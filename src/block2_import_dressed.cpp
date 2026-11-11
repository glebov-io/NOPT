// block2_import_dressed — encode a TOTAL dressed active-space operator (0/1/2/3-body) as one
// spin-adapted GeneralFCIDUMP -> GeneralMPO and swap it into the engine for the dressed solve, plus
// the bare-state snapshot and the bare-vs-dressed state overlap that map the dressed roots back.
// The bare fcidump/hamil stay intact so the RDM read-out (block2_dmrg.cpp) keeps working.

#include "block2_dmrg_engine.h"   // dmrgci_engine, block2 API, shared helpers

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common_vars.h"          // out_stream
#include "cdas_fold.h"            // assemble_g3 (the three-body recombination)
#include "ipea_matrices.h"        // ipea_matrices (the IP/EA metric and Hamiltonian matrices)
#include "tensor_rotate.h"        // rotate1/rotate2/rotate3 (active-space basis transforms)
#include "matr.h"
#include "timer.h"

using namespace block2;
using namespace nopt_block2;

namespace {

// Gather one axis-permuted tensor: out[i,j,...] = in[perm[i], perm[j], ...] over `rank` axes of
// length n. This is block2's FCIDUMP::reorder direction (TInt/V1Int::reorder, integral.hpp:107/176:
// site i takes input orbital perm[i]), so the dressed tensors land on the bare solve's lattice.
static std::vector<double> reorder_gather(const double *in, int n, int rank,
                                          const std::vector<uint16_t> &perm) {
    size_t len = 1;
    for (int r = 0; r < rank; r++) len *= (size_t)n;
    std::vector<double> out(len);
    if (perm.empty()) { // no frozen order => identity copy
        std::copy(in, in + len, out.begin());
        return out;
    }
    std::vector<int> idx(rank);
    for (size_t f = 0; f < len; f++) {
        size_t rem = f;
        for (int r = rank - 1; r >= 0; r--) { idx[r] = (int)(rem % n); rem /= n; }
        size_t src = 0;
        for (int r = 0; r < rank; r++) src = src * n + perm[idx[r]];
        out[f] = in[src];
    }
    return out;
}

// Append one dense-tensor group (cutoff 0, axis->slot rperm). Mirrors the classic add path:
// push the coupling string, then add_sum_term over the full dense tensor.
static void add_group(const std::shared_ptr<GeneralFCIDUMP<double>> &gfd, const std::string &expr,
                      const std::vector<double> &T, int n, int rank, double factor,
                      std::vector<uint16_t> rperm) {
    std::vector<int> shape(rank, n);
    std::vector<size_t> strides(rank);
    size_t s = 1;
    for (int r = rank - 1; r >= 0; r--) { strides[r] = s; s *= (size_t)n; }
    gfd->exprs.push_back(expr);
    gfd->add_sum_term(T.data(), s, shape, strides, 0.0, factor, {}, rperm);
}

// The PT2 increments and the IP/EA blocks combine the stored g1/g2 with read-out RDMs. Under
// internal localization the former are held in the localized basis while the latter come back
// delocalized (block2_dmrg.cpp), so the mix is uncertified and must refuse.
static void require_no_internal_localization(const dmrgci_engine &e, const char *what) {
    if (!e.localize_on) return;
    fprintf(out_stream, "ERROR: %s is not supported with solver-internal localization"
                        " (stored integrals and returned RDMs live in different bases)\n", what);
    exit(EXIT_FAILURE);
}

// Null or all-zero 3-body tensor => skip the 3-body group.
static bool all_zero(const double *T, size_t len) {
    if (T == nullptr) return true;
    for (size_t i = 0; i < len; i++)
        if (T[i] != 0.0) return false;
    return true;
}

// A converged MPS is the precondition of the snapshot and the overlap.
static void require_solved(const dmrgci_engine &e, const char *what) {
    if (e.mps == nullptr || e.mps_info == nullptr) {
        fprintf(out_stream, "ERROR: DMRG %s needs a converged MPS (call solve first)\n", what);
        exit(EXIT_FAILURE);
    }
}

// Drop a previous bare-state snapshot and its scratch files.
static void drop_snapshot(dmrgci_engine &e) {
    drop_state_set(e, "snap");
    e.snap_set = -1;
}

} // namespace

// Move a one-site end-centered MPS to the other end so bra and ket share a center (the
// MovingEnvironment constructor asserts they agree). Lossless gauge sweep at the MPS's own bond
// dimension; the one-dot restriction of block2's DMRGDriver::align_mps_center.
void nopt_block2::align_one_dot_center(dmrgci_engine &e,
                                       const std::shared_ptr<MPS<SU2, double>> &mps, int target) {
    if (mps->center == target) return;
    if (mps->dot != 1 || (target != 0 && target != mps->n_sites - 1)) {
        fprintf(out_stream, "ERROR: the MPS overlap expects one-site end-centered states"
                            " (dot=%d, target center %d of %d sites)\n",
                mps->dot, target, mps->n_sites);
        exit(EXIT_FAILURE);
    }
    std::shared_ptr<CG<SU2>> cg = e.hamil->opf->cg;
    mps->load_mutable();
    mps->info->load_mutable();
    mps->info->bond_dim = std::max(mps->info->bond_dim, mps->info->get_max_bond_dimension());
    if (target == 0)
        while (mps->center != 0)
            mps->move_left(cg, nullptr);
    else {
        mps->canonical_form[0] = 'K';
        while (mps->center != mps->n_sites - 1)
            mps->move_right(cg, nullptr);
    }
    mps->save_data();
}

std::shared_ptr<MPO<SU2, double>>
nopt_block2::build_general_mpo(dmrgci_engine &e, const double *h1_total, const double *h2_total,
                               const double *h3_total, double const_total,
                               const std::string &tag) {
    const int n = e.n_act;

    const size_t len1 = (size_t)n * n, len2 = len1 * len1, len3 = len2 * len1;
    const bool have_3body = !all_zero(h3_total, len3);

    // Native basis in, solver basis out: the localizing rotation first (exactly as the bare import
    // does), then every axis onto the frozen lattice (identity if reorder_perm is empty).
    std::vector<double> l1, l2, l3;
    const double *r1 = h1_total, *r2 = h2_total, *r3 = h3_total;
    if (e.localize_on) {
        l1.resize(len1);
        l2.resize(len2);
        rotate1(h1_total, e.U_loc.data(), n, l1.data(), /*forward=*/true);
        rotate2(h2_total, e.U_loc.data(), n, l2.data(), /*forward=*/true);
        r1 = l1.data();
        r2 = l2.data();
        if (have_3body) {
            l3.resize(len3);
            rotate3(h3_total, e.U_loc.data(), n, l3.data(), /*forward=*/true);
            r3 = l3.data();
        }
    }
    std::vector<double> h1 = reorder_gather(r1, n, 2, e.reorder_perm);
    std::vector<double> h2 = reorder_gather(r2, n, 4, e.reorder_perm);
    std::vector<double> h3;
    if (have_3body)
        h3 = reorder_gather(r3, n, 6, e.reorder_perm);

    // One spin-adapted GeneralFCIDUMP; factors 2^{k/2}/k! complete the 1/k! generator sums.
    auto gfd = std::make_shared<GeneralFCIDUMP<double>>(ElemOpTypes::SU2);
    gfd->const_e = const_total;
    add_group(gfd, "(C+D)0", h1, n, 2, std::sqrt(2.0), {});
    add_group(gfd, "((C+(C+D)0)1+D)0", h2, n, 4, 1.0, {0, 3, 1, 2});
    if (have_3body)
        add_group(gfd, "((C+((C+(C+D)0)1+D)0)1+D)0", h3, n, 6, std::sqrt(2.0) / 3.0,
                  {0, 5, 1, 4, 2, 3});
    std::shared_ptr<GeneralFCIDUMP<double>> afd = gfd->adjust_order();

    // General site Hamiltonian (same vacuum/n/orbsym as the bare path) -> exact FastBipartite MPO.
    SU2 vacuum(0);
    std::vector<typename SU2::pg_t> orbsym(n, 0); // C1
    auto hamil = std::make_shared<GeneralHamiltonian<SU2, double>>(vacuum, n, orbsym);
    auto gmpo = std::make_shared<GeneralMPO<SU2, double>>(
        hamil, afd, MPOAlgorithmTypes::FastBipartite, 0.0, -1, 0, tag);
    gmpo->build();
    std::shared_ptr<MPO<SU2, double>> mpo = std::make_shared<SimplifiedMPO<SU2, double>>(
        gmpo, std::make_shared<Rule<SU2, double>>(), false, false);
    // const_e reaches the sweep once (added at sweep_algorithm.hpp:717 after the const-free eigs);
    // IdentityAddedMPO copies const_e unchanged and injects only a coeff-1 identity operator, so it
    // adds no second constant in a sweep, while making the operator usable for expectations.
    mpo = std::make_shared<IdentityAddedMPO<SU2, double>>(mpo);
    return mpo;
}

// The dressed operator is the OP_DRESSED handle, selected on import: solve() runs it, warm off the
// retained MPS once a snapshot exists.
void block2_casci_wrap::import_dressed_operator(const double *h1_total, const double *h2_total,
                                                const double *h3_total, double const_total) {
    import_named_operator(OP_DRESSED, h1_total, h2_total, h3_total, const_total);
    select_operator(OP_DRESSED);
}

// One persistent single-root MPS per root, extracted from the converged MultiMPS exactly as the RDM
// read-outs do; the intermediate MultiMPS extract is transient. The dressed solve overwrites the
// retained tag (warm) or drops it (cold), so this is the only copy of the bare states left.
void block2_casci_wrap::snapshot_states(int i_set) {
    dmrgci_engine &e = *impl_;
    drop_snapshot(e);
    save_state_set("snap"); // the bare states are the named set calc_S answers for
    e.snap_set = i_set;
}

// One identity-MPO expectation per (dressed, bare) pair. Both sets live on the same frozen lattice
// in the same basis, so no rotation enters and the value is exactly the CI overlap; it is symmetric
// and real, so the bare state takes the bra slot and the transient dressed extract is the one moved
// into a common center.
void block2_casci_wrap::calc_S(double *S_track, int a, int b) {
    dmrgci_engine &e = *impl_;
    require_solved(e, "calc_S");
    const auto snap = e.named_sets.find("snap");
    if (a != 0 || e.snap_set < 0 || b != e.snap_set || snap == e.named_sets.end() ||
        (int)snap->second.mps.size() != e.n_s) {
        fprintf(out_stream, "ERROR: DMRG calc_S compares the current MPS set (a=0) against the"
                            " snapshot set %d (got a=%d b=%d)\n", e.snap_set, a, b);
        exit(EXIT_FAILURE);
    }
    const std::vector<std::shared_ptr<MPS<SU2, double>>> &snap_mps = snap->second.mps;
    host_threads_guard htg;
    const int ns = e.n_s;

    std::shared_ptr<MPO<SU2, double>> impo = std::make_shared<IdentityMPO<SU2, double>>(e.hamil);
    impo = std::make_shared<SimplifiedMPO<SU2, double>>(impo, std::make_shared<Rule<SU2, double>>());

    for (int d = 0; d < ns; d++) {
        const std::string xtag = e.mps_info->tag + "-ov" + std::to_string(d);
        const std::string stag = xtag + "-s";
        std::shared_ptr<MPS<SU2, double>> dmps = extract_root_single(e, d, xtag, stag);
        align_one_dot_center(e, dmps, snap_mps[0]->center);

        for (int q = 0; q < ns; q++) {
            auto ome = std::make_shared<MovingEnvironment<SU2, double, double>>(
                impo, snap_mps[q], dmps, "OVLP");
            ome->init_environments(false);
            auto ex = std::make_shared<Expect<SU2, double, double>>(ome, (ubond_t)e.cfg.m,
                                                                    (ubond_t)e.cfg.m);
            ex->iprint = 0; // silence the per-site Expect log
            // propagate = false: one blocking at the built center, so neither MPS is touched
            S_track[(size_t)d * ns + q] = ex->solve(false, dmps->center != 0);
            ome->remove_partition_files();
        }
        remove_tag_files(xtag); // the per-root extract and its single-MPS copy are transient
        remove_tag_files(stag);
    }
    impo->deallocate();
    assert_stack_clean("bare-vs-dressed CI overlap");
}

void block2_casci_wrap::PT2_import_data(double * ext_T3,
                                       double * ext_T3_AB,
                                       double * ext_T2,
                                       double * ext_T2_AB,
                                       double * ext_T1,
                                       double   ext_T0){

    require_no_internal_localization(*impl_, "block2_casci_wrap::PT2_import_data");

    int error=0;
    if(g1.size()!=n_act_*n_act_            ) error=1;
    if(g2.size()!=n_act_*n_act_*n_act_*n_act_) error=1;
    if(error){
        fprintf(out_stream,"ERROR: block2_casci_wrap::PT2_import_data found inconsistent n_act\n");
        fprintf(out_stream,"       check your code\n");
        exit(0);
    }
    // std::vector<double> h1((size_t)n_act_*n_act_);
    // std::vector<double> h2((size_t)n_act_*n_act_*n_act_*n_act_);
    // std::copy(H_AA,    H_AA+(size_t)n_act_*n_act_, h1.begin());
    // std::copy(act_INTS, act_INTS+(size_t)n_act_*n_act_*n_act_*n_act_, h2.begin());
        
    for(size_t i=0;i<(size_t)n_act_*n_act_;i++)g1[i]+=ext_T1[i];
    
    for(int a=0;a<n_act_;a++) 
    for(int c=0;c<n_act_;c++) 
    for(int b=0;b<n_act_;b++)
    for(int d=0;d<n_act_;d++){
        g2[((a*n_act_+c)*n_act_+b)*n_act_+d]+=ext_T2_AB[((a*n_act_+b)*n_act_+c)*n_act_+d];
    }
    
    g3.resize(n_act_*n_act_*n_act_*n_act_*n_act_*n_act_);
    assemble_g3(n_act_, ext_T3_AB, g3.data());
    
    
    
    double const_total = g0 + ext_T0;
    
    import_dressed_operator(g1.data(), g2.data(), g3.data(), const_total);
    
    
    return;
}

int average_DM_aldet_diag(double * G_out, double * G, std::vector<double> avecoe,int na_p, int n_s);

int block2_casci_wrap::calc_IPEA_single(double * U_IP, double * H_IP, 
                                        double * U_EA, double * H_EA,
                                        int a, std::vector<double> avecoe) {

    require_no_internal_localization(*impl_, "block2_casci_wrap::calc_IPEA_single");

    int n_s = n_states();
    
    double * gamma = new double[n_s*n_act_*n_act_];
    set_zero_matr(gamma,n_act_*n_act_*n_s);
    calc_DM_diag(gamma,a);
    //average_DM_aldet_diag scales the leading block of its input in place
    double * gamma_av = new double[n_act_*n_act_];
    average_DM_aldet_diag(gamma_av,gamma, avecoe,n_act_*n_act_,n_s);
    
    
    double * GAMMA = new double[n_s*n_act_*n_act_*n_act_*n_act_];
    set_zero_matr(GAMMA,n_s*n_act_*n_act_*n_act_*n_act_);
    G_calc(GAMMA);
    
    ipea_matrices(n_act_, g1.data(), g2.data(), gamma_av, GAMMA, U_IP, H_IP, U_EA, H_EA);
    
    printf_timer("calculation of IPEA matrices");
    delete[] GAMMA;
    delete[] gamma_av;
    delete[] gamma;
    return 0;
}
           


