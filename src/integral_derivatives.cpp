#include "integral_derivatives.h"

#include <cmath>
#include <limits>
#include <omp.h>
#include <vector>
#include <array>

#include <libint2/engine.h>
#include <libint2/config.h>

#include "common_vars.h"
#include "etc.h"

using libint2::BraKet;
using libint2::Engine;
using libint2::Operator;
using libint2::Shell;

namespace{

std::vector<Engine> ov_engines, kin_engines, nuc_engines, eri_engines, ri3_engines, ri2_engines;

// first AO of every shell, and the total
std::vector<int> ao_offsets(const std::vector<Shell> & s, int * total = nullptr){
    std::vector<int> off(s.size());
    int n = 0;
    for(size_t i = 0; i < s.size(); i++){ off[i] = n; n += (int)s[i].size(); }
    if(total) *total = n;
    return off;
}

void shell_limits(const std::vector<Shell> & s, int * nprim, int * lmax){
    for(const Shell & sh : s){
        *nprim = std::max(*nprim, (int)sh.nprim());
        *lmax  = std::max(*lmax, (int)sh.contr[0].l);
    }
}

// add sum(w * buf) over a block to g[3*atom + xyz] for each centre
inline void accumulate(double * g, const int * atom, int ncentres, const double * w, size_t nblock,
                       const libint2::Engine::target_ptr_vec & res){
    // libint marks a screened-out shell set by setting ONLY the first target pointer to nullptr; the others keep pointing at the buffers of the
    // previous shell set. Testing each pointer separately therefore added stale derivatives (invisible on water, 2e-3 Eh/bohr on butadiene).
    if(res[0] == nullptr) return;
    for(int c = 0; c < ncentres; c++) for(int q = 0; q < 3; q++){
        const double * b = res[3 * c + q];
        if(b == nullptr) continue;
        double s = 0.0;
        for(size_t i = 0; i < nblock; i++) s += w[i] * b[i];
        g[3 * atom[c] + q] += s;
    }
}

struct Reduce{
    int nat;
    std::vector<double> buf;
    explicit Reduce(int natoms) : nat(natoms), buf((size_t)num_threads * 3 * natoms, 0.0) {}
    double * mine(){ return buf.data() + (size_t)omp_get_thread_num() * 3 * nat; }
    void add_to(double * g) const {
        for(int t = 0; t < num_threads; t++) for(int i = 0; i < 3 * nat; i++) g[i] += buf[(size_t)t * 3 * nat + i];
    }
};

int check_ready(const std::vector<Engine> & e, const char * what){
    if((int)e.size() < num_threads){
        fprintf(out_stream, "ERROR: derivative engine for %s is not initialized "
                            "(nopt_deriv_engines_initialize)\n", what);
        exit(EXIT_FAILURE);
    }
    return 0;
}

int one_body(molecule * M, std::vector<Engine> & engines, const double * W, double * g, const char * what){
    check_ready(engines, what);
    const std::vector<Shell> & s = M->s;
    const std::vector<int> off = ao_offsets(s);
    const int n_ao = M->n_ao, ns = (int)s.size();
    Reduce red(M->n_atoms);
#pragma omp parallel num_threads(num_threads)
    {
        Engine & eng = engines[omp_get_thread_num()];
        double * gt = red.mine();
        std::vector<double> w;
        for(int i = 0; i < ns; i++) for(int j = 0; j < ns; j++){
            if((i * ns + j) % num_threads != omp_get_thread_num()) continue;
            const int ni = (int)s[i].size(), nj = (int)s[j].size();
            eng.compute(s[i], s[j]);
            w.resize((size_t)ni * nj);
            for(int a = 0; a < ni; a++) for(int b = 0; b < nj; b++) w[a * nj + b] = W[(size_t)(off[i] + a) * n_ao + off[j] + b];
            const int atom[2] = {M->shell_center[i], M->shell_center[j]};
            accumulate(gt, atom, 2, w.data(), (size_t)ni * nj, eng.results());
        }
    }
    red.add_to(g);
    return 0;
}

}

// derivative orders the linked libint was generated with (libint2/config.h; absent: none)
#ifdef LIBINT_INCLUDE_ONEBODY
static const int libint_d1b = LIBINT_INCLUDE_ONEBODY;
#else
static const int libint_d1b = -1;
#endif
#ifdef LIBINT_INCLUDE_ERI
static const int libint_d4c = LIBINT_INCLUDE_ERI;
#else
static const int libint_d4c = -1;
#endif
#ifdef LIBINT_INCLUDE_ERI3
static const int libint_d3c = LIBINT_INCLUDE_ERI3;
#else
static const int libint_d3c = -1;
#endif
#ifdef LIBINT_INCLUDE_ERI2
static const int libint_d2c = LIBINT_INCLUDE_ERI2;
#else
static const int libint_d2c = -1;
#endif

int deriv_check_supported(molecule * M){
    int bad = 0;
    // without this check the first derivative engine aborts inside libint (an assertion on the derivative order)
    if(LIBINT2_MAX_DERIV_ORDER < 1 || libint_d1b < 1 || (RI ? (libint_d3c < 1 || libint_d2c < 1) : libint_d4c < 1)){
        fprintf(out_stream, "ERROR: the linked libint has no first derivatives of the integrals this run needs (derivative orders: 1-body %d, "
                            "4-center %d, 3-center %d, 2-center %d); build one with tests/gradients/libint-deriv.sh\n", libint_d1b, libint_d4c, libint_d3c, libint_d2c);
        return 1;
    }
    if(!M->PP.empty() || !M->SO_PP.empty()){
        fprintf(out_stream, "ERROR: analytic gradients do not support ECP/pseudopotentials yet\n"); bad++; }
    if(M->F_x != 0.0 || M->F_y != 0.0 || M->F_z != 0.0){
        fprintf(out_stream, "ERROR: analytic gradients do not support an external electric field yet\n"); bad++; }
    if(IS_SYM){
        fprintf(out_stream, "ERROR: analytic gradients support C1 runs only (symmetry is on)\n"); bad++; }
    return bad;
}

int nopt_deriv_engines_initialize(molecule * M){
    nopt_deriv_engines_finalize();
    int nprim = 1, lmax = 0, nprim_ri = 1, lmax_ri = 0;
    shell_limits(M->s, &nprim, &lmax);
    shell_limits(M->ri_s, &nprim_ri, &lmax_ri);
    // the same primitive screening as the energy engines (libint_functions.cpp: max_engine_precision = epsilon / 1e10): with a looser
    // precision the derivative engines drop primitive quartets that the energy keeps (3e-8 Eh/bohr on butadiene, exact ERIs)
    const double eps = std::numeric_limits<double>::epsilon() / 1e10;
    const int T = num_threads;
    ov_engines.resize(T); kin_engines.resize(T); nuc_engines.resize(T); eri_engines.resize(T);
    ri3_engines.resize(T); ri2_engines.resize(T);
    const bool have_ri = M->ri_s.size() > 0;
    // the 3- and 2-center engines see AO and auxiliary shells alike
    const int np_ri = std::max(nprim, nprim_ri), l_ri = std::max(lmax, lmax_ri);
#pragma omp parallel for num_threads(T)
    for(int t = 0; t < T; t++){
        ov_engines[t]  = Engine(Operator::overlap, nprim, lmax, 1, eps);
        kin_engines[t] = Engine(Operator::kinetic, nprim, lmax, 1, eps);
        nuc_engines[t] = Engine(Operator::nuclear, nprim, lmax, 1, eps);
        if(RI == 0)
            eri_engines[t] = Engine(Operator::coulomb, nprim, lmax, 1, eps);
        if(have_ri){
            ri3_engines[t] = Engine(Operator::coulomb, np_ri, l_ri, 1, eps, libint2::default_params(Operator::coulomb), BraKet::xs_xx);
            ri2_engines[t] = Engine(Operator::coulomb, np_ri, l_ri, 1, eps, libint2::default_params(Operator::coulomb), BraKet::xs_xs);
        }
    }
    return 0;
}

int nopt_deriv_engines_finalize(){
    ov_engines.clear(); kin_engines.clear(); nuc_engines.clear(); eri_engines.clear();
    ri3_engines.clear(); ri2_engines.clear();
    return 0;
}

int deriv_overlap_contract(molecule * M, const double * W, double * g){ return one_body(M, ov_engines, W, g, "overlap"); }
int deriv_kinetic_contract(molecule * M, const double * W, double * g){ return one_body(M, kin_engines, W, g, "kinetic"); }

int deriv_nuclear_contract(molecule * M, const double * W, double * g){
    check_ready(nuc_engines, "nuclear attraction");
    // point charges: atoms with a nonzero effective charge, in atom order
    std::vector<std::pair<double, std::array<double, 3>>> q;
    std::vector<int> q_atom;
    for(int i = 0; i < M->n_atoms; i++){
        if(M->nucl_charges_calc[i] == 0.0) continue;
        q.push_back({M->nucl_charges_calc[i], {M->atom_coord[3 * i], M->atom_coord[3 * i + 1], M->atom_coord[3 * i + 2]}});
        q_atom.push_back(i);
    }
    const std::vector<Shell> & s = M->s;
    const std::vector<int> off = ao_offsets(s);
    const int n_ao = M->n_ao, ns = (int)s.size(), nq = (int)q.size();
    Reduce red(M->n_atoms);
#pragma omp parallel num_threads(num_threads)
    {
        Engine & eng = nuc_engines[omp_get_thread_num()];
        eng.set_params(q);
        double * gt = red.mine();
        std::vector<double> w;
        std::vector<int> atom(2 + nq);
        for(int i = 0; i < ns; i++) for(int j = 0; j < ns; j++){
            if((i * ns + j) % num_threads != omp_get_thread_num()) continue;
            const int ni = (int)s[i].size(), nj = (int)s[j].size();
            eng.compute(s[i], s[j]);
            w.resize((size_t)ni * nj);
            for(int a = 0; a < ni; a++) for(int b = 0; b < nj; b++) w[a * nj + b] = W[(size_t)(off[i] + a) * n_ao + off[j] + b];
            atom[0] = M->shell_center[i]; atom[1] = M->shell_center[j];
            for(int c = 0; c < nq; c++) atom[2 + c] = q_atom[c];
            accumulate(gt, atom.data(), 2 + nq, w.data(), (size_t)ni * nj, eng.results());
        }
    }
    red.add_to(g);
    return 0;
}

int deriv_nuclear_repulsion(molecule * M, double * g){
    for(int i = 0; i < M->n_atoms; i++) for(int j = i + 1; j < M->n_atoms; j++){
        if(M->atom_is_ghost[i] * M->atom_is_ghost[j] != 0) continue;   // as E_nuc_calc
        double d[3], r2 = 0.0;
        for(int q = 0; q < 3; q++){ d[q] = M->atom_coord[3 * i + q] - M->atom_coord[3 * j + q]; r2 += d[q] * d[q]; }
        const double r = std::sqrt(r2);
        const double f = -M->nucl_charges_calc[i] * M->nucl_charges_calc[j] / (r2 * r);   // d(ZiZj/r)/dRi = -ZiZj d/r^3
        for(int q = 0; q < 3; q++){ g[3 * i + q] += f * d[q]; g[3 * j + q] -= f * d[q]; }
    }
    return 0;
}

int deriv_eri_contract(molecule * M, const deriv_weight_fn & weights, double * g){
    check_ready(eri_engines, "4-center ERI (RI=0 only)");
    const std::vector<Shell> & s = M->s;
    const std::vector<int> off = ao_offsets(s);
    const int ns = (int)s.size();
    Reduce red(M->n_atoms);
#pragma omp parallel num_threads(num_threads)
    {
        Engine & eng = eri_engines[omp_get_thread_num()];
        double * gt = red.mine();
        std::vector<double> w;
        for(int i = 0; i < ns; i++) for(int j = 0; j < ns; j++){
            if((i * ns + j) % num_threads != omp_get_thread_num()) continue;
            for(int k = 0; k < ns; k++) for(int l = 0; l < ns; l++){
                const int sh[4] = {i, j, k, l};
                const int o[4] = {off[i], off[j], off[k], off[l]};
                const int n[4] = {(int)s[i].size(), (int)s[j].size(), (int)s[k].size(), (int)s[l].size()};
                const size_t nb = (size_t)n[0] * n[1] * n[2] * n[3];
                eng.compute(s[i], s[j], s[k], s[l]);
                w.resize(nb);
                weights(sh, o, n, w.data());
                const int atom[4] = {M->shell_center[i], M->shell_center[j], M->shell_center[k], M->shell_center[l]};
                accumulate(gt, atom, 4, w.data(), nb, eng.results());
            }
        }
    }
    red.add_to(g);
    return 0;
}

int deriv_ri3_contract(molecule * M, const deriv_weight_fn & weights, double * g){
    check_ready(ri3_engines, "3-center RI integrals");
    const std::vector<Shell> & s = M->s, & ax = M->ri_s;
    const std::vector<int> off = ao_offsets(s), aoff = ao_offsets(ax);
    const int ns = (int)s.size(), na = (int)ax.size();
    Reduce red(M->n_atoms);
#pragma omp parallel num_threads(num_threads)
    {
        Engine & eng = ri3_engines[omp_get_thread_num()];
        double * gt = red.mine();
        std::vector<double> w;
        for(int p = 0; p < na; p++){
            if(p % num_threads != omp_get_thread_num()) continue;
            for(int i = 0; i < ns; i++) for(int j = 0; j < ns; j++){
                const int sh[3] = {p, i, j};
                const int o[3] = {aoff[p], off[i], off[j]};
                const int n[3] = {(int)ax[p].size(), (int)s[i].size(), (int)s[j].size()};
                const size_t nb = (size_t)n[0] * n[1] * n[2];
                eng.compute(ax[p], s[i], s[j]);
                w.resize(nb);
                weights(sh, o, n, w.data());
                const int atom[3] = {M->ri_shell_center[p], M->shell_center[i], M->shell_center[j]};
                accumulate(gt, atom, 3, w.data(), nb, eng.results());
            }
        }
    }
    red.add_to(g);
    return 0;
}

int deriv_ri2_contract(molecule * M, const deriv_weight_fn & weights, double * g){
    check_ready(ri2_engines, "2-center RI integrals");
    const std::vector<Shell> & ax = M->ri_s;
    const std::vector<int> aoff = ao_offsets(ax);
    const int na = (int)ax.size();
    Reduce red(M->n_atoms);
#pragma omp parallel num_threads(num_threads)
    {
        Engine & eng = ri2_engines[omp_get_thread_num()];
        double * gt = red.mine();
        std::vector<double> w;
        for(int p = 0; p < na; p++) for(int q = 0; q < na; q++){
            if((p * na + q) % num_threads != omp_get_thread_num()) continue;
            const int sh[2] = {p, q};
            const int o[2] = {aoff[p], aoff[q]};
            const int n[2] = {(int)ax[p].size(), (int)ax[q].size()};
            const size_t nb = (size_t)n[0] * n[1];
            eng.compute(ax[p], ax[q]);
            w.resize(nb);
            weights(sh, o, n, w.data());
            const int atom[2] = {M->ri_shell_center[p], M->ri_shell_center[q]};
            accumulate(gt, atom, 2, w.data(), nb, eng.results());
        }
    }
    red.add_to(g);
    return 0;
}

int deriv_eri_JK_contract(molecule * M, const double * D, double cJ, double cK, double * g){
    const int n = M->n_ao;
    return deriv_eri_contract(M, [=](const int *, const int * o, const int * nn, double * w){
        for(int a = 0; a < nn[0]; a++) for(int b = 0; b < nn[1]; b++)
        for(int c = 0; c < nn[2]; c++) for(int d = 0; d < nn[3]; d++){
            const int A = o[0] + a, B = o[1] + b, C = o[2] + c, Dd = o[3] + d;
            w[((size_t)(a * nn[1] + b) * nn[2] + c) * nn[3] + d] =
                cJ * D[(size_t)A * n + B] * D[(size_t)C * n + Dd] + cK * D[(size_t)A * n + C] * D[(size_t)B * n + Dd];
        }
    }, g);
}

// Diagnostic: translational invariance of every 3-center derivative shell triplet, sum over the 3 centers of d(P|ab)/dR = 0 element by element.
// Prints up to max_report failing triplets (angular momenta, primitives, atoms, which result pointers are null); returns the number of failures.
int deriv_ri3_translation_check(molecule * M, int max_report){
    check_ready(ri3_engines, "3-center RI integrals");
    const std::vector<Shell> & s = M->s, & ax = M->ri_s;
    const int ns = (int)s.size(), na = (int)ax.size();
    Engine & eng = ri3_engines[0];
    int nfail = 0;
    double worst = 0.0;
    for(int p = 0; p < na; p++) for(int i = 0; i < ns; i++) for(int j = 0; j < ns; j++){
        eng.compute(ax[p], s[i], s[j]);
        const auto & res = eng.results();
        const size_t nb = ax[p].size() * s[i].size() * s[j].size();
        if(res[0] == nullptr) continue;
        double mx = 0.0, sc = 0.0; int nnull = 0;
        for(int c = 0; c < 3; c++) for(int q = 0; q < 3; q++) if(res[3 * c + q] == nullptr) nnull++;
        for(int q = 0; q < 3; q++) for(size_t k = 0; k < nb; k++){
            double t = 0.0;
            for(int c = 0; c < 3; c++){ const double * b = res[3 * c + q]; if(b){ t += b[k]; sc = std::max(sc, std::fabs(b[k])); } }
            mx = std::max(mx, std::fabs(t));
        }
        worst = std::max(worst, mx);
        if(mx > 1e-8 * std::max(1.0, sc)){
            if(nfail < max_report)
                fprintf(out_stream, "  ri3 translation FAIL: aux shell %d (l=%d, nprim %d, atom %d), ao shells %d (l=%d, nprim %d, atom %d), %d (l=%d, nprim %d, atom %d): "
                                    "max |sum| %.2e, scale %.2e, null results %d\n",
                        p, (int)ax[p].contr[0].l, (int)ax[p].nprim(), M->ri_shell_center[p], i, (int)s[i].contr[0].l, (int)s[i].nprim(), M->shell_center[i],
                        j, (int)s[j].contr[0].l, (int)s[j].nprim(), M->shell_center[j], mx, sc, nnull);
            nfail++;
        }
    }
    fprintf(out_stream, "ri3 translation check: %d failing shell triplets, worst %.2e\n", nfail, worst);
    {   // the shell -> atom maps must be in range and consistent with the shell origins
        int bad = 0;
        auto chk = [&](const std::vector<Shell> & sh, const std::vector<int> & cen, const char * what){
            if(cen.size() < sh.size()){ fprintf(out_stream, "  %s: %zu shells but %zu centre entries\n", what, sh.size(), cen.size()); bad++; return; }
            for(size_t k = 0; k < sh.size(); k++){
                const int a = cen[k];
                if(a < 0 || a >= M->n_atoms){ if(bad < 10) fprintf(out_stream, "  %s shell %zu: atom index %d out of range\n", what, k, a); bad++; continue; }
                double d = 0; for(int x = 0; x < 3; x++) d = std::max(d, std::fabs(sh[k].O[x] - M->atom_coord[3 * a + x]));
                if(d > 1e-10){ if(bad < 10) fprintf(out_stream, "  %s shell %zu: origin differs from atom %d by %.3e\n", what, k, a, d); bad++; }
            }
        };
        chk(s, M->shell_center, "AO"); chk(ax, M->ri_shell_center, "aux");
        fprintf(out_stream, "shell centre maps: %d problem(s)\n", bad);
        nfail += bad;
    }
    return nfail;
}
