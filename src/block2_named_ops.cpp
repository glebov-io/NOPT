// block2_named_ops — named operator handles on the DMRG engine: folded/dressed general MPOs built
// alongside the bare one and the selection the solves run on. Also the branch machinery: truncated
// zero-noise continuations of the retained state, named checkpoints and named state sets, with
// their cross-set overlaps and per-root expectation values.

#include "block2_dmrg_engine.h"   // dmrgci_engine, block2 API, shared helpers

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common_vars.h"          // out_stream
#include "dmrg_log.h"             // per-solve block2 sweep log (cout/cerr redirect)

using namespace block2;
using namespace nopt_block2;

namespace {

// A converged MPS is the precondition of an expectation value.
static void require_solved(const dmrgci_engine &e, const char *what) {
    if (e.mps == nullptr || e.mps_info == nullptr) {
        fprintf(out_stream, "ERROR: DMRG %s needs a converged MPS (call solve first)\n", what);
        exit(EXIT_FAILURE);
    }
}

// The engine slot holding one operator kind; null until that kind is imported.
static std::shared_ptr<MPO<SU2, double>> &handle_of(dmrgci_engine &e, int kind) {
    if (kind == OP_BARE)
        return e.mpo_bare;
    else if (kind == OP_FOLDED)
        return e.mpo_f;
    else if (kind == OP_DRESSED)
        return e.mpo_d;
    else {
        fprintf(out_stream, "ERROR: unknown CI operator kind %d"
                            " (bare=%d, folded=%d, dressed=%d)\n",
                kind, (int)OP_BARE, (int)OP_FOLDED, (int)OP_DRESSED);
        exit(EXIT_FAILURE);
    }
}

// A named state set with one stored state per root is the precondition of every set read-out.
static const state_set &require_set(const dmrgci_engine &e, const char *name) {
    auto it = e.named_sets.find(name);
    if (it == e.named_sets.end() || (int)it->second.mps.size() != e.n_s) {
        fprintf(out_stream, "ERROR: DMRG state set '%s' is not stored with %d roots\n",
                name, e.n_s);
        exit(EXIT_FAILURE);
    }
    return it->second;
}

// Is a scratch tag one a named checkpoint or a named state set owns? A working tag never is, by
// construction; a replaced one is only removed when it is not.
static bool tag_is_named(const dmrgci_engine &e, const std::string &tag) {
    for (const auto &c : e.named_ckpt)
        if (c.second == tag)
            return true;
    for (const auto &s : e.named_sets)
        for (const std::string &t : s.second.tags)
            if (t == tag)
                return true;
    return false;
}

} // namespace

void nopt_block2::drop_state_set(dmrgci_engine &e, const std::string &name) {
    auto it = e.named_sets.find(name);
    if (it == e.named_sets.end())
        return;
    for (const std::string &t : it->second.tags)
        remove_tag_files(t);
    e.named_sets.erase(it);
}

// Build the folded or dressed handle; the bare one is import_integrals's. The selected kind is
// untouched, so a handle may be prepared while the solves stay on the current operator; replacing
// the kind that is selected moves the selection onto the new handle.
void block2_casci_wrap::import_named_operator(int kind, const double *h1, const double *h2,
                                              const double *h3, double c) {
    dmrgci_engine &e = *impl_;
    if (kind != OP_FOLDED && kind != OP_DRESSED) {
        fprintf(out_stream, "ERROR: only the folded and dressed operators are imported;"
                            " the bare one comes from import_integrals (kind %d)\n", kind);
        exit(EXIT_FAILURE);
    }
    // A Fiedler lattice must already be frozen by the bare import; without it the imported
    // tensors have no site->orbital map to land on.
    if (e.cfg.loc_order == DMRG_LOCORDER_FIEDLER && e.reorder_perm.empty()) {
        fprintf(out_stream, "ERROR: named operator import before bare import"
                            " (no frozen Fiedler order)\n");
        exit(EXIT_FAILURE);
    }
    host_threads_guard htg;
    const std::string tag = (kind == OP_FOLDED) ? "HF" : "HD";
    std::shared_ptr<MPO<SU2, double>> &h = handle_of(e, kind);
    h = build_general_mpo(e, h1, h2, h3, c, tag);
    if (e.op_kind == kind) e.mpo = h;   // e.mpo == handle(op_kind) after every replacement
}

// Selecting a general handle marks the engine dressed, so solve()'s dressed-warm gate and its
// frozen-order protection cover the folded operator exactly as they cover the dressed one.
void block2_casci_wrap::select_operator(int kind) {
    dmrgci_engine &e = *impl_;
    std::shared_ptr<MPO<SU2, double>> &h = handle_of(e, kind);
    if (h == nullptr) {
        fprintf(out_stream, "ERROR: CI operator kind %d has no handle (import it first)\n", kind);
        exit(EXIT_FAILURE);
    }
    e.mpo = h;
    e.op_kind = kind;
    e.dressed_mpo = (kind != OP_BARE);
}

// One branch of the retained state set: a zero-noise continuation under operator kind at bond
// dimension m, entered from the working tag on disk. No RNG is drawn, so two branches of the same
// start under the same handle are the same map. The caller's selection comes back on return.
int block2_casci_wrap::solve_branch(int kind, int m, int n_sweeps, double dav_tol,
                                    bool one_site_tail) {
    dmrgci_engine &e = *impl_;
    if (e.mps == nullptr || e.mps_info == nullptr) {
        fprintf(out_stream, "ERROR: DMRG solve_branch has no retained state to branch from"
                            " (call solve first)\n");
        exit(EXIT_FAILURE);
    }
    if (m < 1 || n_sweeps < 1) {
        fprintf(out_stream, "ERROR: DMRG solve_branch needs m >= 1 and n_sweeps >= 1"
                            " (got m=%d, n_sweeps=%d)\n", m, n_sweeps);
        exit(EXIT_FAILURE);
    }
    dmrg_log_guard log(true); // block2's sweep output for this branch only
    host_threads_guard htg;
    e.d2_valid = false;     // new wavefunction -> any cached 2-RDM is stale
    e.dmfull_valid = false; // ... and the cached property 1-RDM
    reload_retained_mps(e); // the in-memory MultiMPS is a shell after any solve

    const std::shared_ptr<MPO<SU2, double>> mpo_keep = e.mpo;
    const int kind_keep = e.op_kind;
    const bool dressed_keep = e.dressed_mpo;
    select_operator(kind);

    // The schedule is the caller's, flat: one bond dimension, no noise, one Davidson threshold.
    std::vector<ubond_t> bond_dims((size_t)n_sweeps, (ubond_t)m);
    std::vector<double> noises((size_t)n_sweeps, 0.0);
    std::vector<double> dav_thrds((size_t)n_sweeps, dav_tol);

    auto me = std::make_shared<MovingEnvironment<SU2, double, double>>(e.mpo, e.mps, e.mps,
                                                                       "DMRG");
    me->delayed_contraction = OpNamesSet::normal_ops();
    me->cached_contraction = true;
    me->init_environments(false);

    auto dmrg = std::make_shared<DMRG<SU2, double, double>>(me, bond_dims, noises);
    dmrg->davidson_conv_thrds = dav_thrds;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollected; // state-averaged perturbative noise
    dmrg->trunc_type = dmrg->trunc_type | TruncationTypes::RealDensityMatrix;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    dmrg->davidson_soft_max_iter = 200;
    dmrg->iprint = DMRG_LOG_IPRINT;
    const bool forward = (e.mps->center == 0); // the direction this branch sweeps in
    dmrg->solve(n_sweeps, forward, e.cfg.sweep_tol);

    // Truncation the branch enters with (its first sweep) and the one its stored state carries (its
    // last two-site sweep), both read before the tail appends to the same history.
    const double dw_none = std::numeric_limits<double>::quiet_NaN();
    e.last_entry_dw = dmrg->discarded_weights.empty()
                          ? dw_none : (double)dmrg->discarded_weights.front();
    e.last_two_dot_dw = dmrg->discarded_weights.empty()
                            ? dw_none : (double)dmrg->discarded_weights.back();

    // Convergence of the variational phase, as the max over roots. One sweep leaves it
    // unmeasurable (NaN), which counts as not converged rather than faking convergence.
    int n2 = (int)dmrg->energies.size();
    if (n2 >= 2) {
        const auto &en1 = dmrg->energies[n2 - 1];
        const auto &en0 = dmrg->energies[n2 - 2];
        const int nr = (int)en1.size() < (int)en0.size() ? (int)en1.size() : (int)en0.size();
        double dmax = 0.0;
        for (int r = 0; r < nr; r++) {
            const double d = std::fabs((double)en1[r] - (double)en0[r]);
            if (d > dmax) dmax = d;
        }
        e.last_sweep_dE = dmax;
    } else {
        e.last_sweep_dE = std::numeric_limits<double>::quiet_NaN();
    }
    e.last_converged = (e.last_sweep_dE < e.cfg.sweep_tol);
    e.last_hit_max = (n2 == n_sweeps && !e.last_converged);

    // A tailed branch closes on the lattice end it entered from: an odd variational phase takes one
    // more zero-noise two-site sweep, so all branches of one checkpoint close on the same end. The
    // convergence record above stays that of the phase block2 stopped.
    if (one_site_tail && n2 % 2 == 1) {
        dmrg->bond_dims.assign(n2 + 1, (ubond_t)m);
        dmrg->noises.assign(n2 + 1, 0.0);
        dmrg->davidson_conv_thrds.assign(n2 + 1, dav_tol);
        dmrg->solve(n2 + 1, dmrg->forward, /*tol=*/0.0, n2);
        n2 = (int)dmrg->energies.size();
        e.last_two_dot_dw = (double)dmrg->discarded_weights.back();
    }

    // The one-site sweeps every solve closes with, at zero noise and tolerance zero: zero noise is a
    // correctness condition, since block2's perturbative noise raises the density-matrix rank.
    e.last_tail_sweeps = 0;
    e.last_tail_dw = 0.0;
    if (one_site_tail && DMRG_ONEDOT_TAIL > 0) {
        const int total = n2 + DMRG_ONEDOT_TAIL;
        dmrg->me->dot = 1;
        dmrg->bond_dims.assign(total, (ubond_t)m);
        dmrg->noises.assign(total, 0.0);
        dmrg->davidson_conv_thrds.assign(total, dav_tol);
        dmrg->solve(total, dmrg->forward, /*tol=*/0.0, n2);
        e.mps->dot = 1;     // the sweep switched me->dot; the MPS must say so too
        e.mps->save_data();
        e.last_tail_sweeps = (int)dmrg->energies.size() - n2;
        for (size_t i = (size_t)n2; i < dmrg->discarded_weights.size(); i++)
            e.last_tail_dw = std::max(e.last_tail_dw, (double)dmrg->discarded_weights[i]);
    }
    // Two-site center and structural record on disk: the entry state of a further branch and the
    // state deep_copy reads. Without a tail the sweep's own form already is that, and this saves it.
    adjust_mps_two_dot(e);

    // Sweep energies of this branch. An RDM read-out would replace them with its own contraction,
    // so the flow copies them out before any density read.
    if (dmrg->energies.empty()) {
        fprintf(out_stream, "ERROR: DMRG branch produced no sweep energies\n");
        exit(EXIT_FAILURE);
    }
    const auto &eng = dmrg->energies.back();
    if ((int)eng.size() < e.n_s) {
        fprintf(out_stream, "ERROR: DMRG branch returned %d roots < n_s=%d\n",
                (int)eng.size(), e.n_s);
        exit(EXIT_FAILURE);
    }
    e.E_states.assign(e.n_s, 0.0);
    for (int s = 0; s < e.n_s; s++) {
        if (!std::isfinite((double)eng[s])) {
            fprintf(out_stream, "ERROR: DMRG branch root %d energy is not finite\n", s);
            exit(EXIT_FAILURE);
        }
        e.E_states[s] = (double)eng[s];
    }

    e.mpo = mpo_keep; // the selection this branch found, restored
    e.op_kind = kind_keep;
    e.dressed_mpo = dressed_keep;

    me->remove_partition_files(); // keep mps/mps_info alive for the read-outs
    assert_stack_clean("branch solve");
    return n2;
}

// Persist the retained state set under a name: one deep copy of the working MultiMPS onto a tag of
// its own. deep_copy reads the on-disk state, so this must follow a solve that wrote it; its return
// is a deallocated shell, and only the tag is kept. A replaced checkpoint takes its files with it.
void block2_casci_wrap::save_checkpoint(const char *name) {
    dmrgci_engine &e = *impl_;
    require_solved(e, "save_checkpoint");
    host_threads_guard htg;
    const std::string key(name);
    auto it = e.named_ckpt.find(key);
    if (it != e.named_ckpt.end()) {
        remove_tag_files(it->second);
        e.named_ckpt.erase(it);
    }
    reload_retained_mps(e); // the in-memory MultiMPS is a shell after any solve
    const std::string tag = e.mps_info->tag + "-" + key;
    e.mps->deep_copy(tag);
    e.mps->deallocate(); // back to the shell state, as block2's own deep_copy ends
    e.mps_info->deallocate_mutable();
    e.named_ckpt[key] = tag;
    assert_stack_clean("state checkpoint");
}

// Restore a checkpoint onto a fresh working tag. The checkpoint is copied, never consumed, so the
// same one starts any number of branches; its files stay until release_named_states. The live state
// ends as a solve leaves it: a shell over the new working tag, authoritative on disk.
void block2_casci_wrap::load_checkpoint(const char *name) {
    dmrgci_engine &e = *impl_;
    const auto it = e.named_ckpt.find(name);
    if (it == e.named_ckpt.end()) {
        fprintf(out_stream, "ERROR: DMRG checkpoint '%s' is not stored\n", name);
        exit(EXIT_FAILURE);
    }
    host_threads_guard htg;
    const std::string prev = (e.mps_info == nullptr) ? std::string() : e.mps_info->tag;
    const std::string work = "w" + std::to_string(e.engine_id) + "_work_" +
                             std::to_string(e.solve_count++);
    auto info = std::make_shared<MultiMPSInfo<SU2>>(e.mpo->n_sites, e.hamil->vacuum,
                                                    std::vector<SU2>{e.target}, e.mpo->basis);
    info->tag = it->second; // the retained-MPS reload recipe, on the checkpoint's tag
    info->load_mutable();
    auto mps = std::make_shared<MultiMPS<SU2, double>>(info);
    mps->load_data();
    mps->load_mutable();
    mps->deep_copy(work);
    mps->deallocate();
    info->deallocate_mutable();
    info->tag = work; // reload_retained_mps takes the tag off the engine's info
    e.mps_info = info;
    reload_retained_mps(e);
    e.mps->deallocate();
    e.mps_info->deallocate_mutable();
    e.d2_valid = false;     // another wavefunction -> any cached 2-RDM is stale
    e.dmfull_valid = false; // ... and the cached property 1-RDM
    if (!prev.empty() && prev != work && !tag_is_named(e, prev))
        remove_tag_files(prev); // the working copy this restore replaces
    assert_stack_clean("state checkpoint restore");
}

// One persistent single-root MPS per root under a name, extracted from the converged MultiMPS
// exactly as the RDM read-outs do; the intermediate MultiMPS extract is transient. A set of the
// same name is dropped first, files included.
void block2_casci_wrap::save_state_set(const char *name) {
    dmrgci_engine &e = *impl_;
    require_solved(e, "save_state_set");
    host_threads_guard htg;
    const std::string key(name);
    drop_state_set(e, key);
    state_set &set = e.named_sets[key];
    set.mps.resize(e.n_s);
    for (int st = 0; st < e.n_s; st++) {
        const std::string xtag = e.mps_info->tag + "-" + key + std::to_string(st);
        const std::string stag = xtag + "-s";
        set.mps[st] = extract_root_single(e, st, xtag, stag);
        set.tags.push_back(stag);
        remove_tag_files(xtag); // the single-MPS copy under stag is self-contained
    }
    assert_stack_clean("named state set");
}

// S[i*n_s+j] = <state i of set a| state j of set b>: bra rows from a, ket columns from b, and a
// cross-set matrix is not symmetric. Both sets sit on the same frozen lattice in the same basis, so
// no rotation enters and the value is exactly the CI overlap. A stored state is never modified: a
// ket centered at the other end is aligned on a disposable copy.
void block2_casci_wrap::overlap_sets(const char *a, const char *b, double *S) {
    dmrgci_engine &e = *impl_;
    require_solved(e, "overlap_sets"); // the disposable copy is named after the working tag
    const state_set &sa = require_set(e, a);
    const state_set &sb = require_set(e, b);
    host_threads_guard htg;
    const int ns = e.n_s;
    const int bra_center = sa.mps[0]->center;

    std::shared_ptr<MPO<SU2, double>> impo = std::make_shared<IdentityMPO<SU2, double>>(e.hamil);
    impo = std::make_shared<SimplifiedMPO<SU2, double>>(impo, std::make_shared<Rule<SU2, double>>());

    for (int j = 0; j < ns; j++) {
        const std::string ktag = e.mps_info->tag + "-ovk" + std::to_string(j);
        std::shared_ptr<MPS<SU2, double>> kmps = sb.mps[j];
        const bool copied = (kmps->center != bra_center);
        if (copied) {
            kmps = kmps->deep_copy(ktag);
            align_one_dot_center(e, kmps, bra_center);
        }
        for (int i = 0; i < ns; i++) {
            auto ome = std::make_shared<MovingEnvironment<SU2, double, double>>(
                impo, sa.mps[i], kmps, "OVLP");
            ome->init_environments(false);
            auto ex = std::make_shared<Expect<SU2, double, double>>(ome, (ubond_t)e.cfg.m,
                                                                    (ubond_t)e.cfg.m);
            ex->iprint = 0; // silence the per-site Expect log
            // propagate = false: one blocking at the built center, so neither MPS is touched
            S[(size_t)i * ns + j] = ex->solve(false, kmps->center != 0);
            ome->remove_partition_files();
        }
        if (copied)
            remove_tag_files(ktag);
    }
    impo->deallocate();
    assert_stack_clean("named state set overlap");
}

// E[st] = <state st of the named set| operator kind |state st>: one nonpropagating Expect sweep per
// root on the stored state. The handle's IdentityAddedMPO carries its const_e into the value once.
void block2_casci_wrap::expect_set(const char *name, int kind, double *E) {
    dmrgci_engine &e = *impl_;
    const state_set &set = require_set(e, name);
    std::shared_ptr<MPO<SU2, double>> &h = handle_of(e, kind);
    if (h == nullptr) {
        fprintf(out_stream, "ERROR: CI operator kind %d has no handle (import it first)\n", kind);
        exit(EXIT_FAILURE);
    }
    host_threads_guard htg;

    for (int st = 0; st < e.n_s; st++) {
        const std::shared_ptr<MPS<SU2, double>> &imps = set.mps[st];
        auto me = std::make_shared<MovingEnvironment<SU2, double, double>>(h, imps, imps, "EXPT");
        me->delayed_contraction = OpNamesSet::normal_ops();
        me->cached_contraction = false;
        me->fused_contraction_rotation = true;
        me->save_environments = false;
        me->init_environments(false);

        auto ex = std::make_shared<Expect<SU2, double, double>>(me, (ubond_t)e.cfg.m,
                                                                (ubond_t)e.cfg.m);
        ex->iprint = 0; // silence the per-site Expect log
        E[st] = ex->solve(false, imps->center != 0);
        me->remove_partition_files();
    }
    assert_stack_clean("named state set expectation");
}

// Largest bond dimension the live MultiMPS's tensors carry, from the StateInfo each canonical form
// owns on a fresh reload. MPSInfo::bond_dim is the requested one and only ever grows, and the array
// of the other direction is left over from an earlier sweep, so neither measures a compression.
int block2_casci_wrap::last_max_bond_dim() const {
    dmrgci_engine &e = *impl_;
    require_solved(e, "last_max_bond_dim");
    host_threads_guard htg;
    reload_retained_mps(e);
    int mx = 0;
    for (int i = 0; i < e.mps->n_sites; i++) {
        const char cf = e.mps->canonical_form[i];
        if (cf == 'L')
            mx = std::max(mx, (int)e.mps_info->left_dims[i + 1]->n_states_total);
        else if (cf == 'R')
            mx = std::max(mx, (int)e.mps_info->right_dims[i]->n_states_total);
    }
    e.mps->deallocate(); // back to the shell state every read-out reloads from
    e.mps_info->deallocate_mutable();
    assert_stack_clean("bond dimension read");
    return mx;
}

// Every named set and checkpoint the branch flow made, files included. The set the dressed re-solve
// owns ("snap") and the live working tag are left alone.
void block2_casci_wrap::release_named_states() {
    dmrgci_engine &e = *impl_;
    for (auto it = e.named_sets.begin(); it != e.named_sets.end();) {
        if (it->first == "snap") {
            ++it;
            continue;
        }
        for (const std::string &t : it->second.tags)
            remove_tag_files(t);
        it = e.named_sets.erase(it);
    }
    for (const auto &c : e.named_ckpt)
        remove_tag_files(c.second);
    e.named_ckpt.clear();
}
