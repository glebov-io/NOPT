// dmrg_wrap — thin casci_solver adapter for the block2 DMRG backend: object lifecycle,
// configuration setters, and query accessors. The engine (integral import, solve, RDMs) lives in
// block2_dmrg.cpp; the leading-determinant read-out in block2_mps_to_det.cpp.

#include <memory>

#include "block2_dmrg_engine.h"

using namespace nopt_block2;

// ---- ctor / dtor ---------------------------------------------------------------------
block2_casci_wrap::block2_casci_wrap(int n_act, int na, int nb, int mult, int n_s,
                                     int print_number, const dmrg_par &cfg)
    // block2 runs spin-adapted (SU2): the solve targets total spin 2S = mult-1 (not the Sz
    // projection na-nb, which is 0 for an M_S=0 open shell). The determinant read-out, however,
    // must expand to the native M_S = (na-nb)/2 sector, so pass 2*M_S = na-nb separately.
    : impl_(std::make_unique<dmrgci_engine>(n_act, na + nb, mult - 1, na - nb, mult, n_s,
                                            print_number, cfg)) {
    host_threads_guard htg;
    int nthr = omp_get_max_threads();
    if (nthr < 1) nthr = 1;
    ensure_block2_runtime(cfg.save_dir, cfg.memory, cfg.main_stack, nthr);
}

// The engine owns its MPS tag namespace, so its final scratch set and the bare-state snapshot leave
// with it: every other tag (per-root extracts, the read-out copies) is removed where it is made.
// Best-effort and silent -- a destructor must not propagate, and the runtime clears the scratch dir.
block2_casci_wrap::~block2_casci_wrap() {
    if (impl_ == nullptr)
        return;
    try {
        for (const std::string &t : impl_->snap_tags)
            remove_tag_files(t);
        if (impl_->mps_info != nullptr)
            remove_tag_files(impl_->mps_info->tag);
    } catch (...) {
    }
}

// ----------------------- casci_solver contract ----------------------------------------
void block2_casci_wrap::init_state_storage(int n_s, int) {
    impl_->E_states.assign(n_s, 0.0);
    impl_->storage_ready = true;
}
bool block2_casci_wrap::has_coef(int) const { return impl_->storage_ready; }
void block2_casci_wrap::set_act_rep_num(int *) {
    // The DMRG block runs in C1, so orbsym stays all-zero and the per-active-orbital irreps
    // are ignored.
}
void block2_casci_wrap::set_localization(const double *U) {
    // Copy the localizing rotation (caller's buffer is transient); nullptr => no localization.
    dmrgci_engine &e = *impl_;
    if (U == nullptr) { e.localize_on = false; return; }
    e.U_loc.assign(U, U + (size_t)e.n_act * e.n_act);
    e.localize_on = true;
}
void block2_casci_wrap::set_active_rotation(const double *R) {
    // Retain the previous-basis -> current-basis active rotation for the next solve's warm restart.
    dmrgci_engine &e = *impl_;
    if (R == nullptr) { e.have_rotation = false; return; }
    e.R_active.assign(R, R + (size_t)e.n_act * e.n_act);
    e.have_rotation = true;
}
void block2_casci_wrap::set_report_rotation(const double *U) {
    // Retain the active-block canonicalization for the leading-configuration read-out (print_states).
    dmrgci_engine &e = *impl_;
    if (U == nullptr) { e.have_canon = false; return; }
    e.U_canon.assign(U, U + (size_t)e.n_act * e.n_act);
    e.have_canon = true;
}
void block2_casci_wrap::set_state_weights(const double *w, int n_s) {
    // The weights the state-averaged 2-RDM is accumulated with; none => equal weights.
    dmrgci_engine &e = *impl_;
    if (w == nullptr) { e.w_state.clear(); return; }
    e.w_state.assign(w, w + n_s);
}

int    block2_casci_wrap::n_act()         const { return n_act_; }
int    block2_casci_wrap::n_states()      const { return impl_->n_s; }
int    block2_casci_wrap::mult()          const { return impl_->mult; }
double block2_casci_wrap::E_core()        const { return g0; }
double block2_casci_wrap::E_state(int i)  const { return impl_->E_states[i]; }
double block2_casci_wrap::S2_state(int)   const { double S = impl_->twos / 2.0; return S * (S + 1.0); }
double block2_casci_wrap::L2_state(int)   const { return 0.0; } // linear-molecule Lambda: deferred
double block2_casci_wrap::P_state(int)    const { return 0.0; } // parity: deferred
double *block2_casci_wrap::E_states_ptr() const { return impl_->E_states.data(); }
double block2_casci_wrap::last_solve_resid() const { return impl_->last_sweep_dE; }
bool block2_casci_wrap::last_solve_hit_max() const { return impl_->last_hit_max; }
double block2_casci_wrap::last_solve_dw() const { return impl_->last_two_dot_dw; }
double block2_casci_wrap::last_solve_trunc_de() const { return impl_->last_trunc_de; }
double block2_casci_wrap::energy_resolution() const { return impl_->last_resolution; }
double block2_casci_wrap::last_order_drift() const { return impl_->last_ord_drift; }
bool block2_casci_wrap::last_solve_cold() const { return impl_->last_cold_fallback; }
void block2_casci_wrap::gen_ext_ind() { /* aldet determinant index tables; n/a for an MPS backend */ }
