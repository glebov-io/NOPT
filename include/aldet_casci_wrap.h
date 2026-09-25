#pragma once
//
// aldet_casci_wrap — casci_solver adapter over NOPT's built-in determinant CI (aldet_data).
//
// Holds a NON-owning aldet_data* (the molecule owns the CI array) and a NON-owning
// dav_par* (CAS_engine owns the Davidson parameters). It owns nothing: solve() spins up
// a fresh local davidson_solver per call, exactly as the original CAS_engine::CI_calc did
// (src/CAS.cpp:306) — a literal transcription, so behaviour is bit-identical to the
// pre-refactor CI path. The determinant CI supports every capability, so all the
// basis-change-gated operations are real forwards and supports_civec_rotation() is true.

#include <vector>           // aldet.h/CI.h use std::vector but assume the includer pulls it in
#include <cmath>            // std::sqrt

#include "casci_solver.h"
#include "aldet.h"          // aldet_data (full def) + ci_rotate_Pi_pair free function
#include "davidson.h"       // davidson_solver
#include "inp_par_read.h"   // dav_par

class aldet_casci_wrap final : public casci_solver {
    aldet_data* ci_;        // non-owning: lives in molecule
    dav_par*    par_;       // non-owning: CAS_engine's live Davidson parameters
    int         n_s_;       // CAS_engine::n_s, for copy_coef (matches the original literally)

public:
    aldet_casci_wrap(aldet_data* ci, dav_par* par, int n_s) : ci_(ci), par_(par), n_s_(n_s) {}
    ~aldet_casci_wrap() override;   // out-of-line: anchors the vtable in the .cpp

    // --- configuration / lifecycle ---
    void init_state_storage(int n_s, int i_set) override { ci_->init_zero_vec(n_s, i_set); }
    bool has_coef(int i_set) const override { return ci_->coef[i_set] != nullptr; }
    // Same copy the warm-guess solve makes, into the caller's slot (calc_S then reads set i_set).
    void snapshot_states(int i_set) override { ci_->copy_coef(i_set, ci_, n_s_, 0, 0); }
    void set_act_rep_num(int* rep_num) override { ci_->act_rep_num = rep_num; }
    void import_integrals(double* aaaa, double* f_act, double e_core) override {
        ci_->simple_import_data(aaaa, aaaa, f_act, e_core);   // aaaa is both the AA and AB block
    }
    void import_lambda(double* lambda_act, double lambda_core) override {
        ci_->Lambda_act = lambda_act;
        ci_->Lambda_core = lambda_core;
    }
    
    void PT2_import_data(double * ext_T3,
                         double * ext_T3_AB,
                         double * ext_T2,
                         double * ext_T2_AB,
                         double * ext_T1,
                         double   ext_T0) override {
        ci_->PT2_import_data(ext_T3,ext_T3_AB,ext_T2,ext_T2_AB,ext_T1,ext_T0);
    }

    int calc_IPEA_single(double * U_IP, double * H_IP, 
                         double * U_EA, double * H_EA,
                         int a, std::vector<double> avecoe)override {
         ci_->calc_IPEA_single(U_IP, H_IP, U_EA, H_EA, a, avecoe);
         return 1;
    }


    // --- solve (transcription of CI_calc:306-314: fresh solver, set_par, H_diag, run) ---
    int solve(int primary, int read, bool use_prev_guess) override {
        if (use_prev_guess) ci_->copy_coef(1, ci_, n_s_, 0, 0);
        davidson_solver dav;
        dav.set_par(ci_, *par_);
        dav.V.H_diag_calc();
        return dav.run(primary, read);
    }
        

    // Davidson stops on dr (squared residual) or de, whichever comes first.
    double energy_resolution() const override {
        const double s = std::sqrt(par_->r_conv);
        return par_->e_conv > s ? par_->e_conv : s;
    }

    // --- reduced density matrices ---
    void calc_DM_diag(double* gamma, int a) override { ci_->calc_DM_diag(gamma, a); }
    void G_calc(double* GAMMA) override { ci_->G_calc(GAMMA); } //works differently then in DMRG !!!
    void calc_DMA(double* g, int a, int b) override { ci_->calc_DMA(g, a, b); }
    void calc_DMB(double* g, int a, int b) override { ci_->calc_DMB(g, a, b); }

    // --- transition-density read-outs (the determinant CI supplies all of them natively) ---
#if 0  // full transition 2-RDM: no consumer, the driver reads G2_calc_diag. Revive for first-order
       // properties -- the 2-body Mbar needs bra != ket densities between the dressed roots.
    bool supports_g2_full() const override { return true; }
    void G_calc_full(double* G) override;                                     // full n_s x n_s 2-RDM
#endif
    bool supports_g3_diag() const override { return true; }
    void G3_calc_diag(double* G3, int state) override;                        // per-state 3-body moment
    bool supports_g2_diag() const override { return true; }
    void G2_calc_diag(double* G) override { ci_->G_calc(G); }                 // n_s per-state blocks
#if 0  // DIRECT lambda3 path (superseded by the explicit lattice-3RDM route; revive for nact >~ 30)
    bool supports_h2caa_overlap() const override { return true; }
    void h2caa_overlap(const double* Tbra, const double* Tket, int np, double* omega) override;
    void h2caa_overlap2(const double* Tbra1, const double* Tket1, int np1, double* omega1,
                        const double* Tbra2, const double* Tket2, int np2, double* omega2) override;
#endif

    // --- queries ---
    int    n_act()         const override { return ci_->n_act; }
    int    n_states()      const override { return ci_->n_states[0]; }
    int    mult()          const override { return ci_->mult; }
    double E_core()        const override { return ci_->E_core; }
    double E_state(int i)  const override { return ci_->E_states[0][i]; }
    double S2_state(int i) const override { return ci_->S2[0][i]; }
    double L2_state(int i) const override { return ci_->L2[0][i]; }
    double P_state(int i)  const override { return ci_->P[0][i]; }
    double* E_states_ptr() const override { return ci_->E_states[0]; }

    // --- wavefunction-vs-rotated-basis ops (all supported by the determinant CI) ---
    bool supports_civec_rotation() const override { return true; }
    void malmqvist(int i_set, double* U) override { ci_->malmqvist(i_set, U); }
    void rotate_pi_pair(int i_set, double s, double c, int pair, int* ind_pi) override {
        ci_rotate_Pi_pair(ci_, i_set, s, c, pair, ind_pi);
    }
    void calc_S(double* S_track, int a, int b) override { ci_->calc_S(S_track, a, b); }

    // --- IO ---
    void gen_ext_ind() override { ci_->gen_ext_ind(); }
    void print_states(int a, int n_s, int print) override { ci_->print_states(a, n_s, print); }
    void write_civec(int i_s, char* name) override { ci_->write_civec(i_s, name); }

    // --- escape hatch ---
    aldet_data* as_aldet() override { return ci_; }
};
