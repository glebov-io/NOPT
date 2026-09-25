#pragma once
//
// block2_casci_wrap — casci_solver backend driving an external block2 DMRG-CI

#include <memory>

#include "casci_solver.h"
#include "inp_par_read.h"   // dmrg_par

struct dmrgci_engine;       // opaque; defined in the .cpp (holds all block2 state)

class block2_casci_wrap final : public casci_solver {
    std::unique_ptr<dmrgci_engine> impl_;   // pimpl; complete type only in the .cpp
    
public:
    //integral storage
                double  g0;
    std::vector<double> g1;
    std::vector<double> g2;
    std::vector<double> g3;
    int n_act_;

    // active-space dims (from M->CI[0]) + DMRG config; inits the block2 runtime once.
    block2_casci_wrap(int n_act, int na, int nb, int mult, int n_s, int print_number,
                      const dmrg_par& cfg);
    ~block2_casci_wrap() override;

    // --- configuration / lifecycle ---
    void init_state_storage(int n_s, int i_set) override;
    bool has_coef(int i_set) const override;
    // Persist the converged state set (one single-root MPS per root) under its own scratch tags:
    // a dressed re-solve overwrites the retained MPS, so this must run before the dressed import.
    void snapshot_states(int i_set) override;
    void set_act_rep_num(int* rep_num) override;
    void set_localization(const double* U) override;
    void set_active_rotation(const double* R) override;
    void set_report_rotation(const double* U) override;
    void set_state_weights(const double* w, int n_s) override;
    void import_integrals(double* aaaa, double* f_act, double e_core) override;
    // Encode a TOTAL dressed active-space operator (F_act+g1, (tu|vw)+g2, g3, E_core+E0) as one
    // spin-adapted GeneralFCIDUMP -> GeneralMPO and swap it into the solve. Tensors arrive in the
    // native active basis; the localizing rotation and the frozen Fiedler reorder are applied
    // internally, as the bare import does. h3 may be null.
    bool supports_dressed_import() const override { return true; }
    void import_dressed_operator(const double* h1_total, const double* h2_total,
                                 const double* h3_total, double const_total) override;
    
    void PT2_import_data(double * ext_T3,
                         double * ext_T3_AB,
                         double * ext_T2,
                         double * ext_T2_AB,
                         double * ext_T1,
                         double   ext_T0) override;

    int calc_IPEA_single(double * U_IP, double * H_IP, 
                         double * U_EA, double * H_EA,
                         int a, std::vector<double> avecoe) override;

    // --- solve ---
    int solve(int primary, int read, bool use_prev_guess) override;

    // --- reduced density matrices ---
    void calc_DM_diag(double* gamma, int a) override;
    void G_calc(double* GAMMA) override;
    void calc_DMA(double* g, int a, int b) override;
    void calc_DMB(double* g, int a, int b) override;

    // --- transition-density read-outs ---
    // Diagonal per-state read-outs, native (delocalized) active basis; both OVERWRITE the caller's
    // buffer: G2_calc_diag n_s consecutive n_act^4 GAMMA blocks, G3_calc_diag one n_act^6 moment
    // G3[p,q,r,i,j,k] = <a+_p a+_q a+_r a_k a_j a_i> spin-summed.
    bool supports_g2_diag() const override { return true; }
    void G2_calc_diag(double* G) override;
    bool supports_g3_diag() const override { return true; }
    void G3_calc_diag(double* G3, int state) override;
#if 0  // DIRECT lambda3 path (superseded by the explicit lattice-3RDM route; revive for nact >~ 30)
    bool supports_h2caa_overlap() const override { return true; }
    void h2caa_overlap(const double* Tbra, const double* Tket, int np, double* omega) override;
#endif

    // --- queries ---
    int    n_act()            const override;
    int    n_states()         const override;
    int    mult()             const override;
    double E_core()           const override;
    double E_state(int i)     const override;
    double S2_state(int i)    const override;
    double L2_state(int i)    const override;
    double P_state(int i)     const override;
    double* E_states_ptr()    const override;
    double last_solve_resid() const override;
    bool last_solve_hit_max() const override;
    double last_solve_dw() const override;
    double last_solve_trunc_de() const override;
    double energy_resolution() const override;
    double last_order_drift() const override;
    bool last_solve_cold() const override;

    // --- IO / diagnostics ---
    void gen_ext_ind() override;
    void print_states(int a, int n_s, int print) override;
    void write_civec(int i_s, char* name) override;

    // --- bare-vs-dressed state overlap (the dressed re-solve root map) ---
    // S_track[i*n_s+j] = <state i of the current MPS set | state j of snapshot set b>, one
    // identity-MPO sweep per pair. Only a=0 against the slot snapshot_states filled is defined;
    // both sets sit on the same frozen lattice, so this is exactly the CI overlap.
    void calc_S(double* S_track, int a, int b) override;

    // supports_civec_rotation() stays false (base default): DMRG re-solves rather than rotating
    // CI vectors, so malmqvist/rotate_pi_pair stay no-ops -- calc_S needs no rotation and is
    // real. as_aldet() stays nullptr
};
