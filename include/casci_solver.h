#pragma once
//
// casci_solver — abstract interface between the CAS-SCF orbital optimizer (CAS_engine)
// and a concrete active-space CI solver (built-in aldet determinant CI, DMRG, ...).
//
// The optimizer consumes only the RDMs (gamma/GAMMA); it never reads a CI internal.
// Any backend that can import the active-space Hamiltonian, solve, and hand back 1- and
// 2-RDMs in NOPT's convention can drive CAS-SCF through this one type.
//
// Method names deliberately mirror the existing aldet_data routines so the CAS_engine
// touch-point conversion is a mechanical CI[0].foo() -> CI->foo() rename (keeps the
// behavioural diff small and reviewable). Determinant-specific operations that do not
// generalise to an MPS backend are capability-gated via supports_civec_rotation().
//
// Pointer parameters stay raw (double*/int*) for zero-copy interop with the existing
// flat-array code; this interface introduces no ownership.

class aldet_data;  // opaque to consumers; only the aldet adapter dereferences it
#include <limits>
#include <vector>

class casci_solver {
public:
    virtual ~casci_solver();

    // --- configuration / lifecycle ---
    virtual void init_state_storage(int n_s, int i_set) = 0;   // allocate coef/E_states/... (aldet: init_zero_vec)
    virtual bool has_coef(int i_set) const = 0;                // is the CI vector storage allocated?
    // Snapshot the current wavefunction set into storage slot i_set, so a later solve on a dressed
    // operator can be overlapped against it through calc_S(S, 0, i_set). aldet copies its CI
    // vectors; an MPS backend persists a tagged copy of its state set. Default aborts loudly.
    virtual void snapshot_states(int i_set);
    virtual void set_act_rep_num(int* rep_num) = 0;            // per-active-orbital irrep numbers
    virtual void import_integrals(double* aaaa,                // active 2-e (tu|vw), chemist, n_act^4
                                  double* f_act,               // embedded 1-e h_tu (core folded in), n_act^2
                                  double e_core) = 0;          // inactive + nuclear scalar
    virtual void import_lambda(double* lambda_act,             // linear-molecule Lambda machinery (optional;
                               double lambda_core) {}          //   no-op unless the backend supports it)
    virtual void PT2_import_data(double * ext_T3,
                                 double * ext_T3_AB,
                                 double * ext_T2,
                                 double * ext_T2_AB,
                                 double * ext_T1,
                                 double   ext_T0) {}
    
    virtual int calc_IPEA_single(double * U_IP, double * H_IP, 
                                 double * U_EA, double * H_EA,
                                 int a, std::vector<double> avecoe);
    

    // Active-space localizing rotation U (n_act x n_act, [a*n_act+p], C_loc=C*U, U^T U=I). The
    // backend solves in the rotated basis and reports RDMs back in the original basis; nullptr or
    // never-called means solve in the supplied basis. aldet ignores it.
    virtual void set_localization(const double* U) {}
    // Active-space rotation R (n_act x n_act, [a*n_act+p]) taking the previous macro-iteration's
    // active basis to the current one. A warm-start backend uses it to rotate its retained
    // wavefunction across the basis change; aldet and cold solves ignore it. Called only when a
    // usable previous wavefunction exists (never on the first solve or after a fallback).
    virtual void set_active_rotation(const double* R) {}
    // Active-block canonicalization U (n_act x n_act, [a*n_act+p], eigenvectors of the active
    // Fock block, ascending eigenvalue -- the same rotation the aldet path applies via malmqvist).
    // A backend that can't rotate its own CI vector uses it to report its leading configurations in
    // the canonical basis. aldet and rotation-capable backends ignore it (they already canonicalize).
    virtual void set_report_rotation(const double* U) {}
    // State-average weights (n_s entries, positive; the backend normalizes by their sum) the
    // optimizer consumes the RDMs with. Only a backend that averages internally needs them; one that
    // hands back per-state RDMs ignores it.
    virtual void set_state_weights(const double* w, int n_s) {}

    // --- solve ---
    // Encapsulates the full diagonalisation (aldet: copy_coef -> set_par -> H_diag_calc -> run).
    // use_prev_guess: this solve continues from the previous solution -- the backend may snapshot
    // that solution (state tracking) and/or reuse it as its starting guess. false marks an
    // independent solve and forces a cold start.
    // Returns an iteration/convergence count.
    virtual int solve(int primary, int read, bool use_prev_guess) = 0;
    
    
    // --- reduced density matrices (the actual contract) ---
    // 1-RDM diagonal-form: spin-summed, symmetric; trace = N_active_el; each diagonal
    // entry (orbital occupation) satisfies 0 <= gamma_tt <= 2.
    virtual void calc_DM_diag(double* gamma, int a) = 0;
    // 2-RDM, NOPT convention: E2 = 1/2 * sum_{tuvw} GAMMA_{tuvw} (tu|vw),
    // packed GAMMA[((t*n_act+u)*n_act+v)*n_act+w]. A backend holding per-state tensors writes n_s
    // consecutive blocks and the optimizer averages them; one that only ever forms the state average
    // (DMRG: the per-state tensors never coexist) writes that single block. CAS_engine sizes GAMMA
    // and skips the averaging accordingly.
    virtual void G_calc(double* GAMMA) = 0;
    // spin-resolved / transition 1-RDM blocks (properties): alpha and beta.
    virtual void calc_DMA(double* g, int a, int b) = 0;
    virtual void calc_DMB(double* g, int a, int b) = 0;

    // --- queries ---
    virtual int    n_act()           const = 0;
    virtual int    n_states()        const = 0;
    virtual int    mult()            const = 0;
    virtual double E_core()          const = 0;
    virtual double E_state(int i)    const = 0;
    virtual double S2_state(int i)   const = 0;
    virtual double L2_state(int i)   const = 0;                // linear molecules
    virtual double P_state(int i)    const = 0;                // parity (linear molecules)
    virtual double* E_states_ptr() const = 0;                 // contiguous state-energy block (raw view, for PrintEnergy)
    // Energy convergence actually achieved by the last solve (DMRG: |dE| between the final two
    // sweeps; iterative CI: final residual). 0 if not tracked. Reported in the CAS-SCF table.
    virtual double last_solve_resid() const { return 0.0; }
    // True if the last solve exhausted its sweep/iteration budget without meeting the convergence
    // threshold (DMRG: reached the schedule's max sweeps with |dE| still above sweep_tol). Flags a
    // possibly under-converged CI vector in the CAS-SCF table. Backends that don't track it: false.
    virtual bool last_solve_hit_max() const { return false; }
    // Staleness of a pinned orbital ordering the backend solves on: its ordering cost over that of
    // an order derived afresh, in the orbitals of the last solve. 1.0 = as good as fresh, larger =
    // the pinned lattice has drifted. NaN where nothing was pinned, and for backends with no lattice.
    virtual double last_order_drift() const { return std::numeric_limits<double>::quiet_NaN(); }
    // Truncation the last solve's stored wavefunction carries: the discarded weight of its final
    // variational (two-site) sweep, worst over that sweep's decimations. A state-averaged solve
    // truncates one averaged density matrix, so this covers all roots. Untruncated backends: NaN.
    virtual double last_solve_dw() const { return std::numeric_limits<double>::quiet_NaN(); }
    // Energy the last solve gave up to its truncation: the stored wavefunction's energy minus the
    // last variational (two-site) sweep's, worst over roots. An energy change below a fraction of
    // it is within the CI method's own scatter. Untruncated backends: 0.
    virtual double last_solve_trunc_de() const { return 0.0; }
    // Energy scale of the last solve set by its own stop thresholds: sqrt(r) for a stop on a
    // squared residual r, the change itself for a stop on an energy change (aldet: max(de,
    // sqrt(dr)); DMRG: sqrt of the final sweep's per-site Davidson threshold). Untracked backends: 0.
    virtual double energy_resolution() const { return 0.0; }
    // True if the last solve was armed for a warm restart but ran cold anyway (DMRG: the
    // basis-change rotation declined, or the host withheld it), rebuilding the wavefunction from
    // scratch: the energy steps there, and an orbital converger's history predates a surface that
    // no longer exists. A uniformly cold backend -- aldet re-solves every iteration -- is false.
    virtual bool last_solve_cold() const { return false; }

    // --- relating the wavefunction across an active-orbital-basis change (capability-gated) ---
    // All three operations need the same thing: representing/comparing the wavefunction
    // when the active orbitals are rotated. A determinant CI has explicit, index-comparable
    // CI vectors, so it rotates them (malmqvist), follows roots by overlap against the
    // previous iteration's vectors (calc_S), and does the linear-molecule pi-pair rotation.
    // An MPS backend CAN do these but only at real cost: block2 rotates an MPS across orbital
    // bases via exp(kappa) time evolution (logm(U) -> anti-Hermitian 1-body MPO -> td_dmrg;
    // see block2-preview docs/.../orbital-rotation.rst), and cross-basis overlap then needs
    // that rotation plus an identity-MPO sweep. Cheap for small near-converged steps, costly
    // for large rotations (e.g. localization). So such a backend typically advertises false and
    // re-solves from a warm-started MPS instead. calc_S across an UNCHANGED basis needs no
    // rotation, so a backend may implement that one alone (block2 does).
    virtual bool supports_civec_rotation() const { return false; }
    virtual void malmqvist(int i_set, double* U) {}                          // rotate CI vector by active-block U
    virtual void rotate_pi_pair(int i_set, double s, double c,               // linear-molecule pi-pair rotation
                                int pair, int* ind_pi) {}
    virtual void calc_S(double* S_track, int a, int b);                      // S[i*n_s+j] = <set a state i|set b state j>
    
        
    
    // --- dressed active-space operator import (capability-gated) ---
    // A backend that can encode a TOTAL dressed operator (F_act+g1, (tu|vw)+g2, g3, E_core+E0) as
    // its own eigenproblem and re-solve it advertises true; the DMRG/block2 backend does. Default is
    // a loud out-of-line abort (our contract, not a physics choice) so no backend silently drops the
    // dressing. h3_total may be null (no 3-body group); tensors are in the native active basis
    // (h2 chemist (tu|vw)), and the backend maps them onto its own lattice.
    virtual bool supports_dressed_import() const { return false; }
    virtual void import_dressed_operator(const double* h1_total, const double* h2_total,
                                         const double* h3_total, double const_total);

    // --- transition-density read-outs (capability-gated) ---
#if 0  // full transition 2-RDM: no consumer, the driver reads G2_calc_diag. Revive for first-order
       // properties -- the 2-body Mbar needs bra != ket densities between the dressed roots.
    // Full n_s x n_s state matrix of the spin-summed 2-RDM: G[(bra*n_s+ket)*n_act^4] blocks in
    // the GAMMA convention above; diagonal = per-state 2-RDM, off-diagonal = <bra|..|ket>.
    // Delocalized basis. Accumulates into G (caller zeroes it). Default aborts loudly.
    virtual bool supports_g2_full() const { return false; }
    virtual void G_calc_full(double* G);
#endif
    // Per-state spin-summed 3-body moment, native active basis, caller buffer n_act^6,
    // overwritten; layout G3[p,q,r,i,j,k] = <a+_p a+_q a+_r a_k a_j a_i>, spin-summed,
    // flat row-major over the six active axes. Default aborts loudly.
    virtual bool supports_g3_diag() const { return false; }
    virtual void G3_calc_diag(double* G3, int state);
    // Diagonal per-state spin-summed 2-RDMs: n_s consecutive n_act^4 blocks, native basis,
    // GAMMA convention as above. Overwritten, not accumulated. Default aborts loudly.
    virtual bool supports_g2_diag() const { return false; }
    virtual void G2_calc_diag(double* G);
#if 0  // DIRECT lambda3 path (superseded by the explicit lattice-3RDM route; revive for nact >~ 30)
    // Complementary six-operator overlap (3-RDM-free lambda3): per root r, overwritten,
    //   omega[r] = sum_{p,spins} Tbra[p,w,x,y] Tket[p,z,u,v] <r| x+_s y+_t w_t z+_q v_q u_s |r>.
    // Tensors [np][n_act^3], axes (external p, creation, free-spin annih., paired annih.),
    // active legs in the frozen lattice basis; unweighted. Default aborts loudly.
    virtual bool supports_h2caa_overlap() const { return false; }
    virtual void h2caa_overlap(const double* Tbra, const double* Tket, int np, double* omega);
    // Two pairs against the same roots, per-pair semantics exactly as above; gated by the same
    // query. A backend whose per-root density build dominates overrides this to build that density
    // once and contract both pairs against it. The default runs the single-pair route twice.
    virtual void h2caa_overlap2(const double* Tbra1, const double* Tket1, int np1, double* omega1,
                                const double* Tbra2, const double* Tket2, int np2, double* omega2) {
        h2caa_overlap(Tbra1, Tket1, np1, omega1);
        h2caa_overlap(Tbra2, Tket2, np2, omega2);
    }
#endif

    // --- IO / diagnostics ---
    virtual void gen_ext_ind() = 0;
    virtual void print_states(int a, int n_s, int print) = 0;
    virtual void write_civec(int i_s, char* name) = 0;

    // --- escape hatch ---
    // Determinant-coupled paths outside the SCF loop (PT/XMCQDPT) keep using aldet_data
    // directly. The aldet adapter returns its wrapped object; other backends return nullptr.
    virtual aldet_data* as_aldet() { return nullptr; }
};
