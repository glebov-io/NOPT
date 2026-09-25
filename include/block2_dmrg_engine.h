#pragma once
//
// Internal state + shared helpers for the block2 DMRG backend. Heavy block2 headers live
// here, so include this ONLY from the block2 backend TUs (block2_casci_wrap / block2_dmrg /
// block2_mps_to_det). The block2-free public interface stays in block2_casci_wrap.h.

// block2 spells MKL's complex types as std::complex and then includes mkl.h. MKL's headers must
// agree, so claim the typedefs before anything pulls them in: otherwise blas_link.h's mkl_lapack.h
// declares crot/zrot with MKL's native struct and block2's mkl_blas.h re-declares the same
// extern "C" functions with std::complex. MKL guards both with #ifndef for exactly this.
#ifdef _MKL
#include <complex>
#define MKL_Complex8  std::complex<float>
#define MKL_Complex16 std::complex<double>
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <limits>
#include <vector>
#include <omp.h>

#include "block2_casci_wrap.h"   // block2_casci_wrap class + dmrg_par (via inp_par_read.h)
#include "common_vars.h"         // out_stream
#include "blas_link.h"           // BLAS thread-count API (OpenBLAS or MKL)

#include "block2_core.hpp"
#include "block2_dmrg.hpp"

using namespace block2;

// Process-wide engine counter. Engines alive at the same time must not share MPS scratch tags.
inline int next_dmrg_engine_id() {
    static std::atomic<int> counter{0};
    return counter++;
}

// -------------------------- engine: all block2 state ----------------------------------
struct dmrgci_engine {
    dmrg_par cfg;
    // twos = 2S (total spin) drives the spin-adapted SU2 solve; twosz = 2*M_S = na-nb is the
    // projection the determinant read-out expands to, so it matches the native (aldet) M_S.
    int n_act, n_elec, twos, twosz, mult, n_s, print_number;
    SU2 target;                       // active-space symmetry sector (C1: pg = 0)
    std::vector<uint8_t> orbsym;      // per-orbital irrep; C1 -> all 0 (set in set_act_rep_num)
    std::vector<double> E_states;     // per-state energies (returned by E_states_ptr)
    bool storage_ready = false;       // has init_state_storage run?

    // Localization. Empty/off => solve in the given basis.
    std::vector<double> U_loc;        // n_act x n_act, [a*n_act+p]; valid when localize_on
    bool localize_on = false;
    std::vector<double> F_loc, g_loc; // integrals rotated into the localized basis (when localize_on)

    // Active-block canonicalization (eigenvectors of the active Fock, [a*n_act+p]) supplied by the
    // host before the final print; rotates the reported leading determinants into the canonical basis.
    std::vector<double> U_canon;
    bool have_canon = false;

    // Warm-start (localization-rotation MPS reuse). R_active takes the previous macro-iteration's
    // active basis to the current one; valid only when have_rotation. Consumed by the next solve().
    std::vector<double> R_active;     // n_act x n_act, [a*n_act+p]
    bool have_rotation = false;

    // MPO simplification rule resolved once from cfg.low_m_opt: -1 unresolved, 0 off, 1 on.
    int low_m_opt_res = -1;

    // block2 objects for the current macro-iteration (rebuilt each import_integrals)
    std::shared_ptr<FCIDUMP<double>> fcidump;
    std::shared_ptr<HamiltonianQC<SU2, double>> hamil;
    std::shared_ptr<MPO<SU2, double>> mpo;
    std::shared_ptr<MultiMPSInfo<SU2>> mps_info;  // persists solve -> RDM read-out
    std::shared_ptr<MultiMPS<SU2, double>> mps;   // the converged (state-averaged) wavefunction
    std::vector<double> d2_av;                    // state-averaged block2 2-RDM: one n_act^4 block
    std::vector<double> d2_states;                // per-state block2 2-RDM: n_s blocks of n_act^4, lattice order
    std::vector<double> d1_states;                // per-state 1-RDM: n_s blocks of n_act^2
    std::vector<double> w_state;                  // host SA weights (n_s); empty => equal weights
    bool d2_valid = false;                        // are d2_av/d2_states/d1_states current for this solve?
    std::vector<double> dmfull_cache;             // full n_s x n_s spin-summed 1-RDM (properties), delocalized
    bool dmfull_valid = false;                     // is dmfull_cache current for this solve?

    // Bare-state snapshot for the dressed re-solve overlap: one persistent single-root MPS per
    // root plus its scratch tag. snap_set is the storage slot calc_S answers for (-1 = none);
    // dressed_mpo marks e.mpo as a dressed general MPO, never to be rebuilt from the bare FCIDUMP.
    std::vector<std::shared_ptr<MPS<SU2, double>>> snap_mps;
    std::vector<std::string> snap_tags;
    int snap_set = -1;
    bool dressed_mpo = false;

    const int engine_id = next_dmrg_engine_id(); // MPS tag namespace of this engine
    int solve_count = 0;                        // macro-iteration index -> unique MPS tag
    int last_n_sweeps = 0;                      // sweeps actually run in the last solve
    double last_sweep_dE = 0.0;                 // |dE| between the final two sweeps (achieved convergence)
    bool last_hit_max = false;                  // last solve used its full sweep budget with dE > sweep_tol
    std::vector<uint16_t> reorder_perm;         // DMRG lattice order (Fiedler); empty => input order
    double last_ord_drift = std::numeric_limits<double>::quiet_NaN(); // pinned order's cost over a
                                                // freshly derived one's; NaN until an order is pinned
    bool last_cold_fallback = false;            // a warm-armed solve that still ran cold
    double last_dw = 0.0;                       // max discarded weight over the last solve's two-site sweeps at the
                                                // schedule's final bond dim, noise-free sweeps preferred
    double last_two_dot_dw = std::numeric_limits<double>::quiet_NaN(); // discarded weight of the last
                                                // two-site sweep: the truncation the stored MPS carries
    std::vector<double> last_two_dot_E;         // last two-site sweep's energy per root
    double last_trunc_de = 0.0;                 // stored MPS's RDM energy minus last_two_dot_E, max over roots
    double last_resolution = 0.0;               // sqrt of the final sweep's Davidson threshold: the solve's energy scale

    dmrgci_engine(int n_act_, int n_elec_, int twos_, int twosz_, int mult_, int n_s_,
                  int print_number_, const dmrg_par &c)
        : cfg(c), n_act(n_act_), n_elec(n_elec_), twos(twos_), twosz(twosz_), mult(mult_),
          n_s(n_s_), print_number(print_number_), target(n_elec_, twos_, 0), orbsym(n_act_, 0),
          E_states(n_s_, 0.0) {}
};

// -------------------------- shared block2-backend helpers ----------------------------------
namespace nopt_block2 {

// BLAS thread count, spelled per backend. Mirrors the pairing blas_link.h uses: OpenBLAS reports
// the current count, MKL the current maximum.
inline int blas_get_num_threads() {
#if defined(_OPENBLAS)
    return openblas_get_num_threads();
#elif defined(_MKL)
    return mkl_get_max_threads();
#else
#error "no BLAS backend selected: define _OPENBLAS or _MKL"
#endif
}

inline void blas_set_num_threads(int n) {
#if defined(_OPENBLAS)
    openblas_set_num_threads(n);
#elif defined(_MKL)
    mkl_set_num_threads(n);
#endif
}

// Pin block2 to the host OpenMP/BLAS thread count on entering a block2 region; restore on exit.
struct host_threads_guard {
    int omp_saved;
    int blas_saved;
    host_threads_guard() : omp_saved(omp_get_max_threads()), blas_saved(blas_get_num_threads()) {}
    ~host_threads_guard() {
        omp_set_num_threads(omp_saved);
        blas_set_num_threads(blas_saved); // restore the host's BLAS count independently of OMP
    }
    host_threads_guard(const host_threads_guard &) = delete;
    host_threads_guard &operator=(const host_threads_guard &) = delete;
};

void ensure_block2_runtime(const std::string &save_dir_root, double memory_gb,
                           double main_stack_gb, int n_threads);
void remove_tag_files(const std::string &tag);
void assert_stack_clean(const char *where);

// One root of the state-averaged MultiMPS as a plain single-root MPS at a one-site end-center.
// Defined in block2_dmrg.cpp; shared with the transition-RDM / overlap read-outs.
std::shared_ptr<MPS<SU2, double>>
extract_root_single(dmrgci_engine &e, int st, const std::string &xtag, const std::string &stag);

// One root pair's spin-summed N-body density in block2's lattice order, from one general-NPDM
// Expect sweep on transient single-root extracts. The result is unscaled (block2's convention);
// callers apply sqrt(2)^N and their own gathers.
std::shared_ptr<GTensor<double>> npdm_lattice(dmrgci_engine &e, int N, int ket_state,
                                              int bra_state, const char *tag);


// State-averaged 2-RDM, the per-state 2-RDMs, 1-RDMs and energies, once per solve.
void ensure_2rdm(dmrgci_engine &e);

// Fit a lower-bond-dim copy of an MPS (identity-MPO Linear) — a cheaper TRIE for the read-out.
// Defined engine-side; called from the read-out TU.
std::shared_ptr<MPS<SU2, double>>
compress_single_mps(dmrgci_engine &e, const std::shared_ptr<MPS<SU2, double>> &ket,
                    int target_m, const std::string &ctag);

} // namespace nopt_block2
