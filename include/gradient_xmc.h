#ifndef NOPT_GRADIENT_XMC_H
#define NOPT_GRADIENT_XMC_H

// Analytic gradient of XMCQDPT2 root energies, RI, determinant CI, no ifitd / d_only. The computed energy is differentiated, including the
// energy-grid interpolation of res_fit (exact derivative of the computed interpolation: nodes, weights, evaluation points; 08_review_decisions #2).
// The chain, from the final eigenproblem backwards, in the frame of QDPT2 (semicanonical orbitals C_sc and the CI vectors c1 = malmqvist(U) c_CAS):
//   E_n   = eigenvalue of H2 = H01 + sym(C^T E2_PP C)                v = its eigenvector, Hbar2 = v v^T          (Hellmann-Feynman)
//   H01   = <c1|H_CAS|c1>:  v^T H01 v = <psi|H|psi>, psi = sum v_a c1_a                                      (bare objects of one vector)
//   E2_PP = res_fit contraction of the resolvent tables with c2 = C c1 and the grid weights                  (res_fit_E2_adjoint)
//   tables(f) = the raw EE block outputs with every denominator shifted by E_fit[f]                           (pt_vjp_EE_<block>, pt_vjp_shift)
//   grid  = from min/max of E0 and E_act                                                                      (rf_grid_vjp)
//   (C, E0) = eigenvectors / eigenvalues of H0PP[a,b] = sum_I c1_Ia c1_Ib E_act(I),  E_act(I) = sum_t eps_a[t] n_t(I)
//   B, H blocks -> C_sc, L, V, h (pt_chain);  eps and the orbitals -> F_SA (pt_canon_vjp: core, ACTIVE, virtual; the active rotation carries the
//   CI vectors: ci_rot_g = T - T^T, T_tu = sum_a <cbar1_a|E_tu|c1_a>)
//   and the response of the SA-CASSCF reference with the zero-weight model states (cas_lagrangian_objects_cbar).
// run_grad sets g_xmc_grad.root; QDPT2 calls the hooks.
#include <vector>
#include "casscf_resp.h"
#include "inp_par_read.h"

class res_fit_data;
class RI_data;
class doCI_data;

struct XmcGradRequest{
    int root = -1;                       // root of the final XMCQDPT2 eigenproblem; < 0: not requested
    int status = 0;                      // nonzero: refused or failed (message printed)
    bool done = false;
    CasRef * ref = nullptr;              // reference snapshot taken at the start of QDPT2
    CasObjects total;                    // objects for the derivative integrals (direct + response)
    std::vector<double> g;               // 3 n_atoms, after xmc_grad_contract
    double E_root = 0.0;                 // the energy being differentiated, as rebuilt by the gradient
};
extern XmcGradRequest g_xmc_grad;

// At the start of QDPT2 (before anything is rotated): checks the supported options, takes the reference snapshot.
int xmc_grad_snapshot(molecule * M, xmc_par * xmc);
// Right after RF.E2_calc(E2) (the tables, grid and rotated CI vectors are still in place). C, E0: H0PP eigenvectors (rows) and eigenvalues; U: the active
// rotation of three_block_diagonalization; eps: orbital energies as the PT used them; H01, E2PP: as QDPT2 built them.
int xmc_grad_main(molecule * M, xmc_par * xmc, doCI_data & D, res_fit_data & RF, RI_data & RI, double * eps, const double * U, const double * C,
                  const double * E0, const double * H01, const double * E2PP, double * H_AV, double * H_CA, double * H_CV);
// After the final eigenproblem: the energy QDPT2 reports for the root must be the one differentiated.
void xmc_grad_check_final(const double * E, int n_s);
// After nopt_deriv_engines_initialize().
int xmc_grad_contract(molecule * M);

#endif
