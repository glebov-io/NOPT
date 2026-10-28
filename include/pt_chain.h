#ifndef NOPT_PT_CHAIN_H
#define NOPT_PT_CHAIN_H

// Chain the adjoints of the PT inputs (PTInputAdj, from the adjoints of the PT blocks) back to the quantities the nuclear
// gradient is built from.
//
//   B-chain:  B_XY[p,q,k] = sum_ab C_X[p,a] C_Y[q,b] Bhat[a,b,k],   Bhat = L V^-1/2          (RI_data::MO_calc)
//   H-chain:  H_XY[p,q]   = sum_ab C_X[p,a] Hc[a,b] C_Y[q,b],       Hc = h + 2J[DM_C] - K[DM_C]   (the core-dressed h)
//             with DM_C = sum_{i in core} C_i C_i^T and J, K the RI Coulomb/exchange matrices.
//
// Outputs (accumulated, caller zeroes): Cbar (n_mo x n_ao, rows like MO_VEC; the UNCONSTRAINED derivative with respect to
// the MO coefficients), Lbar (n_ao^2 x naux, like RI_AA), Vbar (naux x naux, symmetric, the derivative with respect to the
// Coulomb metric V itself -- V^-1/2 is differentiated through ri_inv_sqrt_vjp), hbar (n_ao x n_ao, symmetric, the derivative with
// respect to the AO core Hamiltonian h = T + V_nuc). Contract Lbar and Vbar with the (P|ab), (P|Q) derivative integrals and
// hbar with T' and V'; Cbar feeds the orbital response / the W term of the Lagrangian.
#include <vector>
#include "pt_input_adj.h"

struct PTChainOut{
    std::vector<double> Cbar, Lbar, Vbar, hbar;
    void resize(int nmo, int n, int naux){
        Cbar.assign((size_t)nmo * n, 0.0); Lbar.assign((size_t)n * n * naux, 0.0); Vbar.assign((size_t)naux * naux, 0.0); hbar.assign((size_t)n * n, 0.0);
    }
};

// C: n_mo x n_ao (rows = MOs: core, active, virtual). L: n_ao^2 x naux. V: naux x naux. A: adjoints of B and H.
// Eeig (optional): output Lbar/Vbar in the eigenbasis of V (as the gradient drivers keep them); without it, in the auxiliary basis.
int pt_chain_B(int n, int naux, int nc, int na, int nv, const double * C, const double * L, const double * V,
               const PTInputAdj & A, PTChainOut * out, const struct RIEig * Eeig = nullptr);
// Ecache (optional): eigendecomposition of V and L U^T for these L, V (RIEig::build); saves its rebuild and the polarization copies.
int pt_chain_H(int n, int naux, int nc, int na, int nv, const double * C, const double * L, const double * V, const double * h,
               const PTInputAdj & A, PTChainOut * out, const struct RIEig * Ecache = nullptr, bool eig_out = false);


#endif
