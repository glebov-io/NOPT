#ifndef NOPT_PT_CANON_H
#define NOPT_PT_CANON_H

// The adjoint of the semicanonicalization of the CAS-SCF orbitals that the PT stages (XMCQDPT2, DSRG-PT2) start from.
//
// CAS_engine::make_canonical builds the state-averaged Fock matrix F_SA = h + J[D] - 1/2 K[D] (RI), D = 2 sum_core c c^T +
// sum_tu gamma_tu c_t c_u^T, and diagonalizes its core-core and virtual-virtual blocks (molecule::diag_X_MO_block): the PT
// reads the rotated orbitals C_sc and the eigenvalues eps_c, eps_v. The PT energy is NOT invariant under rotations inside
// these blocks (diagonal eps), so the energy as a function of the stationary orbitals C_0 is E(C_sc(C_0), eps(C_0)).
//
// With F_X = C_X F C_X^T = U^T diag(eps) U (rows of U = eigenvectors), C_sc = U C_0, and  Omega = dU U^T,
// Omega_ij = (u_i^T dF u_j) / (eps_i - eps_j), the adjoint (in the gauge of C_sc, i.e. the orbitals every other chain uses) of
//   phi = <Cbar_sc, C_sc> + <ebar, eps>
// is, with M_ij = sum_mu Cbar_sc[i,mu] C_sc[j,mu] and Fbar_sc = diag(ebar) + 1/2 K, K_ij = (M_ij - M_ji)/(eps_i - eps_j):
//   Cbar_extra_X = 2 Fbar_sc C_X F ,   Fbar_AO = C_X^T Fbar_sc C_X   (the caller adds Cbar_sc itself),
// and Fbar_AO is chained to h, D (-> C core/active, gamma_SA) and (L, V) with  Dbar = J[Fbar] - 1/2 K[Fbar].
// Pairs closer than `degtol` in eps are dropped (the PT energy is invariant under rotations inside a degenerate subspace,
// which makes M_ij - M_ji vanish there). Everything is accumulated into `out` (Cbar rows like MO_VEC, hbar, Lbar, Vbar) and gbar
// (n_act^2, the adjoint of the state-averaged 1-RDM; NOT symmetrized, entries are independent).
#include "pt_chain.h"

// C: n_mo x n_ao semicanonical orbitals (core, active, virtual). eps: n_mo orbital energies (only core and virtual used).
// Cbar_sc: n_mo x n_ao adjoint of the PT w.r.t. C_sc (only the core and virtual rows are used here); ec_bar (nc), ev_bar (nv).
// Cbar_sc may be out->Cbar itself: it is read first, and out->Cbar then holds Cbar_sc + the extra terms (the total adjoint).
// ea_bar (optional, n_act): adjoints of the eigenvalues of the ACTIVE block of F_SA (HOMO / USE_ORB_FOR_ENERGY modes, where every eps_a is one such eigenvalue); the
// active orbitals are canonical in F_SA and the energy is invariant under their rotations (equal eps_a), so only the Hellmann-Feynman terms are added.
// ci_rot_g (optional, n_act^2, antisymmetric): g_tu = <cbar | G_tu | c>, the derivative of the energy with respect to the rotation of the ACTIVE orbitals t -> u
// applied to the CI vectors (a per-orbital eps_a makes the energy depend on the active gauge, which the canonicalization fixes); with it the active block gets the
// full 1/(eps_t - eps_u) rotation term, orbitals and CI together. Requires ea_bar.
// gamma_sa: n_act^2. F_out (optional, n_ao^2): F_SA in the AO basis as rebuilt here (for diagnostics).
// fbar_mo (optional, n_mo x n_mo): adjoint of the Fock matrix in the semicanonical MO basis, C_sc F C_sc^T (e.g. the DSRG cross blocks f_ca, f_cv, f_av);
// its orbital part joins Cbar_sc before the rotation terms are formed.
int pt_canon_vjp(int n, int naux, int nc, int na, int nv, const double * C, const double * eps, const double * L, const double * V,
                 const double * h, const double * gamma_sa, const double * Cbar_sc, const double * ec_bar, const double * ev_bar,
                 double degtol, PTChainOut * out, double * gbar, double * F_out = nullptr, const double * ea_bar = nullptr, const double * ci_rot_g = nullptr,
                 const struct RIEig * Ecache = nullptr, const double * fbar_mo = nullptr,
                 bool eig_out = false);   // eig_out: Lbar/Vbar accumulated in the eigenbasis of V (the convention of the gradient drivers)


#endif
