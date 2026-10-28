#ifndef NOPT_RI_GRADIENT_H
#define NOPT_RI_GRADIENT_H

// Reverse-mode (adjoint) rules for the density-fitting layer (analytic gradients).
//
//   g^RI = L V^-1 L^T,   L_{x P} = (x|P), x = (mu nu) an ordered AO pair, V_{PQ} = (P|Q).
//
// Layouts are row-major as in NOPT: L[(a*n + b)*naux + P] (the RI_AA array), V[P*naux + Q].
// Everything assumes V is positive definite with full rank: the forward code (S05_calc) drops
// only non-positive eigenvalues, and the derivative of a moving retained subspace is not
// implemented. ri_check_full_rank() tells the caller whether that assumption holds.
//
// Memory: O(n^2 * naux), the size of L itself. No n^4 array is formed.

// Smallest eigenvalue of V (V is not modified). Gradient drivers refuse V with min eigenvalue <= tol.
double ri_min_eigenvalue(int naux, const double * V);
// Both ends of the spectrum (condition number lmax/lmin), for diagnostics.
void ri_eigenvalue_range(int naux, const double * V, double * lmin, double * lmax);

// The adjoint for an explicit pair-space seed Q (npair x npair, general, row-major). For tests only:
//   dE = sum_{xy} Q_xy dg_xy  =>  Lbar = (Q + Q^T) L V^-1,  Vbar = -sym(V^-1 L^T Q L V^-1),
// so that dE = sum Lbar_{xP} dL_{xP} + sum Vbar_{PQ} dV_{PQ} (V varied symmetrically). Both are overwritten.
void ri_vjp_dense(int npair, int naux, const double * L, const double * Vinv, const double * Q,
                  double * Lbar, double * Vbar);

// Seeds of the Coulomb and exchange contractions of a symmetric density P in the fitted integrals:
//   E = cJ sum P_ab P_cd g_abcd + cK sum P_ac P_bd g_abcd,   g = L V^-1 L^T.
// Lbar and Vbar as for ri_vjp_dense, without ever forming Q. Lbar (n*n*naux) and Vbar (naux*naux)
// are overwritten. Takes V itself (not V^-1): the work is done in the eigenbasis of V, because an
// explicit V^-1 and the product V^-1 M V^-1 lose the digits of the near-null modes of the metric.
void ri_jk_vjp(int n, int naux, const double * L, const double * V, const double * P,
               double cJ, double cK, double * Lbar, double * Vbar);

// The adjoint of F(V) = V^-1/2 for symmetric positive definite V (n x n):
//   given Fbar = dE/dF (any matrix; only its symmetric part acts), Vbar = D[V^-1/2]^* (Fbar),
// with dE = sum Vbar_{PQ} dV_{PQ}. Divided differences are used for the eigenvalue pairs, with the
// derivative f'(l) = -l^-3/2 for (near-)repeated eigenvalues.
void ri_inv_sqrt_vjp(int n, const double * V, const double * Fbar, double * Vbar);

// Coulomb and exchange matrices of a symmetric density D in the fitted integrals, with V^-1 handled in the eigenbasis:
//   J_ab = sum_cd D_cd (ab|cd),   K_ab = sum_cd D_cd (ac|bd)          (n x n, overwritten)
void ri_fock_JK(int n, int naux, const double * L, const double * V, const double * D, double * J, double * K);

// Adjoint of the active-space two-electron energy  E_lam = 1/2 sum_{tuvw} lam_tuvw (tu|vw)_RI  (lam on the active
// MO pairs, (tu|vw) = chemists', symmetrized internally over the pair exchange). Cact[a*na + t] are the active MO
// coefficients in the AO basis. Adds to Lbar (n*n*naux) and Vbar (naux*naux) the seeds of E = Q : (L V^-1 L^T) (never forming the
// pair-space Q), and, if GC != null (n x na, [a*na+t]), the derivative of E_lam with respect to the active
// coefficients (the unconstrained orbital gradient, input of W). *E receives the energy.
void ri_active_vjp(int n, int naux, const double * L, const double * V, const double * Cact, int na, const double * lam,
                   double * Lbar, double * Vbar, double * GC, double * E);

// ---- the same rules on a cached eigendecomposition of the metric (for drivers that call them many times) ---------------
// V = U^T diag(lam) U (rows of U are eigenvectors), Lt = L U^T ([ab*naux + i]). The *_eig functions work on seeds in this eigenbasis
// ("Lbt", "Vbt", same layouts as Lbar/Vbar) and ACCUMULATE into them; ri_eig_to_aux maps them back (Lbar = Lbt U, Vbar = U^T Vbt U).
#include <vector>
struct RIEig{
    int n = 0, naux = 0;
    std::vector<double> U, lam, Lt;
    void build(int n, int naux, const double * L, const double * V);
    bool empty() const { return Lt.empty(); }
};
void ri_eig_to_aux(const RIEig & E, const double * Lbt, const double * Vbt, double * Lbar, double * Vbar, bool accumulate);
// The same in place (row batches; no second n^2 naux array), and the inverse map accumulating (Lbt += Lbar U^T, Vbt += U Vbar U^T).
void ri_eig_to_aux_inplace(const RIEig & E, double * Lbar, double * Vbar);
void ri_aux_to_eig(const RIEig & E, const double * Lbar, const double * Vbar, double * Lbt, double * Vbt);
// J[D], K[D] (overwritten); D2 (may be null): also J2 = J[D2], K2 = K[D2] in the same pass
void ri_fock_JK_eig(const RIEig & E, const double * D, double * J, double * K, const double * D2 = nullptr, double * J2 = nullptr, double * K2 = nullptr);
// The same for a density given also in factored form D = X Y^T (X, Y: n x r, row-major [a*r+k]): J from D, the exchange from the factors at
// 3 n^2 r per auxiliary function (instead of 2 n^3). For the orbital densities of the CAS-SCF (r = n_core + n_act ... ).
void ri_fock_JK_eig_lr(const RIEig & E, const double * D, const double * X, const double * Y, int r, double * J, double * K);
// Seeds of the SYMMETRIC BILINEAR form  B(P,Q) = cJ sum P_ab Q_cd g_abcd + cK sum P_ac Q_bd g_abcd  (P, Q symmetric); B(P,P) is the
// energy of ri_jk_vjp, and the directional derivative of the ri_jk_vjp seeds of P along dP is 2 x the seeds of B(P, dP). Lbt/Vbt may be null.
void ri_jk_vjp_eig(const RIEig & E, const double * P, const double * Q, double cJ, double cK, double * Lbt, double * Vbt);
// Active transform La[(t*na+u)*naux + i] = (Ca^T Lt_i Ca)_tu ;  with dCa != null: the derivative dLa along dCa instead
void ri_active_La(const RIEig & E, const double * Cact, int na, double * La, const double * dCa = nullptr);
// ri_active_vjp on the cache (seeds in the eigenbasis, accumulated). La: from ri_active_La (or null: computed here).
void ri_active_vjp_eig(const RIEig & E, const double * Cact, int na, const double * lam, const double * La,
                       double * Lbt, double * Vbt, double * GC, double * Eout);
// Directional derivative of the outputs of ri_active_vjp_eig (Lbt, Vbt, GC) along (dCa, dlam): dLbt/dVbt accumulated (may be null),
// dGC (n x na) overwritten (may be null). La, dLa: from ri_active_La.
void ri_active_jvp_eig(const RIEig & E, const double * Cact, const double * dCa, int na, const double * lam, const double * dlam,
                       const double * La, const double * dLa, double * dLbt, double * dVbt, double * dGC);

#endif
