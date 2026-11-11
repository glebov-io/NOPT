#ifndef NOPT_INTEGRAL_DERIVATIVES_H
#define NOPT_INTEGRAL_DERIVATIVES_H

// Nuclear-coordinate derivatives of the AO integrals, contracted with caller-supplied weights
// (analytic gradients). Needs a libint built with first derivatives (see
// tests/gradients/libint-deriv.sh); the energy-only libint of the published images has none.
//
// Conventions
//   * g is the gradient with respect to the atom Cartesian coordinates in bohr, layout
//     g[3*atom + xyz], atom index as in molecule::atom_coord. Every function ADDS to g.
//   * AO matrices are in the Cartesian AO basis of molecule::s / molecule::ri_s (the basis the
//     integrals are computed in, n_ao), i.e. before the D5 transformation to pure functions.
//   * A weight matrix is contracted as given: sum_ab W[a*n_ao+b] * d(a|O|b)/dx. It does not have to
//     be symmetric, so the caller states exactly which sum it wants differentiated.
//   * Several shells on one atom, and both bra and ket on the same atom, add up: every shell
//     centre is mapped to its atom through molecule::shell_center / ri_shell_center.
//   * No screening beyond libint's own: the engines are built with machine-epsilon precision.
//   * Not supported (the callers refuse them): ECP, external field, symmetry-adapted bases.
//
// The two-electron contractions take a functor that fills the weights of one shell block, so no
// n_ao^4 (or n_ao^2 n_aux) array has to exist. For a block of shells sh[0..3] with first AO
// off[0..3] and sizes n[0..3] the functor writes w[((a*n1+b)*n2+c)*n3+d] (row-major, the order of
// libint's result buffer) for a = 0..n0-1, ... The sum is over ALL shell quartets/triples/pairs,
// without assuming any permutational symmetry of the weights.

#include <functional>
#include "molecule.h"

typedef std::function<void(const int * sh, const int * off, const int * n, double * w)> deriv_weight_fn;

// Returns 0 if the integral derivatives cover this molecule's one-electron Hamiltonian, else the
// number of unsupported features (each reported on out_stream): ECP, external electric field,
// symmetry-adapted run. Gradient drivers call it first and refuse to continue silently.
int deriv_check_supported(molecule * M);

// per-thread first-derivative engines; call after nopt_initialize() and the molecule is read.
// max_nprim / max_l are taken from the shells (not from GTO_MAX), which keeps the scratch small.
int nopt_deriv_engines_initialize(molecule * M);
int nopt_deriv_engines_finalize();

// sum_ab W_ab dS_ab/dx  and  sum_ab W_ab dT_ab/dx
int deriv_overlap_contract(molecule * M, const double * W, double * g);
int deriv_kinetic_contract(molecule * M, const double * W, double * g);

// sum_ab W_ab dV_ab/dx, V = sum_C -Z_C/|r-R_C|: derivatives of the two shell centres AND of the
// charge positions (the operator centres), which fall on the atoms carrying the charges.
int deriv_nuclear_contract(molecule * M, const double * W, double * g);

// V_NN = sum_{i<j} Z_i Z_j / R_ij, with the pairs that molecule's E_nuc_calc includes.
int deriv_nuclear_repulsion(molecule * M, double * g);

// sum_{abcd} G_abcd d(ab|cd)/dx; w is filled by the functor.
int deriv_eri_contract(molecule * M, const deriv_weight_fn & weights, double * g);

// sum_{P a b} G_Pab d(P|ab)/dx and sum_{P Q} G_PQ d(P|Q)/dx over the auxiliary shells molecule::ri_s.
// In the three-index case sh[0] is the auxiliary shell, sh[1], sh[2] are AO shells.
int deriv_ri3_contract(molecule * M, const deriv_weight_fn & weights, double * g);
// Diagnostic: element-wise translational invariance of every 3-center derivative shell triplet; prints failures, returns their number.
int deriv_ri3_translation_check(molecule * M, int max_report);
int deriv_ri2_contract(molecule * M, const deriv_weight_fn & weights, double * g);

// Convenience: G_abcd = cJ D_ab D_cd + cK D_ac D_bd (the J and K contractions of a density D).
int deriv_eri_JK_contract(molecule * M, const double * D, double cJ, double cK, double * g);

#endif
