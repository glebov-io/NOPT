#ifndef NOPT_ALDET_RDM_ADJ_H
#define NOPT_ALDET_RDM_ADJ_H

// Adjoints of the determinant-CI density routines with respect to the CI vectors (analytic gradient of DSRG-PT2). Each routine is a
// transposed copy of the forward loops of aldet.cpp: where the forward accumulates  R += CI_0 * CI_e * sign,  the adjoint accumulates
// cb_0 += Rb * sign * CI_e  and  cb_e += Rb * sign * CI_0. Layouts and arguments are those of the forward calls (state stride n_s, ld = n_s);
// (i_th, n_th) split the outer string loop like the forward workers (cb must then be private to the thread).
#include <vector>

// aldet_calc_DM_2body_AA_diag / _AB_diag (diagonal, n_s states): Gb n_s x N^4 state-major
void dm2_AA_diag_vjp(const double * Gb, int n_s, const double * ci, double * cb, int N, int na, int Na, int Nb, int * fa, int * vec_a, int i_th = 0, int n_th = 1);
void dm2_AB_diag_vjp(const double * Gb, int n_s, const double * ci, double * cb, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b, int i_th = 0, int n_th = 1);
// aldet_calc_DM_3body_AAA / _AAB as G3_calc_diag calls them (one state, element 0 of a stride-ld coefficient block): Gb N^6
void dm3_AAA_vjp(const double * Gb, int ld, const double * ci, double * cb, int N, int na, int Na, int Nb, int * fa, int * vec_a, int i_th = 0, int n_th = 1);
void dm3_AAB_vjp(const double * Gb, int ld, const double * ci, double * cb, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b, int i_th = 0, int n_th = 1);

class CasRef;
// cb (nvec x Nd, state-major) += d/dc of  sum_I w_I [ <g1b, gamma(c_I)> + <G2b, G2(c_I)> + <G3b, G3(c_I)> ]  for the densities of the determinant CI:
// gamma = calc_DM_diag (spin-summed, [t*na+u]), G2 = G2_calc_diag (= G_calc: AA alpha + AA beta + AB), G3 = G3_calc_diag (<p+ q+ r+ k j i>, na^6).
// Any seed may be null. c: nvec x Nd state-major.
void aldet_rdm_vjp(const CasRef & R, const double * c, int nvec, const double * w, const double * g1b, const double * G2b, const double * G3b, double * cb);
// Dot-product test of aldet_rdm_vjp against central differences of the production routines (calc_DM_diag, G_calc, G3_calc_diag). Returns the failures.
int aldet_rdm_vjp_selftest(const CasRef & R, const double * c);

#endif
