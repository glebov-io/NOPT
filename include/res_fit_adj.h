#ifndef NOPT_RES_FIT_ADJ_H
#define NOPT_RES_FIT_ADJ_H

// Adjoint of the contraction of the XMC resolvent tables with the CI vectors (res_fit.cpp: res_fit_calc_0body ... 3body_AB, called by res_fit_data::E2_calc):
// E2[i_s,j_s] is linear in the tables RF_PS, RF_PH, RF_PV_JK, RF_PV_AB, RF_P3_JK, RF_P3_AB (energy grid innermost); for weights W[i_s,j_s] on the
// elements of E2 the seeds Rbar are the derivatives of sum W E2 with respect to every table entry, in the layout of the tables. Exact (a copy of the loops with
// the accumulation transposed), no symmetry assumed.
#include <vector>
#include <functional>
#include "common_vars.h"

class res_fit_data;

struct ResFitSeeds{
    std::vector<double> PS, PH, PV_JK, PV_AB, P3_JK, P3_AB;     // N_fit innermost
};

// Outputs of one contraction routine (each pointer may be null): Rbar - seeds of the tables (layout of the tables), Rfwd - the FORWARD tables (needed for the
// CI and weight seeds), cibar - derivative with respect to the CI vectors (layout of the vectors handed to the routine), Cbar - derivative with respect to the
// interpolation weights C_fit, [(det*n_s + i_s)*(2 n_fit_pol+1) + l].
// Rbar may hold a window of the grid only: grid points f0 .. f0 + nb - 1, [index * nb + f - f0] (the whole grid: f0 = 0, nb = Nf).
struct ResFitAdjOut{
    double * Rbar = nullptr;
    const double * Rfwd = nullptr;
    double * cibar = nullptr;
    double * Cbar = nullptr;
    int Nf = 0, f0 = 0, nb = 0;
};

// The complete derivative of  Phi = sum_{ij} W_ij E2_ij  (E2 as res_fit_data::E2_calc contracts it, before any later symmetrization or rotation):
//   tables      - seeds of the resolvent tables (R-bar, N_fit innermost),
//   cibar       - derivative with respect to ci->coef[0] ([det*n_s + s], the alpha and beta variants folded together),
//   e0bar       - derivative with respect to E0[alpha] and Eactbar - with respect to E_act(det) (det = a*Nb + b), through the evaluation points x = E0 - E_act,
//                 the interpolation weights and the energy grid (exact derivative of the computed interpolation).
struct ResFitAdjResult{
    ResFitSeeds tables;
    std::vector<double> cibar, e0bar, Eactbar;
};

int res_fit_adj_0body(const double * W, int n_s, int ld, double * ci, int Na, int Nb, const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th);
int res_fit_adj_1body(const double * W, int n_s, int ld, double * ci, int N, int na, int Na, int Nb, int * fa, int * vec_a, const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th);
int res_fit_adj_2body_AA(const double * W, int n_s, int ld, double * ci, int N, int na, int Na, int Nb, int * fa, int * vec_a, const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th);
int res_fit_adj_2body_AB(const double * W, int n_s, int ld, double * ci, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b, const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th);
int res_fit_adj_3body_AAA(const double * W, int n_s, int ld, double * ci, int N, int na, int Na, int Nb, int * fa, int * vec_a, const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th);
int res_fit_adj_3body_AB(const double * W, int n_s, int ld, double * ci, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b, const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th);

// All of E2_calc's contractions at once (the pair of coef / coef_bas variants).
// with_3body = false leaves tables.P3_JK / P3_AB empty (the CI and grid derivatives still include the three-body terms); their seeds then come
// window by window from res_fit_E2_adjoint_3body. The contraction loops run on all OpenMP threads.
int res_fit_E2_adjoint(res_fit_data & RF, const double * W, ResFitAdjResult * R, bool want_ci_and_grid = true, bool with_3body = true);
// Seeds of the three-body tables (RF_P3_JK, RF_P3_AB layouts, N_fit replaced by the window) for grid points f0 .. f0 + nb - 1.
int res_fit_E2_adjoint_3body(res_fit_data & RF, const double * W, int f0, int nb, double * P3J, double * P3A);
// Dot-product tests against the production contractions: tables (random), CI vectors and E0 / E_act (finite differences). NOPT_XMC_ADJ_TEST=1 in QDPT2. Overwrites the tables.
int res_fit_adj_selftest(res_fit_data & RF);

#endif
