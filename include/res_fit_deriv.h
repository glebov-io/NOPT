#ifndef NOPT_RES_FIT_DERIV_H
#define NOPT_RES_FIT_DERIV_H

// Derivatives of the resolvent-fitting (RF) interpolation of XMCQDPT2, for the energy AS CURRENTLY
// COMPUTED: the exact derivative of the computed interpolated energy, nodes and weights included, with the grid and the
// stencil rule left as they are (not the interpolation of the exact derivative, which is a different quantity).
//
// The forward rule of res_fit_calc_arrays: for the evaluation point x = e0_alpha - E_act_I and the grid g_0..g_{N-1}
//   * i = first node with g_i >= x, moved to i-1 if that node is nearer; stencil [s, s+L) = [max(0,i-dim), min(N-1,i+dim)]
//   * weights = the Lagrange basis of the stencil nodes,  l_k(x; g) = prod_{m != k} (x - g_m) / (g_k - g_m)
//     (barycentric in the code; exact node hit gives the one-hot vector).
// The stencil choice is piecewise constant, so everything below is the derivative of the interpolation functional on the
// stencil selected at the central geometry; at a switch between stencils (x halfway between nodes) the functional is
// continuous but not smooth -- rf_stencil_margin() tells how far x is from such a switch.
//
// The grid (res_fit_data::gen_grid): g_j = (1-t_j) g_0 + t_j g_{N-1}, t_j = j/(N-1), with
//   g_0 = min_alpha e0_alpha - max_I E_act_I,    g_{N-1} = max_alpha e0_alpha - min_I E_act_I

#include <vector>

// Stencil and weights exactly as res_fit_calc_arrays chooses them, from the grid g[0..N-1]. c[L] are the weights.
int rf_stencil(double x, const double * g, int N, int dim, int * s, int * L);
void rf_weights(double x, const double * g, int s, int L, double * c);

// First derivatives of the weights on the stencil: dc_dx[k] = d l_k / dx, dc_dg[k*L + m] = d l_k / d g_{s+m}.
// Uses products only (no division by x - g_m), so it is exact at a node hit too. Returns 1 if two stencil nodes
// coincide (zero-width or repeated nodes: the weights are not defined), else 0.
int rf_weights_deriv(double x, const double * g, int s, int L, double * dc_dx, double * dc_dg);

// Reverse mode. Given cbar[k] = dE/dl_k: xbar += sum_k cbar_k dl_k/dx, gbar[s+m] += sum_k cbar_k dl_k/dg_{s+m}.
int rf_weights_vjp(double x, const double * g, int s, int L, const double * cbar, double * xbar, double * gbar);

// Grid extrema with the tie-breaking of find_min_max_minus_E_ci_normal / add_min_max_E_arr (first index of a strict extremum).
struct RFGridBranch{ int a_min, a_max, I_min, I_max; };
RFGridBranch rf_grid_branch(const double * e0, int n_s, const double * E_act, long nI);

// Grid values from the endpoints (what fill_grid yields up to rounding of the running sum).
void rf_grid(double g0, double gN, int N, double * g);

// Adjoint of the grid construction: gbar[N] -> adjoints of e0 and E_act through the endpoints and branch.
void rf_grid_vjp(const RFGridBranch & b, int N, const double * gbar, double * e0bar, double * Eactbar);

// Adjoint of the evaluation point x = e0[alpha] - E_act[I].
inline void rf_x_vjp(double xbar, int alpha, long I, double * e0bar, double * Eactbar){ e0bar[alpha] += xbar; Eactbar[I] -= xbar; }

// Distance of x from the nearest stencil switch, in units of the local node spacing (0.5 = exactly at a switch).
double rf_stencil_margin(double x, const double * g, int N);

#endif
