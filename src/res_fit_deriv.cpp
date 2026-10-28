#include "res_fit_deriv.h"

#include <cmath>
#include <algorithm>

int rf_stencil(double x, const double * g, int N, int dim, int * s, int * L){
    int i = 0;
    while(i < N && g[i] < x) i++;                       // as res_fit_calc_arrays (without reading g[N])
    if(i > 0 && i < N && std::fabs(g[i - 1] - x) < std::fabs(g[i] - x)) i--;
    *s = std::max(0, i - dim);
    const int e = std::min(N - 1, i + dim);
    *L = e + 1 - *s;
    return 0;
}

void rf_weights(double x, const double * g, int s, int L, double * c){
    for(int k = 0; k < L; k++){
        double p = 1.0;
        for(int m = 0; m < L; m++) if(m != k) p *= (x - g[s + m]) / (g[s + k] - g[s + m]);
        c[k] = p;
    }
}

int rf_weights_deriv(double x, const double * g, int s, int L, double * dc_dx, double * dc_dg){
    for(int k = 0; k < L; k++) for(int m = k + 1; m < L; m++) if(g[s + k] == g[s + m]) return 1;
    std::fill(dc_dg, dc_dg + (size_t)L * L, 0.0);
    for(int k = 0; k < L; k++){
        const double gk = g[s + k];
        // factors f_m = (x - g_m)/(g_k - g_m), m != k
        double dx = 0.0, dgk = 0.0;
        for(int j = 0; j < L; j++){
            if(j == k) continue;
            const double gj = g[s + j];
            // d l_k / dx : only the factor j differs from f_j
            double pj = 1.0 / (gk - gj);
            for(int m = 0; m < L; m++) if(m != k && m != j) pj *= (x - g[s + m]) / (gk - g[s + m]);
            dx += pj;
            // d l_k / d g_j : (x - g_k)/(g_k - g_j)^2 times the other factors
            double qj = (x - gk) / ((gk - gj) * (gk - gj));
            for(int m = 0; m < L; m++) if(m != k && m != j) qj *= (x - g[s + m]) / (gk - g[s + m]);
            dc_dg[(size_t)k * L + j] = qj;
            // d l_k / d g_k collects -(x - g_j)/(g_k - g_j)^2 times the other factors, for every j
            double rj = -(x - gj) / ((gk - gj) * (gk - gj));
            for(int m = 0; m < L; m++) if(m != k && m != j) rj *= (x - g[s + m]) / (gk - g[s + m]);
            dgk += rj;
        }
        dc_dx[k] = dx;
        dc_dg[(size_t)k * L + k] = dgk;
    }
    return 0;
}

int rf_weights_vjp(double x, const double * g, int s, int L, const double * cbar, double * xbar, double * gbar){
    std::vector<double> dx(L), dg((size_t)L * L);
    if(rf_weights_deriv(x, g, s, L, dx.data(), dg.data())) return 1;
    for(int k = 0; k < L; k++){
        *xbar += cbar[k] * dx[k];
        for(int m = 0; m < L; m++) gbar[s + m] += cbar[k] * dg[(size_t)k * L + m];
    }
    return 0;
}

RFGridBranch rf_grid_branch(const double * e0, int n_s, const double * E_act, long nI){
    RFGridBranch b{0, 0, 0, 0};
    double vmin = 1e18, vmax = -1e18;                    // v = -E_act: min of v is the largest E_act
    for(long i = 0; i < nI; i++){
        const double v = -E_act[i];
        if(v < vmin){ vmin = v; b.I_max = (int)i; }       // g_0 uses -max E_act
        if(v > vmax){ vmax = v; b.I_min = (int)i; }       // g_{N-1} uses -min E_act
    }
    double emin = 1e18, emax = -1e18;
    for(int a = 0; a < n_s; a++){
        if(e0[a] < emin){ emin = e0[a]; b.a_min = a; }
        if(e0[a] > emax){ emax = e0[a]; b.a_max = a; }
    }
    return b;
}

void rf_grid(double g0, double gN, int N, double * g){
    for(int j = 0; j < N; j++){ const double t = (double)j / (N - 1); g[j] = (1.0 - t) * g0 + t * gN; }
}

void rf_grid_vjp(const RFGridBranch & b, int N, const double * gbar, double * e0bar, double * Eactbar){
    double b0 = 0.0, bN = 0.0;
    for(int j = 0; j < N; j++){ const double t = (double)j / (N - 1); b0 += (1.0 - t) * gbar[j]; bN += t * gbar[j]; }
    e0bar[b.a_min] += b0;  Eactbar[b.I_max] -= b0;
    e0bar[b.a_max] += bN;  Eactbar[b.I_min] -= bN;
}

double rf_stencil_margin(double x, const double * g, int N){
    int i = 0;
    while(i < N && g[i] < x) i++;
    if(i == 0 || i >= N) return 0.0;
    const double d = g[i] - g[i - 1];
    if(d == 0.0) return 0.0;
    // the switch is at the midpoint of [g_{i-1}, g_i]
    return 0.5 - std::fabs((x - g[i - 1]) / d - 0.5);
}
