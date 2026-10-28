// rftest: derivatives of the resolvent-fitting interpolation of XMCQDPT2 against finite differences.
// Pure math, no molecule: the derivative functions of include/res_fit_deriv.h against central finite differences of the
// PRODUCTION functions res_fit_calc_arrays / find_min_max_minus_E_ci_normal / add_min_max_E_arr / fill_grid, including
// exact node hits, stencils clipped at the grid ends, and a moving grid built from moving e0 and E_act.
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "res_fit_deriv.h"

int res_fit_calc_arrays(double E, double * E_fit, int N_fit, int dim, int * s, int * l, double * appr_coef);
int find_min_max_minus_E_ci_normal(double * min, double * max, double * E, int n);
int add_min_max_E_arr(double * min_out, double * max_out, double * E, int n);
int fill_grid(double * G, int N);

static std::mt19937_64 gen(4242);
static double rnd(){ return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); }
static int fails = 0;
static void report(const char * name, double err, double tol){
    const bool ok = err < tol;
    printf("%-64s max error %.2e  %s\n", name, err, ok ? "PASS" : "FAIL");
    if(!ok) fails++;
}

// the production grid from e0 (n_s) and E_act (nI)
static std::vector<double> prod_grid(std::vector<double> e0, std::vector<double> Eact, int N){
    std::vector<double> g(N);
    find_min_max_minus_E_ci_normal(g.data(), g.data() + N - 1, Eact.data(), (int)Eact.size());
    add_min_max_E_arr(g.data(), g.data() + N - 1, e0.data(), (int)e0.size());
    fill_grid(g.data(), N);
    return g;
}
static double rfun(double d, double eta){ return d / (d * d + eta); }
static double rfun_p(double d, double eta){ const double q = d * d + eta; return (eta - d * d) / (q * q); }

int main(){
    const double eta = 0.0;
    // ---- 1. forward weights and stencil against the production function ------------------------------
    {
        double worst = 0; int stencil_mismatch = 0, cases = 0;
        for(int rep = 0; rep < 400; rep++){
            const int N = 12 + (rep % 30), dim = 1 + rep % 4;
            std::vector<double> g(N);
            const double g0 = -2.0 + rnd(), gN = 1.5 + rnd();
            rf_grid(g0, gN, N, g.data());
            double x = (rep % 5 == 0) ? g[rep % N] : g0 + (gN - g0) * (0.5 + 0.5 * rnd());   // every fifth: an exact node
            std::vector<double> c(2 * dim + 1);
            int s, L; res_fit_calc_arrays(x, g.data(), N, dim, &s, &L, c.data());
            int s2, L2; rf_stencil(x, g.data(), N, dim, &s2, &L2);
            std::vector<double> c2(L2);
            rf_weights(x, g.data(), s2, L2, c2.data());
            cases++;
            if(s != s2 || L != L2){ stencil_mismatch++; continue; }
            for(int k = 0; k < L; k++) worst = std::max(worst, std::fabs(c[k] - c2[k]));
        }
        printf("forward: %d cases (a fifth at exact nodes, dim 1..4, N 12..41), stencil mismatches: %d\n", cases, stencil_mismatch);
        report("weights = production barycentric weights", worst + stencil_mismatch, 1e-11);
    }
    // ---- 2. d l / dx and d l / d g against finite differences of the production weights ------------
    {
        double wx = 0, wg = 0; int used = 0;
        for(int rep = 0; rep < 400; rep++){
            const int N = 12 + (rep % 30), dim = 1 + rep % 4;
            std::vector<double> g(N); const double g0 = -2.0 + rnd(), gN = 1.5 + rnd();
            rf_grid(g0, gN, N, g.data());
            const double x = (rep % 5 == 0) ? g[rep % N] : g0 + (gN - g0) * (0.5 + 0.5 * rnd());
            if(rf_stencil_margin(x, g.data(), N) < 1e-3 && rep % 5 != 0) continue;     // too close to a stencil switch
            int s, L; rf_stencil(x, g.data(), N, dim, &s, &L);
            std::vector<double> dx(L), dg((size_t)L * L);
            if(rf_weights_deriv(x, g.data(), s, L, dx.data(), dg.data())) continue;
            used++;
            const double h = 1e-6;
            auto weights = [&](double xx, const std::vector<double> & gg){ std::vector<double> c(2 * dim + 1); int ss, ll; res_fit_calc_arrays(xx, const_cast<double *>(gg.data()), N, dim, &ss, &ll, c.data()); c.resize(ll); return c; };
            if(x + h > g[N - 1] || x - h < g[0]) continue;
            auto cp = weights(x + h, g), cm = weights(x - h, g);
            if(x + h > g[N - 1] || x - h < g[0]) continue;   // the production search would read g[N] beyond the grid: x never leaves [g_min, g_max] in XMC
            if((int)cp.size() != L || (int)cm.size() != L){ printf("   stencil changed under the FD step: rep %d x=%g L=%d, sizes %zu %zu\n", rep, x, L, cp.size(), cm.size()); continue; }
            for(int k = 0; k < L; k++){ const double e = std::fabs(dx[k] - (cp[k] - cm[k]) / (2 * h)) / (1.0 + std::fabs(dx[k])); if(e > 1e-4 && wx < 1) printf("   dx mismatch rep %d k=%d analytic %g FD %g (x=%.6g, margin %.3g, N=%d dim=%d s=%d L=%d)\n", rep, k, dx[k], (cp[k]-cm[k])/(2*h), x, rf_stencil_margin(x,g.data(),N), N, dim, s, L); wx = std::max(wx, e); }
            for(int m = 0; m < L; m++){
                std::vector<double> gp = g, gm = g; gp[s + m] += h; gm[s + m] -= h;
                auto ep = weights(x, gp), em = weights(x, gm);
                for(int k = 0; k < L; k++) wg = std::max(wg, std::fabs(dg[(size_t)k * L + m] - (ep[k] - em[k]) / (2 * h)) / (1.0 + std::fabs(dg[(size_t)k * L + m])));
            }
        }
        printf("derivatives: %d cases away from stencil switches (including exact node hits)\n", used);
        report("d l_k / dx  vs FD of production weights", wx, 1e-6);
        report("d l_k / d g_m vs FD of production weights (one node moved)", wg, 1e-6);
    }
    // ---- 3. the whole interpolation T(x) = sum l_k r(d + g_k) with a MOVING grid -------------
    {
        double worst = 0, worst_hit = 0; int used = 0, hits = 0;
        for(int rep = 0; rep < 300; rep++){
            const int n_s = 3 + rep % 3, nI = 20 + rep % 25, N = 15 + rep % 20, dim = 1 + rep % 3;
            std::vector<double> e0(n_s), Eact(nI);
            for(double & v : e0) v = -0.5 + 0.3 * rnd();
            for(double & v : Eact) v = -1.0 + 0.6 * rnd();
            const double d0 = 1.3 + 0.2 * rnd();                   // the orbital part d^orb of the denominator (positive: far from poles)
            const int alpha = rep % n_s, I = (rep * 7) % nI;
            auto forward = [&](const std::vector<double> & e, const std::vector<double> & E, double d){
                auto g = prod_grid(e, E, N);
                const double x = e[alpha] - E[I];
                std::vector<double> c(2 * dim + 1); int s, L; res_fit_calc_arrays(x, g.data(), N, dim, &s, &L, c.data());
                double t = 0; for(int k = 0; k < L; k++) t += c[k] * rfun(d + g[s + k], eta);
                return t;
            };
            auto g = prod_grid(e0, Eact, N);
            const double x = e0[alpha] - Eact[I];
            if(rf_stencil_margin(x, g.data(), N) < 1e-3) continue;
            const bool at_hit = (rep % 6 == 0);
            std::vector<double> e0u = e0, Eu = Eact;
            int hit_node = -1;
            if(at_hit){   // move e0[alpha] so that x is exactly an interior node (nodes are moved by the same amount only through the endpoints)
                // only possible without changing the extrema when alpha / I are not the extremal ones
                RFGridBranch b = rf_grid_branch(e0.data(), n_s, Eact.data(), nI);
                if(alpha == b.a_min || alpha == b.a_max || I == b.I_min || I == b.I_max) continue;
                const int h = 2 + (rep % (N - 4));
                e0u[alpha] = g[h] + Eact[I];                           // x == g[h] up to rounding of g[h] + E - E
                hit_node = h;
            }
            auto gu = prod_grid(e0u, Eu, N);
            const double xu = e0u[alpha] - Eu[I];
            if(at_hit && (hit_node < 0 || xu != gu[hit_node])) continue;   // count only bitwise node hits (the production one-hot branch needs equality)
            int s, L; rf_stencil(xu, gu.data(), N, dim, &s, &L);
            std::vector<double> c(L); rf_weights(xu, gu.data(), s, L, c.data());
            // analytic: T = sum c_k r(d + g_{s+k})
            // reverse mode with seed 1: cbar_k = r(d+g), gbar_{s+k} += c_k r'(d+g), dbar += sum c_k r'
            std::vector<double> cbar(L), gbar(N, 0.0); double xbar = 0, dbar = 0;
            for(int k = 0; k < L; k++){ cbar[k] = rfun(d0 + gu[s + k], eta); gbar[s + k] += c[k] * rfun_p(d0 + gu[s + k], eta); dbar += c[k] * rfun_p(d0 + gu[s + k], eta); }
            if(rf_weights_vjp(xu, gu.data(), s, L, cbar.data(), &xbar, gbar.data())) continue;
            std::vector<double> e0bar(n_s, 0.0), Eabar(nI, 0.0);
            rf_grid_vjp(rf_grid_branch(e0u.data(), n_s, Eu.data(), nI), N, gbar.data(), e0bar.data(), Eabar.data());
            rf_x_vjp(xbar, alpha, I, e0bar.data(), Eabar.data());
            // finite differences of the production forward in random directions of (e0, E_act, d)
            std::vector<double> de(n_s), dE(nI); for(double & v : de) v = rnd(); for(double & v : dE) v = rnd(); const double dd = rnd();
            double ana = dbar * dd; for(int i = 0; i < n_s; i++) ana += e0bar[i] * de[i]; for(int i = 0; i < nI; i++) ana += Eabar[i] * dE[i];
            const double h = 1e-6;
            auto at = [&](double k){ auto e = e0u; auto E = Eu; for(int i = 0; i < n_s; i++) e[i] += k * h * de[i]; for(int i = 0; i < nI; i++) E[i] += k * h * dE[i]; return forward(e, E, d0 + k * h * dd); };
            const double fd = (at(1) - at(-1)) / (2 * h);
            const double err = std::fabs(ana - fd) / (1e-8 + std::fabs(fd));
            if(at_hit){ hits++; worst_hit = std::max(worst_hit, err); } else { used++; worst = std::max(worst, err); }
        }
        printf("moving grid: %d generic points and %d exact-node-hit points (grid rebuilt from e0, E_act by the production code)\n", used, hits);
        report("d/d(e0, E_act, d) of the interpolated r: analytic vs FD  (generic)", worst, 1e-5);
        report("same at an exact node hit (non-zero derivative: the one-hot weights are not constant)", worst_hit, 1e-5);
    }
    // ---- 4. derivative of the computed interpolation differs from the interpolation of the derivative ------
    {
        const int N = 12, dim = 2; std::vector<double> g(N); rf_grid(-1.0, 1.0, N, g.data());
        const double x = 0.123, d0 = 1.4;
        int s, L; rf_stencil(x, g.data(), N, dim, &s, &L); std::vector<double> c(L), dx(L), dg((size_t)L * L);
        rf_weights(x, g.data(), s, L, c.data()); rf_weights_deriv(x, g.data(), s, L, dx.data(), dg.data());
        double dcomp = 0, interp = 0;
        for(int k = 0; k < L; k++){ dcomp += dx[k] * rfun(d0 + g[s + k], eta); interp += c[k] * rfun_p(d0 + g[s + k], eta); }
        printf("informational: d/dx of the interpolated r = % .8f, interpolation of r'(x) = % .8f, exact r'(x) = % .8f\n",
               dcomp, interp, rfun_p(d0 + x, eta));
    }
    // ---- 5. degenerate grids: the weights are not defined --------------------------------------------
    {
        std::vector<double> g(10); rf_grid(0.3, 0.3, 10, g.data());
        int s, L; rf_stencil(0.3, g.data(), 10, 2, &s, &L);
        std::vector<double> dx(L), dg((size_t)L * L);
        const int err = rf_weights_deriv(0.3, g.data(), s, L, dx.data(), dg.data());
        std::vector<double> c(5); int ss, ll; res_fit_calc_arrays(0.3, g.data(), 10, 2, &ss, &ll, c.data());
        printf("zero-width grid: rf_weights_deriv reports 'undefined' = %d ; the production weights there are", err);
        for(int k = 0; k < ll; k++) printf(" %.3g", c[k]);
        printf("\n");
        report("zero-width grid is refused (not silently differentiated)", err ? 0.0 : 1.0, 0.5);
    }
    printf("\n%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
