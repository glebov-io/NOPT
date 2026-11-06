// ritest: the density-fitting adjoints of include/ri_gradient.h, on random data (no molecule).
//
//   1. directional derivative of E = Q : (L V^-1 L^T) along random dL and symmetric dV
//      against sum(Lbar*dL) + sum(Vbar*dV), for arbitrary non-symmetric Q, for L-only, V-only and both.
//   2. Negative control: a seed without the metric term must be detected as wrong.
//   3. ri_jk_vjp (no n^4 Q) against ri_vjp_dense with the explicit Coulomb/exchange Q.
//   4. adjoint of V^-1/2, including repeated eigenvalues.
#include <stdio.h>
#include <cmath>
#include <vector>
#include <functional>

#include "blas_link.h"
#include "matr.h"
#include "ri_gradient.h"

typedef std::vector<double> V;
static unsigned long long st = 88172645463325252ULL;
static double rnd(){ st ^= st << 13; st ^= st >> 7; st ^= st << 17; return ((st >> 11) * (1.0 / 9007199254740992.0)) * 2 - 1; }
static V rv(size_t n){ V v(n); for(double & x : v) x = rnd(); return v; }
static int fails = 0;

static V sym_fn(int n, const V & A, const std::function<double(double)> & f){   // f(A), A symmetric
    V U = A, ev(n); lapack_diag(U.data(), ev.data(), n);
    V R((size_t)n * n, 0.0);
    for(int k = 0; k < n; k++) for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) R[i * n + j] += U[k * n + i] * f(ev[k]) * U[k * n + j];
    return R;
}
static V spd(int n, double shift, bool repeated = false){
    V M = rv((size_t)n * n), A((size_t)n * n, 0.0);
    for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) for(int k = 0; k < n; k++) A[i * n + j] += M[i * n + k] * M[j * n + k];
    for(int i = 0; i < n; i++) A[i * n + i] += shift;
    if(repeated){   // build V = U diag U^T with a doubled eigenvalue
        V Q = A, ev(n); lapack_diag(Q.data(), ev.data(), n);
        ev[1] = ev[0];
        V R((size_t)n * n, 0.0);
        for(int k = 0; k < n; k++) for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) R[i * n + j] += Q[k * n + i] * ev[k] * Q[k * n + j];
        return R;
    }
    return A;
}
static double dot(const V & a, const V & b){ double s = 0; for(size_t i = 0; i < a.size(); i++) s += a[i] * b[i]; return s; }
static void report(const char * name, double err, double scale, bool expect_fail = false){
    const bool good = expect_fail ? err > 1e-3 * scale : err < 1e-7 * std::max(1.0, scale);
    printf("%-52s |analytic - FD| = %.2e (scale %.2e)  %s\n", name, err, scale, good ? (expect_fail ? "DETECTED" : "PASS") : "FAIL");
    if(!good) fails++;
}

int main(){
    const int n = 4, npair = n * n, naux = 7;
    V L = rv((size_t)npair * naux), Vm = spd(naux, 1.0), Q = rv((size_t)npair * npair);
    auto inv = [&](const V & A){ return sym_fn(naux, A, [](double x){ return 1.0 / x; }); };
    auto energy = [&](const V & L_, const V & V_, const V & Q_){
        V Vi = inv(V_), LA((size_t)npair * naux, 0.0);
        for(int x = 0; x < npair; x++) for(int p = 0; p < naux; p++) for(int q = 0; q < naux; q++) LA[x * naux + p] += L_[x * naux + q] * Vi[q * naux + p];
        double e = 0;   // sum_xy Q_xy sum_P LA_xP L_yP
        for(int x = 0; x < npair; x++) for(int y = 0; y < npair; y++){
            double g = 0; for(int p = 0; p < naux; p++) g += LA[x * naux + p] * L_[y * naux + p];
            e += Q_[x * npair + y] * g; }
        return e;
    };
    V Lbar((size_t)npair * naux), Vbar((size_t)naux * naux), Vinv = inv(Vm);
    ri_vjp_dense(npair, naux, L.data(), Vinv.data(), Q.data(), Lbar.data(), Vbar.data());

    const double h = 1e-5;
    V dL = rv((size_t)npair * naux), dV0 = rv((size_t)naux * naux), dV((size_t)naux * naux);
    for(int i = 0; i < naux; i++) for(int j = 0; j < naux; j++) dV[i * naux + j] = 0.5 * (dV0[i * naux + j] + dV0[j * naux + i]);
    auto fd = [&](bool vL, bool vV){
        V Lp = L, Lm = L, Vp = Vm, Vmm = Vm;
        for(size_t i = 0; i < L.size(); i++){ if(vL){ Lp[i] += h * dL[i]; Lm[i] -= h * dL[i]; } }
        for(size_t i = 0; i < Vm.size(); i++){ if(vV){ Vp[i] += h * dV[i]; Vmm[i] -= h * dV[i]; } }
        return (energy(Lp, Vp, Q) - energy(Lm, Vmm, Q)) / (2 * h);
    };
    const double aL = dot(Lbar, dL), aV = dot(Vbar, dV);
    report("L-only perturbation", std::fabs(aL - fd(true, false)), std::fabs(aL));
    report("V-only perturbation", std::fabs(aV - fd(false, true)), std::fabs(aV));
    report("L and V", std::fabs(aL + aV - fd(true, true)), std::fabs(aL + aV));
    report("negative control: metric term dropped", std::fabs(aL - fd(true, true)), std::fabs(aL + aV), true);

    // J/K seeds against the explicit Q
    {
        V P = rv(n * n); for(int i = 0; i < n; i++) for(int j = 0; j < i; j++){ double a = 0.5 * (P[i * n + j] + P[j * n + i]); P[i * n + j] = P[j * n + i] = a; }
        const double cJ = 0.5, cK = -0.25;
        V Qjk((size_t)npair * npair);
        for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) for(int c = 0; c < n; c++) for(int d = 0; d < n; d++)
            Qjk[(a * n + b) * npair + c * n + d] = cJ * P[a * n + b] * P[c * n + d] + cK * P[a * n + c] * P[b * n + d];
        V L1((size_t)npair * naux), V1((size_t)naux * naux), L2((size_t)npair * naux), V2((size_t)naux * naux);
        ri_vjp_dense(npair, naux, L.data(), Vinv.data(), Qjk.data(), L1.data(), V1.data());
        ri_jk_vjp(n, naux, L.data(), Vm.data(), P.data(), cJ, cK, L2.data(), V2.data());
        double eL = 0, eV = 0, sc = 0;
        for(size_t i = 0; i < L1.size(); i++){ eL = std::max(eL, std::fabs(L1[i] - L2[i])); sc = std::max(sc, std::fabs(L1[i])); }
        for(size_t i = 0; i < V1.size(); i++) eV = std::max(eV, std::fabs(V1[i] - V2[i]));
        report("ri_jk_vjp vs explicit Q: Lbar", eL, sc);
        report("ri_jk_vjp vs explicit Q: Vbar", eV, sc);
    }

    // inverse square root
    for(int rep = 0; rep < 2; rep++){
        V A = spd(naux, 0.5, rep == 1), Fb = rv((size_t)naux * naux), Vb((size_t)naux * naux);
        ri_inv_sqrt_vjp(naux, A.data(), Fb.data(), Vb.data());
        auto Ef = [&](const V & X){ V F = sym_fn(naux, X, [](double x){ return 1.0 / std::sqrt(x); }); return dot(Fb, F); };
        V Ap = A, Am = A;
        for(size_t i = 0; i < A.size(); i++){ Ap[i] += h * dV[i]; Am[i] -= h * dV[i]; }
        const double a = dot(Vb, dV);
        report(rep ? "V^-1/2 adjoint, repeated eigenvalue" : "V^-1/2 adjoint", std::fabs(a - (Ef(Ap) - Ef(Am)) / (2 * h)), std::fabs(a));
    }
    printf("\n%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
