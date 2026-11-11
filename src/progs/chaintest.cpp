// chaintest: the chains of include/pt_chain.h and the semicanonicalization adjoint of include/pt_canon.h, on random data
// (no molecule). Every check compares sum(adjoint * direction) with a central finite difference of a scalar built from the
// forward definitions written out here, along random directions of all inputs at once.
//
//   1. pt_chain_B: phi = sum_blocks <Bbar_XY, B_XY>, B_XY[p,q,k] = sum_ab C_X[p,a] C_Y[q,b] (L V^-1/2)[ab,k],
//      over the five blocks VC, VA, CA, CC, AA; directions in C, L (symmetric in ab) and V (symmetric);
//      the adjoint of L and V both in the auxiliary basis and in the eigenbasis of V.
//   2. pt_chain_H: phi = sum_blocks <Hbar_XY, C_X Hc C_Y^T>, Hc = h + 2J[DM] - K[DM], DM = sum_core c c^T (RI J, K);
//      directions in C, L, V and h, both bases.
//   3. pt_canon_vjp: C_0 -> semicanonical C_sc = U C_0 (core and virtual blocks of F_SA = h + J[D] - K[D]/2 diagonalized,
//      D = 2 sum_core c c^T + sum_tu gamma_tu c_t c_u^T), phi = <Cbar_sc, C_sc> + <ebar, eps> (+ <fbar, C_sc F C_sc^T>);
//      directions in C_0, h, L, V and gamma; with and without fbar_mo, both bases.
//   4. Negative controls: the semicanonicalization without its rotation terms, and pt_chain_H without the J/K dressing,
//      must be detected as wrong.
#include <stdio.h>
#include <cmath>
#include <vector>
#include <functional>

#include "blas_link.h"
#include "matr.h"
#include "ri_gradient.h"
#include "pt_canon.h"

typedef std::vector<double> V;
static unsigned long long st = 88172645463325252ULL;
static double rnd(){ st ^= st << 13; st ^= st >> 7; st ^= st << 17; return ((st >> 11) * (1.0 / 9007199254740992.0)) * 2 - 1; }
static V rv(size_t n){ V v(n); for(double & x : v) x = rnd(); return v; }
static int fails = 0;

static double dot(const V & a, const V & b){ double s = 0; for(size_t i = 0; i < a.size(); i++) s += a[i] * b[i]; return s; }
static V axpy(const V & x, double h, const V & d){ V y = x; for(size_t i = 0; i < y.size(); i++) y[i] += h * d[i]; return y; }
static V sym_fn(int n, const V & A, const std::function<double(double)> & f){   // f(A), A symmetric
    V U = A, ev(n); lapack_diag(U.data(), ev.data(), n);
    V R((size_t)n * n, 0.0);
    for(int k = 0; k < n; k++) for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) R[i * n + j] += U[k * n + i] * f(ev[k]) * U[k * n + j];
    return R;
}
static V spd(int n, double shift){
    V M = rv((size_t)n * n), A((size_t)n * n, 0.0);
    for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) for(int k = 0; k < n; k++) A[i * n + j] += M[i * n + k] * M[j * n + k];
    for(int i = 0; i < n; i++) A[i * n + i] += shift;
    return A;
}
static V sym_rand(int n){ V A = rv((size_t)n * n); for(int i = 0; i < n; i++) for(int j = 0; j < i; j++) A[i * n + j] = A[j * n + i]; return A; }
static V pair_sym_rand(int n, int naux){   // L[(a n + b) naux + P] symmetric in a, b
    V L = rv((size_t)n * n * naux);
    for(int a = 0; a < n; a++) for(int b = 0; b < a; b++) for(int P = 0; P < naux; P++) L[((size_t)a * n + b) * naux + P] = L[((size_t)b * n + a) * naux + P];
    return L;
}
static void report(const char * name, double an, double fd, bool expect_fail = false){
    const double err = std::fabs(an - fd), scale = std::max(std::fabs(an), std::fabs(fd));
    const bool good = expect_fail ? err > 1e-3 * scale : err < 1e-7 * std::max(1.0, scale);
    printf("%-64s analytic % .10e  FD % .10e  |diff| %.1e  %s\n", name, an, fd, err, good ? (expect_fail ? "DETECTED" : "PASS") : "FAIL");
    if(!good) fails++;
}

// the fitted integrals (ab|cd) = sum_PQ L[ab,P] V^-1[P,Q] L[cd,Q]; J[D], K[D]
struct Eri{
    int n; V g;
    Eri(int n_, int naux, const V & L, const V & Vm) : n(n_), g((size_t)n_ * n_ * n_ * n_, 0.0){
        V Vi = sym_fn(naux, Vm, [](double x){ return 1.0 / x; }), LV((size_t)n * n * naux, 0.0);
        for(int x = 0; x < n * n; x++) for(int P = 0; P < naux; P++) for(int Q = 0; Q < naux; Q++) LV[(size_t)x * naux + Q] += L[(size_t)x * naux + P] * Vi[P * naux + Q];
        for(int x = 0; x < n * n; x++) for(int y = 0; y < n * n; y++){ double s = 0; for(int Q = 0; Q < naux; Q++) s += LV[(size_t)x * naux + Q] * L[(size_t)y * naux + Q]; g[(size_t)x * n * n + y] = s; }
    }
    double at(int a, int b, int c, int d) const { return g[(((size_t)a * n + b) * n + c) * n + d]; }
    void JK(const V & D, V & J, V & K) const {
        J.assign((size_t)n * n, 0.0); K.assign((size_t)n * n, 0.0);
        for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) for(int c = 0; c < n; c++) for(int d = 0; d < n; d++){
            J[a * n + b] += D[c * n + d] * at(a, b, c, d);
            K[a * n + b] += D[c * n + d] * at(a, c, b, d);
        }
    }
};
// X^T-free block product: (C_X M C_Y^T)[p,q] for the rows X0.., Y0.. of C
static V block(const V & C, int n, int X0, int nX, int Y0, int nY, const V & M){
    V R((size_t)nX * nY, 0.0);
    for(int p = 0; p < nX; p++) for(int q = 0; q < nY; q++){
        double s = 0;
        for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) s += C[(size_t)(X0 + p) * n + a] * M[(size_t)a * n + b] * C[(size_t)(Y0 + q) * n + b];
        R[(size_t)p * nY + q] = s;
    }
    return R;
}

int main(){
    const int n = 6, naux = 11, nc = 2, na = 2, nv = 2, nmo = nc + na + nv;
    const double h = 1e-5;
    V L = pair_sym_rand(n, naux), Vm = spd(naux, 2.0), H0 = sym_rand(n);
    V dL = pair_sym_rand(n, naux), dV = sym_rand(naux), dh = sym_rand(n);
    RIEig E; E.build(n, naux, L.data(), Vm.data());
    auto to_aux = [&](PTChainOut & o){ ri_eig_to_aux_inplace(E, o.Lbar.data(), o.Vbar.data()); };

    // ---- 1. B-chain --------------------------------------------------------------------------------------------------------
    {
        V C = rv((size_t)nmo * n), dC = rv((size_t)nmo * n);
        struct Blk{ int X0, nX, Y0, nY; } blk[5] = {{nc + na, nv, 0, nc}, {nc + na, nv, nc, na}, {0, nc, nc, na}, {0, nc, 0, nc}, {nc, na, nc, na}};
        std::vector<V> adj(5);
        for(int b = 0; b < 5; b++) adj[b] = rv((size_t)blk[b].nX * blk[b].nY * naux);
        PTInputAdj A; A.VC = adj[0].data(); A.VA = adj[1].data(); A.CA = adj[2].data(); A.CC = adj[3].data(); A.AA = adj[4].data();
        auto phi = [&](const V & C_, const V & L_, const V & V_){
            V Vm05 = sym_fn(naux, V_, [](double x){ return 1.0 / std::sqrt(x); }), Bh((size_t)n * n * naux, 0.0);
            for(int x = 0; x < n * n; x++) for(int j = 0; j < naux; j++) for(int k = 0; k < naux; k++) Bh[(size_t)x * naux + k] += L_[(size_t)x * naux + j] * Vm05[j * naux + k];
            double s = 0;
            for(int b = 0; b < 5; b++) for(int p = 0; p < blk[b].nX; p++) for(int q = 0; q < blk[b].nY; q++) for(int k = 0; k < naux; k++){
                double v = 0;
                for(int a = 0; a < n; a++) for(int c = 0; c < n; c++) v += C_[(size_t)(blk[b].X0 + p) * n + a] * C_[(size_t)(blk[b].Y0 + q) * n + c] * Bh[((size_t)a * n + c) * naux + k];
                s += adj[b][((size_t)p * blk[b].nY + q) * naux + k] * v;
            }
            return s;
        };
        const double fd = (phi(axpy(C, h, dC), axpy(L, h, dL), axpy(Vm, h, dV)) - phi(axpy(C, -h, dC), axpy(L, -h, dL), axpy(Vm, -h, dV))) / (2 * h);
        for(int eig = 0; eig < 2; eig++){
            PTChainOut o; o.resize(nmo, n, naux);
            pt_chain_B(n, naux, nc, na, nv, C.data(), L.data(), Vm.data(), A, &o, eig ? &E : nullptr);
            if(eig) to_aux(o);
            report(eig ? "pt_chain_B: C, L, V (adjoint in the eigenbasis of V)" : "pt_chain_B: C, L, V", dot(o.Cbar, dC) + dot(o.Lbar, dL) + dot(o.Vbar, dV), fd);
        }
    }

    // ---- 2. H-chain --------------------------------------------------------------------------------------------------------
    {
        V C = rv((size_t)nmo * n), dC = rv((size_t)nmo * n);
        V aAV = rv((size_t)na * nv), aCA = rv((size_t)nc * na), aCV = rv((size_t)nc * nv);
        PTInputAdj A; A.H_AV = aAV.data(); A.H_CA = aCA.data(); A.H_CV = aCV.data();
        auto phi = [&](const V & C_, const V & L_, const V & V_, const V & h_, bool dressed){
            Eri G(n, naux, L_, V_);
            V DM((size_t)n * n, 0.0), J, K, Hc((size_t)n * n);
            for(int i = 0; i < nc; i++) for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) DM[a * n + b] += C_[(size_t)i * n + a] * C_[(size_t)i * n + b];
            G.JK(DM, J, K);
            for(int k = 0; k < n * n; k++) Hc[k] = h_[k] + (dressed ? 2.0 * J[k] - K[k] : 0.0);
            return dot(aAV, block(C_, n, nc, na, nc + na, nv, Hc)) + dot(aCA, block(C_, n, 0, nc, nc, na, Hc)) + dot(aCV, block(C_, n, 0, nc, nc + na, nv, Hc));
        };
        auto fd = [&](bool dressed){
            return (phi(axpy(C, h, dC), axpy(L, h, dL), axpy(Vm, h, dV), axpy(H0, h, dh), dressed)
                  - phi(axpy(C, -h, dC), axpy(L, -h, dL), axpy(Vm, -h, dV), axpy(H0, -h, dh), dressed)) / (2 * h);
        };
        const double f = fd(true);
        double an = 0;
        for(int eig = 0; eig < 2; eig++){
            PTChainOut o; o.resize(nmo, n, naux);
            pt_chain_H(n, naux, nc, na, nv, C.data(), L.data(), Vm.data(), H0.data(), A, &o, eig ? &E : nullptr, eig == 1);
            if(eig) to_aux(o);
            an = dot(o.Cbar, dC) + dot(o.Lbar, dL) + dot(o.Vbar, dV) + dot(o.hbar, dh);
            report(eig ? "pt_chain_H: C, L, V, h (adjoint in the eigenbasis of V)" : "pt_chain_H: C, L, V, h", an, f);
        }
        report("negative control: pt_chain_H against the undressed h", an, fd(false), true);
    }

    // ---- 3. semicanonicalization -----------------------------------------------------------------------------------------
    {
        V gam = sym_rand(na), dgam = sym_rand(na);
        for(int t = 0; t < na; t++) gam[t * na + t] += 1.0;
        // F_SA(C) = h + J[D] - K[D]/2
        auto fock = [&](const V & C_, const V & L_, const V & V_, const V & h_, const V & g_){
            Eri G(n, naux, L_, V_);
            V D((size_t)n * n, 0.0), J, K, F((size_t)n * n);
            for(int a = 0; a < n; a++) for(int b = 0; b < n; b++){
                double s = 0;
                for(int i = 0; i < nc; i++) s += 2.0 * C_[(size_t)i * n + a] * C_[(size_t)i * n + b];
                for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) s += g_[t * na + u] * C_[(size_t)(nc + t) * n + a] * C_[(size_t)(nc + u) * n + b];
                D[a * n + b] = s;
            }
            G.JK(D, J, K);
            for(int k = 0; k < n * n; k++) F[k] = h_[k] + J[k] - 0.5 * K[k];
            return F;
        };
        // the semicanonical orbitals and energies of C_0: core and virtual blocks diagonalized, eigenvectors ordered by energy,
        // signs fixed against the identity (the base point below is already semicanonical, so U = 1 + O(h) there)
        auto canon = [&](const V & C0, const V & F, V & Csc, V & eps){
            Csc = C0; eps.assign(nmo, 0.0);
            const int o[2] = {0, nc + na}, m[2] = {nc, nv};
            for(int b = 0; b < 2; b++){
                V Fx = block(C0, n, o[b], m[b], o[b], m[b], F), ev(m[b]);
                lapack_diag(Fx.data(), ev.data(), m[b]);
                for(int i = 0; i < m[b]; i++){
                    const double sg = Fx[i * m[b] + i] < 0 ? -1.0 : 1.0;
                    eps[o[b] + i] = ev[i];
                    for(int a = 0; a < n; a++){
                        double s = 0; for(int j = 0; j < m[b]; j++) s += sg * Fx[i * m[b] + j] * C0[(size_t)(o[b] + j) * n + a];
                        Csc[(size_t)(o[b] + i) * n + a] = s;
                    }
                }
            }
        };
        V C0, eps0;
        { V Ci = rv((size_t)nmo * n); canon(Ci, fock(Ci, L, Vm, H0, gam), C0, eps0); }
        { V Ct, et; canon(C0, fock(C0, L, Vm, H0, gam), Ct, et); double d = 0; for(size_t i = 0; i < C0.size(); i++) d = std::max(d, std::fabs(Ct[i] - C0[i]));
          printf("base point: semicanonical to %.1e\n", d); }
        V dC = rv((size_t)nmo * n), Cbar = rv((size_t)nmo * n), ec = rv(nc), ev = rv(nv), fbar = rv((size_t)nmo * nmo);
        auto phi = [&](const V & C_, const V & L_, const V & V_, const V & h_, const V & g_, bool with_f){
            V F = fock(C_, L_, V_, h_, g_), Csc, eps;
            canon(C_, F, Csc, eps);
            double s = dot(Cbar, Csc);
            for(int i = 0; i < nc; i++) s += ec[i] * eps[i];
            for(int i = 0; i < nv; i++) s += ev[i] * eps[nc + na + i];
            if(with_f) s += dot(fbar, block(Csc, n, 0, nmo, 0, nmo, F));
            return s;
        };
        auto fd = [&](bool with_f){
            return (phi(axpy(C0, h, dC), axpy(L, h, dL), axpy(Vm, h, dV), axpy(H0, h, dh), axpy(gam, h, dgam), with_f)
                  - phi(axpy(C0, -h, dC), axpy(L, -h, dL), axpy(Vm, -h, dV), axpy(H0, -h, dh), axpy(gam, -h, dgam), with_f)) / (2 * h);
        };
        for(int with_f = 0; with_f < 2; with_f++){
            const double f = fd(with_f);
            for(int eig = 0; eig < 2; eig++){
                PTChainOut o; o.resize(nmo, n, naux); o.Cbar = Cbar;       // Cbar_sc aliases out->Cbar: out->Cbar becomes the total adjoint
                V gbar((size_t)na * na, 0.0);
                pt_canon_vjp(n, naux, nc, na, nv, C0.data(), eps0.data(), L.data(), Vm.data(), H0.data(), gam.data(), o.Cbar.data(), ec.data(), ev.data(),
                             1e-8, &o, gbar.data(), nullptr, nullptr, nullptr, eig ? &E : nullptr, with_f ? fbar.data() : nullptr, eig == 1);
                if(eig) to_aux(o);
                char name[128];
                snprintf(name, sizeof name, "pt_canon_vjp: C, L, V, h, gamma%s%s", with_f ? ", Fock blocks" : "", eig ? " (eigenbasis of V)" : "");
                report(name, dot(o.Cbar, dC) + dot(o.Lbar, dL) + dot(o.Vbar, dV) + dot(o.hbar, dh) + dot(gbar, dgam), f);
            }
        }
        // negative control: the orbital adjoint without the semicanonicalization (C_sc taken as C_0, eps ignored)
        report("negative control: semicanonicalization ignored", dot(Cbar, dC), fd(false), true);
    }
    printf("\n%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
