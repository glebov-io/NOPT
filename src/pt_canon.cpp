#include "pt_canon.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "blas_link.h"
#include "CAS.h"
#include "common_vars.h"
#include "matr.h"
#include "molecule.h"
#include "RI.h"
#include "ri_gradient.h"
#include "libint_link.h"

namespace{
void gemm(int m, int n, int k, const double * A, const double * B, double * C, double alpha, double beta, bool tA, bool tB, int lda, int ldb, int ldc){
    cblas_dgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// D = 2 sum_core c c^T + sum_tu gamma_tu c_t c_u^T (n x n); rows of C are MOs
void build_D(int n, int nc, int na, const double * C, const double * gamma, std::vector<double> & D){
    D.assign((size_t)n * n, 0.0);
    if(nc) gemm(n, n, nc, C, C, D.data(), 2.0, 0.0, true, false, n, n, n);
    if(na){
        std::vector<double> T((size_t)na * n);
        gemm(na, n, na, gamma, C + (size_t)nc * n, T.data(), 1.0, 0.0, false, false, na, n, n);                   // gamma C_a
        gemm(n, n, na, C + (size_t)nc * n, T.data(), D.data(), 1.0, 1.0, true, false, n, n, n);                   // + C_a^T gamma C_a
    }
}
}

int pt_canon_vjp(int n, int naux, int nc, int na, int nv, const double * C, const double * eps, const double * L, const double * V,
                 const double * h, const double * gamma_sa, const double * Cbar_sc, const double * ec_bar, const double * ev_bar,
                 double degtol, PTChainOut * out, double * gbar, double * F_out, const double * ea_bar, const double * ci_rot_g, const RIEig * Ecache,
                 const double * fbar_mo, bool eig_out){
    const size_t n2 = (size_t)n * n;
    RIEig Eown;
    if(!Ecache){ Eown.build(n, naux, L, V); Ecache = &Eown; }
    const RIEig & E = *Ecache;
    std::vector<double> D, J(n2), K(n2), F(n2);
    build_D(n, nc, na, C, gamma_sa, D);
    {
        // D = 2 sum_core c c^T + sum_tu gamma_tu c_t c_u^T in factored form for the exchange
        const int r = nc + na;
        std::vector<double> X((size_t)n * std::max(r, 1), 0.0), Y((size_t)n * std::max(r, 1), 0.0);
        for(int a = 0; a < n; a++){
            for(int k = 0; k < nc; k++){ X[(size_t)a * r + k] = 2.0 * C[(size_t)k * n + a]; Y[(size_t)a * r + k] = C[(size_t)k * n + a]; }
            for(int t = 0; t < na; t++){
                X[(size_t)a * r + nc + t] = C[(size_t)(nc + t) * n + a];
                double v = 0; for(int u = 0; u < na; u++) v += gamma_sa[(size_t)t * na + u] * C[(size_t)(nc + u) * n + a];
                Y[(size_t)a * r + nc + t] = v;
            }
        }
        ri_fock_JK_eig_lr(E, D.data(), X.data(), Y.data(), r, J.data(), K.data());
    }
    for(size_t i = 0; i < n2; i++) F[i] = h[i] + J[i] - 0.5 * K[i];
    if(F_out) std::memcpy(F_out, F.data(), sizeof(double) * n2);

    // Cbar_sc may alias out->Cbar (the chains accumulate there): read it before the extra terms are added
    std::vector<double> Cb_in;
    if(Cbar_sc) Cb_in.assign(Cbar_sc, Cbar_sc + (size_t)(nc + na + nv) * n);
    Cbar_sc = Cbar_sc ? Cb_in.data() : nullptr;

    std::vector<double> Fbar(n2, 0.0);
    if(fbar_mo){
        // phi += sum_pq fbar[p,q] (C F C^T)[p,q]  (Fock blocks read in the semicanonical basis, e.g. the DSRG cross blocks):
        //   Cbar_sc += (fbar + fbar^T) C F  (before the rotation terms below read it),  Fbar += C^T fbar C
        const int nmo = nc + na + nv;
        if(!Cbar_sc){ Cb_in.assign((size_t)nmo * n, 0.0); Cbar_sc = Cb_in.data(); }
        std::vector<double> CF((size_t)nmo * n), fs((size_t)nmo * nmo), T((size_t)nmo * n);
        gemm(nmo, n, n, C, F.data(), CF.data(), 1.0, 0.0, false, false, n, n, n);
        for(int p = 0; p < nmo; p++) for(int q = 0; q < nmo; q++) fs[(size_t)p * nmo + q] = fbar_mo[(size_t)p * nmo + q] + fbar_mo[(size_t)q * nmo + p];
        gemm(nmo, n, nmo, fs.data(), CF.data(), Cb_in.data(), 1.0, 1.0, false, false, nmo, n, n);       // seen by the rotation terms below
        gemm(nmo, n, nmo, fs.data(), CF.data(), out->Cbar.data(), 1.0, 1.0, false, false, nmo, n, n);   // and part of the total adjoint
        gemm(nmo, n, nmo, fbar_mo, C, T.data(), 1.0, 0.0, false, false, nmo, n, n);
        gemm(n, n, nmo, C, T.data(), Fbar.data(), 1.0, 1.0, true, false, n, n, n);
    }
    // diagnostics: pairs dropped as degenerate (their antisymmetric seed should vanish by invariance) and the smallest gap kept
    int n_skip = 0; double skip_asym = 0.0, min_kept = 1e300;
    const int first[3] = {0, nc + na, nc}, dim[3] = {nc, nv, na};
    const double * ebar[3] = {ec_bar, ev_bar, ea_bar};
    for(int b = 0; b < (ea_bar ? 3 : 2); b++){
        const int m = dim[b], o = first[b];
        if(m == 0) continue;
        // the active block: with equal eps_a the energy is invariant under rotations of the active orbitals (no rotation term); with a
        // per-orbital eps_a (XMCQDPT2) it is not, and the rotation of the active orbitals must carry the CI vectors: ci_rot_g supplies that half of the gradient
        const bool gauge_free = (b == 2) && (ci_rot_g == nullptr);
        const double * Cx = C + (size_t)o * n;
        std::vector<double> Mx((size_t)m * m, 0.0), Fs((size_t)m * m, 0.0);
        if(Cbar_sc && !gauge_free) gemm(m, m, n, Cbar_sc + (size_t)o * n, Cx, Mx.data(), 1.0, 0.0, false, true, n, n, m);       // M_ij = Cbar_i . C_j
        if(b == 2 && ci_rot_g) for(int i = 0; i < m; i++) for(int j = 0; j < m; j++) Mx[(size_t)i * m + j] += 0.5 * ci_rot_g[(size_t)i * m + j];   // CI part: M_ij - M_ji gets g_ij
        for(int i = 0; i < m; i++){
            if(ebar[b]) Fs[(size_t)i * m + i] += ebar[b][i];
            for(int j = 0; j < m; j++){
                if(i == j || gauge_free) continue;
                const double d = eps[o + i] - eps[o + j];
                if(std::fabs(d) < degtol){ if(i < j){ n_skip++; skip_asym = std::max(skip_asym, std::fabs(Mx[(size_t)i * m + j] - Mx[(size_t)j * m + i])); } continue; }
                min_kept = std::min(min_kept, std::fabs(d));
                Fs[(size_t)i * m + j] += 0.5 * (Mx[(size_t)i * m + j] - Mx[(size_t)j * m + i]) / d;
            }
        }
        // Fbar_AO += C_X^T Fs C_X ;  Cbar_extra += 2 Fs (C_X F)
        std::vector<double> T((size_t)m * n), T2((size_t)m * n);
        gemm(m, n, m, Fs.data(), Cx, T.data(), 1.0, 0.0, false, false, m, n, n);
        gemm(n, n, m, Cx, T.data(), Fbar.data(), 1.0, 1.0, true, false, n, n, n);
        gemm(m, n, n, Cx, F.data(), T2.data(), 1.0, 0.0, false, false, n, n, n);
        gemm(m, n, m, Fs.data(), T2.data(), out->Cbar.data() + (size_t)o * n, 2.0, 1.0, false, false, m, n, n);
    }
    if(n_skip) fprintf(out_stream, "canonicalization adjoint: %d orbital pair(s) with |eps_i - eps_j| < %.0e treated as degenerate (largest dropped antisymmetric seed %.2e%s)\n",
                       n_skip, degtol, skip_asym, skip_asym > 1e-6 ? " -- NOT negligible, the gradient is unreliable here" : "");
    if(min_kept < 1e-5) fprintf(out_stream, "WARNING: canonicalization adjoint: smallest orbital-energy gap kept is %.2e (rounding errors are amplified as 1/gap)\n", min_kept);
    else if(min_kept < 1e-3) fprintf(out_stream, "canonicalization adjoint: smallest orbital-energy gap kept %.2e\n", min_kept);
    for(int i = 0; i < n; i++) for(int j = 0; j < i; j++){ const double a = 0.5 * (Fbar[(size_t)i * n + j] + Fbar[(size_t)j * n + i]); Fbar[(size_t)i * n + j] = a; Fbar[(size_t)j * n + i] = a; }

    // F = h + J[D] - 1/2 K[D]
    for(size_t i = 0; i < n2; i++) out->hbar[i] += Fbar[i];
    std::vector<double> Dbar(n2), Jf(n2), Kf(n2);
    ri_fock_JK_eig(E, Fbar.data(), Jf.data(), Kf.data());
    for(size_t i = 0; i < n2; i++) Dbar[i] = Jf[i] - 0.5 * Kf[i];
    {   // L, V: seeds of the symmetric bilinear form B(Fbar, D) = sum Fbar (J[D] - K[D]/2)
        if(eig_out) ri_jk_vjp_eig(E, Fbar.data(), D.data(), 1.0, -0.5, out->Lbar.data(), out->Vbar.data());
        else{
            std::vector<double> Lbt(n2 * naux, 0.0), Vbt((size_t)naux * naux, 0.0);
            ri_jk_vjp_eig(E, Fbar.data(), D.data(), 1.0, -0.5, Lbt.data(), Vbt.data());
            ri_eig_to_aux(E, Lbt.data(), Vbt.data(), out->Lbar.data(), out->Vbar.data(), true);
        }
    }
    // D = 2 sum_core c c^T + sum_tu gamma_tu c_t c_u^T
    if(nc) gemm(nc, n, n, C, Dbar.data(), out->Cbar.data(), 4.0, 1.0, false, false, n, n, n);
    if(na){
        const double * Ca = C + (size_t)nc * n;
        std::vector<double> T((size_t)na * n);
        gemm(na, n, n, Ca, Dbar.data(), T.data(), 1.0, 0.0, false, false, n, n, n);                              // (Dbar c_u)^T rows
        gemm(na, n, na, gamma_sa, T.data(), out->Cbar.data() + (size_t)nc * n, 2.0, 1.0, false, false, na, n, n);
        if(gbar) gemm(na, na, n, T.data(), Ca, gbar, 1.0, 1.0, false, true, n, n, na);                           // gbar_tu += c_t Dbar c_u
    }
    return 0;
}
