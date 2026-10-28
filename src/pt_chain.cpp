#include "pt_chain.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "blas_link.h"
#include "common_vars.h"
#include "matr.h"
#include "molecule.h"
#include "RI.h"
#include "ri_gradient.h"
#include "libint_link.h"

namespace{
inline void gemm(int m, int n, int k, const double * A, const double * B, double * C, double beta = 0.0, bool tA = false, bool tB = false, int lda = -1, int ldb = -1, int ldc = -1){
    cblas_dgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans, m, n, k, 1.0,
                A, lda >= 0 ? lda : (tA ? m : k), B, ldb >= 0 ? ldb : (tB ? k : n), beta, C, ldc >= 0 ? ldc : n);
}

// V^-1/2 as S05_calc builds it (eigen decomposition), in my own storage
void inv_sqrt(int naux, const double * V, std::vector<double> & Vm05){
    std::vector<double> U(V, V + (size_t)naux * naux), lam(naux);
    lapack_diag(U.data(), lam.data(), naux);
    Vm05.assign((size_t)naux * naux, 0.0);
    std::vector<double> sU((size_t)naux * naux);
    for(int i = 0; i < naux; i++) for(int j = 0; j < naux; j++) sU[(size_t)i * naux + j] = U[(size_t)i * naux + j] / std::sqrt(lam[i]);
    gemm(naux, naux, naux, U.data(), sU.data(), Vm05.data(), 0.0, true, false);     // U^T diag(lam^-1/2) U
}
}

int pt_chain_B(int n, int naux, int nc, int na, int nv, const double * C, const double * L, const double * V,
               const PTInputAdj & A, PTChainOut * out, const RIEig * Eeig){
    const size_t n2 = (size_t)n * n, na_ = naux;
    std::vector<double> Vm05; inv_sqrt(naux, V, Vm05);

    struct Blk{ const double * adj; int X0, nX, Y0, nY; };
    const Blk blocks[5] = {
        {A.VC, nc + na, nv, 0, nc},          // VC[a,i,k]
        {A.VA, nc + na, nv, nc, na},         // VA[a,t,k]
        {A.CA, 0, nc, nc, na},               // CA[i,t,k]
        {A.CC, 0, nc, 0, nc},                // CC[i,j,k]
        {A.AA, nc, na, nc, na},              // AA[t,u,k]
    };
    // Bhat = L V^-1/2 mixes the auxiliary index once; everything after it is linear in the fitted index k, so the chain runs over batches of k:
    // only O(n^2 kb) of Bhat and of its adjoint exist at a time (instead of two n^2 naux arrays). Per batch K:
    //   Bhat_K = L Vm05[:,K];  Bhb_K = sum_blocks CX^T t1 (t1 = CY^T Bbar_K);  Cbar += t1 Bhat_K^T, t2 Bhat_K;  Lbar += Bhb_K Vm05[:,K]^T;  Fb[:,K] = L^T Bhb_K.
    const int kb = (int)std::max<long>(16, std::min<long>(naux, (long)(4000000 / std::max<size_t>(n2, 1))));   // ~32 MB per batch array
    std::vector<double> Fb(na_ * na_, 0.0), Vb(na_ * na_);
    std::vector<double> Bh, Bhb, VmK, slice, t1, t2;
    for(int k0 = 0; k0 < naux; k0 += kb){
        const int K = std::min(kb, naux - k0);
        VmK.assign((size_t)naux * K, 0.0);
        for(int j = 0; j < naux; j++) for(int kk = 0; kk < K; kk++) VmK[(size_t)j * K + kk] = Vm05[(size_t)j * naux + k0 + kk];
        std::vector<double> WK;   // the Lbar map of this batch: Vm05[:,K] (auxiliary basis) or U Vm05[:,K] (eigenbasis of V: Lbt = Lbar U^T)
        if(Eeig){ WK.assign((size_t)naux * K, 0.0); gemm(naux, K, naux, Eeig->U.data(), VmK.data(), WK.data()); }
        Bh.assign(n2 * K, 0.0); Bhb.assign(n2 * K, 0.0);
        gemm((int)n2, K, naux, L, VmK.data(), Bh.data());                                  // Bhat_K = L Vm05[:,K]
        for(const Blk & b : blocks){
            if(b.adj == nullptr || b.nX == 0 || b.nY == 0) continue;
            const double * CX = C + (size_t)b.X0 * n, * CY = C + (size_t)b.Y0 * n;
            // the batch of the adjoint: Bbar[p][q][kk] = adj[(p nY + q) naux + k0 + kk]
            slice.assign((size_t)b.nX * b.nY * K, 0.0);
            for(size_t pq = 0; pq < (size_t)b.nX * b.nY; pq++) std::memcpy(slice.data() + pq * K, b.adj + pq * na_ + k0, sizeof(double) * K);
            // t1[p][b][kk] = sum_q CY[q][b] Bbar[p][q][kk]
            t1.assign((size_t)b.nX * n * K, 0.0);
            for(int p = 0; p < b.nX; p++) gemm(n, K, b.nY, CY, slice.data() + (size_t)p * b.nY * K, t1.data() + (size_t)p * n * K, 0.0, true, false);
            // Bhb[a][b][kk] += sum_p CX[p][a] t1[p][b][kk]
            gemm(n, n * K, b.nX, CX, t1.data(), Bhb.data(), 1.0, true, false);
            // Cbar_X[p][a] += sum_{b,kk} t1[p][b][kk] Bhat[a][b][kk]
            gemm(b.nX, n, n * K, t1.data(), Bh.data(), out->Cbar.data() + (size_t)b.X0 * n, 1.0, false, true);
            // t2[q][a][kk] = sum_p CX[p][a] Bbar[p][q][kk];   Cbar_Y[q][b] += sum_{a,kk} t2[q][a][kk] Bhat[a][b][kk]   (Bhat symmetric in a,b)
            t2.assign((size_t)b.nY * n * K, 0.0);
            std::vector<double> Bq((size_t)b.nX * K);
            for(int q = 0; q < b.nY; q++){
                for(int p = 0; p < b.nX; p++) std::memcpy(Bq.data() + (size_t)p * K, slice.data() + ((size_t)p * b.nY + q) * K, sizeof(double) * K);
                gemm(n, K, b.nX, CX, Bq.data(), t2.data() + (size_t)q * n * K, 0.0, true, false);
            }
            gemm(b.nY, n, n * K, t2.data(), Bh.data(), out->Cbar.data() + (size_t)b.Y0 * n, 1.0, false, true);
        }
        // Lbar += Bhb_K Vm05[:,K]^T ;  Fb[:,K] = L^T Bhb_K
        gemm((int)n2, naux, K, Bhb.data(), Eeig ? WK.data() : VmK.data(), out->Lbar.data(), 1.0, false, true);
        std::vector<double> FbK((size_t)naux * K);
        gemm(naux, K, (int)n2, L, Bhb.data(), FbK.data(), 0.0, true, false);
        for(int i = 0; i < naux; i++) for(int kk = 0; kk < K; kk++) Fb[(size_t)i * naux + k0 + kk] = FbK[(size_t)i * K + kk];
    }
    // Bhat = L Vm05: the metric part through ri_inv_sqrt_vjp
    ri_inv_sqrt_vjp(naux, V, Fb.data(), Vb.data());
    if(Eeig) ri_aux_to_eig(*Eeig, nullptr, Vb.data(), nullptr, out->Vbar.data());
    else for(size_t k = 0; k < na_ * na_; k++) out->Vbar[k] += Vb[k];
    return 0;
}

int pt_chain_H(int n, int naux, int nc, int na, int nv, const double * C, const double * L, const double * V, const double * h,
               const PTInputAdj & A, PTChainOut * out, const RIEig * Ecache, bool eig_out){
    const size_t n2 = (size_t)n * n;
    RIEig Eown;
    if(!Ecache){ Eown.build(n, naux, L, V); Ecache = &Eown; }
    const RIEig & E = *Ecache;
    // Hc and the AO adjoint of the three blocks
    std::vector<double> J(n2), K(n2), DM(n2, 0.0), Hc(n2);
    if(nc) gemm(n, n, nc, C, C, DM.data(), 0.0, true, false);                     // DM = sum_core C_i C_i^T
    ri_fock_JK_eig(E, DM.data(), J.data(), K.data());
    for(size_t k = 0; k < n2; k++) Hc[k] = h[k] + 2.0 * J[k] - K[k];

    struct Blk{ const double * adj; int X0, nX, Y0, nY; };
    const Blk blocks[3] = { {A.H_AV, nc, na, nc + na, nv}, {A.H_CA, 0, nc, nc, na}, {A.H_CV, 0, nc, nc + na, nv} };
    std::vector<double> Hb(n2, 0.0), tmp;
    for(const Blk & b : blocks){
        if(b.adj == nullptr || b.nX == 0 || b.nY == 0) continue;
        const double * CX = C + (size_t)b.X0 * n, * CY = C + (size_t)b.Y0 * n;
        // H_XY[p][q] = sum_ab CX[p][a] Hc[a][b] CY[q][b]
        // Hbar_AO[a][b] += sum_pq CX[p][a] Hbar[p][q] CY[q][b]
        tmp.assign((size_t)b.nX * n, 0.0);
        gemm(b.nX, n, b.nY, b.adj, CY, tmp.data());                                  // Hbar CY    (nX x n)
        gemm(n, n, b.nX, CX, tmp.data(), Hb.data(), 1.0, true, false);               // CX^T (Hbar CY)
        // Cbar_X[p][a] += sum_q Hbar[p][q] (Hc CY^T)[a][q]  ;  Cbar_Y[q][b] += sum_p Hbar[p][q] (CX Hc)[p][b]
        std::vector<double> HcCY((size_t)n * b.nY), CXHc((size_t)b.nX * n);
        gemm(n, b.nY, n, Hc.data(), CY, HcCY.data(), 0.0, false, true);              // Hc CY^T    (n x nY)
        gemm(b.nX, n, b.nY, b.adj, HcCY.data(), out->Cbar.data() + (size_t)b.X0 * n, 1.0, false, true);
        gemm(b.nX, n, n, CX, Hc.data(), CXHc.data());                                // CX Hc
        gemm(b.nY, n, b.nX, b.adj, CXHc.data(), out->Cbar.data() + (size_t)b.Y0 * n, 1.0, true, false);
    }
    // Hc is symmetric: only the symmetric part of its adjoint acts
    std::vector<double> Hs(n2);
    for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) Hs[(size_t)a * n + b] = 0.5 * (Hb[(size_t)a * n + b] + Hb[(size_t)b * n + a]);
    for(size_t k = 0; k < n2; k++) out->hbar[k] += Hs[k];
    // 2J[DM] - K[DM] contracted with Hs: DMbar = 2J[Hs] - K[Hs];  RI seeds by polarization of cJ J(P,P) + cK K(P,P), cJ = 2, cK = -1
    std::vector<double> Js(n2), Ks(n2), DMb(n2);
    ri_fock_JK_eig(E, Hs.data(), Js.data(), Ks.data());
    for(size_t k = 0; k < n2; k++) DMb[k] = 2.0 * Js[k] - Ks[k];
    // DM = sum_i C_i C_i^T  ->  Cbar_i[a] += 2 sum_b DMbar[a][b] C_i[b]   (DMbar symmetric)
    for(int i = 0; i < nc; i++) for(int a = 0; a < n; a++){
        double sum = 0.0; for(int b = 0; b < n; b++) sum += DMb[(size_t)a * n + b] * C[(size_t)i * n + b];
        out->Cbar[(size_t)i * n + a] += 2.0 * sum;
    }
    {   // seeds of s = sum Hs_ab (2J[DM] - K[DM])_ab with respect to L and V: the symmetric bilinear form B(Hs, DM), cJ = 2, cK = -1
        if(eig_out) ri_jk_vjp_eig(E, Hs.data(), DM.data(), 2.0, -1.0, out->Lbar.data(), out->Vbar.data());
        else{
            std::vector<double> Lbt(n2 * naux, 0.0), Vbt((size_t)naux * naux, 0.0);
            ri_jk_vjp_eig(E, Hs.data(), DM.data(), 2.0, -1.0, Lbt.data(), Vbt.data());
            ri_eig_to_aux(E, Lbt.data(), Vbt.data(), out->Lbar.data(), out->Vbar.data(), true);
        }
    }
    return 0;
}
