#include "ri_gradient.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>

#include "blas_link.h"
#include "matr.h"
#include "common_vars.h"

namespace{

// y = A * B for row-major (m x k)(k x n)
inline void mm(int m, int n, int k, const double * A, const double * B, double * C, double beta = 0.0, double alpha = 1.0,
               bool tA = false, bool tB = false){
    cblas_dgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                A, tA ? m : k, B, tB ? k : n, beta, C, n);
}

}

void ri_eigenvalue_range(int naux, const double * V, double * lmin, double * lmax){
    std::vector<double> A(V, V + (size_t)naux * naux), ev(naux);
    lapack_diag(A.data(), ev.data(), naux);
    *lmin = ev[0]; *lmax = ev[naux - 1];   // ascending
}

double ri_min_eigenvalue(int naux, const double * V){
    double lo, hi;
    ri_eigenvalue_range(naux, V, &lo, &hi);
    return lo;
}

void ri_vjp_dense(int npair, int naux, const double * L, const double * Vinv, const double * Q,
                  double * Lbar, double * Vbar){
    const size_t np = npair, na = naux;
    std::vector<double> LA(np * na), QLA(np * na), QtLA(np * na), tmp(na * na);
    mm(npair, naux, naux, L, Vinv, LA.data());                     // L V^-1
    mm(npair, naux, npair, Q, LA.data(), QLA.data());              // Q  L V^-1
    mm(npair, naux, npair, Q, LA.data(), QtLA.data(), 0.0, 1.0, true, false);   // Q^T L V^-1
    for(size_t i = 0; i < np * na; i++) Lbar[i] = QLA[i] + QtLA[i];
    // Vbar = -sym( (L V^-1)^T Q (L V^-1) )
    mm(naux, naux, npair, LA.data(), QLA.data(), tmp.data(), 0.0, 1.0, true, false);
    for(int p = 0; p < naux; p++) for(int q = 0; q < naux; q++)
        Vbar[p * na + q] = -0.5 * (tmp[p * na + q] + tmp[q * na + p]);
}

void ri_jk_vjp(int n, int naux, const double * L, const double * V, const double * P,
               double cJ, double cK, double * Lbar, double * Vbar){
    // Everything is done in the eigenbasis of the metric, V = U^T diag(lam) U (rows of U are eigenvectors):
    // with Lt = L U^T the fitted integrals are g = Lt diag(1/lam) Lt^T, and V^-1 never appears as a matrix.
    // (Forming V^-1 and then V^-1 M V^-1 loses the digits of the near-null modes: with a metric of
    // condition number 6e5 that was a 4e-7 Eh/bohr error in the gradient, 1e-9 once done this way.)
    const size_t n2 = (size_t)n * n, na = naux;
    std::vector<double> U(V, V + na * na), lam(na);
    lapack_diag(U.data(), lam.data(), naux);
    std::vector<double> Lt(n2 * na), Lbt(n2 * na), Vbt(na * na);
    mm((int)n2, naux, naux, L, U.data(), Lt.data(), 0.0, 1.0, false, true);            // Lt[ab][i] = sum_P L[ab][P] U[i][P]

    // Coulomb: t_i = sum_ab P_ab Lt_ab,i ; c_i = t_i / lam_i ; Lbt_J = 2 cJ P_ab c_i ; Vbt_J = -cJ c_i c_j
    std::vector<double> t(na, 0.0), c(na, 0.0);
    mm(1, naux, (int)n2, P, Lt.data(), t.data());
    for(size_t i = 0; i < na; i++) c[i] = t[i] / lam[i];
    for(size_t ab = 0; ab < n2; ab++) for(size_t i = 0; i < na; i++) Lbt[ab * na + i] = 2.0 * cJ * P[ab] * c[i];
    for(size_t i = 0; i < na; i++) for(size_t j = 0; j < na; j++) Vbt[i * na + j] = -cJ * c[i] * c[j];

    if(cK != 0.0){
        // Exchange: T_i = P Lt_i P (Lt_i the n x n matrix Lt[:, :, i]); Mt_ij = sum_ab Lt_ab,i T_j,ab
        std::vector<double> TQ(n2 * na);                                               // TQ[ab][i] = T_i,ab
#pragma omp parallel
        {
            std::vector<double> Li(n2), X(n2), Y(n2);
#pragma omp for
            for(int i = 0; i < naux; i++){
                for(size_t ab = 0; ab < n2; ab++) Li[ab] = Lt[ab * na + i];
                mm(n, n, n, P, Li.data(), X.data());
                mm(n, n, n, X.data(), P, Y.data());
                for(size_t ab = 0; ab < n2; ab++) TQ[ab * na + i] = Y[ab];
            }
        }
        std::vector<double> Mt(na * na);
        mm(naux, naux, (int)n2, Lt.data(), TQ.data(), Mt.data(), 0.0, 1.0, true, false);
        // Lbt_K[ab][i] = 2 cK T_i,ab / lam_i ; Vbt_K = -cK sym(Mt_ij / (lam_i lam_j))
        for(size_t ab = 0; ab < n2; ab++) for(size_t i = 0; i < na; i++) Lbt[ab * na + i] += 2.0 * cK * TQ[ab * na + i] / lam[i];
        for(size_t i = 0; i < na; i++) for(size_t j = 0; j < na; j++)
            Vbt[i * na + j] += -cK * 0.5 * (Mt[i * na + j] + Mt[j * na + i]) / (lam[i] * lam[j]);
    }
    // back to the auxiliary basis: Lbar = Lbt U, Vbar = U^T Vbt U
    mm((int)n2, naux, naux, Lbt.data(), U.data(), Lbar);
    std::vector<double> tmp(na * na);
    mm(naux, naux, naux, U.data(), Vbt.data(), tmp.data(), 0.0, 1.0, true, false);
    mm(naux, naux, naux, tmp.data(), U.data(), Vbar);
}

void ri_inv_sqrt_vjp(int n, const double * V, const double * Fbar, double * Vbar){
    std::vector<double> U(V, V + (size_t)n * n), lam(n);
    lapack_diag(U.data(), lam.data(), n);                           // rows of U are the eigenvectors
    // Y = U Fbar_sym U^T in the eigenbasis, Y_ij *= Gamma_ij, Vbar = U^T Y U
    std::vector<double> Fs((size_t)n * n), X((size_t)n * n), Y((size_t)n * n);
    for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) Fs[i * n + j] = 0.5 * (Fbar[i * n + j] + Fbar[j * n + i]);
    mm(n, n, n, U.data(), Fs.data(), X.data());
    mm(n, n, n, X.data(), U.data(), Y.data(), 0.0, 1.0, false, true);
    for(int i = 0; i < n; i++) for(int j = 0; j < n; j++){
        const double li = lam[i], lj = lam[j];
        double gam;
        if(std::fabs(li - lj) > 1e-9 * std::max(li, lj)) gam = (1.0 / std::sqrt(li) - 1.0 / std::sqrt(lj)) / (li - lj);
        else gam = -0.5 * std::pow(0.5 * (li + lj), -1.5);
        Y[i * n + j] *= gam;
    }
    mm(n, n, n, U.data(), Y.data(), X.data(), 0.0, 1.0, true, false);
    mm(n, n, n, X.data(), U.data(), Vbar);
}

// ---- Fock matrices and the active-space term, in the eigenbasis of the metric ----------------------------
void ri_fock_JK(int n, int naux, const double * L, const double * V, const double * D, double * J, double * K){
    const size_t n2 = (size_t)n * n, na = naux;
    std::vector<double> U(V, V + na * na), lam(na);
    lapack_diag(U.data(), lam.data(), naux);
    std::vector<double> Lt(n2 * na);
    mm((int)n2, naux, naux, L, U.data(), Lt.data(), 0.0, 1.0, false, true);
    std::fill(J, J + n2, 0.0); std::fill(K, K + n2, 0.0);
    std::vector<double> Li(n2), X(n2), Y(n2);
    for(size_t i = 0; i < na; i++){
        for(size_t ab = 0; ab < n2; ab++) Li[ab] = Lt[ab * na + i];
        double t = 0.0; for(size_t ab = 0; ab < n2; ab++) t += D[ab] * Li[ab];
        const double w = 1.0 / lam[i];
        for(size_t ab = 0; ab < n2; ab++) J[ab] += w * t * Li[ab];
        mm(n, n, n, Li.data(), D, X.data());              // Li D
        mm(n, n, n, X.data(), Li.data(), Y.data());       // Li D Li
        for(size_t ab = 0; ab < n2; ab++) K[ab] += w * Y[ab];
    }
}

void ri_active_vjp(int n, int naux, const double * L, const double * V, const double * Cact, int na_, const double * lam4,
                   double * Lbar, double * Vbar, double * GC, double * E){
    const size_t n2 = (size_t)n * n, na = naux, a2 = (size_t)na_ * na_;
    std::vector<double> U(V, V + na * na), lam(na);
    lapack_diag(U.data(), lam.data(), naux);
    std::vector<double> Lt(n2 * na);
    mm((int)n2, naux, naux, L, U.data(), Lt.data(), 0.0, 1.0, false, true);

    // pair-exchange symmetrized lam on the active pairs
    std::vector<double> ls(a2 * a2);
    for(size_t x = 0; x < a2; x++) for(size_t y = 0; y < a2; y++) ls[x * a2 + y] = 0.5 * (lam4[x * a2 + y] + lam4[y * a2 + x]);

    // active-MO transform of every eigen-auxiliary matrix:  La[tu, i] = Cact^T Li Cact
    std::vector<double> La(a2 * na), Li(n2), X((size_t)n * na_), Y(a2);
    for(size_t i = 0; i < na; i++){
        for(size_t ab = 0; ab < n2; ab++) Li[ab] = Lt[ab * na + i];
        mm(n, na_, n, Li.data(), Cact, X.data());                                   // Li Cact   (n x na_)
        mm(na_, na_, n, Cact, X.data(), Y.data(), 0.0, 1.0, true, false);           // Cact^T (Li Cact)
        for(size_t tu = 0; tu < a2; tu++) La[tu * na + i] = Y[tu];
    }
    std::vector<double> Zf(a2 * na);                                                // Zf[tu,i] = sum_vw ls[tu,vw] La[vw,i]
    mm((int)a2, naux, (int)a2, ls.data(), La.data(), Zf.data());
    double e = 0.0;
    for(size_t i = 0; i < na; i++){ double s = 0; for(size_t tu = 0; tu < a2; tu++) s += La[tu * na + i] * Zf[tu * na + i]; e += 0.5 * s / lam[i]; }
    *E = e;

    // Vbt_ij = -sym( M_ij / (lam_i lam_j) ),  M = 1/2 La^T Zf
    std::vector<double> M(na * na), Vbt(na * na), Lbt(n2 * na, 0.0);
    mm(naux, naux, (int)a2, La.data(), Zf.data(), M.data(), 0.0, 0.5, true, false);
    for(size_t i = 0; i < na; i++) for(size_t j = 0; j < na; j++) Vbt[i * na + j] = -0.5 * (M[i * na + j] + M[j * na + i]) / (lam[i] * lam[j]);
    // Lbt[ab,i] = sum_tu Cact_at Cact_bu Zf[tu,i] / lam_i
    std::vector<double> Zi(a2), T1((size_t)n * na_), Pm(n2);
    for(size_t i = 0; i < na; i++){
        for(size_t tu = 0; tu < a2; tu++) Zi[tu] = Zf[tu * na + i] / lam[i];
        mm(n, na_, na_, Cact, Zi.data(), T1.data());                                // Cact Z      (n x na_)
        mm(n, n, na_, T1.data(), Cact, Pm.data(), 0.0, 1.0, false, true);           // (Cact Z) Cact^T
        for(size_t ab = 0; ab < n2; ab++) Lbt[ab * na + i] = Pm[ab];
    }
    // back to the auxiliary basis and add to the seeds
    std::vector<double> tmpL(n2 * na), tmpV(na * na), tmpV2(na * na);
    mm((int)n2, naux, naux, Lbt.data(), U.data(), tmpL.data());
    for(size_t k = 0; k < n2 * na; k++) Lbar[k] += tmpL[k];
    mm(naux, naux, naux, U.data(), Vbt.data(), tmpV.data(), 0.0, 1.0, true, false);
    mm(naux, naux, naux, tmpV.data(), U.data(), tmpV2.data());
    for(size_t k = 0; k < na * na; k++) Vbar[k] += tmpV2[k];

    if(GC){   // G[a,t] = sum_i sum_u (Z_tu,i + Z_ut,i) (Li Cact)[a,u],  Z = Zf / lam_i
        std::fill(GC, GC + (size_t)n * na_, 0.0);
        std::vector<double> LC((size_t)n * na_), Zs(a2);
        for(size_t i = 0; i < na; i++){
            for(size_t ab = 0; ab < n2; ab++) Li[ab] = Lt[ab * na + i];
            mm(n, na_, n, Li.data(), Cact, LC.data());
            for(int t = 0; t < na_; t++) for(int u = 0; u < na_; u++) Zs[t * na_ + u] = (Zf[((size_t)t * na_ + u) * na + i] + Zf[((size_t)u * na_ + t) * na + i]) / lam[i];
            mm(n, na_, na_, LC.data(), Zs.data(), GC, 1.0, 1.0, false, true);        // GC[a,t] += sum_u LC[a,u] Zs[t,u]
        }
    }
}

// ---- cached eigendecomposition --------------------------------------------------------------------------------------
void RIEig::build(int n_, int naux_, const double * L, const double * V){
    n = n_; naux = naux_;
    const size_t n2 = (size_t)n * n, na = naux;
    U.assign(V, V + na * na); lam.resize(na);
    lapack_diag(U.data(), lam.data(), naux);
    Lt.resize(n2 * na);
    mm((int)n2, naux, naux, L, U.data(), Lt.data(), 0.0, 1.0, false, true);
}

void ri_eig_to_aux(const RIEig & E, const double * Lbt, const double * Vbt, double * Lbar, double * Vbar, bool accumulate){
    const int n = E.n, naux = E.naux; const size_t n2 = (size_t)n * n, na = naux;
    const double beta = accumulate ? 1.0 : 0.0;
    if(Lbt && Lbar) mm((int)n2, naux, naux, Lbt, E.U.data(), Lbar, beta);
    if(Vbt && Vbar){
        std::vector<double> tmp(na * na);
        mm(naux, naux, naux, E.U.data(), Vbt, tmp.data(), 0.0, 1.0, true, false);
        mm(naux, naux, naux, tmp.data(), E.U.data(), Vbar, beta);
    }
}

void ri_fock_JK_eig(const RIEig & E, const double * D, double * J, double * K, const double * D2, double * J2, double * K2){
    const int n = E.n, naux = E.naux; const size_t n2 = (size_t)n * n, na = naux;
    const int nd = D2 ? 2 : 1;
    const double * Ds[2] = {D, D2}; double * Js[2] = {J, J2}; double * Ks[2] = {K, K2};
    for(int s = 0; s < nd; s++){
        // Coulomb: t_i = sum_ab D_ab Lt_ab,i ; J = Lt (t / lam)
        std::vector<double> t(na);
        mm(1, naux, (int)n2, Ds[s], E.Lt.data(), t.data());
        for(size_t i = 0; i < na; i++) t[i] /= E.lam[i];
        mm((int)n2, 1, naux, E.Lt.data(), t.data(), Js[s]);
        std::fill(Ks[s], Ks[s] + n2, 0.0);
    }
    // exchange: K = sum_i (Li D Li) / lam_i, per-thread accumulators
#pragma omp parallel
    {
        std::vector<double> Li(n2), X(n2), Y(n2), Kl((size_t)nd * n2, 0.0);
#pragma omp for schedule(dynamic)
        for(int i = 0; i < naux; i++){
            for(size_t ab = 0; ab < n2; ab++) Li[ab] = E.Lt[ab * na + i];
            const double w = 1.0 / E.lam[i];
            for(int s = 0; s < nd; s++){
                mm(n, n, n, Li.data(), Ds[s], X.data());
                mm(n, n, n, X.data(), Li.data(), Y.data());
                for(size_t ab = 0; ab < n2; ab++) Kl[(size_t)s * n2 + ab] += w * Y[ab];
            }
        }
#pragma omp critical
        for(int s = 0; s < nd; s++) for(size_t ab = 0; ab < n2; ab++) Ks[s][ab] += Kl[(size_t)s * n2 + ab];
    }
}

void ri_fock_JK_eig_lr(const RIEig & E, const double * D, const double * X, const double * Y, int r, double * J, double * K){
    const int n = E.n, naux = E.naux; const size_t n2 = (size_t)n * n, na = naux;
    std::fill(K, K + n2, 0.0);
    if(r == 0){
        std::vector<double> t(na);
        mm(1, naux, (int)n2, D, E.Lt.data(), t.data());
        for(size_t i = 0; i < na; i++) t[i] /= E.lam[i];
        mm((int)n2, 1, naux, E.Lt.data(), t.data(), J);
        return;
    }
    // exchange of D = X Y^T:  K_ab = sum_i sum_k (L_i X)_ak (L_i Y)_bk / lam_i  with (L_i X)_ak = sum_c L[a,c,i] X[c,k].
    // For each a the slice L[a,:,:] (n x naux) is contiguous: A[a][k][i] = X^T L[a] (r x naux), likewise B (scaled by 1/lam), then K = A B^T over (k,i).
    // Coulomb from the same factors: t_i / lam_i = sum_ab D_ab L_ab,i / lam_i = sum_{a,k} X[a,k] B[a][k][i]; J = Lt (t / lam): one pass over L.
    const size_t ri = (size_t)r * na;
    std::vector<double> A((size_t)n * ri), B((size_t)n * ri);
    set_blas_non_par();
#pragma omp parallel for schedule(static)
    for(int a2 = 0; a2 < n; a2++){
        const double * La = E.Lt.data() + (size_t)a2 * n * na;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r, naux, n, 1.0, X, r, La, naux, 0.0, A.data() + (size_t)a2 * ri, naux);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r, naux, n, 1.0, Y, r, La, naux, 0.0, B.data() + (size_t)a2 * ri, naux);
        double * Bb = B.data() + (size_t)a2 * ri;
        for(int k = 0; k < r; k++) for(size_t i = 0; i < na; i++) Bb[(size_t)k * na + i] /= E.lam[i];
    }
    set_blas_par();
    std::vector<double> c(na);
    cblas_dgemv(CblasRowMajor, CblasTrans, (int)((size_t)n * r), naux, 1.0, B.data(), naux, X, 1, 0.0, c.data(), 1);
    mm((int)n2, 1, naux, E.Lt.data(), c.data(), J);
    mm(n, n, (int)ri, A.data(), B.data(), K, 0.0, 1.0, false, true);
    (void)D;
}

void ri_jk_vjp_eig(const RIEig & E, const double * P, const double * Q, double cJ, double cK, double * Lbt, double * Vbt){
    const int n = E.n, naux = E.naux; const size_t n2 = (size_t)n * n, na = naux;
    const bool same = (P == Q);
    std::vector<double> tP(na), tQ(na);
    mm(1, naux, (int)n2, P, E.Lt.data(), tP.data());
    if(same) tQ = tP; else mm(1, naux, (int)n2, Q, E.Lt.data(), tQ.data());
    for(size_t i = 0; i < na; i++){ tP[i] /= E.lam[i]; tQ[i] /= E.lam[i]; }
    // Coulomb: Lbt += cJ (P cQ + Q cP),  Vbt -= cJ sym(cP cQ^T)
    if(Lbt) for(size_t ab = 0; ab < n2; ab++) for(size_t i = 0; i < na; i++) Lbt[ab * na + i] += cJ * (P[ab] * tQ[i] + Q[ab] * tP[i]);
    if(Vbt) for(size_t i = 0; i < na; i++) for(size_t j = 0; j < na; j++) Vbt[i * na + j] -= 0.5 * cJ * (tP[i] * tQ[j] + tQ[i] * tP[j]);
    if(cK == 0.0) return;
    // exchange: T_i = sym(P Li Q);  Lbt += 2 cK T_i / lam_i ;  Vbt -= cK sym(Lt^T T)_ij / (lam_i lam_j)
    // The auxiliary functions are processed in batches: the T_i of one batch (n^2 x cb) exist at a time, and Lt^T T is formed batch by batch.
    const int cb = (int)std::max<long>(8, std::min<long>(naux, (long)(16000000 / std::max<size_t>(n2, 1))));
    std::vector<double> Mt(Vbt ? na * na : 0, 0.0), Tb(n2 * cb);
    for(int i0 = 0; i0 < naux; i0 += cb){
        const int nb = std::min(cb, naux - i0);
#pragma omp parallel
        {
            std::vector<double> Li(n2), X(n2), Y(n2), Y2(same ? 0 : n2);
#pragma omp for schedule(dynamic)
            for(int ii = 0; ii < nb; ii++){
                const int i = i0 + ii;
                for(size_t ab = 0; ab < n2; ab++) Li[ab] = E.Lt[ab * na + i];
                mm(n, n, n, P, Li.data(), X.data());
                mm(n, n, n, X.data(), Q, Y.data());                  // P Li Q
                if(!same){ mm(n, n, n, Q, Li.data(), X.data()); mm(n, n, n, X.data(), P, Y2.data()); for(size_t ab = 0; ab < n2; ab++) Y[ab] = 0.5 * (Y[ab] + Y2[ab]); }
                const double w = 2.0 * cK / E.lam[i];
                if(Lbt) for(size_t ab = 0; ab < n2; ab++) Lbt[ab * na + i] += w * Y[ab];
                if(Vbt) for(size_t ab = 0; ab < n2; ab++) Tb[ab * cb + ii] = Y[ab];
            }
        }
        if(Vbt){   // Mt[:, i0:i0+nb] = Lt^T T_batch
            std::vector<double> Mb(na * nb);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, naux, nb, (int)n2, 1.0, E.Lt.data(), naux, Tb.data(), cb, 0.0, Mb.data(), nb);
            for(size_t i = 0; i < na; i++) for(int jj = 0; jj < nb; jj++) Mt[i * na + i0 + jj] = Mb[i * nb + jj];
        }
    }
    if(Vbt) for(size_t i = 0; i < na; i++) for(size_t j = 0; j < na; j++) Vbt[i * na + j] -= cK * 0.5 * (Mt[i * na + j] + Mt[j * na + i]) / (E.lam[i] * E.lam[j]);
}

void ri_eig_to_aux_inplace(const RIEig & E, double * Lbar, double * Vbar){
    const int n = E.n, naux = E.naux; const size_t n2 = (size_t)n * n, na = naux;
    if(Lbar){
        const size_t rb = std::max<size_t>(64, std::min<size_t>(n2, 4000000 / std::max<size_t>(na, 1)));
        std::vector<double> tmp(rb * na);
        for(size_t r0 = 0; r0 < n2; r0 += rb){
            const size_t m = std::min(rb, n2 - r0);
            mm((int)m, naux, naux, Lbar + r0 * na, E.U.data(), tmp.data());
            std::memcpy(Lbar + r0 * na, tmp.data(), sizeof(double) * m * na);
        }
    }
    if(Vbar){
        std::vector<double> tmp(na * na), out(na * na);
        mm(naux, naux, naux, E.U.data(), Vbar, tmp.data(), 0.0, 1.0, true, false);
        mm(naux, naux, naux, tmp.data(), E.U.data(), out.data());
        std::memcpy(Vbar, out.data(), sizeof(double) * na * na);
    }
}

void ri_aux_to_eig(const RIEig & E, const double * Lbar, const double * Vbar, double * Lbt, double * Vbt){
    const int n = E.n, naux = E.naux; const size_t n2 = (size_t)n * n, na = naux;
    if(Lbar && Lbt) mm((int)n2, naux, naux, Lbar, E.U.data(), Lbt, 1.0, 1.0, false, true);     // Lbt += Lbar U^T
    if(Vbar && Vbt){
        std::vector<double> tmp(na * na);
        mm(naux, naux, naux, E.U.data(), Vbar, tmp.data());                                     // U Vbar
        mm(naux, naux, naux, tmp.data(), E.U.data(), Vbt, 1.0, 1.0, false, true);              // Vbt += U Vbar U^T
    }
}

namespace{
// X[a][u][i] = (Lt_i Cv)[a,u]   (Cv: n x na, [b*na+u])
void active_X(const RIEig & E, const double * Cv, int na_, std::vector<double> * X){
    const int n = E.n, naux = E.naux;
    X->assign((size_t)n * na_ * naux, 0.0);
#pragma omp parallel for schedule(static)
    for(int a = 0; a < n; a++)
        mm(na_, naux, n, Cv, E.Lt.data() + (size_t)a * n * naux, X->data() + (size_t)a * na_ * naux, 0.0, 1.0, true, false);
}
// Lbt[a][b][i] += sum_tu A[a,t] B[b,u] Z[(t,u)][i]
void add_AZB(const RIEig & E, const double * A, const double * B, const double * Z, int na_, double * Lbt){
    const int n = E.n, naux = E.naux;
    std::vector<double> T1((size_t)n * na_ * naux);
    mm(n, na_ * naux, na_, A, Z, T1.data());                                      // T1[a][(u,i)] = sum_t A[a,t] Z[t][(u,i)]
#pragma omp parallel for schedule(static)
    for(int a = 0; a < n; a++)
        mm(n, naux, na_, B, T1.data() + (size_t)a * na_ * naux, Lbt + (size_t)a * n * naux, 1.0);
}
void pair_sym(const double * lam4, size_t a2, std::vector<double> * ls){
    ls->resize(a2 * a2);
    for(size_t x = 0; x < a2; x++) for(size_t y = 0; y < a2; y++) (*ls)[x * a2 + y] = 0.5 * (lam4[x * a2 + y] + lam4[y * a2 + x]);
}
}

void ri_active_La(const RIEig & E, const double * Cact, int na_, double * La, const double * dCa){
    const int n = E.n, naux = E.naux; const size_t a2 = (size_t)na_ * na_;
    std::vector<double> X; active_X(E, Cact, na_, &X);
    if(!dCa){ mm(na_, na_ * naux, n, Cact, X.data(), La, 0.0, 1.0, true, false); return; }
    // dLa_tu = Y_tu + Y_ut with Y = dCa^T Lt Ca (Lt_i symmetric)
    std::vector<double> Y(a2 * naux);
    mm(na_, na_ * naux, n, dCa, X.data(), Y.data(), 0.0, 1.0, true, false);
    for(int t = 0; t < na_; t++) for(int u = 0; u < na_; u++) for(int i = 0; i < naux; i++)
        La[((size_t)t * na_ + u) * naux + i] = Y[((size_t)t * na_ + u) * naux + i] + Y[((size_t)u * na_ + t) * naux + i];
}

void ri_active_vjp_eig(const RIEig & E, const double * Cact, int na_, const double * lam4, const double * La_in,
                       double * Lbt, double * Vbt, double * GC, double * Eout){
    const int n = E.n, naux = E.naux; const size_t na = naux, a2 = (size_t)na_ * na_;
    std::vector<double> La_own;
    const double * La = La_in;
    if(!La){ La_own.resize(a2 * na); ri_active_La(E, Cact, na_, La_own.data()); La = La_own.data(); }
    std::vector<double> ls; pair_sym(lam4, a2, &ls);
    std::vector<double> Zf(a2 * na);
    mm((int)a2, naux, (int)a2, ls.data(), La, Zf.data());
    double e = 0.0;
    for(size_t tu = 0; tu < a2; tu++) for(size_t i = 0; i < na; i++) e += 0.5 * La[tu * na + i] * Zf[tu * na + i] / E.lam[i];
    if(Eout) *Eout = e;
    if(Vbt){
        std::vector<double> M(na * na);
        mm(naux, naux, (int)a2, La, Zf.data(), M.data(), 0.0, 0.5, true, false);
        for(size_t i = 0; i < na; i++) for(size_t j = 0; j < na; j++) Vbt[i * na + j] -= 0.5 * (M[i * na + j] + M[j * na + i]) / (E.lam[i] * E.lam[j]);
    }
    std::vector<double> Z(a2 * na);
    for(size_t tu = 0; tu < a2; tu++) for(size_t i = 0; i < na; i++) Z[tu * na + i] = Zf[tu * na + i] / E.lam[i];
    if(Lbt) add_AZB(E, Cact, Cact, Z.data(), na_, Lbt);
    if(GC){
        std::vector<double> X; active_X(E, Cact, na_, &X);
        std::vector<double> Zs(a2 * na);
        for(int t = 0; t < na_; t++) for(int u = 0; u < na_; u++) for(size_t i = 0; i < na; i++)
            Zs[((size_t)t * na_ + u) * na + i] = Z[((size_t)t * na_ + u) * na + i] + Z[((size_t)u * na_ + t) * na + i];
        mm(n, na_, na_ * naux, X.data(), Zs.data(), GC, 0.0, 1.0, false, true);   // GC[a,t] = sum_(u,i) X[a][(u,i)] Zs[t][(u,i)]
    }
}

void ri_active_jvp_eig(const RIEig & E, const double * Cact, const double * dCa, int na_, const double * lam4, const double * dlam4,
                       const double * La, const double * dLa, double * dLbt, double * dVbt, double * dGC){
    const int n = E.n, naux = E.naux; const size_t na = naux, a2 = (size_t)na_ * na_;
    std::vector<double> ls, dls; pair_sym(lam4, a2, &ls); pair_sym(dlam4, a2, &dls);
    std::vector<double> Zf(a2 * na), dZf(a2 * na);
    mm((int)a2, naux, (int)a2, ls.data(), La, Zf.data());
    mm((int)a2, naux, (int)a2, dls.data(), La, dZf.data());
    mm((int)a2, naux, (int)a2, ls.data(), dLa, dZf.data(), 1.0);
    if(dVbt){
        std::vector<double> M(na * na);
        mm(naux, naux, (int)a2, dLa, Zf.data(), M.data(), 0.0, 0.5, true, false);
        mm(naux, naux, (int)a2, La, dZf.data(), M.data(), 1.0, 0.5, true, false);
        for(size_t i = 0; i < na; i++) for(size_t j = 0; j < na; j++) dVbt[i * na + j] -= 0.5 * (M[i * na + j] + M[j * na + i]) / (E.lam[i] * E.lam[j]);
    }
    std::vector<double> Z(a2 * na), dZ(a2 * na);
    for(size_t k = 0; k < a2 * na; k++){ Z[k] = Zf[k] / E.lam[k % na]; dZ[k] = dZf[k] / E.lam[k % na]; }
    if(dLbt){
        add_AZB(E, dCa, Cact, Z.data(), na_, dLbt);
        add_AZB(E, Cact, Cact, dZ.data(), na_, dLbt);
        add_AZB(E, Cact, dCa, Z.data(), na_, dLbt);
    }
    if(dGC){
        auto zsym = [&](const std::vector<double> & A, std::vector<double> * S){
            S->resize(a2 * na);
            for(int t = 0; t < na_; t++) for(int u = 0; u < na_; u++) for(size_t i = 0; i < na; i++)
                (*S)[((size_t)t * na_ + u) * na + i] = A[((size_t)t * na_ + u) * na + i] + A[((size_t)u * na_ + t) * na + i];
        };
        std::vector<double> Zs, dZs, X, dX;
        zsym(Z, &Zs); zsym(dZ, &dZs);
        active_X(E, Cact, na_, &X); active_X(E, dCa, na_, &dX);
        mm(n, na_, na_ * naux, dX.data(), Zs.data(), dGC, 0.0, 1.0, false, true);
        mm(n, na_, na_ * naux, X.data(), dZs.data(), dGC, 1.0, 1.0, false, true);
    }
}
