// Adjoints of the batched DF terms of DSRG-PT2 (dsrg_sf_batch.cpp: CCVV, CAVV, CCAV), in the same batched, never-materialized form:
// the four-index integrals of one batch index pair are built by GEMM, the energy and its adjoints accumulated, and the adjoint of the integrals
// is returned to the B-tensors by GEMM. Memory O(n_v^2) / O(n_v n_a) per thread instead of the n_v^2 n_c^2 tensors of the tape.
#include "dsrg_grad.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>

#include "blas_link.h"
#include "tensor_tape.h"

extern int num_threads;

namespace{
inline double q_of(double D, double s){ return tt::reg_value(tt::Reg::Q, s, D); }
inline double dq_of(double D, double s){ return tt::reg_deriv(tt::Reg::Q, s, D); }
inline double R_of(double D, double s){ return tt::reg_value(tt::Reg::R, s, D); }
inline double dR_of(double D, double s){ return tt::reg_deriv(tt::Reg::R, s, D); }
}

// CCVV: E = sum_{i<=j} w_ij sum_ef F(D) J_ef (2 J_ef - J_fe),  J = (ie|jf),  F = (1+q) R  (1/D with ccvv_zero),  w = 1 (i == j) or 2
double dsrg_batch_CCVV(const DsrgTapeIn & in, double seed, std::vector<double> * VCbp, std::vector<double> * ecbp, std::vector<double> * evbp){
    const bool adj = VCbp != nullptr && seed != 0.0;
    const int nc = in.nc, nv = in.nv; const long nQ = in.naux; const double s = in.s;
    if(nc == 0 || nv == 0) return 0.0;
    const double * VC = in.VC;
    std::vector<std::pair<int,int>> ij; for(int i = 0; i < nc; i++) for(int j = i; j < nc; j++) ij.emplace_back(i, j);
    double E = 0.0;
    std::vector<std::vector<double>> ec_t(num_threads, std::vector<double>(nc, 0.0)), ev_t(num_threads, std::vector<double>(nv, 0.0));
    set_blas_non_par();
#pragma omp parallel reduction(+:E)
    {
        const int th = omp_get_thread_num();
        std::vector<double> J((size_t)nv * nv), Jb((size_t)nv * nv);
#pragma omp for schedule(dynamic)
        for(long p = 0; p < (long)ij.size(); p++){
            const int i = ij[p].first, j = ij[p].second;
            const double w = (i == j) ? 1.0 : 2.0;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nv, nv, nQ, 1.0, VC + (long)i * nQ, nc * nQ, VC + (long)j * nQ, nc * nQ, 0.0, J.data(), nv);
            double loc = 0.0, dci = 0.0;
            for(int e = 0; e < nv; e++) for(int f = 0; f < nv; f++){
                const double D = in.ec[i] + in.ec[j] - in.ev[e] - in.ev[f];
                double F, dF;
                if(in.ccvv_zero){ F = 1.0 / D; dF = -1.0 / (D * D); }
                else { const double q = q_of(D, s), R = R_of(D, s); F = (1.0 + q) * R; dF = dq_of(D, s) * R + (1.0 + q) * dR_of(D, s); }
                const double Jef = J[(size_t)e * nv + f], Jfe = J[(size_t)f * nv + e];
                loc += F * Jef * (2.0 * Jef - Jfe);
                if(!adj) continue;
                Jb[(size_t)e * nv + f] = seed * w * (4.0 * F * Jef - 2.0 * F * Jfe);      // F symmetric in (e,f)
                const double Db = seed * w * dF * Jef * (2.0 * Jef - Jfe);
                dci += Db; ev_t[th][e] -= Db; ev_t[th][f] -= Db;
            }
            E += w * loc;
            if(!adj) continue;
            ec_t[th][i] += dci; ec_t[th][j] += dci;
            std::vector<double> & VCb = *VCbp;
#pragma omp critical(dsrg_ccvv)
            {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nv, nQ, nv, 1.0, Jb.data(), nv, VC + (long)j * nQ, nc * nQ, 1.0, VCb.data() + (long)i * nQ, nc * nQ);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nv, nQ, nv, 1.0, Jb.data(), nv, VC + (long)i * nQ, nc * nQ, 1.0, VCb.data() + (long)j * nQ, nc * nQ);
            }
        }
    }
    set_blas_par();
    if(adj) for(int t = 0; t < num_threads; t++){ for(int i = 0; i < nc; i++) (*ecbp)[i] += ec_t[t][i]; for(int e = 0; e < nv; e++) (*evbp)[e] += ev_t[t][e]; }
    return E;
}

// CAVV: for (i, c): J1[e,u] = (ie|cu), J2[e,u] = (eu|ci), D = e_c[i] + e_a[u] - e_v[c] - e_v[e];
//   jk = (2 J1 - J2) R(D),  j1s = J1 (1 + q(D)),  C[v,u] = sum_{i,c,e} jk[e,v] j1s[e,u]   (the energy closes it with L1, the dressed operator folds it).
// Cb == nullptr: forward, C (na x na) returned. Otherwise the adjoint for the seed Cb on C, accumulated into the B-tensor and eps adjoints.
void dsrg_batch_CAVV(const DsrgTapeIn & in, std::vector<double> * Cout, const double * Cb, std::vector<double> * VCbp, std::vector<double> * VAbp,
                     std::vector<double> * ecbp, std::vector<double> * eabp, std::vector<double> * evbp){
    const int nc = in.nc, na = in.na, nv = in.nv; const long nQ = in.naux; const double s = in.s;
    const bool adj = Cb != nullptr;
    if(Cout) Cout->assign((size_t)na * na, 0.0);
    if(nc == 0 || na == 0 || nv == 0) return;
    const double * VC = in.VC, * VA = in.VA;
    std::vector<std::vector<double>> ec_t(num_threads, std::vector<double>(nc, 0.0)), ea_t(num_threads, std::vector<double>(na, 0.0)),
                                     ev_t(num_threads, std::vector<double>(nv, 0.0)), C_t(num_threads, std::vector<double>((size_t)na * na, 0.0));
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> J1((size_t)nv * na), J2((size_t)nv * na), jk((size_t)nv * na), j1s((size_t)nv * na), jkb((size_t)nv * na), j1sb((size_t)nv * na),
                            J1b((size_t)nv * na), J2b((size_t)nv * na), VCci(nQ, 0.0);
#pragma omp for schedule(dynamic) collapse(2)
        for(int i = 0; i < nc; i++) for(int c = 0; c < nv; c++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nv, na, nQ, 1.0, VC + (long)i * nQ, nc * nQ, VA + (long)c * na * nQ, nQ, 0.0, J1.data(), na);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, nv * na, nQ, 1.0, VA, nQ, VC + ((long)c * nc + i) * nQ, 1, 0.0, J2.data(), 1);
            for(int e = 0; e < nv; e++) for(int u = 0; u < na; u++){
                const double D = in.ec[i] + in.ea[u] - in.ev[c] - in.ev[e];
                jk[(size_t)e * na + u] = (2.0 * J1[(size_t)e * na + u] - J2[(size_t)e * na + u]) * R_of(D, s);
                j1s[(size_t)e * na + u] = J1[(size_t)e * na + u] * (1.0 + q_of(D, s));
            }
            if(!adj){   // C[v,u] += sum_e jk[e,v] j1s[e,u]
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, na, nv, 1.0, jk.data(), na, j1s.data(), na, 1.0, C_t[th].data(), na);
                continue;
            }
            // jkb[e,v] = sum_u Cb[v,u] j1s[e,u];  j1sb[e,u] = sum_v jk[e,v] Cb[v,u]
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nv, na, na, 1.0, j1s.data(), na, Cb, na, 0.0, jkb.data(), na);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nv, na, na, 1.0, jk.data(), na, Cb, na, 0.0, j1sb.data(), na);
            double dci = 0.0, dcc = 0.0;
            for(int e = 0; e < nv; e++) for(int u = 0; u < na; u++){
                const size_t k = (size_t)e * na + u;
                const double D = in.ec[i] + in.ea[u] - in.ev[c] - in.ev[e];
                const double R = R_of(D, s), q = q_of(D, s);
                J1b[k] = 2.0 * R * jkb[k] + (1.0 + q) * j1sb[k];
                J2b[k] = -R * jkb[k];
                const double Db = jkb[k] * (2.0 * J1[k] - J2[k]) * dR_of(D, s) + j1sb[k] * J1[k] * dq_of(D, s);
                dci += Db; ea_t[th][u] += Db; dcc -= Db; ev_t[th][e] -= Db;
            }
            ec_t[th][i] += dci; ev_t[th][c] += dcc;
            // J2[e,u] = VA[e,u] . VC[c,i]  =>  VCbar[c,i] += sum_eu J2b VA[e,u]
            cblas_dgemv(CblasRowMajor, CblasTrans, nv * na, nQ, 1.0, VA, nQ, J2b.data(), 1, 0.0, VCci.data(), 1);
            std::vector<double> & VCb = *VCbp, & VAb = *VAbp;
#pragma omp critical(dsrg_cavv)
            {
                // J1 = VC_i VA_c^T: VCbar_i += J1b VA_c,  VAbar_c += J1b^T VC_i
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nv, nQ, na, 1.0, J1b.data(), na, VA + (long)c * na * nQ, nQ, 1.0, VCb.data() + (long)i * nQ, nc * nQ);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, nQ, nv, 1.0, J1b.data(), na, VC + (long)i * nQ, nc * nQ, 1.0, VAb.data() + (long)c * na * nQ, nQ);
                // J2: VAbar[e,u] += J2b[e,u] VC[c,i]
                cblas_dger(CblasRowMajor, nv * na, nQ, 1.0, J2b.data(), 1, VC + ((long)c * nc + i) * nQ, 1, VAb.data(), nQ);
                double * dst = VCb.data() + ((long)c * nc + i) * nQ;
                for(long Q = 0; Q < nQ; Q++) dst[Q] += VCci[Q];
            }
        }
    }
    set_blas_par();
    for(int t = 0; t < num_threads; t++){
        if(adj){
            for(int i = 0; i < nc; i++) (*ecbp)[i] += ec_t[t][i];
            for(int u = 0; u < na; u++) (*eabp)[u] += ea_t[t][u];
            for(int e = 0; e < nv; e++) (*evbp)[e] += ev_t[t][e];
        }
        else if(Cout) for(size_t k = 0; k < Cout->size(); k++) (*Cout)[k] += C_t[t][k];
    }
}

// CCAV: for virtual c: J[u,m,n] = (um|cn), jk = 2 J[u,m,n] - J[u,n,m], D = e_c[m] + e_c[n] - e_v[c] - e_a[u];
//   js = J (1 + q), jks = jk R,  C[v,u] = sum_{c,m,n} js[v,m,n] jks[u,m,n]   (closed with Eta = 2 - L1 by the energy). Same two modes as CAVV.
void dsrg_batch_CCAV(const DsrgTapeIn & in, std::vector<double> * Cout, const double * Cb, std::vector<double> * VCbp, std::vector<double> * CAbp,
                     std::vector<double> * ecbp, std::vector<double> * eabp, std::vector<double> * evbp){
    const int nc = in.nc, na = in.na, nv = in.nv; const long nQ = in.naux; const double s = in.s;
    const bool adj = Cb != nullptr;
    const size_t a2 = (size_t)na * na, nmn = (size_t)nc * nc;
    if(Cout) Cout->assign(a2, 0.0);
    if(nc == 0 || na == 0 || nv == 0) return;
    const double * VC = in.VC, * CA = in.CA;
    // Bac[(u,m)][Q] = CA[(m,u)][Q]
    std::vector<double> Bac((size_t)na * nc * nQ), Bacb(adj ? (size_t)na * nc * nQ : 0, 0.0);
    for(int u = 0; u < na; u++) for(int m = 0; m < nc; m++) for(long Q = 0; Q < nQ; Q++) Bac[((size_t)u * nc + m) * nQ + Q] = CA[((size_t)m * na + u) * nQ + Q];
    std::vector<std::vector<double>> ec_t(num_threads, std::vector<double>(nc, 0.0)), ea_t(num_threads, std::vector<double>(na, 0.0)),
                                     ev_t(num_threads, std::vector<double>(nv, 0.0)), C_t(num_threads, std::vector<double>(a2, 0.0));
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> J(na * nmn), jk(na * nmn), js(na * nmn), jks(na * nmn), jsb(na * nmn), jksb(na * nmn), Jb(na * nmn), jkb(na * nmn);
#pragma omp for schedule(dynamic)
        for(int c = 0; c < nv; c++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na * nc, nc, nQ, 1.0, Bac.data(), nQ, VC + (long)c * nc * nQ, nQ, 0.0, J.data(), nc);
            for(int u = 0; u < na; u++) for(int m = 0; m < nc; m++) for(int n = 0; n < nc; n++){
                const size_t k = ((size_t)u * nc + m) * nc + n;
                jk[k] = 2.0 * J[k] - J[((size_t)u * nc + n) * nc + m];
                const double D = in.ec[m] + in.ec[n] - in.ev[c] - in.ea[u];
                js[k] = J[k] * (1.0 + q_of(D, s)); jks[k] = jk[k] * R_of(D, s);
            }
            if(!adj){   // C[v,u] += sum_mn js[v,(m,n)] jks[u,(m,n)]
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na, na, (int)nmn, 1.0, js.data(), (int)nmn, jks.data(), (int)nmn, 1.0, C_t[th].data(), na);
                continue;
            }
            // jsb[v,mn] = sum_u Cb[v,u] jks[u,mn];  jksb[u,mn] = sum_v Cb[v,u] js[v,mn]
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na, (int)nmn, na, 1.0, Cb, na, jks.data(), (int)nmn, 0.0, jsb.data(), (int)nmn);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, (int)nmn, na, 1.0, Cb, na, js.data(), (int)nmn, 0.0, jksb.data(), (int)nmn);
            double dcc = 0.0;
            for(int u = 0; u < na; u++) for(int m = 0; m < nc; m++) for(int n = 0; n < nc; n++){
                const size_t k = ((size_t)u * nc + m) * nc + n;
                const double D = in.ec[m] + in.ec[n] - in.ev[c] - in.ea[u];
                const double q = q_of(D, s), R = R_of(D, s);
                jkb[k] = jksb[k] * R;
                Jb[k] = jsb[k] * (1.0 + q);
                const double Db = jsb[k] * J[k] * dq_of(D, s) + jksb[k] * jk[k] * dR_of(D, s);
                ec_t[th][m] += Db; ec_t[th][n] += Db; dcc -= Db; ea_t[th][u] -= Db;
            }
            ev_t[th][c] += dcc;
            // jk = 2 J[u,m,n] - J[u,n,m]
            for(int u = 0; u < na; u++) for(int m = 0; m < nc; m++) for(int n = 0; n < nc; n++)
                Jb[((size_t)u * nc + m) * nc + n] += 2.0 * jkb[((size_t)u * nc + m) * nc + n] - jkb[((size_t)u * nc + n) * nc + m];
            std::vector<double> & VCb = *VCbp;
#pragma omp critical(dsrg_ccav)
            {
                // J = Bac VC_c^T  =>  Bacb += Jb VC_c,  VCbar_c += Jb^T Bac
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na * nc, nQ, nc, 1.0, Jb.data(), nc, VC + (long)c * nc * nQ, nQ, 1.0, Bacb.data(), nQ);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nc, nQ, na * nc, 1.0, Jb.data(), nc, Bac.data(), nQ, 1.0, VCb.data() + (long)c * nc * nQ, nQ);
            }
        }
    }
    set_blas_par();
    if(adj){
        for(int u = 0; u < na; u++) for(int m = 0; m < nc; m++) for(long Q = 0; Q < nQ; Q++) (*CAbp)[((size_t)m * na + u) * nQ + Q] += Bacb[((size_t)u * nc + m) * nQ + Q];
        for(int t = 0; t < num_threads; t++){
            for(int i = 0; i < nc; i++) (*ecbp)[i] += ec_t[t][i];
            for(int u = 0; u < na; u++) (*eabp)[u] += ea_t[t][u];
            for(int e = 0; e < nv; e++) (*evbp)[e] += ev_t[t][e];
        }
    }
    else if(Cout) for(int t = 0; t < num_threads; t++) for(size_t k = 0; k < a2; k++) (*Cout)[k] += C_t[t][k];
}

// AAVV: per virtual e, G[x][f][y] = (ex|fy);  t2[(x,y),f] = G R(D),  vt[f,(x,y)] = G (1 + q(D)),  D = e_a[x] + e_a[y] - e_v[e] - e_v[f]  (q is even);
//   s2[(u,v),f] = 2 t2[(u,v),f] - t2[(v,u),f];  MT[(u,v),(w,x)] = sum_{e,f} t2[(u,v),f] vt[f,(w,x)],  MS likewise with s2.
// Forward (MTb == nullptr): MT, MS (n_a^4 each). Adjoint: the seeds MTb, MSb back to VA and the orbital energies.
void dsrg_batch_AAVV(const DsrgTapeIn & in, std::vector<double> * MT, std::vector<double> * MS, const double * MTb, const double * MSb,
                     std::vector<double> * VAbp, std::vector<double> * eabp, std::vector<double> * evbp){
    const int na = in.na, nv = in.nv; const long nQ = in.naux; const double s = in.s;
    const size_t a2 = (size_t)na * na, a4 = a2 * a2, fy = (size_t)nv * na;
    const bool adj = MTb != nullptr;
    if(MT) MT->assign(a4, 0.0);
    if(MS) MS->assign(a4, 0.0);
    if(na == 0 || nv == 0) return;
    const double * VA = in.VA;
    std::vector<std::vector<double>> MT_t(adj ? 0 : num_threads, std::vector<double>(a4, 0.0)), MS_t(adj ? 0 : num_threads, std::vector<double>(a4, 0.0));
    std::vector<std::vector<double>> ea_t(num_threads, std::vector<double>(na, 0.0)), ev_t(num_threads, std::vector<double>(nv, 0.0));
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> G((size_t)na * fy), t2(a2 * nv), vt((size_t)nv * a2), s2(a2 * nv), t2b(a2 * nv), s2b(a2 * nv), vtb((size_t)nv * a2), Gb((size_t)na * fy);
#pragma omp for schedule(dynamic)
        for(int e = 0; e < nv; e++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na, (int)fy, nQ, 1.0, VA + (long)e * na * nQ, nQ, VA, nQ, 0.0, G.data(), (int)fy);   // G[x][(f,y)]
            for(int x = 0; x < na; x++) for(int f = 0; f < nv; f++) for(int y = 0; y < na; y++){
                const double g = G[(size_t)x * fy + (size_t)f * na + y], D = in.ea[x] + in.ea[y] - in.ev[e] - in.ev[f];
                t2[((size_t)x * na + y) * nv + f] = g * R_of(D, s);
                vt[(size_t)f * a2 + (size_t)x * na + y] = g * (1.0 + q_of(D, s));
            }
            for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int f = 0; f < nv; f++)
                s2[((size_t)u * na + v) * nv + f] = 2.0 * t2[((size_t)u * na + v) * nv + f] - t2[((size_t)v * na + u) * nv + f];
            if(!adj){
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)a2, (int)a2, nv, 1.0, t2.data(), nv, vt.data(), (int)a2, 1.0, MT_t[th].data(), (int)a2);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)a2, (int)a2, nv, 1.0, s2.data(), nv, vt.data(), (int)a2, 1.0, MS_t[th].data(), (int)a2);
                continue;
            }
            // t2b = MTb vt^T, s2b = MSb vt^T;  vtb = t2^T MTb + s2^T MSb
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)a2, nv, (int)a2, 1.0, MTb, (int)a2, vt.data(), (int)a2, 0.0, t2b.data(), nv);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)a2, nv, (int)a2, 1.0, MSb, (int)a2, vt.data(), (int)a2, 0.0, s2b.data(), nv);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nv, (int)a2, (int)a2, 1.0, t2.data(), nv, MTb, (int)a2, 0.0, vtb.data(), (int)a2);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nv, (int)a2, (int)a2, 1.0, s2.data(), nv, MSb, (int)a2, 1.0, vtb.data(), (int)a2);
            // s2 = 2 t2 - t2(v,u)
            for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int f = 0; f < nv; f++)
                t2b[((size_t)u * na + v) * nv + f] += 2.0 * s2b[((size_t)u * na + v) * nv + f] - s2b[((size_t)v * na + u) * nv + f];
            double dce = 0.0;
            for(int x = 0; x < na; x++) for(int f = 0; f < nv; f++) for(int y = 0; y < na; y++){
                const size_t gi = (size_t)x * fy + (size_t)f * na + y, ti = ((size_t)x * na + y) * nv + f, vi = (size_t)f * a2 + (size_t)x * na + y;
                const double g = G[gi], D = in.ea[x] + in.ea[y] - in.ev[e] - in.ev[f];
                Gb[gi] = t2b[ti] * R_of(D, s) + vtb[vi] * (1.0 + q_of(D, s));
                const double Db = t2b[ti] * g * dR_of(D, s) + vtb[vi] * g * dq_of(D, s);
                ea_t[th][x] += Db; ea_t[th][y] += Db; dce -= Db; ev_t[th][f] -= Db;
            }
            ev_t[th][e] += dce;
            std::vector<double> & VAb = *VAbp;
#pragma omp critical(dsrg_aavv)
            {
                // G = VA[e] VA^T:  VAbar[e] += Gb VA,  VAbar += Gb^T VA[e]
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na, nQ, (int)fy, 1.0, Gb.data(), (int)fy, VA, nQ, 1.0, VAb.data() + (long)e * na * nQ, nQ);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)fy, nQ, na, 1.0, Gb.data(), (int)fy, VA + (long)e * na * nQ, nQ, 1.0, VAb.data(), nQ);
            }
        }
    }
    set_blas_par();
    if(adj){
        for(int t = 0; t < num_threads; t++){
            for(int u = 0; u < na; u++) (*eabp)[u] += ea_t[t][u];
            for(int e = 0; e < nv; e++) (*evbp)[e] += ev_t[t][e];
        }
    }
    else for(int t = 0; t < num_threads; t++) for(size_t k = 0; k < a4; k++){ if(MT) (*MT)[k] += MT_t[t][k]; if(MS) (*MS)[k] += MS_t[t][k]; }
}
