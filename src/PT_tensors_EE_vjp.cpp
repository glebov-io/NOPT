#include "pt_vjp.h"

#include <cmath>
#include <vector>
#include <omp.h>

#include "blas_link.h"
#include "common_vars.h"

extern int num_threads;

double pt_vjp_shift = 0.0;   // uniform shift of every PT denominator (the XMC energy-grid value E_fit); zero for the EE kernels

namespace{

inline double r_of(double dE, double eta){ return dE / (dE * dE + eta); }
inline double rp_of(double dE, double eta){ const double q = dE * dE + eta; return (eta - dE * dE) / (q * q); }

// e_IP and e_EA are copies of e_a on the plain EE path (set_par), so their adjoints belong to e_a; with
// separate outputs the caller asks for them in A.e_IP / A.e_EA instead.
inline void fold_ea(PTInputAdj & A, const std::vector<double> & ip, const std::vector<double> & ea, int n){
    for(int i = 0; i < n; i++){
        if(A.e_IP) A.e_IP[i] += ip[i]; else if(A.e_a) A.e_a[i] += ip[i];
        if(A.e_EA) A.e_EA[i] += ea[i]; else if(A.e_a) A.e_a[i] += ea[i];
    }
}

}

// ---- CCVV -----------------------------------------------------------------------------------
// forward (calc_EE_2_CCVV, RI branch), for a <= b, i <= j:
//   JK = VC[a] VC[b]^T,  J = JK[i][j], K = JK[j][i],  C = J - K
//   V  = 2 (J^2 + K^2) [/2 if a == b] [/2 if i == j] + 2 C^2
//   dE = e_c[j] + e_c[i] - e_v[a] - e_v[b],   RF_PS += V * r(dE)
int pt_vjp_EE_CCVV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_CCVV supports the RI branch only\n"); return 1; }
    const int nc = T.n_c, nv = T.n_v;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const double * VC = T.integrals->VC_RI_M;
    std::vector<int> live; for(int f = 0; f < K; f++) if(Sg[f].PS != 0.0) live.push_back(f);
    if(live.empty()) return 0;

    std::vector<std::vector<double>> ec_th(num_threads, std::vector<double>(nc, 0.0)), ev_th(num_threads, std::vector<double>(nv, 0.0));
    std::vector<std::vector<double>> sh_th(num_threads, std::vector<double>(K, 0.0));   // adjoints of the denominator shifts
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> JK((size_t)nc * nc), JKb((size_t)nc * nc), tmp((size_t)nc * aux);
        for(int a = th; a < nv; a += num_threads) for(int b = a; b < nv; b++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nc, nc, aux, 1.0,
                        VC + (size_t)a * nc * aux, aux, VC + (size_t)b * nc * aux, aux, 0.0, JK.data(), nc);
            std::fill(JKb.begin(), JKb.end(), 0.0);
            for(int i = 0; i < nc; i++) for(int j = i; j < nc; j++){
                const double J = JK[i * nc + j], K_ = JK[j * nc + i], C = J - K_;
                double w = 1.0;
                if(a == b) w *= 0.5;
                if(i == j) w *= 0.5;
                const double V = 2.0 * w * (J * J + K_ * K_) + 2.0 * C * C;
                const double dE0 = T.e_c[j] + T.e_c[i] - T.e_v[a] - T.e_v[b];
                double Vb = 0.0, dEbt = 0.0;
                for(int f : live){
                    const double dE = dE0 + shg[f];
                    Vb += Sg[f].PS * r_of(dE, eta);                           // dE/dV
                    const double dEb = Sg[f].PS * V * rp_of(dE, eta);         // dE/d(dE)
                    dEbt += dEb; sh_th[th][f] += dEb;
                }
                ec_th[th][j] += dEbt; ec_th[th][i] += dEbt; ev_th[th][a] -= dEbt; ev_th[th][b] -= dEbt;
                JKb[i * nc + j] += Vb * (4.0 * w * J + 4.0 * C);
                JKb[j * nc + i] += Vb * (4.0 * w * K_ - 4.0 * C);
            }
            // JK = VC[a] VC[b]^T  =>  VCbar[a] += JKb VC[b],  VCbar[b] += JKb^T VC[a]
            if(A.VC != nullptr){
#pragma omp critical(ccvv_vc)
                {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nc, aux, nc, 1.0,
                                JKb.data(), nc, VC + (size_t)b * nc * aux, aux, 1.0, A.VC + (size_t)a * nc * aux, aux);
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nc, aux, nc, 1.0,
                                JKb.data(), nc, VC + (size_t)a * nc * aux, aux, 1.0, A.VC + (size_t)b * nc * aux, aux);
                }
            }
        }
    }
    set_blas_par();
    for(int t = 0; t < num_threads; t++) for(int f = 0; f < K; f++) shadj[f] += sh_th[t][f];
    for(int t = 0; t < num_threads; t++){
        if(A.e_c) for(int i = 0; i < nc; i++) A.e_c[i] += ec_th[t][i];
        if(A.e_v) for(int a = 0; a < nv; a++) A.e_v[a] += ev_th[t][a];
    }
    return 0;
}
int pt_vjp_EE_CCVV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){ const double sh = pt_vjp_shift; return pt_vjp_EE_CCVV_grid(T, &S, &sh, 1, A, &A.shift_adj); }

// ---- CAVV -----------------------------------------------------------------------------------
// forward (calc_EE_2_CAVV, RI branch), for a <= b:
//   J[i,t] = sum_Q VC[a,i,Q] VA[b,t,Q],   K[i,t] = sum_Q VC[b,i,Q] VA[a,t,Q]
//   V2(i,t,v) = (J_it - K_it)(J_iv - K_iv) + J_it J_iv + [a != b] K_it K_iv
//   dE(i,t)   = e_c[i] + e_IP[t] - e_v[a] - e_v[b]
//   RF_PH[t,v] += V2(i,t,v) r(dE(i,t))                       (raw, before symmetrization)
int pt_vjp_EE_CAVV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_CAVV supports the RI branch only\n"); return 1; }
    std::vector<int> live; for(int f = 0; f < K; f++) if(Sg[f].PH != nullptr) live.push_back(f);
    if(live.empty()) return 0;
    const int nc = T.n_c, na = T.n_a, nv = T.n_v;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const double * VC = T.integrals->VC_RI_M, * VA = T.integrals->VA_RI_M;

    std::vector<std::vector<double>> ec_th(num_threads, std::vector<double>(nc, 0.0)), eip_th(num_threads, std::vector<double>(na, 0.0)), eea_th(num_threads, std::vector<double>(na, 0.0)),
                                     ev_th(num_threads, std::vector<double>(nv, 0.0));
    std::vector<std::vector<double>> sh_th(num_threads, std::vector<double>(K, 0.0));
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> J((size_t)nc * na), K_((size_t)nc * na), Jb((size_t)nc * na), Kb((size_t)nc * na), W((size_t)na);
        for(int a = th; a < nv; a += num_threads) for(int b = a; b < nv; b++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nc, na, aux, 1.0,
                        VC + (size_t)a * nc * aux, aux, VA + (size_t)b * na * aux, aux, 0.0, J.data(), na);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nc, na, aux, 1.0,
                        VC + (size_t)b * nc * aux, aux, VA + (size_t)a * na * aux, aux, 0.0, K_.data(), na);
            std::fill(Jb.begin(), Jb.end(), 0.0); std::fill(Kb.begin(), Kb.end(), 0.0);
            const double kk = (a != b) ? 2.0 : 1.0;                      // 1 + [a != b]
            for(int i = 0; i < nc; i++) for(int t = 0; t < na; t++){
                const double dE0 = T.e_c[i] + T.e_IP[t] - T.e_v[a] - T.e_v[b];
                const double Jt = J[i * na + t], Kt = K_[i * na + t];
                // V2(i,t,v) for every v, then the grid: W[v] = sum_f PH_f[t,v] r_f (the seed on V2), and each grid point's shift adjoint
                std::fill(W.begin(), W.end(), 0.0);
                double dEbt = 0.0;
                for(int f : live){
                    const double dE = dE0 + shg[f];
                    const double r = r_of(dE, eta), rp = rp_of(dE, eta);
                    const double * PH = Sg[f].PH + (size_t)t * na;
                    double s2 = 0.0;
                    for(int v = 0; v < na; v++){
                        const double Jv = J[i * na + v], Kv = K_[i * na + v];
                        const double V2 = (Jt - Kt) * (Jv - Kv) + Jt * Jv + ((a != b) ? Kt * Kv : 0.0);
                        s2 += PH[v] * V2;
                        W[v] += PH[v] * r;
                    }
                    const double dEb = s2 * rp;
                    dEbt += dEb; sh_th[th][f] += dEb;
                }
                for(int v = 0; v < na; v++){
                    const double Sv = W[v];
                    const double Jv = J[i * na + v], Kv = K_[i * na + v];
                    // slot t:  dV2/dJ_t = 2 J_v - K_v,  dV2/dK_t = -J_v + kk K_v
                    Jb[i * na + t] += Sv * (2.0 * Jv - Kv);
                    Kb[i * na + t] += Sv * (-Jv + kk * Kv);
                    // slot v:  dV2/dJ_v = 2 J_t - K_t,  dV2/dK_v = -J_t + kk K_t
                    Jb[i * na + v] += Sv * (2.0 * Jt - Kt);
                    Kb[i * na + v] += Sv * (-Jt + kk * Kt);
                }
                ec_th[th][i] += dEbt; eip_th[th][t] += dEbt; ev_th[th][a] -= dEbt; ev_th[th][b] -= dEbt;
            }
#pragma omp critical(cavv_acc)
            {
                if(A.VC){
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nc, aux, na, 1.0, Jb.data(), na, VA + (size_t)b * na * aux, aux, 1.0, A.VC + (size_t)a * nc * aux, aux);
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nc, aux, na, 1.0, Kb.data(), na, VA + (size_t)a * na * aux, aux, 1.0, A.VC + (size_t)b * nc * aux, aux);
                }
                if(A.VA){
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, aux, nc, 1.0, Jb.data(), na, VC + (size_t)a * nc * aux, aux, 1.0, A.VA + (size_t)b * na * aux, aux);
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, aux, nc, 1.0, Kb.data(), na, VC + (size_t)b * nc * aux, aux, 1.0, A.VA + (size_t)a * na * aux, aux);
                }
            }
        }
    }
    set_blas_par();
    for(int t = 0; t < num_threads; t++) for(int f = 0; f < K; f++) shadj[f] += sh_th[t][f];
    for(int t = 0; t < num_threads; t++){
        if(A.e_c) for(int i = 0; i < nc; i++) A.e_c[i] += ec_th[t][i];
        fold_ea(A, eip_th[t], eea_th[t], na);
        if(A.e_v) for(int a = 0; a < nv; a++) A.e_v[a] += ev_th[t][a];
    }
    return 0;
}
int pt_vjp_EE_CAVV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){ const double sh = pt_vjp_shift; return pt_vjp_EE_CAVV_grid(T, &S, &sh, 1, A, &A.shift_adj); }

// ---- AAVV -----------------------------------------------------------------------------------
// forward (calc_EE_2_AAVV, RI branch), for a <= b, Ja = VA[a] VA[b]^T (n_a x n_a):
//   JK = (Ja_tu - Ja_ut)(Ja_vw - Ja_wv),  AB = Ja_tu Ja_vw + [a != b] Ja_ut Ja_wv
//   dE = e_IP[t] + e_IP[u] - e_v[a] - e_v[b]
//   RF_PV_JK[((u,t),v),w] += JK r(dE),   RF_PV_AB[((t,u),v),w] += AB r(dE)       (raw)
int pt_vjp_EE_AAVV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_AAVV supports the RI branch only\n"); return 1; }
    const int na = T.n_a, nv = T.n_v;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const double * VA = T.integrals->VA_RI_M;
    std::vector<int> live; for(int f = 0; f < K; f++) if(Sg[f].PV_JK != nullptr || Sg[f].PV_AB != nullptr) live.push_back(f);
    if(live.empty()) return 0;
    auto id4 = [&](int a, int b, int c, int d){ return ((size_t)(a * na + b) * na + c) * na + d; };
    const size_t n2 = (size_t)na * na;

    std::vector<std::vector<double>> eip_th(num_threads, std::vector<double>(na, 0.0)), eea_th(num_threads, std::vector<double>(na, 0.0)), ev_th(num_threads, std::vector<double>(nv, 0.0));
    std::vector<std::vector<double>> sh_th(num_threads, std::vector<double>(K, 0.0));
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> Ja(n2), Jb(n2), WJ(n2), WA(n2);
        for(int a = th; a < nv; a += num_threads) for(int b = a; b < nv; b++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na, na, aux, 1.0,
                        VA + (size_t)a * na * aux, aux, VA + (size_t)b * na * aux, aux, 0.0, Ja.data(), na);
            std::fill(Jb.begin(), Jb.end(), 0.0);
            const double kk = (a != b) ? 1.0 : 0.0;
            for(int t = 0; t < na; t++) for(int u = 0; u < na; u++){
                const double dE0 = T.e_IP[t] + T.e_IP[u] - T.e_v[a] - T.e_v[b];
                const double Dtu = Ja[t * na + u] - Ja[u * na + t];
                // the grid folded into effective seeds WJ[v,w] = sum_f PV_JK_f[u,t,v,w] r_f, WA[v,w] = sum_f PV_AB_f[t,u,v,w] r_f
                std::fill(WJ.begin(), WJ.end(), 0.0); std::fill(WA.begin(), WA.end(), 0.0);
                double dEbt = 0.0;
                for(int f : live){
                    const double dE = dE0 + shg[f];
                    const double r = r_of(dE, eta), rp = rp_of(dE, eta);
                    double s = 0.0;
                    if(Sg[f].PV_JK){ const double * y = Sg[f].PV_JK + id4(u, t, 0, 0);
                        for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){ const double yb = y[v * na + w]; s += yb * Dtu * (Ja[v * na + w] - Ja[w * na + v]); WJ[v * na + w] += yb * r; } }
                    if(Sg[f].PV_AB){ const double * y = Sg[f].PV_AB + id4(t, u, 0, 0);
                        for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){ const double yb = y[v * na + w];
                            s += yb * (Ja[t * na + u] * Ja[v * na + w] + kk * Ja[u * na + t] * Ja[w * na + v]); WA[v * na + w] += yb * r; } }
                    const double dEb = s * rp;
                    dEbt += dEb; sh_th[th][f] += dEb;
                }
                for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
                    const double Dvw = Ja[v * na + w] - Ja[w * na + v];
                    const double sj = WJ[v * na + w], sa = WA[v * na + w];
                    Jb[t * na + u] += sj * Dvw;  Jb[u * na + t] -= sj * Dvw;
                    Jb[v * na + w] += sj * Dtu;  Jb[w * na + v] -= sj * Dtu;
                    Jb[t * na + u] += sa * Ja[v * na + w];  Jb[v * na + w] += sa * Ja[t * na + u];
                    if(a != b){ Jb[u * na + t] += sa * Ja[w * na + v];  Jb[w * na + v] += sa * Ja[u * na + t]; }
                }
                eip_th[th][t] += dEbt; eip_th[th][u] += dEbt; ev_th[th][a] -= dEbt; ev_th[th][b] -= dEbt;
            }
            if(A.VA){
#pragma omp critical(aavv_acc)
                {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na, aux, na, 1.0, Jb.data(), na, VA + (size_t)b * na * aux, aux, 1.0, A.VA + (size_t)a * na * aux, aux);
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, aux, na, 1.0, Jb.data(), na, VA + (size_t)a * na * aux, aux, 1.0, A.VA + (size_t)b * na * aux, aux);
                }
            }
        }
    }
    set_blas_par();
    for(int t = 0; t < num_threads; t++) for(int f = 0; f < K; f++) shadj[f] += sh_th[t][f];
    for(int t = 0; t < num_threads; t++){
        fold_ea(A, eip_th[t], eea_th[t], na);
        if(A.e_v) for(int a = 0; a < nv; a++) A.e_v[a] += ev_th[t][a];
    }
    return 0;
}
int pt_vjp_EE_AAVV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){ const double sh = pt_vjp_shift; return pt_vjp_EE_AAVV_grid(T, &S, &sh, 1, A, &A.shift_adj); }

// ---- CCAV -----------------------------------------------------------------------------------
// forward (calc_EE_2_CCAV, RI branch), for i <= j:
//   J[a,t] = VC[a,i] . CA[j,t],   K[a,t] = VC[a,j] . CA[i,t]            (dot products over the auxiliary index)
//   V2(a,t,v) = (J_at - K_at)(J_av - K_av) + J_at J_av + [i != j] K_at K_av
//   dE(a,t)   = e_c[i] + e_c[j] - e_EA[t] - e_v[a]
//   P[v,t]   -= V2(a,t,v) r(dE(a,t));   then RF_PS -= 2 sum_t P[t,t];  RF_PH[w,t] += P[w,t]    (raw)
// so the seed on P is  Pbar[v,t] = PHbar[v,t] - 2 PSbar delta_vt.
int pt_vjp_EE_CCAV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_CCAV supports the RI branch only\n"); return 1; }
    const int nc = T.n_c, na = T.n_a, nv = T.n_v;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const double * VC = T.integrals->VC_RI_M, * CA = T.integrals->CA_RI_M;
    const size_t n2 = (size_t)na * na;
    // seeds on P per grid point: Pbar[v,t] = PHbar[v,t] - 2 PSbar delta_vt
    std::vector<int> live;
    std::vector<double> Pb((size_t)K * n2, 0.0);
    for(int f = 0; f < K; f++){
        bool any = false;
        for(int v = 0; v < na; v++) for(int t = 0; t < na; t++){
            const double x = (Sg[f].PH ? Sg[f].PH[v * na + t] : 0.0) - (v == t ? 2.0 * Sg[f].PS : 0.0);
            Pb[(size_t)f * n2 + v * na + t] = x; any = any || x != 0.0;
        }
        if(any) live.push_back(f);
    }
    if(live.empty()) return 0;

    std::vector<std::vector<double>> ec_th(num_threads, std::vector<double>(nc, 0.0)), eip_th(num_threads, std::vector<double>(na, 0.0)), eea_th(num_threads, std::vector<double>(na, 0.0)),
                                     ev_th(num_threads, std::vector<double>(nv, 0.0));
    std::vector<std::vector<double>> sh_th(num_threads, std::vector<double>(K, 0.0));
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> J((size_t)nv * na), K_((size_t)nv * na), Jb((size_t)nv * na), Kb((size_t)nv * na), W(na);
        for(int i = th; i < nc; i += num_threads) for(int j = i; j < nc; j++){
            for(int a = 0; a < nv; a++) for(int t = 0; t < na; t++){
                J[a * na + t] = cblas_ddot(aux, VC + ((size_t)a * nc + i) * aux, 1, CA + ((size_t)j * na + t) * aux, 1);
                K_[a * na + t] = cblas_ddot(aux, VC + ((size_t)a * nc + j) * aux, 1, CA + ((size_t)i * na + t) * aux, 1);
            }
            std::fill(Jb.begin(), Jb.end(), 0.0); std::fill(Kb.begin(), Kb.end(), 0.0);
            const double kk = (i != j) ? 2.0 : 1.0;
            for(int a = 0; a < nv; a++) for(int t = 0; t < na; t++){
                const double dE0 = T.e_c[i] + T.e_c[j] - T.e_EA[t] - T.e_v[a];
                const double Jt = J[a * na + t], Kt = K_[a * na + t];
                std::fill(W.begin(), W.end(), 0.0);
                double dEbt = 0.0;
                for(int f : live){
                    const double dE = dE0 + shg[f];
                    const double r = r_of(dE, eta), rp = rp_of(dE, eta);
                    double s = 0.0;
                    for(int v = 0; v < na; v++){
                        const double Pv = -Pb[(size_t)f * n2 + v * na + t];        // dE_total/d(V2 r) with the minus sign of the forward
                        const double Jv = J[a * na + v], Kv = K_[a * na + v];
                        const double V2 = (Jt - Kt) * (Jv - Kv) + Jt * Jv + ((i != j) ? Kt * Kv : 0.0);
                        s += Pv * V2; W[v] += Pv * r;
                    }
                    const double dEb = s * rp;
                    dEbt += dEb; sh_th[th][f] += dEb;
                }
                for(int v = 0; v < na; v++){
                    const double Sv = W[v];
                    const double Jv = J[a * na + v], Kv = K_[a * na + v];
                    Jb[a * na + t] += Sv * (2.0 * Jv - Kv);   Kb[a * na + t] += Sv * (-Jv + kk * Kv);
                    Jb[a * na + v] += Sv * (2.0 * Jt - Kt);   Kb[a * na + v] += Sv * (-Jt + kk * Kt);
                }
                ec_th[th][i] += dEbt; ec_th[th][j] += dEbt; eea_th[th][t] -= dEbt; ev_th[th][a] -= dEbt;
            }
#pragma omp critical(ccav_acc)
            {
                if(A.VC){   // VCbar[:,i,:] += Jb CA[j] ;  VCbar[:,j,:] += Kb CA[i]
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nv, aux, na, 1.0, Jb.data(), na, CA + (size_t)j * na * aux, aux, 1.0, A.VC + (size_t)i * aux, (size_t)nc * aux);
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nv, aux, na, 1.0, Kb.data(), na, CA + (size_t)i * na * aux, aux, 1.0, A.VC + (size_t)j * aux, (size_t)nc * aux);
                }
                if(A.CA){   // CAbar[j] += Jb^T VC[:,i,:] ;  CAbar[i] += Kb^T VC[:,j,:]
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, aux, nv, 1.0, Jb.data(), na, VC + (size_t)i * aux, (size_t)nc * aux, 1.0, A.CA + (size_t)j * na * aux, aux);
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, aux, nv, 1.0, Kb.data(), na, VC + (size_t)j * aux, (size_t)nc * aux, 1.0, A.CA + (size_t)i * na * aux, aux);
                }
            }
        }
    }
    set_blas_par();
    for(int t = 0; t < num_threads; t++) for(int f = 0; f < K; f++) shadj[f] += sh_th[t][f];
    for(int t = 0; t < num_threads; t++){
        if(A.e_c) for(int i = 0; i < nc; i++) A.e_c[i] += ec_th[t][i];
        fold_ea(A, eip_th[t], eea_th[t], na);
        if(A.e_v) for(int a = 0; a < nv; a++) A.e_v[a] += ev_th[t][a];
    }
    return 0;
}
int pt_vjp_EE_CCAV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){ const double sh = pt_vjp_shift; return pt_vjp_EE_CCAV_grid(T, &S, &sh, 1, A, &A.shift_adj); }

// ---- CCAA -----------------------------------------------------------------------------------
// forward (calc_EE_2_CCAA, RI branch), for i <= j, Ja = CA[i] CA[j]^T (n_a x n_a):
//   V2 = (Ja_tu - Ja_ut)(Ja_vw - Ja_wv),  AB = Ja_tu Ja_vw + [i != j] Ja_ut Ja_wv,  dE = e_c[i] + e_c[j] - e_EA[v] - e_EA[w]
//   PJK[u,t,v,w] += V2 r(dE),  PAB[t,u,v,w] += AB r(dE)           (accumulated over i <= j)
// and afterwards linear maps of the accumulated PJK / PAB into RF_PH and RF_PS (see the end of the kernel):
//   RF_PV_JK += PJK, RF_PV_AB += PAB, RF_PH += 1/4 (PJK terms) - PAB[t,w,t,u], RF_PS += 1/2 (PJK terms) + PAB[t,u,t,u].
int pt_vjp_EE_CCAA(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_CCAA supports the RI branch only\n"); return 1; }
    const int nc = T.n_c, na = T.n_a;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const double * CA = T.integrals->CA_RI_M;
    auto id4 = [&](int a, int b, int c, int d){ return ((size_t)(a * na + b) * na + c) * na + d; };
    const size_t n4 = (size_t)na * na * na * na;

    // effective seeds on the accumulated PJK and PAB
    std::vector<double> yJ(n4, 0.0), yA(n4, 0.0);
    if(S.PV_AB) for(size_t k = 0; k < n4; k++) yA[k] += S.PV_AB[k];
    if(S.PV_JK) for(size_t k = 0; k < n4; k++) yJ[k] += S.PV_JK[k];
    if(S.PH){
        for(int a = 0; a < na; a++) for(int b = 0; b < na; b++) for(int c = 0; c < na; c++){
            const double hb = S.PH[a * na + b];
            (void)c;
            // RF_PH[v,u] += PJK[(t*n+v)*n+t)*n+u]/4  (v=a,u=b, t=c)
            yJ[id4(c, a, c, b)] += 0.25 * hb;
            // RF_PH[v,t] -= PJK[(u*n+v)*n+t)*n+u]/4  (v=a,t=b,u=c)
            yJ[id4(c, a, b, c)] -= 0.25 * hb;
            // RF_PH[w,u] -= PJK[(w*n+v)*n+v)*n+u]/4  (w=a,u=b,v=c)
            yJ[id4(a, c, c, b)] -= 0.25 * hb;
            // RF_PH[w,t] += PJK[(w*n+u)*n+t)*n+u]/4  (w=a,t=b,u=c)
            yJ[id4(a, c, b, c)] += 0.25 * hb;
            // RF_PH[w,u] -= PAB[(t*n+w)*n+t)*n+u]    (w=a,u=b,t=c)
            yA[id4(c, a, c, b)] -= hb;
        }
    }
    for(int u = 0; u < na; u++) for(int v = 0; v < na; v++){
        yJ[id4(u, v, v, u)] += 0.5 * S.PS;                          // RF_PS += PJK[u,v,v,u]/2
        yJ[id4(u, v, u, v)] -= 0.5 * S.PS;                          // RF_PS -= PJK[w,v,w,v]/2   (w -> u)
        yA[id4(u, v, u, v)] += S.PS;                                // RF_PS += PAB[t,u,t,u]
    }

    std::vector<std::vector<double>> ec_th(num_threads, std::vector<double>(nc, 0.0)), eip_th(num_threads, std::vector<double>(na, 0.0)), eea_th(num_threads, std::vector<double>(na, 0.0));
    std::vector<double> sh_th(num_threads, 0.0);   // adjoint of the uniform denominator shift
    set_blas_non_par();
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> Ja((size_t)na * na), Jb((size_t)na * na);
        for(int i = th; i < nc; i += num_threads) for(int j = i; j < nc; j++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na, na, aux, 1.0,
                        CA + (size_t)i * na * aux, aux, CA + (size_t)j * na * aux, aux, 0.0, Ja.data(), na);
            std::fill(Jb.begin(), Jb.end(), 0.0);
            const double kk = (i != j) ? 1.0 : 0.0;
            double dEb_c = 0.0;
            for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
                const double dE = -T.e_EA[v] - T.e_EA[w] + T.e_c[i] + T.e_c[j] + pt_vjp_shift;
                const double r = r_of(dE, eta), rp = rp_of(dE, eta);
                const double Dvw = Ja[v * na + w] - Ja[w * na + v];
                double dEb = 0.0;
                for(int t = 0; t < na; t++) for(int u = 0; u < na; u++){
                    const double Dtu = Ja[t * na + u] - Ja[u * na + t];
                    const double yj = yJ[id4(u, t, v, w)], ya = yA[id4(t, u, v, w)];
                    const double AB = Ja[t * na + u] * Ja[v * na + w] + kk * Ja[u * na + t] * Ja[w * na + v];
                    dEb += yj * Dtu * Dvw * rp + ya * AB * rp;
                    const double sj = yj * r, sa = ya * r;
                    Jb[t * na + u] += sj * Dvw;  Jb[u * na + t] -= sj * Dvw;
                    Jb[v * na + w] += sj * Dtu;  Jb[w * na + v] -= sj * Dtu;
                    Jb[t * na + u] += sa * Ja[v * na + w];  Jb[v * na + w] += sa * Ja[t * na + u];
                    if(i != j){ Jb[u * na + t] += sa * Ja[w * na + v];  Jb[w * na + v] += sa * Ja[u * na + t]; }
                }
                dEb_c += dEb; sh_th[th] += dEb;
                eea_th[th][v] -= dEb; eea_th[th][w] -= dEb;
            }
            ec_th[th][i] += dEb_c; ec_th[th][j] += dEb_c;
            if(A.CA){
#pragma omp critical(ccaa_acc)
                {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na, aux, na, 1.0, Jb.data(), na, CA + (size_t)j * na * aux, aux, 1.0, A.CA + (size_t)i * na * aux, aux);
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, aux, na, 1.0, Jb.data(), na, CA + (size_t)i * na * aux, aux, 1.0, A.CA + (size_t)j * na * aux, aux);
                }
            }
        }
    }
    set_blas_par();
    for(int t = 0; t < num_threads; t++) A.shift_adj += sh_th[t];
    for(int t = 0; t < num_threads; t++){
        if(A.e_c) for(int i = 0; i < nc; i++) A.e_c[i] += ec_th[t][i];
        fold_ea(A, eip_th[t], eea_th[t], na);
    }
    return 0;
}

// ---- CV -------------------------------------------------------------------------------------
// forward (calc_EE_2_CV, RI branch). Intermediates
//   VCAA[a,i,x,y] = sum_Q VC[a,i,Q] AA[x,y,Q]  =: g(x,y),        CAVA[i,x,a,y] = sum_Q CA[i,x,Q] VA[a,y,Q]  =: c(x,y)
// for every (i,a), with rr(t,u) = r(dE), dE = e_c[i] + e_IP[t] - e_v[a] - e_EA[u] (t != u), e_c[i] - e_v[a] (t == u),
// r0 = r(e_c[i] - e_v[a]), H = H_CV[i,a]:
//   MM[t,u,v,w] += rr (g(u,t) - c(u,t)) (g(w,v) - c(v,w))     MJ += rr (g(u,t) - c(u,t)) g(w,v)
//   JM[t,u,v,w] += rr  g(u,t)           (g(w,v) - c(v,w))     JJ += rr  g(u,t) g(w,v)       AB += rr c(u,t) c(v,w)
//   MH[t,u] += rr (g(u,t) - c(u,t)) H   HM[t,u] += r0 (g(u,t) - c(u,t)) H
//   JH[t,u] += rr  g(u,t) H             HJ[t,u] += r0  g(u,t) H           RF_H += r0 H^2
// and, at the end of the kernel,
//   PV_JK[u,t,v,w] += MM+JJ at [t,v,u,w] - [u,v,t,w] - [t,w,u,v] + [u,w,t,v]
//   PV_AB[t,u,v,w] += (JM+MJ)[t,v,u,w] + (JM+MJ)[u,w,t,v] - AB[t,w,u,v] - AB[u,v,t,w]
//   PH[t,w] += HM[w,t] + MH[t,w] + HJ[w,t] + JH[t,w] + sum_u (MM+JJ+AB)[t,u,u,w];   PS += 2 RF_H.
int pt_vjp_EE_CV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_CV supports the RI branch only\n"); return 1; }
    const int nc = T.n_c, na = T.n_a, nv = T.n_v;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const size_t n2 = (size_t)na * na, n4 = n2 * n2;
    auto id4 = [&](int a, int b, int c, int d){ return ((size_t)(a * na + b) * na + c) * na + d; };

    // intermediates, as in the forward
    std::vector<double> VCAA((size_t)nc * nv * n2), CAVA((size_t)na * nc * na * nv);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nc * nv, na * na, aux, 1.0, T.integrals->VC_RI_M, aux, T.integrals->AA_RI_M, aux, 0.0, VCAA.data(), na * na);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na * nc, na * nv, aux, 1.0, T.integrals->CA_RI_M, aux, T.integrals->VA_RI_M, aux, 0.0, CAVA.data(), na * nv);

    // seeds on the intermediate accumulators (inverse of the final assembly)
    std::vector<double> yMM(n4, 0.0), yJJ(n4, 0.0), yJM(n4, 0.0), yMJ(n4, 0.0), yAB(n4, 0.0);
    std::vector<double> yHM(n2, 0.0), yMH(n2, 0.0), yHJ(n2, 0.0), yJH(n2, 0.0);
    if(S.PV_JK) for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
        const double y = S.PV_JK[id4(u, t, v, w)];
        yMM[id4(t, v, u, w)] += y; yMM[id4(u, v, t, w)] -= y; yMM[id4(t, w, u, v)] -= y; yMM[id4(u, w, t, v)] += y;
        yJJ[id4(t, v, u, w)] += y; yJJ[id4(u, v, t, w)] -= y; yJJ[id4(t, w, u, v)] -= y; yJJ[id4(u, w, t, v)] += y;
    }
    if(S.PV_AB) for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
        const double z = S.PV_AB[id4(t, u, v, w)];
        yJM[id4(t, v, u, w)] += z; yMJ[id4(t, v, u, w)] += z; yJM[id4(u, w, t, v)] += z; yMJ[id4(u, w, t, v)] += z;
        yAB[id4(t, w, u, v)] -= z; yAB[id4(u, v, t, w)] -= z;
    }
    if(S.PH) for(int t = 0; t < na; t++) for(int w = 0; w < na; w++){
        const double p = S.PH[t * na + w];
        yHM[w * na + t] += p; yMH[t * na + w] += p; yHJ[w * na + t] += p; yJH[t * na + w] += p;
        for(int u = 0; u < na; u++){ yMM[id4(t, u, u, w)] += p; yJJ[id4(t, u, u, w)] += p; yAB[id4(t, u, u, w)] += p; }
    }
    const double yH = 2.0 * S.PS;

    std::vector<double> VCAAb((size_t)nc * nv * n2, 0.0), CAVAb((size_t)na * nc * na * nv, 0.0);
    std::vector<std::vector<double>> ec_th(num_threads, std::vector<double>(nc, 0.0)), eip_th(num_threads, std::vector<double>(na, 0.0)), eea_th(num_threads, std::vector<double>(na, 0.0)),
                                     ev_th(num_threads, std::vector<double>(nv, 0.0));
    const bool haveH = (A.H_CV != nullptr);
    std::vector<double> sh_th(num_threads, 0.0);   // adjoint of the uniform denominator shift
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
        std::vector<double> g(n2), c(n2), gb(n2), cb(n2);
#pragma omp for collapse(2) schedule(dynamic)
        for(int i = 0; i < nc; i++) for(int a = 0; a < nv; a++){
            for(int x = 0; x < na; x++) for(int y = 0; y < na; y++){
                g[x * na + y] = VCAA[((size_t)(a * nc + i) * na + x) * na + y];
                c[x * na + y] = CAVA[(((size_t)i * na + x) * nv + a) * na + y];
            }
            std::fill(gb.begin(), gb.end(), 0.0); std::fill(cb.begin(), cb.end(), 0.0);
            const double H = T.H_CV[i * nv + a];
            const double dE0 = T.e_c[i] - T.e_v[a] + pt_vjp_shift;
            const double r0 = r_of(dE0, eta), rp0 = rp_of(dE0, eta);
            double Hb = 0.0, dE0b = 0.0;
            for(int t = 0; t < na; t++) for(int u = 0; u < na; u++){
                double dE = (t == u) ? dE0 : T.e_c[i] + T.e_IP[t] - T.e_v[a] - T.e_EA[u] + pt_vjp_shift;
                const double rr = r_of(dE, eta), rrp = rp_of(dE, eta);
                const double g1 = g[u * na + t], c1 = c[u * na + t], F1 = g1 - c1;
                double rrb = 0.0, F1b = 0.0, G1b = 0.0, C1b = 0.0;
                for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
                    const double g2 = g[w * na + v], c2 = c[v * na + w], F2 = g2 - c2;
                    const double a1 = yMM[id4(t, u, v, w)], a2 = yMJ[id4(t, u, v, w)], a3 = yJM[id4(t, u, v, w)],
                                 a4 = yJJ[id4(t, u, v, w)], a5 = yAB[id4(t, u, v, w)];
                    rrb += a1 * F1 * F2 + a2 * F1 * g2 + a3 * g1 * F2 + a4 * g1 * g2 + a5 * c1 * c2;
                    F1b += rr * (a1 * F2 + a2 * g2);
                    G1b += rr * (a3 * F2 + a4 * g2);
                    C1b += rr * a5 * c2;
                    const double F2b = rr * (a1 * F1 + a3 * g1);
                    const double G2b = rr * (a2 * F1 + a4 * g1);
                    const double C2b = rr * a5 * c1;
                    gb[w * na + v] += F2b + G2b;  cb[v * na + w] += -F2b + C2b;
                }
                // two-index terms
                const double hMH = yMH[t * na + u], hHM = yHM[t * na + u], hJH = yJH[t * na + u], hHJ = yHJ[t * na + u];
                rrb += hMH * F1 * H + hJH * g1 * H;
                F1b += H * (hMH * rr + hHM * r0);
                G1b += H * (hJH * rr + hHJ * r0);
                Hb  += hMH * F1 * rr + hHM * F1 * r0 + hJH * g1 * rr + hHJ * g1 * r0;
                dE0b += (hHM * F1 + hHJ * g1) * H * rp0;
                const double dEb = rrb * rrp;
                gb[u * na + t] += F1b + G1b;  cb[u * na + t] += -F1b + C1b;
                ec_th[th][i] += dEb; ev_th[th][a] -= dEb; sh_th[th] += dEb;
                if(t != u){ eip_th[th][t] += dEb; eea_th[th][u] -= dEb; }
            }
            // RF_H += r0 H^2
            Hb += yH * 2.0 * H * r0;
            dE0b += yH * H * H * rp0;
            ec_th[th][i] += dE0b; ev_th[th][a] -= dE0b; sh_th[th] += dE0b;
            if(haveH){
#pragma omp atomic
                A.H_CV[i * nv + a] += Hb;
            }
            for(int x = 0; x < na; x++) for(int y = 0; y < na; y++){
                VCAAb[((size_t)(a * nc + i) * na + x) * na + y] += gb[x * na + y];
                CAVAb[(((size_t)i * na + x) * nv + a) * na + y] += cb[x * na + y];
            }
        }
    }
    // back to the RI tensors: VCAA = VC AA^T, CAVA = CA VA^T
    if(A.VC) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nc * nv, aux, na * na, 1.0, VCAAb.data(), na * na, T.integrals->AA_RI_M, aux, 1.0, A.VC, aux);
    if(A.AA) cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na * na, aux, nc * nv, 1.0, VCAAb.data(), na * na, T.integrals->VC_RI_M, aux, 1.0, A.AA, aux);
    if(A.CA) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na * nc, aux, na * nv, 1.0, CAVAb.data(), na * nv, T.integrals->VA_RI_M, aux, 1.0, A.CA, aux);
    if(A.VA) cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na * nv, aux, na * nc, 1.0, CAVAb.data(), na * nv, T.integrals->CA_RI_M, aux, 1.0, A.VA, aux);
    for(int t = 0; t < num_threads; t++){ A.shift_adj += sh_th[t];
        if(A.e_c) for(int i = 0; i < nc; i++) A.e_c[i] += ec_th[t][i];
        fold_ea(A, eip_th[t], eea_th[t], na);
        if(A.e_v) for(int a = 0; a < nv; a++) A.e_v[a] += ev_th[t][a];
    }
    return 0;
}

// ---- AV -------------------------------------------------------------------------------------
// forward (calc_EE_2_AV, RI branch). Intermediate g_a(p;q,r) = VAAA[a,p,q,r] = sum_Q VA[a,p,Q] AA[q,r,Q].
// Per (a,t,u,v): rr = r(dE), dE = -e_v[a] + e_IP[t] + e_IP[u] - e_EA[v]; if v == t: -e_v[a] + e_IP[u]; if v == u: -e_v[a] + e_IP[t].
// Per (a,t): rH = r(-e_v[a] + e_IP[t]); H = H_AV[t,a].  With F1 = g(t;v,u) - g(u;v,t), G1 = g(u;v,t),
// F2 = g(x;w,y) - g(y;w,x), G2 = g(x;w,y):
//   P3_JK[t,u,v,y,x,w] += rr F1 F2      JJ += rr G1 G2     MJ += rr F1 G2     JM += rr G1 F2
//   PH[t,w] += rH H_AV[t,a] H_AV[w,a]
//   HM[t,y,x,w] += rH H_AV[t,a] F2      HJ[t,y,x,w] += rH H_AV[t,a] G2
//   MH[t,u,v,w] += rr F1 H_AV[w,a]      JH[t,u,v,w] += rr G1 H_AV[w,a]
// and at the end PV_JK, PV_AB, P3_AB are linear maps of HM, MH, HJ, JH, JJ, MJ, JM (see the kernel).
int pt_vjp_EE_AV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_AV supports the RI branch only\n"); return 1; }
    const int na = T.n_a, nv = T.n_v;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const size_t n2 = (size_t)na * na, n3 = n2 * na, n4 = n2 * n2, n6 = n3 * n3;
    auto id4 = [&](int a, int b, int c, int d){ return ((size_t)(a * na + b) * na + c) * na + d; };
    auto id6 = [&](int a, int b, int c, int d, int e, int f){ return ((((size_t)(a * na + b) * na + c) * na + d) * na + e) * na + f; };

    std::vector<double> VAAA((size_t)nv * na * n2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nv * na, na * na, aux, 1.0, T.integrals->VA_RI_M, aux, T.integrals->AA_RI_M, aux, 0.0, VAAA.data(), na * na);

    // seeds on the accumulators
    std::vector<double> yJK(n6, 0.0), yJJ(n6, 0.0), yMJ(n6, 0.0), yJM(n6, 0.0);
    std::vector<double> yHM(n4, 0.0), yMH(n4, 0.0), yHJ(n4, 0.0), yJH(n4, 0.0);
    if(S.P3_JK) for(size_t k = 0; k < n6; k++) yJK[k] = S.P3_JK[k];
    if(S.PV_JK) for(int w = 0; w < na; w++) for(int t = 0; t < na; t++) for(int v = 0; v < na; v++) for(int u = 0; u < na; u++){
        const double z = S.PV_JK[id4(w, t, v, u)];
        yHM[id4(t, u, v, w)] += z;  yHM[id4(w, u, v, t)] -= z;  yMH[id4(t, w, v, u)] -= z;  yMH[id4(t, w, u, v)] += z;
    }
    if(S.PV_AB) for(int w = 0; w < na; w++) for(int t = 0; t < na; t++) for(int v = 0; v < na; v++) for(int u = 0; u < na; u++){
        const double z = S.PV_AB[id4(t, u, v, w)];
        yHJ[id4(t, w, v, u)] += z;  yJH[id4(u, t, w, v)] += z;  yHJ[id4(u, v, w, t)] += z;  yJH[id4(t, u, v, w)] += z;
        for(int x = 0; x < na; x++){ yJJ[id6(t, u, x, v, w, x)] += z;  yJJ[id6(u, t, x, w, v, x)] += z; }
    }
    if(S.P3_AB) for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int x = 0; x < na; x++)
                for(int v = 0; v < na; v++) for(int w = 0; w < na; w++) for(int y = 0; y < na; y++){
        const double z = S.P3_AB[id6(t, u, x, v, w, y)];
        yMJ[id6(t, u, v, y, w, x)] -= z;  yMJ[id6(t, u, w, y, v, x)] += z;
        yJM[id6(x, t, y, w, v, u)] += z;  yJM[id6(x, u, y, w, v, t)] -= z;
        yJJ[id6(t, x, v, w, y, u)] += z;  yJJ[id6(t, x, w, v, y, u)] -= z;
        yJJ[id6(u, x, v, w, y, t)] -= z;  yJJ[id6(u, x, w, v, y, t)] += z;
    }

    std::vector<double> VAAAb((size_t)nv * na * n2, 0.0);
    std::vector<std::vector<double>> eip_th(num_threads, std::vector<double>(na, 0.0)), eea_th(num_threads, std::vector<double>(na, 0.0)), ev_th(num_threads, std::vector<double>(nv, 0.0));
    auto gref = [&](int a, int p, int q, int r) -> double { return VAAA[(((size_t)a * na + p) * na + q) * na + r]; };
    std::vector<double> sh_th(num_threads, 0.0);   // adjoint of the uniform denominator shift
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for(int a = 0; a < nv; a++){
            double * gb = VAAAb.data() + (size_t)a * na * n2;           // [p][q][r] slice of this a
            auto G = [&](int p, int q, int r) -> double { return gref(a, p, q, r); };
            auto Gb = [&](int p, int q, int r) -> double & { return gb[((size_t)p * na + q) * na + r]; };
            // ---- terms with the two-index denominators (H) ----
            for(int t = 0; t < na; t++){
                const double dEH = -T.e_v[a] + T.e_IP[t] + pt_vjp_shift;
                const double rH = r_of(dEH, eta), rHp = rp_of(dEH, eta);
                const double H1 = T.H_AV[t * nv + a];
                double dEHb = 0.0;
                for(int w = 0; w < na; w++){
                    const double H2 = T.H_AV[w * nv + a];
                    const double pb = S.PH ? S.PH[t * na + w] : 0.0;
                    dEHb += pb * H1 * H2 * rHp;
                    if(A.H_AV){
#pragma omp atomic
                        A.H_AV[t * nv + a] += pb * H2 * rH;
#pragma omp atomic
                        A.H_AV[w * nv + a] += pb * H1 * rH;
                    }
                }
                for(int y = 0; y < na; y++) for(int x = 0; x < na; x++) for(int w = 0; w < na; w++){
                    const double F2 = G(x, w, y) - G(y, w, x), G2 = G(x, w, y);
                    const double hm = yHM[id4(t, y, x, w)], hj = yHJ[id4(t, y, x, w)];
                    dEHb += (hm * H1 * F2 + hj * H1 * G2) * rHp;
                    if(A.H_AV){
#pragma omp atomic
                        A.H_AV[t * nv + a] += (hm * F2 + hj * G2) * rH;
                    }
                    const double F2b = hm * H1 * rH, G2b = hj * H1 * rH;
                    Gb(x, w, y) += F2b + G2b;  Gb(y, w, x) -= F2b;
                }
                eip_th[th][t] += dEHb;  ev_th[th][a] -= dEHb; sh_th[th] += dEHb;
            }
            // ---- terms with the three-index denominators ----
            for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int t = 0; t < na; t++){
                double dE = -T.e_v[a] + T.e_IP[t] + T.e_IP[u] - T.e_EA[v] + pt_vjp_shift;
                int kind = 0;                                              // 0: all four; 1: v == t; 2: v == u
                if(v == t){ dE = -T.e_v[a] + T.e_IP[u] + pt_vjp_shift; kind = 1; }
                if(v == u){ dE = -T.e_v[a] + T.e_IP[t] + pt_vjp_shift; kind = 2; }
                const double rr = r_of(dE, eta), rrp = rp_of(dE, eta);
                const double F1 = G(t, v, u) - G(u, v, t), G1 = G(u, v, t);
                double rrb = 0.0, F1b = 0.0, G1b = 0.0;
                for(int y = 0; y < na; y++) for(int x = 0; x < na; x++) for(int w = 0; w < na; w++){
                    const double F2 = G(x, w, y) - G(y, w, x), G2 = G(x, w, y);
                    const size_t k = id6(t, u, v, y, x, w);
                    const double a1 = yJK[k], a2 = yJJ[k], a3 = yMJ[k], a4 = yJM[k];
                    rrb += a1 * F1 * F2 + a2 * G1 * G2 + a3 * F1 * G2 + a4 * G1 * F2;
                    F1b += rr * (a1 * F2 + a3 * G2);
                    G1b += rr * (a2 * G2 + a4 * F2);
                    const double F2b = rr * (a1 * F1 + a4 * G1), G2b = rr * (a2 * G1 + a3 * F1);
                    Gb(x, w, y) += F2b + G2b;  Gb(y, w, x) -= F2b;
                }
                // MH and JH: [t,u,v,w] with H_AV[w,a]
                for(int w = 0; w < na; w++){
                    const double H2 = T.H_AV[w * nv + a];
                    const double mh = yMH[id4(t, u, v, w)], jh = yJH[id4(t, u, v, w)];
                    rrb += (mh * F1 + jh * G1) * H2;
                    F1b += rr * mh * H2;  G1b += rr * jh * H2;
                    if(A.H_AV){
#pragma omp atomic
                        A.H_AV[w * nv + a] += rr * (mh * F1 + jh * G1);
                    }
                }
                Gb(t, v, u) += F1b;  Gb(u, v, t) += -F1b + G1b;
                const double dEb = rrb * rrp;
                sh_th[th] += dEb;
                ev_th[th][a] -= dEb;
                if(kind == 0){ eip_th[th][t] += dEb; eip_th[th][u] += dEb; eea_th[th][v] -= dEb; }
                else if(kind == 1){ eip_th[th][u] += dEb; }
                else { eip_th[th][t] += dEb; }
            }
        }
    }
    if(A.VA) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nv * na, aux, na * na, 1.0, VAAAb.data(), na * na, T.integrals->AA_RI_M, aux, 1.0, A.VA, aux);
    if(A.AA) cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na * na, aux, nv * na, 1.0, VAAAb.data(), na * na, T.integrals->VA_RI_M, aux, 1.0, A.AA, aux);
    for(int t = 0; t < num_threads; t++){ A.shift_adj += sh_th[t];
        fold_ea(A, eip_th[t], eea_th[t], na);
        if(A.e_v) for(int a = 0; a < nv; a++) A.e_v[a] += ev_th[t][a];
    }
    return 0;
}

// ---- CA -------------------------------------------------------------------------------------
// forward (calc_EE_2_CA, RI branch). Intermediate k_i(p;q,r) = CAAA[i,p,q,r] = sum_Q CA[i,p,Q] AA[q,r,Q].
// Per (i,t,u,v): rr = r(dE), dE = e_c[i] + e_IP[t] - e_EA[u] - e_EA[v]; if t == u: e_c[i] - e_EA[v]; if t == v: e_c[i] - e_EA[u].
// Per (i,t): rH = r(e_c[i] - e_EA[t]), H = H_CA[i,t]. With V = k(v;u,t) k(w;x,y):
//   ss[t,u,v,y,x,w] += rr V     os[t,u,v,y,x,w] += rr V   (the two accumulators receive identical values)
//   MH[t,u,v,w] += rr (k(v;u,t) - k(u;v,t)) H_CA[i,w]     JH[t,u,v,w] += rr k(v;u,t) H_CA[i,w]
//   HM[t,y,x,w] += rH H_CA[i,t] (k(w;x,y) - k(x;w,y))     HJ[t,y,x,w] += rH H_CA[i,t] k(w;x,y)
//   HH[t,u] += rH H_CA[i,t] H_CA[i,u]
// followed by linear maps of ss, os, MH, HM, JH, HJ, HH into P3_JK, PV_JK, P3_AB, PV_AB, PH, PS. The seed
// transposition of those maps below is GENERATED from the forward statements (each "OUT[a] += c IN[b]" becomes
// "yIN[b] += c Ybar_OUT[a]") and kept as plain code.
int pt_vjp_EE_CA(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A){
    if(RI == 0){ fprintf(out_stream, "ERROR: pt_vjp_EE_CA supports the RI branch only\n"); return 1; }
    const int nc = T.n_c, na = T.n_a;
    const long aux = T.integrals->aux_n_ao;
    const double eta = T.edshift;
    const size_t n2 = (size_t)na * na, n4 = n2 * n2, n6 = n4 * n2;
    auto id4 = [&](int a, int b, int c, int d){ return ((size_t)(a * na + b) * na + c) * na + d; };
    auto id6 = [&](int a, int b, int c, int d, int e, int f){ return ((((size_t)(a * na + b) * na + c) * na + d) * na + e) * na + f; };

    std::vector<double> CAAA((size_t)nc * na * n2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nc * na, na * na, aux, 1.0, T.integrals->CA_RI_M, aux, T.integrals->AA_RI_M, aux, 0.0, CAAA.data(), na * na);

    std::vector<double> ySS(n6, 0.0), yOS(n6, 0.0), yMH(n4, 0.0), yHM(n4, 0.0), yJH(n4, 0.0), yHJ(n4, 0.0), yHH(n2, 0.0);
    // ---- seeds on the accumulators: transpose of the final assembly (generated) ----
    if(S.P3_JK){
        for(int t=0; t<na; t++)
        for(int v=0; v<na; v++)
        for(int u=0; u<na; u++)
        for(int w=0; w<na; w++)
        for(int x=0; x<na; x++)
        for(int y=0; y<na; y++)
        {
            ySS[((((t*na+v)*na+x)*na+y)*na+w)*na+u] += S.P3_JK[((((t*na+u)*na+v)*na+y)*na+x)*na+w];
            ySS[((((t*na+v)*na+y)*na+x)*na+w)*na+u] -= S.P3_JK[((((t*na+u)*na+v)*na+y)*na+x)*na+w];
            ySS[((((u*na+v)*na+x)*na+y)*na+w)*na+t] -= S.P3_JK[((((t*na+u)*na+v)*na+y)*na+x)*na+w];
            ySS[((((u*na+v)*na+y)*na+x)*na+w)*na+t] += S.P3_JK[((((t*na+u)*na+v)*na+y)*na+x)*na+w];
        }
    }
    if(S.PV_JK){
        for(int t=0; t<na; t++)
        for(int v=0; v<na; v++)
        for(int u=0; u<na; u++)
        for(int w=0; w<na; w++)
        for(int x=0; x<na; x++)
        {
            ySS[((((t*na+v)*na+x)*na+w)*na+x)*na+u] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((u*na+v)*na+x)*na+w)*na+x)*na+t] += S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((t*na+w)*na+x)*na+v)*na+x)*na+u] += S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((u*na+w)*na+x)*na+v)*na+x)*na+t] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((t*na+x)*na+v)*na+w)*na+u)*na+x] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((t*na+v)*na+x)*na+w)*na+u)*na+x] += S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((u*na+x)*na+v)*na+w)*na+t)*na+x] += S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((u*na+v)*na+x)*na+w)*na+t)*na+x] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((t*na+x)*na+w)*na+v)*na+u)*na+x] += S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((t*na+w)*na+x)*na+v)*na+u)*na+x] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((u*na+x)*na+w)*na+v)*na+t)*na+x] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            ySS[((((u*na+w)*na+x)*na+v)*na+t)*na+x] += S.PV_JK[((u*na+t)*na+v)*na+w];
            yOS[((((t*na+v)*na+x)*na+w)*na+u)*na+x] += S.PV_JK[((u*na+t)*na+v)*na+w];
            yOS[((((u*na+v)*na+x)*na+w)*na+t)*na+x] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            yOS[((((t*na+w)*na+x)*na+v)*na+u)*na+x] -= S.PV_JK[((u*na+t)*na+v)*na+w];
            yOS[((((u*na+w)*na+x)*na+v)*na+t)*na+x] += S.PV_JK[((u*na+t)*na+v)*na+w];
        }
    }
    if(S.PV_JK){
        for(int t=0; t<na; t++)
        for(int v=0; v<na; v++)
        for(int u=0; u<na; u++)
        for(int w=0; w<na; w++)
        {
            yMH[((t*na+w)*na+v)*na+u] += S.PV_JK[((u*na+t)*na+v)*na+w];
            yMH[((u*na+v)*na+w)*na+t] += S.PV_JK[((u*na+t)*na+v)*na+w];
            yHM[((v*na+w)*na+t)*na+u] += S.PV_JK[((u*na+t)*na+v)*na+w];
            yHM[((w*na+v)*na+u)*na+t] += S.PV_JK[((u*na+t)*na+v)*na+w];
        }
    }
    if(S.P3_AB){
        for(int t=0; t<na; t++)
        for(int u=0; u<na; u++)
        for(int x=0; x<na; x++)
        for(int v=0; v<na; v++)
        for(int w=0; w<na; w++)
        for(int y=0; y<na; y++)
        {
            yOS[((((t*na+v)*na+y)*na+w)*na+u)*na+x] -= S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            yOS[((((u*na+v)*na+y)*na+w)*na+t)*na+x] += S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            yOS[((((t*na+w)*na+y)*na+v)*na+u)*na+x] += S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            yOS[((((u*na+w)*na+y)*na+v)*na+t)*na+x] -= S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            ySS[((((t*na+v)*na+w)*na+y)*na+x)*na+u] -= S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            ySS[((((u*na+v)*na+w)*na+y)*na+x)*na+t] += S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            ySS[((((t*na+w)*na+v)*na+y)*na+x)*na+u] += S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            ySS[((((u*na+w)*na+v)*na+y)*na+x)*na+t] -= S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            yOS[((((x*na+y)*na+v)*na+w)*na+u)*na+t] -= S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            yOS[((((x*na+y)*na+v)*na+w)*na+t)*na+u] += S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            yOS[((((x*na+y)*na+w)*na+v)*na+u)*na+t] += S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
            yOS[((((x*na+y)*na+w)*na+v)*na+t)*na+u] -= S.P3_AB[((((t*na+u)*na+x)*na+v)*na+w)*na+y];
        }
    }
    if(S.PV_AB){
        for(int t=0; t<na; t++)
        for(int u=0; u<na; u++)
        for(int v=0; v<na; v++)
        for(int w=0; w<na; w++)
        for(int x=0; x<na; x++)
        {
            yOS[((((t*na+x)*na+w)*na+v)*na+x)*na+u] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            ySS[((((t*na+x)*na+v)*na+w)*na+u)*na+x] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            ySS[((((t*na+v)*na+x)*na+w)*na+u)*na+x] += S.PV_AB[((t*na+u)*na+v)*na+w];
            ySS[((((u*na+w)*na+x)*na+v)*na+x)*na+t] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            ySS[((((u*na+w)*na+x)*na+v)*na+t)*na+x] += S.PV_AB[((t*na+u)*na+v)*na+w];
            yOS[((((u*na+x)*na+v)*na+w)*na+x)*na+t] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            ySS[((((u*na+x)*na+w)*na+v)*na+t)*na+x] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            yOS[((((u*na+w)*na+x)*na+v)*na+t)*na+x] += S.PV_AB[((t*na+u)*na+v)*na+w];
            ySS[((((t*na+v)*na+x)*na+w)*na+x)*na+u] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            yOS[((((t*na+v)*na+x)*na+w)*na+u)*na+x] += S.PV_AB[((t*na+u)*na+v)*na+w];
        }
    }
    if(S.PV_AB){
        for(int t=0; t<na; t++)
        for(int u=0; u<na; u++)
        for(int v=0; v<na; v++)
        for(int w=0; w<na; w++)
        {
            yJH[((u*na+w)*na+v)*na+t] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            yHJ[((v*na+w)*na+u)*na+t] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            yJH[((t*na+v)*na+w)*na+u] -= S.PV_AB[((t*na+u)*na+v)*na+w];
            yHJ[((w*na+v)*na+t)*na+u] -= S.PV_AB[((t*na+u)*na+v)*na+w];
        }
    }
    if(S.PH){
        for(int t=0; t<na; t++)
        for(int u=0; u<na; u++)
        for(int v=0; v<na; v++)
        for(int y=0; y<na; y++)
        {
            ySS[((((t*na+y)*na+v)*na+u)*na+v)*na+y] -= S.PH[t*na+u];
            ySS[((((t*na+v)*na+y)*na+u)*na+v)*na+y] += S.PH[t*na+u];
            yOS[((((t*na+v)*na+y)*na+u)*na+v)*na+y] += S.PH[t*na+u];
        }
    }
    if(S.PH){
        for(int t=0; t<na; t++)
        for(int u=0; u<na; u++)
        for(int y=0; y<na; y++)
        {
            yMH[((t*na+u)*na+y)*na+y] += S.PH[t*na+u];
            yJH[((t*na+u)*na+y)*na+y] += S.PH[t*na+u];
            yHM[((y*na+u)*na+t)*na+y] += S.PH[t*na+u];
            yHJ[((y*na+u)*na+t)*na+y] += S.PH[t*na+u];
        }
    }
    if(S.PH){
        for(int t=0; t<na; t++)
        for(int u=0; u<na; u++)
        {
            yHH[u*na+t] -= S.PH[t*na+u];
        }
    }
    for(int t=0; t<na; t++)
    {
        yHH[t*na+t] += 2.0*S.PS;
    }

    std::vector<double> CAAAb((size_t)nc * na * n2, 0.0);
    std::vector<std::vector<double>> ec_th(num_threads, std::vector<double>(nc, 0.0)), eip_th(num_threads, std::vector<double>(na, 0.0)), eea_th(num_threads, std::vector<double>(na, 0.0));
    std::vector<double> sh_th(num_threads, 0.0);   // adjoint of the uniform denominator shift
#pragma omp parallel
    {
        const int th = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for(int i = 0; i < nc; i++){
            double * kb = CAAAb.data() + (size_t)i * na * n2;
            auto K  = [&](int p, int q, int r) -> double { return CAAA[(((size_t)i * na + p) * na + q) * na + r]; };
            auto Kb = [&](int p, int q, int r) -> double & { return kb[((size_t)p * na + q) * na + r]; };
            // two-index denominators (H terms)
            for(int t = 0; t < na; t++){
                const double dEH = T.e_c[i] - T.e_EA[t] + pt_vjp_shift;
                const double rH = r_of(dEH, eta), rHp = rp_of(dEH, eta);
                const double H1 = T.H_CA[i * na + t];
                double dEHb = 0.0, H1b = 0.0;
                for(int u = 0; u < na; u++){
                    const double H2 = T.H_CA[i * na + u];
                    const double hh = yHH[t * na + u];
                    dEHb += hh * H1 * H2 * rHp;
                    H1b += rH * hh * H2;
                    if(A.H_CA){
#pragma omp atomic
                        A.H_CA[i * na + u] += rH * hh * H1;
                    }
                }
                for(int y = 0; y < na; y++) for(int x = 0; x < na; x++) for(int w = 0; w < na; w++){
                    const double F2 = K(w, x, y) - K(x, w, y), G2 = K(w, x, y);
                    const double hm = yHM[id4(t, y, x, w)], hj = yHJ[id4(t, y, x, w)];
                    dEHb += H1 * (hm * F2 + hj * G2) * rHp;
                    H1b += rH * (hm * F2 + hj * G2);
                    Kb(w, x, y) += rH * H1 * (hm + hj);  Kb(x, w, y) -= rH * H1 * hm;
                }
                if(A.H_CA){
#pragma omp atomic
                    A.H_CA[i * na + t] += H1b;
                }
                ec_th[th][i] += dEHb;  eea_th[th][t] -= dEHb; sh_th[th] += dEHb;
            }
            // three-index denominators
            for(int v = 0; v < na; v++) for(int u = 0; u < na; u++) for(int t = 0; t < na; t++){
                double dE = T.e_c[i] + T.e_IP[t] - T.e_EA[u] - T.e_EA[v] + pt_vjp_shift;
                int kind = 0;                                                  // 0: all three; 1: t == u; 2: t == v
                if(t == u){ dE = T.e_c[i] - T.e_EA[v] + pt_vjp_shift; kind = 1; }
                if(t == v){ dE = T.e_c[i] - T.e_EA[u] + pt_vjp_shift; kind = 2; }
                const double rr = r_of(dE, eta), rrp = rp_of(dE, eta);
                const double k1 = K(v, u, t), k1x = K(u, v, t);
                double rrb = 0.0, k1b = 0.0;
                for(int y = 0; y < na; y++) for(int x = 0; x < na; x++) for(int w = 0; w < na; w++){
                    const double k2 = K(w, x, y);
                    const double a = ySS[id6(t, u, v, y, x, w)] + yOS[id6(t, u, v, y, x, w)];
                    rrb += a * k1 * k2;
                    k1b += rr * a * k2;
                    Kb(w, x, y) += rr * a * k1;
                }
                for(int w = 0; w < na; w++){
                    const double H2 = T.H_CA[i * na + w];
                    const double mh = yMH[id4(t, u, v, w)], jh = yJH[id4(t, u, v, w)];
                    rrb += (mh * (k1 - k1x) + jh * k1) * H2;
                    k1b += rr * (mh + jh) * H2;
                    Kb(u, v, t) -= rr * mh * H2;
                    if(A.H_CA){
#pragma omp atomic
                        A.H_CA[i * na + w] += rr * (mh * (k1 - k1x) + jh * k1);
                    }
                }
                Kb(v, u, t) += k1b;
                const double dEb = rrb * rrp;
                sh_th[th] += dEb;
                ec_th[th][i] += dEb;
                if(kind == 0){ eip_th[th][t] += dEb; eea_th[th][u] -= dEb; eea_th[th][v] -= dEb; }
                else if(kind == 1){ eea_th[th][v] -= dEb; }
                else { eea_th[th][u] -= dEb; }
            }
        }
    }
    if(A.CA) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nc * na, aux, na * na, 1.0, CAAAb.data(), na * na, T.integrals->AA_RI_M, aux, 1.0, A.CA, aux);
    if(A.AA) cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na * na, aux, nc * na, 1.0, CAAAb.data(), na * na, T.integrals->CA_RI_M, aux, 1.0, A.AA, aux);
    for(int t = 0; t < num_threads; t++){ A.shift_adj += sh_th[t];
        if(A.e_c) for(int i = 0; i < nc; i++) A.e_c[i] += ec_th[t][i];
        fold_ea(A, eip_th[t], eea_th[t], na);
    }
    return 0;
}

// ---- end of E2_calc_EE: symmetrization and the RF_P3_AB permutation ---------------------------------
PTRawSeeds PTRawSeedStore::view() const {
    PTRawSeeds v;
    v.PS = PS; v.PH = PH.data(); v.PV_JK = PV_JK.data(); v.PV_AB = PV_AB.data(); v.P3_JK = P3_JK.data(); v.P3_AB = P3_AB.data();
    return v;
}

namespace{
// A_ij <- (A_ij + A_ji)/2 for i < j, in place, on an m x m matrix (the forward symmetrization(); self-adjoint)
void sym_matrix(std::vector<double> & A, size_t m){
    for(size_t i = 0; i < m; i++) for(size_t j = i + 1; j < m; j++){
        const double x = 0.5 * (A[i * m + j] + A[j * m + i]);
        A[i * m + j] = x; A[j * m + i] = x;
    }
}
}

int pt_vjp_EE_final(int n_a, double PS, const std::vector<double> & PH, const std::vector<double> & PV_JK, const std::vector<double> & PV_AB,
                    const std::vector<double> & P3_JK, const std::vector<double> & P3_AB, PTRawSeedStore * raw){
    const size_t n = n_a, n2 = n * n, n3 = n2 * n, n4 = n2 * n2, n6 = n3 * n3;
    raw->PS = PS;
    raw->PH = PH; raw->PV_JK = PV_JK; raw->PV_AB = PV_AB; raw->P3_JK = P3_JK;
    // forward: raw -> sym -> permutation; reverse: permutation^T first, then sym
    raw->P3_AB.assign(n6, 0.0);
    for(size_t t = 0; t < n; t++) for(size_t u = 0; u < n; u++) for(size_t v = 0; v < n; v++)
    for(size_t w = 0; w < n; w++) for(size_t x = 0; x < n; x++) for(size_t y = 0; y < n; y++)
        raw->P3_AB[((((t * n + u) * n + x) * n + v) * n + w) * n + y] = P3_AB[((((t * n + u) * n + v) * n + w) * n + x) * n + y];
    sym_matrix(raw->PH, n);
    sym_matrix(raw->PV_JK, n2); sym_matrix(raw->PV_AB, n2);
    sym_matrix(raw->P3_JK, n3); sym_matrix(raw->P3_AB, n3);
    (void)n4;
    return 0;
}

int pt_vjp_EE_all(PT_tensors & T, double PS, const std::vector<double> & PH, const std::vector<double> & PV_JK, const std::vector<double> & PV_AB,
                  const std::vector<double> & P3_JK, const std::vector<double> & P3_AB, PTInputAdj & A){
    PTRawSeedStore raw;
    pt_vjp_EE_final(T.n_a, PS, PH, PV_JK, PV_AB, P3_JK, P3_AB, &raw);
    const PTRawSeeds S = raw.view();
    int bad = 0;
    bad += pt_vjp_EE_CCVV(T, S, A); bad += pt_vjp_EE_CAVV(T, S, A); bad += pt_vjp_EE_AAVV(T, S, A); bad += pt_vjp_EE_CCAV(T, S, A);
    bad += pt_vjp_EE_CCAA(T, S, A); bad += pt_vjp_EE_CV(T, S, A);   bad += pt_vjp_EE_AV(T, S, A);   bad += pt_vjp_EE_CA(T, S, A);
    return bad;
}
