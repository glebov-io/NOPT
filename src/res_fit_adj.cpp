#include <algorithm>
// Adjoint (transpose with respect to the resolvent tables) of the contraction routines of res_fit.cpp:
//   forward:  P[i_s,j_s] += ddot(L, R+S, C) * ci[det*ld+i_s] * sign * ci[detext*ld+j_s]      (for every term of every routine)
//   adjoint:  Rbar[S+l]  += C[l] * sign * ci[det*ld+i_s] * sum_j W[i_s,j] ci[detext*ld+j]     (same loops, same terms)
// Rbar has the layout of the tables (N_fit innermost). The loops are copied from res_fit.cpp; res_fit_adj_test (NOPT_XMC_ADJ_TEST) checks every routine with
// <W, P(tables)> = <Rbar(W), tables> on random tables.
#include "res_fit.h"
#include "from_hash.h"
#include "res_fit_adj.h"
#include "res_fit_deriv.h"

#include <cmath>
#include <cstring>
#include <omp.h>

#include "blas_link.h"
#include "common_vars.h"
#include "matr.h"

extern int n_fit_pol;

namespace{
// One term of the forward routines:   T = ddot(L, R+S, C) * sign * CI_0[i_s] * sum_j W[i_s,j] CI_e[j]   (for every i_s), with R the table row of the term.
// Accumulated: table seeds (o.Rbar), CI seeds (o.cibar; CI_0 = ci + k*ld, CI_e = ci + k'*ld), weight seeds (o.Cbar[(det*n_s+i_s)*np1 + l]).
inline void adj_term(const ResFitAdjOut & o, size_t idx, int n_s, const double * W, const double * ci, const double * CI_0, const double * CI_e, double sign, int det,
                     const int * L_fit, const int * S_fit, const double * C_fit){
    const size_t row = idx * o.Nf;                  // the forward tables: all grid points
    const size_t rowb = idx * o.nb;                 // the seeds: grid points f0 .. f0 + nb - 1
    const int np1 = 2 * n_fit_pol + 1;
    for(int i_s = 0; i_s < n_s; i_s++){
        double we = 0.0;
        for(int j = 0; j < n_s; j++) we += W[i_s * n_s + j] * CI_e[j];       // sum_j W[i_s,j] CI_e[j]
        const int L = L_fit[det * n_s + i_s], S = S_fit[det * n_s + i_s];
        const double * C = C_fit + (size_t)np1 * (det * n_s + i_s);
        const double coef = sign * CI_0[i_s] * we;
        if(o.Rbar){
            for(int l = 0; l < L; l++){
                const int f = S + l - o.f0;
                if(f < 0 || f >= o.nb) continue;
#pragma omp atomic
                o.Rbar[rowb + f] += coef * C[l];
            }
        }
        if(o.Rfwd && (o.cibar || o.Cbar)){
            double Rc = 0.0;
            for(int l = 0; l < L; l++) Rc += o.Rfwd[row + S + l] * C[l];
            if(o.Cbar){
                for(int l = 0; l < L; l++){
#pragma omp atomic
                    o.Cbar[(size_t)np1 * (det * n_s + i_s) + l] += coef * o.Rfwd[row + S + l];
                }
            }
            if(o.cibar){
                double * cb0 = o.cibar + (CI_0 - ci);
                double * cbe = o.cibar + (CI_e - ci);
#pragma omp atomic
                cb0[i_s] += Rc * sign * we;
                for(int j = 0; j < n_s; j++){
#pragma omp atomic
                    cbe[j] += Rc * sign * CI_0[i_s] * W[i_s * n_s + j];
                }
            }
        }
    }
}
}

int res_fit_adj_0body(const double * W, int n_s, int ld, double * ci, int Na, int Nb, const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th){
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th)
    for(int j_CI = 0; j_CI < Nb; j_CI++){
        const int det = i_CI * Nb + j_CI;
        double * c0 = ci + det * ld;
        adj_term(o, 0, n_s, W, ci, c0, c0, 1.0, det, L_fit, S_fit, C_fit);
    }
    return 0;
}

int res_fit_adj_1body(const double * W, int n_s, int ld, double * ci, int N, int na, int Na, int Nb, int * fa, int * vec_a,
                      const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th){
    int * bit_a = new int[N];
    int * buf = new int[na + 1];
    buf[0] = 0;
    double sign, sign1, sign2;
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th){
        memset(bit_a, 0, N * sizeof(int));
        for(int k = 1; k < na + 1; k++) bit_a[vec_a[i_CI * (na + 1) + k] - 1] = 1;
        sign1 = 1;
        for(int t = 0; t < N; t++) if(bit_a[t] != 0){
            sign2 = 1;
            bit_a[t] = 0;
            for(int w = 0; w < N; w++){
                if(bit_a[w] == 0){
                    bit_a[w] = 1;
                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                    sign = sign1 * sign2;
                    for(int j_CI = 0; j_CI < Nb; j_CI++)
                        adj_term(o, (size_t)(t * N + w), n_s, W, ci, ci + (i_CI * Nb + j_CI) * ld, ci + (i_CIext * Nb + j_CI) * ld, sign, i_CI * Nb + j_CI, L_fit, S_fit, C_fit);
                    bit_a[w] = 0;
                }
                else
                    sign2 = -sign2;
            }
            bit_a[t] = 1;
            sign1 = -sign1;
        }
    }
    delete[] bit_a;
    delete[] buf;
    return 0;
}

int res_fit_adj_2body_AA(const double * W, int n_s, int ld, double * ci, int N, int na, int Na, int Nb, int * fa, int * vec_a,
                         const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th){
    int * bit_a = new int[N];
    int * buf = new int[na + 1];
    buf[0] = 0;
    double sign, sign1, sign2, sign3;
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th){
        memset(bit_a, 0, N * sizeof(int));
        for(int k = 1; k < na + 1; k++) bit_a[vec_a[i_CI * (na + 1) + k] - 1] = 1;
        sign1 = 1;
        for(int t = 0; t < N; t++) if(bit_a[t] != 0){
            sign1 = 1;
            bit_a[t] = 0;
            for(int u = t + 1; u < N; u++){
                if(bit_a[u] != 0){
                    sign2 = 1;
                    bit_a[u] = 0;
                    for(int v = 0; v < N; v++){
                        if(bit_a[v] == 0){
                            sign3 = sign2;
                            bit_a[v] = 1;
                            for(int w = v + 1; w < N; w++){
                                if(bit_a[w] == 0){
                                    bit_a[w] = 1;
                                    sign = sign1 * sign2 * sign3;
                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                    const size_t R = (size_t)(((u * N + t) * N + v) * N + w);
                                    for(int j_CI = 0; j_CI < Nb; j_CI++)
                                        adj_term(o, R, n_s, W, ci, ci + (i_CI * Nb + j_CI) * ld, ci + (i_CIext * Nb + j_CI) * ld, sign, i_CI * Nb + j_CI, L_fit, S_fit, C_fit);
                                    bit_a[w] = 0;
                                }
                                else
                                    sign3 = -sign3;
                            }
                            bit_a[v] = 0;
                        }
                        else
                            sign2 = -sign2;
                    }
                    bit_a[u] = 1;
                    sign1 = -sign1;
                }
            }
            bit_a[t] = 1;
        }
    }
    delete[] bit_a;
    delete[] buf;
    return 0;
}

int res_fit_adj_2body_AB(const double * W, int n_s, int ld, double * ci, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                         const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th){
    const int Na_zv = (int) std::lround(tgammal(N) / tgammal(na) / tgammal(N - na + 1));
    const int Nb_zv = (int) std::lround(tgammal(N) / tgammal(nb) / tgammal(N - nb + 1));
    int * Ia = new int[Na_zv];
    int * Ia_friends = new int[Na_zv * (N - na) * 3];
    int * Ib = new int[Nb_zv];
    int * Ib_friends = new int[Nb_zv * (N - nb) * 3];
    double sign;
    for(int t = 0; t < N; t++){
        get_I_I_friends(na, N, Na, fa, t, vec_a, Ia, Ia_friends);
        for(int u = 0; u < N; u++){
            get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib, Ib_friends);
            for(int i_CI = i_th; i_CI < Na_zv; i_CI += n_th){
                int k_CI = Ia[i_CI];
                for(int j_CI = 0; j_CI < Nb_zv; j_CI++){
                    int l_CI = Ib[j_CI];
                    const double * CI_0 = ci + (k_CI * Nb + l_CI) * ld;
                    const int det = k_CI * Nb + l_CI;
                    adj_term(o, (size_t)(((t * N + u) * N + t) * N + u), n_s, W, ci, CI_0, CI_0, 1.0, det, L_fit, S_fit, C_fit);
                    for(int n = 0; n < (N - nb); n++){
                        int * ind_n = Ib_friends + j_CI * (N - nb) * 3 + n * 3;
                        const double * CI_e = ci + (k_CI * Nb + ind_n[0]) * ld;
                        adj_term(o, (size_t)(((t * N + u) * N + t) * N + ind_n[1]), n_s, W, ci, CI_0, CI_e, (double)ind_n[2], det, L_fit, S_fit, C_fit);
                    }
                    for(int m = 0; m < (N - na); m++){
                        int * ind_m = Ia_friends + i_CI * (N - na) * 3 + m * 3;
                        const double * CI_e = ci + (ind_m[0] * Nb + l_CI) * ld;
                        adj_term(o, (size_t)(((t * N + u) * N + ind_m[1]) * N + u), n_s, W, ci, CI_0, CI_e, (double)ind_m[2], det, L_fit, S_fit, C_fit);
                        for(int n = 0; n < (N - nb); n++){
                            int * ind_n = Ib_friends + j_CI * (N - nb) * 3 + n * 3;
                            const double * CI_e2 = ci + (ind_m[0] * Nb + ind_n[0]) * ld;
                            sign = ind_m[2] * ind_n[2];
                            adj_term(o, (size_t)(((t * N + u) * N + ind_m[1]) * N + ind_n[1]), n_s, W, ci, CI_0, CI_e2, sign, det, L_fit, S_fit, C_fit);
                        }
                    }
                }
            }
        }
    }
    delete[] Ia; delete[] Ia_friends; delete[] Ib; delete[] Ib_friends;
    return 0;
}

int res_fit_adj_3body_AAA(const double * W, int n_s, int ld, double * ci, int N, int na, int Na, int Nb, int * fa, int * vec_a,
                          const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th){
    int * bit_a = new int[N];
    int * buf = new int[na + 1];
    buf[0] = 0;
    double sign, sign1, sign2, sign3, sign4;
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th){
        memset(bit_a, 0, N * sizeof(int));
        for(int k = 1; k < na + 1; k++) bit_a[vec_a[i_CI * (na + 1) + k] - 1] = 1;
        for(int t = 0; t < N; t++){
            if(bit_a[t] != 0){
                bit_a[t] = 0;
                sign1 = 1;
                for(int u = t + 1; u < N; u++) if(bit_a[u] != 0){
                    bit_a[u] = 0;
                    sign2 = 1;
                    for(int v = 0; v < N; v++){
                        if(bit_a[v] == 0){
                            bit_a[v] = 1;
                            sign3 = 1;
                            for(int w = 0; w < N; w++) if(bit_a[w] != 0){
                                bit_a[w] = 0;
                                for(int x = 0; x < N; x++) if(bit_a[x] == 0){
                                    sign4 = 1;
                                    for(int y = x + 1; y < N; y++){
                                        if(bit_a[y] == 0){
                                            bit_a[x] = 1;
                                            bit_a[y] = 1;
                                            auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                            sign = sign1 * sign2 * sign3 * sign4;
                                            const size_t R = (size_t)(((((t * N + u) * N + v) * N + y) * N + x) * N + w);
                                            for(int j_CI = 0; j_CI < Nb; j_CI++)
                                                adj_term(o, R, n_s, W, ci, ci + (i_CI * Nb + j_CI) * ld, ci + (i_CIext * Nb + j_CI) * ld, sign, i_CI * Nb + j_CI, L_fit, S_fit, C_fit);
                                            bit_a[x] = 0;
                                            bit_a[y] = 0;
                                        }
                                        else
                                            sign4 = -sign4;
                                    }
                                }
                                bit_a[w] = 1;
                                sign3 = -sign3;
                            }
                            bit_a[v] = 0;
                        }
                        else
                            sign2 = -sign2;
                    }
                    bit_a[u] = 1;
                    sign1 = -sign1;
                }
                bit_a[t] = 1;
            }
        }
    }
    delete[] bit_a;
    delete[] buf;
    return 0;
}

int res_fit_adj_3body_AB(const double * W, int n_s, int ld, double * ci, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                         const ResFitAdjOut & o, int * L_fit, int * S_fit, double * C_fit, int N_fit, int i_th, int n_th){
    int * bit_a = new int[N];
    int * bit_b = new int[N];
    int * buf = new int[na + 1];
    buf[0] = 0;
    double sign, sign1, sign2, sign4, sign5, sign6;
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th)
    for(int j_CI = 0; j_CI < Nb; j_CI++){
        memset(bit_a, 0, N * sizeof(int));
        memset(bit_b, 0, N * sizeof(int));
        for(int k = 1; k < na + 1; k++) bit_a[vec_a[i_CI * (na + 1) + k] - 1] = 1;
        for(int k = 1; k < nb + 1; k++) bit_b[vec_b[j_CI * (nb + 1) + k] - 1] = 1;
        for(int t = 0; t < N; t++){
            if(bit_a[t] != 0){
                bit_a[t] = 0;
                sign1 = 1;
                for(int u = t + 1; u < N; u++) if(bit_a[u] != 0){
                    bit_a[u] = 0;
                    sign2 = 1;
                    for(int v = 0; v < N; v++){
                        if(bit_a[v] == 0){
                            bit_a[v] = 1;
                            sign5 = sign2;
                            for(int w = v + 1; w < N; w++){
                                if(bit_a[w] == 0){
                                    bit_a[w] = 1;
                                    sign4 = 1;
                                    for(int x = 0; x < N; x++) if(bit_b[x] != 0){
                                        bit_b[x] = 0;
                                        sign6 = 1;
                                        for(int y = 0; y < N; y++){
                                            if(bit_b[y] == 0){
                                                bit_b[y] = 1;
                                                auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                auto j_CIext = get_ind_from_ON(bit_b, N, nb, fb, buf);
                                                sign = sign1 * sign2 * sign4 * sign5 * sign6;
                                                const size_t R = (size_t)(((((t * N + u) * N + v) * N + w) * N + x) * N + y);
                                                adj_term(o, R, n_s, W, ci, ci + (i_CI * Nb + j_CI) * ld, ci + (i_CIext * Nb + j_CIext) * ld, sign, i_CI * Nb + j_CI, L_fit, S_fit, C_fit);
                                                bit_b[y] = 0;
                                            }
                                            else
                                                sign6 = -sign6;
                                        }
                                        bit_b[x] = 1;
                                        sign4 = -sign4;
                                    }
                                    bit_a[w] = 0;
                                }
                                else
                                    sign5 = -sign5;
                            }
                            bit_a[v] = 0;
                        }
                        else
                            sign2 = -sign2;
                    }
                    bit_a[u] = 1;
                    sign1 = -sign1;
                }
                bit_a[t] = 1;
            }
        }
    }
    delete[] bit_a; delete[] bit_b; delete[] buf;
    return 0;
}

// ---- the whole E2 matrix ------------------------------------------------------------------------------------------
namespace{
// Runs body(i_th, n_th) on every OpenMP thread (the contraction routines split their outer string loop that way; their writes are atomic).
template <class F> void on_threads(F body){
#pragma omp parallel
    body(omp_get_thread_num(), omp_get_num_threads());
}
}

int res_fit_E2_adjoint(res_fit_data & RF, const double * W, ResFitAdjResult * R, bool want_ci_and_grid, bool with_3body){
    const int nA = RF.n_a, Nf = RF.N_fit, ns = RF.n_s;
    const size_t n2 = (size_t)nA * nA, n4 = n2 * n2, n6 = n4 * n2;
    const size_t nd = (size_t)RF.ci->Na * RF.ci->Nb;
    const int np1 = 2 * n_fit_pol + 1;
    ResFitSeeds & S = R->tables;
    S.PS.assign(Nf, 0.0); S.PH.assign(n2 * Nf, 0.0); S.PV_JK.assign(n4 * Nf, 0.0); S.PV_AB.assign(n4 * Nf, 0.0);
    if(with_3body){ S.P3_JK.assign(n6 * Nf, 0.0); S.P3_AB.assign(n6 * Nf, 0.0); }
    else{ S.P3_JK.clear(); S.P3_AB.clear(); }
    aldet_data * ci = RF.ci;
    // the beta variants run on coef_bas (b-major) with transposed fit arrays: separate accumulators, folded below
    std::vector<double> cib, cib_t, Cb, Cb_t;
    if(want_ci_and_grid){
        cib.assign(nd * ns, 0.0); cib_t.assign(nd * ns, 0.0); Cb.assign(nd * ns * np1, 0.0); Cb_t.assign(nd * ns * np1, 0.0);
    }
    // the forward tables are read in place (they are needed for the CI and weight seeds only)
    auto out = [&](double * Rbar, const double * Rfwd, bool tr){
        ResFitAdjOut o; o.Rbar = Rbar; o.Nf = Nf; o.f0 = 0; o.nb = Nf;
        if(want_ci_and_grid){ o.Rfwd = Rfwd; o.cibar = tr ? cib_t.data() : cib.data(); o.Cbar = tr ? Cb_t.data() : Cb.data(); }
        return o;
    };
    double * P3J = with_3body ? S.P3_JK.data() : nullptr, * P3A = with_3body ? S.P3_AB.data() : nullptr;
    const ResFitAdjOut o0 = out(S.PS.data(), RF.RF_PS, false), o1 = out(S.PH.data(), RF.RF_PH, false), o1t = out(S.PH.data(), RF.RF_PH, true),
                       o2a = out(S.PV_AB.data(), RF.RF_PV_AB, false), o2j = out(S.PV_JK.data(), RF.RF_PV_JK, false), o2jt = out(S.PV_JK.data(), RF.RF_PV_JK, true),
                       o3j = out(P3J, RF.RF_P3_JK, false), o3jt = out(P3J, RF.RF_P3_JK, true), o3a = out(P3A, RF.RF_P3_AB, false), o3at = out(P3A, RF.RF_P3_AB, true);
    on_threads([&](int it, int nt){
        res_fit_adj_0body(W, ns, ns, ci->coef[0], ci->Na, ci->Nb, o0, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
        res_fit_adj_1body(W, ns, ns, ci->coef[0], nA, ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a, o1, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
        res_fit_adj_1body(W, ns, ns, ci->coef_bas[0], nA, ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b, o1t, RF.L_fit_tr, RF.S_fit_tr, RF.C_fit_tr, Nf, it, nt);
        res_fit_adj_2body_AB(W, ns, ns, ci->coef[0], nA, ci->na, ci->nb, ci->Na, ci->Nb, ci->fa, ci->fb, ci->vec_a, ci->vec_b, o2a, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
        res_fit_adj_2body_AA(W, ns, ns, ci->coef[0], nA, ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a, o2j, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
        res_fit_adj_2body_AA(W, ns, ns, ci->coef_bas[0], nA, ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b, o2jt, RF.L_fit_tr, RF.S_fit_tr, RF.C_fit_tr, Nf, it, nt);
        if(with_3body || want_ci_and_grid){
            res_fit_adj_3body_AAA(W, ns, ns, ci->coef[0], nA, ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a, o3j, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
            res_fit_adj_3body_AAA(W, ns, ns, ci->coef_bas[0], nA, ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b, o3jt, RF.L_fit_tr, RF.S_fit_tr, RF.C_fit_tr, Nf, it, nt);
            res_fit_adj_3body_AB(W, ns, ns, ci->coef[0], nA, ci->na, ci->nb, ci->Na, ci->Nb, ci->fa, ci->fb, ci->vec_a, ci->vec_b, o3a, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
            res_fit_adj_3body_AB(W, ns, ns, ci->coef_bas[0], nA, ci->nb, ci->na, ci->Nb, ci->Na, ci->fb, ci->fa, ci->vec_b, ci->vec_a, o3at, RF.L_fit_tr, RF.S_fit_tr, RF.C_fit_tr, Nf, it, nt);
        }
    });
    if(!want_ci_and_grid) return 0;

    // fold the beta variants: coef_bas[b*Na + a, s] = coef[a*Nb + b, s]
    const int Na = ci->Na, Nb = ci->Nb;
    R->cibar = cib;
    for(int a = 0; a < Na; a++) for(int b = 0; b < Nb; b++){
        for(int s = 0; s < ns; s++) R->cibar[((size_t)a * Nb + b) * ns + s] += cib_t[((size_t)b * Na + a) * ns + s];
        for(int s = 0; s < ns; s++) for(int l = 0; l < np1; l++) Cb[(((size_t)a * Nb + b) * ns + s) * np1 + l] += Cb_t[(((size_t)b * Na + a) * ns + s) * np1 + l];
    }
    // weights -> evaluation points and grid:  x = E0[s] - E_act(det)
    R->e0bar.assign(ns, 0.0); R->Eactbar.assign(nd, 0.0);
    std::vector<double> gbar(Nf, 0.0);
    for(size_t det = 0; det < nd; det++) for(int s = 0; s < ns; s++){
        const int L = RF.L_fit[det * ns + s], St = RF.S_fit[det * ns + s];
        const double x = RF.E0[s] - ci->E_act[0][det];
        double xbar = 0.0;
        rf_weights_vjp(x, RF.E_fit, St, L, &Cb[(det * ns + s) * np1], &xbar, gbar.data());
        rf_x_vjp(xbar, s, (long)det, R->e0bar.data(), R->Eactbar.data());
    }
    RFGridBranch br = rf_grid_branch(RF.E0, ns, ci->E_act[0], (long)nd);
    rf_grid_vjp(br, Nf, gbar.data(), R->e0bar.data(), R->Eactbar.data());
    return 0;
}

int res_fit_E2_adjoint_3body(res_fit_data & RF, const double * W, int f0, int nb, double * P3J, double * P3A){
    const int nA = RF.n_a, Nf = RF.N_fit, ns = RF.n_s;
    const size_t n6 = (size_t)nA * nA * nA * nA * nA * nA;
    std::fill(P3J, P3J + n6 * nb, 0.0); std::fill(P3A, P3A + n6 * nb, 0.0);
    aldet_data * ci = RF.ci;
    auto out = [&](double * Rbar){ ResFitAdjOut o; o.Rbar = Rbar; o.Nf = Nf; o.f0 = f0; o.nb = nb; return o; };
    const ResFitAdjOut oj = out(P3J), oa = out(P3A);
    on_threads([&](int it, int nt){
        res_fit_adj_3body_AAA(W, ns, ns, ci->coef[0], nA, ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a, oj, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
        res_fit_adj_3body_AAA(W, ns, ns, ci->coef_bas[0], nA, ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b, oj, RF.L_fit_tr, RF.S_fit_tr, RF.C_fit_tr, Nf, it, nt);
        res_fit_adj_3body_AB(W, ns, ns, ci->coef[0], nA, ci->na, ci->nb, ci->Na, ci->Nb, ci->fa, ci->fb, ci->vec_a, ci->vec_b, oa, RF.L_fit, RF.S_fit, RF.C_fit, Nf, it, nt);
        res_fit_adj_3body_AB(W, ns, ns, ci->coef_bas[0], nA, ci->nb, ci->na, ci->Nb, ci->Na, ci->fb, ci->fa, ci->vec_b, ci->vec_a, oa, RF.L_fit_tr, RF.S_fit_tr, RF.C_fit_tr, Nf, it, nt);
    });
    return 0;
}

// ---- tests ----------------------------------------------------------------------------------------------------------
#include <random>
namespace{
double E2_dot(res_fit_data & RF, const double * W){
    const int ns = RF.n_s;
    std::vector<double> E2((size_t)ns * ns, 0.0);
    RF.calc_0body(E2.data(), 0, 1); RF.calc_1body_A(E2.data(), 0, 1); RF.calc_1body_B(E2.data(), 0, 1);
    RF.calc_2body_AA(E2.data(), 0, 1); RF.calc_2body_AB(E2.data(), 0, 1); RF.calc_2body_BB(E2.data(), 0, 1);
    RF.calc_3body_AAA(E2.data(), 0, 1); RF.calc_3body_AAB(E2.data(), 0, 1); RF.calc_3body_ABB(E2.data(), 0, 1); RF.calc_3body_BBB(E2.data(), 0, 1);
    double l = 0; for(int i = 0; i < ns * ns; i++) l += W[i] * E2[i];
    return l;
}
}

int res_fit_adj_selftest(res_fit_data & RF){
    const int nA = RF.n_a, Nf = RF.N_fit, ns = RF.n_s;
    const size_t n2 = (size_t)nA * nA, n4 = n2 * n2, n6 = n4 * n2;
    std::mt19937_64 gen(99);
    auto rnd = [&](){ return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); };
    auto fill = [&](double * a, size_t n){ for(size_t i = 0; i < n; i++) a[i] = rnd(); };
    fill(RF.RF_PS, Nf); fill(RF.RF_PH, n2 * Nf); fill(RF.RF_PV_JK, n4 * Nf); fill(RF.RF_PV_AB, n4 * Nf); fill(RF.RF_P3_JK, n6 * Nf); fill(RF.RF_P3_AB, n6 * Nf);
    std::vector<double> W((size_t)ns * ns);
    for(double & w : W) w = rnd();
    ResFitAdjResult R;
    res_fit_E2_adjoint(RF, W.data(), &R);
    ResFitSeeds & S = R.tables;
    int fails = 0;
    auto dotv = [](const std::vector<double> & a, const double * b){ double s = 0; for(size_t i = 0; i < a.size(); i++) s += a[i] * b[i]; return s; };

    // (1) tables: <W, E2(tables)> = <Rbar(W), tables>; the error relative to the sum of |table parts| (the parts can cancel)
    const double lhs = E2_dot(RF, W.data());
    const double rk[6] = {dotv(S.PS, RF.RF_PS), dotv(S.PH, RF.RF_PH), dotv(S.PV_JK, RF.RF_PV_JK), dotv(S.PV_AB, RF.RF_PV_AB), dotv(S.P3_JK, RF.RF_P3_JK), dotv(S.P3_AB, RF.RF_P3_AB)};
    double rhs = 0, scale = std::fabs(lhs);
    for(int k = 0; k < 6; k++){ rhs += rk[k]; scale += std::fabs(rk[k]); }
    double err = std::fabs(lhs - rhs) / (1e-12 + scale);
    fprintf(out_stream, "\nXMC contraction adjoint, tables:  <W, E2(tables)> = % .12e   <Rbar(W), tables> = % .12e   rel.err %.1e %s\n", lhs, rhs, err, err < 1e-11 ? "ok" : "MISMATCH");
    if(err >= 1e-11) fails++;
    const char * names[6] = {"PS", "PH", "PV_JK", "PV_AB", "P3_JK", "P3_AB"};
    double * tabs[6] = {RF.RF_PS, RF.RF_PH, RF.RF_PV_JK, RF.RF_PV_AB, RF.RF_P3_JK, RF.RF_P3_AB};
    size_t sizes[6] = {(size_t)Nf, n2 * Nf, n4 * Nf, n4 * Nf, n6 * Nf, n6 * Nf};
    const std::vector<double> * seeds[6] = {&S.PS, &S.PH, &S.PV_JK, &S.PV_AB, &S.P3_JK, &S.P3_AB};
    std::vector<std::vector<double>> saved(6);
    for(int k = 0; k < 6; k++) saved[k].assign(tabs[k], tabs[k] + sizes[k]);
    for(int k = 0; k < 6; k++){
        for(int j = 0; j < 6; j++){ if(j == k) std::copy(saved[j].begin(), saved[j].end(), tabs[j]); else std::fill(tabs[j], tabs[j] + sizes[j], 0.0); }
        const double l = E2_dot(RF, W.data());
        const double r = dotv(*seeds[k], tabs[k]);
        const double e = std::fabs(l - r) / (1e-12 + std::fabs(l));
        const bool ok = e < 1e-11 || (std::fabs(l) < 1e-12 && std::fabs(r) < 1e-12);
        fprintf(out_stream, "   table %-6s <W, E2> = % .12e   <Rbar, table> = % .12e   rel.err %.1e %s\n", names[k], l, r, e, ok ? "ok" : "MISMATCH");
        if(!ok) fails++;
    }
    for(int k = 0; k < 6; k++) std::copy(saved[k].begin(), saved[k].end(), tabs[k]);

    // (2) CI vectors: central differences of <W, E2> with respect to ci->coef[0] (coef_bas follows through transpose_ci); fits fixed
    {
        aldet_data * ci = RF.ci;
        const size_t nd = (size_t)ci->Na * ci->Nb;
        std::vector<double> dx(nd * ns), save(ci->coef[0], ci->coef[0] + nd * ns);
        for(double & v : dx) v = rnd();
        double ana = 0; for(size_t i = 0; i < dx.size(); i++) ana += R.cibar[i] * dx[i];
        const double h = 1e-4;
        auto at = [&](double k){ for(size_t i = 0; i < dx.size(); i++) ci->coef[0][i] = save[i] + k * h * dx[i]; ci->transpose_ci(0); return E2_dot(RF, W.data()); };
        const double fd = (-at(2) + 8 * at(1) - 8 * at(-1) + at(-2)) / (12 * h);
        std::copy(save.begin(), save.end(), ci->coef[0]); ci->transpose_ci(0);
        err = std::fabs(ana - fd) / (1e-12 + std::fabs(ana));
        fprintf(out_stream, "XMC contraction adjoint, CI vectors: analytic % .12e  FD % .12e  rel.err %.1e %s\n", ana, fd, err, err < 1e-8 ? "ok" : "MISMATCH");
        if(err >= 1e-8) fails++;
    }
    // (3) E0 and E_act: central differences of <W, E2> with the fits (stencils, weights, grid) regenerated from the perturbed values; the tables are the same numbers.
    // The interpolation is smooth only while every evaluation point keeps its stencil and the grid its end points: the step shrinks until no
    // stencil (S, L) and no grid branch changes over the four displaced points.
    {
        aldet_data * ci = RF.ci;
        const size_t nd = (size_t)ci->Na * ci->Nb;
        std::vector<double> dE0(ns), dEa(nd), E0s(RF.E0, RF.E0 + ns), Eas(ci->E_act[0], ci->E_act[0] + nd);
        for(double & v : dE0) v = rnd();
        for(double & v : dEa) v = rnd();
        double ana = 0; for(int i = 0; i < ns; i++) ana += R.e0bar[i] * dE0[i]; for(size_t i = 0; i < nd; i++) ana += R.Eactbar[i] * dEa[i];
        std::vector<double> tabsave[6];
        for(int k = 0; k < 6; k++) tabsave[k].assign(tabs[k], tabs[k] + sizes[k]);
        const std::vector<int> S0(RF.S_fit, RF.S_fit + nd * ns), L0(RF.L_fit, RF.L_fit + nd * ns);
        const RFGridBranch b0 = rf_grid_branch(RF.E0, ns, ci->E_act[0], (long)nd);
        long switches = 0;
        auto at = [&](double k, double h){
            for(int i = 0; i < ns; i++) RF.E0[i] = E0s[i] + k * h * dE0[i];
            for(size_t i = 0; i < nd; i++) ci->E_act[0][i] = Eas[i] + k * h * dEa[i];
            RF.gen_grid(Nf, n_fit_pol);                      // regenerates the grid and the fits (and reallocates the tables: refill them)
            double * tt[6] = {RF.RF_PS, RF.RF_PH, RF.RF_PV_JK, RF.RF_PV_AB, RF.RF_P3_JK, RF.RF_P3_AB};
            for(int j = 0; j < 6; j++) std::copy(tabsave[j].begin(), tabsave[j].end(), tt[j]);
            for(size_t i = 0; i < nd * ns; i++) switches += (RF.S_fit[i] != S0[i] || RF.L_fit[i] != L0[i]);
            const RFGridBranch b = rf_grid_branch(RF.E0, ns, ci->E_act[0], (long)nd);
            switches += (b.a_min != b0.a_min || b.a_max != b0.a_max || b.I_min != b0.I_min || b.I_max != b0.I_max);
            return E2_dot(RF, W.data());
        };
        double h = 1e-5, fd = 0;
        for(; ; h *= 0.1){
            switches = 0;
            fd = (-at(2, h) + 8 * at(1, h) - 8 * at(-1, h) + at(-2, h)) / (12 * h);
            if(switches == 0 || h < 1e-8) break;
            fprintf(out_stream, "   E0 and E_act, step %.0e: %ld stencil or grid switches over the displaced points, smaller step\n", h, switches);
        }
        // back to the unperturbed fits and tables
        std::copy(E0s.begin(), E0s.end(), RF.E0); std::copy(Eas.begin(), Eas.end(), ci->E_act[0]);
        RF.gen_grid(Nf, n_fit_pol);
        { double * tt[6] = {RF.RF_PS, RF.RF_PH, RF.RF_PV_JK, RF.RF_PV_AB, RF.RF_P3_JK, RF.RF_P3_AB};
          for(int j = 0; j < 6; j++) std::copy(tabsave[j].begin(), tabsave[j].end(), tt[j]); }
        err = std::fabs(ana - fd) / (1e-12 + std::fabs(ana));
        const bool ok = err < 1e-6 && switches == 0;
        fprintf(out_stream, "XMC contraction adjoint, E0 and E_act (weights and grid regenerated, step %.0e%s): analytic % .12e  FD % .12e  rel.err %.1e %s\n",
                h, switches ? ", stencils still switch" : "", ana, fd, err, ok ? "ok" : "MISMATCH");
        if(!ok) fails++;
    }
    return fails;
}
