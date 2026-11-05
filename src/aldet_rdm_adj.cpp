#include "aldet_rdm_adj.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <omp.h>

#include "aldet.h"
#include "aldet_casci_wrap.h"
#include "casscf_resp.h"
#include "common_vars.h"
#include "from_hash.h"

// Adjoint of aldet_calc_DM_2body_AA_diag(G, n_s, n_s, ci, N, na, Na, Nb, fa, vec_a, 0, 1): Gb (n_s x N^4, state-major) -> cb (layout of ci), +=.
// The forward loops are copied; R[s] += CI_0[s] CI_e[s] sign becomes cb_0 += Rb sign CI_e, cb_e += Rb sign CI_0.
void dm2_AA_diag_vjp(const double * Gb, int n_s, const double * ci, double * cb, int N, int na, int Na, int Nb, int * fa, int * vec_a, int i_th, int n_th){
    const size_t N4 = (size_t)N * N * N * N;
    std::vector<double> Bb(N4 * n_s, 0.0);
    // post-processing: G[i,j,k,l] -= B[i,k,j,l], G[k,j,i,l] += B, G[i,l,k,j] += B, G[k,l,i,j] -= B   (k >= i, l >= j)
    for(int s = 0; s < n_s; s++){
        const double * g = Gb + s * N4;
        for(int i = 0; i < N; i++) for(int j = 0; j < N; j++) for(int k = i; k < N; k++) for(int l = j; l < N; l++)
            Bb[((((size_t)i * N + k) * N + j) * N + l) * n_s + s] += -g[(((size_t)i * N + j) * N + k) * N + l] + g[(((size_t)k * N + j) * N + i) * N + l]
                                                                    + g[(((size_t)i * N + l) * N + k) * N + j] - g[(((size_t)k * N + l) * N + i) * N + j];
    }
    std::vector<int> bit_a(N), buf(na + 1); buf[0] = 0;
    double sign, sign2, sign3, sign4;
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th){
        std::fill(bit_a.begin(), bit_a.end(), 0);
        for(int k = 1; k < na + 1; k++) bit_a[vec_a[i_CI * (na + 1) + k] - 1] = 1;
        for(int t = 0; t < N; t++) if(bit_a[t] != 0){
            sign2 = 1;
            bit_a[t] = 0;
            for(int u = t; u < N; u++){
                if(bit_a[u] != 0){
                    sign3 = 1;
                    bit_a[u] = 0;
                    for(int v = 0; v < N; v++){
                        if(bit_a[v] == 0){
                            sign4 = 1;
                            bit_a[v] = 1;
                            for(int w = v; w < N; w++){
                                if(bit_a[w] == 0){
                                    bit_a[w] = 1;
                                    sign = sign2 * sign4;
                                    const int i_CIext = get_ind_from_ON(bit_a.data(), N, na, fa, buf.data());
                                    const double * Rb = Bb.data() + ((((size_t)t * N + u) * N + v) * N + w) * n_s;
                                    for(int j_CI = 0; j_CI < Nb; j_CI++){
                                        const double * CI_0 = ci + ((size_t)i_CI * Nb + j_CI) * n_s, * CI_e = ci + ((size_t)i_CIext * Nb + j_CI) * n_s;
                                        double * cb_0 = cb + ((size_t)i_CI * Nb + j_CI) * n_s, * cb_e = cb + ((size_t)i_CIext * Nb + j_CI) * n_s;
                                        for(int s = 0; s < n_s; s++){ cb_0[s] += Rb[s] * sign * CI_e[s]; cb_e[s] += Rb[s] * sign * CI_0[s]; }
                                    }
                                    bit_a[w] = 0;
                                }
                                else sign4 = -sign4;
                            }
                            bit_a[v] = 0;
                        }
                        else sign3 = -sign3;
                    }
                    bit_a[u] = 1;
                    sign2 = -sign2;
                }
            }
            bit_a[t] = 1;
        }
    }
}

// Adjoint of aldet_calc_DM_2body_AB_diag(G, n_s, n_s, ci, N, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1) (loops copied).
void dm2_AB_diag_vjp(const double * Gb, int n_s, const double * ci, double * cb, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b, int i_th, int n_th){
    const size_t N4 = (size_t)N * N * N * N;
    std::vector<double> Bb(N4 * n_s);
    for(int s = 0; s < n_s; s++) for(size_t i = 0; i < N4; i++) Bb[i * n_s + s] = Gb[s * N4 + i];      // G[s*N4+i] += BUF[i*n_s+s]
    const int Na_zv = (int)std::lround(tgammal(N) / tgammal(na) / tgammal(N - na + 1));
    const int Nb_zv = (int)std::lround(tgammal(N) / tgammal(nb) / tgammal(N - nb + 1));
    std::vector<int> Ia(Na_zv), Ia_friends((size_t)Na_zv * (N - na) * 3 + 1), Ib(Nb_zv), Ib_friends((size_t)Nb_zv * (N - nb) * 3 + 1);
    auto B = [&](int p, int q, int r, int s){ return Bb.data() + ((((size_t)p * N + q) * N + r) * N + s) * n_s; };
    for(int t = 0; t < N; t++){
        get_I_I_friends(na, N, Na, fa, t, vec_a, Ia.data(), Ia_friends.data());
        for(int u = 0; u < N; u++){
            get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib.data(), Ib_friends.data());
            for(int i_CI = i_th; i_CI < Na_zv; i_CI += n_th){
                const int k_CI = Ia[i_CI];
                for(int j_CI = 0; j_CI < Nb_zv; j_CI++){
                    const int l_CI = Ib[j_CI];
                    const double * CI_0 = ci + ((size_t)k_CI * Nb + l_CI) * n_s;
                    double * cb_0 = cb + ((size_t)k_CI * Nb + l_CI) * n_s;
                    {
                        const double * RA = B(t, t, u, u), * RB = B(u, u, t, t);
                        for(int s = 0; s < n_s; s++) cb_0[s] += 2.0 * (RA[s] + RB[s]) * CI_0[s];
                    }
                    for(int n = 0; n < (N - nb); n++){
                        const int * ind_n = Ib_friends.data() + (size_t)j_CI * (N - nb) * 3 + n * 3;
                        const size_t e = (size_t)k_CI * Nb + ind_n[0];
                        const double * RA = B(t, t, u, ind_n[1]), * RB = B(u, ind_n[1], t, t);
                        for(int s = 0; s < n_s; s++){ const double r = (RA[s] + RB[s]) * ind_n[2]; cb_0[s] += r * ci[e * n_s + s]; cb[e * n_s + s] += r * CI_0[s]; }
                    }
                    for(int m = 0; m < (N - na); m++){
                        const int * ind_m = Ia_friends.data() + (size_t)i_CI * (N - na) * 3 + m * 3;
                        {
                            const size_t e = (size_t)ind_m[0] * Nb + l_CI;
                            const double * RA = B(t, ind_m[1], u, u), * RB = B(u, u, t, ind_m[1]);
                            for(int s = 0; s < n_s; s++){ const double r = (RA[s] + RB[s]) * ind_m[2]; cb_0[s] += r * ci[e * n_s + s]; cb[e * n_s + s] += r * CI_0[s]; }
                        }
                        for(int n = 0; n < (N - nb); n++){
                            const int * ind_n = Ib_friends.data() + (size_t)j_CI * (N - nb) * 3 + n * 3;
                            const size_t e = (size_t)ind_m[0] * Nb + ind_n[0];
                            const double * RA = B(t, ind_m[1], u, ind_n[1]), * RB = B(u, ind_n[1], t, ind_m[1]);
                            const double sign = ind_m[2] * ind_n[2];
                            for(int s = 0; s < n_s; s++){ const double r = (RA[s] + RB[s]) * sign; cb_0[s] += r * ci[e * n_s + s]; cb[e * n_s + s] += r * CI_0[s]; }
                        }
                    }
                }
            }
        }
    }
}


// Adjoint of aldet_calc_DM_3body_AAA(G, 1, ld, ci, ...) (G3_calc_diag: one state, element 0 of each coefficient block of stride ld).
void dm3_AAA_vjp(const double * Gb, int ld, const double * ci, double * cb, int N, int na, int Na, int Nb, int * fa, int * vec_a, int i_th, int n_th){
    std::vector<int> bit_a(N), buf(na + 1); buf[0] = 0;
    double sign, sign1, sign2, sign3, sign4, sign5, sign6;
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th){
        std::fill(bit_a.begin(), bit_a.end(), 0);
        for(int k = 1; k < na + 1; k++) bit_a[vec_a[i_CI * (na + 1) + k] - 1] = 1;
        sign1 = 1;
        for(int t = 0; t < N; t++) if(bit_a[t] != 0){
            sign2 = 1;
            bit_a[t] = 0;
            for(int u = 0; u < N; u++){
                if(bit_a[u] != 0){
                    sign3 = 1;
                    bit_a[u] = 0;
                    for(int v = 0; v < N; v++){
                        if(bit_a[v] != 0){
                            sign4 = 1;
                            bit_a[v] = 0;
                            for(int w = 0; w < N; w++){
                                if(bit_a[w] == 0){
                                    sign5 = 1;
                                    bit_a[w] = 1;
                                    for(int x = 0; x < N; x++){
                                        if(bit_a[x] == 0){
                                            sign6 = 1;
                                            bit_a[x] = 1;
                                            for(int y = 0; y < N; y++){
                                                if(bit_a[y] == 0){
                                                    bit_a[y] = 1;
                                                    sign = sign1 * sign2 * sign3 * sign4 * sign5 * sign6;
                                                    const int i_CIext = get_ind_from_ON(bit_a.data(), N, na, fa, buf.data());
                                                    const double r = Gb[(((((size_t)t * N + u) * N + v) * N + w) * N + x) * N + y] * sign;
                                                    if(r != 0.0)
                                                    for(int j_CI = 0; j_CI < Nb; j_CI++){
                                                        const size_t o0 = ((size_t)i_CI * Nb + j_CI) * ld, oe = ((size_t)i_CIext * Nb + j_CI) * ld;
                                                        cb[o0] += r * ci[oe]; cb[oe] += r * ci[o0];
                                                    }
                                                    bit_a[y] = 0;
                                                }
                                                else sign6 = -sign6;
                                            }
                                            bit_a[x] = 0;
                                        }
                                        else sign5 = -sign5;
                                    }
                                    bit_a[w] = 0;
                                }
                                else sign4 = -sign4;
                            }
                            bit_a[v] = 1;
                            sign3 = -sign3;
                        }
                    }
                    bit_a[u] = 1;
                    sign2 = -sign2;
                }
            }
            bit_a[t] = 1;
            sign1 = -sign1;
        }
    }
}

// Adjoint of aldet_calc_DM_3body_AAB(G, 1, ld, ci, ...).
void dm3_AAB_vjp(const double * Gb, int ld, const double * ci, double * cb, int N, int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b, int i_th, int n_th){
    std::vector<int> bit_a(N), bit_b(N), buf(std::max(na + 1, nb + 1) + 1); buf[0] = 0;
    double sign, sign1, sign2, sign3, sign4, sign5, sign6;
    for(int i_CI = i_th; i_CI < Na; i_CI += n_th)
    for(int j_CI = 0; j_CI < Nb; j_CI++){
        std::fill(bit_a.begin(), bit_a.end(), 0);
        for(int k = 1; k < na + 1; k++) bit_a[vec_a[i_CI * (na + 1) + k] - 1] = 1;
        std::fill(bit_b.begin(), bit_b.end(), 0);
        for(int k = 1; k < nb + 1; k++) bit_b[vec_b[j_CI * (nb + 1) + k] - 1] = 1;
        sign1 = 1;
        for(int t = 0; t < N; t++) if(bit_a[t] != 0){
            sign2 = 1;
            bit_a[t] = 0;
            for(int u = 0; u < N; u++){
                if(bit_a[u] != 0){
                    sign3 = 1;
                    bit_a[u] = 0;
                    for(int v = 0; v < N; v++){
                        if(bit_b[v] != 0){
                            sign4 = 1;
                            bit_b[v] = 0;
                            for(int w = 0; w < N; w++){
                                if(bit_a[w] == 0){
                                    sign5 = 1;
                                    bit_a[w] = 1;
                                    for(int x = 0; x < N; x++){
                                        if(bit_a[x] == 0){
                                            sign6 = 1;
                                            bit_a[x] = 1;
                                            for(int y = 0; y < N; y++){
                                                if(bit_b[y] == 0){
                                                    bit_b[y] = 1;
                                                    sign = sign1 * sign2 * sign3 * sign4 * sign5 * sign6;
                                                    const int i_CIext = get_ind_from_ON(bit_a.data(), N, na, fa, buf.data());
                                                    const int j_CIext = get_ind_from_ON(bit_b.data(), N, nb, fb, buf.data());
                                                    const double r = Gb[(((((size_t)t * N + u) * N + v) * N + w) * N + x) * N + y] * sign;
                                                    if(r != 0.0){
                                                        const size_t o0 = ((size_t)i_CI * Nb + j_CI) * ld, oe = ((size_t)i_CIext * Nb + j_CIext) * ld;
                                                        cb[o0] += r * ci[oe]; cb[oe] += r * ci[o0];
                                                    }
                                                    bit_b[y] = 0;
                                                }
                                                else sign6 = -sign6;
                                            }
                                            bit_a[x] = 0;
                                        }
                                        else sign5 = -sign5;
                                    }
                                    bit_a[w] = 0;
                                }
                                else sign4 = -sign4;
                            }
                            bit_b[v] = 1;
                            sign3 = -sign3;
                        }
                    }
                    bit_a[u] = 1;
                    sign2 = -sign2;
                }
            }
            bit_a[t] = 1;
            sign1 = -sign1;
        }
    }
}

namespace{
void setup_aldet(const CasRef & R, aldet_data & X, int nvec, const double * c){
    omp_set_num_threads(num_threads);
    X.get_dim(R.ci_template->n_act, R.ci_template->na, R.ci_template->nb, 1, R.ci_template->mult, 0);
    X.init_zero_vec(nvec, 0);
    for(int s = 0; s < nvec; s++) for(size_t d = 0; d < R.Nd; d++) X.coef[0][d * nvec + s] = c[(size_t)s * R.Nd + d];
    X.transpose_ci(0);
}
inline size_t i6(int n, int a, int b, int c, int d, int e, int f){ return (((((size_t)a * n + b) * n + c) * n + d) * n + e) * n + f; }
}

void aldet_rdm_vjp(const CasRef & R, const double * c, int nvec, const double * w, const double * g1b, const double * G2b, const double * G3b, double * cb){
    const int N = R.na; const size_t Nd = R.Nd, a4 = (size_t)N * N * N * N, a6 = a4 * N * N;
    if(g1b){
        std::vector<double> Hc((size_t)nvec * Nd);
        cas_onebody_apply(R, g1b, c, nvec, Hc.data());
        for(int I = 0; I < nvec; I++) for(size_t d = 0; d < Nd; d++) cb[(size_t)I * Nd + d] += 2.0 * w[I] * Hc[(size_t)I * Nd + d];
    }
    if(!G2b && !G3b) return;
    aldet_data X;
    setup_aldet(R, X, nvec, c);
    const int Na = (int)X.Na, Nb = (int)X.Nb;
    std::vector<double> cba((size_t)Na * Nb * nvec, 0.0), cbb((size_t)Na * Nb * nvec, 0.0);   // coef layout, coef_bas layout
    // the loops are split over alpha (or beta) strings like the forward workers; each thread accumulates into its own copy, summed in thread order
    const int nth = std::max(1, num_threads);
    auto par = [&](const std::function<void(double * a, double * b, int t, int n)> & job){
        std::vector<std::vector<double>> ta(nth, std::vector<double>(cba.size(), 0.0)), tb(nth, std::vector<double>(cbb.size(), 0.0));
        #pragma omp parallel num_threads(nth)
        { const int t = omp_get_thread_num(); job(ta[t].data(), tb[t].data(), t, nth); }
        for(int t = 0; t < nth; t++) for(size_t k = 0; k < cba.size(); k++){ cba[k] += ta[t][k]; cbb[k] += tb[t][k]; }
    };
    if(G2b){
        std::vector<double> Gs((size_t)nvec * a4);
        for(int I = 0; I < nvec; I++) for(size_t k = 0; k < a4; k++) Gs[I * a4 + k] = w[I] * G2b[k];
        par([&](double * a, double * b, int t, int nt){
            dm2_AA_diag_vjp(Gs.data(), nvec, X.coef[0], a, N, X.na, Na, Nb, X.fa, X.vec_a, t, nt);
            dm2_AA_diag_vjp(Gs.data(), nvec, X.coef_bas[0], b, N, X.nb, Nb, Na, X.fb, X.vec_b, t, nt);
            dm2_AB_diag_vjp(Gs.data(), nvec, X.coef[0], a, N, X.na, X.nb, Na, Nb, X.fa, X.fb, X.vec_a, X.vec_b, t, nt);
        });
    }
    if(G3b){
        // G3[p,q,r,i,j,k] = A3[i,j,k,r,q,p] + B3[j,k,i,r,q,p] + B3[i,k,j,r,p,q] + B3[j,i,k,p,q,r]   (aldet_casci_wrap::G3_calc_diag)
        std::vector<double> A3b(a6, 0.0), B3b(a6, 0.0);
        for(int p = 0; p < N; p++) for(int q = 0; q < N; q++) for(int r = 0; r < N; r++)
        for(int i = 0; i < N; i++) for(int j = 0; j < N; j++) for(int k = 0; k < N; k++){
            const double g = G3b[i6(N, p, q, r, i, j, k)];
            A3b[i6(N, i, j, k, r, q, p)] += g;
            B3b[i6(N, j, k, i, r, q, p)] += g; B3b[i6(N, i, k, j, r, p, q)] += g; B3b[i6(N, j, i, k, p, q, r)] += g;
        }
        for(int I = 0; I < nvec; I++){
            std::vector<double> A(a6), B(a6);
            for(size_t k = 0; k < a6; k++){ A[k] = w[I] * A3b[k]; B[k] = w[I] * B3b[k]; }
            par([&](double * a, double * b, int t, int nt){
                dm3_AAA_vjp(A.data(), nvec, X.coef[0] + I, a + I, N, X.na, Na, Nb, X.fa, X.vec_a, t, nt);
                dm3_AAA_vjp(A.data(), nvec, X.coef_bas[0] + I, b + I, N, X.nb, Nb, Na, X.fb, X.vec_b, t, nt);
                dm3_AAB_vjp(B.data(), nvec, X.coef[0] + I, a + I, N, X.na, X.nb, Na, Nb, X.fa, X.fb, X.vec_a, X.vec_b, t, nt);
                dm3_AAB_vjp(B.data(), nvec, X.coef_bas[0] + I, b + I, N, X.nb, X.na, Nb, Na, X.fb, X.fa, X.vec_b, X.vec_a, t, nt);
            });
        }
    }
    for(int ia = 0; ia < Na; ia++) for(int ib = 0; ib < Nb; ib++){
        const size_t d = (size_t)ia * Nb + ib, db = (size_t)ib * Na + ia;
        for(int I = 0; I < nvec; I++) cb[(size_t)I * Nd + d] += cba[d * nvec + I] + cbb[db * nvec + I];
    }
}

int aldet_rdm_vjp_selftest(const CasRef & R, const double * c){
    const int N = R.na; const size_t Nd = R.Nd, a2 = (size_t)N * N, a4 = a2 * a2, a6 = a4 * a2;
    std::mt19937_64 gen(4242);
    auto rnd = [&](){ return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); };
    std::vector<double> g1(a2), G2(a4), G3(a6);
    for(double & v : g1) v = rnd(); for(double & v : G2) v = rnd(); for(double & v : G3) v = rnd();
    const double one = 1.0;
    // functional of one vector through the production routines
    auto phi = [&](const std::vector<double> & v, int which){
        aldet_data X;
        setup_aldet(R, X, 1, v.data());
        double f = 0;
        if(which == 0){ std::vector<double> g(a2, 0.0); X.calc_DM_diag(g.data(), 0); for(size_t k = 0; k < a2; k++) f += g1[k] * g[k]; }
        if(which == 1){ std::vector<double> g(a4, 0.0); X.G_calc(g.data()); for(size_t k = 0; k < a4; k++) f += G2[k] * g[k]; }
        if(which == 2){ aldet_casci_wrap W(&X, nullptr, 1); std::vector<double> g(a6, 0.0); W.G3_calc_diag(g.data(), 0); for(size_t k = 0; k < a6; k++) f += G3[k] * g[k]; }
        return f;
    };
    int fails = 0;
    const char * names[3] = {"1-RDM", "2-RDM", "3-RDM"};
    for(int which = 0; which < 3; which++){
        std::vector<double> cb(Nd, 0.0);
        aldet_rdm_vjp(R, c, 1, &one, which == 0 ? g1.data() : nullptr, which == 1 ? G2.data() : nullptr, which == 2 ? G3.data() : nullptr, cb.data());
        std::vector<double> dc(Nd); for(double & v : dc) v = rnd();
        double ana = 0; for(size_t d = 0; d < Nd; d++) ana += cb[d] * dc[d];
        const double h = 1e-3;
        std::vector<double> cp(c, c + Nd), cm(c, c + Nd);
        for(size_t d = 0; d < Nd; d++){ cp[d] += h * dc[d]; cm[d] -= h * dc[d]; }
        const double fd = (phi(cp, which) - phi(cm, which)) / (2 * h);    // quadratic in c: exact up to rounding
        const double err = std::fabs(ana - fd) / std::max(1e-12, std::max(std::fabs(ana), std::fabs(fd)));
        const bool ok = err < 1e-9;
        if(!ok) fails++;
        fprintf(out_stream, "   CI adjoint of the %s: analytic % .12e  FD % .12e  rel.err %.1e  %s\n", names[which], ana, fd, err, ok ? "ok" : "MISMATCH");
    }
    return fails;
}
