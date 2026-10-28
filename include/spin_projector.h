#ifndef spin_projector_h
#define spin_projector_h
# include "aldet.h"
# include "from_hash.h"
# include <algorithm>
# include <cmath>
# include <cstdlib>
# include <vector>

// Projection onto the alpha-beta exchange parity of the multiplicity (n_alp = n_bet, Na = Nb): C[b,a] = sym_ab C[a,b] for every
// column, sym_ab = (-1)^((mult-1)/2) as in aldet_data. H is spin-free, so this keeps the Davidson space in one spin parity: a
// singlet run has no triplets. States with S differing by 2 (quintets in a singlet run) are removed by spin_projector below.
inline void spin_parity_project(double * c, int N, int ncol, int ld, double sym_ab){
    for(int a = 0; a < N; a++){
        for(int k = 0; k < ncol; k++){
            double & d = c[((size_t)a * N + a) * ld + k];
            if(sym_ab < 0) d = 0.0;
        }
        for(int b = a + 1; b < N; b++)
            for(int k = 0; k < ncol; k++){
                double & u = c[((size_t)a * N + b) * ld + k];
                double & l = c[((size_t)b * N + a) * ld + k];
                const double v = 0.5 * (u + sym_ab * l);
                u = v; l = sym_ab * v;
            }
    }
}

// Number of CSFs (spin eigenfunctions) with spin S = two_s/2 for N electrons in n orbitals (Weyl-Paldus formula); every
// determinant space with |Ms| <= S holds exactly this many states of that spin.
inline double spin_n_csf(int n, int N, int two_s){
    if(two_s < 0 || (N - two_s) % 2) return 0;
    auto binom = [](int a, int b){ return (b < 0 || b > a) ? 0.0 : std::round(std::tgamma(a + 1.0) / std::tgamma(b + 1.0) / std::tgamma(a - b + 1.0)); };
    return std::round((two_s + 1.0) / (n + 1.0) * binom(n + 1, (N - two_s) / 2) * binom(n + 1, (N + two_s) / 2 + 1));
}

// Full spin projection (dav_par::spin_proj) in the determinant space of V, Ms = (na-nb)/2, onto S = (mult-1)/2.
// S^2 = S+ S- + Sz^2 - Sz, S- = sum_t b+_t a_t maps into the (na-1, nb+1) string space with the signs of
// aldet_data::calc_spin_matr, S+ is its transpose. The Lowdin projector
//     P_S = prod_{S' != S} (S^2 - S'(S'+1)) / (S(S+1) - S'(S'+1))
// runs over the spins present, |Ms| <= S' <= min(na+nb, 2 n_act-na-nb)/2. For na = nb the alpha-beta parity projection
// goes first and removes every S' of the other parity, so only S' = S +- 2, +- 4, ... remain for the product.
class spin_projector{
    public:
        int ok = 0;          // the target spin is possible in this space
        int parity = 0;      // na = nb: alpha-beta exchange parity first
        double sym_ab = 0;
        double S = 0, ss = 0;
        double n_states = 0;  // states of spin S in the space
        std::vector<double> remove; // S'(S'+1) removed by the product

        // use_parity = 0 skips the parity step for na = nb (the product then runs over all S'; for tests)
        spin_projector(aldet_data & V, int use_parity = 1){
            n_act = V.n_act; na = V.na; nb = V.nb; Na = V.Na; Nb = V.Nb;
            a_sign = V.a_spin_sign; b_sign = V.b_spin_sign;
            sz2 = (na - nb) / 2.0; sz2 = sz2 * sz2 - sz2;
            S = (V.mult - 1) / 2.0; ss = S * (S + 1);
            const int two_ms = abs(na - nb), two_smax = std::min(na + nb, 2 * n_act - na - nb);
            ok = V.mult >= 1 && V.mult - 1 >= two_ms && V.mult - 1 <= two_smax && (V.mult - 1 - two_ms) % 2 == 0;
            if(!ok) return;
            n_states = spin_n_csf(n_act, na + nb, V.mult - 1);
            parity = use_parity && (na == nb && Na == Nb);
            sym_ab = V.sym_ab;
            for(int ts = two_ms; ts <= two_smax; ts += 2){
                if(ts == V.mult - 1) continue;
                if(parity && ((ts - V.mult + 1) / 2) % 2) continue;
                remove.push_back(ts / 2.0 * (ts / 2.0 + 1));
            }
            down = (na > 0 && nb < n_act);
            if(down) build_tables(V);
        }

        // out = S^2 in for k columns; in has row stride ld, out is dense (row stride k)
        void S2(double * out, const double * in, int k, int ld){
            const int n = n_act;
            for(ci_int i = 0; i < Na; i++)
                for(ci_int j = 0; j < Nb; j++)
                    for(int c = 0; c < k; c++)
                        out[(i * Nb + j) * k + c] = sz2 * in[(i * Nb + j) * ld + c];
            if(!down) return;
            M.assign((size_t)NaM * NbM * k, 0.0);
            // M = S- in, gathered per (na-1)-string: t not in a', t in b'
#pragma omp parallel for schedule(dynamic, 16)
            for(ci_int iM = 0; iM < NaM; iM++)
                for(int t = 0; t < n; t++){
                    const ci_int i = aM_up[iM * n + t];
                    if(i < 0) continue;
                    const double sa = a_sign[i * n + t];
                    for(ci_int jM = 0; jM < NbM; jM++){
                        const ci_int j = bM_dn[jM * n + t];
                        if(j < 0) continue;
                        const double s = sa * b_sign[j * n + t];
                        double * m = &M[(iM * NbM + jM) * k];
                        const double * v = in + (i * Nb + j) * ld;
                        for(int c = 0; c < k; c++) m[c] += s * v[c];
                    }
                }
            // out += S+ M, gathered per na-string: t in a, t not in b
#pragma omp parallel for schedule(dynamic, 16)
            for(ci_int i = 0; i < Na; i++)
                for(int t = 0; t < n; t++){
                    const ci_int iM = a_dn[i * n + t];
                    if(iM < 0) continue;
                    const double sa = a_sign[i * n + t];
                    for(ci_int j = 0; j < Nb; j++){
                        const ci_int jM = b_up[j * n + t];
                        if(jM < 0) continue;
                        const double s = sa * b_sign[j * n + t];
                        double * o = out + (i * Nb + j) * k;
                        const double * m = &M[(iM * NbM + jM) * k];
                        for(int c = 0; c < k; c++) o[c] += s * m[c];
                    }
                }
        }

        // project ncol columns of c (row stride ld) in place
        void project(double * c, int ncol, int ld){
            if(!ok) return;
            if(parity) spin_parity_project(c, Na, ncol, ld, sym_ab);
            if(remove.empty()) return;
            const ci_int Nd = Na * Nb;
            for(int c0 = 0; c0 < ncol; c0 += KC){
                const int k = std::min(KC, ncol - c0);
                Y.resize((size_t)Nd * k);
                for(double sp : remove){
                    S2(Y.data(), c + c0, k, ld);
                    const double f = 1.0 / (ss - sp);
                    for(ci_int d = 0; d < Nd; d++)
                        for(int q = 0; q < k; q++){
                            double & v = c[d * ld + c0 + q];
                            v = (Y[d * k + q] - sp * v) * f;
                        }
                }
            }
        }

        // max over columns of |(S^2 - S(S+1)) c| for unit columns
        double residual(const double * c, int ncol, int ld){
            if(!ok) return 0;
            const ci_int Nd = Na * Nb;
            Y.resize((size_t)Nd * ncol);
            S2(Y.data(), c, ncol, ld);
            double r = 0;
            for(int q = 0; q < ncol; q++){
                double s = 0, nn = 0;
                for(ci_int d = 0; d < Nd; d++){
                    const double v = c[d * ld + q], e = Y[d * ncol + q] - ss * v;
                    s += e * e; nn += v * v;
                }
                if(nn > 0) r = std::max(r, sqrt(s / nn));
            }
            return r;
        }

    private:
        static const int KC = 4;
        int n_act, na, nb, down = 0;
        ci_int Na, Nb, NaM = 0, NbM = 0;
        double sz2;
        const double * a_sign, * b_sign;
        std::vector<ci_int> a_dn, b_up, aM_up, bM_dn; // string tables, -1 where the replacement does not exist
        std::vector<double> M, Y;

        void build_tables(aldet_data & V){
            const int n = n_act;
            NaM = (ci_int) std::lround(tgammal(n + 1) / tgammal(na) / tgammal(n - na + 2));
            NbM = (ci_int) std::lround(tgammal(n + 1) / tgammal(nb + 2) / tgammal(n - nb));
            std::vector<int> faM((size_t)(na - 1) * n + 1), fbM((size_t)(nb + 1) * n + 1), bit(n), buf(n + 2, 0);
            get_factorials(na - 1, n, faM.data());
            get_factorials(nb + 1, n, fbM.data());
            a_dn.assign((size_t)Na * n, -1); aM_up.assign((size_t)NaM * n, -1);
            b_up.assign((size_t)Nb * n, -1); bM_dn.assign((size_t)NbM * n, -1);
            for(ci_int i = 0; i < Na; i++){
                std::fill(bit.begin(), bit.end(), 0);
                for(int q = 1; q <= na; q++) bit[V.vec_a[i * (na + 1) + q] - 1] = 1;
                for(int t = 0; t < n; t++) if(bit[t]){
                    bit[t] = 0;
                    const ci_int iM = get_ind_from_ON(bit.data(), n, na - 1, faM.data(), buf.data());
                    bit[t] = 1;
                    a_dn[i * n + t] = iM; aM_up[iM * n + t] = i;
                }
            }
            for(ci_int j = 0; j < Nb; j++){
                std::fill(bit.begin(), bit.end(), 0);
                for(int q = 1; q <= nb; q++) bit[V.vec_b[j * (nb + 1) + q] - 1] = 1;
                for(int t = 0; t < n; t++) if(!bit[t]){
                    bit[t] = 1;
                    const ci_int jM = get_ind_from_ON(bit.data(), n, nb + 1, fbM.data(), buf.data());
                    bit[t] = 0;
                    b_up[j * n + t] = jM; bM_dn[jM * n + t] = j;
                }
            }
        }
};

#endif
