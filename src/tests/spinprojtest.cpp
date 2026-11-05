// Algebraic checks of the spin projector of the determinant CI (include/spin_projector.h), no integrals needed.
//   make run_spinprojtest && ./run_spinprojtest [max_n_act=7] [big_n_act=10]
// For every active space n_act <= max_n_act, every n_alp >= n_bet and every S that the space holds:
//   - P_S is symmetric and idempotent, and (S^2 - S(S+1)) P_S = 0;
//   - trace P_S is the number of CSFs (Weyl-Paldus formula);
//   - for n_alp = n_bet, the parity step changes nothing: P_S with it equals the plain Lowdin product;
//   - sum over S of P_S is the identity;
//   - <v|S^2|v> of the projector's S^2 equals aldet_data::calc_spin_matr.
// The big_n_act spaces (half filling and one off it) repeat the vector checks on random vectors.
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <random>
# include <vector>
# include "blas_link.h"
# include "common_vars.h"
# include "spin_projector.h"

static int n_fail = 0, n_check = 0;
static void check(bool ok, const char * what, int n, int na, int nb, int mult, double val){
    n_check++;
    if(ok) return;
    n_fail++;
    printf("FAIL  n_act=%d n_alp=%d n_bet=%d mult=%d: %s (%.3e)\n", n, na, nb, mult, what, val);
}

static void random_fill(std::vector<double> & v, std::mt19937 & g){
    std::uniform_real_distribution<double> u(-1, 1);
    for(auto & x : v) x = u(g);
}

// checks on k random vectors, for spaces too big for the full matrix
static void vector_checks(int n, int na, int nb, std::mt19937 & g, double & worst){
    const int two_ms = na - nb, two_smax = std::min(na + nb, 2 * n - na - nb);
    const int k = 3;
    aldet_data V; V.get_dim(n, na, nb, 1, two_ms + 1, 4);
    const size_t Nd = V.Nd;
    std::vector<double> v(Nd * k), sum(Nd * k, 0.0), w, y(Nd * k);
    random_fill(v, g);
    for(int two_s = two_ms; two_s <= two_smax; two_s += 2){
        aldet_data W; W.get_dim(n, na, nb, 1, two_s + 1, 4);
        spin_projector P(W);
        w = v; P.project(w.data(), k, k);
        for(size_t i = 0; i < Nd * k; i++) sum[i] += w[i];
        const double res = P.residual(w.data(), k, k);
        check(res < 1e-10, "|(S^2 - S(S+1)) P v| / |P v|", n, na, nb, two_s + 1, res);
        y = w; P.project(y.data(), k, k);
        double d = 0, nn = 0;
        for(size_t i = 0; i < Nd * k; i++){ d = std::max(d, fabs(y[i] - w[i])); nn = std::max(nn, fabs(w[i])); }
        check(d < 1e-10 * std::max(1.0, nn), "P P v - P v", n, na, nb, two_s + 1, d);
        worst = std::max(worst, std::max(res, d));
    }
    double d = 0;
    for(size_t i = 0; i < Nd * k; i++) d = std::max(d, fabs(sum[i] - v[i]));
    check(d < 1e-10, "sum_S P_S v - v", n, na, nb, 0, d);
    worst = std::max(worst, d);
    printf("  n_act=%2d n_alp=%2d n_bet=%2d  Nd=%8zu  random vectors: max deviation %.1e\n", n, na, nb, Nd, d);
}

int main(int argc, char ** argv){
    const int max_n = argc > 1 ? atoi(argv[1]) : 7;
    const int big_n = argc > 2 ? atoi(argv[2]) : 10;
    FILE * keep = out_stream;
    out_stream = fopen("/dev/null", "w"); // get_dim prints an engine warning for every space
    std::mt19937 g(12345);
    double worst = 0;
    int n_spaces = 0, n_proj = 0;

    for(int n = 1; n <= max_n; n++)
    for(int na = 0; na <= n; na++)
    for(int nb = 0; nb <= na; nb++){
        const int two_ms = na - nb, two_smax = std::min(na + nb, 2 * n - na - nb);
        aldet_data V0; V0.get_dim(n, na, nb, 1, two_ms + 1, 4);
        const size_t Nd = V0.Nd;
        if(Nd > 1500) continue;
        n_spaces++;
        std::vector<double> sum(Nd * Nd, 0.0), I(Nd * Nd), Pm, Q, R(Nd * Nd);
        for(size_t i = 0; i < Nd; i++) for(size_t j = 0; j < Nd; j++) I[i * Nd + j] = (i == j);

        for(int two_s = two_ms; two_s <= two_smax; two_s += 2){
            const int mult = two_s + 1;
            aldet_data V; V.get_dim(n, na, nb, 1, mult, 4);
            spin_projector P(V);
            check(P.ok, "spin not accepted", n, na, nb, mult, 0);
            if(!P.ok) continue;
            n_proj++;
            Pm = I; P.project(Pm.data(), Nd, Nd);

            double tr = 0, asym = 0;
            for(size_t i = 0; i < Nd; i++){
                tr += Pm[i * Nd + i];
                for(size_t j = 0; j < i; j++) asym = std::max(asym, fabs(Pm[i * Nd + j] - Pm[j * Nd + i]));
            }
            const double ncsf = spin_n_csf(n, na + nb, two_s);
            check(fabs(tr - ncsf) < 1e-8, "trace P - n_CSF", n, na, nb, mult, tr - ncsf);
            check(P.n_states == ncsf, "spin_projector::n_states - n_CSF", n, na, nb, mult, P.n_states - ncsf);
            check(asym < 1e-12, "P - P^T", n, na, nb, mult, asym);

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nd, Nd, Nd, 1.0, Pm.data(), Nd, Pm.data(), Nd, 0.0, R.data(), Nd);
            double idem = 0;
            for(size_t i = 0; i < Nd * Nd; i++) idem = std::max(idem, fabs(R[i] - Pm[i]));
            check(idem < 1e-12, "P P - P", n, na, nb, mult, idem);

            // (S^2 - S(S+1)) P = 0, absolute: columns of P can be (numerically) zero
            P.S2(R.data(), Pm.data(), Nd, Nd);
            double res = 0;
            for(size_t i = 0; i < Nd * Nd; i++) res = std::max(res, fabs(R[i] - P.ss * Pm[i]));
            check(res < 1e-10, "(S^2 - S(S+1)) P", n, na, nb, mult, res);

            if(na == nb){
                spin_projector P0(V, 0);
                Q = I; P0.project(Q.data(), Nd, Nd);
                double d = 0;
                for(size_t i = 0; i < Nd * Nd; i++) d = std::max(d, fabs(Q[i] - Pm[i]));
                check(d < 1e-12, "parity + product - product alone", n, na, nb, mult, d);
                worst = std::max(worst, d);
            }
            for(size_t i = 0; i < Nd * Nd; i++) sum[i] += Pm[i];
            worst = std::max(worst, std::max(std::max(asym, idem), std::max(res, fabs(tr - ncsf))));
        }
        double d = 0;
        for(size_t i = 0; i < Nd * Nd; i++) d = std::max(d, fabs(sum[i] - I[i]));
        check(d < 1e-12, "sum_S P_S - 1", n, na, nb, 0, d);
        worst = std::max(worst, d);

        // S^2 of the projector against calc_spin_matr, on random vectors
        if(Nd > 1){
            const int k = 3;
            std::vector<double> v(Nd * k), y(Nd * k), O(k * k), O2(k * k);
            random_fill(v, g);
            spin_projector P(V0);
            V0.calc_spin_matr(O.data(), v.data(), k, k);
            // residual() needs a projector; S2 itself is the quantity compared here
            std::vector<double> s2v(Nd * k);
            P.S2(s2v.data(), v.data(), k, k);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, k, Nd, 1.0, v.data(), k, s2v.data(), k, 0.0, O2.data(), k);
            double e = 0;
            for(int i = 0; i < k * k; i++) e = std::max(e, fabs(O[i] - O2[i]) / (1 + fabs(O[i])));
            check(e < 1e-12, "<v|S^2|w> - calc_spin_matr", n, na, nb, 0, e);
            worst = std::max(worst, e);
        }
    }
    out_stream = keep;
    printf("full matrices: %d active spaces, %d projectors (n_act <= %d, N_det <= 1500)\n", n_spaces, n_proj, max_n);
    out_stream = fopen("/dev/null", "w");
    if(big_n > 0){
        printf("random vectors:\n");
        const int nh = big_n;
        int list[][2] = {{nh / 2, nh / 2}, {nh / 2 + 1, nh / 2}, {nh / 2 + 1, nh / 2 - 1}, {nh / 2 + 2, nh / 2 - 1}};
        for(auto & e : list) vector_checks(nh, e[0], e[1], g, worst);
    }
    printf("checks: %d, failed: %d, largest deviation %.1e\n", n_check, n_fail, worst);
    printf(n_fail ? "spinprojtest: FAIL\n" : "spinprojtest: PASS\n");
    return n_fail ? 1 : 0;
}
