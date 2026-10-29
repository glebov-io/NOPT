// First-derivative libint (built by libint-deriv.sh): for each integral kind the
// harness finds, by central finite differences, which result buffer holds d/d(centre, xyz),
// prints that ordering table, and reports the residual. A centre with no buffer of its own is
// checked through translational invariance (the derivatives over all centres sum to zero).
// Kinds: overlap, kinetic, nuclear attraction, 4-center ERI, 3-center (P|ab), 2-center (P|Q).
// Exit 1 if any residual exceeds 1e-6.
#include <libint2.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>
using namespace libint2;
typedef std::vector<double> V;
typedef std::vector<std::array<double, 3>> Pos;

static Shell mk(int l, double a, const std::array<double, 3> & c){
    return Shell{{a, a * 0.37}, {{l, false, {1.0, 0.6}}}, {{c[0], c[1], c[2]}}};
}
static int ncart(int l){ return (l + 1) * (l + 2) / 2; }

static bool g_ok = true;

// eval0(pos): order-0 integrals; eval1(pos): all first-derivative buffers
static void test(const char * name, const Pos & pos, size_t nint,
                 std::function<V(const Pos &)> eval0, std::function<std::vector<V>(const Pos &)> eval1){
    const double h = 1e-5;
    auto bufs = eval1(pos);
    printf("%s: %zu integrals, %zu derivative buffers\n", name, nint, bufs.size());
    std::vector<V> fd(pos.size() * 3);
    for(size_t c = 0; c < pos.size(); c++) for(int q = 0; q < 3; q++){
        Pos p = pos, m = pos; p[c][q] += h; m[c][q] -= h;
        V a = eval0(p), b = eval0(m), d(nint);
        for(size_t i = 0; i < nint; i++) d[i] = (a[i] - b[i]) / (2 * h);
        fd[c * 3 + q] = d;
    }
    std::vector<int> used(pos.size() * 3, -1);
    double worst = 0;
    for(size_t k = 0; k < fd.size(); k++){
        double best = 1e300; int bi = -1;
        for(size_t j = 0; j < bufs.size(); j++){
            double e = 0; for(size_t i = 0; i < nint; i++) e = std::max(e, std::fabs(fd[k][i] - bufs[j][i]));
            if(e < best){ best = e; bi = (int)j; }
        }
        if(best < 1e-6){ used[k] = bi; worst = std::max(worst, best); }
        else used[k] = -1;
    }
    // centres without their own buffer: translational invariance
    for(size_t k = 0; k < fd.size(); k++){
        const char * ax = "xyz";
        if(used[k] >= 0) { printf("   centre %zu %c  <- buffer %d\n", k / 3, ax[k % 3], used[k]); continue; }
        // sum of the other centres' analytic buffers must equal -FD of this one
        double e = 0; bool others = true;
        for(size_t i = 0; i < nint; i++){
            double s = 0;
            for(size_t c = 0; c < pos.size(); c++){ int u = used[c * 3 + k % 3]; if(c == k / 3) continue;
                if(u < 0) { others = false; break; } s += bufs[u][i]; }
            if(!others) break;
            e = std::max(e, std::fabs(-s - fd[k][i]));
        }
        bool ok = others && e < 1e-6;
        printf("   centre %zu %c  <- translational invariance (%s, err %.1e)\n", k / 3, ax[k % 3], ok ? "ok" : "FAILED", e);
        if(!ok) g_ok = false; else worst = std::max(worst, e);
    }
    printf("   worst |analytic - FD| = %.2e\n", worst);
    if(worst >= 1e-6) g_ok = false;
}

int main(){
    setvbuf(stdout, nullptr, _IONBF, 0);
    initialize();
    const Pos A = {{0.0, 0.1, -0.4}, {0.3, -0.2, 0.5}, {0.7, 0.4, 0.2}, {-0.5, 0.6, 0.1}};
    // one-body: (a|b) with the highest AM the library supports (l = 7)
    for(auto lab : std::vector<std::pair<int,int>>{{2, 3}, {7, 7}}){
        int la = lab.first, lb = lab.second; size_t nint = ncart(la) * ncart(lb);
        Pos pos = {A[0], A[1]};
        for(auto op : {Operator::overlap, Operator::kinetic}){
            char nm[64]; snprintf(nm, sizeof nm, "%s (l=%d,%d)", op == Operator::overlap ? "overlap" : "kinetic", la, lb);
            test(nm, pos, nint,
                [=](const Pos & p){ Engine e(op, 2, 7, 0); e.compute(mk(la, 1.1, p[0]), mk(lb, 0.9, p[1]));
                                    const double * r = e.results()[0]; return V(r, r + nint); },
                [=](const Pos & p){ Engine e(op, 2, 7, 1); e.compute(mk(la, 1.1, p[0]), mk(lb, 0.9, p[1]));
                                    std::vector<V> o; for(size_t k = 0; k < e.results().size(); k++) o.push_back(V(e.results()[k], e.results()[k] + nint)); return o; });
        }
    }
    {   // nuclear attraction with three point charges: charge-centre derivatives too
        std::vector<std::pair<double, std::array<double,3>>> q = {{1.0, A[2]}, {2.0, A[3]}};
        int la = 3, lb = 2; size_t nint = ncart(la) * ncart(lb);
        Pos pos = {A[0], A[1], A[2], A[3]};
        test("nuclear (l=3,2), 2 charges", pos, nint,
            [=](const Pos & p){ Engine e(Operator::nuclear, 2, 4, 0);
                e.set_params(std::vector<std::pair<double, std::array<double,3>>>{{1.0, p[2]}, {2.0, p[3]}});
                e.compute(mk(la, 1.1, p[0]), mk(lb, 0.9, p[1])); const double * r = e.results()[0]; return V(r, r + nint); },
            [=](const Pos & p){ Engine e(Operator::nuclear, 2, 4, 1);
                e.set_params(std::vector<std::pair<double, std::array<double,3>>>{{1.0, p[2]}, {2.0, p[3]}});
                e.compute(mk(la, 1.1, p[0]), mk(lb, 0.9, p[1]));
                std::vector<V> o; for(size_t k = 0; k < e.results().size(); k++) o.push_back(V(e.results()[k], e.results()[k] + nint)); return o; });
        (void)q;
    }
    {   // 4-center ERI (dd|fp)
        size_t nint = ncart(2) * ncart(2) * ncart(3) * ncart(1);
        Pos pos = {A[0], A[1], A[2], A[3]};
        auto sh = [](const Pos & p){ return std::vector<Shell>{mk(2, 1.1, p[0]), mk(2, 0.9, p[1]), mk(3, 0.8, p[2]), mk(1, 1.3, p[3])}; };
        test("ERI (dd|fp)", pos, nint,
            [=](const Pos & p){ Engine e(Operator::coulomb, 2, 3, 0); auto s = sh(p); e.compute(s[0], s[1], s[2], s[3]); const double * r = e.results()[0]; return V(r, r + nint); },
            [=](const Pos & p){ Engine e(Operator::coulomb, 2, 3, 1); auto s = sh(p); e.compute(s[0], s[1], s[2], s[3]);
                std::vector<V> o; for(size_t k = 0; k < e.results().size(); k++) o.push_back(V(e.results()[k], e.results()[k] + nint)); return o; });
    }
    for(int lp : {2, 7}){   // 3-center (P|ab); the fitting shell goes up to l = 7
        size_t nint = ncart(lp) * ncart(2) * ncart(3);
        Pos pos = {A[0], A[1], A[2]};
        char nm[64]; snprintf(nm, sizeof nm, "(P|ab) 3-center l_P=%d, (d f)", lp);
        auto run = [=](const Pos & p, int d){ Engine e(Operator::coulomb, 2, std::max(lp, 3), d); e.set(BraKet::xs_xx);
            e.compute(mk(lp, 0.8, p[0]), mk(2, 1.1, p[1]), mk(3, 0.9, p[2])); return e; };
        test(nm, pos, nint,
            [=](const Pos & p){ auto e = run(p, 0); const double * r = e.results()[0]; return V(r, r + nint); },
            [=](const Pos & p){ auto e = run(p, 1); std::vector<V> o; for(size_t k = 0; k < e.results().size(); k++) o.push_back(V(e.results()[k], e.results()[k] + nint)); return o; });
    }
    {   // 2-center (P|Q)
        size_t nint = ncart(5) * ncart(7);
        Pos pos = {A[0], A[1]};
        auto run = [=](const Pos & p, int d){ Engine e(Operator::coulomb, 2, 7, d); e.set(BraKet::xs_xs);
            e.compute(mk(5, 0.8, p[0]), mk(7, 0.9, p[1])); return e; };
        test("(P|Q) 2-center l=(5,7)", pos, nint,
            [=](const Pos & p){ auto e = run(p, 0); const double * r = e.results()[0]; return V(r, r + nint); },
            [=](const Pos & p){ auto e = run(p, 1); std::vector<V> o; for(size_t k = 0; k < e.results().size(); k++) o.push_back(V(e.results()[k], e.results()[k] + nint)); return o; });
    }
    {   // the maximum: (kk|kk) first derivative engine exists and runs
        Engine e(Operator::coulomb, 2, 7, 1); Shell s = mk(7, 0.5, {0, 0, 0});
        e.compute(s, s, s, s);
        printf("ERI (7 7|7 7) first derivative: ok (%zu buffers)\n", e.results().size());
    }
    finalize();
    printf("%s\n", g_ok ? "PASS" : "FAIL");
    return g_ok ? 0 : 1;
}
