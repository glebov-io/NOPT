// tapetest: the reverse-mode tape of include/tensor_tape.h against central finite differences (no molecule).
//
// Each case builds a scalar on a fresh tape from random leaves, runs backward(), and compares sum_leaves <g, d> with the
// finite difference along random directions d of all leaves at once:
//   1. einsum: a matrix product, a three-operand contraction with summed letters, a diagonal (repeated letter), a scalar operand
//   2. reg: every regularizer kind, with the orbital-energy adjoints, and the Taylor branch of R near D = 0
//   3. auxgemm (RI three-index -> four-index) and lin
//   4. einsum_into accumulating into an existing tensor, and a constant leaf (no adjoint)
//   5. a contraction big enough to be split over threads
#include <stdio.h>
#include <cmath>
#include <functional>
#include <vector>

#include "tensor_tape.h"

typedef std::vector<double> V;
static unsigned long long st = 88172645463325252ULL;
static double rnd(){ st ^= st << 13; st ^= st >> 7; st ^= st << 17; return ((st >> 11) * (1.0 / 9007199254740992.0)) * 2 - 1; }
static V rv(size_t n){ V v(n); for(double & x : v) x = rnd(); return v; }
static int fails = 0;

static void report(const char * name, double an, double fd, double tol = 1e-7){
    const double err = std::fabs(an - fd), scale = std::max(std::fabs(an), std::fabs(fd));
    const bool good = err < tol * std::max(1.0, scale);
    printf("%-58s analytic % .10e  FD % .10e  |diff| %.1e  %s\n", name, an, fd, err, good ? "PASS" : "FAIL");
    if(!good) fails++;
}

// leaves: shapes and values; build(tape, leaves) returns the scalar. Checks the tape gradient of every leaf along d.
struct Case{
    std::vector<std::vector<int>> dims;
    std::vector<V> x, d;
    std::vector<bool> constant;
    void add(const std::vector<int> & dim, bool is_const = false, double shift = 0.0){
        size_t n = 1; for(int k : dim) n *= k;
        V v = rv(n); for(double & a : v) a += shift;
        dims.push_back(dim); x.push_back(v); d.push_back(rv(n)); constant.push_back(is_const);
    }
};
static void check(const char * name, Case & c, const std::function<tt::P(tt::Tape &, std::vector<tt::P> &)> & build, double h = 1e-5, double tol = 1e-7){
    auto value = [&](double s){
        tt::Tape T; std::vector<tt::P> L;
        for(size_t k = 0; k < c.x.size(); k++){ V v = c.x[k]; if(!c.constant[k]) for(size_t i = 0; i < v.size(); i++) v[i] += s * c.d[k][i]; L.push_back(T.leaf(c.dims[k], v.data(), !c.constant[k])); }
        return build(T, L)->v[0];
    };
    tt::Tape T; std::vector<tt::P> L;
    for(size_t k = 0; k < c.x.size(); k++) L.push_back(T.leaf(c.dims[k], c.x[k].data(), !c.constant[k]));
    tt::P out = build(T, L);
    T.backward(out);
    double an = 0;
    for(size_t k = 0; k < L.size(); k++){
        if(c.constant[k]){ if(!L[k]->g.empty()){ printf("%s: constant leaf %zu got an adjoint  FAIL\n", name, k); fails++; } continue; }
        for(size_t i = 0; i < L[k]->g.size(); i++) an += L[k]->g[i] * c.d[k][i];
    }
    report(name, an, (value(h) - value(-h)) / (2 * h), tol);
}

int main(){
    {   // 1. einsum
        Case c; c.add({3, 4}); c.add({4, 5}); c.add({5, 3, 2}); c.add({2}); c.add({3, 3}); c.add({});
        check("einsum: product, 3 operands, diagonal, scalar operand", c, [](tt::Tape & T, std::vector<tt::P> & L){
            tt::P X = T.einsum("ij,jk->ik", {L[0], L[1]});                       // 3 x 5
            tt::P Y = T.einsum("ik,kip,p->", {X, L[2], L[3]});                   // three operands, all letters summed
            tt::P Z = T.einsum("ii,ik,ik,->", {L[4], X, X, L[5]}, 0.5);           // diagonal of L[4], X twice, scalar L[5]
            return T.lin({{1.0, Y}, {-0.3, Z}});
        });
    }
    {   // 2. regularizers, each kind, D = e1[i] - e2[j] + e3[k] away from 0
        const tt::Reg kinds[5] = {tt::Reg::Q, tt::Reg::ONE_PLUS_Q, tt::Reg::R, tt::Reg::QR, tt::Reg::INV};
        const char * names[5] = {"reg Q = exp(-s D^2)", "reg 1 + Q", "reg R = (1 - Q)/D", "reg (1 + Q) R", "reg 1/D"};
        for(int k = 0; k < 5; k++){
            Case c; c.add({3, 4, 2}); c.add({3}, false, 3.0); c.add({4}); c.add({2}); c.add({3, 4, 2});
            const tt::Reg kind = kinds[k];
            char name[96]; snprintf(name, sizeof name, "%s, with the eps adjoints", names[k]);
            check(name, c, [kind](tt::Tape & T, std::vector<tt::P> & L){
                tt::P R = T.reg(L[0], kind, 0.7, {L[1], L[2], L[3]}, {1.0, -1.0, 1.0});
                return T.einsum("ijk,ijk->", {R, L[4]});
            });
        }
        // the Taylor branch of R: sqrt(s)|D| < 1e-3 (eps differ by ~1e-4); the polynomial is differentiated as such
        Case c; c.add({3, 3}); c.add({3}); c.add({3}); c.add({3, 3});
        for(int i = 0; i < 3; i++){ c.x[1][i] = 0.1 * i; c.x[2][i] = 0.1 * i + 1e-4 * (i + 1); c.d[1][i] *= 1e-5; c.d[2][i] *= 1e-5; }
        check("reg R, Taylor branch near D = 0", c, [](tt::Tape & T, std::vector<tt::P> & L){
            tt::P R = T.reg(L[0], tt::Reg::R, 0.5, {L[1], L[2]}, {1.0, -1.0});
            return T.einsum("ij,ij->", {R, L[3]});
        }, 1e-7);
    }
    {   // 3. auxgemm and lin
        Case c; c.add({3, 2, 7}); c.add({2, 4, 7}); c.add({3, 2, 2, 4}); c.add({3, 2, 2, 4});
        check("auxgemm [p,q,P] x [r,s,P] and lin", c, [](tt::Tape & T, std::vector<tt::P> & L){
            tt::P G = T.auxgemm(L[0], L[1]);
            tt::P M = T.lin({{2.0, G}, {-1.5, L[2]}});
            return T.einsum("pqrs,pqrs,pqrs->", {M, G, L[3]});
        });
    }
    {   // 4. einsum_into accumulation, and a constant leaf
        Case c; c.add({4, 4}); c.add({4, 4}, true); c.add({4, 4});
        check("einsum_into accumulation, constant leaf", c, [](tt::Tape & T, std::vector<tt::P> & L){
            tt::P acc = T.einsum("ij,jk->ik", {L[0], L[1]});
            T.einsum_into(acc, "ij,kj->ik", {L[0], L[2]}, -0.5);
            T.einsum_into(acc, "ij->ji", {L[2]});
            return T.einsum("ik,ik->", {acc, acc});
        });
    }
    {   // 5. big enough for the thread split (work > 2e5 per einsum), forward and backward
        Case c; c.add({24, 30}); c.add({30, 26}); c.add({26, 24, 20}); c.add({20});
        check("threaded einsum (work > 2e5)", c, [](tt::Tape & T, std::vector<tt::P> & L){
            tt::P X = T.einsum("ij,jk->ik", {L[0], L[1]});
            return T.einsum("ik,kip,p->", {X, L[2], L[3]});
        });
    }
    printf("\n%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
