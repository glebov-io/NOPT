#include "tensor_tape.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "blas_link.h"
#include <algorithm>
#include <omp.h>

namespace tt{

namespace{
[[noreturn]] void fail(const std::string & msg){ fprintf(stderr, "tensor_tape: %s\n", msg.c_str()); abort(); }

// letter loop plan: nl distinct letters (output letters first), extents, and per tensor the stride of each letter
struct Plan{
    std::vector<int> ext;
    std::vector<std::vector<long>> stride;   // [tensor][letter]
};

// lead: the letter to put first (the one the loops are split over between threads); 0 = the output letters first
Plan make_plan(const std::string & spec, const std::vector<const Ten *> & ts, char lead = 0){
    const size_t arrow = spec.find("->");
    if(arrow == std::string::npos) fail("einsum spec without '->': " + spec);
    std::vector<std::string> subs;
    {
        std::string lhs = spec.substr(0, arrow), cur;
        for(char c : lhs){ if(c == ','){ subs.push_back(cur); cur.clear(); } else if(c != ' ') cur += c; }
        subs.push_back(cur);
    }
    subs.push_back(spec.substr(arrow + 2));
    if(subs.size() != ts.size()) fail("einsum operand count does not match: " + spec);
    std::vector<char> letters; std::map<char, int> pos;
    auto add = [&](char c){ if(!pos.count(c)){ pos[c] = (int)letters.size(); letters.push_back(c); } };
    if(lead) add(lead);
    for(char c : subs.back()) add(c);
    for(size_t k = 0; k + 1 < subs.size(); k++) for(char c : subs[k]) add(c);
    Plan p; p.ext.assign(letters.size(), -1); p.stride.assign(ts.size(), std::vector<long>(letters.size(), 0));
    for(size_t k = 0; k < ts.size(); k++){
        const std::string & s = subs[k];
        if(s.size() != ts[k]->dim.size()) fail("einsum operand rank does not match its subscripts: " + spec);
        long st = 1;
        for(int a = (int)s.size() - 1; a >= 0; a--){
            const int l = pos[s[a]], d = ts[k]->dim[a];
            if(p.ext[l] < 0) p.ext[l] = d; else if(p.ext[l] != d) fail("einsum extent mismatch for letter in: " + spec);
            p.stride[k][l] += st; st *= d;
        }
    }
    // output letters must not repeat (a diagonal output is not supported)
    { std::map<char, int> cnt; for(char c : subs.back()) if(++cnt[c] > 1) fail("repeated output letter: " + spec); }
    return p;
}

// the loops over all letters with letter 0 restricted to [lo, hi) (hi < 0: its full range)
template<class F> void run(const Plan & p, F f, int lo = 0, int hi = -1){
    const int nl = (int)p.ext.size(), nt = (int)p.stride.size();
    for(int e : p.ext) if(e == 0) return;
    std::vector<long> off(nt, 0);
    if(nl == 0){ if(lo == 0) f(off.data()); return; }
    if(hi < 0) hi = p.ext[0];
    if(lo >= hi) return;
    std::vector<int> idx(nl, 0);
    if(nl == 1){
        for(int t = 0; t < nt; t++) off[t] = (long)lo * p.stride[t][0];
        for(int i = lo; i < hi; i++){ f(off.data()); for(int t = 0; t < nt; t++) off[t] += p.stride[t][0]; }
        return;
    }
    // letter 0 is the outermost: iterate it explicitly, the rest with the odometer
    for(int i0 = lo; i0 < hi; i0++){
        std::fill(off.begin(), off.end(), 0);
        for(int t = 0; t < nt; t++) off[t] = (long)i0 * p.stride[t][0];
        std::fill(idx.begin(), idx.end(), 0);
        const int L = nl - 1;
        while(true){
            for(int i = 0; i < p.ext[L]; i++){ f(off.data()); for(int t = 0; t < nt; t++) off[t] += p.stride[t][L]; }
            for(int t = 0; t < nt; t++) off[t] -= (long)p.ext[L] * p.stride[t][L];
            int l = L - 1;
            while(l >= 1){
                idx[l]++;
                for(int t = 0; t < nt; t++) off[t] += p.stride[t][l];
                if(idx[l] < p.ext[l]) break;
                for(int t = 0; t < nt; t++) off[t] -= (long)p.ext[l] * p.stride[t][l];
                idx[l] = 0; l--;
            }
            if(l < 1) break;
        }
    }
}

// total work of a plan (product of the extents)
double work(const Plan & p){ double w = 1; for(int e : p.ext) w *= e; return w; }

// split letter 0 between OpenMP threads; body(lo, hi, thread) runs one chunk. Small loops stay serial.
template<class B> void split(const Plan & p, B body){
    const int n0 = p.ext.empty() ? 1 : p.ext[0];
    int nth = omp_get_max_threads();
    if(work(p) < 2e5 || n0 < 2 || nth < 2){ body(0, n0, 0); return; }
    nth = std::min(nth, n0);
    #pragma omp parallel num_threads(nth)
    {
        const int t = omp_get_thread_num(), lo = (int)((long)n0 * t / nth), hi = (int)((long)n0 * (t + 1) / nth);
        body(lo, hi, t);
    }
}

const double TAYLOR_SMALL = 1e-3;
}

double reg_value(Reg kind, double s, double D){
    const double q = std::exp(-s * D * D);
    auto R = [&](){
        const double sq = std::sqrt(s), Z = sq * D;
        if(std::fabs(Z) < TAYLOR_SMALL){ double value = Z, tmp = Z; for(int x = 0; x < 3; x++){ tmp *= -Z * Z / (x + 2); value += tmp; } return value * sq; }
        return (1.0 - q) / D;
    };
    switch(kind){
        case Reg::Q: return q;
        case Reg::ONE_PLUS_Q: return 1.0 + q;
        case Reg::R: return R();
        case Reg::QR: return (1.0 + q) * R();
        case Reg::INV: return 1.0 / D;
    }
    return 0.0;
}

double reg_deriv(Reg kind, double s, double D){
    const double q = std::exp(-s * D * D), dq = -2.0 * s * D * q;
    auto R = [&](){
        const double sq = std::sqrt(s), Z = sq * D;
        if(std::fabs(Z) < TAYLOR_SMALL){ double value = Z, tmp = Z; for(int x = 0; x < 3; x++){ tmp *= -Z * Z / (x + 2); value += tmp; } return value * sq; }
        return (1.0 - q) / D;
    };
    auto dR = [&](){
        const double sq = std::sqrt(s), Z = sq * D;
        // the Taylor polynomial sq (Z - Z^3/2 + Z^5/6 - Z^7/24), differentiated as such
        if(std::fabs(Z) < TAYLOR_SMALL){ const double Z2 = Z * Z; return s * (1.0 - 1.5 * Z2 + (5.0 / 6.0) * Z2 * Z2 - (7.0 / 24.0) * Z2 * Z2 * Z2); }
        return (2.0 * s * D * D * q - (1.0 - q)) / (D * D);
    };
    switch(kind){
        case Reg::Q: return dq;
        case Reg::ONE_PLUS_Q: return dq;
        case Reg::R: return dR();
        case Reg::QR: return dq * R() + (1.0 + q) * dR();
        case Reg::INV: return -1.0 / (D * D);
    }
    return 0.0;
}

P Tape::leaf(const std::vector<int> & dim, const double * data, bool grad){
    P t = std::make_shared<Ten>();
    t->dim = dim; t->grad = grad;
    t->v.assign(t->size(), 0.0);
    if(data) for(size_t i = 0; i < t->size(); i++) t->v[i] = data[i];
    t->g.assign(grad ? t->size() : 0, 0.0);
    return t;
}
P Tape::zeros(const std::vector<int> & dim, bool grad){ return leaf(dim, nullptr, grad); }

void Tape::einsum_into(const P & out, const std::string & spec, const std::vector<P> & ins, double alpha){
    std::vector<const Ten *> ts; for(const P & x : ins) ts.push_back(x.get()); ts.push_back(out.get());
    const int n = (int)ins.size();
    // forward: split over an output letter (disjoint writes); a scalar output is reduced per thread and summed in thread order
    {
        const Plan plan = make_plan(spec, ts);
        std::vector<const double *> in(n); for(int k = 0; k < n; k++) in[k] = ins[k]->v.data();
        double * o = out->v.data();
        if(!out->dim.empty())
            split(plan, [&](int lo, int hi, int){ run(plan, [&](const long * off){ double pr = alpha; for(int k = 0; k < n; k++) pr *= in[k][off[k]]; o[off[n]] += pr; }, lo, hi); });
        else{
            std::vector<double> part(omp_get_max_threads() + 1, 0.0);
            split(plan, [&](int lo, int hi, int t){ double acc = 0; run(plan, [&](const long * off){ double pr = alpha; for(int k = 0; k < n; k++) pr *= in[k][off[k]]; acc += pr; }, lo, hi); part[t] = acc; });
            for(double v : part) o[0] += v;
        }
    }
    // backward: one pass per operand, split over a letter of that operand (disjoint writes into its adjoint)
    std::vector<std::shared_ptr<Plan>> bplans(n);
    {
        const size_t arrow = spec.find("->");
        std::vector<std::string> subs; std::string lhs = spec.substr(0, arrow), cur;
        for(char c : lhs){ if(c == ','){ subs.push_back(cur); cur.clear(); } else if(c != ' ') cur += c; }
        subs.push_back(cur);
        for(int k = 0; k < n; k++) bplans[k] = std::make_shared<Plan>(make_plan(spec, ts, subs[k].empty() ? 0 : subs[k][0]));
    }
    back_.push_back([bplans, ins, out, alpha, n](){
        if(!out->grad) return;
        std::vector<const double *> in(n); for(int k = 0; k < n; k++) in[k] = ins[k]->v.data();
        const double * go = out->g.data();
        for(int k = 0; k < n; k++){
            if(!ins[k]->grad) continue;
            double * gk = ins[k]->g.data();
            const Plan & plan = *bplans[k];
            auto body = [&](const long * off){
                const double gv = go[off[n]];
                if(gv == 0.0) return;
                double pr = alpha * gv;
                for(int j = 0; j < n; j++) if(j != k) pr *= in[j][off[j]];
                gk[off[k]] += pr;
            };
            if(!ins[k]->dim.empty()) split(plan, [&](int lo, int hi, int){ run(plan, body, lo, hi); });
            else{
                // a scalar operand: per-thread partial sums
                std::vector<double> part(omp_get_max_threads() + 1, 0.0);
                split(plan, [&](int lo, int hi, int t){
                    double acc = 0;
                    run(plan, [&](const long * off){ const double gv = go[off[n]]; if(gv == 0.0) return; double pr = alpha * gv; for(int j = 0; j < n; j++) if(j != k) pr *= in[j][off[j]]; acc += pr; }, lo, hi);
                    part[t] = acc; });
                for(double v : part) gk[0] += v;
            }
        }
    });
}

P Tape::einsum(const std::string & spec, const std::vector<P> & ins, double alpha){
    // the output extents from the operands
    const size_t arrow = spec.find("->");
    if(arrow == std::string::npos) fail("einsum spec without '->': " + spec);
    std::map<char, int> ext;
    {
        std::string lhs = spec.substr(0, arrow); size_t k = 0, a = 0;
        for(char c : lhs){
            if(c == ','){ k++; a = 0; continue; }
            if(c == ' ') continue;
            if(k >= ins.size() || a >= ins[k]->dim.size()) fail("einsum subscripts do not match the operands: " + spec);
            ext[c] = ins[k]->dim[a]; a++;
        }
    }
    std::vector<int> dim;
    for(char c : spec.substr(arrow + 2)){ if(!ext.count(c)) fail("output letter absent from the inputs: " + spec); dim.push_back(ext[c]); }
    bool grad = false; for(const P & x : ins) grad = grad || x->grad;
    P out = zeros(dim, grad);
    einsum_into(out, spec, ins, alpha);
    return out;
}

P Tape::lin(const std::vector<std::pair<double, P>> & terms){
    if(terms.empty()) fail("lin without terms");
    bool grad = false; for(auto & t : terms) grad = grad || t.second->grad;
    P out = zeros(terms[0].second->dim, grad);
    for(auto & t : terms){
        if(t.second->dim != out->dim) fail("lin: shape mismatch");
        for(size_t i = 0; i < out->size(); i++) out->v[i] += t.first * t.second->v[i];
    }
    back_.push_back([terms, out](){
        if(!out->grad) return;
        for(auto & t : terms) if(t.second->grad) for(size_t i = 0; i < out->size(); i++) t.second->g[i] += t.first * out->g[i];
    });
    return out;
}

P Tape::reg(const P & x, Reg kind, double s, const std::vector<P> & eps, const std::vector<double> & sign){
    const int r = (int)x->dim.size();
    if((int)eps.size() != r || (int)sign.size() != r) fail("reg: one eps (or null) and one sign per axis");
    for(int a = 0; a < r; a++) if(eps[a] && (int)eps[a]->size() != x->dim[a]) fail("reg: eps length does not match the axis");
    bool grad = x->grad; for(const P & e : eps) if(e) grad = grad || e->grad;
    P out = zeros(x->dim, grad);
    const size_t n = x->size();
    std::vector<long> st(r, 1); for(int a = r - 2; a >= 0; a--) st[a] = st[a + 1] * x->dim[a + 1];
    auto Dof = [=](size_t i){ double D = 0; for(int a = 0; a < r; a++) if(eps[a]) D += sign[a] * eps[a]->v[(i / st[a]) % x->dim[a]]; return D; };
    #pragma omp parallel for schedule(static) if(n > 100000)
    for(long i = 0; i < (long)n; i++) out->v[i] = x->v[i] * reg_value(kind, s, Dof(i));
    back_.push_back([=](){
        if(!out->grad) return;
        // the eps adjoints (short vectors) are accumulated per thread and summed in thread order
        const int nth = n > 100000 ? omp_get_max_threads() : 1;
        std::vector<std::vector<std::vector<double>>> eg(nth, std::vector<std::vector<double>>(r));
        for(int t = 0; t < nth; t++) for(int a = 0; a < r; a++) if(eps[a] && eps[a]->grad) eg[t][a].assign(eps[a]->size(), 0.0);
        #pragma omp parallel for schedule(static) num_threads(nth)
        for(long i = 0; i < (long)n; i++){
            const double go = out->g[i];
            if(go == 0.0) continue;
            const double D = Dof(i);
            if(x->grad) x->g[i] += go * reg_value(kind, s, D);
            const double dD = go * x->v[i] * reg_deriv(kind, s, D);
            auto & mine = eg[omp_get_thread_num()];
            for(int a = 0; a < r; a++) if(!mine[a].empty()) mine[a][(i / st[a]) % x->dim[a]] += sign[a] * dD;
        }
        for(int t = 0; t < nth; t++) for(int a = 0; a < r; a++) for(size_t k = 0; k < eg[t][a].size(); k++) eps[a]->g[k] += eg[t][a][k];
    });
    return out;
}

P Tape::auxgemm(const P & A, const P & B){
    if(A->dim.size() != 3 || B->dim.size() != 3 || A->dim[2] != B->dim[2]) fail("auxgemm: operands must be [p,q,P] with the same P");
    const int rA = A->dim[0] * A->dim[1], rB = B->dim[0] * B->dim[1], np = A->dim[2];
    P out = zeros({A->dim[0], A->dim[1], B->dim[0], B->dim[1]}, A->grad || B->grad);
    if(rA && rB && np) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rA, rB, np, 1.0, A->v.data(), np, B->v.data(), np, 0.0, out->v.data(), rB);
    back_.push_back([=](){
        if(!out->grad || !rA || !rB || !np) return;
        if(A->grad) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rA, np, rB, 1.0, out->g.data(), rB, B->v.data(), np, 1.0, A->g.data(), np);
        if(B->grad) cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rB, np, rA, 1.0, out->g.data(), rB, A->v.data(), np, 1.0, B->g.data(), np);
    });
    return out;
}

void Tape::backward(const P & scalar, double seed){
    if(scalar->size() != 1) fail("backward: the output must be a scalar");
    if(!scalar->grad) return;
    scalar->g[0] += seed;
    for(auto it = back_.rbegin(); it != back_.rend(); ++it) (*it)();
}

}   // namespace tt
