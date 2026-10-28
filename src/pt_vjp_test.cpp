// Dot-product test of the PT block adjoints (include/pt_vjp.h) against finite differences,
// run where the RI tensors, the orbital energies and H_* already exist; xmc_vjp_test drives it from QDPT2 (NOPT_XMC_VJP_TEST=<blocks>).
#include "pt_vjp.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common_vars.h"
#include "matr.h"
#include "blas_link.h"

std::function<void(PT_tensors &, const std::string &)> pt_vjp_forward_override;

namespace{

std::mt19937_64 gen(20261024);
double rnd(){ return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); }

struct Group{ std::string name; double * x; size_t n; double * adj; bool is_energy; };

void run_block(PT_tensors & T, const std::string & blk){
    if(pt_vjp_forward_override){ pt_vjp_forward_override(T, blk); return; }
    if(blk == "ALL") T.E2_calc_EE();
    else if(blk == "CCVV") T.calc_EE_2_CCVV();
    else if(blk == "CAVV") T.calc_EE_2_CAVV();
    else if(blk == "AAVV") T.calc_EE_2_AAVV();
    else if(blk == "CCAV") T.calc_EE_2_CCAV();
    else if(blk == "CCAA") T.calc_EE_2_CCAA();
    else if(blk == "CV") T.calc_EE_2_CV();
    else if(blk == "AV") T.calc_EE_2_AV();
    else if(blk == "CA") T.calc_EE_2_CA();
    else { fprintf(out_stream, "pt_vjp_selftest: block %s is not implemented\n", blk.c_str()); }
}

int vjp_block(PT_tensors & T, const std::string & blk, const PTRawSeeds & S, PTInputAdj & A){
    if(blk == "ALL"){   // the seeds are on the FINAL arrays here
        const size_t n2 = (size_t)T.n_a * T.n_a, n4 = n2 * n2, n6 = n4 * n2;
        auto vec = [](const double * p, size_t n){ return p ? std::vector<double>(p, p + n) : std::vector<double>(n, 0.0); };
        return pt_vjp_EE_all(T, S.PS, vec(S.PH, n2), vec(S.PV_JK, n4), vec(S.PV_AB, n4), vec(S.P3_JK, n6), vec(S.P3_AB, n6), A);
    }
    if(blk == "CCVV") return pt_vjp_EE_CCVV(T, S, A);
    if(blk == "CAVV") return pt_vjp_EE_CAVV(T, S, A);
    if(blk == "AAVV") return pt_vjp_EE_AAVV(T, S, A);
    if(blk == "CCAV") return pt_vjp_EE_CCAV(T, S, A);
    if(blk == "CCAA") return pt_vjp_EE_CCAA(T, S, A);
    if(blk == "CV") return pt_vjp_EE_CV(T, S, A);
    if(blk == "AV") return pt_vjp_EE_AV(T, S, A);
    if(blk == "CA") return pt_vjp_EE_CA(T, S, A);
    return 1;
}

}


// The end of E2_calc_EE (symmetrization + permutation) as a standalone copy, checked against the production
// E2_calc_EE in pt_vjp_selftest_final(), and used for the dot-product test of pt_vjp_EE_final().
static void finalize_copy(int na, std::vector<double> & PH, std::vector<double> & JK, std::vector<double> & AB,
                          std::vector<double> & J3, std::vector<double> & A3){
    auto sym = [](std::vector<double> & A, size_t m){ symmetrization(A.data(), (int)m); };
    sym(PH, na); sym(JK, (size_t)na * na); sym(AB, (size_t)na * na); sym(J3, (size_t)na * na * na); sym(A3, (size_t)na * na * na);
    std::vector<double> tmp(A3.size());
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++)
    for(int w = 0; w < na; w++) for(int x = 0; x < na; x++) for(int y = 0; y < na; y++)
        tmp[((((t * na + u) * na + v) * na + w) * na + x) * na + y] = A3[((((t * na + u) * na + x) * na + v) * na + w) * na + y];
    A3 = tmp;
}

int pt_vjp_selftest_final(PT_tensors & T){
    const int na = T.n_a;
    const size_t n2 = (size_t)na * na, n4 = n2 * n2, n6 = n4 * n2;
    int fails = 0;
    // (1) the copy equals the production end of E2_calc_EE: raw sum of the eight blocks -> copy vs E2_calc_EE
    T.RF_PS = 0;
    set_zero_matr(T.RF_PH, n2); set_zero_matr(T.RF_PV_JK, n4); set_zero_matr(T.RF_PV_AB, n4);
    set_zero_matr(T.RF_P3_JK, n6); set_zero_matr(T.RF_P3_AB, n6);
    for(int t = 0; t < na; t++){ T.e_IP[t] = T.e_a[t]; T.e_EA[t] = T.e_a[t]; }
    T.calc_EE_2_CCVV(); T.calc_EE_2_CAVV(); T.calc_EE_2_AAVV(); T.calc_EE_2_CCAV(); T.calc_EE_2_CCAA();
    T.calc_EE_2_CV(); T.calc_EE_2_AV(); T.calc_EE_2_CA();
    std::vector<double> PH(T.RF_PH, T.RF_PH + n2), JK(T.RF_PV_JK, T.RF_PV_JK + n4), AB(T.RF_PV_AB, T.RF_PV_AB + n4),
                        J3(T.RF_P3_JK, T.RF_P3_JK + n6), A3(T.RF_P3_AB, T.RF_P3_AB + n6);
    const double PS = T.RF_PS;
    finalize_copy(na, PH, JK, AB, J3, A3);
    T.E2_calc_EE();
    double d = std::fabs(PS - T.RF_PS);
    for(size_t i = 0; i < n2; i++) d = std::max(d, std::fabs(PH[i] - T.RF_PH[i]));
    for(size_t i = 0; i < n4; i++) d = std::max(d, std::max(std::fabs(JK[i] - T.RF_PV_JK[i]), std::fabs(AB[i] - T.RF_PV_AB[i])));
    for(size_t i = 0; i < n6; i++) d = std::max(d, std::max(std::fabs(J3[i] - T.RF_P3_JK[i]), std::fabs(A3[i] - T.RF_P3_AB[i])));
    fprintf(out_stream, "\nFINAL: copy of the end of E2_calc_EE vs production: max |difference| = %.2e %s\n", d, d < 1e-12 ? "ok" : "MISMATCH");
    if(d >= 1e-12) fails++;

    // (2) dot product: <ybar, F(x)> = <F^T ybar, x> with random raw x and random final seeds
    std::vector<double> x1(n2), x2(n4), x3(n4), x4(n6), x5(n6), y1(n2), y2(n4), y3(n4), y4(n6), y5(n6);
    for(auto * v : {&x1, &x2, &x3, &x4, &x5, &y1, &y2, &y3, &y4, &y5}) for(double & e : *v) e = rnd();
    std::vector<double> f1 = x1, f2 = x2, f3 = x3, f4 = x4, f5 = x5;
    finalize_copy(na, f1, f2, f3, f4, f5);
    double lhs = 0;
    for(size_t i = 0; i < n2; i++) lhs += y1[i] * f1[i];
    for(size_t i = 0; i < n4; i++) lhs += y2[i] * f2[i] + y3[i] * f3[i];
    for(size_t i = 0; i < n6; i++) lhs += y4[i] * f4[i] + y5[i] * f5[i];
    PTRawSeedStore raw;
    pt_vjp_EE_final(na, 0.0, y1, y2, y3, y4, y5, &raw);
    double rhs = 0;
    for(size_t i = 0; i < n2; i++) rhs += raw.PH[i] * x1[i];
    for(size_t i = 0; i < n4; i++) rhs += raw.PV_JK[i] * x2[i] + raw.PV_AB[i] * x3[i];
    for(size_t i = 0; i < n6; i++) rhs += raw.P3_JK[i] * x4[i] + raw.P3_AB[i] * x5[i];
    const double err = std::fabs(lhs - rhs) / (1e-12 + std::fabs(lhs));
    fprintf(out_stream, "FINAL: <ybar, F(x)> = % .12e   <F^T ybar, x> = % .12e   rel.err %.1e %s\n", lhs, rhs, err, err < 1e-12 ? "ok" : "MISMATCH");
    if(err >= 1e-12) fails++;
    return fails;
}

int pt_vjp_selftest(PT_tensors & T, const char * blocks){
    const int nc = T.n_c, na = T.n_a, nv = T.n_v;
    const long aux = T.integrals->aux_n_ao;
    const size_t n2 = (size_t)na * na, n4 = n2 * n2, n6 = n4 * n2;
    RI_data * R = T.integrals;

    std::vector<double> adjVC((size_t)nv * nc * aux), adjVA((size_t)nv * na * aux), adjCA((size_t)nc * na * aux),
                        adjCC((size_t)nc * nc * aux), adjAA((size_t)na * na * aux),
                        adj_ec(nc), adj_ea(na), adj_ev(nv), adjHAV((size_t)na * nv), adjHCA((size_t)nc * na), adjHCV((size_t)nc * nv);
    std::vector<Group> groups = {
        {"B(VC)", R->VC_RI_M, adjVC.size(), adjVC.data(), false},
        {"B(VA)", R->VA_RI_M, adjVA.size(), adjVA.data(), false},
        {"B(CA)", R->CA_RI_M, adjCA.size(), adjCA.data(), false},
        {"B(CC)", R->CC_RI_M, adjCC.size(), adjCC.data(), false},
        {"B(AA)", R->AA_RI_M, adjAA.size(), adjAA.data(), false},
        {"eps_c", T.e_c, (size_t)nc, adj_ec.data(), true},
        {"eps_a", T.e_a, (size_t)na, adj_ea.data(), true},
        {"eps_v", T.e_v, (size_t)nv, adj_ev.data(), true},
        {"H_AV", T.H_AV, adjHAV.size(), adjHAV.data(), false},
        {"H_CA", T.H_CA, adjHCA.size(), adjHCA.data(), false},
        {"H_CV", T.H_CV, adjHCV.size(), adjHCV.data(), false},
    };
    std::vector<double> adjShift(1, 0.0);
    if(pt_vjp_forward_override) groups.push_back({"shift", &pt_vjp_shift, 1, adjShift.data(), true});   // only meaningful when the forward pass honours the shift (XMC tables)

    // random seeds on every raw output
    std::vector<double> yPH(n2), yJK(n4), yAB(n4), y3J(n6), y3A(n6);
    for(double & v : yPH) v = rnd(); for(double & v : yJK) v = rnd(); for(double & v : yAB) v = rnd();
    for(double & v : y3J) v = rnd(); for(double & v : y3A) v = rnd();
    const double yPS = rnd();

    auto functional = [&](const std::string & blk){
        T.RF_PS = 0;
        set_zero_matr(T.RF_PH, n2); set_zero_matr(T.RF_PV_JK, n4); set_zero_matr(T.RF_PV_AB, n4);
        set_zero_matr(T.RF_P3_JK, n6); set_zero_matr(T.RF_P3_AB, n6);
        for(int t = 0; t < na; t++){ T.e_IP[t] = T.e_a[t]; T.e_EA[t] = T.e_a[t]; }
        run_block(T, blk);
        double s = yPS * T.RF_PS;
        for(size_t i = 0; i < n2; i++) s += yPH[i] * T.RF_PH[i];
        for(size_t i = 0; i < n4; i++) s += yJK[i] * T.RF_PV_JK[i] + yAB[i] * T.RF_PV_AB[i];
        for(size_t i = 0; i < n6; i++) s += y3J[i] * T.RF_P3_JK[i] + y3A[i] * T.RF_P3_AB[i];
        return s;
    };

    int fails = 0;
    std::string list = blocks;
    if(list.find("FINAL") != std::string::npos){ fails += pt_vjp_selftest_final(T); }
    size_t pos = 0;
    while(pos < list.size()){
        size_t c = list.find(',', pos);
        std::string blk = list.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        pos = c == std::string::npos ? list.size() : c + 1;
        if(blk.empty() || blk == "FINAL") continue;

        for(auto & g : groups) std::fill(g.adj, g.adj + g.n, 0.0);
        PTRawSeeds S; S.PS = yPS; S.PH = yPH.data(); S.PV_JK = yJK.data(); S.PV_AB = yAB.data(); S.P3_JK = y3J.data(); S.P3_AB = y3A.data();
        // negative control (NOPT_VJP_NEGCTRL=1): the adjoint is not given the 3-body seeds the functional contains
        // -- AV, CA and ALL must then be reported as MISMATCH
        if(getenv("NOPT_VJP_NEGCTRL") && *getenv("NOPT_VJP_NEGCTRL")){ S.P3_JK = nullptr; S.P3_AB = nullptr; }
        PTInputAdj A;
        A.VC = adjVC.data(); A.VA = adjVA.data(); A.CA = adjCA.data(); A.CC = adjCC.data(); A.AA = adjAA.data();
        A.e_c = adj_ec.data(); A.e_a = adj_ea.data(); A.e_v = adj_ev.data();
        A.H_AV = adjHAV.data(); A.H_CA = adjHCA.data(); A.H_CV = adjHCV.data();
        A.shift_adj = 0.0;
        if(vjp_block(T, blk, S, A)){ fprintf(out_stream, "VJP %-5s not available\n", blk.c_str()); fails++; continue; }
        adjShift[0] = A.shift_adj;

        fprintf(out_stream, "\nVJP test, block %s (dot product <adjoint, dx> against central differences):\n", blk.c_str());
        for(auto & g : groups){
            if(g.x == nullptr) continue;
            std::vector<double> dx(g.n), save(g.x, g.x + g.n);
            double scale = 0; for(size_t i = 0; i < g.n; i++) scale = std::max(scale, std::fabs(save[i]));
            if(scale == 0) scale = 1;
            for(size_t i = 0; i < g.n; i++) dx[i] = rnd() * scale * 0.5;
            double ana = 0; for(size_t i = 0; i < g.n; i++) ana += g.adj[i] * dx[i];
            const double h = g.is_energy ? 1e-4 : 1e-4;
            auto at = [&](double k){ for(size_t i = 0; i < g.n; i++) g.x[i] = save[i] + k * h * dx[i]; return functional(blk); };
            const double fd = (-at(2) + 8 * at(1) - 8 * at(-1) + at(-2)) / (12 * h);
            memcpy(g.x, save.data(), sizeof(double) * g.n);
            for(int t = 0; t < na; t++){ T.e_IP[t] = T.e_a[t]; T.e_EA[t] = T.e_a[t]; }
            const double err = std::fabs(ana - fd) / (1e-12 + std::max(std::fabs(ana), std::fabs(fd)));
            const bool zero = std::fabs(ana) < 1e-14 && std::fabs(fd) < 1e-10;
            const bool ok = zero || err < 1e-6;
            if(!ok) fails++;
            fprintf(out_stream, "   %-6s analytic % .10e  FD % .10e  rel.err %.1e  %s\n", g.name.c_str(), ana, fd, err, ok ? "ok" : "MISMATCH");
        }
    }
    return fails;
}
