// derivtest: the derivative integrals of the analytic gradients against finite differences.
//
// Reads an ordinary NOPT input (geometry, basis, RI=0/1), then compares every derivative-integral
// contraction of include/integral_derivatives.h with a fourth-order central finite difference of the
// SAME scalar built from NOPT's own energy integrals (the production AO routines, or the energy
// engines for the 4-center ERI), over every Cartesian coordinate of every atom. The weights are
// fixed random numbers, not a physical density, so a wrong term cannot hide behind a
// cancellation. Also reported: the sum over atoms of each gradient (translational invariance).
//
//   run_derivtest -i input.inp          exit status 1 if any deviation exceeds the tolerance
#include <stdio.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "molecule.h"
#include "inp_par_read.h"
#include "common_vars.h"
#include "nopt_libint_engines.h"
#include "libint_link.h"
#include "libint_functions.h"
#include "integral_derivatives.h"
#include "ri_gradient.h"
#include "RI.h"
#include "matr.h"

static unsigned long long rng_state = 0x9e3779b97f4a7c15ULL;
static double rnd(){   // deterministic in [-1, 1)
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((rng_state >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}
static void fill_random(std::vector<double> & v){ for(double & x : v) x = rnd(); }
static void symmetrize(std::vector<double> & m, int n){
    for(int i = 0; i < n; i++) for(int j = 0; j < i; j++){ double a = 0.5 * (m[i * n + j] + m[j * n + i]); m[i * n + j] = m[j * n + i] = a; }
}

static int failures = 0;
static const double TOL = 1e-7;

// scalar(R) -> central 4th-order FD gradient over all atom coordinates
static std::vector<double> fd_gradient(molecule * M, const std::function<double()> & scalar, double h){
    std::vector<double> g(3 * M->n_atoms);
    for(int i = 0; i < 3 * M->n_atoms; i++){
        const double x0 = M->atom_coord[i];
        double f[4];
        const double d[4] = {2 * h, h, -h, -2 * h};
        for(int k = 0; k < 4; k++){
            M->atom_coord[i] = x0 + d[k];
            M->sync_shell_to_geom();
            nopt_engines_link_mol(M);
            f[k] = scalar();
        }
        M->atom_coord[i] = x0;
        M->sync_shell_to_geom();
        nopt_engines_link_mol(M);
        g[i] = (-f[0] + 8 * f[1] - 8 * f[2] + f[3]) / (12 * h);
    }
    return g;
}

static void compare(const char * name, molecule * M, const std::vector<double> & an, const std::vector<double> & fd, bool check_translation){
    double worst = 0, scale = 0;
    int wi = 0;
    for(size_t i = 0; i < an.size(); i++){
        double e = std::fabs(an[i] - fd[i]);
        if(e > worst){ worst = e; wi = (int)i; }
        scale = std::max(scale, std::fabs(fd[i]));
    }
    double tr[3] = {0, 0, 0};
    for(int a = 0; a < M->n_atoms; a++) for(int q = 0; q < 3; q++) tr[q] += an[3 * a + q];
    const double trm = std::max(std::fabs(tr[0]), std::max(std::fabs(tr[1]), std::fabs(tr[2])));
    const bool ok = worst < TOL * std::max(1.0, scale) && (!check_translation || trm < 1e-9 * std::max(1.0, scale));
    printf("%-26s max|analytic-FD| = %.2e (worst atom %d xyz %d)  max|grad| = %.2e  translation sum = %.1e  %s\n",
           name, worst, wi / 3, wi % 3, scale, check_translation ? trm : 0.0, ok ? "PASS" : "FAIL");
    if(!ok) failures++;
}

int main(int argc, char ** argv){
    out_stream = stdout;
    nQM = 0;
    molecule M;
    char * P_name = nullptr;
    inp_par P;
    for(int i = 1; i < argc; i++) if(strstr(argv[i], "-i") != NULL){ P_name = argv[i + 1]; i++; }
    if(!P_name){ fprintf(stderr, "usage: run_derivtest -i input\n"); return 2; }
    P.read(P_name);
    nopt_initialize(P.num_threads);
    molecule_read_by_inp_par(&M, &P);
    nopt_engines_link_mol(&M);
    M.gen_1el_data();
    if(deriv_check_supported(&M)) return 2;
    nopt_deriv_engines_initialize(&M);

    const int n = M.n_ao, na = M.n_atoms;
    printf("\n== derivtest: %d atoms, %d Cartesian AOs, %zu shells, RI=%d (%zu auxiliary shells), %d threads ==\n\n",
           na, n, M.s.size(), RI, M.ri_s.size(), num_threads);
    const double h = 2e-3;

    std::vector<double> Wg(n * n), Ws(n * n);
    fill_random(Wg); fill_random(Ws); symmetrize(Ws, n);

    // ---- one-body ----------------------------------------------------------
    auto onebody = [&](const char * name, char kind, const std::vector<double> & W,
                       int (*contract)(molecule *, const double *, double *)){
        std::vector<double> an(3 * na, 0.0);
        contract(&M, W.data(), an.data());
        auto fd = fd_gradient(&M, [&]{
            std::vector<double> A(n * n, 0.0);
            AO_1el_from_2shells(A.data(), M.s, M.s, n, n, kind, 0);
            double s = 0; for(int i = 0; i < n * n; i++) s += W[i] * A[i];
            return s; }, h);
        compare(name, &M, an, fd, kind != 'v' || true);
    };
    onebody("overlap (general W)", 's', Wg, deriv_overlap_contract);
    onebody("overlap (symmetric W)", 's', Ws, deriv_overlap_contract);
    onebody("kinetic (symmetric W)", 't', Ws, deriv_kinetic_contract);
    onebody("nuclear attraction", 'v', Ws, deriv_nuclear_contract);

    // ---- nuclear repulsion --------------------------------------------------
    {
        std::vector<double> an(3 * na, 0.0);
        deriv_nuclear_repulsion(&M, an.data());
        auto fd = fd_gradient(&M, [&]{ return E_nuc_calc(&M); }, h);
        compare("nuclear repulsion", &M, an, fd, true);
    }

    if(RI == 0){
        // ---- 4-center ERI, general tensor and the J/K convenience ---------------
        std::vector<size_t> off = map_shell_to_basis_function(M.s);
        auto eri_scalar = [&](const std::function<double(int, int, int, int)> & G){
            double s = 0;
            const int ns = (int)M.s.size();
            auto & eng = e_g_engines[0];
            for(int i = 0; i < ns; i++) for(int j = 0; j < ns; j++) for(int k = 0; k < ns; k++) for(int l = 0; l < ns; l++){
                eng.compute(M.s[i], M.s[j], M.s[k], M.s[l]);
                const double * b = eng.results()[0];
                if(!b) continue;
                const int n0 = (int)M.s[i].size(), n1 = (int)M.s[j].size(), n2 = (int)M.s[k].size(), n3 = (int)M.s[l].size();
                for(int a = 0; a < n0; a++) for(int bb = 0; bb < n1; bb++) for(int c = 0; c < n2; c++) for(int d = 0; d < n3; d++)
                    s += G(off[i] + a, off[j] + bb, off[k] + c, off[l] + d) * b[((a * n1 + bb) * n2 + c) * n3 + d];
            }
            return s;
        };
        // a random rank-4 tensor from a hash of the indices, with no symmetry whatsoever
        auto Gfun = [&](int a, int b, int c, int d){
            unsigned long long x = ((((unsigned long long)a * 1315423911ULL) ^ ((unsigned long long)b * 2654435761ULL)
                                   ^ ((unsigned long long)c * 97531ULL) ^ ((unsigned long long)d * 40503ULL)) * 0x9e3779b97f4a7c15ULL);
            x ^= x >> 29; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 32;
            return ((x >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
        };
        {
            std::vector<double> an(3 * na, 0.0);
            deriv_eri_contract(&M, [&](const int *, const int * o, const int * nn, double * w){
                for(int a = 0; a < nn[0]; a++) for(int b = 0; b < nn[1]; b++) for(int c = 0; c < nn[2]; c++) for(int d = 0; d < nn[3]; d++)
                    w[((a * nn[1] + b) * nn[2] + c) * nn[3] + d] = Gfun(o[0] + a, o[1] + b, o[2] + c, o[3] + d);
            }, an.data());
            compare("ERI (random rank-4 G)", &M, an, fd_gradient(&M, [&]{ return eri_scalar(Gfun); }, h), true);
        }
        {
            std::vector<double> D(n * n); fill_random(D); symmetrize(D, n);
            std::vector<double> an(3 * na, 0.0);
            deriv_eri_JK_contract(&M, D.data(), 0.5, -0.25, an.data());
            compare("ERI J/K (cJ=.5 cK=-.25)", &M, an, fd_gradient(&M, [&]{
                return eri_scalar([&](int a, int b, int c, int d){ return 0.5 * D[a * n + b] * D[c * n + d] - 0.25 * D[a * n + c] * D[b * n + d]; }); }, h), true);
        }
    }else{
        // ---- RI: (P|ab) and (P|Q) -------------------------------------------------
        const int naux = (int)M.ri_s.size() ? [&]{ int t = 0; for(auto & s : M.ri_s) t += (int)s.size(); return t; }() : 0;
        std::vector<size_t> aoff = map_shell_to_basis_function(M.ri_s);
        std::vector<double> G3((size_t)n * n * naux), G2((size_t)naux * naux);
        fill_random(G3); fill_random(G2);
        // (P|ab) is symmetric in a,b, so only the symmetric part of G3 in (a,b) is probed
        for(int a = 0; a < n; a++) for(int b = 0; b < a; b++) for(int p = 0; p < naux; p++){
            double s = 0.5 * (G3[((size_t)a * n + b) * naux + p] + G3[((size_t)b * n + a) * naux + p]);
            G3[((size_t)a * n + b) * naux + p] = G3[((size_t)b * n + a) * naux + p] = s;
        }
        symmetrize(G2, naux);   // (P|Q) is symmetric
        {
            std::vector<double> an(3 * na, 0.0);
            std::vector<size_t> off = map_shell_to_basis_function(M.s);
            deriv_ri3_contract(&M, [&](const int * sh, const int * o, const int * nn, double * w){
                (void)sh;
                for(int p = 0; p < nn[0]; p++) for(int a = 0; a < nn[1]; a++) for(int b = 0; b < nn[2]; b++)
                    w[(p * nn[1] + a) * nn[2] + b] = G3[((size_t)(o[1] + a) * n + o[2] + b) * naux + o[0] + p];
            }, an.data());
            deriv_ri3_translation_check(&M, 20);
            compare("RI (P|ab)", &M, an, fd_gradient(&M, [&]{
                std::vector<double> I((size_t)n * n * naux);
                calc_2el_AO_INTS_RI(M.s, n, M.ri_s, naux, I.data());
                double s = 0; for(size_t i = 0; i < I.size(); i++) s += G3[i] * I[i];
                return s; }, h), true);
        }
        {
            std::vector<double> an(3 * na, 0.0);
            deriv_ri2_contract(&M, [&](const int *, const int * o, const int * nn, double * w){
                for(int p = 0; p < nn[0]; p++) for(int q = 0; q < nn[1]; q++)
                    w[p * nn[1] + q] = G2[(size_t)(o[0] + p) * naux + o[1] + q];
            }, an.data());
            compare("RI (P|Q)", &M, an, fd_gradient(&M, [&]{
                std::vector<double> V((size_t)naux * naux, 0.0);
                RI_V_calc(M.ri_s, naux, V.data());
                double s = 0; for(size_t i = 0; i < V.size(); i++) s += G2[i] * V[i];
                return s; }, h), true);
        }
    }

    if(RI != 0){
        // ---- the whole RI Coulomb/exchange chain against the PRODUCTION Fock builder -----------
        // E = sum DM_ab (2 J_ab - K_ab) for a fixed set of random (non-orthonormal) occupied rows C,
        // DM = C^T C, with J, K from DM_to_F_transform_RI (L, V^-1/2 as the SCF uses them). The
        // analytic side is ri_jk_vjp (P = 2 DM, cJ = 1/2, cK = -1/4) contracted with the derivative
        // integrals. No SCF involved: this isolates the integrals and the fitting formulas.
        const int nc = M.n_el_calc / 2;
        long naux = 0; for(auto & sh : M.ri_s) naux += sh.size();
        std::vector<double> C((size_t)nc * n); fill_random(C);
        for(double & x : C) x *= 0.5;
        std::vector<double> DM((size_t)n * n, 0.0), P((size_t)n * n);
        for(int k = 0; k < nc; k++) for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) DM[a * n + b] += C[k * n + a] * C[k * n + b];
        for(size_t i = 0; i < DM.size(); i++) P[i] = 2.0 * DM[i];

        std::vector<double> L((size_t)n * n * naux), Vm((size_t)naux * naux, 0.0);
        calc_2el_AO_INTS_RI(M.s, n, M.ri_s, (int)naux, L.data());
        RI_V_calc(M.ri_s, (int)naux, Vm.data());
        double lmin, lmax; ri_eigenvalue_range((int)naux, Vm.data(), &lmin, &lmax);
        printf("RI metric: %ld auxiliary functions, eigenvalues %.3e .. %.3e (condition %.1e)\n", naux, lmin, lmax, lmax / lmin);
        std::vector<double> U(Vm), ev(naux), Vinv((size_t)naux * naux, 0.0);
        lapack_diag(U.data(), ev.data(), (int)naux);
        for(long k = 0; k < naux; k++) for(long i = 0; i < naux; i++) for(long j = 0; j < naux; j++) Vinv[i * naux + j] += U[k * naux + i] * U[k * naux + j] / ev[k];
        std::vector<double> Lbar(L.size()), Vbar((size_t)naux * naux);
        ri_jk_vjp(n, (int)naux, L.data(), Vm.data(), P.data(), 0.5, -0.25, Lbar.data(), Vbar.data());
        std::vector<double> an(3 * na, 0.0);
        deriv_ri3_contract(&M, [&](const int *, const int * o, const int * nn, double * w){
            for(int p = 0; p < nn[0]; p++) for(int a = 0; a < nn[1]; a++) for(int b = 0; b < nn[2]; b++)
                w[(p * nn[1] + a) * nn[2] + b] = Lbar[((size_t)(o[1] + a) * n + o[2] + b) * naux + o[0] + p]; }, an.data());
        deriv_ri2_contract(&M, [&](const int *, const int * o, const int * nn, double * w){
            for(int p = 0; p < nn[0]; p++) for(int q = 0; q < nn[1]; q++) w[p * nn[1] + q] = Vbar[(size_t)(o[0] + p) * naux + o[1] + q]; }, an.data());
        auto prod_energy = [&]{
            regen_RI_AA(&M);
            std::vector<double> J(n * n), K(n * n);
            DM_to_F_transform_RI(J.data(), K.data(), DM.data(), C.data(), nc, n);
            double e = 0; for(int i = 0; i < n * n; i++) e += DM[i] * (2 * J[i] - K[i]);
            return e; };
        {   // the energy itself: production builder against E = 1/2 sum P P [J - K/2] with explicit L and V^-1
            double e_prod = prod_energy();
            double e_form = 0;
            std::vector<double> t(naux, 0.0), c(naux, 0.0);
            for(size_t ab = 0; ab < (size_t)n * n; ab++) for(long p = 0; p < naux; p++) t[p] += P[ab] * L[ab * naux + p];
            for(long p = 0; p < naux; p++) for(long q = 0; q < naux; q++) c[p] += Vinv[p * naux + q] * t[q];
            double ej = 0; for(long p = 0; p < naux; p++) ej += t[p] * c[p];
            double ek = 0;   // sum P_ac P_bd g_abcd, via T_Q = P L_Q P and g = L Vinv L^T
            std::vector<double> LQ((size_t)n * n), X((size_t)n * n);
            std::vector<double> Tq((size_t)naux * n * n);
            for(long q = 0; q < naux; q++){
                for(size_t ab = 0; ab < (size_t)n * n; ab++) LQ[ab] = L[ab * naux + q];
                for(int a = 0; a < n; a++) for(int b = 0; b < n; b++){ double s2 = 0; for(int cc = 0; cc < n; cc++) s2 += P[a * n + cc] * LQ[cc * n + b]; X[a * n + b] = s2; }
                for(int a = 0; a < n; a++) for(int b = 0; b < n; b++){ double s2 = 0; for(int cc = 0; cc < n; cc++) s2 += X[a * n + cc] * P[cc * n + b]; Tq[(size_t)q * n * n + a * n + b] = s2; }
            }
            for(long p = 0; p < naux; p++) for(long q = 0; q < naux; q++){ double m = 0; for(size_t ab = 0; ab < (size_t)n * n; ab++) m += L[ab * naux + p] * Tq[(size_t)q * n * n + ab]; ek += Vinv[p * naux + q] * m; }
            e_form = 0.5 * ej - 0.25 * ek;
            printf("RI J/K energy at the central geometry: production %.15f  formula %.15f  diff %.2e\n", e_prod, e_form, e_prod - e_form);
        }
        // several step sizes: a smooth model error would not move with h, rounding noise would
        for(double hh : {4e-3, 2e-3, 1e-3}){
            char nm[64]; snprintf(nm, sizeof nm, "RI J/K, production builder h=%g", hh);
            compare(nm, &M, an, fd_gradient(&M, prod_energy, hh), true);
        }
    }

    nopt_deriv_engines_finalize();
    nopt_finalize();
    printf("\n%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
