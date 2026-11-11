// resptest: checks of the CAS-SCF response machinery (casscf_resp.h) on a converged SA-CASSCF.
//   run_resptest -i input.inp
//   1. the CI kernel against the production CAS: active integrals, H c = E c, densities;
//   2. the residual r(x*) (orbital rotations and CI) is zero;
//   3. the Hessian J = d r / d x by central differences: symmetry (random pairs of directions) and the dense solve against the CG solve.
#include <stdio.h>
#include <string.h>
#include <random>

#include "molecule.h"
#include "inp_par_read.h"
#include "inp_out.h"
#include "common_vars.h"
#include "nopt_libint_engines.h"
#include "QM_calc.h"
#include "CAS.h"
#include "blas_link.h"
#include "matr.h"
#include "casscf_resp.h"

int main(int argc, char ** argv){
    out_stream = stdout;
    nQM = 0;
    molecule M;
    char * P_name = nullptr;
    inp_par P;
    for(int i = 1; i < argc; i++) if(strstr(argv[i], "-i") != NULL){ P_name = argv[i + 1]; i++; }
    if(!P_name){ fprintf(stderr, "usage: run_resptest -i input\n"); return 2; }
    P.read(P_name);
    nopt_initialize(P.num_threads);
    molecule_read_by_inp_par(&M, &P);
    nopt_engines_link_mol(&M);
    single_point_calc(&P, &M);

    CasRef R;
    if(R.build(&M)) return 3;
    CAS_engine * CAS = M.CAS;
    const int na = R.na;
    printf("\nRESPTEST: n_ao %d, core %d, active %d, virtual %d, naux %d, states %d, Nd %zu, nk %zu, nx %zu\n", R.n, R.nc, na, R.nv, R.naux, R.ns, R.Nd, R.nk(), R.nx());

    // 1. kernel against production: reference residual and energies
    std::vector<double> r0; double Esa;
    cas_residual(R, R.C0.data(), R.c0.data(), &r0, nullptr, &Esa);
    double rk = 0, rc = 0;
    for(size_t k = 0; k < R.nk(); k++) rk = std::max(rk, std::fabs(r0[k]));
    for(size_t k = R.nk(); k < R.nx(); k++) rc = std::max(rc, std::fabs(r0[k]));
    printf("RESPTEST residual at the reference: max |r_kappa| = %.3e, max |r_delta| = %.3e   E_SA(rebuilt) = %.12f\n", rk, rc, Esa);
    CasObjects ob; std::vector<double> wsa = R.w;
    cas_objects(R, R.C0.data(), R.c0.data(), R.ns, wsa.data(), false, &ob);
    printf("RESPTEST E_SA from densities %.12f, CI eigenvalues:", ob.E);
    for(int I = 0; I < R.ns; I++) printf(" %.12f", R.ci_template->E_states[0][R.states[I]]);
    printf("\n");
    for(int I = 0; I < R.ns; I++){
        std::vector<double> e(R.ns, 0.0); e[I] = 1.0; CasObjects oi;
        cas_objects(R, R.C0.data(), R.c0.data(), R.ns, e.data(), false, &oi);
        printf("RESPTEST root %d: energy from densities %.12f, CI eigenvalue %.12f, difference %.2e\n", R.states[I], oi.E, R.ci_template->E_states[0][R.states[I]], oi.E - R.ci_template->E_states[0][R.states[I]]);
    }

    // 3. Hessian symmetry and dense vs CG
    std::mt19937_64 gen(5);
    auto rnd = [&](){ return std::uniform_real_distribution<double>(-1, 1)(gen); };
    std::vector<double> p(R.nx()), q(R.nx()), Jp, Jq;
    for(double & v : p) v = rnd();
    for(double & v : q) v = rnd();
    {   // keep the CI parts in the complement
        for(auto * v : {&p, &q}){
            for(int I = 0; I < R.ns; I++) for(int J = 0; J < R.ns; J++){ double o = 0; for(size_t d = 0; d < R.Nd; d++) o += R.c0[(size_t)J * R.Nd + d] * (*v)[R.nk() + (size_t)I * R.Nd + d];
                for(size_t d = 0; d < R.Nd; d++) (*v)[R.nk() + (size_t)I * R.Nd + d] -= o * R.c0[(size_t)J * R.Nd + d]; }
        }
    }
    cas_hessian_vec(R, p, &Jp); cas_hessian_vec(R, q, &Jq);
    double pJq = 0, qJp = 0; for(size_t i = 0; i < R.nx(); i++){ pJq += p[i] * Jq[i]; qJp += q[i] * Jp[i]; }
    printf("RESPTEST Hessian symmetry: p.J.q = % .10e   q.J.p = % .10e   rel. diff %.2e\n", pJq, qJp, std::fabs(pJq - qJp) / std::max(std::fabs(pJq), 1e-30));
    double pJp = 0; for(size_t i = 0; i < R.nx(); i++) pJp += p[i] * Jp[i];
    printf("RESPTEST p.J.p = % .6e (positive for a minimum)\n", pJp);
    // orbital-only and CI-only blocks (is the CI-orbital coupling there?)
    {
        std::vector<double> po(p), pc(p);
        for(size_t i = R.nk(); i < R.nx(); i++) po[i] = 0;
        for(size_t i = 0; i < R.nk(); i++) pc[i] = 0;
        std::vector<double> Jo, Jc; cas_hessian_vec(R, po, &Jo); cas_hessian_vec(R, pc, &Jc);
        double oo = 0, oc = 0, co = 0, cc = 0;
        for(size_t i = 0; i < R.nk(); i++){ oo += po[i] * Jo[i]; oc += po[i] * Jc[i]; }
        for(size_t i = R.nk(); i < R.nx(); i++){ co += pc[i] * Jo[i]; cc += pc[i] * Jc[i]; }
        printf("RESPTEST blocks: oo % .6e  oc % .6e  co % .6e  cc % .6e\n", oo, oc, co, cc);
    }
    // 4. exact J p (cas_jvp) against the 4-point difference of the analytic residual (the former implementation, now the oracle)
    {
        std::vector<double> Jfd, Jan;
        cas_hessian_vec_fd(R, p, &Jfd); cas_hessian_vec(R, p, &Jan);
        double e = 0, sc = 0, eo = 0, ec = 0;
        for(size_t i = 0; i < R.nx(); i++){ e = std::max(e, std::fabs(Jfd[i] - Jan[i])); sc = std::max(sc, std::fabs(Jfd[i])); if(i < R.nk()) eo = std::max(eo, std::fabs(Jfd[i] - Jan[i])); else ec = std::max(ec, std::fabs(Jfd[i] - Jan[i])); }
        printf("RESPTEST exact J.p vs FD oracle: max |diff| = %.2e (orbital %.2e, CI %.2e), scale %.2e, rel %.2e  %s\n", e, eo, ec, sc, e / sc, e / sc < 1e-7 ? "OK" : "MISMATCH");
        // also the pure-orbital and pure-CI directions
        for(int part = 0; part < 2; part++){
            std::vector<double> pp(p);
            for(size_t i = 0; i < R.nx(); i++) if((part == 0) != (i < R.nk())) pp[i] = 0.0;
            cas_hessian_vec_fd(R, pp, &Jfd); cas_hessian_vec(R, pp, &Jan);
            double e2 = 0, s2 = 0; for(size_t i = 0; i < R.nx(); i++){ e2 = std::max(e2, std::fabs(Jfd[i] - Jan[i])); s2 = std::max(s2, std::fabs(Jfd[i])); }
            printf("RESPTEST exact J.p vs FD, %s-only direction: rel %.2e  %s\n", part ? "CI" : "orbital", e2 / s2, e2 / s2 < 1e-7 ? "OK" : "MISMATCH");
        }
        std::vector<double> Jq2; cas_hessian_vec(R, q, &Jq2);
        double a = 0, b = 0; for(size_t i = 0; i < R.nx(); i++) a += p[i] * Jq2[i];
        std::vector<double> Jp2; cas_hessian_vec(R, p, &Jp2);
        for(size_t i = 0; i < R.nx(); i++) b += q[i] * Jp2[i];
        printf("RESPTEST exact Hessian symmetry: p.J.q = % .12e  q.J.p = % .12e  rel %.2e  %s\n", a, b, std::fabs(a - b) / std::fabs(a), std::fabs(a - b) / std::fabs(a) < 1e-8 ? "OK" : "MISMATCH");
    }
    // 5. exact derivative of the SA objects (D, Lbar, Vbar, Wao) against the 4-point difference along p
    {
        CasObjects an; cas_jvp(R, p.data(), nullptr, &an, true);
        double mx = 0; for(double v : p) mx = std::max(mx, std::fabs(v));
        for(double step : {2e-3, 5e-4}){
        const double eps = step / mx, k[4] = {2.0, 1.0, -1.0, -2.0}, f[4] = {-1.0, 8.0, -8.0, 1.0};
        printf("RESPTEST object derivatives, FD step %.0e / max|p|:\n", step);
        CasObjects o[4];
        for(int i = 0; i < 4; i++){ std::vector<double> Cd, cd, rr; double Es; cas_displace(R, p.data(), k[i] * eps, &Cd, &cd); cas_residual(R, Cd.data(), cd.data(), &rr, &o[i], &Es); }
        auto cmp = [&](const char * name, std::vector<double> CasObjects::* m){
            double e = 0, sc = 0;
            for(size_t j = 0; j < (an.*m).size(); j++){
                double fd = 0; for(int i = 0; i < 4; i++) fd += f[i] * (o[i].*m)[j] / (12.0 * eps);
                e = std::max(e, std::fabs(fd - (an.*m)[j])); sc = std::max(sc, std::fabs(fd));
            }
            printf("RESPTEST exact d%s vs FD: max |diff| %.2e, scale %.2e, rel %.2e  %s\n", name, e, sc, e / std::max(sc, 1e-300), e / std::max(sc, 1e-300) < 1e-7 ? "OK" : "MISMATCH");
        };
        cmp("D", &CasObjects::D); cmp("Lbar", &CasObjects::Lbar); cmp("Vbar", &CasObjects::Vbar); cmp("Wao", &CasObjects::Wao); cmp("GC", &CasObjects::GC);
        }
    }
    // 6. the CI generator of an active rotation: d/dtheta <cbar| malmqvist(U(theta)) c> against the transition density (random vectors, not stationary)
    {
        const int ns = R.ns, na = R.na;
        std::vector<double> cr((size_t)ns * R.Nd), cb((size_t)ns * R.Nd);
        for(double & v : cr) v = rnd();
        for(double & v : cb) v = rnd();
        std::vector<double> T((size_t)na * na), Tt((size_t)na * na);
        cas_transition_rdm(R, cb.data(), cr.data(), ns, T.data());     // <cbar|E_tu|c>
        for(int tr = 0; tr < 2; tr++){
            std::vector<double> g((size_t)na * na);
            cas_ci_rotation_g_fd(R, cr.data(), cb.data(), ns, g.data(), tr == 1);
            double ep = 0, em = 0, sc = 0;
            for(int t = 0; t < na; t++) for(int u = t + 1; u < na; u++){
                const double a = T[t * na + u] - T[u * na + t];
                ep = std::max(ep, std::fabs(g[t * na + u] - a)); em = std::max(em, std::fabs(g[t * na + u] + a)); sc = std::max(sc, std::fabs(a));
            }
            printf("RESPTEST CI rotation generator (malmqvist U %s): |g - (T_tu - T_ut)| = %.2e, |g + (T_tu - T_ut)| = %.2e, scale %.2e\n", tr ? "transposed" : "as built", ep, em, sc);
        }
    }
    nopt_finalize();
    return 0;
}
