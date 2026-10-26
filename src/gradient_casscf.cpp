#include "gradient_casscf.h"

#include <cmath>
#include <vector>

#include "blas_link.h"
#include "CAS.h"
#include "common_vars.h"
#include "integral_derivatives.h"
#include "libint_link.h"
#include "matr.h"
#include "ri_gradient.h"

static inline void gemm(int m, int n, int k, const double * A, const double * B, double * C, double beta = 0.0, bool tA = false, bool tB = false){
    cblas_dgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans, m, n, k, 1.0,
                A, tA ? m : k, B, tB ? k : n, beta, C, n);
}

int casscf_sa_gradient(molecule * M, double * g, double * E_out){
    if(deriv_check_supported(M)) return 1;
    if(RI == 0){ fprintf(out_stream, "ERROR: the CAS-SCF gradient is implemented for the RI energy only\n"); return 1; }
    CAS_engine * C = M->CAS;
    if(C == nullptr){ fprintf(out_stream, "ERROR: casscf_sa_gradient needs a molecule with a CAS_SCF calculation behind it\n"); return 1; }
    const int n = M->n_ao, nc = C->n_core, na = C->n_act, nmo = M->n_mo, na_atoms = M->n_atoms;
    const size_t n2 = (size_t)n * n, a2 = (size_t)na * na, a4 = a2 * a2;

    // averaged densities as the orbital optimizer builds them (gamma, GAMMA averaged with wstate_actual)
    const double E_cas = C->SA_grad_hess_calc();

    // orbital sets as AO matrices: rows of MO_VEC are the MOs
    std::vector<double> Cc((size_t)n * std::max(nc, 1)), Ca((size_t)n * std::max(na, 1)), Call((size_t)n * nmo);
    for(int a = 0; a < n; a++){
        for(int i = 0; i < nc; i++) Cc[(size_t)a * nc + i] = M->MO_VEC[(size_t)i * n + a];
        for(int t = 0; t < na; t++) Ca[(size_t)a * na + t] = M->MO_VEC[(size_t)(nc + t) * n + a];
        for(int p = 0; p < nmo; p++) Call[(size_t)a * nmo + p] = M->MO_VEC[(size_t)p * n + a];
    }
    // d = 2 C_c C_c^T (spin-summed core density), gam_AO = C_a gamma C_a^T, D = d + gam_AO
    std::vector<double> d(n2, 0.0), gam_ao(n2, 0.0), D(n2), tmp((size_t)n * std::max(na, 1));
    if(nc) gemm(n, n, nc, Cc.data(), Cc.data(), d.data(), 0.0, false, true);
    for(double & x : d) x *= 2.0;
    if(na){
        gemm(n, na, na, Ca.data(), C->gamma, tmp.data());
        gemm(n, n, na, tmp.data(), Ca.data(), gam_ao.data(), 0.0, false, true);
    }
    for(size_t k = 0; k < n2; k++) D[k] = d[k] + gam_ao[k];

    // cumulant-like remainder  lam_tuvw = Gamma_tuvw - gamma_tu gamma_vw + 1/2 gamma_tv gamma_uw   (pairing (tu|vw))
    std::vector<double> lam(a4);
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
        const double gm = C->gamma[t * na + u] * C->gamma[v * na + w] - 0.5 * C->gamma[t * na + v] * C->gamma[u * na + w];
        lam[(((size_t)t * na + u) * na + v) * na + w] = C->GAMMA[(((size_t)t * na + u) * na + v) * na + w] - gm;
    }

    // RI integrals of this geometry
    long naux = 0; for(auto & s : M->ri_s) naux += s.size();
    std::vector<double> L(n2 * naux), Vm((size_t)naux * naux, 0.0);
    calc_2el_AO_INTS_RI(M->s, n, M->ri_s, (int)naux, L.data());
    RI_V_calc(M->ri_s, (int)naux, Vm.data());
    double lmin, lmax; ri_eigenvalue_range((int)naux, Vm.data(), &lmin, &lmax);
    if(!(lmin > 1e-12)){ fprintf(out_stream, "ERROR: the RI Coulomb metric is (nearly) rank deficient (lowest eigenvalue %.3e)\n", lmin); return 1; }
    fprintf(out_stream, "RI Coulomb metric: %ld auxiliary functions, eigenvalues %.3e .. %.3e (condition number %.2e)\n", naux, lmin, lmax, lmax / lmin);

    // energy rebuilt from the same pieces
    std::vector<double> J(n2), K(n2), Fd(n2);
    ri_fock_JK(n, (int)naux, L.data(), Vm.data(), D.data(), J.data(), K.data());
    double e1 = 0, ej = 0, ek = 0;
    for(size_t k = 0; k < n2; k++){ e1 += D[k] * M->H_AO[k]; ej += D[k] * J[k]; ek += D[k] * K[k]; }
    double e_act = 0, e_lam = 0;
    {
        std::vector<double> dummyL(n2 * naux, 0.0), dummyV((size_t)naux * naux, 0.0);
        ri_active_vjp(n, (int)naux, L.data(), Vm.data(), Ca.data(), na, lam.data(), dummyL.data(), dummyV.data(), nullptr, &e_lam);
        (void)e_act;
    }
    const double E_rebuilt = M->V_nuc + e1 + 0.5 * ej - 0.25 * ek + e_lam;
    fprintf(out_stream, "CAS-SCF functional identity: E_SA(NOPT) = % .12f   rebuilt from densities and integrals = % .12f   difference % .2e\n",
            E_cas, E_rebuilt, E_rebuilt - E_cas);
    if(E_out) *E_out = E_rebuilt;

    // ---- gradient ----------------------------------------------------------------------------------
    std::vector<double> grad(3 * na_atoms, 0.0);
    deriv_nuclear_repulsion(M, grad.data());
    deriv_kinetic_contract(M, D.data(), grad.data());
    deriv_nuclear_contract(M, D.data(), grad.data());

    std::vector<double> Lbar(n2 * naux), Vbar((size_t)naux * naux);
    ri_jk_vjp(n, (int)naux, L.data(), Vm.data(), D.data(), 0.5, -0.25, Lbar.data(), Vbar.data());
    std::vector<double> GCa((size_t)n * std::max(na, 1), 0.0);
    double e_lam2;
    if(na) ri_active_vjp(n, (int)naux, L.data(), Vm.data(), Ca.data(), na, lam.data(), Lbar.data(), Vbar.data(), GCa.data(), &e_lam2);
    deriv_ri3_contract(M, [&](const int *, const int * o, const int * nn, double * w){
        for(int p = 0; p < nn[0]; p++) for(int a = 0; a < nn[1]; a++) for(int b = 0; b < nn[2]; b++)
            w[(p * nn[1] + a) * nn[2] + b] = Lbar[((size_t)(o[1] + a) * n + o[2] + b) * naux + o[0] + p];
    }, grad.data());
    deriv_ri2_contract(M, [&](const int *, const int * o, const int * nn, double * w){
        for(int p = 0; p < nn[0]; p++) for(int q = 0; q < nn[1]; q++) w[p * nn[1] + q] = Vbar[(size_t)(o[0] + p) * naux + o[1] + q];
    }, grad.data());

    // W: unconstrained orbital gradient G_C (n x nmo, columns of core and active MOs), W_mo = 1/2 sym(C^T G_C) / ...
    for(size_t k = 0; k < n2; k++) Fd[k] = M->H_AO[k] + J[k] - 0.5 * K[k];            // F[D] = h + J[D] - K[D]/2
    std::vector<double> GC((size_t)n * nmo, 0.0), FC((size_t)n * std::max(nc, na)), tmp2;
    if(nc){
        gemm(n, nc, n, Fd.data(), Cc.data(), FC.data());                              // F C_c
        for(int a = 0; a < n; a++) for(int i = 0; i < nc; i++) GC[(size_t)a * nmo + i] = 4.0 * FC[(size_t)a * nc + i];
    }
    if(na){
        std::vector<double> FCa((size_t)n * na), FCg((size_t)n * na);
        gemm(n, na, n, Fd.data(), Ca.data(), FCa.data());                             // F C_a
        gemm(n, na, na, FCa.data(), C->gamma, FCg.data());                            // (F C_a) gamma      (gamma symmetric)
        for(int a = 0; a < n; a++) for(int t = 0; t < na; t++) GC[(size_t)a * nmo + nc + t] = 2.0 * FCg[(size_t)a * na + t] + GCa[(size_t)a * na + t];
    }
    std::vector<double> CtG((size_t)nmo * nmo), Wmo((size_t)nmo * nmo), CW((size_t)n * nmo), Wao(n2), mWao(n2);
    gemm(nmo, nmo, n, Call.data(), GC.data(), CtG.data(), 0.0, true, false);        // C^T G_C
    for(int p = 0; p < nmo; p++) for(int q = 0; q < nmo; q++) Wmo[(size_t)p * nmo + q] = 0.25 * (CtG[(size_t)p * nmo + q] + CtG[(size_t)q * nmo + p]);   // 1/2 * sym
    gemm(n, nmo, nmo, Call.data(), Wmo.data(), CW.data());
    gemm(n, n, nmo, CW.data(), Call.data(), Wao.data(), 0.0, false, true);
    for(size_t k = 0; k < n2; k++) mWao[k] = -Wao[k];
    deriv_overlap_contract(M, mWao.data(), grad.data());

    // a measure of how well the orbitals are converged: the antisymmetric part of C^T G_C (the rotation gradient)
    double rot = 0; for(int p = 0; p < nmo; p++) for(int q = 0; q < p; q++) rot = std::max(rot, std::fabs(CtG[(size_t)p * nmo + q] - CtG[(size_t)q * nmo + p]));
    fprintf(out_stream, "CAS-SCF orbital rotation gradient (max |antisymmetric part of C^T G_C|) = %.3e\n", rot);

    for(int i = 0; i < 3 * na_atoms; i++) g[i] = grad[i];
    return 0;
}
