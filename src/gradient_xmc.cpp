#include "PT_tensors.h"
#undef RES_FIT_H      // PT_tensors.h and res_fit.h share the include guard RES_FIT_H (a production quirk): include both
#include "res_fit.h"
#include "gradient_xmc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "blas_link.h"
#include "common_vars.h"
#include "doCI_data.h"
#include "integral_derivatives.h"
#include "matr.h"
#include "pt_canon.h"
#include "pt_chain.h"
#include "pt_vjp.h"
#include "res_fit_adj.h"
#include "res_fit_deriv.h"
#include "RI.h"
#include <omp.h>
#include "gradient_driver.h"

extern int n_fit_pol;

XmcGradRequest g_xmc_grad;

namespace{
double dot(const double * a, const double * b, size_t n){ double s = 0; for(size_t i = 0; i < n; i++) s += a[i] * b[i]; return s; }
int fail(const char * msg){ fprintf(out_stream, "ERROR: XMCQDPT2 gradient: %s\n", msg); g_xmc_grad.status = 1; return 1; }
}

int xmc_grad_snapshot(molecule * M, xmc_par * xmc){
    if(g_xmc_grad.root < 0) return 0;
    g_xmc_grad.status = 0; g_xmc_grad.done = false;
    if(xmc->d_only) return fail("D_ONLY imports the second-order Hamiltonian from an external file: the energy is not a function of the geometry "
                                "computed here, so it has no gradient");
    if(SO) return fail("spin-orbit runs are not supported");
    delete g_xmc_grad.ref;
    g_xmc_grad.ref = new CasRef;
    CasRef & R = *g_xmc_grad.ref;
    if(R.build(M)){ g_xmc_grad.status = 1; return 1; }
    const int n_s = xmc->cas->n_s;
    if(n_s > R.ns_all) return fail("the XMC model space has more states than the CAS-SCF computed");
    if(g_xmc_grad.root >= n_s){ fprintf(out_stream, "ERROR: XMCQDPT2 gradient: root %d is not among the %d model states\n", g_xmc_grad.root, n_s); g_xmc_grad.status = 1; return 1; }
    // the Fock operator of QDPT2 averages the 1-RDMs of the first w_state.size() roots with the raw w_state; the SA-CASSCF functional uses the actual
    // (tracked) weights: they must be the same functional
    const std::vector<double> & wx = xmc->cas->w_state;
    double sx = 0; for(double v : wx) sx += v;
    std::vector<double> wxn(R.ns_all, 0.0);
    for(size_t i = 0; i < wx.size() && (int)i < R.ns_all; i++) wxn[i] = wx[i] / sx;
    std::vector<double> wr(R.ns_all, 0.0);
    for(int I = 0; I < R.ns; I++) wr[R.states[I]] = R.w[I];
    for(int I = 0; I < R.ns_all; I++) if(std::fabs(wxn[I] - wr[I]) > 1e-12)
        return fail("the XMC Fock-operator weights (w_state as given) differ from the actual SA-CASSCF weights (state tracking?)");
    return 0;
}

int xmc_grad_main(molecule * M, xmc_par * xmc, doCI_data & D, res_fit_data & RF, RI_data & RI, double * eps, const double * U, const double * C,
                  const double * E0, const double * H01, const double * E2PP, double * H_AV, double * H_CA, double * H_CV){
    if(g_xmc_grad.root < 0 || g_xmc_grad.status) return g_xmc_grad.status;
    double tw = omp_get_wtime();
    auto lap = [&](const char * what){ const double t = omp_get_wtime(); fprintf(out_stream, "XMCQDPT2 gradient: %-46s %8.2f s   (memory %.0f MB, peak %.0f MB)\n", what, t - tw, grad_rss_mb(), grad_peak_rss_mb()); fflush(out_stream); tw = t; };
    if(!g_xmc_grad.ref) return fail("no reference snapshot");
    CasRef & R = *g_xmc_grad.ref;
    const int n = R.n, nc = R.nc, na = R.na, nv = R.nv, nmo = R.nmo, naux = R.naux, ns = RF.n_s, root = g_xmc_grad.root;
    aldet_data & ci = D.aldet;
    const int Na = ci.Na, Nb = ci.Nb;
    const size_t Nd = (size_t)Na * Nb, a2 = (size_t)na * na, a4 = a2 * a2, a6 = a4 * a2;
    const int Nf = RF.N_fit;
    if(Nd != R.Nd) return fail("CI dimensions differ from the reference");

    // ---- 0. the frame of QDPT2: active orbitals rotated by U, CI vectors carried along; checks against what QDPT2 holds ----------------------
    {
        double off = 0; for(int i = 0; i < na; i++) for(int j = 0; j < na; j++) if(i != j) off = std::max(off, std::fabs(U[(size_t)i * na + j]));
        fprintf(out_stream, "XMCQDPT2 gradient: active rotation of QDPT2 relative to the CAS-SCF orbitals: max off-diagonal |U| = %.3e, diagonal", off);
        for(int i = 0; i < na; i++) fprintf(out_stream, " % .6f", U[(size_t)i * na + i]);
        fprintf(out_stream, "\n");
    }
    R.rotate_active(U);
    {
        // active rows: the rotation by U must reproduce QDPT2's orbitals; core and virtual rows: QDPT2's block eigenvectors span the same spaces (the SA functional
        // is invariant under rotations within them), so they are taken over as they are
        double dact = 0, leak = 0;
        for(int p = nc; p < nc + na; p++) for(int a = 0; a < n; a++) dact = std::max(dact, std::fabs(R.C0[(size_t)p * n + a] - D.VEC[0][(size_t)p * n + a]));
        // squared norm of the projection of each new core/virtual row on the old block (AO metric S): 1 for a rotation within the block
        M->calc_S_AO();
        std::vector<double> SC((size_t)nmo * n, 0.0);           // SC[k][a] = (S C0_k)[a]
        for(int k = 0; k < nmo; k++) for(int a = 0; a < n; a++){ double t = 0; for(int b = 0; b < n; b++) t += M->S_AO[(size_t)a * n + b] * R.C0[(size_t)k * n + b]; SC[(size_t)k * n + a] = t; }
        for(int blk = 0; blk < 2; blk++){
            const int o = blk ? nc + na : 0, m = blk ? nv : nc;
            for(int i = 0; i < m; i++){
                double w = 0;
                for(int k = 0; k < m; k++){ const double t = dot(&D.VEC[0][(size_t)(o + i) * n], &SC[(size_t)(o + k) * n], n); w += t * t; }
                leak = std::max(leak, std::fabs(1.0 - w));
            }
        }
        fprintf(out_stream, "XMCQDPT2 gradient: active orbitals vs the rotated reference %.2e, core/virtual spaces leak %.2e\n", dact, leak);
        if(dact > 1e-8 || leak > 1e-8) return fail("the semicanonical orbitals of QDPT2 are not a rotation of the reference within its blocks");
        for(int p = 0; p < nc; p++) for(int a = 0; a < n; a++) R.C0[(size_t)p * n + a] = D.VEC[0][(size_t)p * n + a];
        for(int p = nc + na; p < nmo; p++) for(int a = 0; a < n; a++) R.C0[(size_t)p * n + a] = D.VEC[0][(size_t)p * n + a];
        R.base_.reset();
    }
    {   // the rotated reference must still be a stationary SA-CASSCF (rotating orbitals and CI vectors together is an invariance)
        std::vector<double> r0; double Esa = 0;
        cas_residual(R, R.C0.data(), R.c0.data(), &r0, nullptr, &Esa);
        double rk = 0, rc = 0;
        for(size_t k = 0; k < R.nk(); k++) rk = std::max(rk, std::fabs(r0[k]));
        for(size_t k = R.nk(); k < R.nx(); k++) rc = std::max(rc, std::fabs(r0[k]));
        std::vector<double> Hc(Nd); double eig = 0;
        for(int I = 0; I < R.ns_all; I++){ std::vector<double> c = R.root_vector(I); cas_H_apply(R, R.C0.data(), c.data(), 1, Hc.data()); const double E = dot(c.data(), Hc.data(), Nd);
            for(size_t d = 0; d < Nd; d++) eig = std::max(eig, std::fabs(Hc[d] - E * c[d])); }
        fprintf(out_stream, "XMCQDPT2 gradient: rotated reference: E_SA = % .12f, max |r_kappa| = %.2e, max |r_delta| = %.2e, max |(H - E) c| over the roots = %.2e\n", Esa, rk, rc, eig);
        if(rk > 1e-6 || rc > 1e-6 || eig > 1e-6) return fail("the reference rotated into the frame of QDPT2 is not a stationary SA-CASSCF (malmqvist?)");
    }
    // c1 = C^T c2 (C has orthonormal rows) against the rotated reference vectors
    std::vector<double> c1((size_t)ns * Nd), c2((size_t)ns * Nd);   // state-major
    for(size_t d = 0; d < Nd; d++) for(int s = 0; s < ns; s++) c2[(size_t)s * Nd + d] = ci.coef[0][d * ns + s];
    for(int i = 0; i < ns; i++) for(size_t d = 0; d < Nd; d++){ double v = 0; for(int j = 0; j < ns; j++) v += C[(size_t)j * ns + i] * c2[(size_t)j * Nd + d]; c1[(size_t)i * Nd + d] = v; }
    {
        double dmax = 0;
        for(int i = 0; i < ns; i++) for(size_t d = 0; d < Nd; d++) dmax = std::max(dmax, std::fabs(c1[(size_t)i * Nd + d] - R.c_all[(size_t)i * Nd + d]));
        if(dmax > 1e-8){ fprintf(out_stream, "ERROR: XMCQDPT2 gradient: the model-space vectors of QDPT2 differ from the rotated reference by %.2e\n", dmax); g_xmc_grad.status = 1; return 1; }
        double smax = 0;
        for(int i = 0; i < ns; i++) for(int j = 0; j < ns; j++) smax = std::max(smax, std::fabs(D.ACT_DET_A[i * ns + j] - (i == j ? 1.0 : 0.0)));
        if(smax > 1e-10){ fprintf(out_stream, "ERROR: XMCQDPT2 gradient: the model-space overlap S0PP differs from 1 by %.2e\n", smax); g_xmc_grad.status = 1; return 1; }
    }

    lap("0. the frame of QDPT2: active orbitals rotated ");
    // ---- 1. the final eigenproblem H2 = H01 + sym(C^T E2PP C) --------------------------------------------------------------------------------
    std::vector<double> E2i((size_t)ns * ns, 0.0), H2((size_t)ns * ns), ev(ns);
    for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++){ double v = 0; for(int j = 0; j < ns; j++) for(int k = 0; k < ns; k++) v += C[(size_t)j * ns + a] * E2PP[(size_t)j * ns + k] * C[(size_t)k * ns + b]; E2i[(size_t)a * ns + b] = v; }
    for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++) H2[(size_t)a * ns + b] = H01[(size_t)a * ns + b] + 0.5 * (E2i[(size_t)a * ns + b] + E2i[(size_t)b * ns + a]);
    std::vector<double> Vh(H2);
    lapack_diag(Vh.data(), ev.data(), ns);                  // rows of Vh are the eigenvectors, ascending
    const double * v = &Vh[(size_t)root * ns];
    g_xmc_grad.E_root = ev[root];
    {
        double gap = 1e300; for(int k = 0; k < ns; k++) if(k != root) gap = std::min(gap, std::fabs(ev[k] - ev[root]));
        fprintf(out_stream, "XMCQDPT2 gradient: root %d, E = % .12f, gap to the nearest other root %.3e\n", root, ev[root], gap);
        if(gap < 1e-6) fprintf(out_stream, "WARNING: XMCQDPT2 gradient: the root is (nearly) degenerate; its gradient is not defined at a crossing\n");
    }
    // Hbar2 = v v^T (symmetric): Ebar on E2i and on H01 is v v^T
    std::vector<double> Eb((size_t)ns * ns);
    for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++) Eb[(size_t)a * ns + b] = v[a] * v[b];

    lap("1. the final eigenproblem H2 = H01 + sym(C^T E2");
    // ---- 2. E2i = C^T E2PP C:  E2PPbar = C Eb C^T,  Cbar = E2PP C Eb + E2PP^T C Eb -------------------------------------------------------------
    std::vector<double> W((size_t)ns * ns, 0.0), Cbar((size_t)ns * ns, 0.0), EC((size_t)ns * ns, 0.0), ETC((size_t)ns * ns, 0.0);
    for(int j = 0; j < ns; j++) for(int k = 0; k < ns; k++){ double s1 = 0; for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++) s1 += C[(size_t)j * ns + a] * Eb[(size_t)a * ns + b] * C[(size_t)k * ns + b]; W[(size_t)j * ns + k] = s1; }
    for(int j = 0; j < ns; j++) for(int b = 0; b < ns; b++){ double s1 = 0, s2 = 0; for(int k = 0; k < ns; k++){ s1 += E2PP[(size_t)j * ns + k] * C[(size_t)k * ns + b]; s2 += E2PP[(size_t)k * ns + j] * C[(size_t)k * ns + b]; } EC[(size_t)j * ns + b] = s1; ETC[(size_t)j * ns + b] = s2; }
    for(int j = 0; j < ns; j++) for(int a = 0; a < ns; a++){ double s1 = 0; for(int b = 0; b < ns; b++) s1 += (EC[(size_t)j * ns + b] + ETC[(size_t)j * ns + b]) * Eb[(size_t)b * ns + a]; Cbar[(size_t)j * ns + a] = s1; }

    lap("2. E2i = C^T E2PP C:  E2PPbar = C Eb C^T,  Cbar");
    // ---- 3. the res_fit contraction ----------------------------------------------------------------------------------------------------------------
    ResFitAdjResult RR;
    res_fit_E2_adjoint(RF, W.data(), &RR, true, false);   // the three-body table seeds follow window by window in step 4
    std::vector<double> e0bar = RR.e0bar, Eactbar = RR.Eactbar;
    std::vector<double> cbar2((size_t)ns * Nd);                      // state-major
    for(size_t d = 0; d < Nd; d++) for(int s = 0; s < ns; s++) cbar2[(size_t)s * Nd + d] = RR.cibar[d * ns + s];

    lap("3. the res_fit contraction");
    // ---- 4. the tables: the raw EE blocks at every grid point, denominators shifted by E_fit[f] -------------------------------------------------
    PT_tensors T;
    T.set_par(&RI, eps, nc, na, nv, H_AV, H_CA, H_CV, xmc->edshift, M);
    std::vector<double> adjVC((size_t)nv * nc * naux, 0.0), adjVA((size_t)nv * na * naux, 0.0), adjCA((size_t)nc * na * naux, 0.0), adjCC((size_t)nc * nc * naux, 0.0),
                        adjAA((size_t)na * na * naux, 0.0), adj_ec(nc, 0.0), adj_ea(na, 0.0), adj_ev(nv, 0.0), adjHAV((size_t)na * nv, 0.0), adjHCA((size_t)nc * na, 0.0),
                        adjHCV((size_t)nc * nv, 0.0);
    PTInputAdj A;
    A.VC = adjVC.data(); A.VA = adjVA.data(); A.CA = adjCA.data(); A.CC = adjCC.data(); A.AA = adjAA.data();
    A.e_c = adj_ec.data(); A.e_a = adj_ea.data(); A.e_v = adj_ev.data();
    A.H_AV = adjHAV.data(); A.H_CA = adjHCA.data(); A.H_CV = adjHCV.data();
    std::vector<double> gbar_grid(Nf, 0.0);
    std::vector<double> PH(a2), PVJ(a4), PVA(a4), P3J(a6), P3A(a6);
    // the seeds of the two three-body tables (n_a^6 x N_fit each) are built for a window of grid points at a time: NOPT_XMC_ADJ_MB bounds the
    // window (for the pair; default the budget of the windows of the energy, NOPT_XMC_WINDOW_MB, 4096 MB); every window repeats the three-body
    // contraction loops
    double adj_mb = 4096.0;
    if(getenv("NOPT_XMC_WINDOW_MB") && *getenv("NOPT_XMC_WINDOW_MB")) adj_mb = std::max(1.0, atof(getenv("NOPT_XMC_WINDOW_MB")));
    if(getenv("NOPT_XMC_ADJ_MB") && *getenv("NOPT_XMC_ADJ_MB")) adj_mb = std::max(1.0, atof(getenv("NOPT_XMC_ADJ_MB")));
    const int win = (int)std::max(1.0, std::min((double)Nf, adj_mb * 1048576.0 / (2.0 * 8.0 * (double)a6)));
    std::vector<double> w3J, w3A;
    int wf0 = -1, wnb = 0, n_windows = 0;
    double t_win = 0.0;
    int n_visited = 0;
    const char * blk_names[8] = {"CCVV", "CAVV", "AAVV", "CCAV", "CCAA", "CV", "AV", "CA"};
    double blk_time[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> gf; std::vector<double> gPS, gPH, gPVJ, gPVA;   // seeds of the visited grid points, for the fused classes
    // NOPT_XMC_BLOCKS (test switch, see res_fit_data::E2_calc): only the classes the forward computed
    const char * blk = getenv("NOPT_XMC_BLOCKS");
    auto use_blk = [blk](const char * b){
        if(!(blk && *blk)) return true;
        const size_t lb = strlen(b);
        for(const char * p = blk; *p; ){ const char * q = strchr(p, ','); const size_t l = q ? (size_t)(q - p) : strlen(p);
            if(l == lb && !strncmp(p, b, lb)) return true; if(!q) break; p = q + 1; }
        return false;
    };
    for(int f = 0; f < Nf; f++){
        double mx = std::fabs(RR.tables.PS[f]);
        for(size_t i = 0; i < a2; i++){ PH[i] = RR.tables.PH[i * Nf + f]; mx = std::max(mx, std::fabs(PH[i])); }
        for(size_t i = 0; i < a4; i++){ PVJ[i] = RR.tables.PV_JK[i * Nf + f]; PVA[i] = RR.tables.PV_AB[i * Nf + f]; mx = std::max(mx, std::max(std::fabs(PVJ[i]), std::fabs(PVA[i]))); }
        if(wf0 < 0 || f >= wf0 + wnb){
            const double t0 = omp_get_wtime();
            wf0 = f; wnb = std::min(win, Nf - f);
            w3J.assign(a6 * wnb, 0.0); w3A.assign(a6 * wnb, 0.0);
            res_fit_E2_adjoint_3body(RF, W.data(), wf0, wnb, w3J.data(), w3A.data());
            n_windows++; t_win += omp_get_wtime() - t0;
        }
        const int fw = f - wf0;
        for(size_t i = 0; i < a6; i++){ P3J[i] = w3J[i * wnb + fw]; mx = std::max(mx, std::fabs(P3J[i])); }
        // RF_P3_AB: res_fit keeps [t,u,v,w,x,y], the raw EE kernels [t,u,x,v,w,y]
        for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int vv = 0; vv < na; vv++) for(int w = 0; w < na; w++) for(int x = 0; x < na; x++) for(int y = 0; y < na; y++){
            const double val = w3A[(((((size_t)(t * na + u) * na + vv) * na + w) * na + x) * na + y) * wnb + fw];
            P3A[((((size_t)(t * na + u) * na + x) * na + vv) * na + w) * na + y] = val; mx = std::max(mx, std::fabs(val));
        }
        if(getenv("NOPT_RESP_DEBUG") && *getenv("NOPT_RESP_DEBUG")){
            double tmax = std::fabs(RF.RF_PS[f]);
            for(size_t i = 0; i < a2; i++) tmax = std::max(tmax, std::fabs(RF.RF_PH[i * Nf + f]));
            for(size_t i = 0; i < a4; i++) tmax = std::max(tmax, std::max(std::fabs(RF.RF_PV_JK[i * Nf + f]), std::fabs(RF.RF_PV_AB[i * Nf + f])));
            if(RF.RF_P3_JK) for(size_t i = 0; i < a6; i++) tmax = std::max(tmax, std::max(std::fabs(RF.RF_P3_JK[i * Nf + f]), std::fabs(RF.RF_P3_AB[i * Nf + f])));
            fprintf(out_stream, "XMC DEBUG grid %2d E_fit % .6f  max|table| %.3e  max|seed| %.3e\n", f, RF.E_fit[f], tmax, mx);
        }
        if(mx == 0.0) continue;
        n_visited++;
        PTRawSeeds S; S.PS = RR.tables.PS[f]; S.PH = PH.data(); S.PV_JK = PVJ.data(); S.PV_AB = PVA.data(); S.P3_JK = P3J.data(); S.P3_AB = P3A.data();
        pt_vjp_shift = RF.E_fit[f];
        A.shift_adj = 0.0;
        int bad = 0;
        auto use = use_blk;
        {
            // the four classes with a fat index pair are fused over the grid below (their integrals built once); the rest run per grid point
            int (*fn[8])(PT_tensors &, const PTRawSeeds &, PTInputAdj &) = {nullptr, nullptr, nullptr, nullptr,
                                                                            pt_vjp_EE_CCAA, pt_vjp_EE_CV, pt_vjp_EE_AV, pt_vjp_EE_CA};
            for(int k = 4; k < 8; k++) if(use(blk_names[k])){ const double t0 = omp_get_wtime(); bad += fn[k](T, S, A); blk_time[k] += omp_get_wtime() - t0; }
            gf.push_back(f);
            gPS.push_back(S.PS);
            gPH.insert(gPH.end(), PH.begin(), PH.end()); gPVJ.insert(gPVJ.end(), PVJ.begin(), PVJ.end()); gPVA.insert(gPVA.end(), PVA.begin(), PVA.end());
        }
        gbar_grid[f] = A.shift_adj;
        if(bad){ pt_vjp_shift = 0.0; return fail("a PT block adjoint failed"); }
    }
    pt_vjp_shift = 0.0;
    {
        const int K = (int)gf.size();
        std::vector<PTRawSeeds> Sg(K);
        std::vector<double> shg(K), shadj(K, 0.0);
        for(int k = 0; k < K; k++){
            Sg[k].PS = gPS[k]; Sg[k].PH = &gPH[(size_t)k * a2]; Sg[k].PV_JK = &gPVJ[(size_t)k * a4]; Sg[k].PV_AB = &gPVA[(size_t)k * a4];
            shg[k] = RF.E_fit[gf[k]];
        }
        int (*fg[4])(PT_tensors &, const PTRawSeeds *, const double *, int, PTInputAdj &, double *) = {pt_vjp_EE_CCVV_grid, pt_vjp_EE_CAVV_grid, pt_vjp_EE_AAVV_grid, pt_vjp_EE_CCAV_grid};
        int bad = 0;
        for(int c = 0; c < 4; c++) if(K && use_blk(blk_names[c])){ const double t0 = omp_get_wtime(); bad += fg[c](T, Sg.data(), shg.data(), K, A, shadj.data()); blk_time[c] += omp_get_wtime() - t0; }
        if(bad) return fail("a PT block adjoint failed");
        for(int k = 0; k < K; k++) gbar_grid[gf[k]] += shadj[k];
    }
    fprintf(out_stream, "XMCQDPT2 gradient: three-body table seeds in %d window(s) of up to %d grid points (%.0f MB), %.2f s\n", n_windows, win,
            2.0 * 8.0 * (double)a6 * win / 1048576.0, t_win);
    w3J.clear(); w3J.shrink_to_fit(); w3A.clear(); w3A.shrink_to_fit();
    fprintf(out_stream, "XMCQDPT2 gradient: block adjoints at %d of %d grid points; time per class:", n_visited, Nf);
    for(int k = 0; k < 8; k++) fprintf(out_stream, " %s %.2f s", blk_names[k], blk_time[k]);
    fprintf(out_stream, "\n");
    if(!(getenv("NOPT_XMC_NEGCTRL") && !strcmp(getenv("NOPT_XMC_NEGCTRL"), "grid"))){
        RFGridBranch br = rf_grid_branch(RF.E0, ns, ci.E_act[0], (long)Nd);
        rf_grid_vjp(br, Nf, gbar_grid.data(), e0bar.data(), Eactbar.data());
    }
    else fprintf(out_stream, "XMCQDPT2 gradient: NEGATIVE CONTROL grid\n");

    lap("4. the tables: the raw EE blocks at every grid ");
    // ---- 5. c2 = C c1 --------------------------------------------------------------------------------------------------------------------------
    std::vector<double> cbar1((size_t)ns * Nd, 0.0);
    for(int i = 0; i < ns; i++) for(size_t d = 0; d < Nd; d++){ double s1 = 0; for(int j = 0; j < ns; j++) s1 += C[(size_t)j * ns + i] * cbar2[(size_t)j * Nd + d]; cbar1[(size_t)i * Nd + d] = s1; }
    for(int j = 0; j < ns; j++) for(int i = 0; i < ns; i++) Cbar[(size_t)j * ns + i] += dot(&cbar2[(size_t)j * Nd], &c1[(size_t)i * Nd], Nd);

    lap("5. c2 = C c1");
    // ---- 6. (C, E0) = eigenpairs of H0PP --------------------------------------------------------------------------------------------------------
    std::vector<double> H0b((size_t)ns * ns, 0.0);
    double gap0 = 1e300;
    for(int k = 0; k < ns; k++){
        for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++) H0b[(size_t)a * ns + b] += e0bar[k] * C[(size_t)k * ns + a] * C[(size_t)k * ns + b];
        for(int l = 0; l < ns; l++){
            if(l == k) continue;
            const double dE = E0[k] - E0[l];
            gap0 = std::min(gap0, std::fabs(dE));
            const double f = dot(&Cbar[(size_t)k * ns], &C[(size_t)l * ns], ns) / dE;    // dC_k = sum_l C_l (C_l dH C_k^T) / (E_k - E_l)
            for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++) H0b[(size_t)a * ns + b] += f * C[(size_t)l * ns + a] * C[(size_t)k * ns + b];
        }
    }
    for(int a = 0; a < ns; a++) for(int b = 0; b < a; b++){ const double m = 0.5 * (H0b[(size_t)a * ns + b] + H0b[(size_t)b * ns + a]); H0b[(size_t)a * ns + b] = H0b[(size_t)b * ns + a] = m; }
    fprintf(out_stream, "XMCQDPT2 gradient: smallest gap between H0PP eigenvalues %.3e\n", gap0);
    if(gap0 < 1e-6) fprintf(out_stream, "WARNING: XMCQDPT2 gradient: H0PP is (nearly) degenerate; the derivative of its eigenvectors is amplified as 1/gap\n");

    lap("6. (C, E0) = eigenpairs of H0PP");
    // ---- 7. H0PP = sum_I c1_Ia c1_Ib E_act(I) ----------------------------------------------------------------------------------------------------
    const double * Eact = ci.E_act[0];
    for(size_t d = 0; d < Nd; d++){
        double s1 = 0;
        for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++) s1 += H0b[(size_t)a * ns + b] * c1[(size_t)a * Nd + d] * c1[(size_t)b * Nd + d];
        Eactbar[d] += s1;
        for(int a = 0; a < ns; a++){ double t = 0; for(int b = 0; b < ns; b++) t += 2.0 * H0b[(size_t)a * ns + b] * c1[(size_t)b * Nd + d]; cbar1[(size_t)a * Nd + d] += t * Eact[d]; }
    }
    lap("7. H0PP = sum_I c1_Ia c1_Ib E_act(I)");
    // ---- 8. E_act(a,b) = sum of eps_a over the occupied orbitals of the alpha string a and the beta string b -------------------------------------
    std::vector<double> ea_bar(adj_ea);
    for(int ia = 0; ia < Na; ia++) for(int ib = 0; ib < Nb; ib++){
        const double eb = Eactbar[(size_t)ia * Nb + ib];
        if(eb == 0.0) continue;
        for(int k = 1; k < ci.na + 1; k++) ea_bar[ci.vec_a[ia * (ci.na + 1) + k] - 1] += eb;
        for(int k = 1; k < ci.nb + 1; k++) ea_bar[ci.vec_b[ib * (ci.nb + 1) + k] - 1] += eb;
    }

    if(xmc->have_ifitd){
        // ifitd: the active orbital energies are constants read from a file -- the energy does not depend on them; it still depends on the canonical
        // active gauge of QDPT2 (the rotation term below, with the true eigenvalues of the active block in the denominators)
        std::fill(ea_bar.begin(), ea_bar.end(), 0.0);
    }
    lap("8. E_act(a,b) = sum of eps_a over the occupied ");
    // ---- 9. H01: v^T H01 v = <psi|H|psi> ----------------------------------------------------------------------------------------------------------
    std::vector<double> psi(Nd, 0.0);
    for(int a = 0; a < ns; a++) for(size_t d = 0; d < Nd; d++) psi[d] += v[a] * c1[(size_t)a * Nd + d];
    CasObjects direct;
    const double one = 1.0;
    cas_objects(R, R.C0.data(), psi.data(), 1, &one, true, &direct);
    const std::vector<double> GC_bare = direct.GC;
    std::vector<double> cbar1_bare((size_t)ns * Nd, 0.0);
    {
        double vh = 0; for(int a = 0; a < ns; a++) for(int b = 0; b < ns; b++) vh += v[a] * H01[(size_t)a * ns + b] * v[b];
        fprintf(out_stream, "XMCQDPT2 gradient: v^T H01 v = % .12f, <psi|H|psi> rebuilt = % .12f, difference %.2e\n", vh, direct.E, direct.E - vh);
        if(std::fabs(direct.E - vh) > 1e-8) return fail("H01 is not the CAS Hamiltonian of the reference (rebuilt <psi|H|psi> differs)");
    }
    {
        std::vector<double> Hpsi(Nd);
        cas_H_apply(R, R.C0.data(), psi.data(), 1, Hpsi.data());
        for(int a = 0; a < ns; a++) for(size_t d = 0; d < Nd; d++){ cbar1[(size_t)a * Nd + d] += 2.0 * v[a] * Hpsi[d]; cbar1_bare[(size_t)a * Nd + d] = 2.0 * v[a] * Hpsi[d]; }
    }

    lap("9. H01: v^T H01 v = <psi|H|psi>");
    // ---- 10. B and H blocks -> C_sc, L, V, h ---------------------------------------------------------------------------------------------------------
    PTChainOut out; out.resize(nmo, n, naux);
    pt_chain_B(n, naux, nc, na, nv, R.C0.data(), R.L(), R.V.data(), A, &out, &R.ri());
    pt_chain_H(n, naux, nc, na, nv, R.C0.data(), R.L(), R.V.data(), R.h.data(), A, &out, &R.ri(), true);

    lap("10. B and H blocks");
    // ---- 11. the active canonicalization carries the CI vectors: g_tu = d/dtheta of the energy for C_t += theta C_u, CI part ---------------------
    // Only the parts of the energy that depend on the canonical rotation U belong here (the PT parts: E2 and H0PP). H01 = <c1|H|c1> is invariant under U
    // (orbitals and CI vectors rotate together), so neither half of its derivative may enter the canonicalization term: its orbital half is not in Cbar_sc
    // (it sits in `direct`), and its CI half 2 v Hpsi is excluded here. (Including the CI half alone broke the invariance whenever the bare energy has a
    // nonzero active-rotation gradient -- water 3 states, weights 1 1 <=0.05.)
    std::vector<double> cbar1_pt(cbar1);
    for(size_t i = 0; i < cbar1_pt.size(); i++) cbar1_pt[i] -= cbar1_bare[i];
    std::vector<double> Ttr(a2), grot(a2);
    cas_transition_rdm(R, cbar1_pt.data(), c1.data(), ns, Ttr.data());
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) grot[(size_t)t * na + u] = Ttr[(size_t)t * na + u] - Ttr[(size_t)u * na + t];
    // negative controls (tests only): NOPT_XMC_NEGCTRL=rot flips the CI part of the active rotation, =grid drops the table-shift (grid) adjoint
    const char * neg = getenv("NOPT_XMC_NEGCTRL");
    if(neg && !strcmp(neg, "rot")){ for(double & x : grot) x = -x; fprintf(out_stream, "XMCQDPT2 gradient: NEGATIVE CONTROL rot\n"); }

    if(getenv("NOPT_RESP_DEBUG") && *getenv("NOPT_RESP_DEBUG")){
        fprintf(out_stream, "XMC DEBUG ea_bar(PT+Eact):"); for(int t = 0; t < na; t++) fprintf(out_stream, " % .6e", ea_bar[t]); fprintf(out_stream, "\n");
        fprintf(out_stream, "XMC DEBUG ea_bar(PT only):"); for(int t = 0; t < na; t++) fprintf(out_stream, " % .6e", adj_ea[t]); fprintf(out_stream, "\n");
        fprintf(out_stream, "XMC DEBUG |grot_tu|:"); for(int t = 0; t < na; t++) for(int u = t + 1; u < na; u++) fprintf(out_stream, " %d%d:% .6e", t, u, grot[(size_t)t * na + u] * (U[(size_t)t * na + t] * U[(size_t)u * na + u])); fprintf(out_stream, "\n");
        fprintf(out_stream, "XMC DEBUG Cbar_sc.C_act (s-corrected):");
        for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) if(t != u){ double m = dot(&out.Cbar[(size_t)(nc + t) * n], &R.C0[(size_t)(nc + u) * n], n); fprintf(out_stream, " %d%d:% .4e", t, u, m * U[(size_t)t * na + t] * U[(size_t)u * na + u]); }
        fprintf(out_stream, "\n");
        double e0 = 0, ea = 0; for(double x : e0bar) e0 += x; for(double x : Eactbar) ea += x;
        fprintf(out_stream, "XMC DEBUG sum e0bar % .6e sum Eactbar % .6e\n", e0, ea);
    }
    lap("11. the active canonicalization carries the CI v");
    // ---- 12. orbital energies and the canonical orbitals -> F_SA -> C, h, L, V, gamma_SA -------------------------------------------------------------
    std::vector<double> gamma_sa; cas_sa_gamma(R, &gamma_sa);
    std::vector<double> gbar(a2, 0.0), Fsa((size_t)n * n);
    std::vector<double> eps_canon(eps, eps + nmo);
    if(xmc->have_ifitd){ std::vector<double> d; cas_sa_fock_diag(R, R.C0.data(), gamma_sa.data(), &d); for(int t = 0; t < na; t++) eps_canon[nc + t] = d[nc + t]; }
    pt_canon_vjp(n, naux, nc, na, nv, R.C0.data(), eps_canon.data(), R.L(), R.V.data(), R.h.data(), gamma_sa.data(), out.Cbar.data(), adj_ec.data(), adj_ev.data(),
                 1e-6, &out, gbar.data(), Fsa.data(), ea_bar.data(), grot.data(), &R.ri(), nullptr, true);
    {   // the orbital energies of QDPT2 must be the semicanonical eigenvalues of this F_SA
        double emax = 0;
        for(int p = 0; p < nmo; p++){
            double d = 0; for(int a = 0; a < n; a++){ double t = 0; for(int b = 0; b < n; b++) t += Fsa[(size_t)a * n + b] * R.C0[(size_t)p * n + b]; d += R.C0[(size_t)p * n + a] * t; }
            emax = std::max(emax, std::fabs(d - eps_canon[p]));
        }
        fprintf(out_stream, "XMCQDPT2 gradient: orbital energies vs the SA Fock operator rebuilt from the reference: max difference %.2e\n", emax);
        if(emax > 1e-8) return fail("the XMC orbital energies are not the eigenvalues of the SA Fock operator of the reference");
    }

    lap("12. orbital energies and the canonical orbitals");
    // ---- 13. totals -----------------------------------------------------------------------------------------------------------------------------------------
    for(size_t k = 0; k < direct.D.size(); k++) direct.D[k] += out.hbar[k];
    for(size_t k = 0; k < direct.Lbar.size(); k++) direct.Lbar[k] += out.Lbar[k];
    for(size_t k = 0; k < direct.Vbar.size(); k++) direct.Vbar[k] += out.Vbar[k];
    for(int p = 0; p < nmo; p++) for(int a = 0; a < n; a++) direct.GC[(size_t)a * nmo + p] += out.Cbar[(size_t)p * n + a];
    {   // diagnostic: the antisymmetric part of C^T G_C in the blocks whose rotations are not response variables must vanish (core-core, virtual-virtual;
        // active-active together with the CI part of the active rotation, which the canonicalization multiplier and the reference response carry)
        std::vector<double> CtG((size_t)nmo * nmo, 0.0);
        for(int p = 0; p < nmo; p++) for(int q = 0; q < nmo; q++){ double t = 0; for(int a = 0; a < n; a++) t += R.C0[(size_t)p * n + a] * direct.GC[(size_t)a * nmo + q]; CtG[(size_t)p * nmo + q] = t; }
        double acc = 0, avv = 0, aaa = 0;
        for(int p = 0; p < nmo; p++) for(int q = p + 1; q < nmo; q++){
            const double a = CtG[(size_t)q * nmo + p] - CtG[(size_t)p * nmo + q];
            if(q < nc) acc = std::max(acc, std::fabs(a));
            else if(p >= nc + na) avv = std::max(avv, std::fabs(a));
            else if(p >= nc && q < nc + na) aaa = std::max(aaa, std::fabs(a));
        }
        fprintf(out_stream, "XMCQDPT2 gradient: antisymmetric C^T G_C in the redundant blocks: core-core %.2e, virtual-virtual %.2e, active-active (orbital part) %.2e\n", acc, avv, aaa);
        // active rotations of the reference (orbitals and every CI vector together) leave the energy unchanged: orbital part + CI part = 0
        std::vector<double> T1(a2), T2(a2), Gc((size_t)R.ns * Nd), gsrc((size_t)R.ns * Nd);
        cas_transition_rdm(R, cbar1.data(), c1.data(), ns, T1.data());     // the full CI derivative (bare part included: the orbital part includes it too)
        cas_onebody_apply(R, gbar.data(), R.c0.data(), R.ns, Gc.data());
        for(int P = 0; P < R.ns; P++) for(size_t d = 0; d < Nd; d++) gsrc[(size_t)P * Nd + d] = 2.0 * R.w[P] * Gc[(size_t)P * Nd + d];
        cas_transition_rdm(R, gsrc.data(), R.c0.data(), R.ns, T2.data());
        double bal = 0, orb = 0;
        for(int t = 0; t < na; t++) for(int u = t + 1; u < na; u++){
            const int p = nc + t, q = nc + u;
            const double o = CtG[(size_t)p * nmo + q] - CtG[(size_t)q * nmo + p];          // orbital part
            const double c = (T1[t * na + u] - T1[u * na + t]) + (T2[t * na + u] - T2[u * na + t]);   // CI part (same orientation: an invariance makes them equal)
            bal = std::max(bal, std::fabs(o - c)); orb = std::max(orb, std::fabs(o));
            if(getenv("NOPT_RESP_DEBUG") && *getenv("NOPT_RESP_DEBUG")) fprintf(out_stream, "   active rotation %d-%d: orbital % .6e  CI(target) % .6e  CI(gamma) % .6e  sum % .3e\n", t, u, o,
                    T1[t * na + u] - T1[u * na + t], T2[t * na + u] - T2[u * na + t], o + c);
        }
        fprintf(out_stream, "XMCQDPT2 gradient: invariance under active rotations of the reference: max |orbital - CI| = %.2e (orbital part up to %.2e)\n", bal, orb);
        if(bal > 1e-6 * std::max(1.0, orb)){ fprintf(out_stream, "ERROR: XMCQDPT2 gradient: the derivative is not invariant under active rotations of the reference\n"); g_xmc_grad.status = 1; return 1; }
        {   // the bare part alone (<psi|H|psi>) must balance too
            std::vector<double> Tb(a2);
            cas_transition_rdm(R, cbar1_bare.data(), c1.data(), ns, Tb.data());
            for(int t = 0; t < na; t++) for(int u = t + 1; u < na; u++){
                const int p = nc + t, q = nc + u;
                double ob = 0, ob2 = 0;
                for(int a = 0; a < n; a++){ ob += R.C0[(size_t)p * n + a] * GC_bare[(size_t)a * nmo + q]; ob2 += R.C0[(size_t)q * n + a] * GC_bare[(size_t)a * nmo + p]; }
                if(getenv("NOPT_RESP_DEBUG") && *getenv("NOPT_RESP_DEBUG"))
                    fprintf(out_stream, "   bare: active rotation %d-%d: orbital % .6e  CI % .6e\n", t, u, ob - ob2, Tb[t * na + u] - Tb[u * na + t]);
            }
        }
    }
    out = PTChainOut();
    for(auto * x : {&adjVC, &adjVA, &adjCA, &adjCC, &adjAA}) std::vector<double>().swap(*x);

    std::vector<double> b_extra;
    cas_gamma_rhs(R, gbar.data(), &b_extra);
    std::vector<double> cbar_all((size_t)R.ns_all * Nd, 0.0);
    std::copy(cbar1.begin(), cbar1.end(), cbar_all.begin());          // model state a = CAS root a
    lap("13. totals");
    if(cas_lagrangian_objects_cbar(M, R, direct, cbar_all, b_extra, &g_xmc_grad.total)){ g_xmc_grad.status = 1; return 1; }
    lap("SA-CASSCF response (Z-vector, response objects)");
    g_xmc_grad.done = true;
    (void)a6;
    return 0;
}

void xmc_grad_check_final(const double * E, int n_s){
    if(g_xmc_grad.root < 0 || !g_xmc_grad.done) return;
    const double d = E[g_xmc_grad.root] - g_xmc_grad.E_root;
    fprintf(out_stream, "XMCQDPT2 gradient: final energy of root %d % .12f, differentiated % .12f, difference %.2e\n", g_xmc_grad.root, E[g_xmc_grad.root], g_xmc_grad.E_root, d);
    if(std::fabs(d) > 1e-9){ fprintf(out_stream, "ERROR: XMCQDPT2 gradient: the differentiated energy is not the one QDPT2 reports\n"); g_xmc_grad.done = false; g_xmc_grad.status = 1; }
    (void)n_s;
}

int xmc_grad_contract(molecule * M){
    g_xmc_grad.g.assign(3 * M->n_atoms, 0.0);
    deriv_nuclear_repulsion(M, g_xmc_grad.g.data());
    return cas_geo_contract(M, *g_xmc_grad.ref, g_xmc_grad.total, g_xmc_grad.g.data());
}
