#include "casscf_resp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include <functional>

#include "blas_link.h"
#include "CAS.h"
#include "RI.h"
#include "common_vars.h"
#include "integral_derivatives.h"
#include "libint_link.h"
#include "matr.h"
#include "ri_gradient.h"

extern long aux_n_ao;
extern double * __restrict__ RI_AA;

namespace{
void gemm(int m, int n, int k, const double * A, const double * B, double * C, double alpha = 1.0, double beta = 0.0, bool tA = false, bool tB = false){
    cblas_dgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                A, tA ? m : k, B, tB ? k : n, beta, C, n);
}
double dot(const double * a, const double * b, size_t n){ double s = 0; for(size_t i = 0; i < n; i++) s += a[i] * b[i]; return s; }
}

// ---- reference ---------------------------------------------------------------------------------------
int CasRef::build(molecule * Mol){
    M = Mol;
    CAS_engine * CAS = M->CAS;
    if(CAS == nullptr){ fprintf(out_stream, "ERROR: CasRef needs a molecule with a CAS_SCF calculation behind it\n"); return 1; }
    ci_template = CAS->CI->as_aldet();
    if(ci_template == nullptr){ fprintf(out_stream, "ERROR: the CAS-SCF response needs the determinant CI solver\n"); return 1; }
    n = M->n_ao; nc = CAS->n_core; na = CAS->n_act; nmo = M->n_mo; nv = nmo - nc - na;
    Enuc = M->V_nuc;
    ns_all = ci_template->n_states[0];
    Nd = (size_t)ci_template->Na * ci_template->Nb;
    C0.assign(M->MO_VEC, M->MO_VEC + (size_t)nmo * n);
    h.assign(M->H_AO, M->H_AO + (size_t)n * n);
    long nx = 0; for(auto & s : M->ri_s) nx += s.size();
    naux = (int)nx;
    V.assign((size_t)naux * naux, 0.0);
    // the three-index integrals: those of the energy calculation (RI_AA, built by the same calc_2el_AO_INTS_RI for this geometry) unless asked otherwise
    if(RI_AA != nullptr && aux_n_ao == naux && !(getenv("NOPT_GRAD_OWN_L") && *getenv("NOPT_GRAD_OWN_L"))){ L_own.clear(); Lp = RI_AA; }
    else { L_own.assign((size_t)n * n * naux, 0.0); calc_2el_AO_INTS_RI(M->s, n, M->ri_s, naux, L_own.data()); Lp = L_own.data(); }
    RI_V_calc(M->ri_s, naux, V.data());
    fprintf(out_stream, "Analytic gradient: n_ao %d, n_mo %d, n_aux %d (core %d, active %d, virtual %d); one AO-pair x auxiliary array (n_ao^2 n_aux) = %.0f MB\n",
            n, nmo, naux, nc, na, nv, (double)n * n * naux * 8.0 / 1048576.0);
    // weights: the roots with a nonzero weight define the SA functional (zero-weight roots, e.g. n_s=7 with three weights given, are spectators of the
    // orbital optimization; a gradient of such a root needs no CI response of its own: it is an eigenvector of the CAS Hamiltonian -- Hellmann-Feynman)
    std::vector<double> wa(ns_all, 1.0);
    if((int)CAS->wstate_actual.size() >= ns_all) for(int I = 0; I < ns_all; I++) wa[I] = CAS->wstate_actual[I];
    states.clear();
    for(int I = 0; I < ns_all; I++) if(wa[I] > 1e-10) states.push_back(I);
    ns = (int)states.size();
    if(ns == 0){ fprintf(out_stream, "ERROR: no CAS-SCF state has a nonzero weight\n"); return 1; }
    double wsum = 0; for(int I : states) wsum += wa[I];
    w.assign(ns, 0.0);
    for(int I = 0; I < ns; I++) w[I] = wa[states[I]] / wsum;
    rho.clear();
    for(int I = 0; I < ns; I++) for(int J = I + 1; J < ns; J++) if(std::fabs(w[I] - w[J]) > 1e-12) rho.push_back({I, J});
    if(!rho.empty()) fprintf(out_stream, "CAS-SCF response: unequal state weights, %d rotation(s) between weighted states are non-redundant variables\n", (int)rho.size());
    c_all.assign((size_t)ns_all * Nd, 0.0);
    for(int I = 0; I < ns_all; I++) for(size_t d = 0; d < Nd; d++) c_all[(size_t)I * Nd + d] = ci_template->coef[0][d * ns_all + I];
    c0.assign((size_t)ns * Nd, 0.0);
    for(int I = 0; I < ns; I++) for(size_t d = 0; d < Nd; d++) c0[(size_t)I * Nd + d] = c_all[(size_t)states[I] * Nd + d];
    pairs.clear();
    for(int p = 0; p < nc; p++) for(int q = nc; q < nc + na; q++) pairs.push_back({p, q});          // core - active
    for(int p = 0; p < nc; p++) for(int q = nc + na; q < nmo; q++) pairs.push_back({p, q});        // core - virtual
    for(int p = nc; p < nc + na; p++) for(int q = nc + na; q < nmo; q++) pairs.push_back({p, q});  // active - virtual
    return 0;
}

// ---- the CI kernel: H c and the densities of a set of vectors, for given active-space integrals ------------------
namespace{
struct ActInts{ std::vector<double> act, HAA; double Ecore = 0.0; };

// column layouts of the MO blocks: Cc (n x nc), Ca (n x na), Call (n x nmo) from C (nmo x n rows)
void columns(const CasRef & R, const double * C, std::vector<double> * Cc, std::vector<double> * Ca, std::vector<double> * Call){
    const int n = R.n, nc = R.nc, na = R.na, nmo = R.nmo;
    if(Cc){ Cc->assign((size_t)n * std::max(nc, 1), 0.0); for(int a = 0; a < n; a++) for(int i = 0; i < nc; i++) (*Cc)[(size_t)a * nc + i] = C[(size_t)i * n + a]; }
    if(Ca){ Ca->assign((size_t)n * std::max(na, 1), 0.0); for(int a = 0; a < n; a++) for(int t = 0; t < na; t++) (*Ca)[(size_t)a * na + t] = C[(size_t)(nc + t) * n + a]; }
    if(Call){ Call->assign((size_t)n * nmo, 0.0); for(int a = 0; a < n; a++) for(int p = 0; p < nmo; p++) (*Call)[(size_t)a * nmo + p] = C[(size_t)p * n + a]; }
}

// (tu|vw) = sum_i La[tu,i] La[vw,i] / lam_i  (and its derivative with dLa: sum_i (dLa La + La dLa) / lam_i)
void act_from_La(const CasRef & R, const double * La, const double * dLa, std::vector<double> * act){
    const int naux = R.naux; const size_t a2 = (size_t)R.na * R.na;
    const RIEig & E = R.ri();
    std::vector<double> Ls(a2 * naux);
    for(size_t tu = 0; tu < a2; tu++) for(int i = 0; i < naux; i++) Ls[tu * naux + i] = La[tu * naux + i] / E.lam[i];
    act->assign(a2 * a2, 0.0);
    if(!dLa){ gemm((int)a2, (int)a2, naux, Ls.data(), La, act->data(), 1.0, 0.0, false, true); return; }
    gemm((int)a2, (int)a2, naux, dLa, Ls.data(), act->data(), 1.0, 0.0, false, true);
    gemm((int)a2, (int)a2, naux, Ls.data(), dLa, act->data(), 1.0, 1.0, false, true);
}

// X Y^T factors of a density given as a sum of products of n x m column blocks: D = sum_k c_k A_k B_k^T
struct LowRank{
    int n = 0, r = 0; std::vector<double> X, Y;
    void add(int nn, int m, const double * A, const double * B, double c){   // A, B: n x m column layouts [a*m+k]
        if(m == 0) return;
        n = nn;
        std::vector<double> X2((size_t)n * (r + m)), Y2((size_t)n * (r + m));
        for(int a = 0; a < n; a++){
            for(int k = 0; k < r; k++){ X2[(size_t)a * (r + m) + k] = X[(size_t)a * r + k]; Y2[(size_t)a * (r + m) + k] = Y[(size_t)a * r + k]; }
            for(int k = 0; k < m; k++){ X2[(size_t)a * (r + m) + r + k] = c * A[(size_t)a * m + k]; Y2[(size_t)a * (r + m) + r + k] = B[(size_t)a * m + k]; }
        }
        X.swap(X2); Y.swap(Y2); r += m;
    }
};

// Active-space Hamiltonian of the orbitals C: core energy (with V_NN), active block of the core Fock h + 2J[DMC] - K[DMC], (tu|vw).
// Optional: DMC (core density, n x n, no factor 2), the core Fock Hc and the active transform La (for cas_objects and the JVP).
void build_active(const CasRef & R, const double * C, ActInts * out, std::vector<double> * DMC_out = nullptr, std::vector<double> * Hc_out = nullptr,
                  std::vector<double> * La_out = nullptr){
    const int n = R.n, nc = R.nc, na = R.na, naux = R.naux; const size_t n2 = (size_t)n * n;
    const RIEig & E = R.ri();
    std::vector<double> DMC(n2, 0.0), J(n2), K(n2), Hc(n2);
    if(nc) gemm(n, n, nc, C, C, DMC.data(), 1.0, 0.0, true, false);
    {
        std::vector<double> Cc; columns(R, C, &Cc, nullptr, nullptr);
        LowRank F; F.add(n, nc, Cc.data(), Cc.data(), 1.0);
        ri_fock_JK_eig_lr(E, DMC.data(), F.X.data(), F.Y.data(), F.r, J.data(), K.data());
    }
    for(size_t k = 0; k < n2; k++) Hc[k] = R.h[k] + 2 * J[k] - K[k];
    std::vector<double> Ca; columns(R, C, nullptr, &Ca, nullptr);
    std::vector<double> T((size_t)n * na);
    out->HAA.assign((size_t)na * na, 0.0);
    gemm(n, na, n, Hc.data(), Ca.data(), T.data());
    gemm(na, na, n, Ca.data(), T.data(), out->HAA.data(), 1.0, 0.0, true, false);
    double ec = R.Enuc;
    for(size_t k = 0; k < n2; k++) ec += DMC[k] * (R.h[k] + Hc[k]);
    out->Ecore = ec;
    std::vector<double> La((size_t)na * na * naux);
    ri_active_La(E, Ca.data(), na, La.data());
    act_from_La(R, La.data(), nullptr, &out->act);
    if(DMC_out) DMC_out->swap(DMC);
    if(Hc_out) Hc_out->swap(Hc);
    if(La_out) La_out->swap(La);
}

// H c and the (per-state) densities. vecs: nvec x Nd state-major; Hv same layout; gamma nvec x na^2, Gamma nvec x na^4 (either may be null).
// Everything is bilinear in the vectors (no normalization here) and H c is linear in (act, HAA, Ecore).
void ci_kernel(const CasRef & R, const ActInts & A, const double * vecs, int nvec, double * Hv, double * gamma, double * Gamma, double * Hdiag = nullptr){
    omp_set_num_threads(num_threads);   // aldet partitions its loops over `num_threads` (global) and indexes per-thread buffers by omp_get_thread_num()
    aldet_data X;
    X.get_dim(R.ci_template->n_act, R.ci_template->na, R.ci_template->nb, 1, R.ci_template->mult, 0);
    X.init_zero_vec(nvec, 0);
    X.simple_import_data(const_cast<double *>(A.act.data()), const_cast<double *>(A.act.data()), const_cast<double *>(A.HAA.data()), A.Ecore);
    X.U_simple_import_data(const_cast<double *>(A.act.data()), const_cast<double *>(A.act.data()), const_cast<double *>(A.act.data()),
                           const_cast<double *>(A.HAA.data()), const_cast<double *>(A.HAA.data()), A.Ecore);
    X.gen_ext_ind();
    const size_t Nd = R.Nd;
    if(Hdiag){ if(X.H_diag == nullptr) X.H_diag_calc(); for(size_t d = 0; d < Nd; d++) Hdiag[d] = X.H_diag[d]; }
    for(int s = 0; s < nvec; s++) for(size_t d = 0; d < Nd; d++) X.coef[0][d * nvec + s] = vecs[(size_t)s * Nd + d];
    if(Hv){
        X.H_mult(0, nvec);
        for(int s = 0; s < nvec; s++) for(size_t d = 0; d < Nd; d++) Hv[(size_t)s * Nd + d] = X.Hv[d * nvec + s];
    }
    if(gamma || Gamma) X.transpose_ci(0);
    if(gamma){ std::fill(gamma, gamma + (size_t)nvec * R.na * R.na, 0.0); X.calc_DM_diag(gamma, 0); }
    if(Gamma){ std::fill(Gamma, Gamma + (size_t)nvec * R.na * R.na * R.na * R.na, 0.0); X.G_calc(Gamma); }
}

void normalized(const CasRef & R, const double * vecs, int nvec, std::vector<double> * out){
    out->assign(vecs, vecs + (size_t)nvec * R.Nd);
    for(int s = 0; s < nvec; s++){ double q = std::sqrt(dot(&(*out)[(size_t)s * R.Nd], &(*out)[(size_t)s * R.Nd], R.Nd)); for(size_t d = 0; d < R.Nd; d++) (*out)[(size_t)s * R.Nd + d] /= q; }
}

// P_perp of the REFERENCE SA space applied to every state vector of v (ns x Nd)
void project_perp(const CasRef & R, double * v){
    for(int I = 0; I < R.ns; I++) for(int J = 0; J < R.ns; J++){
        const double o = dot(&R.c0[(size_t)J * R.Nd], &v[(size_t)I * R.Nd], R.Nd);
        for(size_t d = 0; d < R.Nd; d++) v[(size_t)I * R.Nd + d] -= o * R.c0[(size_t)J * R.Nd + d];
    }
}

// CI part of a direction x: dc_I = P_perp delta_I + sum_J rho_IJ c0_J (rho antisymmetric)
void ci_direction(const CasRef & R, const double * x, std::vector<double> * dc){
    dc->assign(x + R.nk(), x + R.ro());
    project_perp(R, dc->data());
    for(size_t k = 0; k < R.nr(); k++){
        const int I = R.rho[k].first, J = R.rho[k].second; const double t = x[R.ro() + k];
        for(size_t d = 0; d < R.Nd; d++){ (*dc)[(size_t)I * R.Nd + d] += t * R.c0[(size_t)J * R.Nd + d]; (*dc)[(size_t)J * R.Nd + d] -= t * R.c0[(size_t)I * R.Nd + d]; }
    }
}

// lam_tuvw = G_tuvw - (g_tu g_vw - 1/2 g_tv g_uw)  (the active two-body density that multiplies (tu|vw) in E, cumulant-like)
void make_lam(int na, const double * g, const double * G, double * lam){
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
        const double gm = g[t * na + u] * g[v * na + w] - 0.5 * g[t * na + v] * g[u * na + w];
        lam[(((size_t)t * na + u) * na + v) * na + w] = G[(((size_t)t * na + u) * na + v) * na + w] - gm;
    }
}
}

std::vector<double> CasRef::root_vector(int root) const{
    return std::vector<double>(c_all.begin() + (size_t)root * Nd, c_all.begin() + (size_t)(root + 1) * Nd);
}

int CasRef::rotate_active(const double * U){
    // orbitals: new active row i = sum_k U[k][i] old_k
    std::vector<double> Ca((size_t)na * n, 0.0);
    for(int i = 0; i < na; i++) for(int k = 0; k < na; k++){ const double u = U[(size_t)k * na + i]; if(u == 0.0) continue;
        for(int a = 0; a < n; a++) Ca[(size_t)i * n + a] += u * C0[(size_t)(nc + k) * n + a]; }
    std::copy(Ca.begin(), Ca.end(), C0.begin() + (size_t)nc * n);
    // CI vectors of every root
    omp_set_num_threads(num_threads);
    aldet_data X;
    X.get_dim(ci_template->n_act, ci_template->na, ci_template->nb, 1, ci_template->mult, 0);
    X.init_zero_vec(ns_all, 0);
    for(int s = 0; s < ns_all; s++) for(size_t d = 0; d < Nd; d++) X.coef[0][d * ns_all + s] = c_all[(size_t)s * Nd + d];
    std::vector<double> Uc(U, U + (size_t)na * na);
    X.malmqvist(0, Uc.data());
    for(int s = 0; s < ns_all; s++) for(size_t d = 0; d < Nd; d++) c_all[(size_t)s * Nd + d] = X.coef[0][d * ns_all + s];
    for(int I = 0; I < ns; I++) for(size_t d = 0; d < Nd; d++) c0[(size_t)I * Nd + d] = c_all[(size_t)states[I] * Nd + d];
    base_.reset();
    return 0;
}
int CasRef::slot(int root) const{ for(int I = 0; I < ns; I++) if(states[I] == root) return I; return -1; }

const RIEig & CasRef::ri() const{ if(ri_.empty()) ri_.build(n, naux, Lp, V.data()); return ri_; }

// ---- densities -> objects --------------------------------------------------------------------------------
namespace{
// The objects of the energy of the densities g (na^2), G (na^4) in the orbitals C. Optional inputs from build_active at the same C (La).
void objects_from_densities(const CasRef & R, const double * C, const double * g, const double * G, bool want_LV, const std::vector<double> * La_in,
                            CasObjects * out, std::vector<double> * J_out = nullptr, std::vector<double> * K_out = nullptr){
    const int n = R.n, nc = R.nc, na = R.na, nmo = R.nmo, naux = R.naux; const size_t n2 = (size_t)n * n, a2 = (size_t)na * na, a4 = a2 * a2;
    const RIEig & E = R.ri();
    std::vector<double> Cc, Ca, Call; columns(R, C, &Cc, &Ca, &Call);
    std::vector<double> d(n2, 0.0), gam_ao(n2, 0.0), tmp((size_t)n * std::max(na, 1));
    if(nc) gemm(n, n, nc, Cc.data(), Cc.data(), d.data(), 2.0, 0.0, false, true);
    if(na){ gemm(n, na, na, Ca.data(), g, tmp.data()); gemm(n, n, na, tmp.data(), Ca.data(), gam_ao.data(), 1.0, 0.0, false, true); }
    out->D.resize(n2);
    for(size_t k = 0; k < n2; k++) out->D[k] = d[k] + gam_ao[k];
    std::vector<double> lam(a4);
    make_lam(na, g, G, lam.data());
    std::vector<double> J(n2), K(n2), Fd(n2);
    {
        LowRank F; F.add(n, nc, Cc.data(), Cc.data(), 2.0);
        if(na){ std::vector<double> Cg((size_t)n * na); gemm(n, na, na, Ca.data(), g, Cg.data()); F.add(n, na, Cg.data(), Ca.data(), 1.0); }
        ri_fock_JK_eig_lr(E, out->D.data(), F.X.data(), F.Y.data(), F.r, J.data(), K.data());
    }
    double e1 = 0, ej = 0, ek = 0;
    for(size_t k = 0; k < n2; k++){ e1 += out->D[k] * R.h[k]; ej += out->D[k] * J[k]; ek += out->D[k] * K[k]; }
    // RI adjoints are kept in the eigenbasis of V (converted once, in cas_geo_contract)
    if(want_LV){
        out->Lbar.assign(n2 * naux, 0.0); out->Vbar.assign((size_t)naux * naux, 0.0);
        ri_jk_vjp_eig(E, out->D.data(), out->D.data(), 0.5, -0.25, out->Lbar.data(), out->Vbar.data());
    }
    else { out->Lbar.clear(); out->Vbar.clear(); }
    std::vector<double> GCa((size_t)n * std::max(na, 1), 0.0);
    double e_lam = 0;
    if(na){
        std::vector<double> La_own;
        const double * La = La_in ? La_in->data() : nullptr;
        if(!La){ La_own.resize(a2 * naux); ri_active_La(E, Ca.data(), na, La_own.data()); La = La_own.data(); }
        ri_active_vjp_eig(E, Ca.data(), na, lam.data(), La, want_LV ? out->Lbar.data() : nullptr, want_LV ? out->Vbar.data() : nullptr, GCa.data(), &e_lam);
    }
    out->E = R.Enuc + e1 + 0.5 * ej - 0.25 * ek + e_lam;

    for(size_t k = 0; k < n2; k++) Fd[k] = R.h[k] + J[k] - 0.5 * K[k];
    out->GC.assign((size_t)n * nmo, 0.0);
    std::vector<double> FC((size_t)n * std::max(nc, na));
    if(nc){ gemm(n, nc, n, Fd.data(), Cc.data(), FC.data()); for(int a = 0; a < n; a++) for(int i = 0; i < nc; i++) out->GC[(size_t)a * nmo + i] = 4.0 * FC[(size_t)a * nc + i]; }
    if(na){
        std::vector<double> FCa((size_t)n * na), FCg((size_t)n * na);
        gemm(n, na, n, Fd.data(), Ca.data(), FCa.data());
        gemm(n, na, na, FCa.data(), g, FCg.data());
        for(int a = 0; a < n; a++) for(int t = 0; t < na; t++) out->GC[(size_t)a * nmo + nc + t] = 2.0 * FCg[(size_t)a * na + t] + GCa[(size_t)a * na + t];
    }
    std::vector<double> CtG((size_t)nmo * nmo), Wmo((size_t)nmo * nmo), CW((size_t)n * nmo);
    gemm(nmo, nmo, n, Call.data(), out->GC.data(), CtG.data(), 1.0, 0.0, true, false);
    for(int p = 0; p < nmo; p++) for(int q = 0; q < nmo; q++) Wmo[(size_t)p * nmo + q] = 0.25 * (CtG[(size_t)p * nmo + q] + CtG[(size_t)q * nmo + p]);
    gemm(n, nmo, nmo, Call.data(), Wmo.data(), CW.data());
    out->Wao.assign(n2, 0.0);
    gemm(n, n, nmo, CW.data(), Call.data(), out->Wao.data(), 1.0, 0.0, false, true);
    if(J_out) J_out->swap(J);
    if(K_out) K_out->swap(K);
}

// The RI adjoints (eigenbasis of V) of the energy of the densities (g, G) in the orbitals C, scaled by s and ACCUMULATED into Lbt / Vbt
// (the Lbar / Vbar part of objects_from_densities without a fresh n^2 naux array).
void lv_accumulate(const CasRef & R, const double * C, const double * g, const double * G, const std::vector<double> * La_in, double s,
                   std::vector<double> & Lbt, std::vector<double> & Vbt){
    const int n = R.n, nc = R.nc, na = R.na; const size_t n2 = (size_t)n * n, a2 = (size_t)na * na, a4 = a2 * a2;
    const RIEig & E = R.ri();
    std::vector<double> Cc, Ca, Call; columns(R, C, &Cc, &Ca, &Call);
    std::vector<double> D(n2, 0.0), gam(n2, 0.0), tmp((size_t)n * std::max(na, 1));
    if(nc) gemm(n, n, nc, Cc.data(), Cc.data(), D.data(), 2.0, 0.0, false, true);
    if(na){ gemm(n, na, na, Ca.data(), g, tmp.data()); gemm(n, n, na, tmp.data(), Ca.data(), gam.data(), 1.0, 0.0, false, true); }
    for(size_t k = 0; k < n2; k++) D[k] += gam[k];
    ri_jk_vjp_eig(E, D.data(), D.data(), 0.5 * s, -0.25 * s, Lbt.data(), Vbt.data());
    if(na){
        std::vector<double> lam(a4); make_lam(na, g, G, lam.data());
        for(double & x : lam) x *= s;
        std::vector<double> La_own; const double * La = La_in ? La_in->data() : nullptr;
        if(!La){ La_own.resize(a2 * R.naux); ri_active_La(E, Ca.data(), na, La_own.data()); La = La_own.data(); }
        double e = 0; std::vector<double> GCa((size_t)n * na);
        ri_active_vjp_eig(E, Ca.data(), na, lam.data(), La, Lbt.data(), Vbt.data(), GCa.data(), &e);
    }
}

// direct += objects(g, G) - objects(g0, G0): the energy of the difference of two density sets (affine in the densities), RI part accumulated in place
void add_objects_difference(const CasRef & R, const double * g, const double * G, const double * g0, const double * G0, const std::vector<double> * La,
                            CasObjects & direct, double * Ediff = nullptr, const double * C = nullptr){
    if(!C) C = R.C0.data();
    CasObjects o1, o0;
    objects_from_densities(R, C, g, G, false, La, &o1);
    objects_from_densities(R, C, g0, G0, false, La, &o0);
    for(size_t k = 0; k < direct.D.size(); k++) direct.D[k] += o1.D[k] - o0.D[k];
    for(size_t k = 0; k < direct.GC.size(); k++) direct.GC[k] += o1.GC[k] - o0.GC[k];
    if(!direct.Lbar.empty()){
        lv_accumulate(R, C, g, G, La, 1.0, direct.Lbar, direct.Vbar);
        lv_accumulate(R, C, g0, G0, La, -1.0, direct.Lbar, direct.Vbar);
    }
    if(Ediff) *Ediff = o1.E - o0.E;
}

// weighted densities of a set of vectors (normalized here)
void weighted_densities(const CasRef & R, const ActInts & A, const double * vecs, int nvec, const double * wts, std::vector<double> * g, std::vector<double> * G){
    const size_t a2 = (size_t)R.na * R.na, a4 = a2 * a2;
    std::vector<double> cn; normalized(R, vecs, nvec, &cn);
    std::vector<double> gam((size_t)nvec * a2), Gam((size_t)nvec * a4);
    ci_kernel(R, A, cn.data(), nvec, nullptr, gam.data(), Gam.data());
    g->assign(a2, 0.0); G->assign(a4, 0.0);
    for(int s = 0; s < nvec; s++){
        for(size_t k = 0; k < a2; k++) (*g)[k] += wts[s] * gam[(size_t)s * a2 + k];
        for(size_t k = 0; k < a4; k++) (*G)[k] += wts[s] * Gam[(size_t)s * a4 + k];
    }
}
}

int cas_objects(const CasRef & R, const double * C, const double * vecs, int nvec, const double * wts, bool want_LV, CasObjects * out){
    ActInts A; std::vector<double> La;
    build_active(R, C, &A, nullptr, nullptr, &La);
    std::vector<double> g, G;
    weighted_densities(R, A, vecs, nvec, wts, &g, &G);
    objects_from_densities(R, C, g.data(), G.data(), want_LV, &La, out);
    return 0;
}

void cas_sa_fock_diag(const CasRef & R, const double * C, const double * gamma, std::vector<double> * d){
    const int n = R.n, nc = R.nc, na = R.na, nmo = R.nmo; const size_t n2 = (size_t)n * n;
    std::vector<double> Cc, Ca; columns(R, C, &Cc, &Ca, nullptr);
    std::vector<double> D(n2, 0.0), J(n2), K(n2), tmp((size_t)n * std::max(na, 1));
    LowRank F;
    if(nc){ gemm(n, n, nc, Cc.data(), Cc.data(), D.data(), 2.0, 0.0, false, true); F.add(n, nc, Cc.data(), Cc.data(), 2.0); }
    if(na){ gemm(n, na, na, Ca.data(), gamma, tmp.data()); gemm(n, n, na, tmp.data(), Ca.data(), D.data(), 1.0, 1.0, false, true); F.add(n, na, tmp.data(), Ca.data(), 1.0); }
    ri_fock_JK_eig_lr(R.ri(), D.data(), F.X.data(), F.Y.data(), F.r, J.data(), K.data());
    d->assign(nmo, 0.0);
    for(int p = 0; p < nmo; p++){
        const double * c = C + (size_t)p * n;
        double e = 0;
        for(int a = 0; a < n; a++){ double t = 0; for(int b = 0; b < n; b++) t += (R.h[(size_t)a * n + b] + J[(size_t)a * n + b] - 0.5 * K[(size_t)a * n + b]) * c[b]; e += c[a] * t; }
        (*d)[p] = e;
    }
}

void cas_active_integrals(const CasRef & R, const double * C, std::vector<double> * HAA, std::vector<double> * act, double * Ecore){
    ActInts A;
    build_active(R, C, &A);
    if(HAA) HAA->swap(A.HAA);
    if(act) act->swap(A.act);
    if(Ecore) *Ecore = A.Ecore;
}

int cas_objects_active_linear(const CasRef & R, const double * C, const double * HAAb, const double * actb, bool want_LV, CasObjects * out){
    // E(g, G) of objects_from_densities is E_core + sum g_tu HAA_tu + 1/2 sum G_tuvw (tu|vw), affine in (g, G) at fixed orbitals, and so are its
    // objects: the objects of the linear functional are objects(g = sym HAAb, G = 2 sym8 actb) - objects(0, 0). Both seeds are symmetrized
    // (HAA is symmetric, (tu|vw) has the 8-fold symmetry), which leaves the functional unchanged and makes its derivatives those of the symmetric forms.
    const int na = R.na; const size_t a2 = (size_t)na * na, a4 = a2 * a2;
    std::vector<double> g(a2), G(a4), g0(a2, 0.0), G0(a4, 0.0);
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) g[(size_t)t * na + u] = 0.5 * (HAAb[(size_t)t * na + u] + HAAb[(size_t)u * na + t]);
    auto ix = [na](int t, int u, int v, int w){ return (((size_t)t * na + u) * na + v) * na + w; };
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int w = 0; w < na; w++)
        G[ix(t, u, v, w)] = 0.25 * (actb[ix(t, u, v, w)] + actb[ix(u, t, v, w)] + actb[ix(t, u, w, v)] + actb[ix(u, t, w, v)]
                                  + actb[ix(v, w, t, u)] + actb[ix(w, v, t, u)] + actb[ix(v, w, u, t)] + actb[ix(w, v, u, t)]);   // 2 x the 8-fold average
    ActInts A; std::vector<double> La;
    build_active(R, C, &A, nullptr, nullptr, &La);
    // objects(g, G) - objects(0, 0), the RI part accumulated into one array
    const size_t n2 = (size_t)R.n * R.n;
    out->D.assign(n2, 0.0); out->GC.assign((size_t)R.n * R.nmo, 0.0); out->Wao.clear();
    if(want_LV){ out->Lbar.assign(n2 * R.naux, 0.0); out->Vbar.assign((size_t)R.naux * R.naux, 0.0); } else { out->Lbar.clear(); out->Vbar.clear(); }
    double ed = 0;
    add_objects_difference(R, g.data(), G.data(), g0.data(), G0.data(), &La, *out, &ed, C);
    out->E = ed;
    // the value must be the functional itself
    double e = 0, sc = 0;
    for(size_t k = 0; k < a2; k++){ e += HAAb[k] * A.HAA[k]; sc += std::fabs(HAAb[k] * A.HAA[k]); }
    for(size_t k = 0; k < a4; k++){ e += actb[k] * A.act[k]; sc += std::fabs(actb[k] * A.act[k]); }
    if(std::fabs(e - out->E) > 1e-10 * std::max(1.0, sc)){
        fprintf(out_stream, "ERROR: objects of the active-integral functional: value %.12e, direct %.12e\n", out->E, e);
        return 1;
    }
    return 0;
}

// ---- the reference SA quantities (for the tangent-linear code) ------------------------------------------------------
struct CasBase{
    std::vector<double> Cc, Ca, Call;     // column layouts of C0
    std::vector<double> g, G, lam;        // SA densities
    CasObjects sa;                        // SA objects at the reference (D, GC, Wao; no Lbar/Vbar)
    std::vector<double> J, K, Fd;         // J[D_SA], K[D_SA], h + J - K/2
    std::vector<double> CtG, Wmo;         // C0^T GC and 1/4 sym of it
    std::vector<double> DMC, Hc, La;      // core density (no factor 2), core Fock, active transform at C0
    ActInts A;                            // active integrals at C0
    std::vector<double> Hc0, Ei;          // H c0 (ns x Nd), state energies <c0_I|H|c0_I>
    // for the exchange of the JVP densities: LX[a][k][i] = sum_b Lt[a,b,i] X[b,k] for X = Cc, Ca, and the same divided by lam_i
    std::vector<double> LCc, LCa, LCc_s, LCa_s;
};

namespace{
// LX[a][k][i] = sum_b Lt[a,b,i] X[b,k]  (X: n x m column layout [b*m+k]); the slices Lt[a,:,:] are contiguous
void lt_times(const RIEig & E, const double * X, int m, std::vector<double> * LX){
    const int n = E.n, naux = E.naux; const size_t mi = (size_t)m * naux;
    LX->assign((size_t)n * mi, 0.0);
    if(m == 0) return;
    set_blas_non_par();
#pragma omp parallel for schedule(static)
    for(int a = 0; a < n; a++)
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, naux, n, 1.0, X, m, E.Lt.data() + (size_t)a * n * naux, naux, 0.0, LX->data() + (size_t)a * mi, naux);
    set_blas_par();
}
}
namespace{
const CasBase & cas_base(const CasRef & R){
    if(R.base_) return *R.base_;
    auto B = std::make_shared<CasBase>();
    const int n = R.n, nmo = R.nmo;
    build_active(R, R.C0.data(), &B->A, &B->DMC, &B->Hc, &B->La);
    weighted_densities(R, B->A, R.c0.data(), R.ns, R.w.data(), &B->g, &B->G);
    B->lam.resize(B->G.size());
    make_lam(R.na, B->g.data(), B->G.data(), B->lam.data());
    objects_from_densities(R, R.C0.data(), B->g.data(), B->G.data(), false, &B->La, &B->sa, &B->J, &B->K);
    columns(R, R.C0.data(), &B->Cc, &B->Ca, &B->Call);
    B->Fd.resize((size_t)n * n);
    for(size_t k = 0; k < B->Fd.size(); k++) B->Fd[k] = R.h[k] + B->J[k] - 0.5 * B->K[k];
    B->CtG.resize((size_t)nmo * nmo); B->Wmo.resize((size_t)nmo * nmo);
    gemm(nmo, nmo, n, B->Call.data(), B->sa.GC.data(), B->CtG.data(), 1.0, 0.0, true, false);
    for(int p = 0; p < nmo; p++) for(int q = 0; q < nmo; q++) B->Wmo[(size_t)p * nmo + q] = 0.25 * (B->CtG[(size_t)p * nmo + q] + B->CtG[(size_t)q * nmo + p]);
    {
        const RIEig & E = R.ri();
        lt_times(E, B->Cc.data(), R.nc, &B->LCc); lt_times(E, B->Ca.data(), R.na, &B->LCa);
        B->LCc_s = B->LCc; B->LCa_s = B->LCa;
        const size_t na_ = R.naux;
        for(size_t k = 0; k < B->LCc_s.size(); k++) B->LCc_s[k] /= E.lam[k % na_];
        for(size_t k = 0; k < B->LCa_s.size(); k++) B->LCa_s[k] /= E.lam[k % na_];
    }
    B->Hc0.resize((size_t)R.ns * R.Nd); B->Ei.resize(R.ns);
    ci_kernel(R, B->A, R.c0.data(), R.ns, B->Hc0.data(), nullptr, nullptr);
    for(int I = 0; I < R.ns; I++) B->Ei[I] = dot(&R.c0[(size_t)I * R.Nd], &B->Hc0[(size_t)I * R.Nd], R.Nd);
    R.base_ = B;
    return *R.base_;
}
}

// ---- residual ------------------------------------------------------------------------------------------------
int cas_residual(const CasRef & R, const double * C, const double * c, std::vector<double> * r, CasObjects * sa_obj, double * E_sa){
    const int n = R.n, nmo = R.nmo, ns = R.ns;
    ActInts A; std::vector<double> La;
    build_active(R, C, &A, nullptr, nullptr, &La);
    std::vector<double> g, G;
    weighted_densities(R, A, c, ns, R.w.data(), &g, &G);
    CasObjects obj;
    objects_from_densities(R, C, g.data(), G.data(), sa_obj != nullptr, &La, &obj);
    if(E_sa) *E_sa = obj.E;
    // orbital residual from C^T G_C
    std::vector<double> Call((size_t)n * nmo), CtG((size_t)nmo * nmo);
    for(int a = 0; a < n; a++) for(int p = 0; p < nmo; p++) Call[(size_t)a * nmo + p] = C[(size_t)p * n + a];
    gemm(nmo, nmo, n, Call.data(), obj.GC.data(), CtG.data(), 1.0, 0.0, true, false);
    r->assign(R.nx(), 0.0);
    for(size_t k = 0; k < R.nk(); k++){ const int p = R.pairs[k].first, q = R.pairs[k].second; (*r)[k] = CtG[(size_t)q * nmo + p] - CtG[(size_t)p * nmo + q]; }
    // CI residual: 2 w_I P_perp (H c_I - E_I c_I) for the normalized vectors, H of the displaced orbitals
    std::vector<double> cn; normalized(R, c, ns, &cn);
    std::vector<double> Hc((size_t)ns * R.Nd);
    ci_kernel(R, A, cn.data(), ns, Hc.data(), nullptr, nullptr);
    for(int I = 0; I < ns; I++){
        const double E = dot(&cn[(size_t)I * R.Nd], &Hc[(size_t)I * R.Nd], R.Nd);
        double * ri = &(*r)[R.nk() + (size_t)I * R.Nd];
        for(size_t d = 0; d < R.Nd; d++) ri[d] = 2.0 * R.w[I] * (Hc[(size_t)I * R.Nd + d] - E * cn[(size_t)I * R.Nd + d]);
    }
    project_perp(R, &(*r)[R.nk()]);
    // rotations between weighted states of different weight: dE_SA/drho_IJ = 2 (w_I - w_J) <c_I|H|c_J>
    for(size_t k = 0; k < R.nr(); k++){
        const int I = R.rho[k].first, J = R.rho[k].second;
        (*r)[R.ro() + k] = 2.0 * (R.w[I] - R.w[J]) * dot(&cn[(size_t)I * R.Nd], &Hc[(size_t)J * R.Nd], R.Nd);
    }
    if(sa_obj) *sa_obj = std::move(obj);
    return 0;
}

void cas_displace(const CasRef & R, const double * x, double eps, std::vector<double> * C, std::vector<double> * c){
    const int n = R.n;
    *C = R.C0;
    for(size_t k = 0; k < R.nk(); k++){
        const int p = R.pairs[k].first, q = R.pairs[k].second; const double t = eps * x[k];
        for(int a = 0; a < n; a++){ (*C)[(size_t)p * n + a] += t * R.C0[(size_t)q * n + a]; (*C)[(size_t)q * n + a] -= t * R.C0[(size_t)p * n + a]; }
    }
    std::vector<double> dc; ci_direction(R, x, &dc);
    *c = R.c0;
    for(size_t i = 0; i < dc.size(); i++) (*c)[i] += eps * dc[i];
}

// ---- tangent-linear code ------------------------------------------------------------------------------------------
// Along x: dC = K C0 (rows: dC_p += x_pq C0_q, dC_q -= x_pq C0_p, the linear displacement of cas_displace) and dc_I = P_perp delta_I.
// Because dc_I is orthogonal to c0_I, the normalization has zero first derivative and the density derivatives are exact polarizations of the
// bilinear forms:  d rho = [rho(c0 + s dc) - rho(c0 - s dc)] / (2 s)  for every s (rho = gamma or Gamma of one state; s only scales the rounding).
double g_jvp_time[6] = {0, 0, 0, 0, 0, 0};   // densities, J/K, active two-electron, CI H, rest, calls
int cas_jvp(const CasRef & R, const double * x, std::vector<double> * Jx, CasObjects * dobj, bool want_LV){
    double tq = omp_get_wtime();
    auto tick = [&](int k){ const double t = omp_get_wtime(); g_jvp_time[k] += t - tq; tq = t; };
    g_jvp_time[5] += 1;
    const CasBase & B = cas_base(R);
    const int n = R.n, nc = R.nc, na = R.na, nmo = R.nmo, naux = R.naux, ns = R.ns; const size_t n2 = (size_t)n * n, a2 = (size_t)na * na, a4 = a2 * a2, Nd = R.Nd;
    const RIEig & E = R.ri();
    // orbital direction
    std::vector<double> dC((size_t)nmo * n, 0.0);
    for(size_t k = 0; k < R.nk(); k++){
        const int p = R.pairs[k].first, q = R.pairs[k].second; const double t = x[k];
        if(t == 0.0) continue;
        for(int a = 0; a < n; a++){ dC[(size_t)p * n + a] += t * R.C0[(size_t)q * n + a]; dC[(size_t)q * n + a] -= t * R.C0[(size_t)p * n + a]; }
    }
    std::vector<double> dCc, dCa, dCall; columns(R, dC.data(), &dCc, &dCa, &dCall);
    // CI direction (complement and rotations within the weighted space) and the density derivatives
    std::vector<double> dc; ci_direction(R, x, &dc);
    std::vector<double> dg(a2, 0.0), dG(a4, 0.0);
    double dmax = 0; for(double v : dc) dmax = std::max(dmax, std::fabs(v));
    if(dmax > 0 && na){
        const double s = 1.0 / std::max(1.0, dmax * std::sqrt((double)Nd));
        std::vector<double> vp(R.c0), vm(R.c0);
        for(size_t i = 0; i < vp.size(); i++){ vp[i] += s * dc[i]; vm[i] -= s * dc[i]; }
        std::vector<double> gp((size_t)ns * a2), gm((size_t)ns * a2), Gp((size_t)ns * a4), Gm((size_t)ns * a4);
        ci_kernel(R, B.A, vp.data(), ns, nullptr, gp.data(), Gp.data());
        ci_kernel(R, B.A, vm.data(), ns, nullptr, gm.data(), Gm.data());
        for(int I = 0; I < ns; I++){
            const double f = R.w[I] / (2.0 * s);
            for(size_t k = 0; k < a2; k++) dg[k] += f * (gp[(size_t)I * a2 + k] - gm[(size_t)I * a2 + k]);
            for(size_t k = 0; k < a4; k++) dG[k] += f * (Gp[(size_t)I * a4 + k] - Gm[(size_t)I * a4 + k]);
        }
    }
    tick(0);
    std::vector<double> dlam(a4);
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int w = 0; w < na; w++){
        const size_t i = (((size_t)t * na + u) * na + v) * na + w;
        const double dgm = dg[t * na + u] * B.g[v * na + w] + B.g[t * na + u] * dg[v * na + w]
                         - 0.5 * (dg[t * na + v] * B.g[u * na + w] + B.g[t * na + v] * dg[u * na + w]);
        dlam[i] = dG[i] - dgm;
    }
    // dD = 2 (dCc Cc^T + Cc dCc^T) + dCa g Ca^T + Ca g dCa^T + Ca dg Ca^T ;  dDMC = dCc Cc^T + Cc dCc^T
    std::vector<double> dD(n2, 0.0), dDMC(n2, 0.0), t1((size_t)n * std::max(na, 1));
    if(nc){ gemm(n, n, nc, dCc.data(), B.Cc.data(), dDMC.data(), 1.0, 0.0, false, true); gemm(n, n, nc, B.Cc.data(), dCc.data(), dDMC.data(), 1.0, 1.0, false, true); }
    for(size_t k = 0; k < n2; k++) dD[k] = 2.0 * dDMC[k];
    if(na){
        gemm(n, na, na, dCa.data(), B.g.data(), t1.data()); gemm(n, n, na, t1.data(), B.Ca.data(), dD.data(), 1.0, 1.0, false, true);
        gemm(n, na, na, B.Ca.data(), B.g.data(), t1.data()); gemm(n, n, na, t1.data(), dCa.data(), dD.data(), 1.0, 1.0, false, true);
        gemm(n, na, na, B.Ca.data(), dg.data(), t1.data()); gemm(n, n, na, t1.data(), B.Ca.data(), dD.data(), 1.0, 1.0, false, true);
    }
    tick(4);
    std::vector<double> dJ(n2), dK(n2), dJ1(n2), dK1(n2);
    {
        // dDMC = dCc Cc^T + Cc dCc^T,  dD = 2 dDMC + (dCa g) Ca^T + Ca (dCa g)^T + Ca dg Ca^T.  With LX = Lt X per auxiliary function (L Cc, L Ca
        // precomputed in the base, L dCc and L dCa here), and the "_s" side divided by lam:
        //   P1 = (L dCc)(L Cc)_s^T,  P2 = (L dCa g)(L Ca)_s^T,  P3 = (L Ca dg)(L Ca)_s^T   (n x n, contracted over (k, i))
        //   K[dDMC] = P1 + P1^T,  K[dD] = 2 K[dDMC] + P2 + P2^T + P3;  the Coulomb coefficients t_i / lam_i from the same factors, both J in one pass.
        const size_t nI = (size_t)naux;
        std::vector<double> LdCc, LdCa;
        lt_times(E, dCc.data(), nc, &LdCc); lt_times(E, dCa.data(), na, &LdCa);
        std::vector<double> LdCag((size_t)n * na * nI, 0.0), LCadg((size_t)n * na * nI, 0.0);
        if(na){
            // (L X g)[a][t][i] = sum_u (L X)[a][u][i] g[u][t]
            set_blas_non_par();
#pragma omp parallel for schedule(static)
            for(int a = 0; a < n; a++){
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, naux, na, 1.0, B.g.data(), na, LdCa.data() + (size_t)a * na * nI, naux, 0.0, LdCag.data() + (size_t)a * na * nI, naux);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, naux, na, 1.0, dg.data(), na, B.LCa.data() + (size_t)a * na * nI, naux, 0.0, LCadg.data() + (size_t)a * na * nI, naux);
            }
            set_blas_par();
        }
        std::vector<double> P1(n2, 0.0), P2(n2, 0.0), P3(n2, 0.0);
        if(nc) gemm(n, n, (int)(nc * nI), LdCc.data(), B.LCc_s.data(), P1.data(), 1.0, 0.0, false, true);
        if(na){ gemm(n, n, (int)(na * nI), LdCag.data(), B.LCa_s.data(), P2.data(), 1.0, 0.0, false, true);
                gemm(n, n, (int)(na * nI), LCadg.data(), B.LCa_s.data(), P3.data(), 1.0, 0.0, false, true); }
        for(int a = 0; a < n; a++) for(int b = 0; b < n; b++){
            const size_t ab = (size_t)a * n + b, ba = (size_t)b * n + a;
            const double k1 = P1[ab] + P1[ba];
            dK1[ab] = k1;
            dK[ab] = 2.0 * k1 + P2[ab] + P2[ba] + P3[ab];
        }
        // Coulomb: c1_i = 2 sum_{a,k} dCc[a,k] (L Cc)_s[a,k,i];  c_i = 2 c1_i + 2 sum (dCa g)[a,t] (L Ca)_s[a,t,i] + sum (Ca dg)[a,t] (L Ca)_s[a,t,i]
        std::vector<double> cc(2 * nI, 0.0);   // [i][0]: dD, [i][1]: dDMC
        std::vector<double> c1(nI, 0.0), c2(nI, 0.0);
        if(nc) cblas_dgemv(CblasRowMajor, CblasTrans, n * nc, naux, 2.0, B.LCc_s.data(), naux, dCc.data(), 1, 0.0, c1.data(), 1);
        if(na){
            std::vector<double> W((size_t)n * na), t1a((size_t)n * na);
            gemm(n, na, na, dCa.data(), B.g.data(), W.data());          // dCa g
            gemm(n, na, na, B.Ca.data(), dg.data(), t1a.data());        // Ca dg
            for(size_t k = 0; k < W.size(); k++) W[k] = 2.0 * W[k] + t1a[k];
            cblas_dgemv(CblasRowMajor, CblasTrans, n * na, naux, 1.0, B.LCa_s.data(), naux, W.data(), 1, 0.0, c2.data(), 1);
        }
        for(size_t i = 0; i < nI; i++){ cc[2 * i] = 2.0 * c1[i] + c2[i]; cc[2 * i + 1] = c1[i]; }
        std::vector<double> JJ(n2 * 2);
        gemm((int)n2, 2, naux, E.Lt.data(), cc.data(), JJ.data());
        for(size_t k = 0; k < n2; k++){ dJ[k] = JJ[2 * k]; dJ1[k] = JJ[2 * k + 1]; }
    }
    tick(1);
    std::vector<double> dFd(n2);
    for(size_t k = 0; k < n2; k++) dFd[k] = dJ[k] - 0.5 * dK[k];
    // active two-electron part
    std::vector<double> dLa(a2 * naux, 0.0), dGCa((size_t)n * std::max(na, 1), 0.0), dLbt, dVbt;
    if(want_LV && dobj){
        // accumulate onto what the caller put there (the direct RI adjoints), otherwise start from zero
        if(dobj->Lbar.size() == n2 * naux && dobj->Vbar.size() == (size_t)naux * naux){ dLbt.swap(dobj->Lbar); dVbt.swap(dobj->Vbar); }
        else { dLbt.assign(n2 * naux, 0.0); dVbt.assign((size_t)naux * naux, 0.0); }
    }
    if(na){
        ri_active_La(E, B.Ca.data(), na, dLa.data(), dCa.data());
        ri_active_jvp_eig(E, B.Ca.data(), dCa.data(), na, B.lam.data(), dlam.data(), B.La.data(), dLa.data(),
                          dLbt.empty() ? nullptr : dLbt.data(), dVbt.empty() ? nullptr : dVbt.data(), dGCa.data());
    }
    tick(2);
    // dGC: core 4 (dFd Cc + Fd dCc); active 2 (dFd Ca g + Fd dCa g + Fd Ca dg) + dGCa
    std::vector<double> dGC((size_t)n * nmo, 0.0);
    if(nc){
        std::vector<double> FC((size_t)n * nc);
        gemm(n, nc, n, dFd.data(), B.Cc.data(), FC.data());
        gemm(n, nc, n, B.Fd.data(), dCc.data(), FC.data(), 1.0, 1.0);
        for(int a = 0; a < n; a++) for(int i = 0; i < nc; i++) dGC[(size_t)a * nmo + i] = 4.0 * FC[(size_t)a * nc + i];
    }
    if(na){
        std::vector<double> FCa((size_t)n * na), FCg((size_t)n * na);
        gemm(n, na, n, dFd.data(), B.Ca.data(), FCa.data());
        gemm(n, na, n, B.Fd.data(), dCa.data(), FCa.data(), 1.0, 1.0);
        gemm(n, na, na, FCa.data(), B.g.data(), FCg.data());
        gemm(n, na, n, B.Fd.data(), B.Ca.data(), FCa.data());
        gemm(n, na, na, FCa.data(), dg.data(), FCg.data(), 1.0, 1.0);
        for(int a = 0; a < n; a++) for(int t = 0; t < na; t++) dGC[(size_t)a * nmo + nc + t] = 2.0 * FCg[(size_t)a * na + t] + dGCa[(size_t)a * na + t];
    }
    // dCtG = dCall^T GC + Call^T dGC
    std::vector<double> dCtG((size_t)nmo * nmo);
    gemm(nmo, nmo, n, dCall.data(), B.sa.GC.data(), dCtG.data(), 1.0, 0.0, true, false);
    gemm(nmo, nmo, n, B.Call.data(), dGC.data(), dCtG.data(), 1.0, 1.0, true, false);

    if(Jx){
        Jx->assign(R.nx(), 0.0);
        for(size_t k = 0; k < R.nk(); k++){ const int p = R.pairs[k].first, q = R.pairs[k].second; (*Jx)[k] = dCtG[(size_t)q * nmo + p] - dCtG[(size_t)p * nmo + q]; }
        // CI: dr_I = 2 w_I P_perp (dH c0_I + (H - E_I) dc_I)   (the dE_I c0_I term is removed by P_perp)
        ActInts dA;
        std::vector<double> dHc(n2);
        for(size_t k = 0; k < n2; k++) dHc[k] = 2.0 * dJ1[k] - dK1[k];
        dA.HAA.assign(a2, 0.0);
        if(na){
            std::vector<double> T((size_t)n * na);
            gemm(n, na, n, B.Hc.data(), B.Ca.data(), T.data());
            gemm(na, na, n, dCa.data(), T.data(), dA.HAA.data(), 1.0, 0.0, true, false);
            gemm(n, na, n, B.Hc.data(), dCa.data(), T.data());
            gemm(na, na, n, B.Ca.data(), T.data(), dA.HAA.data(), 1.0, 1.0, true, false);
            gemm(n, na, n, dHc.data(), B.Ca.data(), T.data());
            gemm(na, na, n, B.Ca.data(), T.data(), dA.HAA.data(), 1.0, 1.0, true, false);
        }
        double dec = 0;
        for(size_t k = 0; k < n2; k++) dec += dDMC[k] * (R.h[k] + B.Hc[k]) + B.DMC[k] * dHc[k];
        dA.Ecore = dec;
        if(na) act_from_La(R, B.La.data(), dLa.data(), &dA.act); else dA.act.assign(a4, 0.0);
        std::vector<double> s1((size_t)ns * Nd), s2((size_t)ns * Nd, 0.0);
        tick(4);
        ci_kernel(R, dA, R.c0.data(), ns, s1.data(), nullptr, nullptr);
        if(dmax > 0) ci_kernel(R, B.A, dc.data(), ns, s2.data(), nullptr, nullptr);
        tick(3);
        double * rc = &(*Jx)[R.nk()];
        for(int I = 0; I < ns; I++) for(size_t d = 0; d < Nd; d++){
            const size_t i = (size_t)I * Nd + d;
            rc[i] = 2.0 * R.w[I] * (s1[i] + s2[i] - B.Ei[I] * dc[i]);
        }
        project_perp(R, rc);
        // rho rows: d [2 (w_I - w_J) <c_I|H|c_J>] = 2 (w_I - w_J) (<dc_I|H c_J> + <H c_I|dc_J> + <c_I|dH c_J>)
        for(size_t k = 0; k < R.nr(); k++){
            const int I = R.rho[k].first, J = R.rho[k].second;
            const double v = dot(&dc[(size_t)I * Nd], &B.Hc0[(size_t)J * Nd], Nd) + dot(&B.Hc0[(size_t)I * Nd], &dc[(size_t)J * Nd], Nd)
                           + dot(&R.c0[(size_t)I * Nd], &s1[(size_t)J * Nd], Nd);
            (*Jx)[R.ro() + k] = 2.0 * (R.w[I] - R.w[J]) * v;
        }
    }
    if(dobj){
        dobj->E = 0.0;
        dobj->D = dD;
        dobj->GC = dGC;
        // dWao = dCall Wmo Call^T + Call Wmo dCall^T + Call dWmo Call^T,  dWmo = 1/4 sym(dCtG)
        std::vector<double> dWmo((size_t)nmo * nmo), CW((size_t)n * nmo);
        for(int p = 0; p < nmo; p++) for(int q = 0; q < nmo; q++) dWmo[(size_t)p * nmo + q] = 0.25 * (dCtG[(size_t)p * nmo + q] + dCtG[(size_t)q * nmo + p]);
        dobj->Wao.assign(n2, 0.0);
        gemm(n, nmo, nmo, dCall.data(), B.Wmo.data(), CW.data());
        gemm(n, n, nmo, CW.data(), B.Call.data(), dobj->Wao.data(), 1.0, 0.0, false, true);
        gemm(n, nmo, nmo, B.Call.data(), B.Wmo.data(), CW.data());
        gemm(n, n, nmo, CW.data(), dCall.data(), dobj->Wao.data(), 1.0, 1.0, false, true);
        gemm(n, nmo, nmo, B.Call.data(), dWmo.data(), CW.data());
        gemm(n, n, nmo, CW.data(), B.Call.data(), dobj->Wao.data(), 1.0, 1.0, false, true);
        if(want_LV){
            // d of the J/K seeds of D: 2 x the bilinear seeds of (D, dD)
            ri_jk_vjp_eig(E, B.sa.D.data(), dD.data(), 1.0, -0.5, dLbt.data(), dVbt.data());
            dobj->Lbar = std::move(dLbt); dobj->Vbar = std::move(dVbt);     // eigenbasis of V, like every CasObjects
        }
        else { dobj->Lbar.clear(); dobj->Vbar.clear(); }
    }
    tick(4);
    return 0;
}

void cas_hessian_vec(const CasRef & R, const std::vector<double> & p, std::vector<double> * Jp){
    cas_jvp(R, p.data(), Jp, nullptr, false);
}

void cas_hessian_vec_fd(const CasRef & R, const std::vector<double> & p, std::vector<double> * Jp){
    double mx = 0; for(double v : p) mx = std::max(mx, std::fabs(v));
    Jp->assign(R.nx(), 0.0);
    if(mx == 0.0) return;
    // 4-point central difference of the analytic residual along p: truncation ~ eps^4, noise ~ 1e-16 |r| / eps
    const double eps = (getenv("NOPT_RESP_HEPS") && *getenv("NOPT_RESP_HEPS") ? atof(getenv("NOPT_RESP_HEPS")) : 2e-4) / mx;
    const double k[4] = {2.0, 1.0, -1.0, -2.0}, f[4] = {-1.0, 8.0, -8.0, 1.0};
    for(int i = 0; i < 4; i++){
        std::vector<double> Cd, cd, rd;
        cas_displace(R, p.data(), k[i] * eps, &Cd, &cd);
        cas_residual(R, Cd.data(), cd.data(), &rd, nullptr, nullptr);
        for(size_t j = 0; j < Jp->size(); j++) (*Jp)[j] += f[i] * rd[j] / (12.0 * eps);
    }
    project_perp(R, &(*Jp)[R.nk()]);
}

// Diagonal preconditioner of the Hessian (positive, approximate): orbital part from the SA Fock diagonal and the occupations, CI part 2 w_I |H_dd - E_I|.
static void cas_precond(const CasRef & R, std::vector<double> * dg){
    const int n = R.n, nmo = R.nmo, nc = R.nc, na = R.na;
    const CasBase & B = cas_base(R);
    std::vector<double> FC((size_t)nmo * n);
    gemm(nmo, n, n, R.C0.data(), B.Fd.data(), FC.data());
    std::vector<double> Fpp(nmo);
    const std::vector<double> & gsa = B.g;
    for(int p = 0; p < nmo; p++) Fpp[p] = dot(&FC[(size_t)p * n], &R.C0[(size_t)p * n], n);
    dg->assign(R.nx(), 1.0);
    for(size_t k = 0; k < R.nk(); k++){
        const int p = R.pairs[k].first, q = R.pairs[k].second; double v;
        if(p < nc && q >= nc + na) v = 4.0 * (Fpp[q] - Fpp[p]);
        else if(p < nc) v = 2.0 * (2.0 - gsa[(size_t)(q - nc) * na + (q - nc)]) * (Fpp[q] - Fpp[p]);
        else v = 2.0 * gsa[(size_t)(p - nc) * na + (p - nc)] * (Fpp[q] - Fpp[p]);
        (*dg)[k] = std::max(v, 0.3);
    }
    std::vector<double> Hc((size_t)R.ns * R.Nd), Hd(R.Nd);
    ci_kernel(R, B.A, R.c0.data(), R.ns, Hc.data(), nullptr, nullptr, Hd.data());
    for(int I = 0; I < R.ns; I++)
        for(size_t d = 0; d < R.Nd; d++) (*dg)[R.nk() + (size_t)I * R.Nd + d] = std::max(2.0 * R.w[I] * std::fabs(Hd[d] - B.Ei[I]), 0.3);
    for(size_t k = 0; k < R.nr(); k++){
        const int I = R.rho[k].first, J = R.rho[k].second;
        (*dg)[R.ro() + k] = std::max(std::fabs(2.0 * (R.w[I] - R.w[J]) * (B.Ei[J] - B.Ei[I])), 1e-2);
    }
}

double g_cas_z_residual = 0.0;   // final relative residual of the last cas_solve_z

// Preconditioned MINRES (Paige-Saunders) for a symmetric, possibly indefinite operator; prec must be symmetric positive definite (here a positive
// diagonal followed by a projection). Stops on its own residual estimate; the caller checks the true residual. Returns the iteration count.
namespace{
int minres(size_t N, const std::function<void(const std::vector<double> &, std::vector<double> *)> & apply,
           const std::function<void(const std::vector<double> &, std::vector<double> *)> & prec,
           const std::vector<double> & b, std::vector<double> * z, double tol, int maxit, const char * label, bool verbose){
    z->assign(N, 0.0);
    std::vector<double> r1 = b, r2 = b, y, v(N), w(N, 0.0), w1(N, 0.0), w2(N, 0.0), Av;
    prec(r1, &y);
    const double beta1 = std::sqrt(std::max(dot(r1.data(), y.data(), N), 0.0));
    if(beta1 == 0.0) return 0;
    double oldb = 0.0, beta = beta1, dbar = 0.0, epsln = 0.0, phibar = beta1, cs = -1.0, sn = 0.0;
    int it = 0;
    for(it = 1; it <= maxit; it++){
        const double s = 1.0 / beta;
        for(size_t i = 0; i < N; i++) v[i] = s * y[i];
        apply(v, &Av);
        y = Av;
        if(it >= 2) for(size_t i = 0; i < N; i++) y[i] -= (beta / oldb) * r1[i];
        const double alfa = dot(v.data(), y.data(), N);
        for(size_t i = 0; i < N; i++) y[i] -= (alfa / beta) * r2[i];
        r1.swap(r2); r2 = y;
        prec(r2, &y);
        oldb = beta;
        beta = std::sqrt(std::max(dot(r2.data(), y.data(), N), 0.0));
        const double oldeps = epsln;
        const double delta = cs * dbar + sn * alfa;
        const double gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;
        const double gamma = std::max(std::sqrt(gbar * gbar + beta * beta), 1e-300);
        cs = gbar / gamma; sn = beta / gamma;
        const double phi = cs * phibar;
        phibar = sn * phibar;
        w1.swap(w2); w2.swap(w);
        for(size_t i = 0; i < N; i++){ w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) / gamma; (*z)[i] += phi * w[i]; }
        const double est = phibar / beta1;
        if(verbose) fprintf(out_stream, "   %s MINRES iteration %3d: estimated |residual| / |rhs| = %.3e\n", label, it, est);
        if(est < tol || beta == 0.0) return it;
    }
    return maxit + 1;
}
}

// The Z-vector: MINRES on the SA Hessian (indefinite e.g. when a zero-weight root lies below a weighted one) with the diagonal preconditioner of cas_precond
// followed by the projection onto the complement of the SA space. Acceptance by the true relative residual |rhs - J z| / |rhs| <= 1e-6, from one more
// Hessian action.
int cas_solve_z(const CasRef & R, const std::vector<double> & rhs, std::vector<double> * z, double tol, int maxit){
    for(double & t : g_jvp_time) t = 0;
    const size_t N = R.nx();
    g_cas_z_residual = 0.0;
    z->assign(N, 0.0);
    std::vector<double> dg; cas_precond(R, &dg);
    auto prec = [&](const std::vector<double> & r, std::vector<double> * y){ y->resize(N); for(size_t i = 0; i < N; i++) (*y)[i] = r[i] / dg[i]; project_perp(R, &(*y)[R.nk()]); };
    auto apply = [&](const std::vector<double> & v, std::vector<double> * Av){ cas_hessian_vec(R, v, Av); };
    std::vector<double> b = rhs;
    project_perp(R, &b[R.nk()]);
    const double bnorm = std::sqrt(dot(b.data(), b.data(), N));
    if(bnorm == 0.0) return 0;
    const int it = minres(N, apply, prec, b, z, tol, maxit, "Z-vector", true);
    std::vector<double> Jz; cas_hessian_vec(R, *z, &Jz);
    double rr = 0; for(size_t i = 0; i < N; i++){ const double d = b[i] - Jz[i]; rr += d * d; }
    g_cas_z_residual = std::sqrt(rr) / bnorm;
    if(it > maxit) fprintf(out_stream, "WARNING: MINRES reached %d iterations; true |residual| / |rhs| = %.2e\n", maxit, g_cas_z_residual);
    const int n_it = std::min(it, maxit);
    return g_cas_z_residual <= 1e-6 ? n_it : -n_it;
}

// ---- geometry contraction --------------------------------------------------------------------------------------
int cas_geo_contract(molecule * M, const CasRef & R, CasObjects & obj, double * g){
    const int n = R.n, naux = R.naux; const size_t n2 = (size_t)n * n;
    // the RI adjoints arrive in the eigenbasis of V: to the auxiliary basis of the integrals, in place
    ri_eig_to_aux_inplace(R.ri(), obj.Lbar.empty() ? nullptr : obj.Lbar.data(), obj.Vbar.empty() ? nullptr : obj.Vbar.data());
    deriv_kinetic_contract(M, obj.D.data(), g);
    deriv_nuclear_contract(M, obj.D.data(), g);
    deriv_ri3_contract(M, [&](const int *, const int * o, const int * nn, double * w){
        for(int p = 0; p < nn[0]; p++) for(int a = 0; a < nn[1]; a++) for(int b = 0; b < nn[2]; b++)
            w[(p * nn[1] + a) * nn[2] + b] = obj.Lbar[((size_t)(o[1] + a) * n + o[2] + b) * naux + o[0] + p];
    }, g);
    deriv_ri2_contract(M, [&](const int *, const int * o, const int * nn, double * w){
        for(int p = 0; p < nn[0]; p++) for(int q = 0; q < nn[1]; q++) w[p * nn[1] + q] = obj.Vbar[(size_t)(o[0] + p) * naux + o[1] + q];
    }, g);
    std::vector<double> mW(n2);
    for(size_t k = 0; k < n2; k++) mW[k] = -obj.Wao[k];
    deriv_overlap_contract(M, mW.data(), g);
    return 0;
}

void cas_sa_gamma(const CasRef & R, std::vector<double> * gamma_sa){
    *gamma_sa = cas_base(R).g;
}

void cas_H_apply(const CasRef & R, const double * C, const double * vecs, int nvec, double * Hv){
    ActInts A; build_active(R, C, &A);
    ci_kernel(R, A, vecs, nvec, Hv, nullptr, nullptr);
}

void cas_transition_rdm(const CasRef & R, const double * bra, const double * ket, int nvec, double * T){
    omp_set_num_threads(num_threads);
    const int na = R.na; const size_t Nd = R.Nd;
    aldet_data X;
    X.get_dim(R.ci_template->n_act, R.ci_template->na, R.ci_template->nb, 2, R.ci_template->mult, 0);
    X.init_zero_vec(nvec, 0); X.init_zero_vec(nvec, 1);
    for(int s = 0; s < nvec; s++) for(size_t d = 0; d < Nd; d++){ X.coef[0][d * nvec + s] = bra[(size_t)s * Nd + d]; X.coef[1][d * nvec + s] = ket[(size_t)s * Nd + d]; }
    X.transpose_ci(0); X.transpose_ci(1);
    std::vector<double> O((size_t)nvec * nvec * na * na, 0.0);
    X.calc_DM(O.data(), 0, 1);
    std::fill(T, T + (size_t)na * na, 0.0);
    for(int s = 0; s < nvec; s++) for(int k = 0; k < na * na; k++) T[k] += O[((size_t)s * nvec + s) * na * na + k];
}

void cas_ci_rotation_g_fd(const CasRef & R, const double * c, const double * cbar, int nvec, double * g, bool transpose_U){
    omp_set_num_threads(num_threads);
    const int na = R.na; const size_t Nd = R.Nd;
    aldet_data X;
    X.get_dim(R.ci_template->n_act, R.ci_template->na, R.ci_template->nb, 1, R.ci_template->mult, 0);
    X.init_zero_vec(nvec, 0);
    std::fill(g, g + (size_t)na * na, 0.0);
    std::vector<double> U((size_t)na * na);
    auto phi = [&](int t, int u, double th){
        std::fill(U.begin(), U.end(), 0.0);
        for(int i = 0; i < na; i++) U[(size_t)i * na + i] = 1.0;
        const double cs = std::cos(th), sn = std::sin(th);
        U[(size_t)t * na + t] = cs; U[(size_t)u * na + u] = cs;
        U[(size_t)t * na + u] = transpose_U ? -sn : sn; U[(size_t)u * na + t] = transpose_U ? sn : -sn;
        for(int s = 0; s < nvec; s++) for(size_t d = 0; d < Nd; d++) X.coef[0][d * nvec + s] = c[(size_t)s * Nd + d];
        X.malmqvist(0, U.data());
        double v = 0;
        for(int s = 0; s < nvec; s++) for(size_t d = 0; d < Nd; d++) v += cbar[(size_t)s * Nd + d] * X.coef[0][d * nvec + s];
        return v;
    };
    const double h = 2e-3;
    for(int t = 0; t < na; t++) for(int u = t + 1; u < na; u++){
        const double gv = (-phi(t, u, 2 * h) + 8 * phi(t, u, h) - 8 * phi(t, u, -h) + phi(t, u, -2 * h)) / (12 * h);
        g[(size_t)t * na + u] = gv; g[(size_t)u * na + t] = -gv;
    }
}

// ---- W from an unconstrained orbital gradient --------------------------------------------------------------
void cas_w_from_gc(const CasRef & R, const double * C, const std::vector<double> & GC, std::vector<double> * Wao, std::vector<double> * CtG_out){
    const int n = R.n, nmo = R.nmo;
    std::vector<double> Call((size_t)n * nmo), CtG((size_t)nmo * nmo), Wmo((size_t)nmo * nmo), CW((size_t)n * nmo);
    for(int a = 0; a < n; a++) for(int p = 0; p < nmo; p++) Call[(size_t)a * nmo + p] = C[(size_t)p * n + a];
    gemm(nmo, nmo, n, Call.data(), GC.data(), CtG.data(), 1.0, 0.0, true, false);
    for(int p = 0; p < nmo; p++) for(int q = 0; q < nmo; q++) Wmo[(size_t)p * nmo + q] = 0.25 * (CtG[(size_t)p * nmo + q] + CtG[(size_t)q * nmo + p]);
    gemm(n, nmo, nmo, Call.data(), Wmo.data(), CW.data());
    Wao->assign((size_t)n * n, 0.0);
    gemm(n, n, nmo, CW.data(), Call.data(), Wao->data(), 1.0, 0.0, false, true);
    if(CtG_out) *CtG_out = CtG;
}

void cas_onebody_apply(const CasRef & R, const double * M, const double * vecs, int nvec, double * out){
    const int na = R.na;
    ActInts A;
    A.act.assign((size_t)na * na * na * na, 0.0);
    A.HAA.assign((size_t)na * na, 0.0);
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) A.HAA[(size_t)t * na + u] = 0.5 * (M[t * na + u] + M[u * na + t]);
    A.Ecore = 0.0;
    ci_kernel(R, A, vecs, nvec, out, nullptr, nullptr);
}

// One-body part of the response right-hand side: dE/dc_I for an energy that depends on the reference CI vectors only through the
// state-averaged 1-RDM, E(gamma_SA), gamma_SA = sum_I w_I gamma_I:  dE/dc_I = 2 w_I P_perp Ghat c_I  with  Ghat = sum_tu gbar_tu E_tu.
void cas_gamma_rhs(const CasRef & R, const double * gbar, std::vector<double> * b_delta){
    const int na = R.na;
    ActInts A;
    A.act.assign((size_t)na * na * na * na, 0.0);
    A.HAA.assign((size_t)na * na, 0.0);
    for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) A.HAA[(size_t)t * na + u] = 0.5 * (gbar[t * na + u] + gbar[u * na + t]);
    A.Ecore = 0.0;
    std::vector<double> Hc((size_t)R.ns * R.Nd);
    ci_kernel(R, A, R.c0.data(), R.ns, Hc.data(), nullptr, nullptr);
    b_delta->assign((size_t)R.ns * R.Nd + R.nr(), 0.0);
    for(int I = 0; I < R.ns; I++) for(size_t d = 0; d < R.Nd; d++) (*b_delta)[(size_t)I * R.Nd + d] = 2.0 * R.w[I] * Hc[(size_t)I * R.Nd + d];
    project_perp(R, b_delta->data());
    for(size_t k = 0; k < R.nr(); k++){
        const int I = R.rho[k].first, J = R.rho[k].second;
        (*b_delta)[(size_t)R.ns * R.Nd + k] = 2.0 * (R.w[I] - R.w[J]) * dot(&R.c0[(size_t)I * R.Nd], &Hc[(size_t)J * R.Nd], R.Nd);
    }
}

// ---- the Lagrangian gradient -----------------------------------------------------------------------------------------
// direct: D (n x n, contracted with T' and V'), Lbar, Vbar and the UNCONSTRAINED orbital gradient GC of the energy at fixed reference
// (sum of every contribution the caller has: bare densities, PT chains, ...). b_delta: its derivative with respect to the reference CI vectors
// (ns*Nd, in the complement; may be empty). On return g (3 n_atoms) holds V_NN' + the contractions + the response.
int cas_lagrangian_objects(molecule * M, const CasRef & R, CasObjects & direct, const std::vector<double> & b_delta, CasObjects * tot_out){
    const int n = R.n, nmo = R.nmo;
    std::vector<double> CtG;
    cas_w_from_gc(R, R.C0.data(), direct.GC, &direct.Wao, &CtG);
    std::vector<double> b(R.nx(), 0.0);
    for(size_t k = 0; k < R.nk(); k++){ const int p = R.pairs[k].first, q = R.pairs[k].second; b[k] = CtG[(size_t)q * nmo + p] - CtG[(size_t)p * nmo + q]; }
    for(size_t i = 0; i < b_delta.size(); i++) b[R.nk() + i] = b_delta[i];   // CI complement, then (optionally) the rho block
    std::vector<double> mb(b); for(double & v : mb) v = -v;
    const double tz0 = omp_get_wtime();
    std::vector<double> z;
    const int it = cas_solve_z(R, mb, &z, (getenv("NOPT_RESP_TOL") && *getenv("NOPT_RESP_TOL")) ? atof(getenv("NOPT_RESP_TOL")) : 1e-9, 400);
    if(it < 0){ fprintf(out_stream, "ERROR: the Z-vector equations did not converge\n"); return 1; }
    fprintf(out_stream, "CAS-SCF response: Z-vector converged in %d iterations, final |residual| / |rhs| = %.2e\n", it, g_cas_z_residual);
    fprintf(out_stream, "CAS-SCF response: %d Hessian actions: CI densities %.2f s, J/K %.2f s, active 2-e %.2f s, CI H %.2f s, rest %.2f s\n",
            (int)g_jvp_time[5], g_jvp_time[0], g_jvp_time[1], g_jvp_time[2], g_jvp_time[3], g_jvp_time[4]);

    // response objects: the exact derivative of the SA objects along z
    const double tr0 = omp_get_wtime();
    CasObjects resp;
    resp.Lbar = std::move(direct.Lbar); resp.Vbar = std::move(direct.Vbar);   // the response RI adjoints accumulate onto the direct ones
    cas_jvp(R, z.data(), nullptr, &resp, true);
    fprintf(out_stream, "CAS-SCF response: Z-vector %.2f s, response objects %.2f s\n", tr0 - tz0, omp_get_wtime() - tr0);

    // total = direct + response, accumulated in place (direct is consumed: its large L-bar is moved, not copied)
    CasObjects & tot = *tot_out;
    tot.D = std::move(direct.D); tot.Lbar = std::move(resp.Lbar); tot.Vbar = std::move(resp.Vbar); tot.Wao = std::move(direct.Wao);
    for(size_t j = 0; j < tot.D.size(); j++) tot.D[j] += resp.D[j];
    for(size_t j = 0; j < tot.Wao.size(); j++) tot.Wao[j] += resp.Wao[j];
    return 0;
}

int cas_lagrangian_objects_cbar(molecule * M, const CasRef & R, CasObjects & direct, const std::vector<double> & cbar_all, const std::vector<double> & b_extra,
                                CasObjects * tot){
    const size_t Nd = R.Nd; const int ns = R.ns, nall = R.ns_all, na = R.na;
    const size_t a2 = (size_t)na * na, a4 = a2 * a2;
    std::vector<double> bd((size_t)ns * Nd + R.nr(), 0.0);
    for(size_t i = 0; i < b_extra.size() && i < bd.size(); i++) bd[i] += b_extra[i];
    if(!cbar_all.empty()){
        const CasBase & B = cas_base(R);
        auto cb = [&](int root){ return &cbar_all[(size_t)root * Nd]; };
        auto cv = [&](int root){ return &R.c_all[(size_t)root * Nd]; };
        std::vector<int> zero; for(int I = 0; I < nall; I++) if(R.slot(I) < 0) zero.push_back(I);
        // weighted states: cbar_P, with the counter-rotation of every zero-weight root (c_Z -= r c_P when c_P += r c_Z), on the complement of the weighted span
        std::vector<double> bw((size_t)ns * Nd + R.nr(), 0.0);
        for(int P = 0; P < ns; P++){
            double * b = &bw[(size_t)P * Nd];
            for(size_t d = 0; d < Nd; d++) b[d] = cb(R.states[P])[d];
            if(!(getenv("NOPT_RESP_DEBUG") && strstr(getenv("NOPT_RESP_DEBUG"), "nocounter")))
            for(int Z : zero){ const double o = dot(cb(Z), cv(R.states[P]), Nd); for(size_t d = 0; d < Nd; d++) b[d] -= o * cv(Z)[d]; }
        }
        project_perp(R, bw.data());
        if(getenv("NOPT_RESP_DEBUG") && *getenv("NOPT_RESP_DEBUG"))
            for(int P = 0; P < ns; P++) for(int Z : zero)
                fprintf(out_stream, "RESP DEBUG: <cbar_P|c_Z> = % .6e, <cbar_Z|c_P> = % .6e, antisymmetric part % .3e (P = %d, Z = %d)\n",
                        dot(cb(R.states[P]), cv(Z), Nd), dot(cb(Z), cv(R.states[P]), Nd), dot(cb(R.states[P]), cv(Z), Nd) - dot(cb(Z), cv(R.states[P]), Nd), R.states[P], Z);
        for(size_t k = 0; k < R.nr(); k++){
            const int I = R.states[R.rho[k].first], J = R.states[R.rho[k].second];
            bw[(size_t)ns * Nd + k] = dot(cb(I), cv(J), Nd) - dot(cb(J), cv(I), Nd);
        }
        for(size_t i = 0; i < bd.size(); i++) bd[i] += bw[i];
        // invariance under rotations among weighted roots of equal weight (redundant SA variables): the target must not depend on them
        {
            double asym = 0.0, scale = 0.0;
            for(int I = 0; I < ns; I++){ scale = std::max(scale, std::sqrt(dot(cb(R.states[I]), cb(R.states[I]), Nd)));
                for(int J = I + 1; J < ns; J++) if(std::fabs(R.w[I] - R.w[J]) <= 1e-12)
                    asym = std::max(asym, std::fabs(dot(cb(R.states[I]), cv(R.states[J]), Nd) - dot(cb(R.states[J]), cv(R.states[I]), Nd))); }
            if(getenv("NOPT_RESP_DEBUG") && *getenv("NOPT_RESP_DEBUG"))
                fprintf(out_stream, "RESP DEBUG: equal-weight antisymmetric part max %.3e, |cbar| max %.3e\n", asym, scale);
            if(asym > 1e-10 * std::max(1.0, scale)){
                // A target that depends on the rotation between weighted roots I, J of EQUAL weight (e.g. a state-specific DSRG on one root of an
                // equal-weight ensemble): that rotation is redundant for E_SA and fixed instead by <c_I|H|c_J> = 0 (the roots are eigenvectors).
                // Multiplier lam_IJ (E_J - E_I) = -a_IJ, a_IJ = <cbar_I|c_J> - <cbar_J|c_I>; its terms: the fixed-reference energy lam <c_I|H|c_J>
                // (objects of the symmetrized transition densities) and the CI source lam P_perp H c_J on delta_I, lam P_perp H c_I on delta_J.
                std::vector<double> Hc((size_t)ns * Nd), E(ns);
                ci_kernel(R, B.A, R.c0.data(), ns, Hc.data(), nullptr, nullptr);
                for(int I = 0; I < ns; I++) E[I] = dot(&R.c0[(size_t)I * Nd], &Hc[(size_t)I * Nd], Nd);
                std::vector<double> vp, vm, bl((size_t)ns * Nd, 0.0);
                int npair = 0;
                for(int I = 0; I < ns; I++) for(int J = I + 1; J < ns; J++){
                    if(std::fabs(R.w[I] - R.w[J]) > 1e-12) continue;
                    const double a = dot(cb(R.states[I]), cv(R.states[J]), Nd) - dot(cb(R.states[J]), cv(R.states[I]), Nd), gap = E[J] - E[I];
                    if(std::fabs(a) <= 1e-14 * std::max(1.0, scale)) continue;
                    if(std::fabs(gap) < 1e-8){ fprintf(out_stream, "ERROR: equally weighted roots %d and %d are degenerate (gap %.2e) and the target depends on their rotation\n",
                                                       R.states[I], R.states[J], gap); return 1; }
                    const double lam = -a / gap;
                    for(size_t d = 0; d < Nd; d++){ bl[(size_t)I * Nd + d] += lam * Hc[(size_t)J * Nd + d]; bl[(size_t)J * Nd + d] += lam * Hc[(size_t)I * Nd + d]; }
                    for(size_t d = 0; d < Nd; d++){ vp.push_back(R.c0[(size_t)I * Nd + d] + lam * R.c0[(size_t)J * Nd + d]); vm.push_back(R.c0[(size_t)I * Nd + d] - lam * R.c0[(size_t)J * Nd + d]); }
                    npair++;
                    fprintf(out_stream, "CAS-SCF response: equal-weight roots %d, %d: the target depends on their rotation (%.3e); multiplier %.3e (gap %.3e)\n",
                            R.states[I], R.states[J], a, lam, gap);
                }
                project_perp(R, bl.data());
                for(size_t i = 0; i < (size_t)ns * Nd; i++) bd[i] += bl[i];
                if(npair){
                    std::vector<double> gp((size_t)npair * a2), gm((size_t)npair * a2), Gp((size_t)npair * a4), Gm((size_t)npair * a4);
                    ci_kernel(R, B.A, vp.data(), npair, nullptr, gp.data(), Gp.data());
                    ci_kernel(R, B.A, vm.data(), npair, nullptr, gm.data(), Gm.data());
                    std::vector<double> g(a2, 0.0), G(a4, 0.0), g0(a2, 0.0), G0(a4, 0.0);
                    for(int k = 0; k < npair; k++){
                        for(size_t q = 0; q < a2; q++) g[q] += 0.25 * (gp[(size_t)k * a2 + q] - gm[(size_t)k * a2 + q]);
                        for(size_t q = 0; q < a4; q++) G[q] += 0.25 * (Gp[(size_t)k * a4 + q] - Gm[(size_t)k * a4 + q]);
                    }
                    add_objects_difference(R, g.data(), G.data(), g0.data(), G0.data(), &B.La, direct);
                }
            }
        }
        // zero-weight roots: multipliers of their eigen-equations
        if(!zero.empty() && !(getenv("NOPT_RESP_DEBUG") && strstr(getenv("NOPT_RESP_DEBUG"), "nospect"))){
            const int nz = (int)zero.size();
            std::vector<double> Hcz((size_t)nz * Nd), cz((size_t)nz * Nd), Ez(nz), Hd(Nd);
            for(int iz = 0; iz < nz; iz++) std::copy(cv(zero[iz]), cv(zero[iz]) + Nd, cz.begin() + (size_t)iz * Nd);
            ci_kernel(R, B.A, cz.data(), nz, Hcz.data(), nullptr, nullptr, Hd.data());
            for(int iz = 0; iz < nz; iz++){
                Ez[iz] = dot(&cz[(size_t)iz * Nd], &Hcz[(size_t)iz * Nd], Nd);
                double rr = 0; for(size_t d = 0; d < Nd; d++){ const double e = Hcz[(size_t)iz * Nd + d] - Ez[iz] * cz[(size_t)iz * Nd + d]; rr += e * e; }
                fprintf(out_stream, "CAS-SCF response: zero-weight root %d is an eigenvector to |(H - E) c| = %.2e\n", zero[iz], std::sqrt(rr));
                if(std::sqrt(rr) > 1e-6) fprintf(out_stream, "WARNING: zero-weight root %d is not converged as an eigenvector (|(H - E) c| = %.2e): the gradient assumes "
                                                             "it is; tighten the CI convergence\n", zero[iz], std::sqrt(rr));
            }
            auto perp_all = [&](double * v){ for(int J = 0; J < nall; J++){ const double o = dot(cv(J), v, Nd); for(size_t d = 0; d < Nd; d++) v[d] -= o * cv(J)[d]; } };
            std::vector<double> T((size_t)nz * Nd, 0.0);
            for(int iz = 0; iz < nz; iz++){
                std::vector<double> b2(cb(zero[iz]), cb(zero[iz]) + Nd);
                perp_all(b2.data());
                const double bn = std::sqrt(dot(b2.data(), b2.data(), Nd));
                if(bn == 0.0) continue;
                for(double & x : b2) x = -x;
                const double E = Ez[iz];
                auto apply = [&](const std::vector<double> & v, std::vector<double> * Av){
                    Av->resize(Nd);
                    std::vector<double> pv(v); perp_all(pv.data());
                    ci_kernel(R, B.A, pv.data(), 1, Av->data(), nullptr, nullptr);
                    for(size_t d = 0; d < Nd; d++) (*Av)[d] -= E * pv[d];
                    perp_all(Av->data());
                };
                auto prec = [&](const std::vector<double> & r, std::vector<double> * y){
                    y->resize(Nd); for(size_t d = 0; d < Nd; d++) (*y)[d] = r[d] / std::max(std::fabs(Hd[d] - E), 1e-2); perp_all(y->data()); };
                std::vector<double> z;
                const int it = minres(Nd, apply, prec, b2, &z, 1e-11, 500, "zero-weight root", false);
                std::vector<double> Az; apply(z, &Az);
                double rr = 0; for(size_t d = 0; d < Nd; d++){ const double e = b2[d] - Az[d]; rr += e * e; }
                const double rel = std::sqrt(rr) / bn;
                fprintf(out_stream, "CAS-SCF response: zero-weight root %d (E = % .10f): multiplier converged in %d iterations, |residual| / |rhs| = %.2e\n", zero[iz], E, it, rel);
                if(rel > 1e-6){ fprintf(out_stream, "ERROR: the multiplier of zero-weight root %d did not converge\n", zero[iz]); return 1; }
                std::copy(z.begin(), z.end(), T.begin() + (size_t)iz * Nd);
            }
            // rotations among zero-weight roots: m (E_Z' - E_Z) = -(<cbar_Z|c_Z'> - <cbar_Z'|c_Z>); the term m <c_Z|H|c_Z'> joins t_Z as m c_Z'
            for(int iz = 0; iz < nz; iz++) for(int jz = iz + 1; jz < nz; jz++){
                const double a = dot(cb(zero[iz]), cv(zero[jz]), Nd) - dot(cb(zero[jz]), cv(zero[iz]), Nd), gap = Ez[jz] - Ez[iz];
                if(a == 0.0) continue;
                if(std::fabs(gap) < 1e-8){ fprintf(out_stream, "ERROR: zero-weight roots %d and %d are degenerate (gap %.2e) and the target depends on their rotation\n", zero[iz], zero[jz], gap); return 1; }
                const double m = -a / gap;
                for(size_t d = 0; d < Nd; d++) T[(size_t)iz * Nd + d] += m * cv(zero[jz])[d];
            }
            // Phi = sum_Z <t_Z|H|c_Z>: densities by polarization ((rho(c+t) - rho(c-t)) / 4), objects of the linear part (obj(rho) - obj(0))
            std::vector<double> vp(cz), vm(cz);
            for(size_t i = 0; i < vp.size(); i++){ vp[i] += T[i]; vm[i] -= T[i]; }
            std::vector<double> gp((size_t)nz * a2), gm((size_t)nz * a2), Gp((size_t)nz * a4), Gm((size_t)nz * a4);
            ci_kernel(R, B.A, vp.data(), nz, nullptr, gp.data(), Gp.data());
            ci_kernel(R, B.A, vm.data(), nz, nullptr, gm.data(), Gm.data());
            std::vector<double> g(a2, 0.0), G(a4, 0.0), g0(a2, 0.0), G0(a4, 0.0);
            for(int iz = 0; iz < nz; iz++){
                for(size_t k = 0; k < a2; k++) g[k] += 0.25 * (gp[(size_t)iz * a2 + k] - gm[(size_t)iz * a2 + k]);
                for(size_t k = 0; k < a4; k++) G[k] += 0.25 * (Gp[(size_t)iz * a4 + k] - Gm[(size_t)iz * a4 + k]);
            }
            double ed = 0;
            add_objects_difference(R, g.data(), G.data(), g0.data(), G0.data(), &B.La, direct, &ed);
            fprintf(out_stream, "CAS-SCF response: multiplier term of the %d zero-weight root(s): % .3e\n", nz, ed);
        }
    }
    return cas_lagrangian_objects(M, R, direct, bd, tot);
}

int cas_lagrangian_gradient(molecule * M, const CasRef & R, CasObjects & direct, const std::vector<double> & b_delta, double * g){
    CasObjects tot;
    if(cas_lagrangian_objects(M, R, direct, b_delta, &tot)) return 1;
    std::vector<double> grad(3 * M->n_atoms, 0.0);
    deriv_nuclear_repulsion(M, grad.data());
    cas_geo_contract(M, R, tot, grad.data());
    for(int i = 0; i < 3 * M->n_atoms; i++) g[i] = grad[i];
    return 0;
}

// ---- root gradient --------------------------------------------------------------------------------------------------
int cas_reference_check(const CasRef & R, const char * who){
    std::vector<double> r0; double Esa = 0;
    cas_residual(R, R.C0.data(), R.c0.data(), &r0, nullptr, &Esa);
    double rk = 0, rc = 0, eig = 0;
    for(size_t k = 0; k < R.nk(); k++) rk = std::max(rk, std::fabs(r0[k]));
    for(size_t k = R.nk(); k < R.nx(); k++) rc = std::max(rc, std::fabs(r0[k]));
    std::vector<double> Hc(R.Nd);
    for(int I = 0; I < R.ns_all; I++){
        const std::vector<double> c = R.root_vector(I);
        cas_H_apply(R, R.C0.data(), c.data(), 1, Hc.data());
        const double E = dot(c.data(), Hc.data(), R.Nd);
        for(size_t d = 0; d < R.Nd; d++) eig = std::max(eig, std::fabs(Hc[d] - E * c[d]));
    }
    fprintf(out_stream, "%s: CAS-SCF reference: max |r_kappa| = %.2e, max |r_delta| = %.2e, max |(H - E) c| over the roots = %.2e\n", who, rk, rc, eig);
    if(rk > 1e-6 || rc > 1e-6 || eig > 1e-6){
        fprintf(out_stream, "ERROR: %s: the CAS-SCF is not converged tightly enough for a gradient (the error is linear in these residuals); "
                            "tighten $CAS grad/step and the CI ($dav de=1e-14 dr=1e-16 $end)\n", who);
        return 1;
    }
    return 0;
}

int casscf_root_gradient(molecule * M, int root, double * g, double * E_out){
    if(deriv_check_supported(M)) return 1;
    if(RI == 0){ fprintf(out_stream, "ERROR: the CAS-SCF gradient is implemented for the RI energy only\n"); return 1; }
    CasRef R;
    if(R.build(M)) return 1;
    if(root < 0 || root >= R.ns_all){ fprintf(out_stream, "ERROR: root %d is not among the %d states of the CAS-SCF\n", root, R.ns_all); return 1; }
    if(cas_reference_check(R, "CAS-SCF root gradient")) return 1;
    const std::vector<double> cn = R.root_vector(root);
    const double one = 1.0;

    CasObjects obj_n;
    cas_objects(R, R.C0.data(), cn.data(), 1, &one, true, &obj_n);
    std::vector<double> r0; double E_sa = 0;
    cas_residual(R, R.C0.data(), R.c0.data(), &r0, nullptr, &E_sa);
    double rmax = 0; for(double v : r0) rmax = std::max(rmax, std::fabs(v));
    fprintf(out_stream, "CAS-SCF response: %d orbital rotations + %d weighted CI vectors of dimension %zu (%d roots); E_SA(rebuilt) = % .12f, E_root(%d) = % .12f; max |residual| = %.2e\n",
            (int)R.nk(), R.ns, R.Nd, R.ns_all, E_sa, root, obj_n.E, rmax);

    // dE_n/dc_I: only the root's own vector, if it is a weighted state: 2 P_perp (H - E_n) c_n (zero for an eigenvector). A zero-weight root is an
    // eigenvector of the CAS Hamiltonian outside the SA variables: Hellmann-Feynman, no CI term.
    std::vector<double> bd((size_t)R.ns * R.Nd, 0.0);
    const int sl = R.slot(root);
    if(sl >= 0){
        const CasBase & B = cas_base(R);
        const std::vector<double> & Hc = B.Hc0;
        const double E = B.Ei[sl];
        for(size_t d = 0; d < R.Nd; d++) bd[(size_t)sl * R.Nd + d] = 2.0 * (Hc[(size_t)sl * R.Nd + d] - E * R.c0[(size_t)sl * R.Nd + d]);
        project_perp(R, bd.data());
    }
    if(cas_lagrangian_gradient(M, R, obj_n, bd, g)) return 1;
    if(E_out) *E_out = obj_n.E;
    return 0;
}
