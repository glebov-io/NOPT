// Spin-free DSRG-PT2 driver: state-specific or SA ensemble reference, optional uncontracted
// relaxation. Structurally modelled on CDAS_PT2 (CAS_engine-based, seam-driven): bare-solve on
// the converged CASSCF orbitals, read spin-summed densities through the casci_solver seam,
// semicanonicalize the generalized Fock, re-form the RI B-tensors and bare 1-e integrals in that
// basis, then drive the dsrg_sf_tensors core plus the batched CCVV/CAVV/CCAV terms.

# include <cstdio>
# include <cstdlib>
# include <cmath>
# include <vector>
# include <algorithm>

# include "blas_link.h"
# include "molecule.h"
# include "matr.h"
# include "timer.h"
# include "defaults.h"
# include "common_vars.h"
# include "inp_par_read.h"
# include "inp_out.h"
# include "RI.h"
# include "CAS.h"
# include "casci_solver.h"
# include "tensor_rotate.h"    // rotate1 / rotate2 / rotate3 (active-basis density rotation)
# include "dsrg_sf_tensors.h"
# include "dsrg_sf_batch.h"
# include "dsrg_pt.h"
# include "dmrg_log.h"         // per-solve block2 sweep log
# include "grabbers.h"

// A relaxed dressed root is assigned to its argmax bare root only above this |CI overlap|.
static const double DSRG_RELAX_OVERLAP_MIN = 0.9;

// cross = Ul^T (m x m) . X (m x n) . Ur (n x n) -> out (m x n). tmp is m*n scratch.
static void rot_cross(const double * Ul, const double * X, const double * Ur,
                      int m, int n, double * out, double * tmp){
    if(m==0 || n==0) return;
    nopt_par_dgemm(CblasRowMajor,CblasTrans  ,CblasNoTrans, m,n,m, 1.0, Ul,m, X ,n, 0.0, tmp,n);
    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, m,n,n, 1.0, tmp,n, Ur,n, 0.0, out,n);
}

// Max off-diagonal magnitude of a dim x dim block after rotation to the semicanonical
// basis (Usub^T Xblock Usub); the diagonal is compared against eps separately by the caller.
static double block_offdiag_max(const double * Xblock, const double * Usub, int dim,
                                std::vector<double> & rot){
    if(dim==0) return 0.0;
    rot.assign((size_t)dim*dim, 0.0);
    rotate1(Xblock, Usub, dim, rot.data(), /*forward=*/true);   // Usub^T Xblock Usub
    double mx=0.0;
    for(int i=0;i<dim;i++)
    for(int j=0;j<dim;j++)
        if(i!=j) mx = std::max(mx, std::fabs(rot[(size_t)i*dim+j]));
    return mx;
}

int SA_DSRG_PT2(molecule * M, dsrg_par * dsrg, char * job_name){
    // CASSCF has published its n_s energies; DSRG replaces them with its own result, or with none.
    E_grabbed.clear();
    dmrg_log_set_tag(dmrg_log_tag::dsrg);

    if(RI==0){
        fprintf(out_stream,"ERROR: DSRG-PT2 requires the RI (density-fitting) path (set RI=1)\n");
        exit(EXIT_FAILURE);
    }
    gen_RI_AA(M);

    dsrg->write_info(M->n_act_el_alp[0], M->n_act_el_bet[0],
                     M->n_act_orb[0],    M->CI[0].mult);

    // ---- CAS_engine on the converged orbitals; bare SA solve ----
    CAS_engine CAS;
    CAS.init(dsrg->cas, M);
    CAS.SCF_alloc();
    CAS.calc_DM_C();
    CAS.CI_calc(1,0,0);

    const int n_ao  = CAS.n_ao;
    const int n_cor = CAS.n_core;
    const int n_act = CAS.n_act;
    const int n_vir = CAS.n_vac;
    const int ns    = CAS.CI->n_states();
    const int root  = dsrg->root;
    const size_t na2 = (size_t)n_act*n_act;
    const size_t na4 = na2*na2;
    const size_t na6 = na4*na2;

    // ---- guards (our-contract; loud, naming the supported alternative) ----
    if(root<0 || root>=ns){
        fprintf(out_stream,"ERROR: $DSRG root=%d out of range; the CAS solve has n_s=%d states (root in [0,%d])\n",
                root, ns, ns-1);
        exit(EXIT_FAILURE);
    }
    if(!CAS.CI->supports_g2_diag()){
        fprintf(out_stream,"ERROR: DSRG-PT2 needs the per-state spin-summed 2-RDM via G2_calc_diag, which this "
                           "CI backend does not provide; use cisolver=aldet\n");
        exit(EXIT_FAILURE);
    }
    if(!CAS.CI->supports_g3_diag()){
        fprintf(out_stream,"ERROR: DSRG-PT2 needs the per-state spin-summed 3-body moment via G3_calc_diag for the "
                           "lambda3 (E3) term, which this CI backend does not provide; use cisolver=aldet\n");
        exit(EXIT_FAILURE);
    }

    const bool ccvv_zero = (dsrg->ccvv_source==DSRG_CCVV_SRC_ZERO);
    const bool sa = (dsrg->sa==1);
    const double s = dsrg->s;

    // ---- energies first (block2's E_state cache is clobbered by the first RDM read; a
    //      dressed re-solve later overwrites E_state, so all per-root energies snapshot here) ----
    const double E_cas_root = CAS.CI->E_state(root);
    std::vector<double> E_bare(ns);
    for(int r=0;r<ns;r++) E_bare[r] = CAS.CI->E_state(r);

    // ---- selected root's spin-summed densities in the converged (original) active basis ----
    // 1-RDM blocks via calc_DMA/DMB (ns x ns), 2-RDMs via G2_calc_diag (ns consecutive
    // per-state blocks); either backend reports the native basis.
    std::vector<double> D1((size_t)ns*ns*na2, 0.0), G2diag((size_t)ns*na4, 0.0);
    CAS.CI->calc_DMA(D1.data(),0,0);
    if(CAS.ci_solver==CISOLVER_ALDET) CAS.CI->calc_DMB(D1.data(),0,0);   // sum alpha+beta (DMRG: DMA is spin-summed)
    CAS.CI->G2_calc_diag(G2diag.data());

    std::vector<double> L1_orig(na2,0.0), GAMMA_orig(na4,0.0);
    if(!sa){
        const size_t od1 = ((size_t)root*ns+root)*na2;
        const size_t og2 = (size_t)root*na4;
        for(size_t i=0;i<na2;i++) L1_orig[i]    = D1[od1+i];
        for(size_t i=0;i<na4;i++) GAMMA_orig[i] = G2diag[og2+i];
    }
    else{
        // SA ensemble: weighted sum of the per-root (s,s) diagonal blocks, normalized by the
        // reference-weight sum (average_DM convention). Seeds calc_F -> SA generalized Fock.
        double wsum=0.0; for(int t=0;t<ns;t++) wsum += CAS.wstate_actual[t];
        if(wsum<=0.0){ fprintf(out_stream,"ERROR: DSRG-PT2 sa=1 has zero total reference weight\n"); exit(EXIT_FAILURE); }
        for(int t=0;t<ns;t++){
            const double w = CAS.wstate_actual[t]/wsum;
            const size_t od1=((size_t)t*ns+t)*na2, og2=(size_t)t*na4;
            for(size_t i=0;i<na2;i++) L1_orig[i]    += w*D1[od1+i];
            for(size_t i=0;i<na4;i++) GAMMA_orig[i] += w*G2diag[og2+i];
        }
    }

    // ---- state-specific generalized Fock from the selected root's 1-RDM (calc_F pattern) ----
    // Seed CAS.gamma with the selected root's 1-RDM and reuse the CAS_engine Fock build; the
    // separate L1_orig copy is preserved for the density rotation below.
    for(size_t i=0;i<na2;i++) CAS.gamma[i] = L1_orig[i];
    CAS.calc_F(CAS.F_tot, CAS.gamma);
    double * F = CAS.F_tot;

    // Extract the original cross + diagonal blocks BEFORE diagonalization (diag_X_MO_block
    // rewrites only the diagonal blocks, leaving the cross blocks stale in the old basis).
    std::vector<double> FCC0((size_t)n_cor*n_cor), FAA0(na2), FVV0((size_t)n_vir*n_vir);
    std::vector<double> FCA0((size_t)n_cor*n_act), FCV0((size_t)n_cor*n_vir), FAV0((size_t)n_act*n_vir);
    for(int i=0;i<n_cor;i++) for(int j=0;j<n_cor;j++) FCC0[(size_t)i*n_cor+j]=F[(size_t)i*n_ao+j];
    for(int t=0;t<n_act;t++) for(int u=0;u<n_act;u++) FAA0[(size_t)t*n_act+u]=F[(size_t)(n_cor+t)*n_ao+(n_cor+u)];
    for(int a=0;a<n_vir;a++) for(int b=0;b<n_vir;b++) FVV0[(size_t)a*n_vir+b]=F[(size_t)(n_cor+n_act+a)*n_ao+(n_cor+n_act+b)];
    for(int i=0;i<n_cor;i++) for(int t=0;t<n_act;t++) FCA0[(size_t)i*n_act+t]=F[(size_t)i*n_ao+(n_cor+t)];
    for(int i=0;i<n_cor;i++) for(int a=0;a<n_vir;a++) FCV0[(size_t)i*n_vir+a]=F[(size_t)i*n_ao+(n_cor+n_act+a)];
    for(int t=0;t<n_act;t++) for(int a=0;a<n_vir;a++) FAV0[(size_t)t*n_vir+a]=F[(size_t)(n_cor+t)*n_ao+(n_cor+n_act+a)];

    // Block-diagonalize C / A / V (normalize_rotation_rows gauge); capture all three
    // rotations. diag_X_MO_block rotates M->MO_VEC (== CAS.MO_VEC) rows and writes eps into
    // M->orb_energy; the returned U is transposed for use as U^T (.) U.
    std::vector<double> Uc((size_t)n_cor*n_cor,0.0), Ua(na2,0.0), Uv((size_t)n_vir*n_vir,0.0);
    M->diag_X_MO_block(F, 0,             n_cor, Uc.data());
    M->diag_X_MO_block(F, n_cor,         n_act, Ua.data());
    M->diag_X_MO_block(F, n_cor+n_act,   n_vir, Uv.data());

    double * eps_c = M->orb_energy;
    double * eps_a = M->orb_energy + n_cor;
    double * eps_v = M->orb_energy + n_cor + n_act;

    // Verify the three diagonal blocks are diagonal in the rotated basis (our-contract: the
    // semicanonicalization actually took). Print the max off-diagonal; abort above a loose bound.
    std::vector<double> rot;
    double off = 0.0;
    off = std::max(off, block_offdiag_max(FCC0.data(), Uc.data(), n_cor, rot));
    off = std::max(off, block_offdiag_max(FAA0.data(), Ua.data(), n_act, rot));
    off = std::max(off, block_offdiag_max(FVV0.data(), Uv.data(), n_vir, rot));
    fprintf(out_stream,"\nDSRG-PT2 semicanonicalization: max |off-diagonal| Fock (C/A/V) = %.3e\n", off);
    if(off > 1e-8){
        fprintf(out_stream,"ERROR: semicanonical Fock block-diagonalization failed (off-diagonal %.3e > 1e-8)\n", off);
        exit(EXIT_FAILURE);
    }

    // Cross Fock blocks in the semicanonical basis: f_ca=Uc^T FCA0 Ua, f_cv=Uc^T FCV0 Uv,
    // f_av=Ua^T FAV0 Uv. Layouts [i*n_a+t], [i*n_v+a], [t*n_v+a] (the set_problem contract).
    std::vector<double> f_ca((size_t)n_cor*n_act,0.0), f_cv((size_t)n_cor*n_vir,0.0), f_av((size_t)n_act*n_vir,0.0);
    {
        const size_t tmpsz = std::max({(size_t)n_cor*n_act, (size_t)n_cor*n_vir, (size_t)n_act*n_vir});
        std::vector<double> tmp(tmpsz+1);
        rot_cross(Uc.data(), FCA0.data(), Ua.data(), n_cor, n_act, f_ca.data(), tmp.data());
        rot_cross(Uc.data(), FCV0.data(), Uv.data(), n_cor, n_vir, f_cv.data(), tmp.data());
        rot_cross(Ua.data(), FAV0.data(), Uv.data(), n_act, n_vir, f_av.data(), tmp.data());
    }

    // ---- densities into the semicanonical active basis: L1'=Ua^T L1 Ua, GAMMA 4-leg rotate ----
    std::vector<double> L1_semi(na2,0.0), GAMMA_semi(na4,0.0), Eta1_semi(na2,0.0);
    rotate1(L1_orig.data(),    Ua.data(), n_act, L1_semi.data(),    /*forward=*/true);
    rotate2(GAMMA_orig.data(), Ua.data(), n_act, GAMMA_semi.data(), /*forward=*/true);
    for(int u=0;u<n_act;u++) for(int v=0;v<n_act;v++)
        Eta1_semi[(size_t)u*n_act+v] = (u==v?2.0:0.0) - L1_semi[(size_t)u*n_act+v];

    // ---- RI B-tensors re-formed in the semicanonical MO basis ----
    RI_core_realloc(n_cor+n_act, n_ao);
    RI_data R;
    R.set_par(M, n_cor, n_act, n_vir);
    R.MO_calc(M->MO_VEC, n_ao);
    const long naux = R.aux_n_ao;
    if(RI) printf_timer("DSRG-PT2 RI orbital transformation");

    // ---- bare 1-e h in the semicanonical basis (kinetic + nuclear attraction = M->H_AO;
    //      NOT the core-Fock-dressed F_core_MO). e_scalar = nuclear repulsion only. ----
    std::vector<double> COR_VEC((size_t)n_ao*n_cor), ACT_VEC((size_t)n_ao*n_act);
    for(int a=0;a<n_ao;a++) for(int i=0;i<n_cor;i++) COR_VEC[(size_t)a*n_cor+i]=M->MO_VEC[(size_t)i*n_ao+a];
    for(int a=0;a<n_ao;a++) for(int t=0;t<n_act;t++) ACT_VEC[(size_t)a*n_act+t]=M->MO_VEC[(size_t)(n_cor+t)*n_ao+a];

    std::vector<double> h_active(na2,0.0), h_core_full((size_t)n_cor*n_cor,0.0), h_core_diag(n_cor,0.0);
    transform_from_col_MO(h_active.data(),    M->H_AO, n_ao, ACT_VEC.data(), n_act, ACT_VEC.data(), n_act);
    if(n_cor>0){
        transform_from_col_MO(h_core_full.data(), M->H_AO, n_ao, COR_VEC.data(), n_cor, COR_VEC.data(), n_cor);
        for(int i=0;i<n_cor;i++) h_core_diag[i]=h_core_full[(size_t)i*n_cor+i];
    }
    const double e_scalar = M->V_nuc;

    // ---- drive the algebra core ----
    dsrg_sf_tensors T;
    T.set_problem(n_cor, n_act, n_vir, s, ccvv_zero,
                  eps_c, eps_a, eps_v, f_ca.data(), f_cv.data(), f_av.data(), &R);
    T.set_densities(L1_semi.data(), GAMMA_semi.data());
    T.build_vt();
    T.build_amplitudes();
    T.compute_e2();

    // ---- lambda3: per-state 3-body moments in the native active basis (the CI vector is never
    //      rotated, all basis motion is tensor-side), weight-averaged before the cumulant for
    //      sa=1 (cumulant of the average). Two n_act^6 buffers here plus two inside compute_e3
    //      -- 512 MB each at n_act=20, 2 MB at the examples' n_act=8. ----
    std::vector<double> G3_avg(na6,0.0), G3_semi(na6,0.0);
    if(!sa)
        CAS.CI->G3_calc_diag(G3_avg.data(), root);
    else{
        // G3_semi is the per-root read buffer here; the rotation below overwrites it.
        double wsum=0.0; for(int t=0;t<ns;t++) wsum += CAS.wstate_actual[t];
        for(int t=0;t<ns;t++){
            const double w = CAS.wstate_actual[t]/wsum;
            CAS.CI->G3_calc_diag(G3_semi.data(), t);
#pragma omp parallel for schedule(static)
            for(long i=0;i<(long)na6;i++) G3_avg[i] += w*G3_semi[i];
        }
    }
    rotate3(G3_avg.data(), Ua.data(), n_act, G3_semi.data(), /*forward=*/true);
    T.compute_e3(G3_semi.data());   // G3_semi becomes SF_L3 and is read again by pilot_dump

#if 0  // DIRECT lambda3 path (superseded by the explicit lattice-3RDM route; revive for nact >~ 30)
    // compute_e3 evaluates the DIRECT lambda3 via CI->h2caa_overlap with semicanonical
    // active-leg T tensors, so the CI vector must sit in the same basis. aldet rotates it
    // (malmqvist); a backend that cannot (DMRG) passes only if the rotation is ~identity --
    // mixed bases corrupt the overlap even when the 3-body RDM vanishes (2-body survivals).
    double u_dev = 0.0;
    for(int i=0;i<n_act;i++) for(int j=0;j<n_act;j++)
        u_dev = std::max(u_dev, std::fabs(Ua[(size_t)i*n_act+j] - (i==j?1.0:0.0)));
    if(CAS.CI->supports_civec_rotation()){
        CAS.CI->malmqvist(0, Ua.data());
    }
    else if(u_dev>1e-8){
        fprintf(out_stream,"ERROR: the DSRG lambda3 (E3) term needs the CI vector in the semicanonical active "
                           "basis, but this CI backend cannot rotate it (active rotation deviates from identity "
                           "by %.3e); use cisolver=aldet\n", u_dev);
        exit(EXIT_FAILURE);
    }
    T.compute_e3(CAS.CI, root);

    // SA ensemble lambda3: compute_e3's <=2-body completions already carry the ensemble
    // density; only the per-root solver moment om_v/om_c is state-specific, so swap om[root]
    // for the weight-averaged moment (cumulant of the average, not average of cumulants).
    if(sa){
        double wsum=0.0, mv=0.0, mc=0.0;
        for(int r=0;r<ns;r++){ const double w=CAS.wstate_actual[r]; wsum+=w; mv+=w*T.omega_v()[r]; mc+=w*T.omega_c()[r]; }
        T.ledger.E3v += mv/wsum - T.omega_v()[root];
        T.ledger.E3c += T.omega_c()[root] - mc/wsum;
    }
#endif

    const double Eref = T.compute_eref(h_core_diag.data(), h_active.data(), e_scalar);

    // ---- batched, never-materialized CCVV/CAVV/CCAV over the RI B-blocks ----
    const double E_CCVV = dsrg_sf_E_CCVV(R.VC_RI_M, n_cor, n_vir, naux, eps_c, eps_v, s, ccvv_zero);
    const double E_CAVV = dsrg_sf_E_CAVV(R.VC_RI_M, R.VA_RI_M, n_cor, n_act, n_vir, naux,
                                         eps_c, eps_a, eps_v, L1_semi.data(), s);
    const double E_CCAV = dsrg_sf_E_CCAV(R.VC_RI_M, R.CA_RI_M, n_cor, n_act, n_vir, naux,
                                         eps_c, eps_a, eps_v, Eta1_semi.data(), s);
    T.ledger.ccvv = E_CCVV;
    T.ledger.cavv = E_CAVV;
    T.ledger.ccav = E_CCAV;

    // ---- env-gated pilot dump (unrelaxed tail; relax=once dumps after the re-solve so the
    //      dressed operator and per-root data reach the §tail) ----
    if(dsrg->relax==DSRG_RELAX_NONE && getenv("DSRG_PILOT_DUMP"))
        T.pilot_dump(getenv("DSRG_PILOT_DUMP"), M->n_act_el_alp[0], M->n_act_el_bet[0], ns, root);

    // ---- Forte six-line breakdown ----
    const dsrg_pt2_ledger & L = T.ledger;
    const double vt2_L1 = L.VT2_L1_incore() + E_CCVV + E_CAVV + E_CCAV;   // in-core + the three DF terms
    const double vt2_L2 = L.VT2_L2();
    const double vt2_L3 = L.E3v + L.E3c;
    const double E_corr = L.E2_incore() + E_CCVV + E_CAVV + E_CCAV;

    fprintf(out_stream,"\n\n");
    if(!sa)
        fprintf(out_stream,"===================== DSRG-PT2 energy summary (root %d) =====================\n\n", root);
    else
        fprintf(out_stream,"=================== DSRG-PT2 energy summary (SA ensemble) ===================\n\n");
    fprintf(out_stream,"  Flow parameter s                   = % .6f%s\n", s, ccvv_zero?"   (ccvv_source = zero)":"");
    fprintf(out_stream,"  Reference energy E0                = % .12f\n", Eref);
    if(!sa)
        fprintf(out_stream,"  CAS energy (root %d)                = % .12f\n", root, E_cas_root);
    else{
        double wsum=0.0, ec=0.0;
        for(int r=0;r<ns;r++){ wsum+=CAS.wstate_actual[r]; ec+=CAS.wstate_actual[r]*E_bare[r]; }
        fprintf(out_stream,"  CAS ensemble energy (SA average)   = % .12f\n", ec/wsum);
    }
    fprintf(out_stream,"\n  Correlation energy breakdown:\n");
    fprintf(out_stream,"    [Fr, T1]                         = % .12f\n", L.E_FT1);
    fprintf(out_stream,"    [Fr, T2]                         = % .12f\n", L.E_FT2);
    fprintf(out_stream,"    [Vr, T1]                         = % .12f\n", L.E_VT1);
    fprintf(out_stream,"    [Vr, T2] L1                      = % .12f\n", vt2_L1);
    fprintf(out_stream,"    [Vr, T2] L2                      = % .12f\n", vt2_L2);
    fprintf(out_stream,"    [Vr, T2] L3                      = % .12f\n", vt2_L3);
    fprintf(out_stream,"      DF CCVV                        = % .12f\n", E_CCVV);
    fprintf(out_stream,"      DF CAVV                        = % .12f\n", E_CAVV);
    fprintf(out_stream,"      DF CCAV                        = % .12f\n", E_CCAV);
    fprintf(out_stream,"\n  Correlation energy E(2)            = % .12f\n", E_corr);
    fprintf(out_stream,"  Total DSRG-PT2 energy              = % .12f\n", Eref + E_corr);

    if(sa && dsrg->relax==DSRG_RELAX_NONE)
        fprintf(out_stream,"\n  NOTE: sa=1 without relax=once reports the ensemble average only; per-state "
                           "DSRG-PT2 energies need relax=once\n");

    if(dsrg->print >= 2){
        fprintf(out_stream,"\n  Per-class in-core [Vr,T2] (L1 / L2):\n");
        fprintf(out_stream,"    AAVV = % .12e / % .12e\n", L.aavv_L1, L.aavv_L2);
        fprintf(out_stream,"    CCAA = % .12e / % .12e\n", L.ccaa_L1, L.ccaa_L2);
        fprintf(out_stream,"    CAAV = % .12e / % .12e\n", L.caav_L1, L.caav_L2);
        fprintf(out_stream,"    CAAA = % .12e / % .12e\n", L.caaa_L1, L.caaa_L2);
        fprintf(out_stream,"    AAAV = % .12e / % .12e\n", L.aaav_L1, L.aaav_L2);
        fprintf(out_stream,"    E3 (v / c) = % .12e / % .12e\n", L.E3v, L.E3c);
        fprintf(out_stream,"    [Fr,T1]/[Fr,T2]/[Vr,T1] = % .12e / % .12e / % .12e\n", L.E_FT1, L.E_FT2, L.E_VT1);
    }
    fprintf(out_stream,"============================================================================\n\n");

    // relax=none publishes the total above (the ensemble's for sa=1); relax=once publishes below.
    if(grab_E==1 && dsrg->relax==DSRG_RELAX_NONE){
        E_grabbed.resize(1);
        E_grabbed[0]=Eref+E_corr;
    }

    // ---- reference relaxation (relax = once): fold the DSRG dressing into a bare active
    //      0/1/2-body operator and re-solve it in the active space, then map dressed roots to
    //      bare roots by wavefunction overlap and permute the reference weights onto the dressed
    //      roots (states are NOT reordered). Unrelaxed output is unchanged. ----
    if(dsrg->relax==DSRG_RELAX_ONCE){
        // aldet re-diagonalizes the dressed integrals directly; a backend that encodes a dressed
        // operator (DMRG/block2) re-solves it on its frozen lattice. Anything else has neither.
        if(CAS.ci_solver!=CISOLVER_ALDET && !CAS.CI->supports_dressed_import()){
            fprintf(out_stream,"ERROR: DSRG-PT2 relax=once needs a CI backend that can re-solve a dressed active "
                               "operator; use cisolver=aldet or cisolver=dmrg\n");
            exit(EXIT_FAILURE);
        }

        // reference weights: normalized ensemble weights (sa=1) or one-hot on the state-specific
        // root (sa=0). The relaxed average uses these, permuted to the matched dressed roots.
        std::vector<double> wref(ns,0.0);
        {
            double wsum=0.0; for(int r=0;r<ns;r++) wsum += CAS.wstate_actual[r];
            if(sa) for(int r=0;r<ns;r++) wref[r]=CAS.wstate_actual[r]/wsum;
            else   wref[root]=1.0;
        }

        // aldet compares CI coefficients index by index, so its bare vectors must sit in the
        // dressed operator's semicanonical basis; the MPS backend never rotates its wavefunction
        // and takes the dressed operator back to the native basis below instead.
        if(CAS.ci_solver==CISOLVER_ALDET)
            CAS.CI->malmqvist(0, Ua.data());

        // snapshot the bare states into storage set 1 before the dressed solve overwrites set 0
        // (aldet coef/E_states, DMRG the retained MPS); E_bare was captured before any RDM read.
        CAS.CI->snapshot_states(1);

        // dressed active operator: Hbar = seed + 1/2[H~1,A] + the two batched CAVV/CCAV Hbar1
        // corrections, de-normal-ordered to a bare 0/1/2-body operator (e0_d, h1_d, h2_d_chem).
        T.build_hbar();
        dsrg_sf_Hbar1_CAVV(R.VC_RI_M, R.VA_RI_M, n_cor, n_act, n_vir, naux,
                           eps_c, eps_a, eps_v, s, T.hbar1_ref().data());
        dsrg_sf_Hbar1_CCAV(R.VC_RI_M, R.CA_RI_M, n_cor, n_act, n_vir, naux,
                           eps_c, eps_a, eps_v, s, T.hbar1_ref().data());
        T.degno();

        // Seam re-solve on the dressed operator (never CI_calc: it re-imports the bare integrals
        // and clobbers the dressing). e0_d is the total-energy scalar, passed as-is.
        std::vector<double> h2d(T.dressed_h2_chem()), h1d(T.dressed_h1());
        if(CAS.ci_solver==CISOLVER_ALDET){
            CAS.CI->import_integrals(h2d.data(), h1d.data(), T.dressed_e0());
            CAS.CI->solve(1, 0, false);   // cold: a warm guess would overwrite the snapshot
        }
        else if(CAS.ci_solver==CISOLVER_DMRG){
            // The MPS is never rotated, so the dressed operator goes to it in the native active
            // basis: undo the semicanonicalization on both tensors (Ua columns are eigenvectors,
            // so native <- semi is forward=false). The lattice and the localization are frozen.
            std::vector<double> h1n(na2), h2n(na4);
            rotate1(h1d.data(), Ua.data(), n_act, h1n.data(), /*forward=*/false);
            rotate2(h2d.data(), Ua.data(), n_act, h2n.data(), /*forward=*/false);
            CAS.CI->import_dressed_operator(h1n.data(), h2n.data(), nullptr, T.dressed_e0());
            CAS.CI->solve(1, 0, true);    // warm off the converged bare MPS, same lattice
        }
        else{
            fprintf(out_stream,"ERROR: unknown CISOLVER (%d); accepted: aldet, dmrg\n",CAS.ci_solver);
            exit(EXIT_FAILURE);
        }

        std::vector<double> E_dressed(ns);
        for(int r=0;r<ns;r++) E_dressed[r] = CAS.CI->E_state(r);

        // root map by overlap argmax: S[d*ns+b] = <dressed_d | bare_b>. A dressed root claims
        // its best bare root only above the threshold; the bare roots are orthonormal, so
        // sum_d |S[d,b]|^2 <= 1 makes that injective and only leaves roots unassigned (d2b=-1).
        std::vector<double> S((size_t)ns*ns,0.0);
        CAS.CI->calc_S(S.data(), 0, 1);
        std::vector<int> d2b(ns,-1), root_map(ns,-1);
        std::vector<double> ov(ns,0.0);
        for(int d=0;d<ns;d++){
            int best=0; double bmax=-1.0;
            for(int b=0;b<ns;b++){ const double a=std::fabs(S[(size_t)d*ns+b]); if(a>bmax){bmax=a;best=b;} }
            ov[d]=bmax;
            if(bmax>DSRG_RELAX_OVERLAP_MIN){ d2b[d]=best; root_map[best]=d; }
        }

        // Every dressed root carries weight: a matched one takes its bare root's, the rest take the
        // unclaimed weights in order (the map is injective, so the two counts agree). The SA average
        // is a summary number and spans the whole manifold. sa=0 keeps its one-hot weight, so an
        // unmatched target root still reports nothing.
        std::vector<double> wd(ns,0.0);
        {
            std::vector<int> spare;
            for(int b=0;b<ns;b++) if(root_map[b]<0) spare.push_back(b);
            size_t k=0;
            for(int d=0;d<ns;d++){
                if(d2b[d]>=0)                 wd[d]=wref[d2b[d]];
                else if(sa && k<spare.size()) wd[d]=wref[spare[k++]];
            }
        }
        double e_relax=0.0, w_cov=0.0;
        for(int d=0;d<ns;d++){ e_relax += wd[d]*E_dressed[d]; w_cov += wd[d]; }

        T.set_root_data(wref.data(), E_bare.data(), E_dressed.data(), root_map.data(), ns);

        if(getenv("DSRG_PILOT_DUMP"))
            T.pilot_dump(getenv("DSRG_PILOT_DUMP"), M->n_act_el_alp[0], M->n_act_el_bet[0], ns, root);

        fprintf(out_stream,"\n");
        fprintf(out_stream,"===================== DSRG-PT2 relaxation (relax = once) =====================\n\n");
        for(int d=0;d<ns;d++){
            char bcol[16];
            if(d2b[d]>=0) snprintf(bcol,sizeof(bcol),"%d",d2b[d]);
            else          snprintf(bcol,sizeof(bcol),"?");
            fprintf(out_stream,"  root %2d : E(dressed) = % .12f   bare root = %2s   |overlap| = %.6f\n",
                    d, E_dressed[d], bcol, ov[d]);
        }
        fprintf(out_stream,"\n");
        // w_cov==0 is the state-specific case whose target root matched nothing: no energy to report.
        if(w_cov<=0.0)
            fprintf(out_stream,"  Relaxed energy                     =  none (no dressed root matched the reference)\n");
        else if(sa) fprintf(out_stream,"  Relaxed SA average                 = % .12f\n", e_relax);
        else        fprintf(out_stream,"  Relaxed DSRG-PT2 energy            = % .12f\n", e_relax);
        int n_unm=0; for(int d=0;d<ns;d++) if(d2b[d]<0) n_unm++;
        if(sa && n_unm)
            fprintf(out_stream,"\n  NOTE: %d dressed root(s) matched no reference root ('?' above); they are averaged\n"
                               "        in with the unclaimed reference weights.\n", n_unm);
        fprintf(out_stream,"============================================================================\n\n");

        // sa=1: every dressed root, in dressed order; sa=0: the relaxed target root, if one matched.
        if(grab_E==1){
            if(sa){
                E_grabbed.resize(ns);
                for(int d=0;d<ns;d++) E_grabbed[d]=E_dressed[d];
            }
            else if(w_cov>0.0){
                E_grabbed.resize(1);
                E_grabbed[0]=e_relax;
            }
        }
    }

    printf_timer("DSRG-PT2");
    return 0;
}
