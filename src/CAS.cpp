//standart
//no

//doCI
# include "blas_link.h"
# include "molecule.h"
# include "matr.h"
# include "doCI_matr.h"
# include "libint_link.h"
# include "converger_2_1.h"
# include "CAS.h"
# include "localizer.h"
# include "localized_dmrg.h"   // build_loc_orbitals (warm-start rotation)
# include "tensor_rotate.h"    // rotate1/rotate2 (active-basis transforms)
# include "dmrg_log.h"         // per-solve block2 sweep log
# include "aldet_casci_wrap.h"
#ifdef NOPT_HAS_BLOCK2
# include "block2_casci_wrap.h"
#endif
# include "defaults.h"
# include "RI.h"
# include "inp_out.h"
# include "trcamm.h"
# include "timer.h"
# include "jacobi.h"
# include "common_vars.h"


extern int num_threads;

int average_DM(double * G, std::vector<double> avecoe,int na_p, int n_s){
    
    double norm=avecoe[0];
    for(int j=0;j<na_p;j++){
        G[j]=G[j]*avecoe[0];
    }
    for(int i=1;i<avecoe.size();i++){
        norm+=avecoe[i];
        for(int j=0;j<na_p;j++){
            G[j]+=G[i*na_p+j]*avecoe[i];
        }
    }
    for(int j=0;j<na_p;j++)
        G[j]=G[j]/norm;
    
//     fprintf(out_stream,"norm = %e\n",norm);
    return 0;
}


int average(double * G, int na_p, int n_s){
    
    double norm=1.0;
    
    for(int i=1;i<n_s;i++){
        norm+=1.0;
        for(int j=0;j<na_p;j++){
            G[j]+=G[i*na_p+j];
        }
    }
    for(int j=0;j<na_p;j++)
        G[j]=G[j]/norm;
    
//     fprintf(out_stream,"norm = %e\n",norm);
    return 0;
}





CAS_engine::CAS_engine(){
    
    F_core_AO   = NULL;
    F_core_MO   = NULL;
    F_act        = NULL;
    act_INTS     = NULL;
    Lambda_act   = NULL;
    
    aaag_ints   = NULL;
    aaaa_ints   = NULL;
    aa_J_ints   = NULL;
    aa_K_ints   = NULL;
    cv_J_ints   = NULL;
    cv_K_ints   = NULL;
    F_tot       = NULL;
    gamma       = NULL;
    G_ga        = NULL;
    GAMMA       = NULL;
    G           = NULL;
    B           = NULL;
    S_track     = NULL;
    MO_BUF      = NULL;
    MO_VEC      = NULL;
    GEN_CVEC    = NULL;
    ACT_CVEC    = NULL;
    DM_C        = NULL;
    DM_A        = NULL;
    J           = NULL;
    K           = NULL;
    H           = NULL;
    
    Prop_AO     = NULL;
    Prop_Act    = NULL;
    Prop_Core   = NULL;
    Prop_value  = NULL;
    Prop_nuc    = NULL;
    
    n_CI=1;
    
}

int CAS_engine::init(cas_par * cas, molecule * ext_M){
    
    
    M =ext_M;
    n_ao  = M->n_ao;
    n_act  = M->n_act_orb[0];//reading from A state 0 fragment 0
    n_core = M->n_cor_orb;   //reading from A state 0 fragment 0
    n_vac  = n_ao-n_core-n_act;
    if(D5)
        n_vac  = M->n_ao_d5-n_core-n_act;
    
    rep_num = M->rep_num;
    
    
    n_s = cas->n_s;
    
    F_core_AO   = new double[n_ao*n_ao];
    F_core_MO   = new double[n_ao*n_ao];
    J           = new double[n_ao*n_ao];
    K           = new double[n_ao*n_ao];
    F_act        = new double[n_act*n_act];
    Lambda_act   = new double[n_act*n_act];
    ACT_CVEC    = new double[n_act*n_ao];
    aaaa_ints   = new double[n_act*n_act*n_act*n_act];
    aa_J_ints   = new double[n_act*n_act*n_ao];
    aa_K_ints   = new double[n_act*n_act*n_ao];
    cv_J_ints   = new double[n_core*n_vac];
    cv_K_ints   = new double[n_core*n_vac];
    DM_C        = new double[n_ao*n_ao];
    gamma       = new double[n_act*n_act*n_s*n_s];
    H           = new double[n_s*n_s];
    S_track     = new double[n_s*n_s];
    DM_A        = new double[n_ao*n_ao];
    
//     memcpy(MO_VEC,ext_MO,sizeof(double)*n_ao*n_ao);
//     memcpy(NO_VEC,ext_MO,sizeof(double)*n_ao*n_core);

    MO_VEC      = M->MO_VEC;
    //transposition of VEC to CVEC
    for(int i=0; i<n_ao ;i++)
    for(int j=0; j<n_act;j++)
        ACT_CVEC[i*n_act+j]=MO_VEC[(j+n_core)*n_ao+i];
    
    
    n_CI=M->n_CI;
    if(n_CI!=1){
        fprintf(out_stream,"ERROR: n_CI=%d is not supported by the casci_solver\n",n_CI);
        exit(0);
    }
    ci_solver=cas->ci_solver;
    if      (cas->ci_solver==CISOLVER_ALDET){
        CI_owner = std::make_unique<aldet_casci_wrap>(M->CI+0, &dav, n_s);
    }
    else if (cas->ci_solver==CISOLVER_DMRG){
#ifdef NOPT_HAS_BLOCK2
        // --- validate the DMRG request before constructing any block2 object ---
        {
            const int na = M->CI[0].na, nb = M->CI[0].nb, mult = M->CI[0].mult;
            const int N = na + nb, twoS = mult - 1;
            int dsz = na - nb; if(dsz < 0) dsz = -dsz;

            // spin sector: block2 targets 2S=mult-1 in the (na-nb)/2 projection
            if(mult < 1){
                fprintf(out_stream,"ERROR: DMRG needs mult>=1 (got %d); mult=0 (all multiplicities) is unsupported\n", mult);
                exit(EXIT_FAILURE);
            }
            if(N < 2){
                fprintf(out_stream,"ERROR: DMRG-CASSCF wants >=2 active electrons to correlate -- CAS(%d,%d)"
                                   " gives it nothing to chew on (one electron is Hartree-Fock's job, zero is nobody's)\n", N, n_act);
                exit(EXIT_FAILURE);
            }
            if((N - twoS) % 2 != 0){
                fprintf(out_stream,"ERROR: DMRG spin sector inconsistent: N=%d and 2S=%d differ in parity\n", N, twoS);
                exit(EXIT_FAILURE);
            }
            if(dsz > twoS){
                fprintf(out_stream,"ERROR: DMRG spin sector inconsistent: |na-nb|=%d exceeds 2S=%d\n", dsz, twoS);
                exit(EXIT_FAILURE);
            }
            if(twoS > N || twoS > 2*n_act - N){
                fprintf(out_stream,"ERROR: DMRG spin 2S=%d exceeds what %d electrons in %d orbitals allow\n", twoS, N, n_act);
                exit(EXIT_FAILURE);
            }
            if(n_s < 1){
                fprintf(out_stream,"ERROR: DMRG needs n_s>=1 (got %d)\n", n_s);
                exit(EXIT_FAILURE);
            }

            // Separate minimization builds one orbital gradient per state, so it reads the per-state
            // 2-RDMs. The DMRG backend only ever forms their state average -- the per-state tensors
            // never coexist -- which is what block2's shared renormalized basis solves for anyway.
            if(cas->method != 1){
                fprintf(out_stream,"ERROR: DMRG supports state-averaged CAS-SCF only (method=1);"
                                   " separate minimization needs the per-state 2-RDMs\n");
                exit(EXIT_FAILURE);
            }

            // symmetry: the block2 backend runs in C1 -- no linear (Lambda/parity) or spatial-irrep targeting
            if(LINEAR){
                fprintf(out_stream,"ERROR: DMRG backend does not support linear-molecule symmetry (Lambda/parity); use cisolver=aldet\n");
                exit(EXIT_FAILURE);
            }
            if(cas->w_state_type == 2){
                fprintf(out_stream,"ERROR: DMRG does not support by-irrep state selection (wstate rep)\n");
                exit(EXIT_FAILURE);
            }

            // state weights: block2 shares one renormalized basis with EQUAL root weights (each 1/nroots).
            // Accept only equal weights -- `wstate all_1` (w_state_type==1, materialized to n_s ones), or an
            // explicit wstate vector of exactly n_s positive, all-equal entries.
            if(cas->w_state_type != 1){
                if((int)cas->w_state.size() != n_s){
                    fprintf(out_stream,"ERROR: DMRG requires exactly n_s state weights (got %d for n_s=%d);"
                                       " use `wstate all_1` or give n_s equal positive weights\n",
                                       (int)cas->w_state.size(), n_s);
                    exit(EXIT_FAILURE);
                }
                for(int i=0;i<n_s;i++)
                    if(cas->w_state[i] <= 0.0){
                        fprintf(out_stream,"ERROR: DMRG state weights must be positive\n");
                        exit(EXIT_FAILURE);
                    }
                for(int i=1;i<n_s;i++)
                    if(fabs(cas->w_state[i]-cas->w_state[0]) > 1e-10){
                        fprintf(out_stream,"ERROR: DMRG state averaging requires equal weights\n");
                        exit(EXIT_FAILURE);
                    }
            }
        }
        CI_owner = std::make_unique<block2_casci_wrap>(
            n_act, M->CI[0].na, M->CI[0].nb, M->CI[0].mult, n_s, M->CI[0].print_number, cas->dmrg);
#else
        fprintf(out_stream,"ERROR: CISOLVER=dmrg selected, but this build was compiled without block2 (set USE_BLOCK2=yes)\n");
        exit(EXIT_FAILURE);
#endif
    }
    else{
        fprintf(out_stream,"ERROR: unknown CISOLVER (%d); accepted: aldet, dmrg\n",cas->ci_solver);
        exit(0);
    }
    CI = CI_owner.get();

    // Active-space localizer (PM): atomic overlap blocks are orbital-independent, so build once;
    // U is recomputed from the current orbitals each iteration in tensors_recalc.
    if(cas->ci_solver==CISOLVER_DMRG && cas->dmrg.localize==DMRG_LOC_PM){
        localizer_ = std::make_unique<pm_localizer>(*M);
        U_loc.resize((size_t)n_act*n_act);
    }
    if(cas->ci_solver==CISOLVER_DMRG){ // cache warm-start config (cas is not kept on the engine)
        warm_start_cfg = cas->dmrg.warm_start;
        warm_after_cfg = cas->dmrg.warm_start_after;
    }

    /*if(n_CI==1)*/ //if not previous CI data - alloc and set zero
    if(!CI->has_coef(0))CI->init_state_storage(n_s,0);
    CI->set_act_rep_num(rep_num+n_core);
    
    dav = cas->dav;
    
    if(LINEAR){
        S_rep     =cas->rep_spin      ;
        P_rep     =cas->rep_sign      ;
        Lambda_rep=cas->rep_lambda    ;
        wstate_rep=cas->w_state_by_rep;
        non_zero_w.resize(wstate_rep.size());
        for(int i=0;i<non_zero_w.size();i++)non_zero_w[i]=wstate_rep[i].size();
    
    }
    
    wstate = cas->w_state;
    if(n_s>wstate.size()){
        wstate.resize(n_s);
    }
    wstate_init=wstate;
    
    track=cas->track;
    if(LINEAR)if(track)if(Lambda_rep.size()!=0){
        track=1;
        fprintf(out_stream,"\nWARNING: states are selected by Lambda, tracking is disabled\n\n");
    }
    if(track && !CI->supports_civec_rotation()){
        fprintf(out_stream,"ERROR: state tracking requested but the CI backend does not support it\n");
        exit(0);
    }

    i_s_opt.resize(0);
    for(int i=0;i<wstate.size();i++)
        if(wstate[i]>1e-10)i_s_opt.push_back(i);
    
    n_s_opt=i_s_opt.size();
    
    if(LINEAR)if(Lambda_rep.size()!=0){
        n_s_opt=0;
        for(int i=0;i<non_zero_w.size();i++)n_s_opt+=non_zero_w[i];
        i_s_opt.resize(n_s_opt);
    }
        
    
        
//     for(int i=0;i<n_s_opt;i++)
//         printf("%d\n",i_s_opt[i]);
//     getchar();
    
    
    rotate_orbs=cas->rotate_orbs;
    if(rotate_orbs && !CI->supports_civec_rotation()){
        fprintf(out_stream,"WARNING: orbital canonicalization (rotate) disabled - the CI "
                           "backend does not support CI-vector rotation\n");
        rotate_orbs=0;
    }
    
    for(int i=0;i<n_s;i++)if(fabs(wstate[i])>1e-8)
        w_num.push_back(i);
    
    w_renum=w_num;
//     fprintf(out_stream,"AVECOE: ");for(const auto&c:wstate)fprintf(out_stream,"%e ",c);
//     fprintf(out_stream,"\n");
//     getchar();
    
    n_prop = 3;
    if(print_dispersion+print_quadrupole)
        n_prop=9;
    Prop_AO = new double*[n_prop];
    Prop_AO[0]=M->Dx_AO;
    Prop_AO[1]=M->Dy_AO;
    Prop_AO[2]=M->Dz_AO;
    if(print_dispersion+print_quadrupole){
        Prop_AO[3]=M->Qxx_AO;
        Prop_AO[4]=M->Qyy_AO;
        Prop_AO[5]=M->Qzz_AO;
        Prop_AO[6]=M->Qxy_AO;
        Prop_AO[7]=M->Qxz_AO;
        Prop_AO[8]=M->Qyz_AO;
    }
    
    
    Prop_Act  = new double[n_prop*n_act*n_act];
    Prop_Core = new double[n_prop            ];
    Prop_value= new double[n_prop*n_s  *n_s  ];
    Prop_nuc  = new double[n_prop];
    Prop_nuc[0]=M->Dx_nuc;
    Prop_nuc[1]=M->Dy_nuc;
    Prop_nuc[2]=M->Dz_nuc;
    if(print_dispersion+print_quadrupole){
        Prop_nuc[3]=0;
        Prop_nuc[4]=0;
        Prop_nuc[5]=0;
        Prop_nuc[6]=0;
        Prop_nuc[7]=0;
        Prop_nuc[8]=0;
    }
    return 0;
}
                    
int CAS_engine::SCF_alloc(){
    
    //allocate memory
    int n_s=CI->n_states();
    aaag_ints = new double[n_act*n_act*n_act*n_ao];
    F_tot     = new double[n_ao*n_ao*n_s_opt];
    G_ga      = new double[n_act*n_ao*n_s_opt];
    // GAMMA takes the CI backend's 2-RDM: n_s per-state tensors, averaged here (aldet), or the
    // single state-averaged one the backend already accumulated (DMRG).
    int n_s_GAMMA;
    if      (ci_solver==CISOLVER_ALDET) n_s_GAMMA = n_s;
    else if (ci_solver==CISOLVER_DMRG ) n_s_GAMMA = 1;
    else{
        fprintf(out_stream,"ERROR: unknown CISOLVER (%d); accepted: aldet, dmrg\n",ci_solver);
        exit(0);
    }
    GAMMA     = new double[n_act*n_act*n_act*n_act*n_s_GAMMA];
    G         = new double[(n_core*n_act+n_core*n_vac+n_act*n_vac)*n_s_opt];
    B         = new double[(n_core*n_act+n_core*n_vac+n_act*n_vac)*n_s_opt];
    MO_BUF    = new double[n_ao*n_ao];
    GEN_CVEC  = new double[n_ao*n_ao];
    
    if(RI)RI_core_realloc(n_core+n_act, n_ao);
    
//     DAV.set_par(&(CI),dav_n_s,dav_eps, dav_max_it, 0);
    
    return 0;

}

int CAS_engine::update_ACT_CVEC(){

    //transposition of VEC to CVEC
    for(int i=0; i<n_ao ;i++)
    for(int j=0; j<n_act;j++)
        ACT_CVEC[i*n_act+j]=MO_VEC[(j+n_core)*n_ao+i];
    
    return 0;
}

// Apply the pending canonicalization to n_blocks consecutive 1-RDM blocks: gamma' = U gamma U^T.
// rotate1 forbids aliasing, so every block goes through the scratch.
int CAS_engine::rotate_pending_gamma(double * g, int n_blocks){

    if(U_pending.empty())return 0;

    const size_t blk = (size_t)n_act*n_act;
    std::vector<double> buf(blk);
    for(int i=0;i<n_blocks;i++){
        double * g_i = g+i*blk;
        rotate1(g_i, U_pending.data(), n_act, buf.data(), /*forward=*/false);
        memcpy(g_i, buf.data(), blk*sizeof(double));
    }

    return 0;
}

// Same for the 2-RDM. The block count must track SCF_alloc's GAMMA sizing -- per-state (aldet) or
// the single state-averaged block (DMRG) -- or one backend is silently left half-rotated.
int CAS_engine::rotate_pending_GAMMA(double * G){

    if(U_pending.empty())return 0;

    int n_blocks;
    if      (ci_solver==CISOLVER_ALDET) n_blocks = CI->n_states();
    else if (ci_solver==CISOLVER_DMRG ) n_blocks = 1;
    else{
        fprintf(out_stream,"ERROR: unknown CISOLVER (%d); accepted: aldet, dmrg\n",ci_solver);
        exit(0);
    }

    const size_t blk = (size_t)n_act*n_act*n_act*n_act;
    std::vector<double> buf(blk);
    for(int i=0;i<n_blocks;i++){
        double * G_i = G+i*blk;
        rotate2(G_i, U_pending.data(), n_act, buf.data(), /*forward=*/false);
        memcpy(G_i, buf.data(), blk*sizeof(double));
    }

    return 0;
}

int CAS_engine::tensors_recalc(int n){

    update_ACT_CVEC();

    //calc 1-el density matrices
    calc_DM_C();
    
    if(RI==0)
        calc_2el_MO_INTS(M->s, n_ao, DM_C, J, K, aaaa_ints, ACT_CVEC, ACT_CVEC, n_act);
    else
        calc_2el_MO_INTS_RI(    n_ao, DM_C, J, K, aaaa_ints, MO_VEC, MO_VEC+n_core*n_ao, MO_VEC+n_core*n_ao, n_core, n_act,0);
    
    for(int i=0;i<n_ao*n_ao;i++)F_core_AO[i]=M->H_AO[i]+2*J[i]-K[i];
    
    transform_from_col_MO(F_act, F_core_AO, n_ao, ACT_CVEC, n_act, ACT_CVEC, n_act);
    
    if(LINEAR){
        transform_from_col_MO(Lambda_act, M->Lambda_AO, n_ao, ACT_CVEC, n_act, ACT_CVEC, n_act);
        CI->import_lambda(Lambda_act, E_1el_calc(M->Lambda_AO, DM_C, n_ao, n_ao));
        
    }
    
    E_core =  E_1el_calc(M->H_AO, DM_C, n_ao, n_ao)*2
             +E_1el_calc(    J   , DM_C, n_ao, n_ao)*2
             -E_1el_calc(    K   , DM_C, n_ao, n_ao)  
             +M->V_nuc ;
             
    const bool warm_on = (warm_start_cfg==DMRG_WARM_ON);
    const int  F       = warm_after_cfg;                   // freeze/warm gate (solve index)

    // Active-space localization: block2 solves in the localized basis. Only when a localizer exists.
    if(localizer_){
        if(warm_on && warm_frozen && warm_ci_calls>F){
            // frozen frame: reuse the pinned localization instead of re-localizing, so the DMRG
            // lattice order (a function of the localized orbitals) stays valid for the retained MPS.
            std::copy(U_loc_frozen.begin(), U_loc_frozen.end(), U_loc.data());
        } else {
            loc_result lr = localizer_->localize(ACT_CVEC, n_ao, n_act, nullptr, U_loc.data());
            if(!lr.converged)
                fprintf(out_stream,"WARNING: active-space localization did not converge; running delocalized\n");
        }
        CI->set_localization(U_loc.data());
    }

    // Warm-start frame + rotation. Independent of localization: the frame orbitals are the localized
    // active orbitals when localizing, else the canonical active orbitals (ACT_CVEC) directly.
    if(warm_on){
        std::vector<double> C_loc_cur;
        const double* C_cur;
        if(localizer_){
            C_loc_cur.resize((size_t)n_ao*n_act);
            build_loc_orbitals(ACT_CVEC, U_loc.data(), n_ao, n_act, C_loc_cur.data());
            C_cur = C_loc_cur.data();
        } else {
            C_cur = ACT_CVEC;                              // canonical active orbitals [ao*n_act+p]
        }
        if(warm_ci_calls==F){
            // freeze point: pin the frame; this solve stays cold (no prior frozen-frame MPS yet).
            if(localizer_) U_loc_frozen.assign(U_loc.begin(), U_loc.end());
            C_loc_prev.assign(C_cur, C_cur + (size_t)n_ao*n_act);
            warm_frozen = true;
        } else if(warm_ci_calls>F){
            // warm: R = C_loc_prev^T S_AO C_cur (n_act x n_act, [a*n_act+p]); prev -> current
            std::vector<double> tmp((size_t)n_ao*n_act), R((size_t)n_act*n_act);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_ao,n_act,n_ao, 1.0,
                        M->S_AO,n_ao, C_cur,n_act, 0.0, tmp.data(),n_act);
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, n_act,n_act,n_ao, 1.0,
                        C_loc_prev.data(),n_act, tmp.data(),n_act, 0.0, R.data(),n_act);
            CI->set_active_rotation(R.data());
            C_loc_prev.assign(C_cur, C_cur + (size_t)n_ao*n_act);
        }
    }
    CI->import_integrals(aaaa_ints, F_act, E_core);
    warm_ci_calls++;

    return 0;
}

int CAS_engine::CI_calc(int primary, int create_track_data,int read){
    
    int n;
    tensors_recalc(0);
    n = CI->solve(primary, read, create_track_data==0);
    U_pending.clear();   // the solve hands back RDMs in the current active basis
    if(primary==-1){
        return n;
    }
    if(primary==0)if(create_track_data==0)if(track){   // track implies supports_civec_rotation() (checked in init)
        CI->calc_S(S_track,0,1);
        
//         fprintf(out_stream,"tracking\n");
//         PrintMatr(S_track, n_s,n_s,0);
        
        for(int i=0;i<w_renum.size();i++){
            int k=w_renum[i];
            w_renum[i]=-1;
            for(int j=0;j<n_s;j++)
            if(S_track[j*n_s+k]*S_track[j*n_s+k]>0.5){
                w_renum[i]=j;
                break;
            }
            if(k!=w_renum[i])
                fprintf(out_stream,"tracking changed state %d to %d\n",w_num[i],w_renum[i]);
            
            if(w_renum[i]==-1){
                fprintf(out_stream,"tracking failed: state %d is lost\n",w_num[i]);
                exit(0);
            }
            
        }
    }
    for(int i=0;i<wstate.size();i++){
        wstate[i]=0;
    }
    
    for(int i=0;i<w_renum.size();i++){
        wstate[w_renum[i]]=wstate_init[w_num[i]];
//         fprintf(out_stream,"%d ",w_renum[i]);
    }
    wstate_actual=wstate;
    
    if(LINEAR)if(Lambda_rep.size()!=0){
        
        for(int i=0;i<wstate_actual.size();i++)wstate_actual[i]=0;
        for(int i_l=0;i_l<Lambda_rep.size();i_l++){
//             printf("l[i] = %d\n",Lambda[i_l]);
            int j=0;
            for(int i=0;i<n_s;i++){
                double dE = CI->E_state(i) - CI->E_state(0);
                if(fabs(CI->S2_state(i)-S_rep[i_l]*(S_rep[i_l]+1))<1e-1)
                if(fabs(CI->L2_state(i)-Lambda_rep[i_l]*Lambda_rep[i_l])<1e-1){
                    if(Lambda_rep[i_l]==0){
                        if(fabs(CI->P_state(i)-P_rep[i_l])<1e-1){
                            wstate_actual[i]=wstate_rep[i_l][j];
//                          wstate_actual[i]=4.0/(exp(dE)+exp(-dE))/(exp(dE)+exp(-dE)); // dyn weight
                            j++;
                        }
                    }
                    else{
                        wstate_actual[i]=wstate_rep[i_l][j];
//                      wstate_actual[i]=4.0/(exp(dE)+exp(-dE))/(exp(dE)+exp(-dE)); // dyn weight
                        j++;
                    }
                }
                if(j==non_zero_w[i_l])break;
            }
            if(j<non_zero_w[i_l]){
                fprintf(out_stream,"ERROR: some states with Lambda=%d",Lambda_rep[i_l]);
                if(Lambda_rep[i_l]<1e-3){
                    if     ((P_rep[i_l]-1.0)<1e-8)printf(" (+)");
                    else if((P_rep[i_l]+1.0)<1e-8)printf(" (-)");
                    else                                 printf(" (?)");
                }
                fprintf(out_stream," are lost\n");
                fprintf(out_stream,"       increase n_s!\n");
                exit(0);
            }
        }
               
    }
    
    
    for(int i=0, j=0;i<wstate_actual.size();i++)
    if(wstate_actual[i]>1e-10){
        i_s_opt[j]=i;
        j++;
    }
//     for(int i=0;i<n_s_opt;i++)
//         printf("%d\n",i_s_opt[i]);
//     getchar();

    //the weights this iteration's RDMs are averaged with (a backend that averages internally)
    CI->set_state_weights(wstate_actual.data(),n_s);
    
    
    
    
//     fprintf(out_stream,"\n");
//     fprintf(out_stream,"AVECOE: ");for(const auto&c:wstate)fprintf(out_stream,"%e ",c);
//     fprintf(out_stream,"\n");
    
    return n;
}

int CAS_engine::calc_DM_C(){
    
    set_zero_matr(DM_C, n_ao*n_ao);        
    
    for(int i=0;i<n_ao  ;i++)
    for(int j=0;j<n_ao  ;j++)
    for(int k=0;k<n_core;k++)
        DM_C[i*n_ao+j]+=MO_VEC[k*n_ao+i]*MO_VEC[k*n_ao+j];
    
    return 0;
}
inline int copy_to_CVEC(double * CVEC, double * VEC, int n_ao){
    
    for(int i=0; i<n_ao;i++)
    for(int j=0; j<n_ao;j++)
        CVEC[i*n_ao+j]=VEC[j*n_ao+i];
    
    return 1;
}

int CAS_engine::AO_to_MO(double *M_out, double *M_in){
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                n_ao,n_ao,n_ao,1.0,
                M_in,n_ao,
                MO_VEC,n_ao,0.0,
                MO_BUF,n_ao);
     
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                n_ao,n_ao,n_ao,1.0,
                MO_VEC,n_ao,
                MO_BUF,n_ao,0.0,
                M_out,n_ao);
    
    return 0;
}

int CAS_engine::calc_gamma(){
    
    set_zero_matr(gamma,n_act*n_act*n_s);
    CI->calc_DM_diag(gamma,0);
    rotate_pending_gamma(gamma,n_s);
    
    return 0;
}

double CAS_engine::calc_E(double *g, double *G){
    
    double E=0;
    for(int t=0;t<n_act  ;t++)
    for(int u=0;u<n_act  ;u++)
        E+=g[t*n_act+u]*F_core_MO[(n_core+t)*n_ao+n_core+u];

    for(int u=0;u<n_act*n_act*n_act;u++)
    for(int t=0;t<n_act            ;t++)
        E+=G[u*n_act+t]*aaag_ints[u*n_ao+n_core+t]*0.5;

    
    return E;
}

int CAS_engine::calc_F(double * F, double *g){
    
    d_o_DMV_gen(DM_A, ACT_CVEC, n_ao, ACT_CVEC, n_ao, g, n_act, MO_BUF);
    for(int i=0;i<n_ao*n_ao;i++)DM_A[i]+=  2*DM_C[i];
    
    if(RI==1)M->nat_orb_calc(g, 1,'T');
    M->calc_F_AO(F, DM_A, 0.5);
    
    AO_to_MO(F, F);
    
    return 0;
}

int CAS_engine::calc_G(double * G_ga_state, double * gamma_state, double *GAMMA_state){
    
    set_zero_matr(G_ga_state,n_act*n_ao);
    for(int r=0;r<n_ao;r++)
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
        G_ga_state[r*n_act+t]+=gamma_state[t*n_act+u]*F_core_MO[r*n_ao+n_core+u];
    
    for(int r=0;r<n_ao             ;r++)
    for(int t=0;t<n_act            ;t++)
    for(int u=0;u<n_act*n_act*n_act;u++)
        G_ga_state[r*n_act+t]+=GAMMA_state[u*n_act+t]*aaag_ints[u*n_ao+r];
    
    return 0;
}

int CAS_engine::calc_grad(double * G_state, double * F_state, double * G_ga_state){
    
    double * g_it = G_state;
    double * g_ia = G_state+n_core*n_act;
    double * g_ta = G_state+n_core*n_act+n_core*n_vac;
    
    for(int i=0;i<n_core;i++)
    for(int t=0;t<n_act;t++){
        g_it[i*n_act+t]=4*F_state[i*n_ao+(t+n_core)]-2*G_ga_state[i*n_act+t];
	if(rep_num[i]!=rep_num[n_core+t]){
            g_it[i*n_act+t]=0;
        }
    }
        
    for(int i=0;i<n_core;i++)
    for(int a=0;a<n_vac;a++){
        g_ia[i*n_vac+a]=4*F_state[i*n_ao+(a+n_core+n_act)];
        if(rep_num[i]!=rep_num[a+n_core+n_act]){
            g_ia[i*n_vac+a]=0;
       }
    }
        
    for(int t=0;t<n_act;t++)
    for(int a=0;a<n_vac  ;a++){
        g_ta[t*n_vac+a]=2*G_ga_state[(a+n_core+n_act)*n_act+t];
        if(rep_num[t+n_core]!=rep_num[a+n_core+n_act]){
            g_ta[t*n_vac+a]=0;
        }
    }
//     printf("g_it:\n");PrintMatr(g_it,n_core,n_act,0);
//     printf("g_ia:\n");PrintMatr(g_ia,n_core,n_vac,0);
//     printf("g_ta:\n");PrintMatr(g_ta,n_act ,n_vac,0);
//     exit(0);
    
    
    
    return 0;
}

int CAS_engine::scale_grad(double * G_state, double * gamma_state){
    
    double * g_it = G_state;
    double * g_ia = G_state+n_core*n_act;
    double * g_ta = G_state+n_core*n_act+n_core*n_vac;
    
    double * sqrt_g=new double[n_act];
    
//     printf("DM:\n");
//     PrintMatr(gamma_state,n_act, n_act,1);
    
    for(int t=0;t<n_act;t++)sqrt_g[t]=sqrt(2.0/(2.0-gamma_state[t*n_act+t]));
    
    for(int i=0;i<n_core;i++)
    for(int t=0;t<n_act;t++){
        g_it[i*n_act+t]=g_it[i*n_act+t]*sqrt_g[t];
    }
    
    for(int t=0;t<n_act;t++)sqrt_g[t]=sqrt(2.0/gamma_state[t*n_act+t]);
    
    for(int t=0;t<n_act;t++)
    for(int a=0;a<n_vac  ;a++){
        g_ta[t*n_vac+a]=g_ta[t*n_vac+a]*sqrt_g[t];
    }
    
    delete[] sqrt_g;
    
//     printf("it:\n");PrintMatr(g_it,n_core,n_act,0);
//     printf("ia:\n");PrintMatr(g_ia,n_core,n_vac,0);
//     printf("ta:\n");PrintMatr(g_ta,n_act ,n_vac,0);
//     getchar();
//     
//  
    
    return 0;
}

int CAS_engine::calc_hess(double * B_state, double * gamma_state, double * F_state, double * G_ga_state){

    double * b_it = B_state;
    double * b_ia = B_state+n_core*n_act;
    double * b_ta = B_state+n_core*n_act+n_core*n_vac;
    
    for(int i=0;i<n_core;i++)
    for(int t=0;t<n_act;t++)
        b_it[i*n_act+t]= 4*F_state[(t+n_core)*n_ao+(t+n_core)]-4*F_state[i*n_ao+i]
                        +2*gamma_state[t*n_act+t]*F_state[i*n_ao+i]-2*G_ga_state[(t+n_core)*n_act+t];
    
    for(int i=0;i<n_core;i++)
    for(int a=0;a<n_vac;a++)
        b_ia[i*n_vac+a]=4*F_state[(a+n_core+n_act)*n_ao+(a+n_core+n_act)]-4*F_state[i*n_ao+i];
        
    for(int t=0;t<n_act;t++)
    for(int a=0;a<n_vac  ;a++){
        b_ta[t*n_vac+a]=2*gamma_state[t*n_act+t]*F_state[(a+n_core+n_act)*n_ao+(a+n_core+n_act)]-2*G_ga_state[(t+n_core)*n_act+t];
    }
    
//     printf("b_it:\n");PrintMatr(b_it,n_core,n_act,0);
//     printf("b_ia:\n");PrintMatr(b_ia,n_core,n_vac,0);
//     printf("b_ta:\n");PrintMatr(b_ta,n_act ,n_vac,0);
//     exit(1);
    
    return 0;
}

int CAS_engine::av_DM_and_F_calc(){
    
    //transposition of VEC to CVEC
    copy_to_CVEC(GEN_CVEC,MO_VEC,n_ao);
    
    
    //calc average DM
    calc_gamma();
    average_DM(gamma, wstate_actual,n_act*n_act,n_s);
    
    calc_F(F_tot,gamma);
    
    return 0;
    
}
    
    
int CAS_engine::make_canonical(bool with_active){
    
    av_DM_and_F_calc();
    
    double * U =new double[n_act*n_act];
    
    M->diag_X_MO_block(F_tot, 0           , n_core, nullptr);
    M->diag_X_MO_block(F_tot, n_core+n_act, n_vac , nullptr);

    // Core and virtual canonicalization touches neither the active orbitals nor the CI vector.
    // The active block rotates only where no solve follows: lapack_diag absorbs the ordering
    // permutation, so its rotation can be improper (det<0), which no warm-started backend carries.
    if(!with_active)return 0;

    // Active-block canonicalization rotates the active orbitals, so the CI vector must
    // follow via malmqvist and the basis-dependent caches must be rebuilt -- a stale
    // ACT_CVEC silently corrupts DM_A and the properties. A backend that can't rotate
    // its CI vector (e.g. DMRG) keeps its wavefunction and rotates the RDMs instead.
    if(CI->supports_civec_rotation()){
        M->diag_X_MO_block(F_tot, n_core   , n_act , U);
        CI->malmqvist(0, U);
        tensors_recalc(0);
    }
    if(!CI->supports_civec_rotation() && n_act>0){
        double * U_canon  = nullptr;//to be move to molecule
        U_canon  = new double[n_act*n_act];
        
        for(int i=0;i<n_act;i++)
            for(int j=0;j<n_act;j++)
                U_canon[i*n_act+j]=F_tot[(i+n_core)*n_ao+(j+n_core)];
        lapack_diag(U_canon, M->orb_energy+n_core, n_act);//ir.rep can be broken -- must be rewritten in the "diag_X_MO_block"-style
        normalize_rotation_rows(U_canon, n_act);

        // The wavefunction sits in the last solved basis, and a canonicalization may still be owed
        // to it, so this one composes with what is pending: U_total = U_canon * U_pending.
        // dgemm must not alias, hence the fresh buffer.
        if(U_pending.empty())
            U_pending.assign(U_canon, U_canon+(size_t)n_act*n_act);
        else{
            std::vector<double> U_total((size_t)n_act*n_act);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_act,n_act,n_act,1.0,
                        U_canon,n_act,
                        U_pending.data(),n_act,0.0,
                        U_total.data(),n_act);
            U_pending.swap(U_total);
        }
        CI->set_report_rotation(U_pending.data());
        
        double * B  = new double[n_act*n_ao];
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    n_act,n_ao,n_act,1.0,
                    U_canon,n_act,
                    M->MO_VEC+n_core*n_ao,n_ao,0.0,
                    B,n_ao);
        memcpy(M->MO_VEC+n_core*n_ao, B, n_act*n_ao*sizeof(double));
        M->check_orb_symmetry();
        delete[] B;
        
        // Refresh the active orbital copy. No tensors_recalc here -- it would re-localize and
        // advance the warm-start frame past the retained wavefunction.
        update_ACT_CVEC();

    }
    
    
    return 0;
}

double CAS_engine::SA_grad_hess_calc(){
    
    int n_s=CI->n_states();
    
    av_DM_and_F_calc();
    
    AO_to_MO(F_core_MO,F_core_AO);
    
    if(RI==0)
        calc_2el_MO_INTS_XXXY(M->s, n_ao, aaag_ints, ACT_CVEC, GEN_CVEC, n_act, n_ao);
    else{
        calc_2el_MO_INTS_AAAG_RI(          aaag_ints, MO_VEC  , n_core  , n_act, n_ao);
    }
    
    //calc 2DM
    CI->G_calc(GAMMA);
    rotate_pending_GAMMA(GAMMA);
    if      (ci_solver==CISOLVER_ALDET){
        average_DM(GAMMA,wstate_actual,n_act*n_act*n_act*n_act,n_s);
    }
    else if (ci_solver==CISOLVER_DMRG ){
        // already state-averaged in the backend, with the same weights
    }
    else{
        fprintf(out_stream,"ERROR: unknown CISOLVER (%d); accepted: aldet, dmrg\n",ci_solver);
        exit(0);
    }


    calc_G(G_ga,gamma,GAMMA);
    
    //calc orb grad
    calc_grad(G, F_tot, G_ga);
    
    //calc orb hess
    calc_hess(B, gamma, F_tot, G_ga);
    
    return E_core+calc_E(gamma,GAMMA);
    
}

double CAS_engine::SM_grad_hess_calc(){
    
    int n_s=CI->n_states();
    
    double ** gamma_state = new double*[n_s_opt];
    double ** GAMMA_state = new double*[n_s_opt];
    double ** F_state     = new double*[n_s_opt];
    double ** G_ga_state  = new double*[n_s_opt];
    double ** G_state     = new double*[n_s_opt];
    double ** B_state     = new double*[n_s_opt];
    
    for(int i=0;i<n_s_opt;i++)gamma_state[i]=gamma+i_s_opt[i]*n_act*n_act            ;
    for(int i=0;i<n_s_opt;i++)GAMMA_state[i]=GAMMA+i_s_opt[i]*n_act*n_act*n_act*n_act;
    
    for(int i=0;i<n_s_opt;i++)   F_state[i] = F_tot + i*n_ao*n_ao;
    for(int i=0;i<n_s_opt;i++)G_ga_state[i] = G_ga  + i*n_ao*n_act;
    for(int i=0;i<n_s_opt;i++)   G_state[i] = G     + i*(n_core*n_act+n_core*n_vac+n_act*n_vac);
    for(int i=0;i<n_s_opt;i++)   B_state[i] = B     + i*(n_core*n_act+n_core*n_vac+n_act*n_vac);
    
    
    step_start:;
    
    copy_to_CVEC(GEN_CVEC,MO_VEC,n_ao);

    calc_gamma();
    
    for(int i=0;i<n_s_opt;i++)calc_F(F_state[i],gamma_state[i]);
    
    // if(no_rot_v==0)if(rotate_orbs==1){
        // average(F_tot,n_ao*n_ao,n_s_opt);
        // M->diag_X_MO_block(F_tot, n_core+n_act,n_vac, nullptr);
        // no_rot_v++;
        // goto step_start;
    // }
    
    AO_to_MO(F_core_MO,F_core_AO);
    
    if(RI==0)
        calc_2el_MO_INTS_XXXY(M->s, n_ao, aaag_ints, ACT_CVEC, GEN_CVEC, n_act, n_ao);
    else{
        calc_2el_MO_INTS_AAAG_RI(          aaag_ints, MO_VEC  , n_core  , n_act, n_ao);
    }
    
    //calc 2DM
    CI->G_calc(GAMMA);
    rotate_pending_GAMMA(GAMMA);
    for(int i=0;i<n_s_opt;i++)calc_G(G_ga_state[i], gamma_state[i], GAMMA_state[i]);
    
    
    //calc orb grad
    for(int i=0;i<n_s_opt;i++)calc_grad(G_state[i], F_state[i], G_ga_state[i]);
    
    for(int i=0;i<n_s_opt;i++)calc_hess(B_state[i], gamma_state[i], F_state[i], G_ga_state[i]);
    
//     for(int i=0;i<n_s_opt;i++)scale_grad(G_state[i], gamma_state[i]);


    average(B,n_core*n_act+n_core*n_vac+n_act*n_vac,n_s_opt);
    
//     for(int i=0;i<n_s_opt;i++){
//         double * g_it = G_state[i];
//         double * g_ia = G_state[i]+n_core*n_act;
//         double * g_ta = G_state[i]+n_core*n_act+n_core*n_vac;
//         double * b_it = B_state[i];
//         double * b_ia = B_state[i]+n_core*n_act;
//         double * b_ta = B_state[i]+n_core*n_act+n_core*n_vac;
//         
//         printf("g_it:\n");PrintMatr(g_it,n_core,n_act,0);
//         printf("g_ia:\n");PrintMatr(g_ia,n_core,n_vac,0);
//         printf("g_ta:\n");PrintMatr(g_ta,n_act ,n_vac,0);
// 
//         printf("b_it:\n");PrintMatr(b_it,n_core,n_act,0);
//         printf("b_ia:\n");PrintMatr(b_ia,n_core,n_vac,0);
//         printf("b_ta:\n");PrintMatr(b_ta,n_act ,n_vac,0);
//     }
//     exit(1);

    
    double * E_state = new double[n_s_opt];
    
    for(int i=0;i<n_s_opt;i++)E_state[i]=E_core+calc_E(gamma_state[i],GAMMA_state[i]);
//     for(int i=0;i<n_s_opt;i++)printf("E[%d] = % 18.10f\n",i,E_state[i]);
    average(E_state,1,n_s_opt);
//     printf("E_av = % 18.10f\n",E_state[0]);
//     getchar();
    
    delete[] gamma_state;
    delete[] GAMMA_state;
    delete[] F_state    ;
    delete[] G_ga_state ;
    delete[] G_state    ;
    delete[] B_state    ;
    double E = E_state[0];
    delete[] E_state;
//     printf(" E = %e\n",E);
//     exit(0);
    return E;

}

int CAS_engine::Prop_calc(){
    
    Prop_calc_with_num(n_prop);
    
    return 0;
}



int CAS_engine::Prop_calc_with_num(int n_calc_prop){
    
    if(n_calc_prop>n_prop){
        fprintf(stdout, "ERROR: can not calculate %d print_properties matrices\n", n_calc_prop);
        fprintf(stdout, "       memory is allocated only for %d matrices\n", n_prop);
        exit(0);
    }
    
    set_zero_matr(Prop_value, n_calc_prop*n_s*n_s);
    
    calc_DM_C();
    for(int i=0; i<n_calc_prop;i++){
        Prop_Core[i] = cblas_ddot(n_ao*n_ao, DM_C,1, Prop_AO[i], 1)*2.0;
//         fprintf(out_stream,"%e\n",Prop_Core[i]);
    }
    
    set_zero_matr(Prop_Act, n_calc_prop*n_act*n_act);
    for(int i=0; i<n_calc_prop; i++)
        transform_from_col_MO(Prop_Act+i*n_act*n_act, Prop_AO[i], n_ao, ACT_CVEC, n_act, ACT_CVEC, n_act);
    
    set_zero_matr(gamma,n_act*n_act*n_s*n_s);
    CI->calc_DMA(gamma,0,0);
    CI->calc_DMB(gamma,0,0);
    rotate_pending_gamma(gamma,n_s*n_s);
    for(int i=0; i<n_calc_prop; i++)
    for(int j=0; j<n_s   ; j++){
        Prop_value[i*n_s*n_s+j*(n_s+1)]=Prop_Core[i]+Prop_nuc[i];
        for(int k=0; k<n_s; k++){
//             d_o_DMV_gen(DM_A, ACT_CVEC, n_ao, ACT_CVEC, n_ao, gamma+(j*n_s+k)*n_act*n_act, n_act);
            Prop_value[i*n_s*n_s+j*n_s+k]+=cblas_ddot(n_act*n_act, gamma+(j*n_s+k)*n_act*n_act,1, Prop_Act+i*n_act*n_act, 1);
        }
    }
    
    return 0;
}

int CAS_engine::TrDM(int a, int b){
    
    double * gamma_ab = new double [n_act*n_act];
    set_zero_matr(gamma,n_act*n_act*n_s*n_s);
    CI->calc_DMA(gamma,0,0);
    CI->calc_DMB(gamma,0,0);
    rotate_pending_gamma(gamma,n_s*n_s);
    
    for(int j=0;j<n_act*n_act;j++){
        gamma_ab[j]=gamma[(a*n_s+b)*n_act*n_act+j];
    }
    if(MO_BUF==NULL)MO_BUF= new double[n_ao*n_ao];
    d_o_DMV_gen(DM_A, ACT_CVEC, n_ao, ACT_CVEC, n_ao, gamma_ab, n_act,MO_BUF);

    if(a==b){
        calc_DM_C();
        for(int i=0;i<n_ao*n_ao;i++)DM_A[i]+=  2*DM_C[i];
    }
//     fprintf(out_stream,"DM:\n");
//     PrintMatr(gamma_ab,n_act,n_act,1);
    delete[] gamma_ab;
    
    return 0;
    
}
    

int CAS_engine::TrC_calc(int a, int b){
    
    TrDM(a,b);
    
    if(a==b)fprintf(out_stream,"\nState %d charges:\n",a);
    if(a!=b)fprintf(out_stream,"\nStates [%d;%d] transition charges:\n",a,b);
    double *A_DM = new double[n_ao*n_ao];
    double q;
    for(int i_a=0;i_a<M->n_atoms;i_a++){
        gen_atomic_DM(A_DM, M->S_AO, i_a, &(M->s), &(M->shell_center), n_ao);
        for(int i=0;i<n_ao*n_ao;i++)A_DM[i]=-A_DM[i];
        q=E_1el_calc(A_DM,DM_A, n_ao, n_ao);
        if(a==b)q+=M->nucl_charges_calc[i_a];
        fprintf(out_stream,"c(%c%c) =% .5e\n",M->atom_names[i_a][0],M->atom_names[i_a][1],i_a, q);
    }
    delete[] A_DM;
    
    
    return 0;
}

int CAS_engine::print_properties(const char* name){
    
    if(print_dipole+print_dispersion+print_quadrupole+print_mulliken)
        Prop_calc();
    
    if(print_dipole){
        fprintf(out_stream,"\n\n%s Dipole moment:\n", name);
        PrintDipole(Prop_value, 
                    Prop_value+n_s*n_s  , 
                    Prop_value+n_s*n_s*2,
                    n_s);
    }
    
    if(print_dispersion){
        fprintf(out_stream,"\n\n%s Electron dispersion:\n",name);    
        PrintDispersion(Prop_value+n_s*n_s*3, 
                        Prop_value+n_s*n_s*4, 
                        Prop_value+n_s*n_s*5,
                        n_s);
    }
    
    if(print_quadrupole){
        fprintf(out_stream,"\n\n%s Quadrupole moment:\n",name);
        PrintQuadrupole(Prop_value+n_s*n_s*3, 
                        Prop_value+n_s*n_s*4, 
                        Prop_value+n_s*n_s*5,
                        Prop_value+n_s*n_s*6, 
                        Prop_value+n_s*n_s*7, 
                        Prop_value+n_s*n_s*8,
                        n_s);
    }
    
    if(print_mulliken){
        fprintf(out_stream,"\n\n%s Mulliken charges:\n",name);
        for(int a=0;a<n_s;a++)
//         for(int b=0;b<n_s;b++)
            TrC_calc(a,a);
    }
    
    return 0;
    
}

int CAS_engine::rotate(){
    
    counter_first_sigma=-1; //// !!!
    counter_for_number_of_Pi_states=0;
    ind_pi_states = new int[wstate_actual.size()];

    double sin_phi_rot_Pi;
    double cos_phi_rot_Pi;
    double dipole_first_in_pi_pair_x;     ////    double dipole_first_in_pi_pair_y;
    double dipole_second_in_pi_pair_x;    ////    double dipole_second_in_pi_pair_y;

    double old_coef_first_in_pi_pair;  //// double **
    double old_coef_second_in_pi_pair;
    double new_coef_first_in_pi_pair;
    double new_coef_second_in_pi_pair;

    for(int i=0;i<wstate_actual.size();i++){
       
        if(CI->L2_state(i)<1e-3)
            if (counter_first_sigma==-1) 
                counter_first_sigma=i;
        
        if((fabs(CI->L2_state(i)-1))<1e-3){
            ind_pi_states[counter_for_number_of_Pi_states]=i;
            counter_for_number_of_Pi_states++;
            //// get number of pi pairs and their indices
        }
    }
     
    Prop_calc_with_num(3);
    for(int new_iterator=0;new_iterator<(counter_for_number_of_Pi_states/2);new_iterator++){

        dipole_first_in_pi_pair_x = Prop_value[counter_first_sigma*wstate_actual.size()+ind_pi_states[new_iterator*2  ]];
        dipole_second_in_pi_pair_x= Prop_value[counter_first_sigma*wstate_actual.size()+ind_pi_states[new_iterator*2+1]];

        sin_phi_rot_Pi = dipole_first_in_pi_pair_x/(sqrt(dipole_first_in_pi_pair_x*dipole_first_in_pi_pair_x+dipole_second_in_pi_pair_x*dipole_second_in_pi_pair_x));
        cos_phi_rot_Pi = dipole_second_in_pi_pair_x/(sqrt(dipole_first_in_pi_pair_x*dipole_first_in_pi_pair_x+dipole_second_in_pi_pair_x*dipole_second_in_pi_pair_x));
           //// x component comes first

        if(!CI->supports_civec_rotation()){
            fprintf(out_stream,"ERROR: Pi-pair rotation needs CI-vector rotation, unsupported by the CI backend\n");
            exit(0);
        }
        CI->rotate_pi_pair(0, sin_phi_rot_Pi, cos_phi_rot_Pi, new_iterator, ind_pi_states); //// see end of aldet.cpp
    }
    
    return 0;
}
int CAS_engine::print_av_table(const char* type_of_calc){
    
    print_av_table_with_prop(type_of_calc, 0, NULL, NULL);
    
    return 0;
    
}

int CAS_engine::print_av_table_with_prop(const char* type_of_calc, int num_prop, double ** P, char ** names){
    
    
    fprintf(out_stream,"\n\n%s\n",type_of_calc);
    fprintf(out_stream,"__________________________________");
    if(LINEAR)fprintf(out_stream,"___________");
    fprintf(out_stream,"________");
    for(int i=0; i<num_prop;i++)fprintf(out_stream,"________________");
    fprintf(out_stream,"\n");
    
    fprintf(out_stream," State| E                 | S^2  |");
    if(LINEAR)fprintf(out_stream," L^2      |");
    fprintf(out_stream," weight |");
    for(int i=0; i<num_prop;i++){
        fprintf(out_stream," ");
        for(int j=0; j<13;j++)fprintf(out_stream,"%c",names[i][j]);
        fprintf(out_stream," |");
    }
    fprintf(out_stream,"\n");
    
    fprintf(out_stream,"______|___________________|______|");
    if(LINEAR)fprintf(out_stream,"__________|");
    fprintf(out_stream,"________|");
    for(int i=0; i<num_prop;i++)fprintf(out_stream,"_______________|");
    fprintf(out_stream,"\n");
    for(int i=0;i<wstate_actual.size();i++)/*if(fabs(wstate_actual[i])>1e-10)*/{
        fprintf(out_stream,"  %3d |% 18.10f | %.2f |",i,CI->E_state(i),CI->S2_state(i));
        if(LINEAR){
            fprintf(out_stream," %.2f",CI->L2_state(i));
            if(CI->L2_state(i)<1e-3){
                if     (fabs(CI->P_state(i)-1.0)<1e-8)printf(" (+)");
                else if(fabs(CI->P_state(i)+1.0)<1e-8)printf(" (-)");
                else                               printf(" (?)");
            }
            else
                printf("    ");
            fprintf(out_stream," |");
        }
        fprintf(out_stream," %.2f   |",wstate_actual[i]);
        for(int j=0; j<num_prop;j++)fprintf(out_stream," % 5e |",P[j][i*n_s+i]);
        fprintf(out_stream,"\n",wstate_actual[i]);
    
    }
    
    fprintf(out_stream,"______|___________________|______|");
    if(LINEAR)fprintf(out_stream,"__________|");
    fprintf(out_stream,"________");
    for(int i=0; i<num_prop;i++)fprintf(out_stream,"________________");
    fprintf(out_stream,"|\n");
    
    return 0;
}



CAS_engine::~CAS_engine(){
    
    if(F_core_AO  != NULL)delete[] F_core_AO  ;
    if(F_core_MO  != NULL)delete[] F_core_MO  ;
    if(F_act       != NULL)delete[] F_act       ;
    if(Lambda_act  != NULL)delete[] Lambda_act  ;
    if(act_INTS    != NULL)delete[] act_INTS    ;
    if(aaag_ints  != NULL)delete[] aaag_ints  ;
    if(aaaa_ints  != NULL)delete[] aaaa_ints  ;
    if(aa_J_ints  != NULL)delete[] aa_J_ints  ;
    if(aa_K_ints  != NULL)delete[] aa_K_ints  ;
    if(cv_J_ints  != NULL)delete[] cv_J_ints  ;
    if(cv_K_ints  != NULL)delete[] cv_K_ints  ;
    if(F_tot      != NULL)delete[] F_tot      ;
    if(gamma      != NULL)delete[] gamma      ;
    if(G_ga       != NULL)delete[] G_ga       ;
    if(GAMMA      != NULL)delete[] GAMMA      ;
    if(G          != NULL)delete[] G          ;
    if(B          != NULL)delete[] B          ;
    if(S_track    != NULL)delete[] S_track    ;
    if(MO_BUF     != NULL)delete[] MO_BUF     ;
//     if(MO_VEC     != NULL)delete[] MO_VEC     ;
    if(GEN_CVEC   != NULL)delete[] GEN_CVEC   ;
    if(ACT_CVEC   != NULL)delete[] ACT_CVEC   ;
    if(DM_C       != NULL)delete[] DM_C       ;
    if(DM_A       != NULL)delete[] DM_A       ;
    if(J          != NULL)delete[] J          ;
    if(K          != NULL)delete[] K          ;
    if(H          != NULL)delete[] H          ;
    if(Prop_AO    != NULL)delete[] Prop_AO    ;
    if(Prop_Act   != NULL)delete[] Prop_Act   ;
    if(Prop_Core  != NULL)delete[] Prop_Core  ;
    if(Prop_value != NULL)delete[] Prop_value ;
    if(Prop_nuc   != NULL)delete[] Prop_nuc   ;
    
    
}


int CAS_SCF(molecule * M, cas_par * cas, char * job_name){
    
    dmrg_log_set_job(job_name);
    M->CI[0].PT2_delete_data();
    
    cas->write_info(M->n_act_el_alp[0],
                    M->n_act_el_bet[0],
                    M->n_act_orb   [0],
                    M->n_cor_orb,
                    M->CI[0].mult);
    
    if(RI){
        gen_RI_AA(M);
    }
    
    M->mc=1;//set multi-configuration
    
    int n_dav_conv;
    double max_grad_el;
    int n_iter=0;
    double E_old=0;
    double E;
    
    int converged=0;
    double rot_step=1.0;
    double grad_old=0;      // max|g| at the point the current rot_step was taken from
    bool any_maxed=false;   // a macro-iter whose CI solve hit its max sweeps while under-converged
    bool any_cold=false;    // a macro-iter whose CI solve fell back to a cold start
    bool any_reset=false;   // a macro-iter whose energy rise restarted the orbital converger
    
    if(IS_SYM){
        int n_ao  = M->n_ao;
        int n_act = M->n_act_orb[0];
        int n_cor = M->n_cor_orb;
        int n_mo  = n_ao;
        if(D5){
            M->calc_d5_n_ao();
            n_mo = M->n_ao_d5;
        }
        int n_virt = n_mo-n_cor-n_act;
        
        M->sort_orbs_by_rep(0,n_cor);
        M->sort_orbs_by_rep(n_cor,n_cor+n_act);
        M->sort_orbs_by_rep(n_cor+n_act,n_cor+n_act+n_virt);
    }
    
    //CAS engine reading from A state 0 fragment 0
    if(M->CAS!=nullptr)delete [] M->CAS;
    M->CAS = new CAS_engine[1];
    CAS_engine * CAS = M->CAS;
    CAS->init(cas ,M);
    CAS->SCF_alloc();
    
    //SOSCF engine
    soscf_engine_MCSCF SOSCF;
    SOSCF.N_LBFGS_VECTORS=20;
    SOSCF.init(std::min(CAS->n_ao,std::min(SOSCF.N_LBFGS_VECTORS,cas->max_it)),CAS->n_core, CAS->n_act,CAS->n_vac,CAS->n_ao,cas->x_max);
    
    jacobi_mcscf_sd_engine j_sd;
    j_sd.init(CAS->G,CAS->B,CAS->n_core, CAS->n_act,CAS->n_vac,CAS->n_ao,CAS->n_s_opt,cas->x_max);
    
    dmrg_log_set_tag(dmrg_log_tag::primary);
    n_dav_conv = CAS->CI_calc(1,0,0);
    
    // U_loc belongs to the active orbitals the solve ran on, so dump before canonicalization moves
    // them: after it the two no longer describe the same orbitals.
    if(cas->dmrg.dump_loc_orbs && CAS->localizer_){
        fprintf(out_stream,"\nDumping localized active orbitals:\n");
        M->LOC_print(job_name, CAS->U_loc.data());
    }
    
    CAS->make_canonical(/*with_active=*/false);
    printf_timer("Primary CI and calculation canonical orbitals");

    
    
    CAS->print_av_table("CAS_SCF density averaging:");
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Start CAS_SCF iterations\n");
    // A DMRG solve reports sweeps, a sweep-to-sweep energy and a discarded weight where a
    // determinant CI reports Davidson iterations and nothing else; the header follows the backend.
    const bool dmrg_ci = (cas->ci_solver==CISOLVER_DMRG);
    const char * ni_head = dmrg_ci ? " N_SWP |" : " N_dav |";
    const char * de_rule = dmrg_ci ? "___________|" : "";
    const char * de_head = dmrg_ci ? " DMRG_DE   |" : "";
    const char * dw_rule = dmrg_ci ? "___________|" : "";
    const char * dw_head = dmrg_ci ? " DMRG_DW   |" : "";
    // The CI backend's lattice order is pinned across warm solves, so its staleness is a run diagnostic.
    const bool ord_col = (cas->ci_solver==CISOLVER_DMRG &&
                          (cas->dmrg.loc_order==DMRG_LOCORDER_FIEDLER ||
                           cas->dmrg.loc_order==DMRG_LOCORDER_GAOPT));
    const char * od_rule = ord_col ? "___________|" : "";
    const char * od_head = ord_col ? " OPTIM.LAT |" : "";
    fprintf(out_stream,"______________________________________________________________________");
    if(dmrg_ci)fprintf(out_stream,"________________________");
    if(ord_col)fprintf(out_stream,"____________");
    fprintf(out_stream,"\n");
    fprintf(out_stream,"  N | E                 | dE         | LAG.ASYM. | ROT.STEP  |%s%s%s%s\n",ni_head,de_head,dw_head,od_head);
    fprintf(out_stream,"____|___________________|____________|___________|___________|_______|%s%s%s\n",de_rule,dw_rule,od_rule);
    disable_print_timers();
    
    while(true){
        if(n_iter>cas->max_it-1){converged=0; break;}
        
        if(cas->method==1)E = CAS->SA_grad_hess_calc();
        if(cas->method==2)E = CAS->SM_grad_hess_calc();
        
        if(cas->method==1)max_grad_el = SOSCF.calc(CAS->G,CAS->B);
        if(cas->method==2)max_grad_el = j_sd.find_max_el();
        
        
        // An energy rise nothing accounts for: an honest step moves the energy by |g||kappa| to first
        // order, and a CI solve resolves it only to a fraction of its truncation energy and to what
        // its stop thresholds bound. Above all three the CI solution itself moved, and the amplitude
        // behind it is not an error vector the history can fit. Restart, as SCF DIIS does.
        bool cold_fb = CAS->CI->last_solve_cold();
        if(cold_fb) any_cold=true;
        double step_ref = rot_step;                                    // SOSCF returns the step it applied
        const double ci_floor  = CAS_RESET_TRUNC_FRAC*CAS->CI->last_solve_trunc_de();
        const double ci_res    = CAS->CI->energy_resolution();
        double rise_ref = grad_old*step_ref;
        if(ci_floor>rise_ref) rise_ref = ci_floor;
        if(ci_res  >rise_ref) rise_ref = ci_res;
        const bool diis_reset = (n_iter>0 && !cold_fb && E-E_old>0 && E-E_old > rise_ref); // a cold solve's rise is expected, its reset done
        if(diis_reset){
            SOSCF.reset_history();
            any_reset=true;
        }
        bool hit_max = CAS->CI->last_solve_hit_max();
        if(hit_max) any_maxed=true;
        char de_val[16]; de_val[0]='\0';
        if(dmrg_ci)snprintf(de_val,sizeof(de_val)," %.3e |",CAS->CI->last_solve_resid());
        char dw_val[16]; dw_val[0]='\0';
        if(dmrg_ci){
            const double dw = CAS->CI->last_solve_dw();
            if(std::isnan(dw))snprintf(dw_val,sizeof(dw_val),"     -     |"); // backend never truncates
            else              snprintf(dw_val,sizeof(dw_val)," %9.2e |",dw);
        }
        char od_val[16]; od_val[0]='\0';
        if(ord_col){
            const double od = CAS->CI->last_order_drift(); // FALSE: a cheaper lattice order is in hand
            if(std::isnan(od))snprintf(od_val,sizeof(od_val),"     -     |"); // nothing pinned to price, or a cold re-pin dropped it
            else snprintf(od_val,sizeof(od_val)," %9s |",od>DMRG_ORD_DRIFT_TOL?"FALSE":"TRUE");
        }
        fprintf(out_stream,"%3d |% 18.10f | % .3e | %.3e | %.3e | %3d   |%s%s%s%s%s%s\n",n_iter,E,E-E_old,max_grad_el, rot_step,n_dav_conv, de_val, dw_val, od_val, hit_max?" *":"", cold_fb?" c":"", diis_reset?" r":"");
        fflush(out_stream);
//         getchar();
//         exit(0);
        if(fabs(E-E_old)<cas->e_conv){converged=1; break;}
        if(max_grad_el  <cas->g_conv){converged=2; break;}
        
        if(cas->method==1)rot_step=SOSCF.step(CAS->MO_VEC,CAS->MO_BUF);
        if(cas->method==2)rot_step=j_sd.step(CAS->MO_VEC,CAS->MO_BUF);
//         printf_timer("CAS_step");
//         converged=0; break;
        dmrg_log_set_tag(dmrg_log_tag::scf);
        n_dav_conv =CAS->CI_calc(0,0,1);
        // That solve rebuilt the wavefunction from scratch: the energy surface the converger's
        // history was accumulated on is gone, so extrapolating across it fits a defunct surface.
        const bool cold_now = CAS->CI->last_solve_cold();
        if(cold_now){
            SOSCF.reset_history();
            any_cold=true;  // this solve may never reach a row of its own
        }
        // rot_step was measured against a surface that no longer exists; judge it one iteration
        // later, once the rebuilt wavefunction has an energy and a gradient of its own.
        if(!cold_now && rot_step  <cas->s_conv){converged=3; break;}
                
//         printf_timer("CAS-CI");
        E_old=E;
        grad_old=max_grad_el;
        n_iter++;
//         PrintMatr(M->nat_orb_occ,M->n_act_orb[0],1,1);
    }
    enable_print_timers();
    set_zero_matr(M->orb_energy,M->n_ao);
    
    char * name = new char[BUF_LINE_LENGTH];
    
    fprintf(out_stream,"____|___________________|____________|___________|___________|_______|%s%s%s\n",de_rule,dw_rule,od_rule);
    if(converged==0)fprintf(out_stream,"\nCASSCF did not converge");
    if(converged==1)fprintf(out_stream,"\nEnergy converged");
    if(converged==2)fprintf(out_stream,"\nLagrangian converged");
    if(converged==3)fprintf(out_stream,"\nRotation matrix converged");
    fprintf(out_stream," after %d iterations\n\n", n_iter);
    if(any_maxed)
        fprintf(out_stream," * CI solve reached the maximum DMRG sweep count without meeting sweep_tol;\n"
                           "   that iteration's CI vector may be under-converged -- raise $DMRG sweeps or m.\n\n");
    if(any_cold)
        fprintf(out_stream," c CI solve fell back to a cold start: the wavefunction was rebuilt from\n"
                           "   scratch, so the energy steps there and the orbital converger was reset.\n\n");
    if(any_reset)
        fprintf(out_stream," r energy rose by more than the applied rotation and the CI resolution account\n"
                           "   for, consistent with that CI solve landing on a different solution: the\n"
                           "   converger history was restarted.\n\n");
    printf_timer("CAS_SCF iterations");
    
    
    //calculate canonical orbitals
    CAS->make_canonical(/*with_active=*/true);
    
    if(LINEAR)CAS->rotate();
    
    printf_timer("Preparation of canonical orbitals");
    
    fprintf(out_stream,"CAS_SCF WaveFunctions:\n\n");
    CAS->CI->gen_ext_ind();
    CAS->CI->print_states(0,CAS->n_s,1);

    CAS->print_av_table("CAS_SCF density averaging:");
    
    cas->w_state=CAS->wstate_actual;
        
    
    fprintf(out_stream,"\n\nCAS_SCF Energy summary:\n");
    PrintEnergy(CAS->CI->E_states_ptr(),CAS->n_s, 1);
    
    CAS->print_properties("CAS_SCF");

    fprintf(out_stream,"\n\n");
    if(write_orbs){
        fprintf(out_stream,"Writing CAS_SCF canonical orbitals:\n");

        M->MO_gamess_format();
        sprintf(name,"%s_CAS.out\0",job_name);
        M->GAMESS_type_out_print(name,-1);
        fprintf(out_stream,"visualization file: %s\n",name);
        
        sprintf(name,"%s_CAS.orb\0",job_name);
        M->MO_print(name);
        fprintf(out_stream,"data file         : %s\n",name);
        fprintf(out_stream,"\n");
        
        fprintf(out_stream,"Writing CAS_SCF natural orbitals:\n");
        
        sprintf(name,"%s_CAS\0",job_name);
        M->NO_print(name, 'T');
        
        fprintf(out_stream,"\n");
    }
    if(write_ci){
        if(cas->ci_solver == CISOLVER_DMRG){
            fprintf(out_stream,"CI/MPS wavefunction output not supported by the DMRG backend -- skipped\n");
        } else {
            fprintf(out_stream,"Writing CAS_SCF WaveFunctions:\n");
            sprintf(name,"%s_CAS.ci\0",job_name);
            CAS->CI->write_civec(0, name);
            fprintf(out_stream,"data file         : %s\n",name);
        }
    }
    
    
    
    fprintf(out_stream,"_______________________________________________________________________\n");
    fprintf(out_stream,"\n");
    fprintf(out_stream,"\n");
    printf_timer("CAS_SCF");

    // A post-SCF module reuses this engine with a solver of its own, already in the canonical
    // basis, so no rotation may be left owed to its RDM pulls.
    CAS->U_pending.clear();

    delete[] name;
//     delete[] S_track;
    
//     libint2::finalize();
    
    return 0;
}

