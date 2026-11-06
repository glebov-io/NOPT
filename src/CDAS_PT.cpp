# include "blas_link.h"
# include "libint_link.h"
# include "matr.h"
# include "timer.h"
# include "defaults.h"
# include "doCI_data.h"
# include "PT_tensors.h"
# include "RI.h"
# include "inp_out.h"
# include "XMCQDPT.h"
# include "CAS.h"
# include "localizer.h"
# include "cdas_gno.h"
# include "dmrg_log.h"         // per-solve block2 sweep log
#ifdef NOPT_HAS_BLOCK2
# include "block2_casci_wrap.h"
#endif

extern int num_threads;

// extern int testing;

inline double ED_with_shift(double E, double edshift){
    
    return E/(E*E+edshift);
}

int copy_MO_to_CVEC(double * V,
                    int d1, int d2, int d3, int d_all,
                    double * O1, double * O2, double* O3){
    
    
    for(int j=0;j<d_all;j++)
    for(int i=0;i<d1;i++)
        O1[j*d1+i]=V[i*d_all+j];
    
    for(int i=0;i<d2   ;i++)
    for(int j=0;j<d_all;j++)
        O2[j*d2+i]=V[(i+d1)*d_all+j];

    for(int j=0;j<d_all;j++)
    for(int i=0;i<d3   ;i++)
        O3[j*d3+i]=V[(i+d1+d2)*d_all+j];
    
       
    return 1;
    
}

int CDAS_PT2(molecule * M, cdas_par * cdas, char * job_name){
    
    dmrg_log_set_job(job_name);
    dmrg_log_set_tag(dmrg_log_tag::cdas);

    
    if(RI==0){
        fprintf(out_stream,"WARNING: RI=0 is not supported for any PT\n");
        exit(0);
    }
    if(RI)gen_RI_AA(M);

    cdas->write_info(M->n_act_el_alp[0],
                   M->n_act_el_bet[0],
                   M->n_act_orb   [0],
                   M->CI[0].mult);
    if(fabs(cdas->edshift)>1E-8)
        fprintf(out_stream,"\n\nWARNING: nonzero ISA denominator shift is not typical for CDAS-PT2\n\n");
    
//     libint2::initialize();
    
    char * name = new char[BUF_LINE_LENGTH];
    
    int n_s=cdas->cas->n_s;
    
    int n_ao  = M->n_ao;
    int n_act = M->n_act_orb[0];
    int n_cor = M->n_cor_orb;
    int n_mo  = n_ao;
    if(D5){
        M->calc_d5_n_ao();
        n_mo = M->n_ao_d5;
    }
    int n_virt = n_mo-n_cor-n_act;
    

    if(RI)RI_core_realloc(n_cor+n_act, n_ao);
    
    
    if(cdas->have_eps){
        if(n_act!=cdas->eps.size()){
            fprintf(out_stream,"ERROR:wrong dim of eps(%d) != n_act(%d)\n",cdas->eps.size(),n_act);
            exit(1);
        }
    }
    if(cdas->orb_e){
        if((cdas->n_orb<1)||(cdas->n_orb>(n_act+1))){
            fprintf(out_stream,"ERROR: could not use orbital %d for active space of size %d\n",cdas->n_orb,n_act);
            exit(0);
        }
    }
    
    
    double * eps = M->orb_energy;
    
    //transposed vectors for calculation of 2e integrals
    double * COR_VEC;//cor
    double * ACT_VEC;//active
    double * VIRT_VEC;//virtual
    
    
    COR_VEC = new double[n_ao * n_cor];
    ACT_VEC = new double[n_ao * n_act];
    VIRT_VEC = new double[n_ao * n_virt];
    
    double * J = new double[n_ao*n_ao];
    double * K = new double[n_ao*n_ao];
    
    double * act_INTS   = new double[n_act*n_act*n_act*n_act];
    
    double * d_x1= new double[n_s*n_s];
    double * d_y1= new double[n_s*n_s];
    double * d_z1= new double[n_s*n_s];
    double * d_x_CV = new double[n_cor*n_virt];
    double * d_y_CV = new double[n_cor*n_virt];
    double * d_z_CV = new double[n_cor*n_virt];
    double * d_x_AV = new double[n_act*n_virt];
    double * d_y_AV = new double[n_act*n_virt];
    double * d_z_AV = new double[n_act*n_virt];
    double * d_x_CA = new double[n_cor*n_act];
    double * d_y_CA = new double[n_cor*n_act];
    double * d_z_CA = new double[n_cor*n_act];
        
    //H1 
    double * H_AV;
    double * H_AA;
    double * H_CA;
    double * H_CV;
    H_AV = new double[n_act*n_virt];
    H_AA = new double[n_act*n_act];
    H_CA = new double[n_cor*n_act];
    H_CV = new double[n_cor*n_virt];
    
    
    
    CAS_engine * CAS = M->CAS;
    double E_core = CAS->CI->E_core();
       
    fflush(out_stream);
    copy_MO_to_CVEC(M->MO_VEC,n_cor,n_act, n_virt,n_ao,COR_VEC,ACT_VEC,VIRT_VEC);
        
    double * eps_a = eps   + n_cor;
    double * eps_e = eps_a + n_act;

    //active-space localization: the $DMRG localize option, DMRG backend only
    int localize_act = (cdas->cas->ci_solver==CISOLVER_DMRG)&&(cdas->cas->dmrg.localize==DMRG_LOC_PM);
    if((cdas->cas->ci_solver!=CISOLVER_DMRG)&&(cdas->cas->dmrg.localize==DMRG_LOC_PM))
        fprintf(out_stream,"NOTE: active-space localization is a DMRG feature -- ignored for cisolver=aldet\n\n");

    fprintf(out_stream,"\n");
    fprintf(out_stream,"Orbital energies (3 blocks):\n");
    fprintf(out_stream,"core     :");fPrintMatr(out_stream,eps  ,1,n_cor,0);
    fprintf(out_stream,"active F :");fPrintMatr(out_stream,eps_a,1,n_act,0);
    if(cdas->have_eps){
        for(int i=0; i<n_act;i++)eps_a[i]=cdas->eps[i];
    }
    if(cdas->HOMO){
        for(int i=0; i<n_act;i++)eps_a[i]=eps_a[M->n_act_el_alp[0]-1];
    }
    if(cdas->sing_e){
        for(int i=0; i<n_act;i++)eps_a[i]=cdas->eps[0];
    }
    if(cdas->orb_e){
        for(int i=0; i<n_act;i++)eps_a[i]=eps_a[cdas->n_orb-1];
    }
    
    if(cdas->have_eps+cdas->HOMO+cdas->sing_e+cdas->orb_e){
        fprintf(out_stream,"active PT:");fPrintMatr(out_stream,eps_a,1,n_act,0);
    }
    
    // block2_casci_wrap DMRG(n_act, M->CI[0].na, M->CI[0].nb, M->CI[0].mult, n_s, M->CI[0].print_number, cas->dmrg);
    // DMRG->init_state_storage(n_s,0);

    //the localized actives belong to the PT stage; the molecule gets its own back before return
    double * ACT_MO_save = nullptr;
    int loc_realized = 0;   //localization actually applied; a fallback to U=I leaves it 0
    std::vector<double> U_loc_keep;   //the localizing rotation, outliving the localizer block
    if(localize_act){
        pm_localizer localizer(*M);

        U_loc_keep.assign((size_t)n_act*n_act,0.0);
        double * U_loc = U_loc_keep.data();

        //U=I fallback of a localizer that did not converge or could not be built; the run stays
        //in the delocalized basis, so nothing is applied
        loc_result lr = localizer.localize(ACT_VEC, n_ao, n_act, nullptr, U_loc);
        if(!lr.converged)
            fprintf(out_stream,"NOTE: active-space localization unavailable; running delocalized\n");
        else{
            double * TMP_MO = new double[M->n_act_orb[0]*M->n_ao];
            double * ACT_MO = M->MO_VEC+M->n_cor_orb*M->n_ao;
            ACT_MO_save = new double[M->n_act_orb[0]*M->n_ao];
            memcpy(ACT_MO_save, ACT_MO, sizeof(double)*M->n_act_orb[0]*M->n_ao);
            nopt_par_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                           M->n_act_orb[0], M->n_ao, M->n_act_orb[0], 1.0,
                           U_loc, M->n_act_orb[0],
                           ACT_MO, n_ao,0.0,
                           TMP_MO,n_ao);

            memcpy(ACT_MO, TMP_MO, sizeof(double)*M->n_act_orb[0]*M->n_ao);
            delete[] TMP_MO;
            copy_MO_to_CVEC(M->MO_VEC,n_cor,n_act, n_virt,n_ao,COR_VEC,ACT_VEC,VIRT_VEC);

            loc_realized = 1;
            fprintf(out_stream,"active-space localization converged; running localized\n");

            M->MO_gamess_format();
            sprintf(name,"%s_DMRG_CDAS_loc.out\0",job_name);
            M->GAMESS_type_out_print(name,-1);
            fprintf(out_stream,"visualization file: %s\n",name);
        }
    }
    fprintf(out_stream,"vacant   :");fPrintMatr(out_stream,eps_e,1,n_virt,0);
    fprintf(out_stream,"\n\n");
        
    
    
    printf_timer("Fock matrix calculation");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
    CAS->calc_DM_C();
    
    double * H_core = new double[n_ao*n_ao];;
    
    set_zero_matr(act_INTS,n_act*n_act*n_act*n_act);
    set_zero_matr(J ,n_ao*n_ao);
    set_zero_matr(K ,n_ao*n_ao);
    if(RI==0)
        calc_2el_MO_INTS(M->s, n_ao, CAS->DM_C, J, K, act_INTS, ACT_VEC, ACT_VEC, n_act);
    else
        calc_2el_MO_INTS_RI(   n_ao, CAS->DM_C, J, K, act_INTS, M->MO_VEC, M->MO_VEC+n_ao*n_cor, M->MO_VEC+n_ao*n_cor, n_cor, n_act,0);
    
    for(int i=0;i<n_ao*n_ao;i++)H_core[i]=M->H_AO[i]+2*J[i]-K[i];//-F_backup[i];
    
    
    
    
    set_zero_matr(d_x1, n_s*n_s);
    set_zero_matr(d_y1, n_s*n_s);
    set_zero_matr(d_z1, n_s*n_s);    
    
    transform_from_col_MO(d_x_CV, M->Dx_AO, n_ao, COR_VEC, n_cor, VIRT_VEC, n_virt);
    transform_from_col_MO(d_y_CV, M->Dy_AO, n_ao, COR_VEC, n_cor, VIRT_VEC, n_virt);
    transform_from_col_MO(d_z_CV, M->Dz_AO, n_ao, COR_VEC, n_cor, VIRT_VEC, n_virt);
    
    transform_from_col_MO(d_x_AV, M->Dx_AO, n_ao, ACT_VEC, n_act, VIRT_VEC, n_virt);
    transform_from_col_MO(d_y_AV, M->Dy_AO, n_ao, ACT_VEC, n_act, VIRT_VEC, n_virt);
    transform_from_col_MO(d_z_AV, M->Dz_AO, n_ao, ACT_VEC, n_act, VIRT_VEC, n_virt);
    
    transform_from_col_MO(d_x_CA, M->Dx_AO, n_ao, COR_VEC, n_cor, ACT_VEC, n_act);
    transform_from_col_MO(d_y_CA, M->Dy_AO, n_ao, COR_VEC, n_cor, ACT_VEC, n_act);
    transform_from_col_MO(d_z_CA, M->Dz_AO, n_ao, COR_VEC, n_cor, ACT_VEC, n_act);
    
    
    
    //H1
    transform_from_col_MO(H_AV, H_core, n_ao, ACT_VEC, n_act, VIRT_VEC, n_virt);
    transform_from_col_MO(H_CA, H_core, n_ao, COR_VEC, n_cor, ACT_VEC, n_act);
    transform_from_col_MO(H_CV, H_core, n_ao, COR_VEC, n_cor, VIRT_VEC, n_virt);
    
    for(int i=0;i<n_ao*n_ao;i++)H_core[i]=M->H_AO[i]+2*J[i]-K[i];
    transform_from_col_MO(H_AA, H_core, n_ao, ACT_VEC, n_act, ACT_VEC, n_act);
    
    fprintf(out_stream,"Start calculation of PT2 correction\n");
//     RI=1;
    RI_data R;
    R.set_par(M,n_cor, n_act, n_virt);
    R.MO_calc(M->MO_VEC, n_ao);
//     RI=RI_backup;
    fprintf(out_stream,"\n");
    fprintf(out_stream,"\n");
    if(RI)printf_timer("RI orbital transformation");
    fprintf(out_stream,"\n");
    fflush(out_stream);
    
    
    //PT — declaration only (ctor nulls pointers); the n_a^6 tables are
    //allocated below by set_par.
    PT_tensors T;

    //the PT stage may drive its own solver; the engine gets its original back before return
    casci_solver * CI_engine = CAS->CI;
#ifdef NOPT_HAS_BLOCK2
    std::unique_ptr<block2_casci_wrap> DMRG;
#endif

    //the ensemble is read off the CAS-stage engine, so it is taken before the PT stage builds its own
    gno_ensemble_data gno_ens;
    if(cdas->gno.on())
        gno_ensemble(CAS, cdas, loc_realized, loc_realized?U_loc_keep.data():nullptr,
                     cdas->gno.skip_scalar==0, gno_ens);

    if(CAS->CI->as_aldet()!=nullptr){
        CAS->CI->as_aldet()->simple_import_data(act_INTS, act_INTS, H_AA, E_core);
    }
    else{
#ifdef NOPT_HAS_BLOCK2
        //the PT stage solves in the orbitals rotated above, so the solver keeps that frame
        dmrg_par pt_dmrg = cdas->cas->dmrg;
        pt_dmrg.localize = DMRG_LOC_OFF;
        //the PT stage may be given a bond dimension of its own
        if(cdas->m_cdas_set) pt_dmrg.m = cdas->m_cdas;
        if(ACT_MO_save!=nullptr)
            fprintf(out_stream,"NOTE: the PT stage runs in the localized orbitals -- solver-internal localization is off\n\n");
        DMRG = std::make_unique<block2_casci_wrap>(n_act, M->CI[0].na, M->CI[0].nb, M->CI[0].mult, n_s, M->CI[0].print_number, pt_dmrg);
        //the weights the internally averaged RDMs carry, the same the driver averages with
        DMRG->set_state_weights(cdas->cas->w_state.data(), n_s);
        DMRG->import_integrals(act_INTS, H_AA, E_core);
        CAS->CI=DMRG.get();
        if(cdas->gno.on() && !CAS->CI->supports_operator_handles()){
            fprintf(out_stream,"ERROR: cdas_mode=trunc_gno|delta_gno needs a backend with named operator handles; use cisolver=dmrg\n");
            exit(EXIT_FAILURE);
        }
        //the bare solution is read by the IP/EA construction only, and the GNO stage builds
        //those matrices from the CAS-stage ensemble instead
        if(cdas->IPEA && !cdas->gno.on()) CAS->CI->solve(1,0,false);
#else
        fprintf(out_stream,"ERROR: CISOLVER=dmrg selected, but this build was compiled without block2 (set USE_BLOCK2=yes)\n");
        exit(EXIT_FAILURE);
#endif
    }
    
    T.set_par(&R, eps, n_cor, n_act, n_virt, H_AV, H_CA, H_CV, cdas->edshift, M);
    if(cdas->IPEA){
        if(cdas->gno.on())
            T.IPEA(H_AA, act_INTS, gno_ens.gamma.data(), gno_ens.GAMMA.data());
        else
            T.IPEA(CAS->CI,cdas->cas->w_state);
        T.E2_calc_IPEA();
    }
    else if(cdas->MPPT){
        if(CAS->CI->as_aldet()==nullptr){
            fprintf(out_stream,"ERROR: MPPT is supported by the aldet solver only; use cisolver=aldet\n\n");
            exit(1);
        }
        T.MPPT(CAS->CI->as_aldet(), 0,cdas->cas->w_state);
        T.E2_calc_EE();
    }
    else{
        T.E2_calc_EE();
    }

    printf_timer("PT tensors calculation");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");


    //every buffer this routine owns, plus the engine and localized-MO restores
    auto release = [&](){

        delete[] H_CV;
        delete[] H_CA;
        delete[] H_AA;
        delete[] H_AV;
        delete[] H_core;
        delete[] COR_VEC;
        delete[] ACT_VEC;
        delete[] VIRT_VEC;

        delete[] d_x1  ;
        delete[] d_y1  ;
        delete[] d_z1  ;
        delete[] d_x_CV;
        delete[] d_y_CV;
        delete[] d_z_CV;
        delete[] d_x_AV;
        delete[] d_y_AV;
        delete[] d_z_AV;
        delete[] d_x_CA;
        delete[] d_y_CA;
        delete[] d_z_CA;

//         delete[] eps_EA;

        delete[] name;

        delete[] J       ;
        delete[] K       ;
        delete[] act_INTS ;

        CAS->CI = CI_engine;

        if(ACT_MO_save != nullptr){
            memcpy(M->MO_VEC+M->n_cor_orb*M->n_ao, ACT_MO_save, sizeof(double)*M->n_act_orb[0]*M->n_ao);
            delete[] ACT_MO_save;
        }

    };

    if(cdas->gno.on()){
        cdas_gno_run(CAS, T, cdas, gno_ens, H_AA, act_INTS, E_core);
        release();
        return 0;
    }

    CAS->CI->PT2_import_data(T.RF_P3_JK,
                T.RF_P3_AB,
                T.RF_PV_JK,
                T.RF_PV_AB,
                T.RF_PH,
                T.RF_PS);
    
    CAS->CI->solve(1,1,true);
    if(LINEAR)CAS->rotate();

//    }
    fprintf(out_stream,"\n\nCDAS-PT2 Energy summary:\n");
    PrintEnergy(CAS->CI->E_states_ptr(),CAS->n_s,1);
    
    double * print_d[3];
    print_d[0]=CAS->Prop_value                  ;
    print_d[1]=CAS->Prop_value+CAS->n_s*CAS->n_s  ; 
    print_d[2]=CAS->Prop_value+CAS->n_s*CAS->n_s*2;
    
    char * print_n[3];
    print_n[0]=new char[BUF_LINE_LENGTH];sprintf(print_n[0],"     d_x     ");
    print_n[1]=new char[BUF_LINE_LENGTH];sprintf(print_n[1],"     d_y     "); 
    print_n[2]=new char[BUF_LINE_LENGTH];sprintf(print_n[2],"     d_z     ");
    
    
    CAS->Prop_calc();
    CAS->print_av_table_with_prop("CDAS-PT2 extended results:",3,print_d, print_n);
    
    fprintf(out_stream,"\n");
    if(write_ci){
         if(CAS->CI->as_aldet()==nullptr){
             fprintf(out_stream,"CI/MPS wavefunction output not supported by the DMRG backend -- skipped\n");
         } else {
             fprintf(out_stream,"Writing CDAS-PT2 WaveFunctions:\n");
             sprintf(name,"%s_CDAS.ci\0",job_name);
             CAS->CI->as_aldet()->write_civec(0, name);
             fprintf(out_stream,"data file         : %s\n",name);
         }
    }
    

    CAS->print_properties("CDAS-PT(0)");    
    
    if(CAS->CI->as_aldet()==nullptr){
        fprintf(out_stream,"first-order properties are not available with cisolver=dmrg -- skipped\n");
        if(write_ci)fprintf(out_stream,"CI/MPS wavefunction output not supported by the DMRG backend -- skipped\n");
        fprintf(out_stream,"\n");
    }
    else{
        double * d_x = CAS->Prop_value                  ;
        double * d_y = CAS->Prop_value+CAS->n_s*CAS->n_s  ;
        double * d_z = CAS->Prop_value+CAS->n_s*CAS->n_s*2;
    
    
        set_zero_matr(d_x1,n_s*n_s);
        set_zero_matr(d_y1,n_s*n_s);
        set_zero_matr(d_z1,n_s*n_s);
    
        if(print_dipole)if(cdas->pt1_d){
            fprintf(out_stream,"PT1 dipole moment - d(1):\n\n");
            if(cdas->IPEA){
                T.P1_calc_IPEA(d_x1, CAS->CI->as_aldet(), d_x_AV, d_x_CA, d_x_CV, 1, 1, 1);
                T.P1_calc_IPEA(d_y1, CAS->CI->as_aldet(), d_y_AV, d_y_CA, d_y_CV, 1, 1, 1);
                T.P1_calc_IPEA(d_z1, CAS->CI->as_aldet(), d_z_AV, d_z_CA, d_z_CV, 1, 1, 1);
            }
            else{
                T.P1_calc_EE(d_x1, CAS->CI->as_aldet(), d_x_AV, d_x_CA, d_x_CV, 1, 1, 1);
                T.P1_calc_EE(d_y1, CAS->CI->as_aldet(), d_y_AV, d_y_CA, d_y_CV, 1, 1, 1);
                T.P1_calc_EE(d_z1, CAS->CI->as_aldet(), d_z_AV, d_z_CA, d_z_CV, 1, 1, 1);
            }
        
            symmetrization_with_scaling(d_x1,n_s,2.0);
            symmetrization_with_scaling(d_y1,n_s,2.0);
            symmetrization_with_scaling(d_z1,n_s,2.0);
        
            for(int i_s=0;i_s<n_s*n_s;i_s++){
                d_x[i_s] += d_x1[i_s];
                d_y[i_s] += d_y1[i_s];
                d_z[i_s] += d_z1[i_s];
            }
        
            fprintf(out_stream,"\n");
            fprintf(out_stream,"\n\nDipole CDAS(0+1):\n");
            PrintDipole(d_x,d_y,d_z,n_s);
        }

        delete[] print_n[0];
        delete[] print_n[1];
        delete[] print_n[2];
    }

        
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
//         fprintf(out_stream,"%e %e %e\n",M->Dx_nuc,M->Dy_nuc,M->Dz_nuc);
//         exit(0);
#ifdef _TRrCAMM
    double *A_DM = new double[n_ao*n_ao];
    double q;
    for(int i_a=0;i_a<M->n_atoms;i_a++){
        
        gen_atomic_DM(A_DM, M->S_MO, i_a, M->basis[0].ao_center, n_ao);
        for(int i=0;i<n_ao*n_ao;i++)A_DM[i]=-A_DM[i];
        fprintf(out_stream,"TrC:\n");
        for(int i_s=0;i_s<n_s;i_s++){
            for(int j_s=0;j_s<n_s;j_s++){
                D.transform_1el_DM(i_s*n_s+j_s);
                q=E_1el_calc(A_DM, D.DM_T, D.n_ao[0], D.n_ao[1])*D.p_SVD+D.ACT_DET_A[i_s*n_s+j_s]*M->nucl_charges[i_a];
                
                fprintf(out_stream,"% .5e ", q);
            }
            fprintf(out_stream,"\n");
        }
        getchar();
        
        double * q1 = new double[n_s*n_s];
        set_zero_matr(q1, n_s*n_s);
        
        double * q_AV;
        double * q_CA;
        double * q_CV;
        q_AV = new double[n_act*n_virt];
        q_CA = new double[n_cor*n_act];
        q_CV = new double[n_cor*n_virt];
        transform_from_col_MO(q_AV, A_DM, n_ao, ACT_VEC, n_act, VIRT_VEC, n_virt);
        transform_from_col_MO(q_CA, A_DM, n_ao, COR_VEC, n_cor, ACT_VEC, n_act);
        transform_from_col_MO(q_CV, A_DM, n_ao, COR_VEC, n_cor, VIRT_VEC, n_virt);
        
    
        T.P1_calc(q1, q_AV, q_CA, q_CV, 1, 1, 1);
        
        
        symmetrization_with_scaling(q1,n_s,2.0);
        fPrintMatr(out_stream,q1,n_s,n_s,1);
        
        
    }
#endif
        
    printf_timer("CDAS-PT2");
    
    
    release();

    return 0;
 
}
