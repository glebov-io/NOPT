# include "blas_link.h"
# include "energy_result.h"
# include "matr.h"
# include "inp_par_read.h"
# include "timer.h"
# include "doCI_matr.h"
# include "converger_2_1.h"
# include "defaults.h"
# include "RI.h"
# include "sad_guess.h"
# include "grabbers.h"

using std::vector;
// using libint2::Shell;
// using libint2::Engine;
// using libint2::Operator;

extern int num_threads;

// int max_n_iter=100;

double conv_crit=5.0E-7;

double extern svd_eps;

int tr_and_diag_re(double * H, double * M, double * B, double * V, double * E, int dim){
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        H,dim,
                        M,dim,0.0,
                        V,dim);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        M,dim,
                        V,dim,0.0,
                        B,dim);
    
//     fprintf(out_stream,"H_tr:\n");
//     PrintMatr(H,dim,dim,0);
    
    
    lapack_right_eig(B,E,dim);
    
//     for(int i=0;i<dim;i++)
//         fprintf(out_stream,"ev_old[%d] = %.10f (%.10e)\n",i,H[i*(dim+1)]/S[i*(dim+1)],S[i*(dim+1)]);

//     for(int i=0;i<dim;i++)
//         fprintf(out_stream,"ev[%d] = %.10f\n",i,E[i]);

    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        B,dim,
                        M,dim,0.0,
                        V,dim);
    
    
    return 0;
    
    
}

int gen_HF_DM(double * DM, double * MO, int n, int n_el){
    
    set_zero_matr(DM,n*n);
    
    for(int k=0; k<n_el; k++)
    for(int i=0; i<n  ; i++)
    for(int j=0; j<n  ; j++)
        DM[i*n+j]+=MO[k*n+i]*MO[k*n+j];
    
    return 0;
}




int RHF(molecule * M, rhf_par * rhf, char * job_name){
    
    
    double E    =0;
    double E_old=1.0E15;
    
    int iter_n=0;
    int n_ao = M->n_ao;
    int n_mo = M->n_mo;
    int n_cor = M->n_el_calc/2;
    if(M->n_el_calc-2*n_cor){
        printf("ERROR: molecule contains odd number of electrons (n_el = %d)\n", M->n_el_calc);
        printf("       check molecule charge, number of ECP frozen electrons etc.\n");
        exit(1);
    }
    
    
    M->mc=0;//set single configuration
    
    rhf->write_info(n_cor);
    
    double * BUF  = new double[n_ao*n_ao];
    double * BUF2 = new double[n_ao*n_ao];
    double * F_AO = new double[n_ao*n_ao];
    double * DM   = new double[n_ao*n_ao];
    
    
    
    
    double max_grad_el;
    soscf_engine SOSCF;
    SOSCF.N_LBFGS_VECTORS=20;
    SOSCF.init(std::min(n_mo,std::min(SOSCF.N_LBFGS_VECTORS,rhf->max_it)),n_cor,n_mo,n_ao);
    if(RI){
        gen_RI_AA(M);
    }
    int guess_diag=0;
    if(M->guess_occ.size()){
        sad_guess_fock(M,F_AO);
        guess_diag=1;
    }
    fprintf(out_stream,"\n\n");
    fprintf(out_stream,"_______________________________________________________________________________\n");
    fprintf(out_stream,"\n\n");
    fprintf(out_stream,"Starting SCF\n");
    
    disable_print_timers();
    while(true){
        
        E_old=E;
        printf_timer("come to next iter");
        gen_HF_DM(DM, M->MO_VEC, n_ao, n_cor);
        printf_timer("gen DM");
        M->calc_F_AO(F_AO, DM, 1.0);//DM is normalized on 1 in J and K
        printf_timer("calc F");
        
        E=E_1el_calc(F_AO, DM, n_ao, n_ao)+E_1el_calc(M->H_AO, DM, n_ao, n_ao)+M->V_nuc;
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,n_ao,n_mo,n_ao,1.0,F_AO              ,n_ao,M->MO_VEC,n_ao,0.0,BUF2,n_mo); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_mo,n_mo,n_ao,1.0,M->MO_VEC,n_ao,BUF2              ,n_mo,0.0,BUF ,n_mo);

        max_grad_el=SOSCF.calc(BUF);
                
        fprintf(out_stream,"ITER %d\tE=% .10f\tdE=% e\torb_grad = %e\n",iter_n,E,E-E_old,max_grad_el);
        fflush(out_stream);
        
        //case of small gradient - simple diagonalization and exit
        if(max_grad_el<rhf->g_conv){
            fprintf(out_stream,"\nGradient converged after %d iterations\n\n", iter_n);
            break;
        }
        if(fabs(E-E_old)<rhf->e_conv){
            fprintf(out_stream,"\nEnergy converged after %d iterations\n\n", iter_n);
            break;
        }
        if(iter_n>rhf->max_it){
            fprintf(out_stream,"\nRHF did not converge after %d iterations\n\n", iter_n);
            break;
        }
        
        //case of large gradient - simple diagonalization
        //a guess that comes from its own Fock does not need it repeated
        if((iter_n==0)&&(guess_diag==0)/*||(max_grad_el>SOSCF.grad_max_soscf)*/){

            if(iter_n)fprintf(out_stream,"WARNING: gradient is too large\n");
            M->diag_X_AO_in_MO(F_AO);
            iter_n++;
            continue;
        }
        
        //calculation of orbital shift
        SOSCF.step(M->MO_VEC,BUF);
        
        E_old=E;
        iter_n++;

    }
    
    
    enable_print_timers();
    M->diag_X_AO_in_MO(F_AO);
    
    char * name = new char[BUF_LINE_LENGTH];
    
    if(write_orbs){
        M->MO_gamess_format();
    
        fprintf(out_stream,"Writing RHF orbitals:\n");
        sprintf(name,"%s_RHF.out\0",job_name);
        M->GAMESS_type_out_print(name,-1);
        fprintf(out_stream,"visualization file: %s\n",name);
        sprintf(name,"%s_RHF.orb\0",job_name);
        M->MO_print(name);
        fprintf(out_stream,"data file         : %s\n",name);
            
//         M->MO_libint_reordr();
    }
    
    if(grab_E==1){
        E_grabbed.resize(1);
        E_grabbed[0]=E;
    }
    
    if(M->reorder){
        fprintf(out_stream,"\n\n");
        fprintf(out_stream,"WARNING: using CAS orbital reordering  with RHF=y\n");
        fprintf(out_stream,"         reordering is done now\n");
        fprintf(out_stream,"         previous files are written before reordering\n");
        M->reorder_orbitals();
        fprintf(out_stream,"\n\n");
    }
    
    fprintf(out_stream,"\n");
    
    
    printf_timer("RHF converged at");
    energy_result_record("rhf","",-1,E);
  
    delete[] BUF ;
    delete[] BUF2;
    delete[] F_AO;
    delete[] DM  ;
    delete[] name;
    
    
    return 0;
}

#ifdef experimental

int ortvec_calc( double * o_v, double * ref_state, double * cur_state, double * S_AO, int n_ao, int n_el, int cb){
    
    double * SMO12 ;
    double * minor ;
    double * cpu_MO;
    double * TMP   ;
    
    SMO12  = new double[n_el*n_el];
    minor  = new double[(n_el-1)*(n_el-1)];
    cpu_MO = new double[(n_ao)*(n_ao)];
    TMP    = new double[(n_ao)*n_el];
    int i,j,k;
    double tmp;
  
    for(i=0;i<n_ao;i++) o_v[i]=0;

    double a1 = 1.0;
    double b1 = 0.0;
    
//     fprintf(out_stream,"SAO\n");
//     PrintMatr(S_AO,n_ao,n_ao,1);
//     
//     fprintf(out_stream,"R\n");
//     PrintMatr(ref_state,n_ao,n_ao,1);
//     
//     fprintf(out_stream,"C\n");
//     PrintMatr(cur_state,n_ao,n_ao,1);
    
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_ao, n_el, n_ao,1.0,
                        S_AO,n_ao,
                        ref_state,n_ao,0.0,
                        TMP,n_el);
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_el, n_el, n_ao,1.0,
                        cur_state,n_ao,
                        TMP,n_el,0.0,
                        SMO12,n_el);
//     fprintf(out_stream,"S_MO\n");
//     PrintMatr(SMO12,n_el,n_el,1);
    
    for(i=0;i<n_el;i++){
        for(j=0;j<n_el-1;j++)for(k=0;k<i;k++){minor[j*(n_el-1)+k]=SMO12[j*n_el+k];}
        for(j=0;j<n_el-1;j++)for(k=i;k<n_el-1;k++){minor[j*(n_el-1)+k]=SMO12[j*n_el+k+1];}
        tmp=mat_det_calc_lapack(minor,n_el-1);
//         printf(
        if(cb)for(j=0;j<n_ao;j++)o_v[j]+=tmp*ref_state[i*n_ao+j];
        else o_v[i]=tmp;
    }
//     for(j=0;j<n_ao;j++)fprintf(out_stream,"ov[%d] = %e\n",j,o_v[j]);
//     getchar();
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                         n_ao, n_el, n_ao,1.0,
//                         S_AO,n_ao,
//                         ref_state,n_ao,0.0,
//                         TMP,n_el);
//     
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                         1, n_el, n_ao,1.0,
//                         o_v,n_ao,
//                         TMP,n_el,0.0,
//                         SMO12,n_el);
//     
//     fprintf(out_stream,"V^OV[0] - ov_calc\n");
//     PrintMatr(SMO12,n_el,1,1);
    delete[] SMO12;
    delete[] TMP;
    delete[] minor;
    return 0;
}

int ort_diag(double * H, double * M, double * P, double * B, double * V, double * E, double * O, int dim, int n_ort){
    
    if(n_ort==0){
        tr_and_diag(H, M, B, V, E, dim);
        return 0;
    }
            
    int i,j,i_o,i_m;
    
    double * red_matr;
    red_matr = new double[(dim-n_ort)*(dim-n_ort)];
    
    double * tmp_basis;
    tmp_basis=new double[dim*dim];

    //transformation V_tr = S^1/2 * V
    //data are transposed,so
    //V_tr^t = V^t * S^1/2 (S is symmetric)
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        V,dim,
                        P,dim,0.0,
                        tmp_basis,dim);

    
    for(int i=0;i<dim*n_ort;i++)B[i] = O[i];
//     fprintf(out_stream,"OV\n");
//     PrintMatr(B,n_ort,dim,1);
    
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ort,dim,dim,1.0,////??????????????
                        B,dim,
                        P,dim,0.0,
                        O,dim);
            
    double max_cos,cos;
    for(i_o=0;i_o<n_ort;i_o++){
        i_m =dim-1;	
        max_cos=0.0;
        make_norm(O,i_o,dim);
        for(i=0;i<dim-i_o;i++){
                         cos=fabs(make_ort(tmp_basis,i*dim,O,i_o*dim,dim));
                         
//                            fprintf(out_stream,"cos(%d^ort[%d])=%e v=\n",i,i_o,cos);
//                            fprintf(out_stream,"v[%d]=",i);
//                            for(j=0;j<dim;j++)fprintf(out_stream,"%f ",tmp_basis[i*dim+j]);
//                            fprintf(out_stream,"\n");
//                            getchar();
                         if(cos>max_cos){
                                         i_m=i;
                                         max_cos=cos;
                                         }
                         make_norm(tmp_basis,i,dim);                
                         }
//       fprintf(out_stream,"i_max=%d\n",i_m);
//     if(i_m!=(dim-n_ort))fprintf(out_stream,"WARNING:ortogonal vector #%d changed\n",i_o);
//     for(i=0;i<i_m;i++)for(j=0;j<dim;j++)ort_basis[i*dim+j]=tmp_basis[i*dim+j];
    for(i=i_m;i<dim-1-i_o;i++)for(j=0;j<dim;j++)tmp_basis[i*dim+j]=tmp_basis[(i+1)*dim+j];
    for(i=0;i<dim-1-i_o;i++)for(j=0; j<i;j++)/*fprintf(out_stream,"cos(%d^%d)=%e\n",i,j,*/make_ort(tmp_basis,i*dim,tmp_basis,j*dim,dim)/*)*/;
    for(i=0;i<dim-1-i_o;i++)/*fprintf(out_stream,"norm(%d)=%e\n",i,*/make_norm(tmp_basis,i,dim)/*)*/;
    }
    
    
    //F_tr = S^-1/2*F*S^-1/2
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        H,dim,
                        M,dim,0.0,
                        V,dim);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        M,dim,
                        V,dim,0.0,
                        B,dim);
    
//     fprintf(out_stream,"F_tr:\n");
//     PrintMatr(B,dim,dim,1);
    
    //
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        dim, dim-n_ort, dim,1.0,
                        B,dim,
                        tmp_basis,dim,0.0,
                        V,dim);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim-n_ort, dim-n_ort, dim,1.0,
                        tmp_basis,dim,
                        V,dim,0.0,
                        red_matr, dim-n_ort);

    
    lapack_right_eig(red_matr,E,dim-n_ort);
        
    set_zero_matr(B,dim*dim);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim-n_ort, dim, dim-n_ort,1.0,
                        red_matr,dim-n_ort,
                        tmp_basis, dim,0.0,
                        B,dim);
    
    
    for(int i=0;i<n_ort*dim;i++)
        B[(dim-n_ort)*dim+i]=O[i];
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        B,dim,
                        M,dim,0.0,
                        V,dim);
    
    return 0;

}

int ort_F_calc(double * H, double * M, double * P, double * B, double * V, double * B2, double * O, double * tmp_basis, int dim, int n_ort){
    
    if(n_ort==0){
        cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,dim,dim,dim,1.0, H,dim,V , dim, 0.0, B2,dim); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,dim,dim,dim,1.0, V,dim,B2, dim, 0.0, B ,dim);
        return 0;
    }
        
    
    int i,j,i_o,i_m;
    
    
//     double * red_matr;
//     red_matr = new double[(dim-n_ort)*(dim-n_ort)];
    
    
//     double * tmp_basis;
    double * O_tmp    ;
//     tmp_basis=new double[dim*dim  ];
    O_tmp    =new double[dim*n_ort];

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        V,dim,
                        P,dim,0.0,
                        tmp_basis,dim);


    
//     for(int i=0;i<dim*n_ort;i++)B[i] = O[i];
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ort,dim,dim,1.0,////??????????????
                        O,dim,
                        P,dim,0.0,
                        O_tmp,dim);
    
        
    double max_cos,cos;
    for(i_o=0;i_o<n_ort;i_o++){
        i_m =dim-1;	
        max_cos=0.0;
        make_norm(O_tmp,i_o,dim);
        for(i=0;i<dim-i_o;i++){
                         cos=fabs(make_ort(tmp_basis,i*dim,O_tmp,i_o*dim,dim));
//                          fprintf(out_stream,"%e %d\n",i,cos);
                         if(cos>max_cos){
                                         i_m=i;
                                         max_cos=cos;
                                         }
                         make_norm(tmp_basis,i,dim);
        }
//         fprintf(out_stream,"I_m=%d,max_cos=%e\n",i_m, max_cos);
        for(i=i_m;i<dim-1-i_o;i++)for(j=0;j<dim;j++)tmp_basis[i*dim+j]=tmp_basis[(i+1)*dim+j];
//         fprintf(out_stream,"oB1:\n");
//         PrintMatr(tmp_basis,dim,dim,0);  
        
        for(i=0;i<dim-1-i_o;i++)for(j=0; j<i;j++)/*fprintf(out_stream,"cos(%d^%d)=%e\n",i,j,*/make_ort(tmp_basis,i*dim,tmp_basis,j*dim,dim)/*)*/;
//         fprintf(out_stream,"oB2:\n");
//         PrintMatr(tmp_basis,dim,dim,0);  
        
        for(i=0;i<dim-1-i_o;i++)/*fprintf(out_stream,"norm(%d)=%e\n",i,*/make_norm(tmp_basis,i,dim)/*)*/;
        for(i=dim-1-i_o;i<dim;i++)for(j=0;j<dim;j++)tmp_basis[i*dim+j]=O_tmp[j];
    }
    
//     fprintf(out_stream,"oB:\n");
//     PrintMatr(tmp_basis,dim,dim,1);  
    
    //F_tr = S^-1/2*F*S^-1/2
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        H,dim,
                        M,dim,0.0,
                        B2,dim);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                        dim,dim,dim,1.0,
                        M,dim,
                        B2,dim,0.0,
                        B,dim);
    
    //
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        dim, dim-n_ort, dim,1.0,
                        B,dim,
                        tmp_basis,dim,0.0,
                        B2,dim);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        dim-n_ort, dim-n_ort, dim,1.0,
                        tmp_basis,dim,
                        B2,dim,0.0,
                        B, dim-n_ort);

//     fprintf(out_stream,"F_ob:\n");
//     PrintMatr(B,dim-n_ort,dim-n_ort,1);
    
        for(int i=0;i<n_ort*dim;i++)
        tmp_basis[(dim-n_ort)*dim+i]=O_tmp[i];
    
    return 0;

}

int gen_dc_DM(double * Q_ab,
              double * Q_ba,
              double * Va,
              double * Vb, int n_ao,int n_el,
              double * S_AO){

    double * S_MO = new double[n_el*n_el]; // <A|S|B>
    double * BUF  = new double[n_el*n_ao];
    double * A    = new double[n_el*n_ao]; //left
    double * B    = new double[n_el*n_ao]; //right
    double * L_MO = new double[n_el*n_el];
    double * R_MO = new double[n_el*n_el];
    double * SVD  = new double[n_el];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_ao,n_el,n_ao,1.0,
                            S_AO,n_ao,
                            Vb,n_ao,0.0,
                            BUF,n_el);
        
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                            n_el,n_el,n_ao,1.0,
                            Va,n_ao,
                            BUF,n_el,0.0,
                            S_MO,n_el);
    
//     PrintMatr(S_MO,n_el,n_el,0);
//     exit(0);
    lapack_svd(S_MO,R_MO,L_MO,SVD,n_el);
//     PrintMatr(SVD,1,n_el,0);
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                n_el,n_ao,n_el,1.0,
                R_MO,n_el,    
                Vb,n_ao,0.0,
                B,n_ao);
    cblas_dgemm(CblasRowMajor,CblasTrans ,CblasNoTrans,
                n_el,n_ao,n_el,1.0,
                L_MO,n_el,    
                Va,n_ao,0.0,
                A,n_ao);
//     PrintMatr(Va,n_el,n_ao,1);
//     PrintMatr(Vb,n_el,n_ao,1);
//     PrintMatr(A,n_el,n_ao,1);
//     PrintMatr(B,n_el,n_ao,1);
    int n_ext=0;
    int i_ext=-1;
    double S=1.0;
    for(int ii=0;ii<n_el;ii++){
        if(fabs(SVD[ii])>svd_eps)S*=SVD[ii];
        else{
            n_ext++;
            if(n_ext==3)break;
            i_ext=ii;
        }
    }
//     fprintf(out_stream,"n_ext = %d, i_ext =%d S=%e\n",n_ext,i_ext,S);
    if(n_ext!=1){
        fprintf(out_stream,"ERROR: DC_SCF is realized only for n_ext = 1\n");
        exit(0);
    }
    int i,j;
//     fprintf(out_stream,"SYM_ctrl in gen DCDM\n");
// fprintf(out_stream,"%e %e %e %e\n",A[13*180+7],A[13*180+43],A[13*180+79],A[13*180+115]);
// fprintf(out_stream,"%e %e %e %e\n",B[13*180+7],B[13*180+43],B[13*180+79],B[13*180+115]);
// fprintf(out_stream,"%e %e %e %e\n",A[14*180+7],A[14*180+43],A[14*180+79],A[14*180+115]);
// fprintf(out_stream,"%e %e %e %e\n",B[14*180+7],B[14*180+43],B[14*180+79],B[14*180+115]);
// getchar();
//     for(i=0;i<n_ao;i++)for(j=0;j<n_ao;j++)Q_ba[i*n_ao+j]=S*B[i_ext*n_ao+i]*A[i_ext*n_ao+j];
    for(i=0;i<n_ao;i++)for(j=0;j<n_ao;j++)Q_ab[i*n_ao+j]=S*A[i_ext*n_ao+i]*B[i_ext*n_ao+j];
//     for(i=0;i<n_ao;i++)for(j=0;j<n_ao;j++)Q_bb[i*n_ao+j]=S*B[i_ext*n_ao+i]*B[i_ext*n_ao+j];
    
//     PrintMatr(Q_aa,n_ao,n_ao,1);
//     PrintMatr(Q_ab,n_ao,n_ao,1);
//     PrintMatr(Q_bb,n_ao,n_ao,1);
    
    
    
    
    delete[] S_MO;
    delete[] BUF ;
    delete[] A;
    delete[] B;
    
    return 0;
}

int DC_OHF(molecule * A, int n_state, int * ex){
    
        
    libint2::initialize();
    
    double E    =0;
    double E_old=1.0E15;
    
    int n_alp_ex=0;
    int n_bet_ex=0;
    for(int i=0;i<n_state;i++){
        if(ex[i]) n_bet_ex++;
        else      n_alp_ex++;
    }
    fprintf(out_stream,"a = %d\nb = %d\n", n_alp_ex, n_bet_ex);
//     getchar();
    
    int iter_n=0;
    int n_ao = A->basis[n_state].n_ao;
    int n_el = A->n_el[n_state]/2;
    
    
    double V_nuc=0.0;
    V_nuc=E_nuc_calc(A);
    fprintf(out_stream,"V_nuc = %e\n",V_nuc);
    
    Engine  s_engine(Operator::overlap,GTO_MAX,L_MAX);
    Engine  t_engine(Operator::kinetic,GTO_MAX,L_MAX);
//     Engine  d_engine(Operator::emultipole1,GTO_MAX,L_MAX);
//     d_engine.set_params(std::array<double,3>{0.0, 0.0, 0.0});
    
    std::vector<libint2::Engine> v_engines(num_threads);
    v_engines[0] = libint2::Engine(Operator::nuclear,GTO_MAX,L_MAX);
    A->libint_point_charges();
    v_engines[0].set_params(A->libint_point_charges());
    for (size_t i = 0; i != num_threads; ++i) {
        v_engines[i] = v_engines[0];
    }

//     std::vector<libint2::Engine> ef_engines(num_threads);
//     ef_engines[0] = libint2::Engine(Operator::nuclear,GTO_MAX,L_MAX); 
//     if(have_efrag){
//         ef_engines[0].set_params(EF->libint_point_charges());
//         for (size_t i = 0; i != num_threads; ++i) {
//             ef_engines[i] = ef_engines[0];
//         }
//     }
    
    std::vector<libint2::Engine> e_engines(num_threads);
    e_engines[0] = libint2::Engine(Operator::coulomb,GTO_MAX,L_MAX);
    for (size_t i = 0; i != num_threads; ++i) {
        e_engines[i] = e_engines[0];
    }
    
    
    std::vector<Shell> s;
    A->basis[n_state].gen_libint_shells(&s);
    
    double * S_AO     = new double[n_ao*n_ao];
    double * BUF      = new double[n_ao*n_ao];
    double * Ort_B_a  = new double[n_ao*n_ao];
    double * Ort_B_b  = new double[n_ao*n_ao];
    double * BUF2     = new double[n_ao*n_ao];
    double * SM05     = new double[n_ao*n_ao];
    double * SP05     = new double[n_ao*n_ao];
    double * H_AO     = new double[n_ao*n_ao];
    double * Fa_AO    = new double[n_ao*n_ao];
    double * Fb_AO    = new double[n_ao*n_ao];
    double * Jab      = new double[n_ao*n_ao];
    double * Ka       = new double[n_ao*n_ao];
    double * Kb       = new double[n_ao*n_ao];
    
    double * DM_S     = new double[n_ao*n_ao];
    double * DM_A     = new double[n_ao*n_ao];
    double * DM_B     = new double[n_ao*n_ao];
    double * DM_A_old = new double[n_ao*n_ao];
    double * DM_B_old = new double[n_ao*n_ao];
    double * Q_ba     /*= new double[n_ao*n_ao]*/;
    double * Q_ab     = new double[n_ao*n_ao];
//     double * Q_bb     = new double[n_ao*n_ao];
    double * QI       = new double[n_ao*n_ao];
    
    double *J[2]   ={Jab ,QI  };
    double *K[2]   ={Ka  ,Kb  };
    double *DM_J[2]={DM_S,Q_ab};
    double *DM_K[2]={DM_A,DM_B};
    
    
//     double * Dx_AO= new double[n_ao*n_ao];
//     double * Dy_AO= new double[n_ao*n_ao];
//     double * Dz_AO= new double[n_ao*n_ao];
    double * orb_e_a  = A->orb_energy[n_state];
    double * orb_e_b  = A->orb_energy_B[n_state];
    double * o_v_a      = new double[n_ao*n_alp_ex];
    double * o_v_b      = new double[n_ao*n_bet_ex];
    
    
    gen_matr_from_2shells(S_AO,s,s,n_ao,n_ao,&(s_engine),0);
//     fprintf(out_stream,"S\n");
//     PrintMatr(S_AO,n_ao,n_ao,1);
    
    S05_calc(S_AO,SM05,SP05,n_ao);
//     fprintf(out_stream,"P\n");
//     PrintMatr(SP05,n_ao,n_ao,1);
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                         n_ao,n_ao,n_ao,1.0,
//                         SP05,n_ao,
//                         SP05,n_ao,0.0,
//                         BUF,n_ao);
//     fprintf(out_stream,"P^2:\n");
//     PrintMatr(BUF,n_ao,n_ao,1);    
    
    gen_matr_from_2shells(S_AO,s,s,n_ao,n_ao,&(s_engine),0);
    
    gen_matr_from_2shells(H_AO,s,s,n_ao,n_ao,&(t_engine),0);
    add_matr_from_2shells(H_AO,s,s,n_ao,n_ao,&(v_engines),1);
    
//     gen_matr_from_2shells(Dx_AO,s,s,n_ao,n_ao,&(d_engine),1);
//     gen_matr_from_2shells(Dy_AO,s,s,n_ao,n_ao,&(d_engine),2);    
//     gen_matr_from_2shells(Dz_AO,s,s,n_ao,n_ao,&(d_engine),3);    
    
//     set_zero_matr(A->MO_VEC[n_state],n_ao*n_ao);
//     fprintf(out_stream,"V:\n");
//     PrintMatr(A->MO_VEC[n_state],n_ao,n_ao,1);
    
//     tr_and_diag(H_AO, SM05, BUF, A->MO_VEC[n_state], orb_energy, n_ao);
    
    if(ex[n_state-1]){
        fprintf(out_stream,"change B\n");
        change_orbs(A->MO_VEC_B[n_state],n_el-1,n_el,n_ao);
    }
    else{
        fprintf(out_stream,"change A\n");
        change_orbs(A->MO_VEC[n_state],n_el-1,n_el,n_ao);
    }
      
    
    
//     fprintf(out_stream,"DM_A:\n");
//     PrintMatr(DM_A,n_ao,n_ao,1);
//     
//     fprintf(out_stream,"DM_B:\n");
//     PrintMatr(DM_B,n_ao,n_ao,1);
    
    double max_grad_el;
//     double max_grad_el_B;
    soscf_engine_UHF SOSCF;
    SOSCF.N_LBFGS_VECTORS=30;
    SOSCF.init(std::min(n_ao  ,std::min(SOSCF.N_LBFGS_VECTORS,max_n_iter)),n_el,n_el,n_ao-n_alp_ex,n_ao-n_bet_ex  /*, A*/);    
    
    fprintf(out_stream,"Starting SCF\n");
    
    while(iter_n<max_n_iter){
//                 fprintf(out_stream,"VEC\n");
//         fprintf(out_stream,"V_a\n");
//         PrintMatr(A->MO_VEC  [n_state],n_ao,n_ao,1);
//         fprintf(out_stream,"V_b\n");
//         PrintMatr(A->MO_VEC_B[n_state],n_ao,n_ao,1);
// fprintf(out_stream,"%e %e %e %e\n",A->MO_VEC  [n_state][13*180+7],A->MO_VEC  [n_state][13*180+43],A->MO_VEC  [n_state][13*180+79],A->MO_VEC  [n_state][13*180+115]);
// fprintf(out_stream,"%e %e %e %e\n",A->MO_VEC_B[n_state][13*180+7],A->MO_VEC_B[n_state][13*180+43],A->MO_VEC_B[n_state][13*180+79],A->MO_VEC_B[n_state][13*180+115]);
// fprintf(out_stream,"%e %e %e %e\n",A->MO_VEC  [n_state][14*180+7],A->MO_VEC  [n_state][14*180+43],A->MO_VEC  [n_state][14*180+79],A->MO_VEC  [n_state][14*180+115]);
// fprintf(out_stream,"%e %e %e %e\n",A->MO_VEC_B[n_state][14*180+7],A->MO_VEC_B[n_state][14*180+43],A->MO_VEC_B[n_state][14*180+79],A->MO_VEC_B[n_state][14*180+115]);
// getchar();
        E_old=E;
        gen_HF_DM(DM_A, A->MO_VEC  [n_state], n_ao, n_el);
        gen_HF_DM(DM_B, A->MO_VEC_B[n_state], n_ao, n_el);
        for(int i=0;i<2;i++)set_zero_matr(J[i],n_ao*n_ao);
        for(int i=0;i<2;i++)set_zero_matr(K[i],n_ao*n_ao);
        for(int i=0;i<n_ao*n_ao;i++)DM_S[i] = DM_A[i]+DM_B[i];
//         gen_dc_DM(Q_ab, Q_ba,A->MO_VEC  [0      ],A->MO_VEC  [n_state],n_ao,n_el,S_AO);
//         gen_dc_DM(Q_ab, Q_ba,A->MO_VEC  [0      ],A->MO_VEC_B[n_state],n_ao,n_el,S_AO);
        gen_dc_DM(Q_ab, Q_ba,A->MO_VEC  [n_state],A->MO_VEC_B[n_state],n_ao,n_el,S_AO);
        DM_to_F_transform(J, K, DM_J, 2, DM_K, 2, s, n_ao, &e_engines, num_threads);
        
        for(int i=0;i<n_ao*n_ao;i++)Fa_AO  [i] = H_AO[i]+J[0][i]-Ka[i];
        for(int i=0;i<n_ao*n_ao;i++)Fb_AO  [i] = H_AO[i]+J[0][i]-Kb[i];
//         for(int i=0;i<n_ao*n_ao;i++)DM_old[i] = DM[i];
        

//         DCSCF_Matr_calc(Fa_AO, Fb_AO, DM_A, DM_B, Q_ab, QI, s, n_ao, &e_engines, num_threads);
//         UHF_Matr_calc(Fa_AO, Fb_AO, DM_A, DM_B, /*Q_ab, QI,*/ s, n_ao, &e_engines, num_threads);

//         fprintf(out_stream,"%e %e %e %e\n",E_1el_calc(Fa_AO, DM_A, n_ao, n_ao)
//                               ,E_1el_calc(Fb_AO, DM_B, n_ao, n_ao)
//                               ,E_1el_calc(H_AO,  DM_A, n_ao, n_ao)
//                               ,E_1el_calc(H_AO,  DM_B, n_ao, n_ao));
//         fprintf(out_stream,"Qab\n");
//         PrintMatr(Q_ab,n_ao,n_ao,1);
//         fprintf(out_stream,"QI\n");
//         PrintMatr(QI,n_ao,n_ao,1);
//         fprintf(out_stream,"QI\n");
//         PrintMatr(QI,n_ao,n_ao,1);
//         fprintf(out_stream,"Q_ab\n");
//         PrintMatr(Q_ab,n_ao,n_ao,1);
               
        fprintf(out_stream,"%e %e\n",0.5*(E_1el_calc(Fa_AO, DM_A, n_ao, n_ao)+
               E_1el_calc(Fb_AO, DM_B, n_ao, n_ao)+
               E_1el_calc(H_AO,  DM_A, n_ao, n_ao)+
               E_1el_calc(H_AO,  DM_B, n_ao, n_ao))+V_nuc,
               E_1el_calc(QI,    Q_ab, n_ao, n_ao));
        E=0.5*(E_1el_calc(Fa_AO, DM_A, n_ao, n_ao)+
               E_1el_calc(Fb_AO, DM_B, n_ao, n_ao)+
               E_1el_calc(H_AO,  DM_A, n_ao, n_ao)+
               E_1el_calc(H_AO,  DM_B, n_ao, n_ao)+
               E_1el_calc(QI,    Q_ab, n_ao, n_ao)*2)+V_nuc;
        
        set_zero_matr(o_v_a,n_state*n_alp_ex);
        set_zero_matr(o_v_b,n_state*n_bet_ex);
        
        for(int i=0,i_a=0,i_b=0;i<n_state;i++){
            if(ex[i]){
                ortvec_calc(o_v_b+i_b*n_ao, A->MO_VEC_B[i], A->MO_VEC_B[n_state], S_AO, n_ao, n_el,1);
                i_b++;
            }
            else{
                ortvec_calc(o_v_a+i_a*n_ao, A->MO_VEC[i], A->MO_VEC[n_state], S_AO, n_ao, n_el,1);
                i_a++;
            }
        }
//         ort_F_calc(Fa_AO, SM05, SP05, BUF, A->MO_VEC[n_state], BUF2, o_v_a, Ort_B_a, n_ao, n_alp_ex);
//         fprintf(out_stream,"Fa:\n");
//         PrintMatr(BUF,n_ao-n_alp_ex,n_ao-n_alp_ex,1);
//         ort_F_calc(Fb_AO, SM05, SP05, BUF, A->MO_VEC_B[n_state], BUF2, o_v_b, Ort_B_b, n_ao, n_bet_ex);
//         fprintf(out_stream,"Fb:\n");
//         PrintMatr(BUF,n_ao-n_bet_ex,n_ao-n_bet_ex,1);

//         set_zero_matr(QI,n_ao*n_ao);
//         for(int i=0;i<n_ao;i++)QI[i*(n_ao+1)]=1;
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_ao,n_ao,n_ao,1.0,
                        QI,n_ao,
                        Q_ab,n_ao,0.0,//!!!!!!!!!!!!!!
                        BUF,n_ao);
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_ao,n_ao,n_ao,1.0,
                        BUF,n_ao,
                        S_AO,n_ao,1.0,//!!!!!!!!!!!!!!
                        Fa_AO,n_ao);
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ao,n_ao,n_ao,1.0,
                        QI,n_ao,
                        Q_ab,n_ao,0.0,///!!!!!!!!!!!!
                        BUF,n_ao);
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_ao,n_ao,n_ao,1.0,
                        BUF,n_ao,
                        S_AO,n_ao,1.0,///!!!!!!!!!!!!
                        Fb_AO,n_ao);
//         fprintf(out_stream,"Ma\n");
//         PrintMatr(Fa_AO,n_ao,n_ao,1);
//         fprintf(out_stream,"Mb\n");
//         PrintMatr(Fb_AO,n_ao,n_ao,1);
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,n_ao,n_ao,n_ao,1.0,Fa_AO              ,n_ao,A->MO_VEC[n_state],n_ao,0.0,BUF2,n_ao); 
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0,A->MO_VEC[n_state],n_ao,BUF2              ,n_ao,0.0,BUF ,n_ao);
// //         fprintf(out_stream,"Fa\n");
//         PrintMatr(BUF,n_ao,n_ao,1);
//         
// //         fprintf(out_stream,"Va\n");
// //         PrintMatr(A->MO_VEC[n_state],n_ao,n_ao,1);
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,n_ao,n_ao,n_ao,1.0,Fb_AO              ,n_ao,A->MO_VEC_B[n_state],n_ao,0.0,BUF2,n_ao); 
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0,A->MO_VEC_B[n_state],n_ao,BUF2              ,n_ao,0.0,BUF ,n_ao);
//         fprintf(out_stream,"Fb\n");
//         PrintMatr(BUF,n_ao,n_ao,1);
        
        
        

        fprintf(out_stream,"ITER %d\tE=%.10f\tdE=%e\t",iter_n,E,E-E_old);
        
//         int n_el_a= n_el -1;
//         int n_el_b= n_el -1;
//         int n_ao_a =n_ao;
//         int n_ao_b =n_ao;

//         n_alp_ex=0;
        ort_F_calc(Fa_AO, SM05, SP05, BUF, A->MO_VEC[n_state], BUF2, o_v_a, Ort_B_a, n_ao, n_alp_ex);
//         fprintf(out_stream,"Fa:\n");
//         PrintMatr(BUF,n_ao-n_alp_ex,n_ao-n_alp_ex,1);
        max_grad_el = SOSCF.calc_a(BUF);
//          fprintf(out_stream,"orb_grad = %e\t",max_grad_el/4);

        ort_F_calc(Fb_AO, SM05, SP05, BUF, A->MO_VEC_B[n_state], BUF2, o_v_b, Ort_B_b, n_ao, n_bet_ex);
       
        
//         fprintf(out_stream,"Fb:\n");
//         PrintMatr(BUF,n_ao-n_bet_ex,n_ao-n_bet_ex,1);
        //calculation of gradient and appr hessian
        max_grad_el = std::max(max_grad_el, SOSCF.calc_b(BUF));
        
        fprintf(out_stream,"orb_grad = %e\n",max_grad_el/4);
//         fprintf(out_stream,"orb_grad_a:\n");
//         PrintMatr(SOSCF.orb_grad_a,n_el,n_ao-n_el-n_alp_ex,1);
//         fprintf(out_stream,"orb_grad_ab:\n");
//         PrintMatr(SOSCF.orb_grad_b,n_el,n_ao-n_el-n_bet_ex,1);
        fflush(out_stream);
        //case of small gradient - simple diagonalization and exit
        if(max_grad_el<SOSCF.grad_min_soscf){
//             ort_diag   (Fa_AO, SM05, SP05, BUF, A->MO_VEC  [n_state], orb_e_a, o_v_a, n_ao, n_alp_ex);
//             ort_diag   (Fb_AO, SM05, SP05, BUF, A->MO_VEC_B[n_state], orb_e_b, o_v_b, n_ao, n_bet_ex);
            
            break;//"converged" signal
        }
        
        //case of large gradient - simple diagonalization
//         if((iter_n==0)||(max_grad_el>SOSCF.grad_max_soscf)){
//             if(iter_n)fprintf(out_stream,"WARNING: gradient is too large %e\n",max_grad_el);
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC  [n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC  [n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//             fprintf(out_stream,"S_AA before ort_diag\n");
//             PrintMatr(BUF2,n_el,n_el,1);
//             ort_diag   (Fa_AO, SM05, SP05, BUF, A->MO_VEC  [n_state], orb_e_a, o_v_a, n_ao, n_alp_ex);
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC  [n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC  [n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//             fprintf(out_stream,"S_AA before ort_diag\n");
//             PrintMatr(BUF2,n_el,n_el,1);
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC_B[n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC_B[n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//             fprintf(out_stream,"S_BB before ort_diag\n");
//             PrintMatr(BUF2,n_el,n_el,1);
//             ort_diag   (Fb_AO, SM05, SP05, BUF, A->MO_VEC_B[n_state], orb_e_b, o_v_b, n_ao, n_bet_ex);
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC_B[n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC_B[n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//             fprintf(out_stream,"S_BB after ort_diag\n");
//             PrintMatr(BUF2,n_el,n_el,1);
//             iter_n++;
//             continue;
//         }
        
        //calculation of orbital shift
        
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC_B[n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC_B[n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//         fprintf(out_stream,"S_BB before rot_matr_calc\n");
//         PrintMatr(BUF2,n_el,n_el,1);
        SOSCF.rot_matr_calc();
        SOSCF.vec_update_OHF(A->MO_VEC  [n_state], Ort_B_a, SM05, BUF, n_alp_ex, 'a');
        SOSCF.vec_update_OHF(A->MO_VEC_B[n_state], Ort_B_b, SM05, BUF, n_bet_ex, 'b');
//         getchar();
        
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC  [n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC  [n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//         fprintf(out_stream,"S_AA\n");
//         PrintMatr(BUF2,n_el,n_el,1);
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC_B[n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC_B[n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//         fprintf(out_stream,"S_BB\n");
//         PrintMatr(BUF2,n_el,n_el,1);
        
        for(int i=0;i<n_ao*n_alp_ex;i++)A->MO_VEC  [n_state][i+n_ao*(n_ao-n_alp_ex)]=o_v_a[i];
        for(int i=0;i<n_ao*n_bet_ex;i++)A->MO_VEC_B[n_state][i+n_ao*(n_ao-n_bet_ex)]=o_v_b[i];
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
//                             n_ao,n_el,n_ao,1.0,
//                             S_AO,n_ao,
//                             A->MO_VEC_B[n_state],n_ao,0.0,
//                             BUF,n_el);
//         
//         cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                             n_el,n_el,n_ao,1.0,
//                             A->MO_VEC  [n_state],n_ao,
//                             BUF,n_el,0.0,
//                             BUF2,n_el);
//         fprintf(out_stream,"S_AB\n");
//         PrintMatr(BUF2,n_el,n_el,1);
        
        
//         
        
        iter_n++;
        
    }
    printf_timer("OHF converged at");
    SOSCF.finalize();
    
//     for(int i=0;i<n_ao*n_alp_ex;i++)A->MO_VEC  [n_state][i+n_ao*(n_ao-n_alp_ex)]=0.0;
//     for(int i=0;i<n_ao*n_bet_ex;i++)A->MO_VEC_B[n_state][i+n_ao*(n_ao-n_bet_ex)]=0.0;
//     
//     for(int i=0;i<n_alp_ex;i++)A->orb_energy  [n_state][i+(n_ao-n_alp_ex)]=100000000.0;
//     for(int i=0;i<n_bet_ex;i++)A->orb_energy_B[n_state][i+(n_ao-n_bet_ex)]=100000000.0;
//     
    
  
//     delete[] orb_e_a;
//     delete[] orb_e_b;
    delete[] S_AO    ;
    delete[] Ort_B_a ;
    delete[] Ort_B_b ;
    delete[] BUF     ;
    delete[] BUF2    ;
    delete[] SM05    ;
    delete[] SP05    ;
    delete[] H_AO    ;
    delete[] Fa_AO   ;
    delete[] Fb_AO   ;
    delete[] DM_A    ;
    delete[] DM_B    ;
    delete[] DM_A_old;
    delete[] DM_B_old;
//     delete[] Dx_AO;
//     delete[] Dy_AO;
//     delete[] Dz_AO;
     
     libint2::finalize();
     return 0;
    
}

int FplusOxP(double * F, double a, double * O, double * P, double * S, double *B, double *B2, int n){
    
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, //S*P
//                         n,n,n,1.0,
//                         S,n,
//                         P,n,0.0,
//                         B,n);
//         
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, //S*P*O
//                         n,n,n,1.0,
//                         B,n,
//                         O,n,0.0,
//                         B2,n);
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,//O*P
                        n,n,n,1.0,
                        O,n,
                        P,n,0.0,
                        B,n);
        
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,//F+a*O*P*S
                        n,n,n,a,
                        B,n,
                        S,n,1.0,
                        F,n);
    
    return 0;
}


int DC_ROHF(molecule * A, int n_state, int * ex){
    
        
    libint2::initialize();
    
    double E    =0;
    double E_old=1.0E15;
    
    int n_alp_ex=0;
    int n_bet_ex=0;
    for(int i=0;i<n_state;i++){
        if(ex[i]) n_bet_ex++;
        else      n_alp_ex++;
    }
    fprintf(out_stream,"a = %d\nb = %d\n", n_alp_ex, n_bet_ex);
//     getchar();
    
    int iter_n=0;
    int n_ao = A->basis[n_state].n_ao;
    int n_el = A->n_el[n_state]/2;
    
    
    double V_nuc=0.0;
    V_nuc=E_nuc_calc(A);
    fprintf(out_stream,"V_nuc = %e\n",V_nuc);
    
    Engine  s_engine(Operator::overlap,GTO_MAX,L_MAX);
    Engine  t_engine(Operator::kinetic,GTO_MAX,L_MAX);
//     Engine  d_engine(Operator::emultipole1,GTO_MAX,L_MAX);
//     d_engine.set_params(std::array<double,3>{0.0, 0.0, 0.0});
    
    std::vector<libint2::Engine> v_engines(num_threads);
    v_engines[0] = libint2::Engine(Operator::nuclear,GTO_MAX,L_MAX);
    A->libint_point_charges();
    v_engines[0].set_params(A->libint_point_charges());
    for (size_t i = 0; i != num_threads; ++i) {
        v_engines[i] = v_engines[0];
    }

//     std::vector<libint2::Engine> ef_engines(num_threads);
//     ef_engines[0] = libint2::Engine(Operator::nuclear,GTO_MAX,L_MAX); 
//     if(have_efrag){
//         ef_engines[0].set_params(EF->libint_point_charges());
//         for (size_t i = 0; i != num_threads; ++i) {
//             ef_engines[i] = ef_engines[0];
//         }
//     }
    
    std::vector<libint2::Engine> e_engines(num_threads);
    e_engines[0] = libint2::Engine(Operator::coulomb,GTO_MAX,L_MAX);
    for (size_t i = 0; i != num_threads; ++i) {
        e_engines[i] = e_engines[0];
    }
    
    
    std::vector<Shell> s;
    A->basis[n_state].gen_libint_shells(&s);
    
    double * S_AO     = new double[n_ao*n_ao];
    double * BUF      = new double[n_ao*n_ao];
    double * Ort_B_a  = new double[n_ao*n_ao];
    double * Ort_B_b  = new double[n_ao*n_ao];
    double * BUF2     = new double[n_ao*n_ao];
    double * SM05     = new double[n_ao*n_ao];
    double * SP05     = new double[n_ao*n_ao];
    double * H_AO     = new double[n_ao*n_ao];
    double * Fd_AO    = new double[n_ao*n_ao];
    double * Fa_AO    = new double[n_ao*n_ao];
    double * Fb_AO    = new double[n_ao*n_ao];
    double * Jd       = new double[n_ao*n_ao];
    double * Ja       = new double[n_ao*n_ao];
    double * Jb       = new double[n_ao*n_ao];
    double * Kd       = new double[n_ao*n_ao];
    double * Ka       = new double[n_ao*n_ao];
    double * Kb       = new double[n_ao*n_ao];
    
    double * DM_d     = new double[n_ao*n_ao];
    double * DM_a     = new double[n_ao*n_ao];
    double * DM_b     = new double[n_ao*n_ao];
    double * DM_A_old = new double[n_ao*n_ao];
    double * DM_B_old = new double[n_ao*n_ao];


//     double * Q_bb     = new double[n_ao*n_ao];
    
    double *J [3]   ={Jd  ,Ja  ,Jb  };
    double *K [3]   ={Kd  ,Ka  ,Kb  };
    double *DM[3]   ={DM_d,DM_a,DM_b};
    
    
//     double * Dx_AO= new double[n_ao*n_ao];
//     double * Dy_AO= new double[n_ao*n_ao];
//     double * Dz_AO= new double[n_ao*n_ao];
    double * orb_e_a  = A->orb_energy[n_state];
    double * orb_e_b  = A->orb_energy_B[n_state];
//     double * o_v_a      = new double[n_ao* n_el   ];
    double * o_v_a      = new double[n_ao*(n_el+1)];
    
    
    gen_matr_from_2shells(S_AO,s,s,n_ao,n_ao,&(s_engine),0);
//     fprintf(out_stream,"S\n");
//     PrintMatr(S_AO,n_ao,n_ao,1);
    
    S05_calc(S_AO,SM05,SP05,n_ao);
//     fprintf(out_stream,"P\n");
//     PrintMatr(SP05,n_ao,n_ao,1);
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                         n_ao,n_ao,n_ao,1.0,
//                         SP05,n_ao,
//                         SP05,n_ao,0.0,
//                         BUF,n_ao);
//     fprintf(out_stream,"P^2:\n");
//     PrintMatr(BUF,n_ao,n_ao,1);    
    
    gen_matr_from_2shells(S_AO,s,s,n_ao,n_ao,&(s_engine),0);
    
    gen_matr_from_2shells(H_AO,s,s,n_ao,n_ao,&(t_engine),0);
    add_matr_from_2shells(H_AO,s,s,n_ao,n_ao,&(v_engines),1);
    
//     gen_matr_from_2shells(Dx_AO,s,s,n_ao,n_ao,&(d_engine),1);
//     gen_matr_from_2shells(Dy_AO,s,s,n_ao,n_ao,&(d_engine),2);    
//     gen_matr_from_2shells(Dz_AO,s,s,n_ao,n_ao,&(d_engine),3);    
    
//     set_zero_matr(A->MO_VEC[n_state],n_ao*n_ao);
//     fprintf(out_stream,"V:\n");
//     PrintMatr(A->MO_VEC[n_state],n_ao,n_ao,1);
    
//     tr_and_diag(H_AO, SM05, BUF, A->MO_VEC[n_state], orb_energy, n_ao);
    
//     if(ex[n_state-1]){
//         fprintf(out_stream,"change B\n");
//         change_orbs(A->MO_VEC_B[n_state],n_el-1,n_el,n_ao);
//     }
//     else{
//         fprintf(out_stream,"change A\n");
//         change_orbs(A->MO_VEC[n_state],n_el-1,n_el,n_ao);
//     }
      
    
    
//     fprintf(out_stream,"DM_A:\n");
//     PrintMatr(DM_A,n_ao,n_ao,1);
//     
//     fprintf(out_stream,"DM_B:\n");
//     PrintMatr(DM_B,n_ao,n_ao,1);
    
    double max_grad_el;
//     double max_grad_el_B;
    soscf_engine_ROHF SOSCF;
    SOSCF.N_LBFGS_VECTORS=30;
    SOSCF.init(std::min(n_ao,std::min(SOSCF.N_LBFGS_VECTORS,max_n_iter)),n_el-1,1,1,n_ao);   
    
    fprintf(out_stream,"Starting SCF\n");
    
    while(iter_n<max_n_iter){
         cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,
                    n_ao,n_ao,n_ao,1.0,
                    S_AO,n_ao,
                    A->MO_VEC[n_state],n_ao,0.0,
                    BUF2,n_ao); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    n_ao,n_ao,n_ao,1.0,
                    A->MO_VEC[n_state],n_ao,
                    BUF2,n_ao,0.0,
                    BUF ,n_ao);
        
//         fprintf(out_stream,"S_AO\n");
//         PrintMatr(BUF,n_ao,n_ao,1);
        E_old=E;
        gen_HF_DM(DM_d, A->MO_VEC  [n_state], n_ao, n_el-1);

        for(int i=0;i<3;i++)set_zero_matr(J[i],n_ao*n_ao);
        for(int i=0;i<3;i++)set_zero_matr(K[i],n_ao*n_ao);
        
        for(int i=0;i<n_ao;i++)for(int j=0;j<n_ao;j++)DM_a[i*n_ao+j] = A->MO_VEC[n_state][(n_el-1)*n_ao+i]*A->MO_VEC[n_state][(n_el-1)*n_ao+j];
        for(int i=0;i<n_ao;i++)for(int j=0;j<n_ao;j++)DM_b[i*n_ao+j] = A->MO_VEC[n_state][ n_el   *n_ao+i]*A->MO_VEC[n_state][ n_el   *n_ao+j];
        
        DM_to_F_transform(J, K, DM, 3, DM, 3, s, n_ao, &e_engines, num_threads);
        
        for(int i=0;i<n_ao*n_ao;i++)Fd_AO[i] = H_AO[i]+2*Jd[i]-Kd[i]+Ja[i]-0.5*Ka[i]+Jb[i]-0.5*Kb[i];
        for(int i=0;i<n_ao*n_ao;i++)Fa_AO[i] = H_AO[i]+2*Jd[i]-Kd[i]+                Jb[i]+    Kb[i];
        for(int i=0;i<n_ao*n_ao;i++)Fb_AO[i] = H_AO[i]+2*Jd[i]-Kd[i]+Ja[i]+    Ka[i]                ;
        
//         fprintf(out_stream,"F\n");
//         PrintMatr(Fd_AO,n_ao,n_ao,1);
        
        
        E=     E_1el_calc(Fd_AO, DM_d, n_ao, n_ao)+
               E_1el_calc( H_AO, DM_d, n_ao, n_ao)+
          0.5*(E_1el_calc(Fa_AO, DM_a, n_ao, n_ao)+
               E_1el_calc(Fb_AO, DM_b, n_ao, n_ao)+
               E_1el_calc( H_AO, DM_a, n_ao, n_ao)+
               E_1el_calc( H_AO, DM_b, n_ao, n_ao))+V_nuc;
               
        ortvec_calc(o_v_a, A->MO_VEC[0], A->MO_VEC[n_state], S_AO, n_ao, n_el,0);
//         fprintf(out_stream,"VEC\n");
//         PrintMatr(A->MO_VEC[n_state],n_ao,n_ao,1);
//         fprintf(out_stream,"O_V\n");
//         PrintMatr(o_v_a,1,n_ao,1);

//         FplusOxP(Fd_AO, 1.5,Kb,DM_a,S_AO,BUF,BUF2,n_ao);
//         FplusOxP(Fd_AO,-1.0,Ja,DM_a,S_AO,BUF,BUF2,n_ao);
//         FplusOxP(Fd_AO, 0.5,Ka,DM_a,S_AO,BUF,BUF2,n_ao);
//         FplusOxP(Fd_AO, 1.5,Ka,DM_b,S_AO,BUF,BUF2,n_ao);
//         FplusOxP(Fd_AO,-1.0,Jb,DM_b,S_AO,BUF,BUF2,n_ao);
//         FplusOxP(Fd_AO, 0.5,Kb,DM_b,S_AO,BUF,BUF2,n_ao);
//         fprintf(out_stream,"V\n");
//         PrintMatr(A->MO_VEC[n_state],n_ao,n_ao,1);
//         fprintf(out_stream,"F_AO\n");
//         PrintMatr(Fd_AO,n_ao,n_ao,1);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,n_ao,n_ao,n_ao,1.0,Fd_AO             ,n_ao,A->MO_VEC[n_state],n_ao,0.0,BUF2 ,n_ao); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0,A->MO_VEC[n_state],n_ao,BUF2              ,n_ao,0.0,BUF  ,n_ao);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,n_ao,n_ao,n_ao,1.0,Fa_AO             ,n_ao,A->MO_VEC[n_state],n_ao,0.0,BUF2 ,n_ao); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0,A->MO_VEC[n_state],n_ao,BUF2              ,n_ao,0.0,Fa_AO,n_ao);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,n_ao,n_ao,n_ao,1.0,Fb_AO             ,n_ao,A->MO_VEC[n_state],n_ao,0.0,BUF2 ,n_ao); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0,A->MO_VEC[n_state],n_ao,BUF2              ,n_ao,0.0,Fb_AO,n_ao);
//         fprintf(out_stream,"F_MO\n");
//         PrintMatr(BUF,n_ao,n_ao,1);
//         fprintf(out_stream,"Fa_MO\n");
//         PrintMatr(Fa_AO,n_ao,n_ao,1);
//         fprintf(out_stream,"Fb_MO\n");
//         PrintMatr(Fb_AO,n_ao,n_ao,1);
        fprintf(out_stream,"ITER %d\tE=%.10f\tdE=% e\t",iter_n,E,E-E_old);
        max_grad_el=SOSCF.calc(BUF,Fa_AO,Fb_AO);
        SOSCF.OHF_step(A->MO_VEC[n_state],o_v_a,BUF2,BUF,1);
        

        if( max_grad_el<SOSCF.grad_min_soscf)break;
            
//         tr_and_diag(Fd_AO, SM05, BUF, A->MO_VEC[n_state], orb_e_a, n_ao);
        
//         fprintf(out_stream,"F_AO\n");
//         PrintMatr(Fd_AO,n_ao,n_ao,1);
        ortvec_calc(o_v_a, A->MO_VEC[0], A->MO_VEC[n_state], S_AO, n_ao, n_el,0);
        
        ///!!!!!!!WORKING HERE!!!!!!!!!
        fprintf(out_stream,"O_V\n");
        PrintMatr(o_v_a,1,n_ao,1);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,
                    n_ao,n_el+1,n_ao,1.0,
                    S_AO,n_ao,
                    A->MO_VEC[n_state],n_ao,0.0,
                    BUF2,n_el+1); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    n_el+1,n_el+1,n_ao,1.0,
                    o_v_a,n_ao,
                    BUF2,n_el+1,0.0,
                    BUF ,n_el+1);
        fprintf(out_stream,"\t<0|1> = % e\n",BUF[n_el]);
        for(int i=0; i<n_ao;i++)A->MO_VEC[n_state][n_el*n_ao+i]-=BUF[0]*o_v_a[i];
//         
        cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,
                    n_ao,1,n_ao,1.0,
                    S_AO,n_ao,
                    A->MO_VEC[n_state]+(n_el)*n_ao,n_ao,0.0,
                    BUF2,1); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    1,1,n_ao,1.0,
                    A->MO_VEC[n_state]+(n_el)*n_ao,n_ao,
                    BUF2,1,0.0,
                    BUF ,1);
        for(int i=0; i<n_ao;i++)A->MO_VEC[n_state][n_el*n_ao+i]=A->MO_VEC[n_state][n_el*n_ao+i]/sqrt(BUF[0]);
        
//         
//         fprintf(out_stream,"\t<fn|fn> = % e\n",BUF[0]);
//         
//         
         cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,
                    n_ao,n_el+1,n_ao,1.0,
                    S_AO,n_ao,
                    A->MO_VEC[n_state],n_ao,0.0,
                    BUF2,n_el+1); 
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    n_el+1,n_el+1,n_ao,1.0,
                    A->MO_VEC[n_state],n_ao,
                    BUF2,n_el+1,0.0,
                    BUF ,n_el+1);
        
        fprintf(out_stream,"S_AO\n");
        PrintMatr(BUF,n_el+1,n_el+1,1);
        
        fprintf(out_stream,"\tdet-1 = % e\n",mat_det_calc_lapack(BUF,n_el+1)-1);
        getchar();
//         PrintMatr(BUF,n_ao,n_ao,1);
         
        
        iter_n++;
        
    }
    printf_timer("\n\nOHF converged at");
    SOSCF.finalize();
    
//     for(int i=0;i<n_ao*n_alp_ex;i++)A->MO_VEC  [n_state][i+n_ao*(n_ao-n_alp_ex)]=0.0;
//     for(int i=0;i<n_ao*n_bet_ex;i++)A->MO_VEC_B[n_state][i+n_ao*(n_ao-n_bet_ex)]=0.0;
//     
//     for(int i=0;i<n_alp_ex;i++)A->orb_energy  [n_state][i+(n_ao-n_alp_ex)]=100000000.0;
//     for(int i=0;i<n_bet_ex;i++)A->orb_energy_B[n_state][i+(n_ao-n_bet_ex)]=100000000.0;
//     
    
  
//     delete[] orb_e_a;
//     delete[] orb_e_b;
    delete[] S_AO    ;
    delete[] BUF     ;
    delete[] Ort_B_a ;
    delete[] Ort_B_b ;
    delete[] BUF2    ;
    delete[] SM05    ;
    delete[] SP05    ;
    delete[] H_AO    ;
    delete[] Fd_AO   ;
    delete[] Fa_AO   ;
    delete[] Fb_AO   ;
    delete[] Jd      ;
    delete[] Ja      ;
    delete[] Jb      ;
    delete[] Kd      ;
    delete[] Ka      ;
    delete[] Kb      ;
    delete[] DM_d    ;
    delete[] DM_a    ;
    delete[] DM_b    ;
    delete[] DM_A_old;
    delete[] DM_B_old;
//     delete[] Dx_AO;
//     delete[] Dy_AO;
//     delete[] Dz_AO;
     
     libint2::finalize();
     return 0;
    
}

#endif
