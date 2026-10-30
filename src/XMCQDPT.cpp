# include "blas_link.h"
# include "molecule2.h"
# include "matr.h"
# include "defaults.h"
# include "doCI_data.h"
# include "doCI_matr.h"
# include "res_fit.h"
# include "RI.h"
# include "inp_out.h"
# include "xmc_read.h"
# include "from_hash.h"
# include "CAS.h"
# include "timer.h"


# define max(a,b)  (((a)<(b))?(b):(a))

extern int num_threads;

int three_block_diagonalization(double * M, double * V, double * e1,
                                int d1, int d2, int d3, int d_all,
                                double * O1, double * O2, double* O3, double * O4){
    
    int dm = max(d3,max(d1,d2));
    double * e2 = e1+d1;
    double * e3 = e2+d2;
    
    double * M1,* M2,* M3;
    M1 = new double[d1*d1];
    M2 = new double[d2*d2];
    M3 = new double[d3*d3];
    
    for(int i=0;i<d1;i++)
    for(int j=0;j<d1;j++)
        M1[i*d1+j]=M[i*d_all+j];
    
    for(int i=0;i<d2;i++)
    for(int j=0;j<d2;j++)
        M2[i*d2+j]=M[(i+d1)*d_all+j+d1];
        
    for(int i=0;i<d3;i++)
    for(int j=0;j<d3;j++)
        M3[i*d3+j]=M[(i+d1+d2)*d_all+j+d1+d2];
    
    if(d1!=0)lapack_diag(M1,e1,d1);
    if(d2!=0)lapack_diag(M2,e2,d2);
    if(d3!=0)lapack_diag(M3,e3,d3);
    
    for(int i=0;i<d2;i++)
    for(int j=0;j<d2;j++)
        O4[i*d2+j]=M2[j*d2+i];

    if(d1!=0)
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
                     d_all,d1,d1,1.0,
                     V,d_all,
                     M1,d1,0.0,
                     O1,d1);
    if(d2!=0)
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
                     d_all,d2,d2,1.0,
                     V+d_all*d1,d_all,
                     M2,d2,0.0,
                     O2,d2);
    if(d3!=0)
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
                     d_all,d3,d3,1.0,
                     V+d_all*(d1+d2),d_all,
                     M3,d3,0.0,
                     O3,d3);
    
    for(int i=0;i<d1   ;i++)
    for(int j=0;j<d_all;j++)
        V[i*d_all+j]=O1[j*d1+i];
    
    for(int i=0;i<d2   ;i++)
    for(int j=0;j<d_all;j++)
        V[(i+d1)*d_all+j]=O2[j*d2+i];
    
    for(int i=0;i<d3   ;i++)
    for(int j=0;j<d_all;j++)
        V[(i+d1+d2)*d_all+j]=O3[j*d3+i];
    
    
    delete[] M1;
    delete[] M2;
    delete[] M3;
    
    return 1;
    
}


int vec_col_link(double * AB, double * A, double * B, int a, int b, int h){
    
    int w=a+b;
    for(int i=0;i<h;i++){
        for(int j=0;j<a;j++)
            AB[i*w+j]=A[i*a+j];
        for(int j=0;j<b;j++)
            AB[i*w+j+a]=B[i*b+j];
        
    }
    
//     fPrintMatr(out_stream,AB,h,w,1);
    
    
    return 1;
}


int QDPT2(molecule * M, xmc_par * xmc, char * job_name){
    
    
    if(RI){
        gen_RI_AA(M);
    }
    else{
        fprintf(out_stream,"ERROR: RI=0 is not supported for any PT\n");
        exit(0);
    }
        
    
    xmc->write_info(M->n_act_el_alp[0],
                   M->n_act_el_bet[0],
                   M->n_act_orb   [0],
                   M->CI[0].mult);
    
    char * name = new char[BUF_LINE_LENGTH];
    doCI_data D;
    
    int n_s=xmc->cas->n_s;
    M->n_states[0]=n_s;
    D.first_alloc(M,M);
    
    D.gen_aldet_data();
    
    
        
    M->mc=1;//set multi-configuration
    
    int n_ao  = D.n_ao[0];
    int n_act = M->n_act_orb[0];
    int n_cor = M->n_cor_orb;
    int n_mo  = n_ao;
    if(D5){
        n_mo = M->n_ao_d5;
    }
    
    RI_core_realloc(n_cor+n_act, n_ao);
    
    
    int n_virt = n_mo-n_cor-n_act;
    
    if(xmc->have_ifitd){
        if(n_act!=xmc->ifitd_energy.size()){
            fprintf(out_stream,"ERROR:wrong dim of ifitd(%d) != n_act(%d)\n",xmc->ifitd_energy.size(),n_act);
            exit(1);
        }
    }
    
    
//     fprintf(out_stream,"%d %d %d %d %d\n",n_cor,n_act,n_virt,n_ao, n_mo);
//     getchar();
        
    
    D.DM_gen_ortogonal(0, 1.0000);    
    
    D.second_alloc();
    D.cpy_VEC_to_ACT();
    
    D.S_calc();
    fprintf(out_stream,"Initial CI vectors overlap matrix:\n");
    fPrintMatr(out_stream,D.ACT_DET_A,D.n_st_total[0],D.n_st_total[1],0);
    
    D.calc_1el_DM();
    
//     if(xmc->cas->w_state.size()!=n_s){
//         fprintf(out_stream,"ERROR: wrong dim of avecoe(%d) != n_s(%d)\n",xmc->cas->w_state.size(),n_s);
//         exit(1);
//     }
    
//     fprintf(out_stream,"AVECOE: ");for(const auto&c:xmc->cas->w_state)fprintf(out_stream,"%e ",c);
    
    D.average_DM(xmc->cas->w_state);
    D.nat_orb_calc_1mol(0,1);
    D.transform_1el_DM(0);
    
    fprintf(out_stream,"\n\n");
    fprintf(out_stream,"Calculation Fock-like operator...\n");
    fflush(out_stream);
    
    double * F;
    F = new double[n_ao*n_ao];
    
    M->calc_F_AO(F, D.DM_T, 0.5);
    D.AO_to_MO(F);
    
//     fprintf(out_stream,"F_mat\n");
//     fPrintMatr(out_stream,F, M->n_ao,M->n_ao,1);
//     exit(0);
       
    double * eps;
    eps = new double[n_ao];
    
    //transposed vectors for calculation of 2e integrals
    double * COR_VEC; //core
    double * ACT_VEC; //active
    double * VIRT_VEC;//virtual
//     double * OCC_VEC; //occupied=cor+act
    
    int n_occ=n_cor+n_act;
    
    COR_VEC = new double[n_ao * n_cor];
    ACT_VEC = new double[n_ao * n_act];
//     OCC_VEC = new double[n_ao * n_int];
    VIRT_VEC = new double[n_ao * n_virt];
    
    double * U;
    U=new double[n_act*n_act];
    
    fprintf(out_stream,"Diagonalization of Fock-like operator in 3 subspaces...\n");
    fflush(out_stream);
    
    three_block_diagonalization(F,D.VEC[0],eps,n_cor,n_act, n_virt,n_ao,COR_VEC,ACT_VEC,VIRT_VEC, U);
    
    double * eps_a;
    double * eps_v;
    eps_a = eps   + n_cor;
    eps_v = eps_a + n_act;
    if(xmc->have_ifitd){
        fprintf(out_stream,"WARNING: active orbital energies were changed by ifitd\n");
        for(int i=0; i<n_act;i++)eps_a[i]=xmc->ifitd_energy[i];
    }
    
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Orbital energies (3 blocks):\n");
    fprintf(out_stream,"core   :");fPrintMatr(out_stream,eps,1,n_cor,0);
    fprintf(out_stream,"active :");fPrintMatr(out_stream,eps_a,1,n_act,0);
    fprintf(out_stream,"virtual:");fPrintMatr(out_stream,eps_v,1,n_virt,0);
    fprintf(out_stream,"\n\n");
    if(write_orbs){
        fprintf(out_stream,"Writing orbitals:\n");
        VEC_energy_cpy(M->MO_VEC,D.VEC[0], M->orb_energy, eps, n_ao);
        M->MO_gamess_format();
        sprintf(name,"%s_XMC_F.out\0",job_name);
        M->GAMESS_type_out_print(name,-1);
        fprintf(out_stream,"visualization file: %s\n",name);
        
        sprintf(name,"%s_XMC_F.orb\0",job_name);
        M->MO_print(name);
        fprintf(out_stream,"data file         : %s\n",name);
//         M->MO_libint_reordr();
    }

    
    printf_timer("Fock matrix calculation");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    D.n_st_total[0]=n_s;
    D.n_st_total[1]=n_s;
    
    
    int Na = (int) std::lround(tgammal(n_act+1) / tgammal(D.n_alp_el[0]+1) / tgammal(n_act-D.n_alp_el[0]+1));
    int Nb = (int) std::lround(tgammal(n_act+1) / tgammal(D.n_bet_el[0]+1) / tgammal(n_act-D.n_bet_el[0]+1)); 
    
//     int ld_ci=n_s+1; //for new version that  must be written
    int ld_ci=(1+num_threads)*n_s+1;/// use ld_ci for old version of CCVV
    
    
    
    fprintf(out_stream,"Construction of H(0)_PP:\n\n");
    
    D.aldet.malmqvist(0,U);
    D.aldet.malmqvist(1,U);
    
    double * E0;
    E0 = new double[n_s];
    set_zero_matr(E0,n_s);
    
    double * H0PP;
    H0PP = new double[n_s*n_s];
    set_zero_matr(H0PP,n_s*n_s);
    
    double * S0PP;
    S0PP = new double[n_s*n_s];
    set_zero_matr(S0PP,n_s*n_s);
    
    double * H01;
    H01 = new double[n_s*n_s];
    set_zero_matr(H01,n_s*n_s);
    
    double * H2;
    H2 = new double[n_s*n_s];
    set_zero_matr(H2,n_s*n_s);
        
    double * E2;
    E2 = new double[n_s*n_s];
    set_zero_matr(E2,n_s*n_s);
    
    double * S2;
    S2 = new double[n_s*n_s];
    set_zero_matr(S2,n_s*n_s);
    
    double * d_x;
    d_x = new double[n_s*n_s];
    set_zero_matr(d_x,n_s*n_s);
    
    double * d_y;
    d_y = new double[n_s*n_s];
    set_zero_matr(d_y,n_s*n_s);
    
    double * d_z;
    d_z = new double[n_s*n_s];
    set_zero_matr(d_z,n_s*n_s);
    
//     double * DM1_o_format;
//     DM1_o_format = new double [n_mo*n_mo*n_s*n_s];
//     set_zero_matr(DM1_o_format,n_mo*n_mo*n_s*n_s);

    
    
    D.aldet.E_act_calc(eps_a, 0);
    D.aldet.F_calc(H0PP, 0);
    fprintf(out_stream,"H(0)_PP:\n");
    fPrintMatr10(out_stream,H0PP,n_s,n_s,0);
    
    if(xmc->d_only) read_H02_with_H0PP(H2, H0PP, n_s, xmc->gamess_file_name);
    
    
    D.S_calc();
    memcpy(S0PP ,D.ACT_DET_A,n_s*n_s*sizeof(double));
//     fprintf(out_stream,"S0:\n");
//     fPrintMatr10(out_stream,D.ACT_DET_A, D.n_st_total[0], D.n_st_total[1],1);
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"S(0)_PP:\n");
    fPrintMatr10(out_stream,S0PP,n_s,n_s,0);

    D.cpy_ACT_VEC_to_ACT(ACT_VEC); // copies ACT to L_ACT and R_ACT

    calc_2el_CI(M,&D);
    calc_2e_ints(D.h_2e, n_act,D.n_alp_el[0],D.n_bet_el[0], D.aldet.coef[0], D.aldet.coef[0], D.act_INTS,0,n_s,n_s,n_s,n_s);
    
    D.calc_1el_DM();
    
    double * H_AO;//renaming of F to H_AO for old style
    H_AO=F;
    M->mc=2;
    M->calc_F_AO(F, D.DM_C, 1.0);

    D.p_SVD=1;
//     fprintf(out_stream,"%e %e %e\n",M->Dx_nuc,M->Dy_nuc,M->Dz_nuc);
//     exit(0);
    for(int i=0;i<n_s;i++)    
    for(int j=0;j<n_s;j++){
      
        double S = D.ACT_DET_A[i*n_s+j];

        D.transform_1el_DM(i*n_s+j);
        double H_1el = E_1el_calc(M->H_AO, D.DM_T, D.n_ao[0], D.n_ao[1])*D.p_SVD;
        double  H_2el = D.H_2el_calc(S) + D.h_2e[i*n_s+j]*D.p_SVD;
        
        S=S*D.p_SVD;
        d_x[i*n_s+j] = E_1el_calc(M->Dx_AO, D.DM_T, D.n_ao[0], D.n_ao[1])*D.p_SVD+M->Dx_nuc*S;
        d_y[i*n_s+j] = E_1el_calc(M->Dy_AO, D.DM_T, D.n_ao[0], D.n_ao[1])*D.p_SVD+M->Dy_nuc*S;
        d_z[i*n_s+j] = E_1el_calc(M->Dz_AO, D.DM_T, D.n_ao[0], D.n_ao[1])*D.p_SVD+M->Dz_nuc*S;

        H01[i*n_s+j]=H_1el+H_2el+M->V_nuc*S;
                
    }
    fprintf(out_stream,"\n");
    fprintf(out_stream,"H(0+1):\n");
    fPrintMatr10(out_stream,H01,n_s,n_s,0);
    
    
    double * B;
    B = new double[n_s*n_s];
    double * C;
    C = new double[n_s*n_s];
    
    HC_SCE(H0PP, S0PP, C, B, E0, n_s);
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"H(0)_PP eigenvalues:\n");
    fPrintMatr10(out_stream,E0,n_s,1,0);
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"H(0)_PP eigenvectors in the basis of initial states:\n");
    PrintRowVec(C,n_s,n_s);
    
    
    ci_vec_lin_tr(D.aldet.coef[0],C,n_s,Na,Nb,n_s);
    D.aldet.transpose_ci(0);
    for(int i=0; i<n_s; i++)D.aldet.E_states[0][i]=E0[i];
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"H(0)_PP eigenvectors in the determinant basis:\n\n");
    D.aldet.print_states(0,n_s,1);
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Writing H(0)_PP WaveFunctions:\n");
    sprintf(name,"%s_XMC_H0PP.ci\0",job_name);
    D.aldet.write_civec(0, name);
    fprintf(out_stream,"data file         : %s\n",name);

    
    
    fprintf(out_stream,"\n");
    printf_timer("construction and diagonalization of H(0)_PP");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
//     vec_col_link(OCC_VEC, COR_VEC, ACT_VEC, n_cor, n_act, n_ao);
    
    
    double * d_x1;
    double * d_y1;
    double * d_z1;
    double * d_x_CV;//CV
    double * d_y_CV;
    double * d_z_CV;
    double * d_x_AV;//AV
    double * d_y_AV;
    double * d_z_AV;
    double * d_x_CA;//CA
    double * d_y_CA;
    double * d_z_CA;
    
    d_x1= new double[n_s*n_s];
    d_y1= new double[n_s*n_s];
    d_z1= new double[n_s*n_s];
    d_x_CV = new double[n_cor*n_virt];
    d_y_CV = new double[n_cor*n_virt];
    d_z_CV = new double[n_cor*n_virt];
    d_x_AV = new double[n_act*n_virt];
    d_y_AV = new double[n_act*n_virt];
    d_z_AV = new double[n_act*n_virt];
    d_x_CA = new double[n_cor*n_act];
    d_y_CA = new double[n_cor*n_act];
    d_z_CA = new double[n_cor*n_act];
    
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
    double * H_AV;
    double * H_CA;
    double * H_CV;
    H_AV = new double[n_act*n_virt];
    H_CA = new double[n_cor*n_act];
    H_CV = new double[n_cor*n_virt];

    transform_from_col_MO(H_AV, H_AO, n_ao, ACT_VEC, n_act, VIRT_VEC, n_virt);
    transform_from_col_MO(H_CA, H_AO, n_ao, COR_VEC, n_cor, ACT_VEC, n_act);
    transform_from_col_MO(H_CV, H_AO, n_ao, COR_VEC, n_cor, VIRT_VEC, n_virt);
    
    
    

    fprintf(out_stream,"Start calculation of PT2 correction\n");
    
    RI_data RI;
    
    RI.set_par(M,n_cor, n_act, n_virt);
    
    RI.MO_calc(D.VEC[0], n_ao);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"\n");
    printf_timer("RI orbital transformation");
    fprintf(out_stream,"\n");
    fflush(out_stream);
    res_fit_data RF;//resolvent fitting
    RF.set_par(&(D.aldet), E0, n_s, &RI, eps, n_cor, n_act, n_virt, H_AV, H_CA, H_CV, xmc->edshift);
    RF.gen_grid(xmc->n_fit, xmc->n_fit_pol);
    
    if(xmc->d_only==0){
        RF.E2_calc        (E2);
    }
    
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"H(2):\n");
    fPrintMatr10(out_stream,E2,n_s,n_s,0);
    
    fprintf(out_stream,"\n");
    printf_timer("PT2 energy calculation");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    if(xmc->pt1_d){
        set_zero_matr(d_x1,n_s*n_s);
        set_zero_matr(d_y1,n_s*n_s);
        set_zero_matr(d_z1,n_s*n_s);
    
        fprintf(out_stream,"PT1 dipole moment - d(1):\n\n");
        fprintf(out_stream,"d_x(1)\n");
        RF.P1_calc(d_x1, d_x_AV, d_x_CA, d_x_CV, 1, 1, 1);
        fprintf(out_stream,"\n");
        fprintf(out_stream,"d_y(1)\n");
        RF.P1_calc(d_y1, d_y_AV, d_y_CA, d_y_CV, 1, 1, 1);
        fprintf(out_stream,"\n");
        fprintf(out_stream,"d_z(1)\n");
        RF.P1_calc(d_z1, d_z_AV, d_z_CA, d_z_CV, 1, 1, 1);
        
        symmetrization_with_scaling(d_x1,n_s,2.0);
        symmetrization_with_scaling(d_y1,n_s,2.0);
        symmetrization_with_scaling(d_z1,n_s,2.0);
        
        fprintf(out_stream,"\n");
        fprintf(out_stream,"d(1):\n");
        PrintDipole(d_x1, d_y1, d_z1, n_s);
        
        fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    }
//     exit(0);
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
        
    
        RF.P1_calc(q1, q_AV, q_CA, q_CV, 1, 1, 1);
        
        
        symmetrization_with_scaling(q1,n_s,2.0);
        fPrintMatr(out_stream,q1,n_s,n_s,1);
        
        
    }
#endif
    
    delete[] H_CV;
    
    fprintf(out_stream,"\n\nXMCQDPT2 FINAL RESULTS:\n\n");
    
    
    fprintf(out_stream,"H(2) in H(0)_PP basis:\n");
    fPrintMatr10(out_stream,E2,n_s,n_s,0);
    
    transform_back(E2,E2,C,B,n_s);
    
    fprintf(out_stream,"\nH(2) in initial basis:\n");
    fPrintMatr10(out_stream,E2,n_s,n_s,0);
    
    symmetrization(E2, n_s);
    fprintf(out_stream,"\nH(2) in initial basis (symmetric form):\n");
    fPrintMatr10(out_stream,E2,n_s,n_s,0);

//     transform_back(S2,S2,C,B,n_s);
    
    if(xmc->pt1_d){
        transform_back(d_x1,d_x1,C,B,n_s);
        transform_back(d_y1,d_y1,C,B,n_s);
        transform_back(d_z1,d_z1,C,B,n_s);
    }
    
    //rotation back to CAS basis
    transpose(C,n_s,n_s);
    ci_vec_lin_tr(D.aldet.coef[0],C,n_s,Na,Nb,n_s);
    
    
    if(xmc->d_only==0)for(int i=0;i<n_s*n_s;i++)H2[i]=H01[i]+E2[i];
    
    fprintf(out_stream,"\nH(0-1):\n");
    fPrintMatr10(out_stream,H01,n_s,n_s,0);    
    
    fprintf(out_stream,"\nH(0-2):\n");
    fPrintMatr10(out_stream,H2,n_s,n_s,0);
//         exit(1);
    
    memcpy(S0PP ,D.ACT_DET_A,n_s*n_s*sizeof(double));    
    HC_SCE(H01, S0PP, C, B, E0, n_s);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"E(CAS):\n");
    PrintEnergy(E0, n_s,0);
    
   if(xmc->d_only){
        set_zero_matr(S0PP,n_s*n_s);
        for(int i=0; i<n_s;i++)S0PP[i*n_s+1]=1;
    }
    memcpy(S0PP ,D.ACT_DET_A,n_s*n_s*sizeof(double));
    HC_SCE(H2, S0PP, C, B, E0, n_s);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"E(MP2):\n");
    PrintEnergy(E0, n_s,1);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"H(0-2) eigenvectors in the basis of initial states:\n");
    PrintRowVec(C,n_s,n_s);
    
    ci_vec_lin_tr(D.aldet.coef[0],C,n_s,Na,Nb,n_s);
    D.aldet.transpose_ci(0);
    for(int i=0; i<n_s; i++)D.aldet.E_states[0][i]=E0[i];
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"H(0-2) eigenvectors in the determinant basis:\n\n");
    D.aldet.print_states(0,n_s,1);
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Writing H(0-2) WaveFunctions:\n");
    sprintf(name,"%s_XMC_H2.ci\0",job_name);
    D.aldet.write_civec(0, name);
    fprintf(out_stream,"data file         : %s\n",name);

    
    
    fprintf(out_stream,"\n\nDipole CAS:\n");
    PrintDipole(d_x, d_y, d_z, n_s);
    
    CAS_engine CAS;
    //CAS.init set CI coef to 0, so it must be stored somewhere else
    CAS.init(xmc->cas ,M);
    //restore CI coef
    memcpy(M->CI[0].coef[0],D.aldet.coef[0],Na*Nb*n_s*sizeof(double));
    M->CI[0].transpose_ci(0);

    transform(d_x ,d_x ,C,B,n_s);
    transform(d_y ,d_y ,C,B,n_s);
    transform(d_z ,d_z ,C,B,n_s);
    CAS.Prop_calc();
    CAS.print_properties("XMC(0)");    
    
    if(xmc->pt1_d){
        fprintf(out_stream,"\n\nDipole XMC(0+1):\n");
        transform(d_x1,d_x1,C,B,n_s);
        transform(d_y1,d_y1,C,B,n_s);
        transform(d_z1,d_z1,C,B,n_s);
        for(int i_s=0;i_s<n_s*n_s;i_s++){
            double S = D.ACT_DET_A[i_s];
            
            d_x[i_s] += d_x1[i_s];
            d_y[i_s] += d_y1[i_s];
            d_z[i_s] += d_z1[i_s];
            
        }
        PrintDipole(d_x, d_y, d_z, n_s);
    }
    
    
    
    printf_timer("XMCQDPT2");
    
    D.clear();
    
    delete[] H_CA;
    delete[] H_AV;
    delete[] F;
    delete[] COR_VEC;
    delete[] ACT_VEC;
//     delete[] OCC_VEC;
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

    delete[] eps;
    delete[] B;
    delete[] C;
    
    delete[] E0;
    delete[] H0PP;
    delete[] S0PP;
    delete[] H01;
    delete[] H2;
    delete[] E2;
    delete[] S2;
    delete[] d_x;
    delete[] d_y;
    delete[] d_z;
    delete[] U;
    delete[] name;
    
 
    return 0;
}
