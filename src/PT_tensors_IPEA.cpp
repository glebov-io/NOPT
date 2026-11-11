# include <stdio.h>
# include "blas_link.h"
# include <omp.h>

# include "CI.h"
# include "matr.h"
# include "timer.h"

# include "from_hash.h"
# include "aldet.h"
# include "RI.h"
# include "PT_tensors.h"
# include "ipea_matrices.h"

// #include "backup_and_old_versions/aldet_Nbody_operators.cpp"

# define max(a,b)  (((a)<(b))?(b):(a))
# define min(a,b)  (((a)>(b))?(b):(a))

extern int num_threads;




PT_tensors::PT_tensors(){
            
    RF_PH    = nullptr;
    RF_PV    = nullptr;
    RF_PV_JK = nullptr;
    RF_PV_AB = nullptr;
    RF_P3_JK = nullptr;
    RF_P3_AB = nullptr;
    
    e_IP     = nullptr;
    e_EA     = nullptr;
    e_IP_2   = nullptr;
    e_EA_2   = nullptr;
    e_IP_AB  = nullptr;
    e_EA_AB  = nullptr;
    
    IP_U     = nullptr;
    EA_U     = nullptr;
    IP_Um    = nullptr;
    EA_Um    = nullptr;
    IP_Up    = nullptr;
    EA_Up    = nullptr;
     
    IP_2_U   = nullptr;
    EA_2_U   = nullptr;
    IP_2_Um  = nullptr;
    EA_2_Um  = nullptr;
    IP_2_Up  = nullptr;
    EA_2_Up  = nullptr;
    
    IP_AB_U  = nullptr;
    EA_AB_U  = nullptr;
    IP_AB_Um = nullptr;
    EA_AB_Um = nullptr;
    IP_AB_Up = nullptr;
    EA_AB_Up = nullptr;
    
    
    

}


int PT_tensors::set_par(RI_data * ext_integrals,
                        double * ext_eps,
                        int ext_n_c, int ext_n_a, int ext_n_v, 
                        double * ext_H_AV, double * ext_H_CA, double * ext_H_CV,
                        double ext_edshift, molecule * A){
    
    integrals      = ext_integrals     ; 
    n_c     = ext_n_c    ; 
    n_a     = ext_n_a    ;
    n_v     = ext_n_v    ;
    H_AV    = ext_H_AV   ;
    H_CA    = ext_H_CA   ;
    H_CV    = ext_H_CV   ;
    edshift = ext_edshift;
        
    e_c     = ext_eps        ; 
    e_a     = ext_eps+n_c    ; 
    e_v     = ext_eps+n_c+n_a;
    
    n_active_el_alp = A->n_act_el_alp[0];
    n_active_el_bet = A->n_act_el_bet[0];
    
    e_IP    = new double[n_a];for(int i=0; i<n_a; i++) e_IP[i]=e_a[i];
    e_EA    = new double[n_a];for(int i=0; i<n_a; i++) e_EA[i]=e_a[i];
    IP_U    = new double[n_a*n_a];
    EA_U    = new double[n_a*n_a];
    IP_Um   = new double[n_a*n_a];
    EA_Um   = new double[n_a*n_a];
    IP_Up   = new double[n_a*n_a];
    EA_Up   = new double[n_a*n_a];
    e_IP_2  = new double[n_a*n_a];
    e_EA_2  = new double[n_a*n_a];
    IP_2_U  = new double[n_a*n_a*n_a*n_a];
    EA_2_U  = new double[n_a*n_a*n_a*n_a];
    IP_2_Um = new double[n_a*n_a*n_a*n_a];
    EA_2_Um = new double[n_a*n_a*n_a*n_a];
    IP_2_Up = new double[n_a*n_a*n_a*n_a];
    EA_2_Up = new double[n_a*n_a*n_a*n_a];

    e_IP_AB  = new double[n_a*n_a];
    e_EA_AB  = new double[n_a*n_a];
    IP_AB_U  = new double[n_a*n_a*n_a*n_a];
    EA_AB_U  = new double[n_a*n_a*n_a*n_a];
    IP_AB_Um = new double[n_a*n_a*n_a*n_a];
    EA_AB_Um = new double[n_a*n_a*n_a*n_a];
    IP_AB_Up = new double[n_a*n_a*n_a*n_a];
    EA_AB_Up = new double[n_a*n_a*n_a*n_a];
        

    
    RF_PH    = new double  [n_a*n_a                ];
    RF_PV    = new double  [n_a*n_a*n_a*n_a        ];
    RF_PV_JK = new double  [n_a*n_a*n_a*n_a        ];
    RF_PV_AB = new double  [n_a*n_a*n_a*n_a        ];
    RF_P3_JK = new double  [n_a*n_a*n_a*n_a*n_a*n_a];
    RF_P3_AB = new double  [n_a*n_a*n_a*n_a*n_a*n_a];
    
    return 0;
}

int M_average(double * A, double * M, int n, int m, std::vector<double> w){
    
    //A -average (output)
    //M - matrix (input)
    //n - dim A
    //l=n*m - dim M
    //w - weight
    
    double norm=w[0];
    int l=n*m;
    double * M_a=M;
    for(int i=0;i<n;i++)
    for(int j=0;j<n;j++){
        A[i*n+j]=M[i*l+j]*w[0];
    }

    for(int a=1;a<w.size();a++){
        norm+=w[a];
        M_a+=n*(l+1);
        for(int i=0;i<n;i++)
        for(int j=0;j<n;j++){
            A[i*n+j]+=M_a[i*l+j]*w[a];
        }
    
    }

    for(int i=0;i<n*n;i++)
        A[i]=A[i]/norm;
    
    return 0;
}

// int gen_IPEA_matr(double * eps, double * U, double * Um, double * Up, 
//                   int dim, int n_s, aldet_data * E, double * E0, std::vector<double> avecoe, double sign){
//     
//    
//     double * S = new double[n_s*dim*n_s*dim];
//     double * H = new double[n_s*dim*n_s*dim];
//     
//     double * H_av = new double[dim*dim  ];
//     
//     E->calc_S(S,0,0);
//     E->H_calc(H,n_s*dim);
//         
//     double * S_part;
//     double * H_part;
//     for(int i=0;i<n_s;i++){
//         S_part = S+i*(n_s*dim*dim+dim);
//         H_part = H+i*(n_s*dim*dim+dim);
//         for(int j=0;j<dim;j++)
//         for(int k=0;k<dim;k++)
//             H_part[j*dim*n_s+k] =sign*(H_part[j*dim*n_s+k]-S_part[j*dim*n_s+k]*E0[i*(n_s+1)]);
//     }
//     
//     
//     
//         
//     M_average(U, S, dim, n_s, avecoe);
//     fprintf(out_stream,"S:\n");
//     PrintMatr(U,dim,dim,0);
//     fprintf(out_stream,"H:\n");
//     PrintMatr(H,dim,dim,0);
//     M_average(H_av, H, dim, n_s, avecoe);
//     HC_SCE_low(H_av  ,U  ,Um  ,Up  , eps  , dim  , 1e-6);
//     
//     
//     delete[] S   ;
//     delete[] H   ;
//     delete[] H_av;
//     
//     
//     return 0;
//     
// }

// int double_inv_vector(double * I_D, double * D, double *V, int n){
//     
//     int n2 = n*(n-1)/2;
// //     double * D = new double[n2*n2];
//     int t=0;
//     int u;
//     for(int i=0  ;i<n;i++)
//     for(int j=i+1;j<n;j++,t++){
//         u=0;
//         for(int k=0  ;k<n;k++)
//         for(int l=k+1;l<n;l++,u++)
//             D[t*n2+u]=V[i*n+k]*V[j*n+l]-V[i*n+l]*V[j*n+k];
//         
//     }
// //     fprintf(out_stream,"\n\n");
// //     PrintMatr(V,n,n,1);
//     
// //     PrintMatr(D,n2,n2,1);
// //     double * D2 = new double[n2*n2];
//     memcpy(I_D,D,sizeof(double)*n2*n2);
//     double * D3 = new double[n2*n2];
//     set_zero_matr(D3,n2*n2);
//     
// //     fprintf(out_stream,"\n\nD");
// //     PrintMatr(I_D,n2,n2,1);
//     
//     inv_matr_constr(I_D,n2);
//     
//     transpose(I_D,n2,n2);
// 
// //     fprintf(out_stream,"\n\nD-1");
// //     PrintMatr(D,n2,n2,1);
// //     set_zero_matr(I_D,n*n*n*n);
// //     t=0;
//     int v;
// //     for(int i=0  ;i<n;i++)
// //     for(int j=i+1;j<n;j++,t++){
// //         u=0;
// //         for(int k=0  ;k<n;k++)
// //         for(int l=k+1;l<n;l++,u++){
// //         v=0;
// //         for(int m=0  ;m<n;m++)
// //         for(int p=m+1;p<n;p++,v++){
// //             I_D[((i*n+j)*n+k)*n+l]= D[u*n2+t];
// //             I_D[((j*n+i)*n+k)*n+l]=-D[u*n2+t];
// //             I_D[((i*n+j)*n+l)*n+k]=-D[u*n2+t];
// //             I_D[((j*n+i)*n+l)*n+k]= D[u*n2+t];
//             
// //             I_D[((i*n+j)*n+k)*n+l]
// //             D3[t*n2+u]+=D[v*n2+u]*(V[i*n+m]*V[j*n+p]-V[i*n+p]*V[j*n+m]);
// //         }
// //         }
// //     }
//     t=0;
//     for(int i=0  ;i<n;i++)
//     for(int j=i+1;j<n;j++,t++){
//         u=0;
//         for(int k=0  ;k<n;k++)
//         for(int l=k+1;l<n;l++,u++){
//         v=0;
//         for(int m=0  ;m<n;m++)
//         for(int p=m+1;p<n;p++,v++){
// //             D3[t*n2+u]+=D[v*n2+u]*(V[i*n+m]*V[j*n+p]-V[i*n+p]*V[j*n+m]);
//             D3[t*n2+u]+=I_D[u*n2+v]*(V[i*n+m]*V[j*n+p]-V[i*n+p]*V[j*n+m]);
//         }
//         }
//     }
// //     for(int t=0;t<n2;t++){
// //     for(int u=0;u<n2;u++){
// //     for(int v=0;v<n2;v++){
// //             D3[t*n2+u]+=D[v*n2+u]*D2[t*n2+v ];//(V[i*n+m]*V[j*n+p]-V[i*n+p]*V[j*n+m]);
// //         }
// //         }
// //     }
// //     fprintf(out_stream,"\n\nE\n");
// //     PrintMatr(D3,n2,n2,1);
//     
//     
// //     delete[] D;
//     
//     return 0;
//     
//     
// }



// int gen_IP_matr(double * eps, double * U, double * Um, double * Up, 
//                   int n_act, int n_s, aldet_data * E, double * E0, std::vector<double> avecoe, double sign){
//     
//    
// //     double * gamma = new double[n_s*n_act*n_s*n_act];
// //     double * H = new double[n_s*n_act*n_s*n_act];
//     
// //     set_zero_matr(gamma,n_act*n_act*n_s*n_s);
//      E->calc_av_DMA(U,0,avecoe);
// //     E->calc_DMB(gamma,0,0);
//     
// //     average_DM_PT(U,gamma, avecoe,n_act*n_act,n_s);
//     
//     exit(0);
//     
// //     double * H_av = new double[n_act*n_act  ];
// /*    
//     E->calc_S(S,0,0);
//     E->H_calc(H,n_s*n_act);
//         
//     double * S_part;
//     double * H_part;
//     for(int i=0;i<n_s;i++){
//         S_part = S+i*(n_s*n_act*n_act+n_act);
//         H_part = H+i*(n_s*n_act*n_act+n_act);
//         for(int j=0;j<n_act;j++)
//         for(int k=0;k<n_act;k++)
//             H_part[j*n_act*n_s+k] =sign*(H_part[j*n_act*n_s+k]-S_part[j*n_act*n_s+k]*E0[i*(n_s+1)]);
//     }
//     
//     
//     
//     M_average(H_av, H, n_act, n_s, avecoe);
//         
//     M_average(U, S, n_act, n_s, avecoe);
//     fprintf(out_stream,"H\n:");
//     PrintMatr(S,n_act,n_act,0);
//     HC_SCE_low(H_av  ,U  ,Um  ,Up  , eps  , n_act  , 1e-6);*/
//     
//     
// //     delete[] S   ;
// //     delete[] H   ;
// //     delete[] H_av;
//     
//     
//     return 0;
//     
// }
// 

int PT_tensors::IPEA(casci_solver * I, std::vector<double> avecoe){
        
    if(n_a!= I->n_act()){
        fprintf(out_stream,"ERROR in IPEA\n");
        exit(0);
    }
    
    int n_s =I->n_states();
    
    double * IP_H   = new double[n_a*n_a];
    double * EA_H   = new double[n_a*n_a];
    
    I->calc_IPEA_single(IP_U, IP_H, EA_U, EA_H, 0 ,avecoe);
    
    
    IPEA_solve(IP_H, EA_H);
    
    delete[] IP_H   ;
    delete[] EA_H   ;
    
    return 0;
    
}

int PT_tensors::IPEA(const double * g1, const double * g2,
                     const double * gamma, const double * GAMMA){
    
    std::vector<double> IP_H(n_a*n_a);
    std::vector<double> EA_H(n_a*n_a);
    
    ipea_matrices(n_a, g1, g2, gamma, GAMMA, IP_U, IP_H.data(), EA_U, EA_H.data());
    
    printf_timer("calculation of IPEA matrices");
    
    IPEA_solve(IP_H.data(), EA_H.data());
    
    return 0;
    
}

int PT_tensors::IPEA_solve(double * IP_H, double * EA_H){
    
    symmetrization(IP_H   , n_a);
    symmetrization(EA_H   , n_a);
    
    symmetrization(IP_U   , n_a);
    symmetrization(EA_U   , n_a);
    
    HC_SCE_low(IP_H   ,IP_U,IP_Um,IP_Up,e_IP,n_a, 1e-6);
    
    fprintf(out_stream, "IP:\n");
    fPrintMatr(out_stream, e_IP,1,n_a,0);
    fprintf(out_stream, "IP coefficients:\n");
    fPrintMatr(out_stream, IP_U,n_a,n_a,0);
    
    HC_SCE_low(EA_H   ,EA_U,EA_Um,EA_Up,e_EA,n_a, 1e-6);
    
    fprintf(out_stream, "EA:\n");
    fPrintMatr(out_stream, e_EA,1,n_a,0);
    fprintf(out_stream, "EA coefficients:\n");
    fPrintMatr(out_stream, EA_U,n_a,n_a,0);
    

    return 0;
    
}

int PT_tensors::MPPT(aldet_data * I, int i_set, std::vector<double> avecoe){
        
    if(n_a!= I->n_act){
        fprintf(out_stream,"ERROR in MPPT\n");
        exit(0);
    }
    
    int n_s =I->n_states[i_set];
    
    double * IP_H   = new double[n_a*n_a];
    double * EA_H   = new double[n_a*n_a];
    
    
    I->calc_IPEA_single(IP_U, IP_H, EA_U, EA_H, 0 ,avecoe);
    
//     PrintMatr(IP_H,n_a,n_a,1);
//     PrintMatr(IP_U,n_a,n_a,1);
    
    
    
    for(int i=0;i<n_a;i++)e_IP[i]=IP_H[i*n_a+i]/IP_U[i*n_a+i];
    for(int i=0;i<n_a;i++)e_EA[i]=EA_H[i*n_a+i]/EA_U[i*n_a+i];
    fprintf(out_stream,"\n\nMPPT orbital enegries:\n");
    fprintf(out_stream,"IP:       ");fPrintMatr(out_stream,e_IP,1,n_a,0);
    fprintf(out_stream,"EA:       ");fPrintMatr(out_stream,e_EA,1,n_a,0);
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
    delete[] IP_H   ;
    delete[] EA_H   ;
    return 0;
    
}



int PT_tensors::set_zero(){
    
    return 0;
}

int PT_tensors::symm(){
    
    return 0;
}

int PT_tensors::E2_calc_IPEA(){
    
    set_zero_matr(RF_P3_JK, n_a*n_a*n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_AB, n_a*n_a*n_a*n_a*n_a*n_a);
    
    set_zero_matr(RF_PV_JK, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_PV_AB, n_a*n_a*n_a*n_a);
    
    set_zero_matr(RF_PH, n_a*n_a);
    
//     set_zero_matr(RF_PS, N_fit);
    RF_PS=0;
    
    calc_IPEA_2_CCVV();printf_timer("CCVV res table");fprintf(out_stream,"\n");fflush(out_stream);
    calc_IPEA_2_CAVV();printf_timer("CAVV res table");fprintf(out_stream,"\n");fflush(out_stream);
    calc_IPEA_2_AAVV();printf_timer("AAVV res table");fprintf(out_stream,"\n");fflush(out_stream);
    calc_IPEA_2_CCAV();printf_timer("CCAV res table");fprintf(out_stream,"\n");fflush(out_stream);
    calc_IPEA_2_CCAA();printf_timer("CCAA res table");fprintf(out_stream,"\n");fflush(out_stream);
    calc_IPEA_2_CV  ();printf_timer("CV   res table");fprintf(out_stream,"\n");fflush(out_stream);
    calc_IPEA_2_AV  ();printf_timer("AV   res table");fprintf(out_stream,"\n");fflush(out_stream);
    calc_IPEA_2_CA  ();printf_timer("CA   res table");fprintf(out_stream,"\n");fflush(out_stream);
    
    symmetrization(RF_PH, n_a);
    symmetrization(RF_PV_JK, n_a*n_a);
    symmetrization(RF_PV_AB, n_a*n_a);
    symmetrization(RF_P3_JK, n_a*n_a*n_a);
    symmetrization(RF_P3_AB, n_a*n_a*n_a);
    
    double * RF_P3_tmp = new double  [n_a*n_a*n_a*n_a*n_a*n_a];
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++) //// ??? TODO parallel #pragma omp parallel for
    for(int y=0; y<n_a; y++){
        RF_P3_tmp[((((t*n_a+u)*n_a+v)*n_a+w)*n_a+x)*n_a+y]=
        RF_P3_AB    [((((t*n_a+u)*n_a+x)*n_a+v)*n_a+w)*n_a+y];
    }
    memcpy(RF_P3_AB,RF_P3_tmp,n_a*n_a*n_a*n_a*n_a*n_a*sizeof(double));
    
    delete[]RF_P3_tmp;
    
    return 0;
    
}

int PT_tensors::calc_IPEA_2_CV(){
    
    double dE;
    double V;
    
    
    double * RF_MM;    // (J-K) * (J-K)
    double * RF_JJ;    //  J(aa)* J(aa)
    double * RF_JM;    //  J(aa)*(J-K)
    double * RF_MJ;    // (J-K) * J(aa)
    double * RF_AB;    //  J(ab)* J(ab)
    
    RF_MM = new double  [n_a*n_a*n_a*n_a];
    RF_JJ = new double  [n_a*n_a*n_a*n_a];
    RF_JM = new double  [n_a*n_a*n_a*n_a];
    RF_MJ = new double  [n_a*n_a*n_a*n_a];
    RF_AB = new double  [n_a*n_a*n_a*n_a];
    set_zero_matr(RF_MM, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_JJ, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_JM, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_MJ, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_AB, n_a*n_a*n_a*n_a);
    
    double * RF_HM;
    double * RF_MH;
    double * RF_HJ;
    double * RF_JH;
    RF_HM = new double  [n_a*n_a];
    RF_MH = new double  [n_a*n_a];
    RF_HJ = new double  [n_a*n_a];
    RF_JH = new double  [n_a*n_a];
    set_zero_matr(RF_HM, n_a*n_a);
    set_zero_matr(RF_MH, n_a*n_a);
    set_zero_matr(RF_HJ, n_a*n_a);
    set_zero_matr(RF_JH, n_a*n_a);
    
    double RF_H=0;
//     RF_H = new double  [n_a];
//     set_zero_matr(RF_H, n_a);
    
    double * VCAA;
    VCAA = new double[n_c*n_v*n_a*n_a];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_c*n_v,n_a*n_a,integrals->aux_n_ao,1.0,
                        integrals->VC_RI_M,integrals->aux_n_ao,
                        integrals->AA_RI_M,integrals->aux_n_ao,0.0,
                        VCAA,n_a*n_a);
    double * CAVA;
    CAVA = new double[n_a*n_c*n_a*n_v];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_a*n_c,n_a*n_v,integrals->aux_n_ao,1.0,
                        integrals->CA_RI_M,integrals->aux_n_ao,
                        integrals->VA_RI_M,integrals->aux_n_ao,0.0,
                        CAVA,n_a*n_v);
    
    
    for(int i=0; i<n_c; i++)
    for(int u=0; u<n_a; u++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++){
        dE=e_c[i]-e_v[a];//is taken negative
        dE=1.0/dE;
        
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //<i(a)t(b)|u(a)a(b)><v(a)a(b)|i(a)w(b)>=(iu|at)(iv|aw)
            V=CAVA[((i*n_a+u)*n_v+a)*n_a+t]*
              CAVA[((i*n_a+v)*n_v+a)*n_a+w];
              
            RF_AB[((t*n_a+u)*n_a+v)*n_a+w]+=V*dE;
            
        }
    }
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE=e_c[i]-e_v[a];//is taken negative
        dE=1.0/dE;
        
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //(<it|au>-<iu|at>)(<av|iw>-<av|wi>)
            V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
               CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
              (VCAA[((a*n_c+i)*n_a+w)*n_a+v]-  //<av|iw>=(ai|wv)
               CAVA[((i*n_a+v)*n_v+a)*n_a+w]); //<av|wi>=(iv|aw)
            
            RF_MM[((t*n_a+u)*n_a+v)*n_a+w]+=V*dE;
        }
    }
    

    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE=e_c[i]-e_v[a];//is taken negative
        dE=1.0/dE;
        
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //(<it|au>-<iu|at>)<av|iw>
            V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
               CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
               VCAA[((a*n_c+i)*n_a+w)*n_a+v];  //<av|iw>=(ai|wv)
            
            RF_MJ[((t*n_a+u)*n_a+v)*n_a+w]+=V*dE;
        }
    }
    
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE=e_c[i]-e_v[a];
        dE=1.0/dE;
        
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //<it|au>(<av|iw>-<av|wi>)
            V= VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
              (VCAA[((a*n_c+i)*n_a+w)*n_a+v]-  //<av|iw>=(ai|wv)
               CAVA[((i*n_a+v)*n_v+a)*n_a+w]); //<av|wi>=(iv|aw)
            
            RF_JM[((t*n_a+u)*n_a+v)*n_a+w]+=V*dE;
        }
    }

    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE=e_c[i]-e_v[a];
        dE=1.0/dE;
        
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //<it|au><av|iw>
            V=VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
              VCAA[((a*n_c+i)*n_a+w)*n_a+v];  //<av|iw>=(ai|wv)
            
            RF_JJ[((t*n_a+u)*n_a+v)*n_a+w]+=V*dE;
        }
    }
//     printf_timer("opt table");
    
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
           CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
           H_CV[i*n_v+a]; 
        dE=e_c[i]-e_v[a];
        RF_MH[t*n_a+u]+=V/dE;
    }
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
           CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
           H_CV[i*n_v+a]; 
        
        dE=e_c[i]-e_v[a];
        
        RF_HM[t*n_a+u]+=V/dE;
    }

    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V= VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
           H_CV[i*n_v+a]; 
        dE=e_c[i]-e_v[a];
        RF_JH[t*n_a+u]+=V/dE;
        
    }
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0  ;t<n_a;t++)
    for(int u=0  ;u<n_a;u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V= VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
           H_CV[i*n_v+a]; 
        dE=e_c[i]-e_v[a];
        RF_HJ[t*n_a+u]+=V/dE;
        
    }
    
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++){
        //H[i,a]*H[i,a]
        V=H_CV[i*n_v+a]*H_CV[i*n_v+a]; 
        dE=e_c[i]-e_v[a];
        RF_H+=V/dE;
        
    }
    
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
        RF_PV_JK[((u*n_a+t)*n_a+v)*n_a+w]+=+RF_MM[((t*n_a+v)*n_a+u)*n_a+w]
                                           +RF_JJ[((t*n_a+v)*n_a+u)*n_a+w]
                                           -RF_MM[((u*n_a+v)*n_a+t)*n_a+w]
                                           -RF_JJ[((u*n_a+v)*n_a+t)*n_a+w]
                                           -RF_MM[((t*n_a+w)*n_a+u)*n_a+v]
                                           -RF_JJ[((t*n_a+w)*n_a+u)*n_a+v]
                                           +RF_MM[((u*n_a+w)*n_a+t)*n_a+v]
                                           +RF_JJ[((u*n_a+w)*n_a+t)*n_a+v]
                                           ;
                                                       
    
    
    
//     set_zero_matr(RF_PV_AB, n_a*n_a*n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
        RF_PV_AB[((t*n_a+u)*n_a+v)*n_a+w]+=+RF_JM[((t*n_a+v)*n_a+u)*n_a+w]
                                           +RF_MJ[((t*n_a+v)*n_a+u)*n_a+w]
                                           +RF_JM[((u*n_a+w)*n_a+t)*n_a+v]
                                           +RF_MJ[((u*n_a+w)*n_a+t)*n_a+v]
                                           -RF_AB[((t*n_a+w)*n_a+u)*n_a+v]
                                           -RF_AB[((u*n_a+v)*n_a+t)*n_a+w];
    
//     set_zero_matr(RF_PH,N_fit*n_a*n_a);
    for(int t=0; t<n_a; t++)
    for(int w=0; w<n_a; w++){
        RF_PH[t*n_a+w]+= RF_HM[w*n_a+t]
                        +RF_MH[t*n_a+w]
                        +RF_HJ[w*n_a+t]
                        +RF_JH[t*n_a+w];
        for(int u=0; u<n_a; u++)
             RF_PH[t*n_a+w]+= RF_MM[((t*n_a+u)*n_a+u)*n_a+w]
                             +RF_JJ[((t*n_a+u)*n_a+u)*n_a+w]
                             +RF_AB[((t*n_a+u)*n_a+u)*n_a+w];
    }
    
    RF_PS+=2*RF_H;
    
    delete[] RF_MM ;
    delete[] RF_JJ ;
    delete[] RF_JM ;
    delete[] RF_MJ ;
    delete[] RF_AB ;
    delete[] RF_HM ;
    delete[] RF_MH ;
    delete[] RF_HJ ;
    delete[] RF_JH ;
    delete[] VCAA  ;
    delete[] CAVA  ;
//     delete[] VCAA_J;
//     delete[] VCAA_K;
    
    return 0;
    
}

int PT_tensors::calc_IPEA_2_AV(){
    
    double dE;
    
    int aux_n_ao = integrals->aux_n_ao;
    
    
    double * VAAA;
    double * VmuAA;
    double * U;
    double * H_muV;
    double * RF_P3_JJ;
    double * RF_P3_JM;
    double * RF_P3_MJ;
    double * RF_P3_HM;
    double * RF_P3_MH;
    double * RF_P3_HJ;
    double * RF_P3_JH;
    
    VAAA     = new double[n_v*n_a*n_a*n_a];
    VmuAA     = new double[n_v*n_a*n_a*n_a];
    U        = new double[n_a*n_a];
    H_muV     = new double[n_v*n_a];
    
    RF_P3_JJ = new double[n_a*n_a*n_a*n_a*n_a*n_a];
    RF_P3_JM = new double[n_a*n_a*n_a*n_a*n_a*n_a];
    RF_P3_MJ = new double[n_a*n_a*n_a*n_a*n_a*n_a];
    
    RF_P3_HM = new double[n_a*n_a*n_a*n_a];
    RF_P3_MH = new double[n_a*n_a*n_a*n_a];
    RF_P3_HJ = new double[n_a*n_a*n_a*n_a];
    RF_P3_JH = new double[n_a*n_a*n_a*n_a];
    
    set_zero_matr(RF_P3_JJ, n_a*n_a*n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_JM, n_a*n_a*n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_MJ, n_a*n_a*n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_HM, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_MH, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_HJ, n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_JH, n_a*n_a*n_a*n_a);
    
    
    
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                n_v*n_a,n_a*n_a,aux_n_ao,1.0,
                integrals->VA_RI_M,aux_n_ao,
                integrals->AA_RI_M,aux_n_ao,0.0,
                VAAA,n_a*n_a);
    
    set_zero_matr(VmuAA,n_v*n_a*n_a*n_a);
    for(int a=0; a<n_v; a++){
        set_zero_matr(U,n_a*n_a);
        for(int u=0; u<n_a; u++){
            dE=1.0/(-e_v[a]+e_IP[u]);
            for(int t=0; t<n_a; t++)
            for(int w=0; w<n_a; w++)
                U[t*n_a+w]+=IP_Um[u*n_a+t]*IP_Up[u*n_a+w]*dE; //// TODO 2 matrix mults -> 1+1 mults ???
        }
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    n_a,n_a*n_a,n_a,1.0,
                    U,n_a,
                    VAAA+a*n_a*n_a*n_a,n_a*n_a,0.0,
                    VmuAA+a*n_a*n_a*n_a,n_a*n_a);
    
    }
    
    set_zero_matr(H_muV,n_v*n_a);
    
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++){
        dE=-e_v[a]+e_IP[t];
        dE=1.0/dE;
            
        for(int t2=0;t2<n_a;t2++)
        for(int t3=0;t3<n_a;t3++){
            H_muV[t2*n_v+a]+=IP_Um[t*n_a+t2]*dE*IP_Up[t*n_a+t3]*H_AV[t3*n_v+a];
            
        }
    }

    
    
    //PT tensors calculation
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int y=0; y<n_a; y++)
    for(int x=0; x<n_a; x++)
    for(int w=0; w<n_a; w++)
    for(int a=0; a<n_v; a++)
    {
        //<aw|xy>-<aw|yx>=(yw|xa)-(xw|ya)
        //<a(a)w(b)|x(a)y(b)>=(yw|xa)
        //MM
        RF_P3_JK[((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w]+=(VmuAA[((a*n_a+t)*n_a+v)*n_a+u]-VmuAA[((a*n_a+u)*n_a+v)*n_a+t])*
                                                           (VAAA[((a*n_a+x)*n_a+w)*n_a+y]-VAAA[((a*n_a+y)*n_a+w)*n_a+x]);
        //JJ    
        RF_P3_JJ[((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w]+= VmuAA[((a*n_a+u)*n_a+v)*n_a+t]*
                                                            VAAA[((a*n_a+x)*n_a+w)*n_a+y];
        //MJ
        RF_P3_MJ[((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w]+=(VmuAA[((a*n_a+t)*n_a+v)*n_a+u]-VmuAA[((a*n_a+u)*n_a+v)*n_a+t])*
                                                            VAAA[((a*n_a+x)*n_a+w)*n_a+y];
        //JM
        RF_P3_JM[((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w]+= VmuAA[((a*n_a+u)*n_a+v)*n_a+t]*
                                                           (VAAA[((a*n_a+x)*n_a+w)*n_a+y]-VAAA[((a*n_a+y)*n_a+w)*n_a+x]);
    }
    //H
    for(int t=0; t<n_a; t++)
    for(int w=0; w<n_a; w++)
    for(int a=0; a<n_v; a++)
        RF_PH[t*n_a+w]+=H_muV[t*n_v+a]*H_AV[w*n_v+a];
    
    //HV
    for(int t=0; t<n_a; t++)
    for(int y=0; y<n_a; y++)
    for(int x=0; x<n_a; x++)
    for(int w=0; w<n_a; w++)
    for(int a=0; a<n_v; a++)
    {
        RF_P3_HJ[((t*n_a+y)*n_a+x)*n_a+w]+=H_muV[t*n_v+a]* VAAA[((a*n_a+x)*n_a+w)*n_a+y];//HJ
        RF_P3_HM[((t*n_a+y)*n_a+x)*n_a+w]+=H_muV[t*n_v+a]*(VAAA[((a*n_a+x)*n_a+w)*n_a+y]-
                                                          VAAA[((a*n_a+y)*n_a+w)*n_a+x]);//HM
    }
    //VH
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int a=0; a<n_v; a++)
    {
        RF_P3_JH[((t*n_a+u)*n_a+v)*n_a+w]+= VmuAA[((a*n_a+u)*n_a+v)*n_a+t]*H_AV[w*n_v+a];//JH//<a(a)w(b)|x(a)y(b)>=(yw|xa)
        RF_P3_MH[((t*n_a+u)*n_a+v)*n_a+w]+= VmuAA[((a*n_a+t)*n_a+v)*n_a+u]*H_AV[w*n_v+a];//MH//<aw|xy>-<aw|yx>    =(yw|xa)-(xw|ya)
        RF_P3_MH[((t*n_a+u)*n_a+v)*n_a+w]+=-VmuAA[((a*n_a+u)*n_a+v)*n_a+t]*H_AV[w*n_v+a];//MH//<aw|xy>-<aw|yx>    =(yw|xa)-(xw|ya)
    }
    
    //reordering
    //RF_PV_JK
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
        RF_PV_JK[((w*n_a+t)*n_a+v)*n_a+u]+= RF_P3_HM[((t*n_a+u)*n_a+v)*n_a+w]
                                           -RF_P3_HM[((w*n_a+u)*n_a+v)*n_a+t]
                                           -RF_P3_MH[((t*n_a+w)*n_a+v)*n_a+u]
                                           +RF_P3_MH[((t*n_a+w)*n_a+u)*n_a+v];
    //RF_PV_AB
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++){
        RF_PV_AB[((t*n_a+u)*n_a+v)*n_a+w]+= RF_P3_HJ[((t*n_a+w)*n_a+v)*n_a+u]
                                           +RF_P3_JH[((u*n_a+t)*n_a+w)*n_a+v]
                                           +RF_P3_HJ[((u*n_a+v)*n_a+w)*n_a+t]
                                           +RF_P3_JH[((t*n_a+u)*n_a+v)*n_a+w];
    
        for(int x=0; x<n_a; x++)
            RF_PV_AB[((t*n_a+u)*n_a+v)*n_a+w] += RF_P3_JJ[((((t*n_a+u)*n_a+x)*n_a+v)*n_a+w)*n_a+x]
                                                +RF_P3_JJ[((((u*n_a+t)*n_a+x)*n_a+w)*n_a+v)*n_a+x];
    }
    //RF_P3_AB - order differs from the original XMCQDPT code (res_fit.cpp)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int x=0; x<n_a; x++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int y=0; y<n_a; y++)
        RF_P3_AB[((((t*n_a+u)*n_a+x)*n_a+v)*n_a+w)*n_a+y]+=-RF_P3_MJ[((((t*n_a+u)*n_a+v)*n_a+y)*n_a+w)*n_a+x]
                                                           +RF_P3_MJ[((((t*n_a+u)*n_a+w)*n_a+y)*n_a+v)*n_a+x]
                                                           +RF_P3_JM[((((x*n_a+t)*n_a+y)*n_a+w)*n_a+v)*n_a+u]
                                                           -RF_P3_JM[((((x*n_a+u)*n_a+y)*n_a+w)*n_a+v)*n_a+t]
                                                           +RF_P3_JJ[((((t*n_a+x)*n_a+v)*n_a+w)*n_a+y)*n_a+u]
                                                           -RF_P3_JJ[((((t*n_a+x)*n_a+w)*n_a+v)*n_a+y)*n_a+u]
                                                           -RF_P3_JJ[((((u*n_a+x)*n_a+v)*n_a+w)*n_a+y)*n_a+t]
                                                           +RF_P3_JJ[((((u*n_a+x)*n_a+w)*n_a+v)*n_a+y)*n_a+t];
                                    
    
    delete[] VAAA  ;
    delete[] VmuAA  ;
    delete[] U     ;
    delete[] H_muV  ;
    delete[] RF_P3_JJ;
    delete[] RF_P3_JM;
    delete[] RF_P3_MJ;
    delete[] RF_P3_HM;
    delete[] RF_P3_MH;
    delete[] RF_P3_HJ;
    delete[] RF_P3_JH;
    
    
    
    return 0;
    
}

int PT_tensors::calc_IPEA_2_CA(){
    
    double dE;
    double H1a;
    double H2a;
    double V;
    
    
    int aux_n_ao = integrals->aux_n_ao;
    
    double * CAAA;
    CAAA = new double[n_c*n_a*n_a*n_a];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_c*n_a,n_a*n_a,aux_n_ao,1.0,
                        integrals->CA_RI_M,aux_n_ao,
                        integrals->AA_RI_M,aux_n_ao,0.0,
                        CAAA,n_a*n_a);
    
    
    double * RF_P3_VV;
    double * RF_P3_VH;
    double * RF_P3_HV;
    double * RF_P3_HH;
    
    RF_P3_VV = new double[n_a*n_a*n_a*n_a*n_a*n_a];
    RF_P3_VH = new double[n_a*n_a*n_a*n_a        ];
    RF_P3_HV = new double[n_a*n_a*n_a*n_a        ];
    RF_P3_HH = new double[n_a*n_a                ];
    
    set_zero_matr(RF_P3_VV, n_a*n_a*n_a*n_a*n_a*n_a);
    set_zero_matr(RF_P3_VH, n_a*n_a*n_a*n_a        );
    set_zero_matr(RF_P3_HV, n_a*n_a*n_a*n_a        );
    set_zero_matr(RF_P3_HH, n_a*n_a                );
    
    
    
    //VV
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE= 1.0/(e_c[i]-e_EA[v]);
        
        for(int v2=0;v2<n_a;v2++)
        for(int v3=0;v3<n_a;v3++)
        for(int y=0; y<n_a; y++)
        for(int x=0; x<n_a; x++)
        for(int w=0; w<n_a; w++){
            
            V=(CAAA[((i*n_a+v3)*n_a+u)*n_a+t]*
               CAAA[((i*n_a+w)*n_a+x)*n_a+y]);
            
            RF_P3_VV[((((t*n_a+u)*n_a+v2)*n_a+y)*n_a+x)*n_a+w]+=EA_Um[v*n_a+v2]*V*dE*EA_Up[v*n_a+v3];
        }
    }
    //VH
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE= 1.0/(e_c[i]-e_EA[v]);
        for(int v2=0;v2<n_a;v2++)
        for(int v3=0;v3<n_a;v3++)
        for(int w=0; w<n_a; w++){
            H2a=H_CA[i*n_a+w];
            V=(CAAA[((i*n_a+v3)*n_a+u)*n_a+t]*
               H2a);
            RF_P3_VH[((t*n_a+u)*n_a+v2)*n_a+w]+=EA_Um[v*n_a+v2]*V*dE*EA_Up[v*n_a+v3];
        }
    }

    //HV
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE= 1.0/(e_c[i]-e_EA[t]);
        
        
        for(int t2=0;t2<n_a;t2++)
        for(int t3=0;t3<n_a;t3++)
        for(int y=0; y<n_a; y++)
        for(int x=0; x<n_a; x++)
        for(int w=0; w<n_a; w++){
            H1a=H_CA[i*n_a+t3];
            V=(H1a*
               CAAA[((i*n_a+w)*n_a+x)*n_a+y]);
            RF_P3_HV[((t2*n_a+y)*n_a+x)*n_a+w]+=EA_Um[t*n_a+t2]*V*dE*EA_Up[t*n_a+t3];
        }
    }
    
    //HH
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE= 1.0/(e_c[i]-e_EA[t]);
        
        for(int t2=0;t2<n_a;t2++)
        for(int t3=0;t3<n_a;t3++)
        for(int u=0; u<n_a; u++){
            H1a=H_CA[i*n_a+t3];
            H2a=H_CA[i*n_a+u];
            V=H1a*H2a;
            RF_P3_HH[t2*n_a+u]+=EA_Um[t*n_a+t2]*V*dE*EA_Up[t*n_a+t3];
        }
    }
    
    
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int y=0; y<n_a; y++)
        //   a(w)a+(x)a+(y)a+(v)a(u)a(t)
        //a+(v)a+(x)a(t) RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+...]
        RF_P3_JK[((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w]+= RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+y)*n_a+w)*n_a+u]
                                                           -RF_P3_VV[((((u*n_a+v)*n_a+x)*n_a+y)*n_a+w)*n_a+t]
                                                           -RF_P3_VV[((((t*n_a+v)*n_a+y)*n_a+x)*n_a+w)*n_a+u]
                                                           +RF_P3_VV[((((u*n_a+v)*n_a+y)*n_a+x)*n_a+w)*n_a+t];
    
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
        RF_PV_JK[((u*n_a+t)*n_a+v)*n_a+w]+=-RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+u]
                                           +RF_P3_VV[((((u*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+t]
                                           +RF_P3_VV[((((t*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+u]
                                           -RF_P3_VV[((((u*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+t]
                                           -RF_P3_VV[((((t*n_a+x)*n_a+v)*n_a+w)*n_a+u)*n_a+x]
                                           +RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x]
                                           +RF_P3_VV[((((u*n_a+x)*n_a+v)*n_a+w)*n_a+t)*n_a+x]
                                           -RF_P3_VV[((((u*n_a+v)*n_a+x)*n_a+w)*n_a+t)*n_a+x]
                                           +RF_P3_VV[((((t*n_a+x)*n_a+w)*n_a+v)*n_a+u)*n_a+x]
                                           -RF_P3_VV[((((t*n_a+w)*n_a+x)*n_a+v)*n_a+u)*n_a+x]
                                           -RF_P3_VV[((((u*n_a+x)*n_a+w)*n_a+v)*n_a+t)*n_a+x]
                                           +RF_P3_VV[((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x]
                                           +RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x]
                                           -RF_P3_VV[((((u*n_a+v)*n_a+x)*n_a+w)*n_a+t)*n_a+x]
                                           -RF_P3_VV[((((t*n_a+w)*n_a+x)*n_a+v)*n_a+u)*n_a+x]
                                           +RF_P3_VV[((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x];
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
        RF_PV_JK[((u*n_a+t)*n_a+v)*n_a+w]+=-RF_P3_VH[((t*n_a+v)*n_a+w)*n_a+u]
                                           -RF_P3_HV[((v*n_a+w)*n_a+u)*n_a+t]
                                           +RF_P3_VH[((u*n_a+v)*n_a+w)*n_a+t]
                                           +RF_P3_HV[((v*n_a+w)*n_a+t)*n_a+u]
                                           +RF_P3_VH[((t*n_a+w)*n_a+v)*n_a+u]
                                           +RF_P3_HV[((w*n_a+v)*n_a+u)*n_a+t]
                                           -RF_P3_VH[((u*n_a+w)*n_a+v)*n_a+t]
                                           -RF_P3_HV[((w*n_a+v)*n_a+t)*n_a+u];

    
//  RF_P3_AB                 - order differs with the original XMCQDPT code (res_fit.cpp)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int x=0; x<n_a; x++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int y=0; y<n_a; y++)
        RF_P3_AB[((((t*n_a+u)*n_a+x)*n_a+v)*n_a+w)*n_a+y]+=-RF_P3_VV[((((t*n_a+v)*n_a+y)*n_a+w)*n_a+u)*n_a+x]
                                                           +RF_P3_VV[((((u*n_a+v)*n_a+y)*n_a+w)*n_a+t)*n_a+x]
                                                           +RF_P3_VV[((((t*n_a+w)*n_a+y)*n_a+v)*n_a+u)*n_a+x]
                                                           -RF_P3_VV[((((u*n_a+w)*n_a+y)*n_a+v)*n_a+t)*n_a+x]
                                                           -RF_P3_VV[((((t*n_a+v)*n_a+w)*n_a+y)*n_a+x)*n_a+u]
                                                           +RF_P3_VV[((((u*n_a+v)*n_a+w)*n_a+y)*n_a+x)*n_a+t]
                                                           +RF_P3_VV[((((t*n_a+w)*n_a+v)*n_a+y)*n_a+x)*n_a+u]
                                                           -RF_P3_VV[((((u*n_a+w)*n_a+v)*n_a+y)*n_a+x)*n_a+t]
                                                           -RF_P3_VV[((((x*n_a+y)*n_a+v)*n_a+w)*n_a+u)*n_a+t]
                                                           +RF_P3_VV[((((x*n_a+y)*n_a+v)*n_a+w)*n_a+t)*n_a+u]
                                                           +RF_P3_VV[((((x*n_a+y)*n_a+w)*n_a+v)*n_a+u)*n_a+t]
                                                           -RF_P3_VV[((((x*n_a+y)*n_a+w)*n_a+v)*n_a+t)*n_a+u];

//     set_zero_matr(RF_PV_AB, n_a*n_a*n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
        RF_PV_AB[((t*n_a+u)*n_a+v)*n_a+w]+=-RF_P3_VV[((((t*n_a+x)*n_a+w)*n_a+v)*n_a+x)*n_a+u]
                                           -RF_P3_VV[((((t*n_a+x)*n_a+v)*n_a+w)*n_a+u)*n_a+x]
                                           +RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x]
                                           -RF_P3_VV[((((u*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+t]
                                           +RF_P3_VV[((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x]
                                           -RF_P3_VV[((((u*n_a+x)*n_a+v)*n_a+w)*n_a+x)*n_a+t]
                                           -RF_P3_VV[((((u*n_a+x)*n_a+w)*n_a+v)*n_a+t)*n_a+x]
                                           +RF_P3_VV[((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x]
                                           -RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+u]
                                           +RF_P3_VV[((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x];
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
        RF_PV_AB[((t*n_a+u)*n_a+v)*n_a+w]+=-RF_P3_VH[((u*n_a+w)*n_a+v)*n_a+t]
                                           -RF_P3_HV[((v*n_a+w)*n_a+u)*n_a+t]
                                           -RF_P3_VH[((t*n_a+v)*n_a+w)*n_a+u]
                                           -RF_P3_HV[((w*n_a+v)*n_a+t)*n_a+u];
    
//     set_zero_matr(RF_PH, n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int y=0; y<n_a; y++)
        RF_PH[t*n_a+u]+=-RF_P3_VV[((((t*n_a+y)*n_a+v)*n_a+u)*n_a+v)*n_a+y]
                        +RF_P3_VV[((((t*n_a+v)*n_a+y)*n_a+u)*n_a+v)*n_a+y]*2;
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int y=0; y<n_a; y++)
        RF_PH[t*n_a+u]+=-RF_P3_VH[((t*n_a+y)*n_a+u)*n_a+y]
                        +RF_P3_VH[((t*n_a+u)*n_a+y)*n_a+y]*2
                        -RF_P3_HV[((y*n_a+u)*n_a+y)*n_a+t]
                        +RF_P3_HV[((y*n_a+u)*n_a+t)*n_a+y]*2;
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
        RF_PH[t*n_a+u]+=-RF_P3_HH[u*n_a+t];
    
//     set_zero_matr(RF_PS, N_fit);
    for(int t=0; t<n_a; t++)
        RF_PS+=RF_P3_HH[t*n_a+t]*2;
    

    delete[] CAAA  ;


    delete[] RF_P3_VV;
    delete[] RF_P3_VH;
    delete[] RF_P3_HV;
    delete[] RF_P3_HH;

    
    return 0;
}

int PT_tensors::calc_IPEA_2_CCVV(){
    
    double * R;
    R = new double[num_threads];
    set_zero_matr(R,num_threads);
    double *V[num_threads];
    double *JK[num_threads];
    for(int i=0;i<num_threads;i++) V[i] = new double[n_c];
    for(int i=0;i<num_threads;i++)JK[i] = new double[n_c*n_c];
    
    set_blas_non_par();
#pragma omp parallel for
    for(int a=0; a<n_v; a++){//fprintf(stderr,"CCVV a=%d\r",a);
    for(int b=a; b<n_v; b++){/*fprintf(stderr,"CCVV a,b=%d,%d\r",a,b);*/
        
        int th_id = omp_get_thread_num();
        double C;
        int n_i = n_c+n_a;
        int iajb=0;
        double dE;
        double Ep;
        double J;
        double K;
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                    n_c,n_c,integrals->aux_n_ao,1.0,
                    integrals->VC_RI_M+a*n_c*integrals->aux_n_ao,integrals->aux_n_ao,
                    integrals->VC_RI_M+b*n_c*integrals->aux_n_ao,integrals->aux_n_ao,0.0,
                    JK[th_id],n_c);
        
        
        for(int i=0; i<n_c; i++){//fprintf(stderr,"i,a=%d,%d\r",i,a);
        for(int j=i; j<n_c; j++){
            J=JK[th_id][i*n_c+j];
            K=JK[th_id][j*n_c+i];
            V[th_id][j]=2*(J*J+K*K);
            
            if(a==b)V[th_id][j]=V[th_id][j]/2;
            if(i==j)V[th_id][j]=V[th_id][j]/2;
            C=J-K;
            V[th_id][j]+=2*C*C;
        }
        
        Ep=e_c[i]-e_v[a]-e_v[b];
        for(int j=i; j<n_c; j++){
            dE=e_c[j]+Ep;
            R[th_id]+=V[th_id][j]/dE;
            
        }
       
        }
    }
    }
    set_blas_par();
    
    for(int i=0;i<num_threads;i++)
        RF_PS+=R[i];
    
    delete[] R;
    for(int i=0;i<num_threads;i++) delete[] V [i];
    for(int i=0;i<num_threads;i++) delete[] JK[i];
    
    
    return 0;
}

int PT_tensors::calc_IPEA_2_CAVV(){
    double **PH_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)PH_th[i]=new double[n_a*n_a];
    for(int i=0;i<num_threads;i++)set_zero_matr(PH_th[i],n_a*n_a);
    
    set_blas_non_par();
    #pragma omp parallel
    {
        int nt = omp_get_thread_num();
        
        double *K;
        double *J;
        K= new double[n_a*n_c];
        J= new double[n_a*n_c];
        
        double *K1;
        double *J1;
        K1= new double[n_a*n_c];
        J1= new double[n_a*n_c];
        
        double *Ku;
        double *Ju;
        Ku= new double[n_a*n_c];
        Ju= new double[n_a*n_c];
        
        
        
        double dE;
        double V2;
        
//         fprintf(out_stream,"CAVV is not parallel\n");
        
        for(int a=nt  ; a<n_v; a+=num_threads)
        for(int b=a  ; b<n_v; b++){//fprintf(stderr,"CAVVa,b=%d,%d\r",a,b);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_c,n_a,integrals->aux_n_ao,1.0,
                            integrals->VC_RI_M+a*n_c*integrals->aux_n_ao,integrals->aux_n_ao,
                            integrals->VA_RI_M+b*n_a*integrals->aux_n_ao,integrals->aux_n_ao,0.0,
                            J,n_a);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_c,n_a,integrals->aux_n_ao,1.0,
                            integrals->VC_RI_M+b*n_c*integrals->aux_n_ao,integrals->aux_n_ao,
                            integrals->VA_RI_M+a*n_a*integrals->aux_n_ao,integrals->aux_n_ao,0.0,
                            K,n_a);
            
            set_zero_matr(K1,n_a*n_c);
            set_zero_matr(J1,n_a*n_c);
            
            for(int i=0;i<n_c;i++)
            for(int t3=0;t3<n_a;t3++){
            for(int t2=0;t2<n_a;t2++)
            {
                K1[i*n_a+t3]+=IP_Up[t3*n_a+t2]*K[i*n_a+t2];
                J1[i*n_a+t3]+=IP_Up[t3*n_a+t2]*J[i*n_a+t2];
            }
            }
            
            for(int i=0;i<n_c;i++)
            for(int t3=0;t3<n_a;t3++){
                dE=e_c[i]+e_IP[t3]-e_v[a]-e_v[b];
                dE=1.0/dE;
                K1[i*n_a+t3]=K1[i*n_a+t3]*dE;
                J1[i*n_a+t3]=J1[i*n_a+t3]*dE;
            }
            
            
            set_zero_matr(Ku,n_a*n_c);
            set_zero_matr(Ju,n_a*n_c);
            
            for(int i=0;i<n_c;i++)
            for(int t=0;t<n_a;t++)
            for(int t3=0;t3<n_a;t3++)
            {
                Ku[i*n_a+t]+=K1[i*n_a+t3]*IP_Um[t3*n_a+t];
                Ju[i*n_a+t]+=J1[i*n_a+t3]*IP_Um[t3*n_a+t];
            }
            
            
            
            
            
            
            for(int i=0;i<n_c;i++)
            for(int t=0;t<n_a;t++)
            for(int v=0;v<n_a;v++)
            {
                V2 =(Ju[i*n_a+t]-Ku[i*n_a+t])*(J[i*n_a+v]-K[i*n_a+v]);
                V2+=Ju[i*n_a+t]*J[i*n_a+v];
                if(a!=b)
                    V2+=Ku[i*n_a+t]*K[i*n_a+v];
                PH_th[nt][t*n_a+v]+=V2;
                
            }
        }
        
        delete[] K;
        delete[] J;
        delete[] Ku;
        delete[] Ju;
        delete[] K1;
        delete[] J1;
    }
    set_blas_par();
    
    for(long j=0; j<num_threads;j++)
    #pragma omp parallel for
        for(long i=0; i<n_a*n_a;i++)
            RF_PH[i]+=PH_th[j][i];
        
    for(int i=0;i<num_threads;i++)delete[] PH_th[i];
    delete[] PH_th;
    
    
    return 0;
    
}

int PT_tensors::calc_IPEA_2_AAVV(){  //// TODO add skipping for 1 active el
    

//// ADDED AND WORKS
    ////if (true){
    if (((n_active_el_alp==1)&&(n_active_el_bet==0))||((n_active_el_alp==0)&&(n_active_el_bet==1))){
        fprintf(out_stream,"only 1 alpha or 1 beta active el is present, skipping AAVV block\n");
        return 0;
    }
//     num_threads=1;
    double **JK_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)JK_th[i]=new double[n_a*n_a*n_a*n_a];
    for(int i=0;i<num_threads;i++)set_zero_matr(JK_th[i],n_a*n_a*n_a*n_a);

    double **AB_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)AB_th[i]=new double[n_a*n_a*n_a*n_a];
    for(int i=0;i<num_threads;i++)set_zero_matr(AB_th[i],n_a*n_a*n_a*n_a);

    
    set_blas_non_par();
    #pragma omp parallel
    {
        int nt =omp_get_thread_num();
        
        double AB,JK,dE;
        
        double * Ja;
        
        Ja = new double[n_a*n_a];
        
        double * Ja2;
        Ja2 = new double[n_a*n_a];
        double * Ja3;
        Ja3 = new double[n_a*n_a];
        double * Ja4;
        Ja4 = new double[n_a*n_a];
        
        
        for(int a=nt; a<n_v; a+=num_threads)
        for(int b=a; b<n_v; b++){//fprintf(stderr,"AAVV a,b=%3d,%3d  (thread %2d)    \r",a,b,nt);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_a,n_a,integrals->aux_n_ao,1.0,
                            integrals->VA_RI_M+a*n_a*integrals->aux_n_ao,integrals->aux_n_ao,
                            integrals->VA_RI_M+b*n_a*integrals->aux_n_ao,integrals->aux_n_ao,0.0,
                            Ja,n_a);
            set_zero_matr(Ja2, n_a*n_a);
            set_zero_matr(Ja3, n_a*n_a);
            set_zero_matr(Ja4, n_a*n_a);
            
            //// TODO 4 cycles -> 3 cycles, matrix mult
            for(int t=0  ;t<n_a;t++)
            for(int u=0  ;u<n_a;u++)
            for(int t3=0  ;t3<n_a;t3++)
            for(int u3=0  ;u3<n_a;u3++){
                Ja3[t*n_a+u]+=Ja[t3*n_a+u3]*IP_Up[t*n_a+t3]*IP_Up[u*n_a+u3];
                Ja4[t*n_a+u]+=Ja[u3*n_a+t3]*IP_Up[t*n_a+t3]*IP_Up[u*n_a+u3];
            }
            //// TODO optimize -- temp objects, 4 cycles with divisions, 5 cycles for rotations
            for(int t=0  ;t<n_a;t++)
            for(int u=0  ;u<n_a;u++)
            for(int t2=0  ;t2<n_a;t2++)
            for(int u2=0  ;u2<n_a;u2++)
            for(int v=0  ;v<n_a;v++)
            for(int w=0  ;w<n_a;w++){
                AB         =Ja3[t*n_a+u]*Ja[v*n_a+w];
                if(a!=b)AB+=Ja4[t*n_a+u]*Ja[w*n_a+v];
                   
                dE=1.0/(e_IP[t]+e_IP[u]-e_v[a]-e_v[b]);
                AB_th[nt][((t2*n_a+u2)*n_a+v)*n_a+w]+=IP_Um[t*n_a+t2]*IP_Um[u*n_a+u2]*AB*dE;
                
                
            }
            
            for(int t=0;t<n_a;t++)
            for(int u=0;u<n_a;u++)
            for(int t3=0;t3<n_a;t3++)
            for(int u3=0;u3<n_a;u3++){
                Ja2[t*n_a+u]+=(Ja  [t3*n_a+u3]-Ja  [u3*n_a+t3])*
                               IP_Up[t*n_a+t3]*IP_Up[u*n_a+u3];
//                               (IP_Up[u*n_a+u3]*IP_Up[t*n_a+t3]
// //                               -IP_Up[t*n_a+u3]*IP_Up[u*n_a+t3]);
                
            }
            
            for(int t=0;t<n_a;t++)
            for(int u=0;u<n_a;u++)
            for(int t2=0;t2<n_a;t2++)
            for(int u2=0;u2<n_a;u2++)
            for(int v=0  ;v<n_a;v++)
            for(int w=0  ;w<n_a;w++){
                JK=IP_Um[t*n_a+t2]*IP_Um[u*n_a+u2]*
                    Ja2[t*n_a+u]*
                   (Ja [v*n_a+w]-Ja [w*n_a+v]);
                   
                dE=1.0/(e_IP[t]+e_IP[u]-e_v[a]-e_v[b]);
                JK_th[nt][((u2*n_a+t2)*n_a+v)*n_a+w]+=JK*dE;///inverse t-u in res_fit_calc_2body_AA (?)
//                 JK_th[nt][((t2*n_a+u2)*n_a+v)*n_a+w]-=JK*dE;///inverse t-u in res_fit_calc_2body_AA (?)
                
            }
                
            
        }
    
       delete[] Ja ;
       delete[] Ja2;
       delete[] Ja3;
       delete[] Ja4;
    
    
    }
    set_blas_par();
    
//     #pragma omp parallel for
    for(long j=0; j<num_threads;j++)
        for(long i=0; i<n_a*n_a*n_a*n_a;i++)
            RF_PV_JK[i]+=JK_th[j][i];
//     #pragma omp parallel for
    for(long j=0; j<num_threads;j++)
        for(long i=0; i<n_a*n_a*n_a*n_a;i++)
            RF_PV_AB[i]+=AB_th[j][i];
        
    for(int i=0;i<num_threads;i++)delete[] JK_th[i];
    delete[] JK_th;
    
    for(int i=0;i<num_threads;i++)delete[] AB_th[i];
    delete[] AB_th;
    return 0;
    
}

int PT_tensors::calc_IPEA_2_CCAV(){
    
    double **RF_PH_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)RF_PH_th[i]   = new double[n_a*n_a];
    for(int i=0;i<num_threads;i++)set_zero_matr(RF_PH_th[i], n_a*n_a);
    
    set_blas_non_par();
    #pragma omp parallel
    {
        int nt =omp_get_thread_num();
        
        double *K;
        K= new double[n_a*n_v];
        double *J;
        J= new double[n_a*n_v];
        
        double dE;
        double V2;
        
        for(int i=nt; i<n_c; i+=num_threads)
        for(int j=i; j<n_c; j++){//fprintf(stderr," CCAV i,j=%d,%d\r",i,j);
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){ //// ??? ddot -> dgemm ???
                J[a*n_a+t]=cblas_ddot(integrals->aux_n_ao, integrals->VC_RI_M+(a*n_c+i)*integrals->aux_n_ao, 1, integrals->CA_RI_M+(j*n_a+t)*integrals->aux_n_ao, 1);
                K[a*n_a+t]=cblas_ddot(integrals->aux_n_ao, integrals->VC_RI_M+(a*n_c+j)*integrals->aux_n_ao, 1, integrals->CA_RI_M+(i*n_a+t)*integrals->aux_n_ao, 1);
            }
            
            for(int a=0;a<n_v;a++)
            for(int t=0;t<n_a;t++)
            for(int t2=0;t2<n_a;t2++)
            for(int t3=0;t3<n_a;t3++)
            for(int v=0;v<n_a;v++){
                V2 =(J[a*n_a+t2]-K[a*n_a+t2])*(J[a*n_a+v]-K[a*n_a+v]);
                V2+=J[a*n_a+t2]*J[a*n_a+v];    //i<j is filled by J
                if(i!=j)                      //i=j is filled only once J[i,i]=K[i,i]
                    V2+=K[a*n_a+t2]*K[a*n_a+v];//i>j is filled by K
                dE=e_c[i]+e_c[j]-e_EA[t3]-e_v[a];
                RF_PH_th[nt][v*n_a+t]-=EA_Up[t3*n_a+t2]*V2/dE*EA_Um[t3*n_a+t];//sign changed!!!!
            }
        }
        
        delete[] K;
        delete[] J;
    }
    set_blas_par();
    
    for(long j=1; j<num_threads;j++)
//     #pragma omp parallel for
        for(long i=0; i<n_a*n_a;i++)
            RF_PH_th[0][i]+=RF_PH_th[j][i];
    
//     #pragma omp parallel for
    for(int t=0; t<n_a; t++)
        RF_PS-=RF_PH_th[0][t*n_a+t]*2;
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
        RF_PH[w*n_a+t]+=RF_PH_th[0][w*n_a+t];
    
    for(int i=0;i<num_threads;i++)delete[] RF_PH_th[i];
    delete[] RF_PH_th;
    
    
    return 0;
    
}

int PT_tensors::calc_IPEA_2_CCAA(){
    
//     num_threads=1;
    
    double ** RF_PV_AB_th = new double * [num_threads];
    double ** RF_PV_JK_th = new double * [num_threads];
    
    for(int i=0;i<num_threads;i++)RF_PV_AB_th[i] = new double[n_a*n_a*n_a*n_a];
    for(int i=0;i<num_threads;i++)RF_PV_JK_th[i] = new double[n_a*n_a*n_a*n_a];
    
    for(int i=0;i<num_threads;i++)set_zero_matr(RF_PV_AB_th[i], n_a*n_a*n_a*n_a);
    for(int i=0;i<num_threads;i++)set_zero_matr(RF_PV_JK_th[i], n_a*n_a*n_a*n_a);
    
    set_blas_non_par();
    #pragma omp parallel
    {
        int nt =omp_get_thread_num();
    
        double dE;
        double Ed;
        
        double V2;
        double AB;
        
        double * Ja;
        double * Ja2;
        double * Ja3;
        double * Ja4;
        Ja = new double[n_a*n_a];
        Ja2 = new double[n_a*n_a];
        Ja3 = new double[n_a*n_a];
        Ja4 = new double[n_a*n_a];
        
        
        for(int i=nt; i<n_c; i+=num_threads)
        for(int j=i; j<n_c; j++){//fprintf(stderr,"CCAA i,j=%3d,%3d  (thread %2d)    \r",i,j,nt);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_a,n_a,integrals->aux_n_ao,1.0,
                            integrals->CA_RI_M+i*n_a*integrals->aux_n_ao,integrals->aux_n_ao,
                            integrals->CA_RI_M+j*n_a*integrals->aux_n_ao,integrals->aux_n_ao,0.0,
                            Ja,n_a);
            
            set_zero_matr(Ja2, n_a*n_a);
            set_zero_matr(Ja3, n_a*n_a);
            set_zero_matr(Ja4, n_a*n_a);
                       
            for(int v=0;v<n_a;v++)
            for(int w=0;w<n_a;w++)
            for(int v3=0;v3<n_a;v3++)
            for(int w3=0;w3<n_a;w3++){
                Ja2[v*n_a+w]+=(Ja  [v3*n_a+w3]-Ja  [w3*n_a+v3])*
                               EA_Up[w*n_a+w3]*EA_Up[v*n_a+v3];
//                               (EA_Up[w*n_a+w3]*EA_Up[v*n_a+v3]
//                               -EA_Up[v*n_a+w3]*EA_Up[w*n_a+v3]);
                
            }

            
            //// TODO move rotation and mult by dE
            for(int t=0  ;t<n_a;t++)
            for(int u=0  ;u<n_a;u++)
            for(int v=0;v<n_a;v++)
            for(int w=0;w<n_a;w++)
            for(int v2=0;v2<n_a;v2++)
            for(int w2=0;w2<n_a;w2++){
                
                V2=(Ja[t*n_a+u]-Ja[u*n_a+t])*Ja2[v*n_a+w]*
                    EA_Um[w*n_a+w2]*EA_Um[v*n_a+v2];
//                    (EA_Um[w*n_a+w2]*EA_Um[v*n_a+v2]
//                    -EA_Um[v*n_a+w2]*EA_Um[w*n_a+v2]);
                
                dE=1.0/(-e_EA[v]-e_EA[w]+e_c[i]+e_c[j]);
                
                RF_PV_JK_th[nt][((u*n_a+t)*n_a+v2)*n_a+w2]+=V2*dE;///inverse t-u in res_fit_calc_2body_AA (?)
//                 RF_PV_JK_th[nt][((u*n_a+t)*n_a+w2)*n_a+v2]-=V2*dE;///inverse t-u in res_fit_calc_2body_AA (?)
            }
            
            
            for(int v=0  ;v<n_a;v++)
            for(int w=0  ;w<n_a;w++)
            for(int v3=0  ;v3<n_a;v3++)
            for(int w3=0  ;w3<n_a;w3++){
                Ja3[v*n_a+w]+=Ja[v3*n_a+w3]*EA_Up[v*n_a+v3]*EA_Up[w*n_a+w3];
                Ja4[v*n_a+w]+=Ja[w3*n_a+v3]*EA_Up[v*n_a+v3]*EA_Up[w*n_a+w3];
            }
            
            
            for(int t=0  ;t<n_a;t++)
            for(int u=0  ;u<n_a;u++)
            for(int v=0  ;v<n_a;v++)
            for(int v2=0  ;v2<n_a;v2++)
            for(int w2=0  ;w2<n_a;w2++)
            for(int w=0  ;w<n_a;w++){
                
                AB         =Ja[t*n_a+u]*Ja3[v*n_a+w];
                if(i!=j)AB+=Ja[u*n_a+t]*Ja4[v*n_a+w];
                
                dE=1.0/(-e_EA[v]-e_EA[w]+e_c[i]+e_c[j]);
                
                RF_PV_AB_th[nt][((t*n_a+u)*n_a+v2)*n_a+w2]+=EA_Um[w*n_a+w2]*EA_Um[v*n_a+v2]*AB*dE;
                
            }
            
        }
        
        delete[] Ja ;
        delete[] Ja2;
        delete[] Ja3;
        delete[] Ja4;
    }
    set_blas_par();
    
    for(long j=1; j<num_threads;j++)
//     #pragma omp parallel for
        for(long i=0; i<n_a*n_a*n_a*n_a;i++)
            RF_PV_JK_th[0][i]+=RF_PV_JK_th[j][i];

    for(long j=1; j<num_threads;j++)
//     #pragma omp parallel for
        for(long i=0; i<n_a*n_a*n_a*n_a;i++)
            RF_PV_AB_th[0][i]+=RF_PV_AB_th[j][i];

    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(long i=0;i<n_a*n_a*n_a*n_a;i++)RF_PV_JK[i]+=RF_PV_JK_th[0][i];
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(long i=0;i<n_a*n_a*n_a*n_a;i++)RF_PV_AB[i]+=RF_PV_AB_th[0][i];
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
        RF_PH[v*n_a+u]+=RF_PV_JK_th[0][((t*n_a+v)*n_a+t)*n_a+u]*0.25;
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
        RF_PH[v*n_a+t]-=RF_PV_JK_th[0][((u*n_a+v)*n_a+t)*n_a+u]*0.25;

    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
        RF_PH[w*n_a+u]-=RF_PV_JK_th[0][((w*n_a+v)*n_a+v)*n_a+u]*0.25;

    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
        RF_PH[w*n_a+t]+=RF_PV_JK_th[0][((w*n_a+u)*n_a+t)*n_a+u]*0.25;

    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
        RF_PH[w*n_a+u]-=RF_PV_AB_th[0][((t*n_a+w)*n_a+t)*n_a+u];



    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
        RF_PS+=RF_PV_JK_th[0][((u*n_a+v)*n_a+v)*n_a+u]*0.5;//A+B

    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
        RF_PS-=RF_PV_JK_th[0][((w*n_a+v)*n_a+w)*n_a+v]*0.5;//A+B
    
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
        RF_PS+=RF_PV_AB_th[0][((t*n_a+u)*n_a+t)*n_a+u];

    
    
    
    for(int i=0;i<num_threads;i++)delete[] RF_PV_JK_th[i];
    for(int i=0;i<num_threads;i++)delete[] RF_PV_AB_th[i];
    
    
    delete[] RF_PV_JK_th;
    delete[] RF_PV_AB_th;
    
    return 0;
}


int sym_JK_T(double *RF_PV_JK, double * RF_PV,int n_a){
    
    #pragma omp parallel for
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
        RF_PV_JK[((t*n_a+u)*n_a+v)*n_a+w]+= RF_PV[((t*n_a+v)*n_a+u)*n_a+w]
                                           +RF_PV[((v*n_a+t)*n_a+w)*n_a+u];
    }
    }
    
    return 0;
}

int sym_JK_CA_T(double *RF_PV_JK, double * RF_PV,int n_a){
    
    #pragma omp parallel for
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
        RF_PV_JK[((t*n_a+u)*n_a+v)*n_a+w]+= RF_PV[((t*n_a+u)*n_a+w)*n_a+v]
                                           +RF_PV[((v*n_a+w)*n_a+u)*n_a+t];
    }
    }
    
    return 0;
}

int sym_AB_T(double * RF_PV_AB, double * RF_PV, int n_a){
    
    #pragma omp parallel for
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
        RF_PV_AB[((t*n_a+u)*n_a+v)*n_a+w]+= RF_PV[((t*n_a+v)*n_a+u)*n_a+w]
                                           +RF_PV[((v*n_a+t)*n_a+w)*n_a+u];
    
    }
    }
    
    return 0;
}

int sym_AB_CA_T(double * RF_PV_AB, double * RF_PV, int n_a){
    
    #pragma omp parallel for
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
        RF_PV_AB[((t*n_a+u)*n_a+v)*n_a+w]+=-RF_PV[((v*n_a+u)*n_a+w)*n_a+t]
                                           -RF_PV[((t*n_a+w)*n_a+u)*n_a+v];
    
    }
    }
    
    return 0;
}

int PT_tensors::calc_IPEA_1_AV(double* P_AV) {    //// AV block
//// TODO parallelism
    double dE;
    double V;

    double* VAAA;
    VAAA = new double[n_v * n_a * n_a * n_a];

    integrals->VAAA_calc(VAAA);

    double* H_muV;
    double* VmuAA;
    H_muV = new double[n_v * n_a];
    VmuAA = new double[n_v * n_a * n_a * n_a];


    set_zero_matr(RF_PV, n_a * n_a * n_a * n_a);
    set_zero_matr(VmuAA,n_v*n_a*n_a*n_a);
    set_zero_matr(H_muV,n_v*n_a);

    double* Uvirt;
    Uvirt = new double[n_a * n_a];

    
    //// ADDED WRONG
    //// #pragma omp parallel for
    for(int a=0; a<n_v; a++){
        set_zero_matr(Uvirt,n_a*n_a);
        for(int mu=0; mu<n_a; mu++){
            dE=1.0/(e_IP[mu] - e_v[a]);
            for(int t=0; t<n_a; t++)
            for(int w=0; w<n_a; w++)
                Uvirt[t*n_a+w]+=IP_Um[mu*n_a+t]*IP_Up[mu*n_a+w]*dE;
        }
    
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    n_a,n_a*n_a,n_a,1.0,
                    Uvirt,n_a,
                    VAAA+a*n_a*n_a*n_a,n_a*n_a,0.0,
                    VmuAA+a*n_a*n_a*n_a,n_a*n_a);

        //// #pragma omp parallel for
        for (int t = 0; t < n_a; t++)
        for (int u = 0; u < n_a; u++)
        for (int v = 0; v < n_a; v++)
        for (int w = 0; w < n_a; w++) {
            //<tu|va>*P[w,a];<tu|va>=(au|tv)
            V = VmuAA[((a * n_a + u) * n_a + t) * n_a + v] * P_AV[w * n_v + a];
            RF_PV[((t * n_a + u) * n_a + v) * n_a + w] += V;
        }
    }

    //// ADDED WRONG - ??? after dE evaluation ???
    //// #pragma omp parallel for
    for (int a = 0; a < n_v; a++) {
        for (int mu = 0; mu < n_a; mu++) {
            dE = 1.0 / (e_IP[mu] - e_v[a]);
            for (int t2 = 0; t2 < n_a; t2++)
                for (int t3 = 0; t3 < n_a; t3++) {
                    H_muV[t2 * n_v + a] += IP_Um[mu * n_a + t2] * dE * IP_Up[mu * n_a + t3] * H_AV[t3 * n_v + a];
                }
        }
    }

    for (int t = 0; t < n_a; t++)
        for (int w = 0; w < n_a; w++)
            //// ADDED WRONG
            //// #pragma omp parallel for
            for (int a = 0; a < n_v; a++) {
                ////~H[t,a]*P[w,a]=H_muV[t,a]*P[w,a]
                V = H_muV[t * n_v + a] * P_AV[w * n_v + a];
                RF_PH[t * n_a + w] += V;
    }


    sym_JK_T(RF_PV_JK, RF_PV, n_a);     

    sym_AB_T(RF_PV_AB, RF_PV, n_a);     

    delete[] VAAA;
    delete[] H_muV;
    delete[] VmuAA;
    delete[] Uvirt;

    return 0;
}

int PT_tensors::calc_IPEA_1_CA(double* P_CA) { //// CA block
//// TODO parallelism
    double dE;
    double V;
    
    double* CAAA;
    
    CAAA = new double[n_c * n_a * n_a * n_a];
    integrals->CAAA_calc(CAAA);
    
    double* RF_PH_tmp = new double[n_a * n_a];
    
    set_zero_matr(RF_PV, n_a * n_a * n_a * n_a);              
    set_zero_matr(RF_PH_tmp, n_a * n_a);

    double* H_Cmu;
    double* CmuAA;
    H_Cmu = new double[n_c * n_a];
    CmuAA = new double[n_c * n_a * n_a * n_a];

    set_zero_matr(H_Cmu, n_c * n_a);
    set_zero_matr(CmuAA, n_c * n_a * n_a * n_a);

    double* Ucore;
    Ucore = new double[n_a * n_a];

    for (int i = 0; i < n_c; i++) {
        set_zero_matr(Ucore, n_a * n_a);
        for (int mu = 0; mu < n_a; mu++) {
            dE = 1.0 / (e_c[i] - e_EA[mu]);
            for (int t = 0; t < n_a; t++)
                for (int w = 0; w < n_a; w++)
                    Ucore[t * n_a + w] += EA_Um[mu * n_a + t] * EA_Up[mu * n_a + w] * dE;
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n_a, n_a * n_a, n_a, 1.0,
            Ucore, n_a,
            CAAA + i * n_a * n_a * n_a, n_a * n_a, 0.0,
            CmuAA + i * n_a * n_a * n_a, n_a * n_a);

        for (int v = 0; v < n_a; v++)
        for (int u = 0; u < n_a; u++)
        for (int t = 0; t < n_a; t++)
        for (int w = 0; w < n_a; w++) {
            //<it|uv>*P[w,a];<it|uv>=(iu|tv)
            V = CmuAA[((i * n_a + u) * n_a + t) * n_a + v] *P_CA[i * n_a + w];   
            RF_PV[((t * n_a + u) * n_a + v) * n_a + w] += V;
        }
    }

    //// ADDED WRONG
    //// #pragma omp parallel for
    for (int i = 0; i < n_c; i++) {
        for (int mu = 0; mu < n_a; mu++) {
            dE = 1.0 / (e_c[i] - e_EA[mu]);
            for (int t2 = 0; t2 < n_a; t2++)
                for (int t3 = 0; t3 < n_a; t3++)
                    H_Cmu[i * n_a + t2] += EA_Um[mu * n_a + t2] * H_CA[i * n_a + t3] * EA_Up[mu * n_a + t3] * dE;
               ////  H_uE[t2 * n_v + a] += IP_Um[mu * n_a + t2] * dE * IP_Up[mu * n_a + t3] * H_AV[t3 * n_v + a];
        }
    }

    //// ADDED - WRONG 
    //// #pragma omp parallel for
    for (int i = 0; i < n_c; i++) 
        for (int t = 0; t < n_a; t++) 
        for (int w = 0; w < n_a; w++) {
            //~H[t,a]*P[w,a]
            V = H_Cmu[i * n_a + t] * P_CA[i * n_a + w];
            RF_PH_tmp[w * n_a + t] -= V;
        }

    //// ADDED - WRONG
    //// #pragma omp parallel for
    for (int t = 0; t < n_a; t++)
        RF_PS -= RF_PH_tmp[t * n_a + t] * 2;

    for (int t = 0; t < n_a; t++)
        //// ADDED -??? 
        //// #pragma omp parallel for
        for (int w = 0; w < n_a; w++)
            RF_PH[w * n_a + t] += RF_PH_tmp[w * n_a + t];

    //// ADDED - WRONG, ? MOVE TO V
    //// #pragma omp parallel for
    for (int w = 0; w < n_a; w++)
        for (int t = 0; t < n_a; t++)
            for (int v = 0; v < n_a; v++)
                RF_PH[t * n_a + w] += 2 * RF_PV[((t * n_a + v) * n_a + w) * n_a + v]
                - RF_PV[((t * n_a + w) * n_a + v) * n_a + v];
    
    sym_JK_CA_T(RF_PV_JK, RF_PV, n_a);
    
    sym_AB_CA_T(RF_PV_AB, RF_PV, n_a);
    
    delete[] CAAA;
    delete[] RF_PH_tmp;
    delete[] H_Cmu;
    delete[] CmuAA; 
    delete[] Ucore;

    return 0;
}



int PT_tensors::calc_IPEA_1_CV(double* P_CV) { //// CV block
//// TODO parallelism
    double V;
    double dE;

    double* VCAA;
    double* CAVA;

    VCAA = new double[n_c * n_v * n_a * n_a]; 
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n_c * n_v, n_a * n_a, integrals->aux_n_ao, 1.0,
        integrals->VC_RI_M, integrals->aux_n_ao,
        integrals->AA_RI_M, integrals->aux_n_ao, 0.0,
        VCAA, n_a * n_a);
    CAVA = new double[n_a * n_c * n_a * n_v];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n_a * n_c, n_a * n_v, integrals->aux_n_ao, 1.0,
        integrals->CA_RI_M, integrals->aux_n_ao,
        integrals->VA_RI_M, integrals->aux_n_ao, 0.0,
        CAVA, n_a * n_v);
    
    for (int i = 0; i < n_c; i++)
        for (int a = 0; a < n_v; a++)
            for (int t = 0; t < n_a; t++)
                for (int u = 0; u < n_a; u++) {
                    //(<it|au>-<iu|at>)*P[i,a]
                    V = (2 * VCAA[((a * n_c + i) * n_a + u) * n_a + t] -
                        CAVA[((i * n_a + u) * n_v + a) * n_a + t]) *
                        P_CV[i * n_v + a];
                                        
                    dE = e_c[i] - e_v[a];
                    RF_PH[t * n_a + u] += V * dE / (dE * dE + 0.0);
                }

    for (int i = 0; i < n_c; i++)
        //// ADDED - WRONG
        //// #pragma omp parallel for
        for (int a = 0; a < n_v; a++) {
            V = H_CV[i * n_v + a] * P_CV[i * n_v + a] * 2;                
            dE = e_c[i] - e_v[a];                                         
            RF_PS += V * dE / (dE * dE + 0.0);
    
        }
    
    delete[] VCAA;
    delete[] CAVA;
    
    return 0;
}


int PT_tensors::P1_calc_IPEA(double* P1, aldet_data* CI, double* P_AV, double* P_CA, double* P_CV, int AV, int CA, int CV) {

         int n_s;
         
         n_s = CI->n_states[0];
         
         aldet_data first_order_props;
         first_order_props.get_dim(CI->n_act, CI->na, CI->nb, 1, CI->mult, CI->print_number);
         first_order_props.copy_coef(0,CI,CI->n_states[0],0,1);

         RF_PS=0;
     
         set_zero_matr(RF_PH      , n_a*n_a        );
         set_zero_matr(RF_PV_JK   , n_a*n_a*n_a*n_a);
         set_zero_matr(RF_PV_AB   , n_a*n_a*n_a*n_a);
         
         if(CA)calc_IPEA_1_CA(P_CA); // CA
         if(CV)calc_IPEA_1_CV(P_CV); // CV
         if(AV)calc_IPEA_1_AV(P_AV); // AV
         
         
         printf_timer("calculation of 1st order correction of one-electron property (dipole moment d(1))");fflush(out_stream);
         
         first_order_props.simple_import_data(RF_PV_JK, RF_PV_AB, RF_PH, RF_PS);
         first_order_props.H_calc(P1,CI->n_states[0]);

         char* timer_words = new char[256];
         if(CV==1)if(CA==1)if(AV==1)sprintf(timer_words, "calculation of first order properties\0");
         if(CV==0)if(CA==1)if(AV==1)sprintf(timer_words, "calculation of CA and AV parts of first order properties\0");
         if(CV==1)if(CA==0)if(AV==1)sprintf(timer_words, "calculation of CV and AV parts of first order properties\0");
         if(CV==1)if(CA==1)if(AV==0)sprintf(timer_words, "calculation of CV and CA parts of first order properties\0");
         if(CV==1)if(CA==0)if(AV==0)sprintf(timer_words, "calculation of CV part of first order properties\0");
         if(CV==0)if(CA==1)if(AV==0)sprintf(timer_words, "calculation of CA part of first order properties\0");
         if(CV==0)if(CA==0)if(AV==1)sprintf(timer_words, "calculation of AV part of first order properties\0");
         if(CV==0)if(CA==0)if(AV==0)sprintf(timer_words, "calculation of first order properties\0");
         
         printf_timer(timer_words);fflush(out_stream);
         delete[] timer_words;
         
    return 0;

}

PT_tensors::~PT_tensors(){
    
    if(RF_PH    != nullptr) delete[] RF_PH   ;
    if(RF_PV    != nullptr) delete[] RF_PV   ;
    if(RF_PV_JK != nullptr) delete[] RF_PV_JK;
    if(RF_PV_AB != nullptr) delete[] RF_PV_AB;
    if(RF_P3_JK != nullptr) delete[] RF_P3_JK;
    if(RF_P3_AB != nullptr) delete[] RF_P3_AB;
    if(e_IP     != nullptr) delete[] e_IP    ;
    if(e_EA     != nullptr) delete[] e_EA    ;
    if(e_IP_2   != nullptr) delete[] e_IP_2  ;
    if(e_EA_2   != nullptr) delete[] e_EA_2  ;
    if(e_IP_AB  != nullptr) delete[] e_IP_AB ;
    if(e_EA_AB  != nullptr) delete[] e_EA_AB ;
    if(IP_U     != nullptr) delete[] IP_U    ;
    if(EA_U     != nullptr) delete[] EA_U    ;
    if(IP_Um    != nullptr) delete[] IP_Um   ;
    if(EA_Um    != nullptr) delete[] EA_Um   ;
    if(IP_Up    != nullptr) delete[] IP_Up   ;
    if(EA_Up    != nullptr) delete[] EA_Up   ;
    if(IP_2_U   != nullptr) delete[] IP_2_U  ;
    if(EA_2_U   != nullptr) delete[] EA_2_U  ;
    if(IP_2_Um  != nullptr) delete[] IP_2_Um ;
    if(EA_2_Um  != nullptr) delete[] EA_2_Um ;
    if(IP_2_Up  != nullptr) delete[] IP_2_Up ;
    if(EA_2_Up  != nullptr) delete[] EA_2_Up ;
    if(IP_AB_U  != nullptr) delete[] IP_AB_U ;
    if(EA_AB_U  != nullptr) delete[] EA_AB_U ;
    if(IP_AB_Um != nullptr) delete[] IP_AB_Um;
    if(EA_AB_Um != nullptr) delete[] EA_AB_Um;
    if(IP_AB_Up != nullptr) delete[] IP_AB_Up;
    if(EA_AB_Up != nullptr) delete[] EA_AB_Up;
    
}


