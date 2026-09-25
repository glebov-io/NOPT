# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <string.h>
# include <ctype.h>

# include "blas_link.h"
# include "converger_2_1.h"
# include "l-bfgs_2_1.h"
# include "molecule.h"
# include "matr.h"

soscf_engine::soscf_engine(){
    
    orb_grad     = NULL;
    orb_shift    = NULL;
    app_orb_hess = NULL;
    rot_matr     = NULL;
    
}

int soscf_engine::init(int n, int n_el_inp,int n_mo_inp, int n_ao_inp){
    
    grad_max_soscf=1;
    grad_min_soscf=0.000005;
    x_max_soscf=0.1;
    n_ao=n_ao_inp;
    n_mo=n_mo_inp;
    n_el=n_el_inp;
//     N_LBFGS_VECTORS=20;

    BFGS.init(n_el*(n_mo-n_el),n);
    
    orb_grad = new double [n_el*(n_mo-n_el)];
    orb_shift = new double [n_el*(n_mo-n_el)];
    for(int i=0;i<n_el*(n_mo-n_el);i++)orb_shift[i]=0;
    app_orb_hess = new double [n_el*(n_mo-n_el)];
    rot_matr=new double[n_ao*n_ao];
//     MOL = A;
    return 0;
}

soscf_engine::~soscf_engine(){
    
    
    if(orb_grad    !=NULL)delete[] orb_grad    ;
    if(orb_shift   !=NULL)delete[] orb_shift   ;
    if(app_orb_hess!=NULL)delete[] app_orb_hess;
    if(rot_matr    !=NULL)delete[] rot_matr    ;
    
}


int soscf_engine::orb_grad_calc(double * F, int n_alp_el, int n_mo){
    
    //     g(ia)=4*F(i,a);
    for(int i=0;i<n_alp_el;i++)
    for(int j=0;j<n_mo-n_alp_el;j++)
        orb_grad[i*(n_mo-n_alp_el)+j]=4*F[(n_alp_el+j)*n_mo+i];
    
    return 0;
}

int soscf_engine::orb_hess_calc(int n_ao, int n_cor, double * F){
  //experimental option  
//   for(int i=0;i<n_ao;i++)cudaMemcpy(alp_orb_energy+i,gpu_F_alp+i*(n_ao+1),sizeof(double),cudaMemcpyDeviceToHost);
  //end of exp. opt.
  
  //H(ia,ia) = 4*(e(i)-e(a))
  for(int i=0;i<n_cor;i++)
      for(int j=0;j<n_mo-n_cor;j++){
          app_orb_hess[j+i*(n_mo-n_cor)]=4*(F[(n_cor+j)*(n_mo+1)]-F[i*(n_mo+1)]); /*fprintf(out_stream,"%d %d %e\n",i,j, app_orb_hess[j+i*(n_ao-n_cor)])*/;}
//           getchar();
  return 0;
      
}

double soscf_engine::calc(double * F){
    
    orb_grad_calc(F   ,n_el,n_mo);
    orb_hess_calc(n_mo,n_el,F   );
    //determination of maximum grad element
    double max=fabs(orb_grad[0]);
    for (int i=1;i<n_el*(n_mo-n_el);i++)
        if(fabs(orb_grad[i])>max)max=fabs(orb_grad[i]);

    return max;
}

int soscf_engine::step(double * VEC, double * BUF){
    
    double zoom = BFGS.step(orb_shift, orb_grad, app_orb_hess, do_nothing, do_nothing, n_el*(n_mo-n_el),x_max_soscf);
    
    if(zoom > -1.0){
        fprintf(out_stream,"SOSCF is scaling rotation angle matrix Xmax=%f\n",-x_max_soscf/zoom);
 
    }
    
    //generation of rot matr
    set_zero_matr(rot_matr, n_ao*n_ao);
    for(int i=0;i<n_ao;i++) rot_matr[i*(n_ao+1)]=1.0;// 1 at diagonal
    for(int i=0;i<n_el;i++)
    for(int j=0;j<(n_mo-n_el);j++){
        rot_matr[i*n_ao+ n_el+j ]= orb_shift[i*(n_mo-n_el)+j]; //upper corner
        rot_matr[i+n_ao*(n_el+j)]=-orb_shift[i*(n_mo-n_el)+j]; //lower corner
    }
    
//     PrintMatr(rot_matr,n_ao,n_ao,0);
//     exit(0);
          
    ortogonalization(rot_matr, n_ao);//ortogonalization of rot matrix
    //rotation of vectors
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0, rot_matr  ,n_ao,VEC,n_ao,0.0,BUF,n_ao);//TMP = C * U^T
    for(int i=0;i<n_ao*n_ao;i++)VEC[i]=BUF[i];
    
//     PrintMatr(VEC,n_ao,n_ao,0);
//     exit(0);
    
    
    return 0;
    
}

// int soscf_engine::OHF_step(double * VEC, double * O_B, double * M, double * BUF, int n_ort){
//     
//     double zoom = BFGS.step(orb_shift, orb_grad, app_orb_hess, do_nothing, do_nothing, n_el*(n_ao-n_el),x_max_soscf);
//     
//     if(zoom > -1.0){
//         fprintf(out_stream,"SOSCF is scaling rotation angle matrix Xmax=%f\n",-x_max_soscf/zoom);
//  
//     }
//     
//     //generation of rot matr
//     set_zero_matr(rot_matr, n_ao*n_ao);
//     for(int i=0;i<n_ao;i++) rot_matr[i*(n_ao+1)]=1.0;// 1 at diagonal
//     for(int i=0;i<n_el;i++)
//     for(int j=0;j<(n_ao-n_el);j++){
//         rot_matr[i*n_ao+ n_el+j ]= orb_shift[i*(n_ao-n_el)+j]; //upper corner
//         rot_matr[i+n_ao*(n_el+j)]=-orb_shift[i*(n_ao-n_el)+j]; //lower corner
//     }
//           
//     ortogonalization(rot_matr, n_ao);//ortogonalization of rot matrix
//     //rotation of vectors
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                         n_ao, n_ao+n_ort, n_ao,1.0,
//                         rot_matr,n_ao,
//                         O_B, n_ao+n_ort,0.0,
//                         BUF,n_ao+n_ort);
//     
// //     fprintf(out_stream,"OB1:\n");
// //     PrintMatr(O_B,n_ao+n_ort,n_ao+n_ort,1);
//     
// //     for(int i=0;i<n_ort*n_ao+n_ort;i++)
// //         B[(n_ao+n_ort-n_ort)*n_ao+n_ort+i]=O[i];
// //     
//     
//     cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
//                         n_ao+n_ort,n_ao+n_ort,n_ao+n_ort,1.0,
//                         BUF,n_ao+n_ort,
//                         M,n_ao+n_ort,0.0,
//                         VEC,n_ao+n_ort);
// //     fprintf(out_stream,"V:\n");
// //     PrintMatr(VEC,n_ao+n_ort,n_ao+n_ort,1);
//     return 0;
//     
// }

int soscf_engine_ROHF::init(int n, int n_d_inp, int n_a_inp, int n_b_inp, int n_ao_inp/*, molecule * A*/){
    
    grad_max_soscf=1;
    grad_min_soscf=0.000005;
    x_max_soscf=0.1;
    n_ao=n_ao_inp;
    n_d=n_d_inp;
    n_a=n_a_inp;
    n_b=n_b_inp;
//     N_LBFGS_VECTORS=20;

    BFGS.init(n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b),n);
    
    orb_grad    = new double[n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b)];
    orb_shift   = new double[n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b)];
    app_orb_hess= new double[n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b)];
    set_zero_matr(orb_shift, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b));
    rot_matr=new double[n_ao*n_ao];
    
    orb_grad_a     = orb_grad     + n_d*(n_ao-n_d);
    orb_grad_b     = orb_grad     + n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a);
    app_orb_hess_a = app_orb_hess + n_d*(n_ao-n_d);
    app_orb_hess_b = app_orb_hess + n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a);
    orb_shift_a    = orb_shift    + n_d*(n_ao-n_d);
    orb_shift_b    = orb_shift    + n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a);
    
//     MOL = A;
    return 0;
}

int soscf_engine_ROHF::finalize(){
    
    
    delete[] orb_grad;
    delete[] orb_shift;
    delete[] app_orb_hess;
    delete[] rot_matr;
    return 0;
}

int soscf_engine_ROHF::orb_grad_part_add(double * g, double *F, int n, int n_v, int n_g, double k){
    
    for(int i=0;i<n  ;i++)
    for(int j=0;j<n_g;j++){
        g[i*n_v+j]+=k*F[j*n_ao+i];
//         fprintf(out_stream,"%d %d, %d %d, %e\n",i,j,i*n_v+j,j*n_ao+i,%
    }
    return 0;
    
}

int soscf_engine_ROHF::orb_grad_calc(double * F, double *F_a, double * F_b){
    
    set_zero_matr(orb_grad, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b));
    
    orb_grad_part_add(orb_grad      , F  + n_d         *n_ao        , n_d, n_ao-n_d        , n_ao-n_d        , 4.0);
    orb_grad_part_add(orb_grad      , F_a+ n_d         *n_ao        , n_d, n_ao-n_d        , n_a             ,-2.0);
    orb_grad_part_add(orb_grad  +n_a, F_b+(n_d+n_a    )*n_ao        , n_d, n_ao-n_d        , n_b             ,-2.0);
    orb_grad_part_add(orb_grad_a    , F_a+(n_d+n_a    )*n_ao+n_d    , n_a, n_ao-n_d-n_a    , n_ao-n_d-n_a    , 2.0);
    orb_grad_part_add(orb_grad_a    , F_b+(n_d+n_a    )*n_ao+n_d    , n_a, n_ao-n_d-n_a    , n_b             ,-2.0);
    orb_grad_part_add(orb_grad_b    , F_b+(n_d+n_a+n_b)*n_ao+n_d+n_a, n_b, n_ao-n_d-n_a-n_b, n_ao-n_d-n_a-n_b, 2.0);
    
//     fprintf(out_stream,"OG:\n");
//     PrintMatr(orb_grad, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b),1,0);
    
    return 0;
}

int soscf_engine_ROHF::orb_hess_part_add(double * h, double *F, int n, int n_v, int n_h, int n_s, double k){
    
    //H(ia,ia) = 4*(e(i)-e(a))
    double * F_v=F+(n_ao+1)*(n+n_s);
    for(int i=0;i<n;i++)
    for(int j=0;j<n_h;j++){
          h[i*n_v+j+n_s]+=k*(F_v[j*(n_ao+1)]-F[i*(n_ao+1)]); /*fprintf(out_stream,"%d %d %e\n",i,j, app_orb_hess[j+i*(n_ao-n_cor)])*/;}
//           getchar();
  return 0;
      
}

int soscf_engine_ROHF::orb_hess_calc(double * F, double *F_a, double * F_b){
    
    set_zero_matr(app_orb_hess, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b));
    
    orb_hess_part_add(app_orb_hess  , F                     , n_d, n_ao-n_d        , n_ao-n_d        ,   0, 4.0);
    orb_hess_part_add(app_orb_hess  , F_a                   , n_d, n_ao-n_d        , n_a             ,   0,-2.0);
    orb_hess_part_add(app_orb_hess  , F_b                   , n_d, n_ao-n_d        , n_b             , n_a,-2.0);
    orb_hess_part_add(app_orb_hess_a, F_a+ n_d     *(n_ao+1), n_a, n_ao-n_d-n_a    , n_ao-n_d-n_a    ,   0, 2.0);
    orb_hess_part_add(app_orb_hess_a, F_b+ n_d     *(n_ao+1), n_a, n_ao-n_d-n_a    , n_b             ,   0,-2.0);
    orb_hess_part_add(app_orb_hess_b, F_b+(n_d+n_a)*(n_ao+1), n_b, n_ao-n_d-n_a-n_b, n_ao-n_d-n_a-n_b,   0, 2.0);
//     fprintf(out_stream,"OH:\n");
//     PrintMatr(app_orb_hess, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b),1,0);
//     exit(0);
    return 0;
}

double soscf_engine_ROHF::calc(double * F, double *F_a, double * F_b){
    
    orb_grad_calc(F, F_a, F_b);
    orb_hess_calc(F, F_a, F_b);
    //determination of maximum grad element
    double max=fabs(orb_grad[0]);
    int i_max=0;
    for (int i=1;i<n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b);i++)
        if(fabs(orb_grad[i])>max){max=fabs(orb_grad[i]);i_max=i;}
    fprintf(out_stream,"orb_grad = %e",max);
    return max;
}

int soscf_engine_ROHF::rot_matr_add(double * R, double *S, int n, int n_v, int n_g){
    
    for(int i=0;i<n  ;i++)
    for(int j=0;j<n_g;j++){
        R[j*n_ao+i]=-S[i*n_v+j];
//         R[j*n_ao+i]=-S[i*n_v+j];
//         fprintf(out_stream,"%d %d, %d %d, %e\n",i,j,i*n_v+j,j*n_ao+i,%
    }
    return 0;
    
}

int soscf_engine_ROHF::step(double * VEC, double * BUF){
    
    double zoom = BFGS.step(orb_shift, orb_grad, app_orb_hess, do_nothing, do_nothing, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b),x_max_soscf);
    
    if(zoom > -1.0){
        fprintf(out_stream,"\nSOSCF is scaling rotation angle matrix Xmax=%f\n",-x_max_soscf/zoom);
 
    }
    
    //generation of rot matr
    set_zero_matr(rot_matr, n_ao*n_ao);
    for(int i=0;i<n_ao;i++) rot_matr[i*(n_ao+1)]=1.0;// 1 at diagonal
    rot_matr_add(rot_matr+ n_d         *n_ao        , orb_shift      , n_d, n_ao-n_d        , n_ao-n_d        );
//     rot_matr_add(rot_matr+ n_d         *n_ao        , orb_shift      , n_d, n_ao-n_d        , n_a             );
//     rot_matr_add(rot_matr+(n_d+n_a    )*n_ao        , orb_shift  +n_a, n_d, n_ao-n_d        , n_b             );
    rot_matr_add(rot_matr+(n_d+n_a    )*n_ao+n_d    , orb_shift_a    , n_a, n_ao-n_d-n_a    , n_ao-n_d-n_a    );
//     rot_matr_add(rot_matr+(n_d+n_a    )*n_ao+n_d    , orb_shift_a    , n_a, n_ao-n_d-n_a    , n_b             );
    rot_matr_add(rot_matr+(n_d+n_a+n_b)*n_ao+n_d+n_a, orb_shift_b    , n_b, n_ao-n_d-n_a-n_b, n_ao-n_d-n_a-n_b);
    
    for(int i=0  ;i<n_ao;i++)
    for(int j=i+1;j<n_ao;j++)
        rot_matr[i*n_ao+j]=-rot_matr[j*n_ao+i];

//     fprintf(out_stream,"RM:\n");
//     PrintMatr(rot_matr,n_ao,n_ao,1);
//     fprintf(out_stream,"OS:\n");
//     PrintMatr(orb_shift, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b),1,1);    
          
    
    ortogonalization(rot_matr, n_ao);//ortogonalization of rot matrix
    //rotation of vectors
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0, rot_matr  ,n_ao,VEC,n_ao,0.0,BUF,n_ao);//TMP = C * U^T
    for(int i=0;i<n_ao*n_ao;i++)VEC[i]=BUF[i];
    
    return 0;
    
}

int soscf_engine_ROHF::OHF_step(double * VEC, double * O_V, double * M, double * BUF, int n_ort){
    
    double zoom = BFGS.step(orb_shift, orb_grad, app_orb_hess, do_nothing, do_nothing, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b),x_max_soscf);
    
    if(zoom > -1.0){
        fprintf(out_stream,"\nSOSCF is scaling rotation angle matrix Xmax=%f\n",-x_max_soscf/zoom);
 
    }
    
    //generation of rot matr
    set_zero_matr(rot_matr, n_ao*n_ao);
    for(int i=0;i<n_ao;i++) rot_matr[i*(n_ao+1)]=1.0;// 1 at diagonal
    rot_matr_add(rot_matr+ n_d         *n_ao        , orb_shift      , n_d, n_ao-n_d        , n_ao-n_d        );
//     rot_matr_add(rot_matr+ n_d         *n_ao        , orb_shift      , n_d, n_ao-n_d        , n_a             );
//     rot_matr_add(rot_matr+(n_d+n_a    )*n_ao        , orb_shift  +n_a, n_d, n_ao-n_d        , n_b             );
    rot_matr_add(rot_matr+(n_d+n_a    )*n_ao+n_d    , orb_shift_a    , n_a, n_ao-n_d-n_a    , n_ao-n_d-n_a    );
//     rot_matr_add(rot_matr+(n_d+n_a    )*n_ao+n_d    , orb_shift_a    , n_a, n_ao-n_d-n_a    , n_b             );
    rot_matr_add(rot_matr+(n_d+n_a+n_b)*n_ao+n_d+n_a, orb_shift_b    , n_b, n_ao-n_d-n_a-n_b, n_ao-n_d-n_a-n_b);
    
    for(int i=0  ;i<n_ao;i++)
    for(int j=i+1;j<n_ao;j++)
        rot_matr[i*n_ao+j]=-rot_matr[j*n_ao+i];

//     fprintf(out_stream,"RM:\n");
//     PrintMatr(rot_matr,n_ao,n_ao,0);
//     fprintf(out_stream,"OV:\n");
//     PrintMatr(O_V,1,n_ao,0);
//     exit(0);
//     fprintf(out_stream,"OS:\n");
//     PrintMatr(orb_shift, n_d*(n_ao-n_d)+n_a*(n_ao-n_d-n_a)+n_b*(n_ao-n_d-n_a-n_b),1,1);    
          
    
    ortogonalization(rot_matr, n_ao);//ortogonalization of rot matrix
    //rotation of vectors
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0, rot_matr  ,n_ao,VEC,n_ao,0.0,BUF,n_ao);//TMP = C * U^T
    for(int i=0;i<n_ao*n_ao;i++)VEC[i]=BUF[i];
    
    return 0;
    
}


int soscf_engine_UHF::init(int n, int n_el_a_inp, int n_el_b_inp,int n_ao_a_inp, int n_ao_b_inp/*, molecule * A*/){
    
    grad_max_soscf=1;
    grad_min_soscf=0.000005;
    x_max_soscf=0.1;
    n_ao_a=n_ao_a_inp;
    n_ao_b=n_ao_b_inp;
    n_el_a=n_el_a_inp;
    n_el_b=n_el_b_inp;
//     N_LBFGS_VECTORS=20;

    BFGS.init(n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b),n);
    
    orb_grad_a     = new double [n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b)];
    orb_shift_a    = new double [n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b)];
    app_orb_hess_a = new double [n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b)];
    orb_grad_b     = orb_grad_a     + n_el_a*(n_ao_a-n_el_a);
    orb_shift_b    = orb_shift_a    + n_el_a*(n_ao_a-n_el_a);
    app_orb_hess_b = app_orb_hess_a + n_el_a*(n_ao_a-n_el_a);
    
    for(int i=0;i<n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b);i++)orb_shift_a[i]=0;
    rot_matr_a=new double[n_ao_a*n_ao_a];
    rot_matr_b=new double[n_ao_b*n_ao_b];
//     MOL = A;
    return 0;
}

int soscf_engine_UHF::finalize(){
    
    
    delete[] orb_grad_a    ;
    delete[] orb_shift_a   ;
    delete[] app_orb_hess_a;
    delete[] rot_matr_a    ;
    delete[] rot_matr_b    ;
    return 0;
}

int soscf_engine_UHF::orb_grad_calc_a(double * F){
    
    //     g(ia)=4*F(i,a);
    for(int i=0;i<n_el_a;i++)
    for(int j=0;j<n_ao_a-n_el_a;j++)
        orb_grad_a[i*(n_ao_a-n_el_a)+j]=4*F[(n_el_a+j)*n_ao_a+i];
    return 0;
}

int soscf_engine_UHF::orb_grad_calc_b(double * F){
    
    //     g(ia)=4*F(i,a);
    for(int i=0;i<n_el_b;i++)
    for(int j=0;j<n_ao_b-n_el_b;j++)
        orb_grad_b[i*(n_ao_b-n_el_b)+j]=4*F[(n_el_b+j)*n_ao_b+i];
    return 0;
}

int soscf_engine_UHF::orb_hess_calc_a(double * F){

  
  //H(ia,ia) = 4*(e(i)-e(a))
  for(int i=0;i<n_el_a;i++)
      for(int j=0;j<n_ao_a-n_el_a;j++){
          app_orb_hess_a[j+i*(n_ao_a-n_el_a)]=4*(F[(n_el_a+j)*(n_ao_a+1)]-F[i*(n_ao_a+1)]);
    }
//           getchar();
  return 0;
      
}

int soscf_engine_UHF::orb_hess_calc_b(double * F){

  
  //H(ia,ia) = 4*(e(i)-e(a))
  for(int i=0;i<n_el_b;i++)
      for(int j=0;j<n_ao_b-n_el_b;j++){
//           fprintf(out_stream,"%e %d\n",app_orb_hess_b[j+i*(n_ao_b-n_el_b)],j+i*(n_ao_b-n_el_b));
          app_orb_hess_b[j+i*(n_ao_b-n_el_b)]=4*(F[(n_el_b+j)*(n_ao_b+1)]-F[i*(n_ao_b+1)]);
//           fprintf(out_stream,"%e %d\n",app_orb_hess_b[j+i*(n_ao_b-n_el_b)],j+i*(n_ao_b-n_el_b));
//           fprintf(out_stream,"%d %d",i,j);
//           getchar();
    }
//           getchar();
  return 0;
      
}

double soscf_engine_UHF::calc_a(double * F){
    
    orb_grad_calc_a(F);
    orb_hess_calc_a(F);
    //determination of maximum grad element
    double max=fabs(orb_grad_a[0]);
    for (int i=1;i<n_el_a*(n_ao_a-n_el_a);i++)
        if(fabs(orb_grad_a[i])>max)max=fabs(orb_grad_a[i]);
//     fprintf(out_stream,"orb_grad = %e\n",max);
    return max;
}

double soscf_engine_UHF::calc_b(double * F){
//     PrintMatr(app_orb_hess_a,1,n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b),1);
    orb_grad_calc_b(F);
//     PrintMatr(app_orb_hess_a,1,n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b),1);
    orb_hess_calc_b(F);
//     PrintMatr(app_orb_hess_a,1,n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b),1);
    //determination of maximum grad element
    double max=fabs(orb_grad_b[0]);
    for (int i=1;i<n_el_b*(n_ao_b-n_el_b);i++)
        if(fabs(orb_grad_b[i])>max)max=fabs(orb_grad_b[i]);
//     fprintf(out_stream,"orb_grad = %e\n",max);
    return max;
}

int soscf_engine_UHF::rot_matr_calc(){
    
//     PrintMatr(orb_grad_a    ,n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b),1,1);
//     PrintMatr(app_orb_hess_a,n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b),1,1);
    
    double zoom = BFGS.step(orb_shift_a, orb_grad_a, app_orb_hess_a, do_nothing, do_nothing, n_el_a*(n_ao_a-n_el_a)+n_el_b*(n_ao_b-n_el_b),x_max_soscf);
    
    if(zoom > -1.0){
        fprintf(out_stream,"SOSCF is scaling rotation angle matrix Xmax=%f\n",-x_max_soscf/zoom);
 
    }
    
    //generation of rot matr_a
    set_zero_matr(rot_matr_a, n_ao_a*n_ao_a);
    for(int i=0;i<n_ao_a;i++) rot_matr_a[i*(n_ao_a+1)]=1.0;// 1 at diagonal
    for(int i=0;i<n_el_a;i++)
    for(int j=0;j<(n_ao_a-n_el_a);j++){
        rot_matr_a[i*n_ao_a+ n_el_a+j ]= orb_shift_a[i*(n_ao_a-n_el_a)+j]; //upper corner
        rot_matr_a[i+n_ao_a*(n_el_a+j)]=-orb_shift_a[i*(n_ao_a-n_el_a)+j]; //lower corner
    }
          
    ortogonalization(rot_matr_a, n_ao_a);//ortogonalization of rot matrix
    
    //generation of rot matr_b
    set_zero_matr(rot_matr_b, n_ao_b*n_ao_b);
    for(int i=0;i<n_ao_b;i++) rot_matr_b[i*(n_ao_b+1)]=1.0;// 1 at diagonal
    for(int i=0;i<n_el_b;i++)
    for(int j=0;j<(n_ao_b-n_el_b);j++){
        rot_matr_b[i*n_ao_b+ n_el_b+j ]= orb_shift_b[i*(n_ao_b-n_el_b)+j]; //upper corner
        rot_matr_b[i+n_ao_b*(n_el_b+j)]=-orb_shift_b[i*(n_ao_b-n_el_b)+j]; //lower corner
    }
          
    ortogonalization(rot_matr_b, n_ao_b);//ortogonalization of rot matrix
    return 0;
}
    
int soscf_engine_UHF::vec_update(double * VEC, double * BUF, char ab){
    double * rot_matr;
    int n_ao;
    
    if(ab=='a'){
//         fprintf(out_stream,"using standart SOSCF for A set\n");
        rot_matr = rot_matr_a;
        n_ao = n_ao_a;
    }
    else if(ab=='b'){
//         fprintf(out_stream,"using standart SOSCF for B set\n");
        rot_matr = rot_matr_b;
        n_ao = n_ao_b;
    }
    else{
//         fprintf(out_stream,"wrong argument ab in vec_update\n");
        exit(1);
    }
    
    
    //rotation of vectors
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_ao,n_ao,n_ao,1.0, rot_matr  ,n_ao,VEC,n_ao,0.0,BUF,n_ao);//TMP = C * U^T
    for(int i=0;i<n_ao*n_ao;i++)VEC[i]=BUF[i];
    
    return 0;
    
}

int soscf_engine_UHF::vec_update_OHF(double * VEC, double * O_B, double * M, double * B,int n_ort, char ab){
    
    if(n_ort==0){
        vec_update(VEC, B, ab);
        return 0;
    }
    
    double * rot_matr;
    int n_ao;
    
    if(ab=='a'){
//         fprintf(out_stream,"using OHF SOSCF for A set\n");
        rot_matr = rot_matr_a;
        n_ao = n_ao_a;
    }
    else if(ab=='b'){
//         fprintf(out_stream,"using OHF SOSCF for B set\n");
        rot_matr = rot_matr_b;
        n_ao = n_ao_b;
    }
    else{
        fprintf(out_stream,"wrong argument ab in vec_update_OHF\n");
        exit(1);
    }
    
    
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ao, n_ao+n_ort, n_ao,1.0,
                        rot_matr,n_ao,
                        O_B, n_ao+n_ort,0.0,
                        B,n_ao+n_ort);
    
//     fprintf(out_stream,"OB1:\n");
//     PrintMatr(O_B,n_ao+n_ort,n_ao+n_ort,1);
    
//     for(int i=0;i<n_ort*n_ao+n_ort;i++)
//         B[(n_ao+n_ort-n_ort)*n_ao+n_ort+i]=O[i];
//     
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ao+n_ort,n_ao+n_ort,n_ao+n_ort,1.0,
                        B,n_ao+n_ort,
                        M,n_ao+n_ort,0.0,
                        VEC,n_ao+n_ort);
//     fprintf(out_stream,"V:\n");
//     PrintMatr(VEC,n_ao+n_ort,n_ao+n_ort,1);
    return 0;
    
}

soscf_engine_MCSCF::soscf_engine_MCSCF(){
    
    orb_grad    = NULL;
    orb_shift   = NULL;
    app_orb_hess= NULL;
    rot_matr    = NULL;
    
}

int soscf_engine_MCSCF::init(int n, int ext_n_c, int ext_n_a, int ext_n_v, int ext_n_ao, double ext_x_max_soscf){
    
    grad_max_soscf=1;
    grad_min_soscf=0.000005;
    n_ao = ext_n_ao;
    n_c  = ext_n_c ;
    n_a  = ext_n_a ;
    n_v  = ext_n_v ;
    n_mo = n_c+n_a+n_v;
    
    x_max_soscf = ext_x_max_soscf;
//     N_LBFGS_VECTORS=20;

    BFGS.init(n_c*n_a+n_c*n_v+n_a*n_v,n);
    
    orb_grad = new double [n_c*n_a+n_c*n_v+n_a*n_v];
    orb_shift = new double [n_c*n_a+n_c*n_v+n_a*n_v];
    for(int i=0;i<n_c*n_a+n_c*n_v+n_a*n_v;i++)orb_shift[i]=0;
    app_orb_hess = new double [n_c*n_a+n_c*n_v+n_a*n_v];
    rot_matr=new double[n_mo*n_mo];
//     MOL = A;
    return 0;
}

// Zero step count = the next step() runs as the first one: no curvature pair is formed from the
// stale previous gradient, and none of the stored ones are used.
void soscf_engine_MCSCF::reset_history(){

    BFGS.lbfgs_step_num = 0;
}

soscf_engine_MCSCF::~soscf_engine_MCSCF(){
    
    if(orb_grad    !=NULL) delete[] orb_grad    ;
    if(orb_shift   !=NULL) delete[] orb_shift   ;
    if(app_orb_hess!=NULL) delete[] app_orb_hess;
    if(rot_matr    !=NULL) delete[] rot_matr    ;
    
}


int soscf_engine_MCSCF::orb_grad_calc(double * G){
    
    //     g is precalculated
    for(int i=0;i<n_c*n_a+n_c*n_v+n_a*n_v;i++)
        orb_grad[i]=G[i];
    
    return 0;
}

int soscf_engine_MCSCF::orb_hess_calc(double * B){
    
    for(int i=0;i<n_c*n_a+n_c*n_v+n_a*n_v;i++)
          app_orb_hess[i]=B[i];

    return 0;
      
}

double soscf_engine_MCSCF::calc(double * G, double * B){
    
    orb_grad_calc(G);
    orb_hess_calc(B);
    //determination of maximum grad element
    double max=fabs(orb_grad[0]);
    for (int i=1;i<n_c*n_a+n_c*n_v+n_a*n_v;i++)
        if(fabs(orb_grad[i])>max)max=fabs(orb_grad[i]);
//     fprintf(out_stream,"orb_grad = %e\n",max);
    return max;
}

double rot_matr_x_max(double * R, int n){
    
    double x_max=0;
    for(int i=0  ; i<n; i++)
    for(int j=i+1; j<n; j++)
        if(fabs(R[i*n+j])>x_max)x_max=fabs(R[i*n+j]);
    
    return x_max;
    
}


double soscf_engine_MCSCF::step(double * VEC, double * BUF){
    
    double zoom = BFGS.step(orb_shift, orb_grad, app_orb_hess, do_nothing, do_nothing, n_c*n_a+n_c*n_v+n_a*n_v,x_max_soscf);
    
    int scaling=0;
    
    if(zoom > -1.0){
        fprintf(out_stream," SOSCF is scaling rotation angle matrix Xmax=%.5e                         |\n",-x_max_soscf/zoom);
        scaling=1;
    }
    
    //generation of rot matr
    set_zero_matr(rot_matr, n_mo*n_mo);
    for(int i=0;i<n_mo;i++) rot_matr[i*(n_mo+1)]=1.0;// 1 at diagonal
    
    //CA
    for(int i=0;i<n_c;i++)
    for(int t=0;t<n_a;t++){
        rot_matr[i*n_mo+ n_c+t ]= orb_shift[i*n_a+t]; //upper corner
        rot_matr[i+n_mo*(n_c+t)]=-orb_shift[i*n_a+t]; //lower corner
    }
    //CV
    for(int i=0;i<n_c;i++)
    for(int a=0;a<n_v;a++){
        rot_matr[i*n_mo+ n_c+n_a+a ]= orb_shift[n_c*n_a+i*n_v+a]; //upper corner
        rot_matr[i+n_mo*(n_c+n_a+a)]=-orb_shift[n_c*n_a+i*n_v+a]; //lower corner
    }
    //AV
    for(int t=0;t<n_a;t++)
    for(int a=0;a<n_v;a++){
        rot_matr[(n_c+t)*n_mo+ n_c+n_a+a ]= orb_shift[n_c*n_a+n_c*n_v+t*n_v+a]; //upper corner
        rot_matr[(n_c+t)+n_mo*(n_c+n_a+a)]=-orb_shift[n_c*n_a+n_c*n_v+t*n_v+a]; //lower corner
    }
    
//     fprintf(out_stream,"R\n");
//     PrintMatr(rot_matr,n_mo,n_mo,0);
    
    double step = rot_matr_x_max(rot_matr,n_mo);
    ortogonalization(rot_matr, n_mo);//ortogonalization of rot matrix
    //rotation of vectors
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_mo,n_ao,n_mo,1.0, rot_matr  ,n_mo,VEC,n_ao,0.0,BUF,n_ao);//TMP = C * U^T
    for(int i=0;i<n_ao*n_ao;i++)VEC[i]=BUF[i];
    
    return step;
    
}

