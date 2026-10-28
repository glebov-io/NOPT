#include <cstring>
#include <cstdlib>
# include <stdio.h>
# include "blas_link.h"
# include <omp.h>


# include "CI.h"
# include "matr.h"
# include "timer.h"

# include "from_hash.h"
# include "aldet.h"
# include "RI.h"
# include "res_fit.h"


# define max(a,b)  (((a)<(b))?(b):(a))
# define min(a,b)  (((a)>(b))?(b):(a))

int n_fit_pol;

extern int num_threads;

inline double ddot(int n, double * A , int a, double * B, int b){
    double res = 0.0;
    for(int i=0;i<n;i++) res+=A[i]*B[i];
//     printf("A              B\n");
//     for(int i=0;i<n;i++)printf("%e   %e\n",A[i],B[i]);
    return res;
    
}

inline double ED_with_shift(double E, double edshift){
    
    return E/(E*E+edshift);
}

int res_fit_calc_arrays(double E, double * E_fit, int N_fit, int dim, int * s, int * l, double * appr_coef){
    
    int i=0;
    
    while((E_fit[i]<E)&&(i<N_fit))i++;
    
//     printf("%e %e %e %d\n", E_fit[i-1], E, E_fit[i],i);
//     printf("%e %e\n",R_fit[i-1],R_fit[i]);
    
    if(i>0 && i<N_fit && fabs(E_fit[i-1]-E)<fabs(E_fit[i]-E)){
        i--;
    }
    s[0] = max(0      ,i-dim);
    int e = min(N_fit-1,i+dim);
    l[0] = e+1-s[0];
//     l[0]=2;
    // Barycentric Lagrange interpolation weights -- stable at any stencil order, unlike
    // the inverse monomial Vandermonde it replaces (exponentially ill-conditioned).
    int L = l[0];
    double * x  = new double[L];
    double * bw = new double[L];

    double cen = 0.5*(E_fit[s[0]] + E_fit[s[0]+L-1]);
    double hlf = 0.5*(E_fit[s[0]+L-1] - E_fit[s[0]]); if(hlf==0.0) hlf=1.0;
    for(int k=0;k<L;k++) x[k]=(E_fit[s[0]+k]-cen)/hlf;
    double Es=(E-cen)/hlf;

    for(int k=0;k<L;k++){
        double p=1.0;
        for(int j=0;j<L;j++) if(j!=k) p*=(x[k]-x[j]);
        bw[k]=1.0/p;
    }

    int hit=-1; double denom=0.0;
    for(int k=0;k<L;k++){
        double d=Es-x[k];
        if(d==0.0){ hit=k; }
        else { appr_coef[k]=bw[k]/d; denom+=appr_coef[k]; }
    }
    if(hit>=0){ for(int k=0;k<L;k++) appr_coef[k]=0.0; appr_coef[hit]=1.0; }
    else      { double iv=1.0/denom; for(int k=0;k<L;k++) appr_coef[k]*=iv; }

    delete[] x;
    delete[] bw;
    return 0;
}

int find_min_max_minus_E_ci_normal(double * min, double * max, double * E, int n){
    
    min[0]= 1E+18;
    max[0]=-1E+18;
    
    double v;
    for(int i=0; i<n;i++){
        v=-E[i];
        if(v<min[0])min[0]=v;
        if(v>max[0])max[0]=v;
    }
    
    return 1;
    
}


int add_min_max_E_arr(double * min_out, double * max_out, double * E, int n){
    
    double min= 1E+18;
    double max=-1E+18;
    
//     double v;
    for(int i=0; i<n;i++){
//         v=det.second[n];
        if(E[i]<min)min=E[i];
        if(E[i]>max)max=E[i];
    }
    
    min_out[0]+=min;
    max_out[0]+=max;
    
    return 1;
    
}


int fill_grid(double * G,int N){
    
    N--;
    double step =(G[N]-G[0])/N;
    
    for(int i=1;i<N;i++)
        G[i]=G[i-1]+step;
    
//     PrintMatr(G,N,1,1);
    
    
    return 1;

}

int res_fit_data::set_par(aldet_data * ext_ci, double * ext_E0, int ext_n_s, RI_data * ext_RI, 
                          double * ext_eps, int ext_n_c, int ext_n_a, int ext_n_v, 
                          double * ext_H_AV, double * ext_H_CA, double * ext_H_CV,
                          double ext_edshift){
    ci      = ext_ci     ; 
    RI      = ext_RI     ; 
    E0      = ext_E0     ; 
    n_s     = ext_n_s    ; 
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
    
    
    return 0;
}

int res_fit_data::gen_grid(int ext_N_fit, int ext_n_fit_pol){
    
    N_fit = ext_N_fit;
    
    n_fit_pol = ext_n_fit_pol;
    
    E_fit = new double[N_fit];
    
    int n_s = ci->n_states[0];
    
    find_min_max_minus_E_ci_normal(E_fit, E_fit+N_fit-1, ci->E_act[0], ci->Na*ci->Nb);
    add_min_max_E_arr(E_fit, E_fit+N_fit-1, E0, n_s);
    fill_grid(E_fit,N_fit);
    
    S_fit = new int[ci->Na*ci->Nb*n_s];
    L_fit = new int[ci->Na*ci->Nb*n_s];
    C_fit = new double[ci->Na*ci->Nb*n_s*(2*n_fit_pol+1)];
    
    S_fit_tr = new int[ci->Na*ci->Nb*n_s];
    L_fit_tr = new int[ci->Na*ci->Nb*n_s];
    C_fit_tr = new double[ci->Na*ci->Nb*n_s*(2*n_fit_pol+1)];
    

    for(int i_CI = 0; i_CI<ci->Na; i_CI++)
    for(int j_CI = 0; j_CI<ci->Nb; j_CI++){
       for(int i_s=0;i_s<n_s;i_s++){
           int i_f = (i_CI*ci->Nb+j_CI)*n_s+i_s;
           res_fit_calc_arrays(E0[i_s]-ci->E_act[0][i_CI*ci->Nb+j_CI],E_fit,N_fit,n_fit_pol, S_fit+i_f, L_fit+i_f,C_fit+ i_f*(2*n_fit_pol+1));
       }
       
    }
    transpose_3d_abc_to_bac_int(S_fit_tr, S_fit, ci->Na, ci->Nb, n_s);
    transpose_3d_abc_to_bac_int(L_fit_tr, L_fit, ci->Na, ci->Nb, n_s);
    transpose_3d_abc_to_bac    (C_fit_tr, C_fit, ci->Na, ci->Nb, n_s*(2*n_fit_pol+1));
    
    int n_a = ci->n_act;
    
    RF_PS    = new double  [                        N_fit];
    RF_PH    = new double  [n_a*n_a                *N_fit];
    RF_PV    = new double  [n_a*n_a*n_a*n_a        *N_fit];
    RF_PV_JK = new double  [n_a*n_a*n_a*n_a        *N_fit];
    RF_PV_AB = new double  [n_a*n_a*n_a*n_a        *N_fit];
    RF_P3_JK = new double  [n_a*n_a*n_a*n_a*n_a*n_a*N_fit];
    RF_P3_AB = new double  [n_a*n_a*n_a*n_a*n_a*n_a*N_fit];

    return 0;
}


int res_fit_calc_0body(double * P, int n_s, int ld,
                       double * ci,
                       int Na, int Nb,
                       double * RF, int * L_fit, int * S_fit, double * C_fit, int N_fit,
                       int i_th, int n_th){
    
    double Rc;
        
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th)//{//fprintf(stderr,"AV(1-body) i_CI=%d  (thread %d)        \r",i_CI, i_th);
    for(int j_CI = 0; j_CI<Nb; j_CI++)
    for(int i_s=0;i_s<n_s;i_s++){
        Rc = cblas_ddot(L_fit[(i_CI*Nb+j_CI)*n_s+i_s],RF+S_fit[(i_CI*Nb+j_CI)*n_s+i_s], 1, C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s), 1);
        Rc =Rc*ci[(i_CI*Nb+j_CI  )*ld+i_s];
        for(int j_s=0;j_s<n_s;j_s++)
            P[i_s*n_s+j_s]+=Rc*ci[(i_CI*Nb+j_CI)*ld+j_s];
    }

    return 0;
}

int res_fit_calc_1body(double * P, int n_s, int ld,
                       double * ci,
                       int N,int na, int Na, int Nb, int * fa, int * vec_a,
                       double * RF, int * L_fit, int * S_fit, double * C_fit, int N_fit,
                       int i_th, int n_th){
    int * bit_a = new int [N];
//     int * bit_b = new int [N];
    int * buf = new int [na + 1];//more space for +a-b and -a+b
    buf[0]=0;
    
    double sign ;
    double sign1;
    double sign2;
    double Rc;
        
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"AV(1-body) i_CI=%d  (thread %d)        \r",i_CI, i_th);
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //A00A
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign2=1;
            bit_a[t]=0;
            for(int w=0;w<N;w++){
                if(bit_a[w]==0){
                    bit_a[w]=1;
                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                    sign=sign1*sign2;
                    for(int j_CI = 0; j_CI<Nb; j_CI++)
                    for(int i_s=0;i_s<n_s;i_s++){
                        Rc = cblas_ddot(L_fit[(i_CI*Nb+j_CI)*n_s+i_s],RF+(t*N+w)*N_fit+S_fit[(i_CI*Nb+j_CI)*n_s+i_s], 1, C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s), 1);
                        
                        Rc =Rc*ci[(i_CI*Nb+j_CI  )*ld+i_s]*sign;
                        for(int j_s=0;j_s<n_s;j_s++)
                            P[i_s*n_s+j_s]+=Rc*ci[(i_CIext*Nb+j_CI)*ld+j_s];
                    }
                    bit_a[w]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
    }
    
    delete[] bit_a;
    delete[] buf;
    
    
    return 0;
}


// t.
// u.
// v'
// w'
// u-t-v-w



int res_fit_calc_2body_AA(double * P, int n_s, int ld,
                          double * ci,
                          int N,int na, int Na, int Nb, int * fa, int * vec_a,
                          double * RF, int * L_fit, int * S_fit, double * C_fit, int N_fit,
                          int i_th, int n_th){
    
    int * bit_a = new int [N];
    int * buf = new int [na + 1];//more space for +a-b and -a+b
    buf[0]=0;
    
    double sign ;
    double sign1;
    double sign2;
    double sign3;
    
    double Rc;
    
    double * R;
    int L;
    int S;
    double * C;
    double * CI_0;
    double * CI_e;
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"AV(2-body-sym) i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //AAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign1=1;
            bit_a[t]=0;
            for(int u=t+1;u<N;u++){
                if(bit_a[u]!=0){
                    sign2=1;
                    bit_a[u]=0;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]==0){
                            sign3=sign2;
                            bit_a[v]=1;
                            for(int w=v+1;w<N;w++){
                                if(bit_a[w]==0){
                                    bit_a[w]=1;
                                    sign=sign1*sign2*sign3;
                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                    R = RF+(((u*N+t)*N+v)*N+w)*N_fit;
                                    for(int j_CI = 0; j_CI<Nb; j_CI++)
//                                     if(t!=v)if(u!=v)
                                    for(int i_s=0;i_s<n_s;i_s++){
                                        Rc =0;
                                        L = L_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                        S = S_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                        C = C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s);
                                        //<tu|wa>*P[a,v]-<tu|va>*P[a,w] to be checked
                                        Rc+=ddot(L,R+S, 1, C, 1);
                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                        Rc =Rc*CI_0[i_s]*sign;
                                        CI_e = ci+(i_CIext*Nb+j_CI)*ld;
                                        for(int j_s=0;j_s<n_s;j_s++)
                                            P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                    }
                                    bit_a[w]=0;
                                }
                                else
                                    sign3=-sign3;
                            }
                            bit_a[v]=0;
                        }
                        else
                            sign2=-sign2;
                    }
                    bit_a[u]=1;
                    sign1=-sign1;
                }
            }
            bit_a[t]=1;
        }
    }
    
    delete[] bit_a;
    delete[] buf;
    
    
    return 0;
}

//     t_a .
//     u_b .
//     v_a '
//     w_b '
//     
//     t-u-v-w
int res_fit_calc_2body_AB(double * P, int n_s, int ld,
                          double * ci,
                          int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                          double * RF, int * L_fit, int * S_fit, double * C_fit, int N_fit,
                          int i_th, int n_th){
    
    
    const int Na_zv = (int) std::lround(tgammal(N) / tgammal(na) / tgammal(N-na+1));
    const int Nb_zv = (int) std::lround(tgammal(N) / tgammal(nb) / tgammal(N-nb+1));
    int * Ia = new int [Na_zv];
    int * Ia_friends = new int [Na_zv * (N-na) * 3];
    int * Ib = new int [Nb_zv];
    int * Ib_friends = new int [Nb_zv * (N-nb) * 3];
//     
    double Rc;
    double sign;
    double * R;
    int L;
    int S;
    double * C;
    double * CI_0;
    double * CI_e;

    
    
    for(int t=0  ;t<N;t++){
        get_I_I_friends(na, N, Na, fa, t, vec_a, Ia, Ia_friends);//if(nt){printf("nt>1\n");exit(0);}
        for(int u=0  ;u<N;u++){//fprintf(stderr,"AV(ABAB) t=%d u=%d  (thread %d)        \r",t,u, i_th);
            get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib, Ib_friends);
            for (int i_CI = i_th; i_CI<Na_zv; i_CI+=n_th){
                int k_CI = Ia[i_CI];
                for (int j_CI = 0; j_CI<Nb_zv; j_CI++){
                    int l_CI = Ib[j_CI];
                    CI_0=ci+(k_CI*Nb+l_CI  )*ld;
                    R=RF+(((t*N+u)*N+t)*N+u)*N_fit;
                    for(int i_s=0;i_s<n_s;i_s++){
                        Rc =0;
                        L=L_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                        S=S_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                        C=C_fit+(2*n_fit_pol+1)*((k_CI*Nb+l_CI)*n_s+i_s);
                        Rc+=ddot(L,R+S, 1, C, 1);
                        Rc =Rc*CI_0[i_s];
//                         printf("Rc = %e",Rc);
//                         getchar();
                        for(int j_s=0;j_s<n_s;j_s++)
                            P[i_s*n_s+j_s]+=Rc*CI_0[j_s];
                    }
                    for (int n = 0; n<(N-nb); n++){
                    int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                        CI_e = ci+(k_CI*Nb+ind_n[0])*ld;
//                         fprintf(stderr,"t = %d\nu = %d\ni_CI = %d(%d)\nj_CI = %d(%d)\nn = %d\nind_n = {%d %d %d}\n",t,u,i_CI,k_CI,j_CI,l_CI,n,ind_n[0],ind_n[1],ind_n[2]);
//                         getchar();
                        R=RF+(((t*N+u)*N+t)*N+ind_n[1])*N_fit;
                        for(int i_s=0;i_s<n_s;i_s++){
                            L=L_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                            S=S_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                            C=C_fit+(2*n_fit_pol+1)*((k_CI*Nb+l_CI)*n_s+i_s);
                            Rc=ddot(L,R+S, 1, C, 1);
                            Rc =Rc*CI_0[i_s]*ind_n[2];
                            for(int j_s=0;j_s<n_s;j_s++)
                                P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                        }
                    }
                    for (int m = 0; m<(N-na); m++){
                        int * ind_m = Ia_friends + i_CI*(N-na)*3 + m*3;
                        CI_e = ci+(ind_m[0]*Nb+l_CI)*ld;
                        R=RF+(((t*N+u)*N+ind_m[1])*N+u)*N_fit;
                        for(int i_s=0;i_s<n_s;i_s++){
                            L=L_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                            S=S_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                            C=C_fit+(2*n_fit_pol+1)*((k_CI*Nb+l_CI)*n_s+i_s);
                            Rc=ddot(L,R+S, 1, C, 1);
                            Rc =Rc*CI_0[i_s]*ind_m[2];
                            for(int j_s=0;j_s<n_s;j_s++)
                                P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                        }
                        for (int n = 0; n<(N-nb); n++){
                            int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                            CI_e = ci+(ind_m[0]*Nb+ind_n[0])*ld;
                            R=RF+(((t*N+u)*N+ind_m[1])*N+ind_n[1])*N_fit;
                            sign = ind_m[2]*ind_n[2];
                            for(int i_s=0;i_s<n_s;i_s++){
                                L=L_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                                S=S_fit[(k_CI*Nb+l_CI)*n_s+i_s];
                                C=C_fit+(2*n_fit_pol+1)*((k_CI*Nb+l_CI)*n_s+i_s);
                                Rc=ddot(L,R+S, 1, C, 1);
                                Rc =Rc*CI_0[i_s]*sign;
                                for(int j_s=0;j_s<n_s;j_s++)
                                    P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                            }
                        }
                    }
                }
            }
        }
    }
    
    delete[] Ia        ;
    delete[] Ia_friends;
    delete[] Ib        ;
    delete[] Ib_friends;
    
    return 0;
       
}

int res_fit_calc_3body_AAA(double * P, int n_s, int ld,
                           double * ci,
                           int N,int na, int Na, int Nb, int * fa, int * vec_a,
                           double * RF, int * L_fit, int * S_fit, double * C_fit, int N_fit,
                           int i_th, int n_th){
    
//     printf("3 body AAA is not written\n");
//     exit(0);
    
    
    int * bit_a = new int [N];
    int * buf = new int [na + 1];//more space for +a-b and -a+b
    buf[0]=0;
    
    double sign ;
    double sign1;
    double sign2;
    double sign3;
    double sign4;
    
    double Rc;
    
    double * R;
    int L;
    int S;
    double * C;
    double * CI_0;
    double * CI_e;
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"AV(2-body-sym) i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;

        for(int t=0  ;t<N;t++){
            if(bit_a[t]!=0){
                bit_a[t]=0;
                sign1=1;
                for(int u=t+1;u<N;u++)if(bit_a[u]!=0){
                    bit_a[u]=0;
                    sign2=1;
                    for(int v=0;v<N;v++){
                        if(bit_a[v]==0){
                            bit_a[v]=1;
                            sign3=1;
                            for(int w=0  ;w<N;w++)if(bit_a[w]!=0){
                                bit_a[w]=0;
                                for(int x=0  ;x<N;x++)if(bit_a[x]==0){
                                    sign4=1;
                                    for(int y=x+1;y<N;y++){
                                        if(bit_a[y]==0){
                                            bit_a[x]=1;
                                            bit_a[y]=1;
                                            
                                            auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                            sign=sign1*sign2*sign3*sign4;
                                            R = RF+(((((t*N+u)*N+v)*N+y)*N+x)*N+w)*N_fit;
                                            for(int j_CI = 0; j_CI<Nb; j_CI++)
                                            for(int i_s=0;i_s<n_s;i_s++){
                                                Rc =0;
                                                L = L_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                S = S_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                C = C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s);
                                                //<tu|wa>*P[a,v]-<tu|va>*P[a,w] to be checked
                                                Rc+=ddot(L,R+S, 1, C, 1);
                                                CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                Rc =Rc*CI_0[i_s]*sign;
                                                CI_e = ci+(i_CIext*Nb+j_CI)*ld;
                                                for(int j_s=0;j_s<n_s;j_s++)
                                                    P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                            }        
                                            
                                            bit_a[x]=0;
                                            bit_a[y]=0;                                            }
                                        else
                                            sign4=-sign4;
                                    }
                                }
                                bit_a[w]=1;
                                sign3=-sign3;                                }
                            bit_a[v]=0;                            }
                        else
                            sign2=-sign2;
                    }
                    bit_a[u]=1;
                    sign1=-sign1;
                }
                bit_a[t]=1;
            }
        }        
    }
    
    delete[] bit_a;
    delete[] buf;
    
    
    return 0;
}


//     t_a .
//     u_a .
//     v_a '
//     w_a '
//     x_b .
//     y_b '
//     
//     t-u-v-w-x-y
int res_fit_calc_3body_AB(double * P, int n_s, int ld,
                          double * ci,
                          int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                          double * RF, int * L_fit, int * S_fit, double * C_fit, int N_fit,
                          int i_th, int n_th){
    
    int * bit_a = new int [N];
    int * bit_b = new int [N];
    int * buf = new int [na + 1];//more space for +a-b and -a+b
    buf[0]=0;
    
    double sign ;
    double sign1;
    double sign2;
    double sign3;
    double sign4;
    double sign5;
    double sign6;
    
    
    double Rc;
    
    double * R;
    int L;
    int S;
    double * C;
    double * CI_0;
    double * CI_e;

    //fprintf(stderr,"\n");
    for (int i_CI = i_th; i_CI<Na; i_CI+=n_th)/// must be optimized!!!!!!!
    for (int j_CI = 0; j_CI<Nb; j_CI++){//fprintf(stderr,"\r3b CI=%d/%d          \n",i_CI*Nb+j_CI,Na*Nb);
//         getchar();

        memset(bit_a,0,N*sizeof(int));
        memset(bit_b,0,N*sizeof(int));
        
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        for(int t=0  ;t<N;t++){
            if(bit_a[t]!=0){
                bit_a[t]=0;
                sign1=1;
                for(int u=t+1;u<N;u++)if(bit_a[u]!=0){
                    bit_a[u]=0;
                    sign2=1;
                    for(int v=0;v<N;v++){
                        if(bit_a[v]==0){
                            bit_a[v]=1;
                            sign5=sign2;
                            for(int w=v+1  ;w<N;w++){
                                if(bit_a[w]==0){
                                    bit_a[w]=1;
                                    sign4=1;
                                    for(int x=0  ;x<N;x++)if(bit_b[x]!=0){
                                        bit_b[x]=0;
                                        sign6=1;
                                        for(int y=0  ;y<N;y++){
                                            if(bit_b[y]==0){
                                                bit_b[y]=1;
                                                auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                auto j_CIext = get_ind_from_ON(bit_b, N, nb, fb, buf);
                                                
                                                sign=sign1*sign2*sign4*sign5*sign6;
                                                
                                                R = RF+(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*N_fit;
                                                for(int i_s=0;i_s<n_s;i_s++){
                                                    Rc =0;
                                                    L = L_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                    S = S_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                    C = C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s);
                                                    Rc+=ddot(L,R+S, 1, C, 1);
                                                    CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                    Rc =Rc*CI_0[i_s]*sign;
                                                    CI_e = ci+(i_CIext*Nb+j_CIext)*ld;
                                                    for(int j_s=0;j_s<n_s;j_s++)
                                                        P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                                }
                                                bit_b[y]=0;
                                                
                                            }
                                            else
                                                sign6=-sign6;
                                        }
                                        bit_b[x]=1;
                                        sign4=-sign4;
                                    }
                                    bit_a[w]=0;
                                }
                                else
                                    sign5=-sign5;
                            }
                            bit_a[v]=0;
                        }
                        else
                            sign2=-sign2;
                    }
                    bit_a[u]=1;
                    sign1=-sign1;
                }
                bit_a[t]=1;
            }
        }
    }
    
    delete[] bit_b;
    delete[] bit_a;
    delete[] buf;

    
    return 0;
}

int res_fit_data::calc_0body(double * E2, int i_th, int n_th){
    
    res_fit_calc_0body(E2, n_s, n_s,
                       ci->coef[0],
                       ci->Na, ci->Nb,
                       RF_PS, L_fit, S_fit, C_fit, N_fit,
                       i_th, n_th);

    return 0;
    
}


int res_fit_data::calc_1body_A(double * E2, int i_th, int n_th){
    
    res_fit_calc_1body(E2, n_s, n_s,
                       ci->coef[0],
                       n_a,ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a,
                       RF_PH, L_fit, S_fit, C_fit, N_fit,
                       i_th, n_th);
    
    return 0;
    
}
    
int res_fit_data::calc_1body_B(double * E2, int i_th, int n_th){
    
    res_fit_calc_1body(E2, n_s, n_s,
                       ci->coef_bas[0],
                       n_a,ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b,
                       RF_PH, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                       i_th, n_th);
    
    return 0;
    
}

int res_fit_data::calc_2body_AB(double * E2, int i_th, int n_th){
    
    res_fit_calc_2body_AB(E2, n_s, n_s,
                          ci->coef[0],
                          n_a,ci->na, ci->nb, ci->Na, ci->Nb, ci->fa, ci->fb, ci->vec_a, ci->vec_b,
                          RF_PV_AB, L_fit, S_fit, C_fit, N_fit,
                          i_th, n_th);
    
    return 0;
    
}

int res_fit_data::calc_2body_AA(double * E2, int i_th, int n_th){
    
    res_fit_calc_2body_AA(E2, n_s, n_s,
                          ci->coef[0],
                          n_a,ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a,
                          RF_PV_JK, L_fit, S_fit, C_fit, N_fit,
                          i_th, n_th);
    
    return 0;
    
}

int res_fit_data::calc_2body_BB(double * E2, int i_th, int n_th){
    
    res_fit_calc_2body_AA(E2, n_s, n_s,
                          ci->coef_bas[0],
                          n_a,ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b,
                          RF_PV_JK, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                          i_th, n_th);

    return 0;
    
}


int res_fit_data::calc_3body_AAA(double * E2, int i_th, int n_th){
    
    res_fit_calc_3body_AAA(E2, n_s, n_s,
                           ci->coef[0],
                           n_a,ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a,
                           RF_P3_JK, L_fit, S_fit, C_fit, N_fit,
                           i_th, n_th);
    
    return 0;
    
}
    
    
int res_fit_data::calc_3body_BBB(double * E2, int i_th, int n_th){

    res_fit_calc_3body_AAA(E2, n_s, n_s,
                           ci->coef_bas[0],
                           n_a,ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b,
                           RF_P3_JK, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                           i_th, n_th);
    return 0;
    
}


int res_fit_data::calc_3body_AAB(double * E2, int i_th, int n_th){
    
    res_fit_calc_3body_AB(E2, n_s, n_s,
                          ci->coef[0],
                          n_a,ci->na, ci->nb, ci->Na, ci->Nb, ci->fa, ci->fb, ci->vec_a, ci->vec_b,
                          RF_P3_AB, L_fit, S_fit, C_fit, N_fit,
                          i_th, n_th);    
    
    return 0;
    
}

int res_fit_data::calc_3body_ABB(double * E2, int i_th, int n_th){
    
    res_fit_calc_3body_AB(E2, n_s, n_s,
                          ci->coef_bas[0],
                          n_a,ci->nb, ci->na, ci->Nb, ci->Na, ci->fb, ci->fa, ci->vec_b, ci->vec_a,
                          RF_P3_AB, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                          i_th, n_th);    
    
    return 0;
    
}


int res_fit_data::E2_calc(double * E2){
    
        
    set_zero_matr(RF_P3_JK, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_AB, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    
    set_zero_matr(RF_PV_JK, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_PV_AB, n_a*n_a*n_a*n_a*N_fit);
    
    set_zero_matr(RF_PH, n_a*n_a*N_fit);
    
    set_zero_matr(RF_PS, N_fit);
    
    // test switch: NOPT_XMC_BLOCKS=CCVV,CV,... keeps only the listed classes (exact names, comma separated; default: all eight)
    const char * blk = getenv("NOPT_XMC_BLOCKS");
    auto use = [&](const char * b){
        if(!(blk && *blk)) return true;
        const size_t lb = strlen(b);
        for(const char * p = blk; *p; ){
            const char * q = strchr(p, ','); const size_t l = q ? (size_t)(q - p) : strlen(p);
            if(l == lb && !strncmp(p, b, lb)) return true;
            if(!q) break; p = q + 1;
        }
        return false;
    };
    if(blk && *blk) fprintf(out_stream, "TEST: XMC PT classes restricted to %s\n", blk);
    if(use("CCVV")){calc_RF_2_CCVV();printf_timer("CCVV res table");printf("\n");fflush(stdout);}
    if(use("CAVV")){calc_RF_2_CAVV();printf_timer("CAVV res table");printf("\n");fflush(stdout);}
    if(use("AAVV")){calc_RF_2_AAVV();printf_timer("AAVV res table");printf("\n");fflush(stdout);}
    if(use("CCAV")){calc_RF_2_CCAV();printf_timer("CCAV res table");printf("\n");fflush(stdout);}
    if(use("CCAA")){calc_RF_2_CCAA();printf_timer("CCAA res table");printf("\n");fflush(stdout);}
    if(use("CV"))  {calc_RF_2_CV  ();printf_timer("CV   res table");printf("\n");fflush(stdout);}
    if(use("AV"))  {calc_RF_2_AV  ();printf_timer("AV   res table");printf("\n");fflush(stdout);}
    if(use("CA"))  {calc_RF_2_CA  ();printf_timer("CA   res table");printf("\n");fflush(stdout);}
    
    
    double **E2_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)E2_th[i]=new double[n_s*n_s];
    for(int i=0;i<num_threads;i++)set_zero_matr(E2_th[i],n_s*n_s);

    #pragma omp parallel
    {
        int nt = omp_get_thread_num();
        
        double * E2_loc=new double[n_s*n_s];
        set_zero_matr(E2_loc,n_s*n_s);
    
        calc_0body    (E2_loc, nt, num_threads);
        calc_1body_A  (E2_loc, nt, num_threads);
        calc_1body_B  (E2_loc, nt, num_threads);
        calc_2body_AA (E2_loc, nt, num_threads);
        calc_2body_AB (E2_loc, nt, num_threads);
        calc_2body_BB (E2_loc, nt, num_threads);
        calc_3body_AAA(E2_loc, nt, num_threads);
        calc_3body_AAB(E2_loc, nt, num_threads);
        calc_3body_ABB(E2_loc, nt, num_threads);
        calc_3body_BBB(E2_loc, nt, num_threads);
                
        for(int i_s=0;i_s<n_s*n_s;i_s++)
            E2_th[nt][i_s]=E2_loc[i_s];
        delete[] E2_loc;
    }
    
    printf_timer("0-3 body PT operators");printf("\n");fflush(stdout);
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<num_threads;i++)
        E2[i_s]+=E2_th[i][i_s];
    for(int i=0;i<num_threads;i++)delete[] E2_th[i];
    delete[] E2_th;
        
    
    return 0;
    
}

int res_fit_data::calc_RF_2_CV(){
    
    double dE;
    double V;
    
    
    double * RF_MM;    // (J-K) * (J-K)
    double * RF_JJ;    //  J(aa)* J(aa)
    double * RF_JM;    //  J(aa)*(J-K)
    double * RF_MJ;    // (J-K) * J(aa)
    double * RF_AB;    //  J(ab)* J(ab)
    
    RF_MM = new double  [n_a*n_a*n_a*n_a*N_fit];
    RF_JJ = new double  [n_a*n_a*n_a*n_a*N_fit];
    RF_JM = new double  [n_a*n_a*n_a*n_a*N_fit];
    RF_MJ = new double  [n_a*n_a*n_a*n_a*N_fit];
    RF_AB = new double  [n_a*n_a*n_a*n_a*N_fit];
    set_zero_matr(RF_MM, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_JJ, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_JM, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_MJ, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_AB, n_a*n_a*n_a*n_a*N_fit);
    
    double * RF_HM;
    double * RF_MH;
    double * RF_HJ;
    double * RF_JH;
    RF_HM = new double  [n_a*n_a*N_fit];
    RF_MH = new double  [n_a*n_a*N_fit];
    RF_HJ = new double  [n_a*n_a*N_fit];
    RF_JH = new double  [n_a*n_a*N_fit];
    set_zero_matr(RF_HM, n_a*n_a*N_fit);
    set_zero_matr(RF_MH, n_a*n_a*N_fit);
    set_zero_matr(RF_HJ, n_a*n_a*N_fit);
    set_zero_matr(RF_JH, n_a*n_a*N_fit);
    
    double * RF_H;
    RF_H = new double  [n_a*N_fit];
    set_zero_matr(RF_H, n_a*N_fit);
    
    
    double * ED;
    ED = new double[N_fit];
    int i_CI=0;
    double * VCAA;
    VCAA = new double[n_c*n_v*n_a*n_a];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_c*n_v,n_a*n_a,RI->aux_n_ao,1.0,
                        RI->VC_RI_M,RI->aux_n_ao,
                        RI->AA_RI_M,RI->aux_n_ao,0.0,
                        VCAA,n_a*n_a);
    double * CAVA;
    CAVA = new double[n_a*n_c*n_a*n_v];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_a*n_c,n_a*n_v,RI->aux_n_ao,1.0,
                        RI->CA_RI_M,RI->aux_n_ao,
                        RI->VA_RI_M,RI->aux_n_ao,0.0,
                        CAVA,n_a*n_v);

    double dE_orb;
    
    for(int i=0; i<n_c; i++)
    for(int u=0; u<n_a; u++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++){
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //<i(a)t(b)|u(a)a(b)><v(a)a(b)|i(a)w(b)>=(iu|at)(iv|aw)
            V=CAVA[((i*n_a+u)*n_v+a)*n_a+t]*
              CAVA[((i*n_a+v)*n_v+a)*n_a+w];
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
            
        }
    }
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //(<it|au>-<iu|at>)(<av|iw>-<av|wi>)
            V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
               CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
              (VCAA[((a*n_c+i)*n_a+w)*n_a+v]-  //<av|iw>=(ai|wv)
               CAVA[((i*n_a+v)*n_v+a)*n_a+w]); //<av|wi>=(iv|aw)
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_MM[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
        }
    }
    

    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //(<it|au>-<iu|at>)<av|iw>
            V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
               CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
               VCAA[((a*n_c+i)*n_a+w)*n_a+v];  //<av|iw>=(ai|wv)
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_MJ[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
        }
    }
    
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //<it|au>(<av|iw>-<av|wi>)
            V= VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
              (VCAA[((a*n_c+i)*n_a+w)*n_a+v]-  //<av|iw>=(ai|wv)
               CAVA[((i*n_a+v)*n_v+a)*n_a+w]); //<av|wi>=(iv|aw)
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_JM[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
        }
    }

    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++){
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int v=0; v<n_a; v++)
        for(int w=0; w<n_a; w++){
            //<it|au><av|iw>
            V=VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
              VCAA[((a*n_c+i)*n_a+w)*n_a+v];  //<av|iw>=(ai|wv)
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_JJ[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
        }
    }
//     printf_timer("opt table");
    
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0  ;t<n_a;t++)
    for(int u=0  ;u<n_a;u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
           CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
           H_CV[i*n_v+a]; 
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                RF_MH[(t*n_a+u)*N_fit+i_f]+=V*dE/(dE*dE+edshift);
        }
    }
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0  ;t<n_a;t++)
    for(int u=0  ;u<n_a;u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V=(VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
           CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
           H_CV[i*n_v+a]; 
        dE_orb=e_c[i]-e_v[a];
        for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                RF_HM[(t*n_a+u)*N_fit+i_f]+=V*dE/(dE*dE+edshift);
        }
    }

    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0  ;t<n_a;t++)
    for(int u=0  ;u<n_a;u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V= VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
           H_CV[i*n_v+a]; 
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                RF_JH[(t*n_a+u)*N_fit+i_f]+=V*dE/(dE*dE+edshift);
        }
    }
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0  ;t<n_a;t++)
    for(int u=0  ;u<n_a;u++){
        //(<it|au>-<iu|at>)*H[i,a]
        V= VCAA[((a*n_c+i)*n_a+u)*n_a+t]*  //<it|au>=(ai|tu)
           H_CV[i*n_v+a]; 
        dE_orb=e_c[i]-e_v[a];
        for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                RF_HJ[(t*n_a+u)*N_fit+i_f]+=V*dE/(dE*dE+edshift);
        }
    }
    
    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++){
        //H[i,a]*H[i,a]
        V=H_CV[i*n_v+a]*H_CV[i*n_v+a]; 
        dE_orb=e_c[i]-e_v[a];
        for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                RF_H[i_f]+=V*dE/(dE*dE+edshift);
        }
    }
//     set_zero_matr(RF_PV_JK, n_a*n_a*n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_JK[(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+i_f]+=+RF_MM[(((t*n_a+v)*n_a+u)*n_a+w)*N_fit+i_f]
                                                       +RF_JJ[(((t*n_a+v)*n_a+u)*n_a+w)*N_fit+i_f]
                                                       -RF_MM[(((u*n_a+v)*n_a+t)*n_a+w)*N_fit+i_f]
                                                       -RF_JJ[(((u*n_a+v)*n_a+t)*n_a+w)*N_fit+i_f]
                                                       -RF_MM[(((t*n_a+w)*n_a+u)*n_a+v)*N_fit+i_f]
                                                       -RF_JJ[(((t*n_a+w)*n_a+u)*n_a+v)*N_fit+i_f]
                                                       +RF_MM[(((u*n_a+w)*n_a+t)*n_a+v)*N_fit+i_f]
                                                       +RF_JJ[(((u*n_a+w)*n_a+t)*n_a+v)*N_fit+i_f]
                                                       ;
                                                       
                                                       
                                                       //to be checked
    
    }
    }
    
    
//     set_zero_matr(RF_PV_AB, n_a*n_a*n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=+RF_JM[(((t*n_a+v)*n_a+u)*n_a+w)*N_fit+i_f]
                                                       +RF_MJ[(((t*n_a+v)*n_a+u)*n_a+w)*N_fit+i_f]
                                                       +RF_JM[(((u*n_a+w)*n_a+t)*n_a+v)*N_fit+i_f]
                                                       +RF_MJ[(((u*n_a+w)*n_a+t)*n_a+v)*N_fit+i_f]
                                                       -RF_AB[(((t*n_a+w)*n_a+u)*n_a+v)*N_fit+i_f]
                                                       -RF_AB[(((u*n_a+v)*n_a+t)*n_a+w)*N_fit+i_f];
    }
    }
    
//     set_zero_matr(RF_PH,N_fit*n_a*n_a);
    for(int t=0; t<n_a; t++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<N_fit; i_f++){
        RF_PH[(t*n_a+w)*N_fit+i_f]+= RF_HM[(w*n_a+t)*N_fit+i_f]
                                    +RF_MH[(t*n_a+w)*N_fit+i_f]
                                    +RF_HJ[(w*n_a+t)*N_fit+i_f]
                                    +RF_JH[(t*n_a+w)*N_fit+i_f];
        for(int u=0; u<n_a; u++)
             RF_PH[(t*n_a+w)*N_fit+i_f]+= RF_MM[(((t*n_a+u)*n_a+u)*n_a+w)*N_fit+i_f]
                                         +RF_JJ[(((t*n_a+u)*n_a+u)*n_a+w)*N_fit+i_f]
                                         +RF_AB[(((t*n_a+u)*n_a+u)*n_a+w)*N_fit+i_f];
    }
    
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PS[i_f]+=2*RF_H[i_f];
    
    delete[] RF_MM ;
    delete[] RF_JJ ;
    delete[] RF_JM ;
    delete[] RF_MJ ;
    delete[] RF_AB ;
    delete[] RF_HM ;
    delete[] RF_MH ;
    delete[] RF_HJ ;
    delete[] RF_JH ;
    delete[] RF_H  ;
    delete[] ED    ;
    delete[] VCAA  ;
    delete[] CAVA  ;
    
    
    return 0;
    
}

int res_fit_data::calc_RF_2_AV(){
    
    
    int n_i=n_c+n_a;
    int n_m=n_i+n_v;
    
    int aux_n_ao = RI->aux_n_ao;
    double * VA=RI->VA_RI_M;
    double * CC=RI->CC_RI_M;
    double * VC=RI->VC_RI_M;
    double * CA=RI->CA_RI_M;
    double * AA=RI->AA_RI_M;
    
    
    double * VAAA;
    VAAA = new double[n_v*n_a*n_a*n_a];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_v*n_a,n_a*n_a,aux_n_ao,1.0,
                        VA,aux_n_ao,
                        AA,aux_n_ao,0.0,
                        VAAA,n_a*n_a);    

    
    
    double * RF_P3_JJ;
    double * RF_P3_JM;
    double * RF_P3_MJ;
    double * RF_P3_HM;
    double * RF_P3_MH;
    double * RF_P3_HJ;
    double * RF_P3_JH;
    
    RF_P3_JJ = new double[n_a*n_a*n_a*n_a*n_a*n_a*N_fit];
    RF_P3_JM = new double[n_a*n_a*n_a*n_a*n_a*n_a*N_fit];
    RF_P3_MJ = new double[n_a*n_a*n_a*n_a*n_a*n_a*N_fit];
    
    RF_P3_HM = new double[n_a*n_a*n_a*n_a*N_fit];
    RF_P3_MH = new double[n_a*n_a*n_a*n_a*N_fit];
    RF_P3_HJ = new double[n_a*n_a*n_a*n_a*N_fit];
    RF_P3_JH = new double[n_a*n_a*n_a*n_a*N_fit];
    
    set_zero_matr(RF_P3_JJ, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_JM, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_MJ, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_HM, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_MH, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_HJ, n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_JH, n_a*n_a*n_a*n_a*N_fit);
    
    
#pragma omp parallel
    {
    //MM == JK
        double dE_orb;
        double V;
        double dE;
        double H1a;
        double H2a;
        
        double * ED;
        ED = new double[N_fit];
        
        int u,v;
        int th_id = omp_get_thread_num();
        
        for(int vu=th_id; vu<n_a*n_a; vu+=num_threads){
            v=vu/n_a;
            u=vu%n_a;
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<N_fit; i_f++){
                    dE=dE_orb+E_fit[i_f];
                    ED[i_f]=dE/(dE*dE+edshift);
                }
                for(int y=0; y<n_a; y++)
                for(int x=0; x<n_a; x++)
                for(int w=0; w<n_a; w++){
                    //<aw|xy>-<aw|yx>=(yw|xa)-(xw|ya)
                    V =(VAAA[((a*n_a+t)*n_a+v)*n_a+u]-
                        VAAA[((a*n_a+u)*n_a+v)*n_a+t])*
                       (VAAA[((a*n_a+x)*n_a+w)*n_a+y]-
                        VAAA[((a*n_a+y)*n_a+w)*n_a+x]);
                    for(int i_f=0; i_f<N_fit; i_f++)
                        RF_P3_JK[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                    
                }
            }
        
            //JJ
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<N_fit; i_f++){
                    dE=dE_orb+E_fit[i_f];
                    ED[i_f]=dE/(dE*dE+edshift);
                }
                for(int y=0; y<n_a; y++)
                for(int x=0; x<n_a; x++)
                for(int w=0; w<n_a; w++){
                    //<a(a)w(b)|x(a)y(b)>=(yw|xa)
                    V=VAAA[((a*n_a+u)*n_a+v)*n_a+t]*
                      VAAA[((a*n_a+x)*n_a+w)*n_a+y];
                    
                    for(int i_f=0; i_f<N_fit; i_f++)
                        RF_P3_JJ[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                    
                }
            }
            //MJ
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<N_fit; i_f++){
                    dE=dE_orb+E_fit[i_f];
                    ED[i_f]=dE/(dE*dE+edshift);
                }
                for(int y=0; y<n_a; y++)
                for(int x=0; x<n_a; x++)
                for(int w=0; w<n_a; w++){
                    //<a(a)w(b)|x(a)y(b)>=(yw|xa)
                    V=(VAAA[((a*n_a+t)*n_a+v)*n_a+u]-
                       VAAA[((a*n_a+u)*n_a+v)*n_a+t])*
                       VAAA[((a*n_a+x)*n_a+w)*n_a+y];
            
                    for(int i_f=0; i_f<N_fit; i_f++)
                        RF_P3_MJ[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                    
                }
            }
            //JM
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<N_fit; i_f++){
                    dE=dE_orb+E_fit[i_f];
                    ED[i_f]=dE/(dE*dE+edshift);
                }
                for(int y=0; y<n_a; y++)
                for(int x=0; x<n_a; x++)
                for(int w=0; w<n_a; w++){
                    //<aw|xy>-<aw|yx>=(yw|xa)-(xw|ya)
            V= VAAA[((a*n_a+u)*n_a+v)*n_a+t]*
              (VAAA[((a*n_a+x)*n_a+w)*n_a+y]-
               VAAA[((a*n_a+y)*n_a+w)*n_a+x]);
            
                    for(int i_f=0; i_f<N_fit; i_f++)
                        RF_P3_JM[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                    
                }
            }
        }
    
        //H
        for(int a=0; a<n_v; a++)
        for(int t=th_id; t<n_a; t+=num_threads){
            dE_orb=-e_v[a]+e_a[t];
            for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                ED[i_f]=dE/(dE*dE+edshift);
            }
            for(int w=0; w<n_a; w++){
                H1a=H_AV[t*n_v+a];
                H2a=H_AV[w*n_v+a];                          
                V=H1a*H2a;
                for(int i_f=0; i_f<N_fit; i_f++)
                    RF_PH[(t*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                
            }
        }
        //HM
        for(int a=0; a<n_v; a++)
        for(int t=th_id; t<n_a; t+=num_threads){
            dE_orb=-e_v[a]+e_a[t];
            for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                ED[i_f]=dE/(dE*dE+edshift);
            }
            for(int y=0; y<n_a; y++)
            for(int x=0; x<n_a; x++)
            for(int w=0; w<n_a; w++){
                H1a=H_AV[t*n_v+a];
     
                //<aw|xy>-<aw|yx>=(yw|xa)-(xw|ya)
// #ifndef _RI 
//                 V=H1a*
//                   (IIIE[(((y+n_c)*n_i+w+n_c)*n_i+x+n_c)*n_v+a]-
//                    IIIE[(((x+n_c)*n_i+w+n_c)*n_i+y+n_c)*n_v+a]);
// #endif      
                V=H1a*
                  (VAAA[((a*n_a+x)*n_a+w)*n_a+y]-
                   VAAA[((a*n_a+y)*n_a+w)*n_a+x]);
                
                for(int i_f=0; i_f<N_fit; i_f++)
                    RF_P3_HM[(((t*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                
            }
        }
        //MH
        for(int vu=th_id; vu<n_a*n_a; vu+=num_threads){
            v=vu/n_a;
            u=vu%n_a;
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<N_fit; i_f++){
                    dE=dE_orb+E_fit[i_f];
                    ED[i_f]=dE/(dE*dE+edshift);
                }
                for(int w=0; w<n_a; w++){
                    H2a=H_AV[w*n_v+a];
                    //<aw|xy>-<aw|yx>=(yw|xa)-(xw|ya)
// #ifndef _RI 
//                     V=(IIIE[(((u+n_c)*n_i+v+n_c)*n_i+t+n_c)*n_v+a]-
//                        IIIE[(((t+n_c)*n_i+v+n_c)*n_i+u+n_c)*n_v+a])*
//                        H2a;
// #endif          

                    V=(VAAA[((a*n_a+t)*n_a+v)*n_a+u]-
                       VAAA[((a*n_a+u)*n_a+v)*n_a+t])*
                       H2a;
                    
                    for(int i_f=0; i_f<N_fit; i_f++)
                        RF_P3_MH[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                    
                }
            }
        }
        //HJ
        for(int a=0; a<n_v; a++)
        for(int t=th_id; t<n_a; t+=num_threads){
            dE_orb=-e_v[a]+e_a[t];
            for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                ED[i_f]=dE/(dE*dE+edshift);
            }
            for(int y=0; y<n_a; y++)
            for(int x=0; x<n_a; x++)
            for(int w=0; w<n_a; w++){
                H1a=H_AV[t*n_v+a];
                //<a(a)w(b)|x(a)y(b)>=(yw|xa)
// #ifndef _RI 
//                 V=H1a*
//                   IIIE[(((y+n_c)*n_i+w+n_c)*n_i+x+n_c)*n_v+a];
// #endif      
                V=H1a*
                  VAAA[((a*n_a+x)*n_a+w)*n_a+y];
                
                for(int i_f=0; i_f<N_fit; i_f++)
                    RF_P3_HJ[(((t*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                
            }
        }
        //JH
        for(int vu=th_id; vu<n_a*n_a; vu+=num_threads){
            v=vu/n_a;
            u=vu%n_a;
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<N_fit; i_f++){
                    dE=dE_orb+E_fit[i_f];
                    ED[i_f]=dE/(dE*dE+edshift);
                }
                for(int w=0; w<n_a; w++){
                    H2a=H_AV[w*n_v+a];
                   //<a(a)w(b)|x(a)y(b)>=(yw|xa)
// #ifndef _RI 
//                     V=IIIE[(((t+n_c)*n_i+v+n_c)*n_i+u+n_c)*n_v+a]*
//                       H2a;
// #endif          
                    V=VAAA[((a*n_a+u)*n_a+v)*n_a+t]*
                      H2a;
                    for(int i_f=0; i_f<N_fit; i_f++)
                        RF_P3_JH[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                    
                }
            }
        }
        delete[] ED;
        
    }
    //RF_PV_JK
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_JK[(((w*n_a+t)*n_a+v)*n_a+u)*N_fit+i_f]+= RF_P3_HM[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]
                                                       -RF_P3_HM[(((w*n_a+u)*n_a+v)*n_a+t)*N_fit+i_f]
                                                       -RF_P3_MH[(((t*n_a+w)*n_a+v)*n_a+u)*N_fit+i_f]
                                                       +RF_P3_MH[(((t*n_a+w)*n_a+u)*n_a+v)*N_fit+i_f];
    //RF_PV_AB
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<N_fit; i_f++){
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+= RF_P3_HJ[(((t*n_a+w)*n_a+v)*n_a+u)*N_fit+i_f]
                                                       +RF_P3_JH[(((u*n_a+t)*n_a+w)*n_a+v)*N_fit+i_f]
                                                       +RF_P3_HJ[(((u*n_a+v)*n_a+w)*n_a+t)*N_fit+i_f]
                                                       +RF_P3_JH[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f];
    
        for(int x=0; x<n_a; x++)
            RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f] += RF_P3_JJ[(((((t*n_a+u)*n_a+x)*n_a+v)*n_a+w)*n_a+x)*N_fit+i_f]
                                                            +RF_P3_JJ[(((((u*n_a+t)*n_a+x)*n_a+w)*n_a+v)*n_a+x)*N_fit+i_f];
    }
    //RF_P3_AB
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int y=0; y<n_a; y++)
    for(int x=0; x<n_a; x++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_P3_AB[(((((t*n_a+u)*n_a+v)*n_a+w)*n_a+x)*n_a+y)*N_fit+i_f]+=-RF_P3_MJ[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+w)*n_a+x)*N_fit+i_f]
                                                                       +RF_P3_MJ[(((((t*n_a+u)*n_a+w)*n_a+y)*n_a+v)*n_a+x)*N_fit+i_f]
                                                                       +RF_P3_JM[(((((x*n_a+t)*n_a+y)*n_a+w)*n_a+v)*n_a+u)*N_fit+i_f]
                                                                       -RF_P3_JM[(((((x*n_a+u)*n_a+y)*n_a+w)*n_a+v)*n_a+t)*N_fit+i_f]
                                                                       +RF_P3_JJ[(((((t*n_a+x)*n_a+v)*n_a+w)*n_a+y)*n_a+u)*N_fit+i_f]
                                                                       -RF_P3_JJ[(((((t*n_a+x)*n_a+w)*n_a+v)*n_a+y)*n_a+u)*N_fit+i_f]
                                                                       -RF_P3_JJ[(((((u*n_a+x)*n_a+v)*n_a+w)*n_a+y)*n_a+t)*N_fit+i_f]
                                                                       +RF_P3_JJ[(((((u*n_a+x)*n_a+w)*n_a+v)*n_a+y)*n_a+t)*N_fit+i_f];
                                                
    
    delete[] VAAA;
    delete[] RF_P3_JJ;
    delete[] RF_P3_JM;
    delete[] RF_P3_MJ;
    delete[] RF_P3_HM;
    delete[] RF_P3_MH;
    delete[] RF_P3_HJ;
    delete[] RF_P3_JH;
    
    return 0;
    
}

int res_fit_data::calc_RF_2_CA(){
    
    double dE;
    double dE_orb;
           
    double H1a;
    double H2a;
    
    double V;
    
    double * ED;
    ED = new double[N_fit];
    int i_CI=0;
    
    
    int aux_n_ao = RI->aux_n_ao;

    double * CC=RI->CC_RI_M;

    double * CA=RI->CA_RI_M;
    double * AA=RI->AA_RI_M;

    
    double * CAAA;
    CAAA = new double[n_c*n_a*n_a*n_a];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_c*n_a,n_a*n_a,aux_n_ao,1.0,
                        CA,aux_n_ao,
                        AA,aux_n_ao,0.0,
                        CAAA,n_a*n_a);
    double * CACC_J;
    CACC_J = new double[n_a*n_c*n_c];
    double * CACC_K;
    CACC_K = new double[n_a*n_c*n_c];
#pragma omp parallel for
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++)
    for(int j=0; j<n_c; j++){
        CACC_J[(i*n_a+t)*n_c+j] = cblas_ddot(aux_n_ao, CA+(i*n_a+t)*aux_n_ao, 1, CC+(j*n_c+j)*aux_n_ao, 1);
        CACC_K[(i*n_a+t)*n_c+j] = cblas_ddot(aux_n_ao, CC+(i*n_c+j)*aux_n_ao, 1, CA+(j*n_a+t)*aux_n_ao, 1);
    }
    
    
    double * RF_P3_VV;
    double * RF_P3_VH;
    double * RF_P3_HV;
    double * RF_P3_HH;
    
    RF_P3_VV = new double[n_a*n_a*n_a*n_a*n_a*n_a*N_fit];
    RF_P3_VH = new double[n_a*n_a*n_a*n_a*        N_fit];
    RF_P3_HV = new double[n_a*n_a*n_a*n_a*        N_fit];
    RF_P3_HH = new double[n_a*n_a*                N_fit];
    
    set_zero_matr(RF_P3_VV, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_P3_VH, n_a*n_a*n_a*n_a*        N_fit);
    set_zero_matr(RF_P3_HV, n_a*n_a*n_a*n_a*        N_fit);
    set_zero_matr(RF_P3_HH, n_a*n_a*                N_fit);
    
    
    
    //VV
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE_orb= e_c[i]+e_a[t]-e_a[u]-e_a[v];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int y=0; y<n_a; y++)
        for(int x=0; x<n_a; x++)
        for(int w=0; w<n_a; w++){
            
// #ifndef _RI 
//             V=AAAC[((t*n_a+u)*n_a+v)*n_c+i]*
//               AAAC[((y*n_a+x)*n_a+w)*n_c+i];
// #endif      
            V=(CAAA[((i*n_a+v)*n_a+u)*n_a+t]*
               CAAA[((i*n_a+w)*n_a+x)*n_a+y]);
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_P3_VV[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
            
        }
    }
    //VH
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE_orb= e_c[i]+e_a[t]-e_a[u]-e_a[v];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int w=0; w<n_a; w++){
            H2a=H_CA[i*n_a+w];
// #ifndef _RI 
//             V=AAAC[((t*n_a+u)*n_a+v)*n_c+i]*
//               H2a;
// #endif      
            V=(CAAA[((i*n_a+v)*n_a+u)*n_a+t]*
               H2a);

            for(int i_f=0; i_f<N_fit; i_f++)
                RF_P3_VH[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
            
        }
    }

    //HV
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE_orb= e_c[i]-e_a[t];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int y=0; y<n_a; y++)
        for(int x=0; x<n_a; x++)
        for(int w=0; w<n_a; w++){
            H1a=H_CA[i*n_a+t];
// #ifndef _RI 
//             V=H1a*
//               AAAC[((y*n_a+x)*n_a+w)*n_c+i];
// #endif      
            V=(H1a*
               CAAA[((i*n_a+w)*n_a+x)*n_a+y]);
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_P3_HV[(((t*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
            
        }
    }
    
    //HH
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE_orb= e_c[i]-e_a[t];

        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int u=0; u<n_a; u++){
            H1a=H_CA[i*n_a+t];
            H2a=H_CA[i*n_a+u];
            V=H1a*H2a;
                            
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_P3_HH[(t*n_a+u)*N_fit+i_f]+=V*ED[i_f];
            
        }
    }
    
    
    
    
//     set_zero_matr(RF_P3_JK, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_P3_JK[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w)*N_fit+i_f]+= RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+y)*n_a+w)*n_a+u)*N_fit+i_f]
                                                                       -RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+y)*n_a+w)*n_a+t)*N_fit+i_f]
                                                                       -RF_P3_VV[(((((t*n_a+v)*n_a+y)*n_a+x)*n_a+w)*n_a+u)*N_fit+i_f]
                                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+y)*n_a+x)*n_a+w)*n_a+t)*N_fit+i_f];
    
//     set_zero_matr(RF_PV_JK, n_a*n_a*n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_JK[(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+i_f]+=-RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+u)*N_fit+i_f]
                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+t)*N_fit+i_f]
                                                       +RF_P3_VV[(((((t*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+u)*N_fit+i_f]
                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+t)*N_fit+i_f]
                                                       -RF_P3_VV[(((((t*n_a+x)*n_a+v)*n_a+w)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((u*n_a+x)*n_a+v)*n_a+w)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+w)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((t*n_a+x)*n_a+w)*n_a+v)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((t*n_a+w)*n_a+x)*n_a+v)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((u*n_a+x)*n_a+w)*n_a+v)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+w)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((t*n_a+w)*n_a+x)*n_a+v)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*N_fit+i_f];
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_JK[(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+i_f]+=-RF_P3_VH[(((t*n_a+v)*n_a+w)*n_a+u)*N_fit+i_f]
                                                       -RF_P3_HV[(((v*n_a+w)*n_a+u)*n_a+t)*N_fit+i_f]
                                                       +RF_P3_VH[(((u*n_a+v)*n_a+w)*n_a+t)*N_fit+i_f]
                                                       +RF_P3_HV[(((v*n_a+w)*n_a+t)*n_a+u)*N_fit+i_f]
                                                       +RF_P3_VH[(((t*n_a+w)*n_a+v)*n_a+u)*N_fit+i_f]
                                                       +RF_P3_HV[(((w*n_a+v)*n_a+u)*n_a+t)*N_fit+i_f]
                                                       -RF_P3_VH[(((u*n_a+w)*n_a+v)*n_a+t)*N_fit+i_f]
                                                       -RF_P3_HV[(((w*n_a+v)*n_a+t)*n_a+u)*N_fit+i_f];

    
//     set_zero_matr(RF_P3_AB, n_a*n_a*n_a*n_a*n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_P3_AB[(((((t*n_a+u)*n_a+v)*n_a+w)*n_a+x)*n_a+y)*N_fit+i_f]+=-RF_P3_VV[(((((t*n_a+v)*n_a+y)*n_a+w)*n_a+u)*n_a+x)*N_fit+i_f]
                                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+y)*n_a+w)*n_a+t)*n_a+x)*N_fit+i_f]
                                                                       +RF_P3_VV[(((((t*n_a+w)*n_a+y)*n_a+v)*n_a+u)*n_a+x)*N_fit+i_f]
                                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+y)*n_a+v)*n_a+t)*n_a+x)*N_fit+i_f]
                                                                       -RF_P3_VV[(((((t*n_a+v)*n_a+w)*n_a+y)*n_a+x)*n_a+u)*N_fit+i_f]
                                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+w)*n_a+y)*n_a+x)*n_a+t)*N_fit+i_f]
                                                                       +RF_P3_VV[(((((t*n_a+w)*n_a+v)*n_a+y)*n_a+x)*n_a+u)*N_fit+i_f]
                                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+v)*n_a+y)*n_a+x)*n_a+t)*N_fit+i_f]
                                                                       -RF_P3_VV[(((((x*n_a+y)*n_a+v)*n_a+w)*n_a+u)*n_a+t)*N_fit+i_f]
                                                                       +RF_P3_VV[(((((x*n_a+y)*n_a+v)*n_a+w)*n_a+t)*n_a+u)*N_fit+i_f]
                                                                       +RF_P3_VV[(((((x*n_a+y)*n_a+w)*n_a+v)*n_a+u)*n_a+t)*N_fit+i_f]
                                                                       -RF_P3_VV[(((((x*n_a+y)*n_a+w)*n_a+v)*n_a+t)*n_a+u)*N_fit+i_f];

//     set_zero_matr(RF_PV_AB, n_a*n_a*n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=-RF_P3_VV[(((((t*n_a+x)*n_a+w)*n_a+v)*n_a+x)*n_a+u)*N_fit+i_f]
                                                       -RF_P3_VV[(((((t*n_a+x)*n_a+v)*n_a+w)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+t)*N_fit+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((u*n_a+x)*n_a+v)*n_a+w)*n_a+x)*n_a+t)*N_fit+i_f]
                                                       -RF_P3_VV[(((((u*n_a+x)*n_a+w)*n_a+v)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*N_fit+i_f]
                                                       -RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+u)*N_fit+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*N_fit+i_f];
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=-RF_P3_VH[(((u*n_a+w)*n_a+v)*n_a+t)*N_fit+i_f]
                                                       -RF_P3_HV[(((v*n_a+w)*n_a+u)*n_a+t)*N_fit+i_f]
                                                       -RF_P3_VH[(((t*n_a+v)*n_a+w)*n_a+u)*N_fit+i_f]
                                                       -RF_P3_HV[(((w*n_a+v)*n_a+t)*n_a+u)*N_fit+i_f];
    
//     set_zero_matr(RF_PH, n_a*n_a*N_fit);
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(t*n_a+u)*N_fit+i_f]+=-RF_P3_VV[(((((t*n_a+y)*n_a+v)*n_a+u)*n_a+v)*n_a+y)*N_fit+i_f]
                                    +RF_P3_VV[(((((t*n_a+v)*n_a+y)*n_a+u)*n_a+v)*n_a+y)*N_fit+i_f]*2;
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(t*n_a+u)*N_fit+i_f]+=-RF_P3_VH[(((t*n_a+y)*n_a+u)*n_a+y)*N_fit+i_f]
                                    +RF_P3_VH[(((t*n_a+u)*n_a+y)*n_a+y)*N_fit+i_f]*2
                                    -RF_P3_HV[(((y*n_a+u)*n_a+y)*n_a+t)*N_fit+i_f]
                                    +RF_P3_HV[(((y*n_a+u)*n_a+t)*n_a+y)*N_fit+i_f]*2;
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(t*n_a+u)*N_fit+i_f]+=-RF_P3_HH[(u*n_a+t)*N_fit+i_f];
    
//     set_zero_matr(RF_PS, N_fit);
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PS[i_f]+=RF_P3_HH[(t*n_a+t)*N_fit+i_f]*2;
    

    delete[] ED;
    delete[] CAAA  ;
    delete[] RF_P3_VV;
    delete[] RF_P3_VH;
    delete[] RF_P3_HV;
    delete[] RF_P3_HH;

    
    return 0;
}

int res_fit_data::calc_RF_2_CCVV(){
    
    double * R_fit;
    R_fit = new double[N_fit*num_threads];
    set_zero_matr(R_fit,N_fit*num_threads);
    double *V[num_threads];
    double *JK[num_threads];
    for(int i=0;i<num_threads;i++) V[i] = new double[n_c];
    for(int i=0;i<num_threads;i++)JK[i] = new double[n_c*n_c];
    
#ifdef _OPENBLAS
    openblas_set_num_threads(1);
#endif
    omp_set_num_threads(num_threads);
    for(int a=0; a<n_v; a++){//fprintf(stderr,"CCVV a=%d\r",a);
#pragma omp parallel for
    for(int b=a; b<n_v; b++){/*fprintf(stderr,"CCVV a,b=%d,%d\r",a,b);*/
        int th_id = omp_get_thread_num();
        double C;
        int n_i = n_c+n_a;
        int iajb=0;
        double dE;
        double Ep;
        double J;
        double K;
        
// #ifdef _RI
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_c,n_c,RI->aux_n_ao,1.0,
                        RI->VC_RI_M+a*n_c*RI->aux_n_ao,RI->aux_n_ao,
                        RI->VC_RI_M+b*n_c*RI->aux_n_ao,RI->aux_n_ao,0.0,
                        JK[th_id],n_c);
        
// #endif
//             
        
        for(int i=0; i<n_c; i++){//fprintf(stderr,"i,a=%d,%d\r",i,a);
        for(int j=i; j<n_c; j++){
// #ifndef _RI
//             iajb = (i*n_i*n_v+j)*n_v;
//             V[th_id][j]=2*(IEIE[iajb+a*n_i*n_v+b]*IEIE[iajb+a*n_i*n_v+b]+
//                     IEIE[iajb+b*n_i*n_v+a]*IEIE[iajb+b*n_i*n_v+a]);
//             if(a==b)V[th_id][j]=V[th_id][j]/2;
//             if(i==j)V[th_id][j]=V[th_id][j]/2;
//             C=IEIE[iajb+a*n_i*n_v+b]-IEIE[iajb+b*n_i*n_v+a];
//             V[th_id][j]+=2*C*C;
// #endif
// #ifdef _RI
            J=JK[th_id][i*n_c+j];
            K=JK[th_id][j*n_c+i];
            V[th_id][j]=2*(J*J+K*K);
            
            if(a==b)V[th_id][j]=V[th_id][j]/2;
            if(i==j)V[th_id][j]=V[th_id][j]/2;
            C=J-K;
            V[th_id][j]+=2*C*C;
// #endif
        }
        
        for(int i_f=0;i_f<N_fit;i_f++){
            Ep=e_c[i]-e_v[a]-e_v[b]+E_fit[i_f];
            for(int j=i; j<n_c; j++){

                dE=e_c[j]+Ep;
                R_fit[th_id*N_fit+i_f]+=V[th_id][j]*dE/(dE*dE+edshift);
                
            }
       }
    }
    }}
    
//     printf_timer("CCVV main cycle");
    
    for(int i_f=0;i_f<N_fit;i_f++)
    for(int i=0;i<num_threads;i++)
        RF_PS[i_f]+=R_fit[i*N_fit+i_f];
    
    delete[] R_fit;
    for(int i=0;i<num_threads;i++) delete[] V [i];
    for(int i=0;i<num_threads;i++) delete[] JK[i];
    
    
    return 0;
}

int res_fit_data::calc_RF_2_CAVV(){
    
    
    
    double **PH_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)PH_th[i]=new double[n_a*n_a*N_fit];
    for(int i=0;i<num_threads;i++)set_zero_matr(PH_th[i],n_a*n_a*N_fit);

#ifdef _OPENBLAS
    openblas_set_num_threads(1);
#endif
    omp_set_num_threads(num_threads);
    
    int n_i=n_c+n_a;
    #pragma omp parallel
    {
        int nt = omp_get_thread_num();
        
        double *K;
        double *J;
        K= new double[n_a*n_c];
        J= new double[n_a*n_c];
        
        double dE;
        double V2;
        
//         printf("CAVV is not parallel\n");
        
        for(int a=nt  ; a<n_v; a+=num_threads)
        for(int b=a  ; b<n_v; b++){//fprintf(stderr,"CAVVa,b=%d,%d\r",a,b);
// #ifndef _RI
//             for(int i=0  ; i<n_c; i++)
//             for(int t=0  ; t<n_a; t++){
//                 J[i*n_a+t]=IEIE[((i*n_v+a)*n_i+t+n_c)*n_v+b];
//                 K[i*n_a+t]=IEIE[((i*n_v+b)*n_i+t+n_c)*n_v+a];
//             }
// #endif
// #ifdef _RI
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_c,n_a,RI->aux_n_ao,1.0,
                            RI->VC_RI_M+a*n_c*RI->aux_n_ao,RI->aux_n_ao,
                            RI->VA_RI_M+b*n_a*RI->aux_n_ao,RI->aux_n_ao,0.0,
                            J,n_a);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_c,n_a,RI->aux_n_ao,1.0,
                            RI->VC_RI_M+b*n_c*RI->aux_n_ao,RI->aux_n_ao,
                            RI->VA_RI_M+a*n_a*RI->aux_n_ao,RI->aux_n_ao,0.0,
                            K,n_a);
// #endif
            for(int i=0;i<n_c;i++)
            for(int t=0;t<n_a;t++)
            for(int v=0;v<n_a;v++){
                V2 =(J[i*n_a+t]-K[i*n_a+t])*(J[i*n_a+v]-K[i*n_a+v]);
                V2+=J[i*n_a+t]*J[i*n_a+v];
                if(a!=b)
                    V2+=K[i*n_a+t]*K[i*n_a+v];
                for(int i_f=0;i_f<N_fit;i_f++){
                    dE=e_c[i]+e_a[t]-e_v[a]-e_v[b]+E_fit[i_f];
                    PH_th[nt][(t*n_a+v)*N_fit+i_f]+=V2*dE/(dE*dE+edshift);
                }
            }
        }
        
        delete[] K;
        delete[] J;
    }
    
    for(long j=0; j<num_threads;j++)
    #pragma omp parallel for
        for(long i=0; i<n_a*n_a*N_fit;i++)
            RF_PH[i]+=PH_th[j][i];
        
    for(int i=0;i<num_threads;i++)delete[] PH_th[i];
    delete[] PH_th;
    
    
    return 0;
    
}

int res_fit_data::calc_RF_2_AAVV(){
    
//     num_threads=1;
    double **JK_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)JK_th[i]=new double[n_a*n_a*n_a*n_a*N_fit];
    for(int i=0;i<num_threads;i++)set_zero_matr(JK_th[i],n_a*n_a*n_a*n_a*N_fit);

    double **AB_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)AB_th[i]=new double[n_a*n_a*n_a*n_a*N_fit];
    for(int i=0;i<num_threads;i++)set_zero_matr(AB_th[i],n_a*n_a*n_a*n_a*N_fit);

#ifdef _OPENBLAS
    openblas_set_num_threads(1);
#endif
    omp_set_num_threads(num_threads);


//     printf("%d %d %d\n", num_threads,omp_get_num_threads(), openblas_get_num_threads());
//     getchar();
    
    
    int n_i=n_c+n_a;
    #pragma omp parallel
    {
        int nt =omp_get_thread_num();
        
        double AB,JK,dE;
        
        double * Ja;
        
        Ja = new double[n_a*n_a];
        
        for(int a=nt; a<n_v; a+=num_threads)
        for(int b=a; b<n_v; b++){//fprintf(stderr,"AAVV a,b=%3d,%3d  (thread %2d)    \r",a,b,nt);
// #ifdef _RI
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_a,n_a,RI->aux_n_ao,1.0,
                            RI->VA_RI_M+a*n_a*RI->aux_n_ao,RI->aux_n_ao,
                            RI->VA_RI_M+b*n_a*RI->aux_n_ao,RI->aux_n_ao,0.0,
                            Ja,n_a);
// #endif        
            for(int t=0  ;t<n_a;t++)
            for(int u=0  ;u<n_a;u++)
            for(int v=0  ;v<n_a;v++)
            for(int w=0  ;w<n_a;w++){
// #ifndef _RI
//                 JK=(IEIE[(((t+n_c)*n_v+a)*n_i+u+n_c)*n_v+b]-
//                     IEIE[(((t+n_c)*n_v+b)*n_i+u+n_c)*n_v+a])*
//                    (IEIE[(((v+n_c)*n_v+a)*n_i+w+n_c)*n_v+b]-
//                     IEIE[(((v+n_c)*n_v+b)*n_i+w+n_c)*n_v+a]);
// #endif      
// #ifdef _RI  
                JK=(Ja[t*n_a+u]-Ja[u*n_a+t])*
                   (Ja[v*n_a+w]-Ja[w*n_a+v]);
// #endif
// #ifndef _RI
//                 AB         =IEIE[(((t+n_c)*n_v+a)*n_i+u+n_c)*n_v+b]*
//                             IEIE[(((v+n_c)*n_v+a)*n_i+w+n_c)*n_v+b];
//                 if(a!=b)AB+=IEIE[(((t+n_c)*n_v+b)*n_i+u+n_c)*n_v+a]*
//                             IEIE[(((v+n_c)*n_v+b)*n_i+w+n_c)*n_v+a];
// #endif      
// #ifdef _RI  
                AB         =Ja[t*n_a+u]*Ja[v*n_a+w];
                if(a!=b)AB+=Ja[u*n_a+t]*Ja[w*n_a+v];
// #endif            
                   
                for(int i_f=0;i_f<N_fit;i_f++){
                    dE=e_a[t]+e_a[u]-e_v[a]-e_v[b]+E_fit[i_f];
                    JK_th[nt][(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+i_f]+=JK*dE/(dE*dE+edshift);///inverse t-u in res_fit_calc_2body_AA (?)
                    AB_th[nt][(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=AB*dE/(dE*dE+edshift);
                }
                
            }
                
            
        }
    
       delete[] Ja;
    
    
    }
    
//     #pragma omp parallel for
    for(long j=0; j<num_threads;j++)
        for(long i=0; i<n_a*n_a*n_a*n_a*N_fit;i++)
            RF_PV_JK[i]+=JK_th[j][i];
//     #pragma omp parallel for
    for(long j=0; j<num_threads;j++)
        for(long i=0; i<n_a*n_a*n_a*n_a*N_fit;i++)
            RF_PV_AB[i]+=AB_th[j][i];
        
    for(int i=0;i<num_threads;i++)delete[] JK_th[i];
    delete[] JK_th;
    
    for(int i=0;i<num_threads;i++)delete[] AB_th[i];
    delete[] AB_th;
    return 0;
}

int res_fit_data::calc_RF_2_CCAV(){
    
    int n_i=n_c+n_a;
    
    
    double **RF_PH_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)RF_PH_th[i] = new double[n_a*n_a*N_fit];
    for(int i=0;i<num_threads;i++)set_zero_matr(RF_PH_th[i], n_a*n_a        *N_fit);
    
#ifdef _OPENBLAS
    openblas_set_num_threads(1);
#endif
    omp_set_num_threads(num_threads);
    
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
// #ifdef _RI        
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                J[a*n_a+t]=cblas_ddot(RI->aux_n_ao, RI->VC_RI_M+(a*n_c+i)*RI->aux_n_ao, 1, RI->CA_RI_M+(j*n_a+t)*RI->aux_n_ao, 1);
                K[a*n_a+t]=cblas_ddot(RI->aux_n_ao, RI->VC_RI_M+(a*n_c+j)*RI->aux_n_ao, 1, RI->CA_RI_M+(i*n_a+t)*RI->aux_n_ao, 1);
            }
// #endif
// #ifndef _RI
//             for(int a=0; a<n_v; a++)
//             for(int t=0  ; t<n_a; t++){
//                 J[a*n_a+t]=IIIE[((i*n_i+t+n_c)*n_i+j)*n_v+a];
//                 K[a*n_a+t]=IIIE[((j*n_i+t+n_c)*n_i+i)*n_v+a];
//             }
// #endif
        
            for(int a=0;a<n_v;a++)
            for(int t=0;t<n_a;t++)
            for(int v=0;v<n_a;v++){
                V2 =(J[a*n_a+t]-K[a*n_a+t])*(J[a*n_a+v]-K[a*n_a+v]);
                V2+=J[a*n_a+t]*J[a*n_a+v];
                if(i!=j)
                    V2+=K[a*n_a+t]*K[a*n_a+v];
                for(int i_f=0;i_f<N_fit;i_f++){
                    dE=e_c[i]+e_c[j]-e_a[t]-e_v[a]+E_fit[i_f];
                    RF_PH_th[nt][(v*n_a+t)*N_fit+i_f]-=V2*dE/(dE*dE+edshift);//sign changed!!!!
                }
            }
        }
        
        delete[] K;
        delete[] J;
    }
    
    for(long j=1; j<num_threads;j++)
//     #pragma omp parallel for
        for(long i=0; i<n_a*n_a*N_fit;i++)
            RF_PH_th[0][i]+=RF_PH_th[j][i];
    
//     #pragma omp parallel for
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PS[i_f]-=RF_PH_th[0][(t*n_a+t)*N_fit+i_f]*2;
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(w*n_a+t)*N_fit+i_f]+=RF_PH_th[0][(w*n_a+t)*N_fit+i_f];
    
    for(int i=0;i<num_threads;i++)delete[] RF_PH_th[i];
    delete[] RF_PH_th;
    
    
    return 0;
    
}

int res_fit_data::calc_RF_2_CCAA(){
    
//     num_threads=1;
    
    double ** RF_PV_AB_th = new double * [num_threads];
    double ** RF_PV_JK_th = new double * [num_threads];
    
    for(int i=0;i<num_threads;i++)RF_PV_AB_th[i] = new double[n_a*n_a*n_a*n_a*N_fit];
    for(int i=0;i<num_threads;i++)RF_PV_JK_th[i] = new double[n_a*n_a*n_a*n_a*N_fit];
    
    for(int i=0;i<num_threads;i++)set_zero_matr(RF_PV_AB_th[i], n_a*n_a*n_a*n_a*N_fit);
    for(int i=0;i<num_threads;i++)set_zero_matr(RF_PV_JK_th[i], n_a*n_a*n_a*n_a*N_fit);
    
    
    
#ifdef _OPENBLAS
    openblas_set_num_threads(1);
#endif
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        int nt =omp_get_thread_num();
    
        double dE;
        double Ed;
        
        double V2;
        double AB;
        
        double * Ja;
        Ja = new double[n_a*n_a];
        
        
        for(int i=nt; i<n_c; i+=num_threads)
        for(int j=i; j<n_c; j++){//fprintf(stderr,"CCAA i,j=%3d,%3d  (thread %2d)    \r",i,j,nt);
// #ifdef _RI
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                            n_a,n_a,RI->aux_n_ao,1.0,
                            RI->CA_RI_M+i*n_a*RI->aux_n_ao,RI->aux_n_ao,
                            RI->CA_RI_M+j*n_a*RI->aux_n_ao,RI->aux_n_ao,0.0,
                            Ja,n_a);
// #endif        
            for(int t=0  ;t<n_a;t++)
            for(int u=0  ;u<n_a;u++)
            for(int v=0  ;v<n_a;v++)
            for(int w=0  ;w<n_a;w++){
                
// #ifndef _RI
//                 V2=(ACAC[((t*n_c+i)*n_a+u)*n_c+j]-
//                     ACAC[((t*n_c+j)*n_a+u)*n_c+i])*
//                    (ACAC[((v*n_c+i)*n_a+w)*n_c+j]-
//                     ACAC[((v*n_c+j)*n_a+w)*n_c+i]);
// #endif      
// #ifdef _RI  
                V2=(Ja[t*n_a+u]-Ja[u*n_a+t])*
                   (Ja[v*n_a+w]-Ja[w*n_a+v]);
// #endif
// #ifndef _RI
//                 AB         =ACAC[((t*n_c+i)*n_a+u)*n_c+j]*
//                             ACAC[((v*n_c+i)*n_a+w)*n_c+j];
//                 if(i!=j)AB+=ACAC[((t*n_c+j)*n_a+u)*n_c+i]*
//                             ACAC[((v*n_c+j)*n_a+w)*n_c+i];
// #endif      
// #ifdef _RI  
                AB         =Ja[t*n_a+u]*Ja[v*n_a+w];
                if(i!=j)AB+=Ja[u*n_a+t]*Ja[w*n_a+v];
// #endif
                for(int i_f=0;i_f<N_fit;i_f++){
                    dE=-e_a[v]-e_a[w]+e_c[i]+e_c[j]+E_fit[i_f];
                    RF_PV_JK_th[nt][(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+i_f]+=V2*dE/(dE*dE+edshift);///inverse t-u in res_fit_calc_2body_AA (?)
                    RF_PV_AB_th[nt][(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=AB*dE/(dE*dE+edshift);
                }
            }
        }
        
        delete[] Ja;
    }
    
    for(long j=1; j<num_threads;j++)
//     #pragma omp parallel for
        for(long i=0; i<n_a*n_a*n_a*n_a*N_fit;i++)
            RF_PV_JK_th[0][i]+=RF_PV_JK_th[j][i];

    for(long j=1; j<num_threads;j++)
//     #pragma omp parallel for
        for(long i=0; i<n_a*n_a*n_a*n_a*N_fit;i++)
            RF_PV_AB_th[0][i]+=RF_PV_AB_th[j][i];

    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(long i=0;i<n_a*n_a*n_a*n_a*N_fit;i++)RF_PV_JK[i]+=RF_PV_JK_th[0][i];
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(long i=0;i<n_a*n_a*n_a*n_a*N_fit;i++)RF_PV_AB[i]+=RF_PV_AB_th[0][i];
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(v*n_a+u)*N_fit+i_f]+=RF_PV_JK_th[0][(((t*n_a+v)*n_a+t)*n_a+u)*N_fit+i_f]*0.25;
    
    #pragma omp parallel for // parallelism is bad but i don't care
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(v*n_a+t)*N_fit+i_f]-=RF_PV_JK_th[0][(((u*n_a+v)*n_a+t)*n_a+u)*N_fit+i_f]*0.25;

    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(w*n_a+u)*N_fit+i_f]-=RF_PV_JK_th[0][(((w*n_a+v)*n_a+v)*n_a+u)*N_fit+i_f]*0.25;

    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(w*n_a+t)*N_fit+i_f]+=RF_PV_JK_th[0][(((w*n_a+u)*n_a+t)*n_a+u)*N_fit+i_f]*0.25;

    #pragma omp parallel for // parallelism is bad but i don't care
    for(int w=0; w<n_a; w++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(w*n_a+u)*N_fit+i_f]-=RF_PV_AB_th[0][(((t*n_a+w)*n_a+t)*n_a+u)*N_fit+i_f];



    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PS[i_f]+=RF_PV_JK_th[0][(((u*n_a+v)*n_a+v)*n_a+u)*N_fit+i_f]*0.5;//A+B

    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PS[i_f]-=RF_PV_JK_th[0][(((w*n_a+v)*n_a+w)*n_a+v)*N_fit+i_f]*0.5;//A+B
    
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PS[i_f]+=RF_PV_AB_th[0][(((t*n_a+u)*n_a+t)*n_a+u)*N_fit+i_f];

    
    
    
    for(int i=0;i<num_threads;i++)delete[] RF_PV_JK_th[i];
    for(int i=0;i<num_threads;i++)delete[] RF_PV_AB_th[i];
    
    
    delete[] RF_PV_JK_th;
    delete[] RF_PV_AB_th;
    
    return 0;
}


int sym_JK(double *RF_PV_JK, double * RF_PV,int n_a, int N_fit){
    
    #pragma omp parallel for
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_JK[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=-RF_PV[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]
                                                       +RF_PV[(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+i_f]
                                                       +RF_PV[(((t*n_a+u)*n_a+w)*n_a+v)*N_fit+i_f]
                                                       -RF_PV[(((u*n_a+t)*n_a+w)*n_a+v)*N_fit+i_f];//to be checked
    
    }
    }
    
    return 0;
}

int sym_JK_CA(double *RF_PV_JK, double * RF_PV,int n_a, int N_fit){
    
    #pragma omp parallel for
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_JK[(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+i_f]+=+RF_PV[(((t*n_a+v)*n_a+w)*n_a+u)*N_fit+i_f]
                                                       -RF_PV[(((t*n_a+w)*n_a+v)*n_a+u)*N_fit+i_f]
                                                       -RF_PV[(((u*n_a+v)*n_a+w)*n_a+t)*N_fit+i_f]
                                                       +RF_PV[(((u*n_a+w)*n_a+v)*n_a+t)*N_fit+i_f];//to be checked
    
    }
    }
    
    return 0;
}


int sym_AB(double * RF_PV_AB, double * RF_PV, int n_a, int N_fit){
    
    #pragma omp parallel for
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+= RF_PV[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]
                                                       +RF_PV[(((u*n_a+t)*n_a+w)*n_a+v)*N_fit+i_f];//to be checked
    
    }
    }
    
    return 0;
}

int sym_AB_CA(double * RF_PV_AB, double * RF_PV, int n_a, int N_fit){
    
    #pragma omp parallel for
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
    for(int w=0; w<n_a; w++){
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=-RF_PV[(((u*n_a+v)*n_a+w)*n_a+t)*N_fit+i_f]
                                                       -RF_PV[(((t*n_a+w)*n_a+v)*n_a+u)*N_fit+i_f];//to be checked
    
    }
    }
    
    return 0;
}


int res_fit_data::calc_RF_1_AV(double * P_AV){
    
    if(RI==0){
        printf("calc_RF_1_AV not realized without RI\n");
        exit(0);
    }
    
    double * VACC_J;
    double * VACC_K;
    double * H1a;
    
    double * VAAA;
    VAAA = new double[n_v*n_a*n_a*n_a];
    
    RI->VAAA_calc(VAAA);
    
    H1a=H_AV;

    
    set_zero_matr(RF_PV, n_a*n_a*n_a*n_a*N_fit);
//     set_zero_matr(RF_PH, n_a*n_a        *N_fit);
    
// #pragma omp parallel
//     {
        double dE_orb;
        double V;
        double dE;
        
        double * ED;
        ED = new double[N_fit];
        
        
// #pragma omp parallel for
        for(int a=0; a<n_v; a++)
        for(int t=0; t<n_a; t++)
        for(int u=0; u<n_a; u++)
        for(int v=0; v<n_a; v++){
            dE_orb=e_a[t]+e_a[u]-e_v[a]-e_a[v];
            for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                ED[i_f]=dE/(dE*dE+edshift);
            }
            for(int w=0; w<n_a; w++){
                //<tu|va>*P[w,a];<tu|va>=(au|tv)
                V=VAAA[((a*n_a+u)*n_a+t)*n_a+v]*P_AV[w*n_v+a];
                for(int i_f=0; i_f<N_fit; i_f++)
                    RF_PV[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                
            }
        }
        
// #pragma omp parallel for
        for(int a=0; a<n_v; a++)
        for(int t=0; t<n_a; t++){
            dE_orb=e_a[t]-e_v[a];
            for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                ED[i_f]=dE/(dE*dE+edshift);
            }
            for(int w=0; w<n_a; w++){
                //H[t,a]*P[w,a]
                V=H1a[t*n_v+a]*P_AV[w*n_v+a];
                for(int i_f=0; i_f<N_fit; i_f++)
                    RF_PH[(t*n_a+w)*N_fit+i_f]+=V*ED[i_f];
                
            }
        }
        delete[] ED;
//     }
    
    sym_JK(RF_PV_JK, RF_PV,n_a, N_fit);
    
    sym_AB(RF_PV_AB, RF_PV,n_a, N_fit);
    
    delete[] VAAA  ;
    
    return 0;
}

int res_fit_data::calc_RF_1_CA(double * P_CA){
    
    if(RI==0){
        printf("calc_RF_1_CA not realized without RI\n");
        return 1;
    }
    
//     double V1;
    double dE;
    double V;
    
    double * ED;
    ED = new double[N_fit];
//     int i_CI=0;
    double * CAAA;
    double * H1a;
    
    CAAA = new double[n_c*n_a*n_a*n_a];
    RI->CAAA_calc(CAAA);
    
    
    H1a=H_CA;
    
    double * RF_PH_tmp = new double[n_a*n_a*N_fit];
    
    set_zero_matr(RF_PV    , n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_PH_tmp, n_a*n_a        *N_fit);

    
    double dE_orb;
    
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++){
        dE_orb=e_c[i]+e_a[t]-e_a[u]-e_a[v];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int w=0; w<n_a; w++){
            //<it|uv>*P[w,a];<it|uv>=(iu|tv)
            V=CAAA[((i*n_a+u)*n_a+t)*n_a+v]*P_CA[i*n_a+w];
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_PV[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+i_f]+=V*ED[i_f];
            
        }
    }
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE_orb=e_c[i]-e_a[t];
        for(int i_f=0; i_f<N_fit; i_f++){
            dE=dE_orb+E_fit[i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int w=0; w<n_a; w++){
            //H[t,a]*P[w,a]
            V=H1a[i*n_a+t]*P_CA[i*n_a+w];
            for(int i_f=0; i_f<N_fit; i_f++)
                RF_PH_tmp[(w*n_a+t)*N_fit+i_f]-=V*ED[i_f];
            
        }
    }
    delete[] ED;
    
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PS[i_f]-=RF_PH_tmp[(t*n_a+t)*N_fit+i_f]*2;
    
    for(int t=0; t<n_a; t++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(w*n_a+t)*N_fit+i_f]+=RF_PH_tmp[(w*n_a+t)*N_fit+i_f];
    
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int i_f=0; i_f<N_fit; i_f++)
        RF_PH[(t*n_a+w)*N_fit+i_f]+=2*RF_PV[(((t*n_a+v)*n_a+w)*n_a+v)*N_fit+i_f]
                                     -RF_PV[(((t*n_a+w)*n_a+v)*n_a+v)*N_fit+i_f];
    
    sym_JK_CA(RF_PV_JK, RF_PV,n_a, N_fit);
    
    sym_AB_CA(RF_PV_AB, RF_PV,n_a, N_fit);
    
    delete[] CAAA;
    delete[] RF_PH_tmp;
    
    return 0;
}

int res_fit_data::calc_RF_1_CV(double * P_CV){
    
    
    if(RI==0){
        printf("calc_RF_1_CA not realized without RI\n");
        return 1;
    }

    double V;
    double dE_orb;
    double dE;
    double * ED;
    ED = new double[N_fit];
    
    double * VCAA;
    double * CAVA;
    
    VCAA = new double[n_c*n_v*n_a*n_a];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_c*n_v,n_a*n_a,RI->aux_n_ao,1.0,
                        RI->VC_RI_M,RI->aux_n_ao,
                        RI->AA_RI_M,RI->aux_n_ao,0.0,
                        VCAA,n_a*n_a);
    CAVA = new double[n_a*n_c*n_a*n_v];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_a*n_c,n_a*n_v,RI->aux_n_ao,1.0,
                        RI->CA_RI_M,RI->aux_n_ao,
                        RI->VA_RI_M,RI->aux_n_ao,0.0,
                        CAVA,n_a*n_v);

    
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++)
    for(int t=0  ;t<n_a;t++)
    for(int u=0  ;u<n_a;u++){
        //(<it|au>-<iu|at>)*P[i,a]
        V=(2*VCAA[((a*n_c+i)*n_a+u)*n_a+t]-  //<it|au>=(ai|tu)
           CAVA[((i*n_a+u)*n_v+a)*n_a+t])* //<it|ua>=(iu|at)
           P_CV[i*n_v+a]; 
        dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
        for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                RF_PH[(t*n_a+u)*N_fit+i_f]+=V*dE/(dE*dE+edshift);
        }
    }
    /// PLACE2
    for(int i=0; i<n_c; i++)
    for(int a=0; a<n_v; a++){
        //H[i,a]*P[i,a]
        V=H_CV[i*n_v+a]*P_CV[i*n_v+a]*2; 
        dE_orb=e_c[i]-e_v[a];
        for(int i_f=0; i_f<N_fit; i_f++){
                dE=dE_orb+E_fit[i_f];
                RF_PS[i_f]+=V*dE/(dE*dE+edshift);
        }
    }
    
    
    
    delete[] ED;
    
    delete[] VCAA;
    delete[] CAVA;
    
    
    return 0;
}


int res_fit_data::P1_calc(double * P1, double * P_AV, double * P_CA, double * P_CV, int AV, int CV, int CA){
    
    
    int n_c;
    int n_a;
    int n_v;
    
    n_c=RI->n_cor;
    n_a=RI->n_act;
    n_v=RI->n_vac; 
    
    set_zero_matr(RF_PS      ,                 N_fit);
    set_zero_matr(RF_PH      , n_a*n_a        *N_fit);
    set_zero_matr(RF_PV_JK   , n_a*n_a*n_a*n_a*N_fit);
    set_zero_matr(RF_PV_AB   , n_a*n_a*n_a*n_a*N_fit);
    
    
    
    //calculation of RF_PV_AB, RF_PV_JK and RF_PH

    if(CA)calc_RF_1_CA(P_CA);
    if(CV)calc_RF_1_CV(P_CV);
    if(AV)calc_RF_1_AV(P_AV);
    
    
    
    
    printf_timer("calculation resolvent table");fflush(stdout);
    double **P_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)P_th[i]=new double[n_s*n_s];
    for(int i=0;i<num_threads;i++)set_zero_matr(P_th[i],n_s*n_s);
    
#pragma omp parallel
    {
        int nt = omp_get_thread_num();
        
        double * P_loc=new double[n_s*n_s];
        set_zero_matr(P_loc,n_s*n_s);
        
        if(AV+CA){
            res_fit_calc_2body_AB(P_loc, n_s, n_s,
                                  ci->coef[0],
                                  n_a,ci->na, ci->nb, ci->Na, ci->Nb, ci->fa, ci->fb, ci->vec_a, ci->vec_b,
                                  RF_PV_AB, L_fit, S_fit, C_fit, N_fit,
                                  nt, num_threads);
//             if(nt==0)printf_timer("\nABAB(by thread 0)");
            
            res_fit_calc_2body_AA(P_loc, n_s, n_s,
                                  ci->coef[0],
                                  n_a,ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a,
                                  RF_PV_JK, L_fit, S_fit, C_fit, N_fit,
                                  nt, num_threads);
//             if(nt==0)printf_timer("\nAAAA(by thread 0)");
            
            res_fit_calc_2body_AA(P_loc, n_s, n_s,
                                  ci->coef_bas[0],
                                  n_a,ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b,
                                  RF_PV_JK, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                                  nt, num_threads);
//             if(nt==0)printf_timer("\nBBBB(by thread 0)");
        }
        if(CV+CA+AV){//if is just for formating, normal (CV+CA+AV)!=0
            res_fit_calc_1body(P_loc, n_s, n_s,
                              ci->coef[0],
                              n_a,ci->na, ci->Na, ci->Nb, ci->fa, ci->vec_a,
                              RF_PH, L_fit, S_fit, C_fit, N_fit,
                              nt, num_threads);
//             if(nt==0)printf_timer("\nAA(by thread 0)");
            
            res_fit_calc_1body(P_loc, n_s, n_s,
                               ci->coef_bas[0],
                               n_a,ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b,
                               RF_PH, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                               nt, num_threads);
//             if(nt==0)printf_timer("\nBB(by thread 0)");
        }
        if(CV+CA){
            res_fit_calc_0body(P_loc, n_s, n_s,
                               ci->coef[0],//was written _tr
                               ci->Na, ci->Nb,
                               RF_PS, L_fit, S_fit, C_fit, N_fit,
                               nt, num_threads);
//             if(nt==0)printf_timer("\nS(by thread 0)");
        }
        for(int i_s=0;i_s<n_s*n_s;i_s++)
            P_th[nt][i_s]=P_loc[i_s];
        delete[] P_loc;
    }
    
    char* timer_words = new char[256];
    if(CV==1)if(CA==1)if(AV==1)sprintf(timer_words, "calculation of first order properties\0");
    if(CV==0)if(CA==1)if(AV==1)sprintf(timer_words, "calculation of CA and AV parts of first order properties\0");
    if(CV==1)if(CA==0)if(AV==1)sprintf(timer_words, "calculation of CV and AV parts of first order properties\0");
    if(CV==1)if(CA==1)if(AV==0)sprintf(timer_words, "calculation of CV and CA parts of first order properties\0");
    if(CV==1)if(CA==0)if(AV==0)sprintf(timer_words, "calculation of CV part of first order properties\0");
    if(CV==0)if(CA==1)if(AV==0)sprintf(timer_words, "calculation of CA part of first order properties\0");
    if(CV==0)if(CA==0)if(AV==1)sprintf(timer_words, "calculation of AV part of first order properties\0");
    if(CV==0)if(CA==0)if(AV==0)sprintf(timer_words, "calculation of first order properties\0");
    
    
    
    
    printf_timer(timer_words);fflush(stdout);
    delete[] timer_words;
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<num_threads;i++)
        P1[i_s]+=P_th[i][i_s];
    for(int i=0;i<num_threads;i++)delete[] P_th[i];
    delete[] P_th;
    
    
    return 0;
    
}


res_fit_data::~res_fit_data(){
    
    //grid data
    delete[] E_fit   ;
    delete[] S_fit   ;
    delete[] L_fit   ;
    delete[] C_fit   ;
    delete[] S_fit_tr;
    delete[] L_fit_tr;
    delete[] C_fit_tr; 
    
    // RF operator tables
    delete[] RF_PV   ;
    delete[] RF_PH   ;

    delete[] RF_PS;
    delete[] RF_PV_JK;
    delete[] RF_PV_AB;
    
    delete[] RF_P3_JK;
    delete[] RF_P3_AB;
    
}
