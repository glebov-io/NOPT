#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
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
    RF_P3_JK = nullptr;
    RF_P3_AB = nullptr;
    if(full_tables){
        RF_P3_JK = new double  [(size_t)n_a*n_a*n_a*n_a*n_a*n_a*N_fit];
        RF_P3_AB = new double  [(size_t)n_a*n_a*n_a*n_a*n_a*n_a*N_fit];
    }
    win_f0 = 0; win_nf = N_fit;
    // E_act[i,j] = Ea(i) + Eb(j): the stencil start S[i,j,s] grows as Eb(j) (Ea(i)) falls, the same for every i (j) and state
    {
        const int Na = ci->Na, Nb = ci->Nb;
        const double * Ea = ci->E_act[0];
        order_b.resize(Nb); order_a.resize(Na);
        for(int j=0;j<Nb;j++) order_b[j]=j;
        for(int i=0;i<Na;i++) order_a[i]=i;
        std::stable_sort(order_b.begin(), order_b.end(), [&](int x, int y){ return Ea[x] > Ea[y]; });            // Eb(j) = E_act[0,j] - E_act[0,0]
        std::stable_sort(order_a.begin(), order_a.end(), [&](int x, int y){ return Ea[(size_t)x*Nb] > Ea[(size_t)y*Nb]; });   // Ea(i)
    }

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
                           int i_th, int n_th, int f0, int nw, const int * order){
    // RF holds the grid points f0 .. f0 + nw - 1 (stride nw); only the part of each interpolation stencil inside them is added. order: the
    // strings of the inner index sorted so that the stencil start S grows along them, the same for every outer string and state.
    
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
    
    std::vector<int> klo(n_s), khi(n_s);
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"AV(2-body-sym) i_CI=%d  (thread %d)        \r",i_CI, i_th);
        // the inner strings whose stencil can touch the window: S in [f0 - (2 n_fit_pol + 1), f0 + nw), one string of slack for rounding
        bool any=false;
        for(int i_s=0;i_s<n_s;i_s++){
            auto Sk=[&](int k){ return S_fit[((long)i_CI*Nb+order[k])*n_s+i_s]; };
            int lo=0, hi=Nb;
            while(lo<hi){ int m=(lo+hi)/2; if(Sk(m) < f0-(2*n_fit_pol+1)) lo=m+1; else hi=m; }
            int lo2=lo, hi2=Nb;
            while(lo2<hi2){ int m=(lo2+hi2)/2; if(Sk(m) < f0+nw) lo2=m+1; else hi2=m; }
            klo[i_s]=lo>0?lo-1:0; khi[i_s]=lo2<Nb?lo2+1:Nb;
            if(klo[i_s]<khi[i_s]) any=true;
        }
        if(!any) continue;
        
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
                                            R = RF+(long)(((((t*N+u)*N+v)*N+y)*N+x)*N+w)*nw - f0;   // R[f] for f in the window
                                            for(int i_s=0;i_s<n_s;i_s++)
                                            for(int k=klo[i_s]; k<khi[i_s]; k++){
                                                const int j_CI = order[k];
                                                L = L_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                S = S_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                const int k0 = S<f0 ? f0-S : 0, k1 = S+L>f0+nw ? f0+nw-S : L;
                                                if(k0>=k1) continue;
                                                C = C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s);
                                                //<tu|wa>*P[a,v]-<tu|va>*P[a,w] to be checked
                                                Rc =0;
                                                for(int q=k0;q<k1;q++) Rc+=R[S+q]*C[q];
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
                          int i_th, int n_th, int f0, int nw){
    // RF holds the grid points f0 .. f0 + nw - 1 (stride nw): a determinant none of whose stencils touches them is skipped
    
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
        {
            bool any=false;
            for(int i_s=0;i_s<n_s&&!any;i_s++){
                const int Sd=S_fit[(i_CI*Nb+j_CI)*n_s+i_s], Ld=L_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                any = Sd<f0+nw && Sd+Ld>f0;
            }
            if(!any) continue;
        }
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
                                                
                                                R = RF+(long)(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*nw - f0;   // R[f] for f in the window
                                                for(int i_s=0;i_s<n_s;i_s++){
                                                    L = L_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                    S = S_fit[(i_CI*Nb+j_CI)*n_s+i_s];
                                                    const int k0 = S<f0 ? f0-S : 0, k1 = S+L>f0+nw ? f0+nw-S : L;
                                                    if(k0>=k1) continue;
                                                    C = C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s);
                                                    Rc =0;
                                                    for(int q=k0;q<k1;q++) Rc+=R[S+q]*C[q];
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
                           i_th, n_th, win_f0, win_nf, order_b.data());
    
    return 0;
    
}
    
    
int res_fit_data::calc_3body_BBB(double * E2, int i_th, int n_th){

    res_fit_calc_3body_AAA(E2, n_s, n_s,
                           ci->coef_bas[0],
                           n_a,ci->nb, ci->Nb, ci->Na, ci->fb, ci->vec_b,
                           RF_P3_JK, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                           i_th, n_th, win_f0, win_nf, order_a.data());
    return 0;
    
}


int res_fit_data::calc_3body_AAB(double * E2, int i_th, int n_th){
    
    res_fit_calc_3body_AB(E2, n_s, n_s,
                          ci->coef[0],
                          n_a,ci->na, ci->nb, ci->Na, ci->Nb, ci->fa, ci->fb, ci->vec_a, ci->vec_b,
                          RF_P3_AB, L_fit, S_fit, C_fit, N_fit,
                          i_th, n_th, win_f0, win_nf);    
    
    return 0;
    
}

int res_fit_data::calc_3body_ABB(double * E2, int i_th, int n_th){
    
    res_fit_calc_3body_AB(E2, n_s, n_s,
                          ci->coef_bas[0],
                          n_a,ci->nb, ci->na, ci->Nb, ci->Na, ci->fb, ci->fa, ci->vec_b, ci->vec_a,
                          RF_P3_AB, L_fit_tr, S_fit_tr, C_fit_tr, N_fit,
                          i_th, n_th, win_f0, win_nf);    
    
    return 0;
    
}


int res_fit_data::E2_windows(double ** E2_th, bool av, bool ca){
    // AV and CA build the three-body tables (and add to the four-index ones); the three-body contraction reads them. All of it is pointwise in the
    // grid, so it runs window by window: only the window of the two three-body tables and the AV intermediates (three more of the same size)
    // exist at a time. With full_tables a single window covers the grid and the tables stay.
    const size_t n6 = (size_t)n_a*n_a*n_a*n_a*n_a*n_a;
    int W = N_fit;
    if(!full_tables){
        double mb = 4096.0;
        if(getenv("NOPT_XMC_WINDOW_MB") && *getenv("NOPT_XMC_WINDOW_MB")) mb = atof(getenv("NOPT_XMC_WINDOW_MB"));
        const double per_point = 5.0 * 8.0 * (double)n6;
        W = (int)(mb * 1048576.0 / per_point);
        if(W < 1) W = 1;
        if(W > N_fit) W = N_fit;
    }
    std::vector<double> wJK, wAB;
    double t_av = 0, t_ca = 0, t_3 = 0;
    int n_win = 0;
    for(int f0 = 0; f0 < N_fit; f0 += W){
        win_f0 = f0; win_nf = (N_fit - f0 < W) ? N_fit - f0 : W;
        if(full_tables){
            set_zero_matr(RF_P3_JK, n6*N_fit);
            set_zero_matr(RF_P3_AB, n6*N_fit);
        }
        else{
            wJK.assign(n6*win_nf, 0.0); wAB.assign(n6*win_nf, 0.0);
            RF_P3_JK = wJK.data(); RF_P3_AB = wAB.data();
        }
        double t0 = omp_get_wtime();
        if(av) calc_RF_2_AV();
        t_av += omp_get_wtime() - t0; t0 = omp_get_wtime();
        if(ca) calc_RF_2_CA();
        t_ca += omp_get_wtime() - t0; t0 = omp_get_wtime();
        #pragma omp parallel
        {
            const int nt = omp_get_thread_num();
            std::vector<double> E2_loc((size_t)n_s*n_s, 0.0);
            calc_3body_AAA(E2_loc.data(), nt, num_threads);
            calc_3body_AAB(E2_loc.data(), nt, num_threads);
            calc_3body_ABB(E2_loc.data(), nt, num_threads);
            calc_3body_BBB(E2_loc.data(), nt, num_threads);
            for(int i = 0; i < n_s*n_s; i++) E2_th[nt][i] += E2_loc[i];
        }
        t_3 += omp_get_wtime() - t0;
        n_win++;
    }
    if(!full_tables){ RF_P3_JK = nullptr; RF_P3_AB = nullptr; }
    win_f0 = 0; win_nf = N_fit;
    fprintf(out_stream, "AV and CA res tables and the 3-body PT operators in %d window(s) of up to %d grid points (%.0f MB per window): "
                        "AV %.2f s, CA %.2f s, 3-body %.2f s\n", n_win, W, 5.0*8.0*(double)n6*W/1048576.0, t_av, t_ca, t_3);
    printf_timer("AV, CA res tables and 3-body PT operators");printf("\n");fflush(stdout);
    return 0;
}

int res_fit_data::E2_calc(double * E2){
    
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
    
    double **E2_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)E2_th[i]=new double[n_s*n_s];
    for(int i=0;i<num_threads;i++)set_zero_matr(E2_th[i],n_s*n_s);

    // AV, CA (they complete the four-index tables too) and the three-body contraction, by windows of the grid
    E2_windows(E2_th, use("AV"), use("CA"));

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
                
        for(int i_s=0;i_s<n_s*n_s;i_s++)
            E2_th[nt][i_s]+=E2_loc[i_s];
        delete[] E2_loc;
    }
    
    printf_timer("0-2 body PT operators");printf("\n");fflush(stdout);
    
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

    // For every output row (t,u), sums over the pairs (i,a) as matrix products, the denominator D(i,a,t,u; f) = dE/(dE^2 + s) with
    // dE = e_i + e_t - e_a - e_u + E_fit[f] folded into the left factor:
    //   RF_X[t,u,v,w,f] += sum_ia R_X[ia,(v,w)] L_X[ia,(t,u)] D(i,a,t,u;f)
    // with L_A = (iu|at), L_M = (ai|tu) - (iu|at), L_J = (ai|tu) and R_A = (iv|aw), R_M = (ai|wv) - (iv|aw), R_J = (ai|wv):
    //   AB = A.A, MM = M.M, MJ = M.J, JM = J.M, JJ = J.J.
    // The H terms: RF_MH, RF_JH[t,u,f] += sum_ia H_ia L_{M,J}[ia,(t,u)] D(i,a,t,u;f) in the same rows; RF_HM, RF_HJ (denominator e_i - e_a) as one product.
    {
        const int n2 = n_a*n_a, nia = n_c*n_v;
        std::vector<double> LA((size_t)nia*n2), LM((size_t)nia*n2), LJ((size_t)nia*n2), RA((size_t)nia*n2), RM((size_t)nia*n2), RJ((size_t)nia*n2);
        for(int i=0; i<n_c; i++) for(int a=0; a<n_v; a++){
            const size_t r=(size_t)(i*n_v+a)*n2;
            for(int t=0; t<n_a; t++) for(int u=0; u<n_a; u++){
                // left factor at (t,u)
                LA[r+t*n_a+u]=CAVA[((i*n_a+u)*n_v+a)*n_a+t];
                LJ[r+t*n_a+u]=VCAA[((a*n_c+i)*n_a+u)*n_a+t];
                LM[r+t*n_a+u]=VCAA[((a*n_c+i)*n_a+u)*n_a+t]-CAVA[((i*n_a+u)*n_v+a)*n_a+t];
                // right factor at (v,w) = (t,u)
                RA[r+t*n_a+u]=CAVA[((i*n_a+t)*n_v+a)*n_a+u];
                RJ[r+t*n_a+u]=VCAA[((a*n_c+i)*n_a+u)*n_a+t];
                RM[r+t*n_a+u]=VCAA[((a*n_c+i)*n_a+u)*n_a+t]-CAVA[((i*n_a+t)*n_v+a)*n_a+u];
            }
        }
        const int K = 1024;   // pairs (i,a) per product
        set_blas_non_par();
#pragma omp parallel
        {
            std::vector<double> GA((size_t)K*N_fit), GM((size_t)K*N_fit), GJ((size_t)K*N_fit);
#pragma omp for schedule(dynamic)
            for(int tu=0; tu<n2; tu++){
                const int t=tu/n_a, u=tu%n_a;
                const size_t ro=(size_t)tu*n2*N_fit;
                for(int k0=0; k0<nia; k0+=K){
                    const int nk=(nia-k0<K)?nia-k0:K;
                    for(int k=0; k<nk; k++){
                        const int ia=k0+k, i=ia/n_v, a=ia%n_v;
                        const double dE_orb=e_c[i]+e_a[t]-e_v[a]-e_a[u];
                        const double la=LA[(size_t)ia*n2+tu], lm=LM[(size_t)ia*n2+tu], lj=LJ[(size_t)ia*n2+tu];
                        for(int i_f=0; i_f<N_fit; i_f++){
                            const double dE=dE_orb+E_fit[i_f];
                            const double d=dE/(dE*dE+edshift);
                            GA[(size_t)k*N_fit+i_f]=la*d; GM[(size_t)k*N_fit+i_f]=lm*d; GJ[(size_t)k*N_fit+i_f]=lj*d;
                        }
                    }
                    const double * rA=RA.data()+(size_t)k0*n2, * rM=RM.data()+(size_t)k0*n2, * rJ=RJ.data()+(size_t)k0*n2;
                    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,nk,1.0,rA,n2,GA.data(),N_fit,1.0,RF_AB+ro,N_fit);
                    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,nk,1.0,rM,n2,GM.data(),N_fit,1.0,RF_MM+ro,N_fit);
                    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,nk,1.0,rJ,n2,GM.data(),N_fit,1.0,RF_MJ+ro,N_fit);
                    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,nk,1.0,rM,n2,GJ.data(),N_fit,1.0,RF_JM+ro,N_fit);
                    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,nk,1.0,rJ,n2,GJ.data(),N_fit,1.0,RF_JJ+ro,N_fit);
                    // RF_MH, RF_JH[t,u,f] += sum_ia H_ia G_{M,J}[ia,f]
                    cblas_dgemv(CblasRowMajor,CblasTrans,nk,N_fit,1.0,GM.data(),N_fit,H_CV+k0,1,1.0,RF_MH+(size_t)tu*N_fit,1);
                    cblas_dgemv(CblasRowMajor,CblasTrans,nk,N_fit,1.0,GJ.data(),N_fit,H_CV+k0,1,1.0,RF_JH+(size_t)tu*N_fit,1);
                }
            }
        }
        set_blas_par();
        // RF_HM, RF_HJ[t,u,f] += sum_ia L_{M,J}[ia,(t,u)] H_ia D0(i,a;f),  D0 with dE = e_i - e_a + E_fit[f]
        std::vector<double> HD((size_t)nia*N_fit);
        for(int ia=0; ia<nia; ia++){
            const int i=ia/n_v, a=ia%n_v;
            for(int i_f=0; i_f<N_fit; i_f++){
                const double dE=e_c[i]-e_v[a]+E_fit[i_f];
                HD[(size_t)ia*N_fit+i_f]=H_CV[ia]*dE/(dE*dE+edshift);
            }
        }
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,nia,1.0,LM.data(),n2,HD.data(),N_fit,1.0,RF_HM,N_fit);
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,nia,1.0,LJ.data(),n2,HD.data(),N_fit,1.0,RF_HJ,N_fit);
    }
    
    double dE_orb;
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
    // the grid points win_f0 .. win_f0 + win_nf - 1: the window tables and intermediates have stride Nw, the full PV/PH/PS tables Nw
    const int Nw = win_nf, f0 = win_f0;
    
    
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
    
    RF_P3_JJ = new double[n_a*n_a*n_a*n_a*n_a*n_a*Nw];
    RF_P3_JM = new double[n_a*n_a*n_a*n_a*n_a*n_a*Nw];
    RF_P3_MJ = new double[n_a*n_a*n_a*n_a*n_a*n_a*Nw];
    
    RF_P3_HM = new double[n_a*n_a*n_a*n_a*Nw];
    RF_P3_MH = new double[n_a*n_a*n_a*n_a*Nw];
    RF_P3_HJ = new double[n_a*n_a*n_a*n_a*Nw];
    RF_P3_JH = new double[n_a*n_a*n_a*n_a*Nw];
    
    set_zero_matr(RF_P3_JJ, n_a*n_a*n_a*n_a*n_a*n_a*Nw);
    set_zero_matr(RF_P3_JM, n_a*n_a*n_a*n_a*n_a*n_a*Nw);
    set_zero_matr(RF_P3_MJ, n_a*n_a*n_a*n_a*n_a*n_a*Nw);
    set_zero_matr(RF_P3_HM, n_a*n_a*n_a*n_a*Nw);
    set_zero_matr(RF_P3_MH, n_a*n_a*n_a*n_a*Nw);
    set_zero_matr(RF_P3_HJ, n_a*n_a*n_a*n_a*Nw);
    set_zero_matr(RF_P3_JH, n_a*n_a*n_a*n_a*Nw);
    
    
    // The four six-index tables, for every (t,u,v) a matrix product over the virtual index a:
    //   RF_X[t,u,v | y,x,w | f] += sum_a B_X[(y,x,w), a] G_X[a, f],   G_X[a,f] = A_X(a,t,u,v) ED(a,t,u,v; f)
    // with the antisymmetrized (M) or plain (J) integrals on either side: JK = M.M, JJ = J.J, MJ = M.J, JM = J.M (first factor: t,u,v).
    {
        const size_t n3 = (size_t)n_a*n_a*n_a;
        std::vector<double> BM(n3*n_v), BJ(n3*n_v);   // [(y,x,w), a]
        for(int y=0; y<n_a; y++) for(int x=0; x<n_a; x++) for(int w=0; w<n_a; w++) for(int a=0; a<n_v; a++){
            const size_t r = ((size_t)(y*n_a+x)*n_a+w)*n_v+a;
            BJ[r] = VAAA[((a*n_a+x)*n_a+w)*n_a+y];
            BM[r] = VAAA[((a*n_a+x)*n_a+w)*n_a+y]-VAAA[((a*n_a+y)*n_a+w)*n_a+x];
        }
        set_blas_non_par();
#pragma omp parallel
        {
            std::vector<double> GM((size_t)n_v*Nw), GJ((size_t)n_v*Nw);
#pragma omp for schedule(dynamic)
            for(long tuv=0; tuv<(long)n3; tuv++){
                const int t=tuv/(n_a*n_a), u=(tuv/n_a)%n_a, v=tuv%n_a;
                for(int a=0; a<n_v; a++){
                    const double dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                    const double Am=VAAA[((a*n_a+t)*n_a+v)*n_a+u]-VAAA[((a*n_a+u)*n_a+v)*n_a+t];
                    const double Aj=VAAA[((a*n_a+u)*n_a+v)*n_a+t];
                    for(int i_f=0; i_f<Nw; i_f++){
                        const double dE=dE_orb+E_fit[f0+i_f];
                        const double ed=dE/(dE*dE+edshift);
                        GM[(size_t)a*Nw+i_f]=Am*ed;
                        GJ[(size_t)a*Nw+i_f]=Aj*ed;
                    }
                }
                const size_t off=(size_t)tuv*n3*Nw;
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n3,Nw,n_v,1.0,BM.data(),n_v,GM.data(),Nw,1.0,RF_P3_JK+off,Nw);
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n3,Nw,n_v,1.0,BJ.data(),n_v,GJ.data(),Nw,1.0,RF_P3_JJ+off,Nw);
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n3,Nw,n_v,1.0,BJ.data(),n_v,GM.data(),Nw,1.0,RF_P3_MJ+off,Nw);
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n3,Nw,n_v,1.0,BM.data(),n_v,GJ.data(),Nw,1.0,RF_P3_JM+off,Nw);
            }
        }
        set_blas_par();
    }

#pragma omp parallel
    {
        double dE_orb;
        double V;
        double dE;
        double H1a;
        double H2a;
        
        double * ED;
        ED = new double[Nw];
        
        int u,v;
        int th_id = omp_get_thread_num();
        
        //H
        for(int a=0; a<n_v; a++)
        for(int t=th_id; t<n_a; t+=num_threads){
            dE_orb=-e_v[a]+e_a[t];
            for(int i_f=0; i_f<Nw; i_f++){
                dE=dE_orb+E_fit[f0+i_f];
                ED[i_f]=dE/(dE*dE+edshift);
            }
            for(int w=0; w<n_a; w++){
                H1a=H_AV[t*n_v+a];
                H2a=H_AV[w*n_v+a];                          
                V=H1a*H2a;
                for(int i_f=0; i_f<Nw; i_f++)
                    RF_PH[(t*n_a+w)*N_fit+f0+i_f]+=V*ED[i_f];
                
            }
        }
        //HM
        for(int a=0; a<n_v; a++)
        for(int t=th_id; t<n_a; t+=num_threads){
            dE_orb=-e_v[a]+e_a[t];
            for(int i_f=0; i_f<Nw; i_f++){
                dE=dE_orb+E_fit[f0+i_f];
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
                
                for(int i_f=0; i_f<Nw; i_f++)
                    RF_P3_HM[(((t*n_a+y)*n_a+x)*n_a+w)*Nw+i_f]+=V*ED[i_f];
                
            }
        }
        //MH
        for(int vu=th_id; vu<n_a*n_a; vu+=num_threads){
            v=vu/n_a;
            u=vu%n_a;
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<Nw; i_f++){
                    dE=dE_orb+E_fit[f0+i_f];
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
                    
                    for(int i_f=0; i_f<Nw; i_f++)
                        RF_P3_MH[(((t*n_a+u)*n_a+v)*n_a+w)*Nw+i_f]+=V*ED[i_f];
                    
                }
            }
        }
        //HJ
        for(int a=0; a<n_v; a++)
        for(int t=th_id; t<n_a; t+=num_threads){
            dE_orb=-e_v[a]+e_a[t];
            for(int i_f=0; i_f<Nw; i_f++){
                dE=dE_orb+E_fit[f0+i_f];
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
                
                for(int i_f=0; i_f<Nw; i_f++)
                    RF_P3_HJ[(((t*n_a+y)*n_a+x)*n_a+w)*Nw+i_f]+=V*ED[i_f];
                
            }
        }
        //JH
        for(int vu=th_id; vu<n_a*n_a; vu+=num_threads){
            v=vu/n_a;
            u=vu%n_a;
            for(int a=0; a<n_v; a++)
            for(int t=0; t<n_a; t++){
                dE_orb=-e_v[a]+e_a[t]+e_a[u]-e_a[v];
                for(int i_f=0; i_f<Nw; i_f++){
                    dE=dE_orb+E_fit[f0+i_f];
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
                    for(int i_f=0; i_f<Nw; i_f++)
                        RF_P3_JH[(((t*n_a+u)*n_a+v)*n_a+w)*Nw+i_f]+=V*ED[i_f];
                    
                }
            }
        }
        delete[] ED;
        
    }
    //RF_PV_JK
#pragma omp parallel for collapse(2)
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PV_JK[(((w*n_a+t)*n_a+v)*n_a+u)*N_fit+f0+i_f]+= RF_P3_HM[(((t*n_a+u)*n_a+v)*n_a+w)*Nw+i_f]
                                                       -RF_P3_HM[(((w*n_a+u)*n_a+v)*n_a+t)*Nw+i_f]
                                                       -RF_P3_MH[(((t*n_a+w)*n_a+v)*n_a+u)*Nw+i_f]
                                                       +RF_P3_MH[(((t*n_a+w)*n_a+u)*n_a+v)*Nw+i_f];
    //RF_PV_AB
#pragma omp parallel for collapse(2)
    for(int w=0; w<n_a; w++)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<Nw; i_f++){
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+f0+i_f]+= RF_P3_HJ[(((t*n_a+w)*n_a+v)*n_a+u)*Nw+i_f]
                                                       +RF_P3_JH[(((u*n_a+t)*n_a+w)*n_a+v)*Nw+i_f]
                                                       +RF_P3_HJ[(((u*n_a+v)*n_a+w)*n_a+t)*Nw+i_f]
                                                       +RF_P3_JH[(((t*n_a+u)*n_a+v)*n_a+w)*Nw+i_f];
    
        for(int x=0; x<n_a; x++)
            RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+f0+i_f] += RF_P3_JJ[(((((t*n_a+u)*n_a+x)*n_a+v)*n_a+w)*n_a+x)*Nw+i_f]
                                                            +RF_P3_JJ[(((((u*n_a+t)*n_a+x)*n_a+w)*n_a+v)*n_a+x)*Nw+i_f];
    }
    //RF_P3_AB
#pragma omp parallel for collapse(2)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int t=0; t<n_a; t++)
    for(int y=0; y<n_a; y++)
    for(int x=0; x<n_a; x++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_P3_AB[(((((t*n_a+u)*n_a+v)*n_a+w)*n_a+x)*n_a+y)*Nw+i_f]+=-RF_P3_MJ[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+w)*n_a+x)*Nw+i_f]
                                                                       +RF_P3_MJ[(((((t*n_a+u)*n_a+w)*n_a+y)*n_a+v)*n_a+x)*Nw+i_f]
                                                                       +RF_P3_JM[(((((x*n_a+t)*n_a+y)*n_a+w)*n_a+v)*n_a+u)*Nw+i_f]
                                                                       -RF_P3_JM[(((((x*n_a+u)*n_a+y)*n_a+w)*n_a+v)*n_a+t)*Nw+i_f]
                                                                       +RF_P3_JJ[(((((t*n_a+x)*n_a+v)*n_a+w)*n_a+y)*n_a+u)*Nw+i_f]
                                                                       -RF_P3_JJ[(((((t*n_a+x)*n_a+w)*n_a+v)*n_a+y)*n_a+u)*Nw+i_f]
                                                                       -RF_P3_JJ[(((((u*n_a+x)*n_a+v)*n_a+w)*n_a+y)*n_a+t)*Nw+i_f]
                                                                       +RF_P3_JJ[(((((u*n_a+x)*n_a+w)*n_a+v)*n_a+y)*n_a+t)*Nw+i_f];
                                                
    
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
    // the grid points win_f0 .. win_f0 + win_nf - 1: the window tables and intermediates have stride Nw, the full PV/PH/PS tables Nw
    const int Nw = win_nf, f0 = win_f0;
    
    double dE;
    double dE_orb;
           
    double H1a;
    double H2a;
    
    double V;
    
    double * ED;
    ED = new double[Nw];
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
    
    RF_P3_VV = new double[n_a*n_a*n_a*n_a*n_a*n_a*Nw];
    RF_P3_VH = new double[n_a*n_a*n_a*n_a*        Nw];
    RF_P3_HV = new double[n_a*n_a*n_a*n_a*        Nw];
    RF_P3_HH = new double[n_a*n_a*                Nw];
    
    set_zero_matr(RF_P3_VV, n_a*n_a*n_a*n_a*n_a*n_a*Nw);
    set_zero_matr(RF_P3_VH, n_a*n_a*n_a*n_a*        Nw);
    set_zero_matr(RF_P3_HV, n_a*n_a*n_a*n_a*        Nw);
    set_zero_matr(RF_P3_HH, n_a*n_a*                Nw);
    
    
    
    // VV and VH, for every (t,u,v) a product over the core index i (denominator D(i,t,u,v; f), dE = e_i + e_t - e_u - e_v + E_fit[f]):
    //   RF_P3_VV[t,u,v,(y,x,w),f] += sum_i (iw|xy) (iv|ut) D,   RF_P3_VH[t,u,v,w,f] += sum_i H_CA[i,w] (iv|ut) D
    {
        const size_t n3 = (size_t)n_a*n_a*n_a;
        std::vector<double> B(n3*n_c);   // [(y,x,w), i] = (iw|xy)
        for(int y=0; y<n_a; y++) for(int x=0; x<n_a; x++) for(int w=0; w<n_a; w++) for(int i=0; i<n_c; i++)
            B[((size_t)(y*n_a+x)*n_a+w)*n_c+i]=CAAA[((i*n_a+w)*n_a+x)*n_a+y];
        std::vector<double> HT((size_t)n_a*n_c);   // [w, i] = H_CA[i,w]
        for(int i=0; i<n_c; i++) for(int w=0; w<n_a; w++) HT[(size_t)w*n_c+i]=H_CA[i*n_a+w];
        set_blas_non_par();
#pragma omp parallel
        {
            std::vector<double> G((size_t)n_c*Nw);
#pragma omp for schedule(dynamic)
            for(long tuv=0; tuv<(long)n3; tuv++){
                const int t=tuv/(n_a*n_a), u=(tuv/n_a)%n_a, v=tuv%n_a;
                for(int i=0; i<n_c; i++){
                    const double dE_o=e_c[i]+e_a[t]-e_a[u]-e_a[v];
                    const double c=CAAA[((i*n_a+v)*n_a+u)*n_a+t];
                    for(int i_f=0; i_f<Nw; i_f++){
                        const double dE=dE_o+E_fit[f0+i_f];
                        G[(size_t)i*Nw+i_f]=c*dE/(dE*dE+edshift);
                    }
                }
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n3,Nw,n_c,1.0,B.data(),n_c,G.data(),Nw,1.0,RF_P3_VV+(size_t)tuv*n3*Nw,Nw);
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_a,Nw,n_c,1.0,HT.data(),n_c,G.data(),Nw,1.0,RF_P3_VH+(size_t)tuv*n_a*Nw,Nw);
            }
        }
        set_blas_par();
    }

    //HV
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE_orb= e_c[i]-e_a[t];
        for(int i_f=0; i_f<Nw; i_f++){
            dE=dE_orb+E_fit[f0+i_f];
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
            for(int i_f=0; i_f<Nw; i_f++)
                RF_P3_HV[(((t*n_a+y)*n_a+x)*n_a+w)*Nw+i_f]+=V*ED[i_f];
            
        }
    }
    
    //HH
    for(int i=0; i<n_c; i++)
    for(int t=0; t<n_a; t++){
        dE_orb= e_c[i]-e_a[t];

        for(int i_f=0; i_f<Nw; i_f++){
            dE=dE_orb+E_fit[f0+i_f];
            ED[i_f]=dE/(dE*dE+edshift);
        }
        for(int u=0; u<n_a; u++){
            H1a=H_CA[i*n_a+t];
            H2a=H_CA[i*n_a+u];
            V=H1a*H2a;
                            
            for(int i_f=0; i_f<Nw; i_f++)
                RF_P3_HH[(t*n_a+u)*Nw+i_f]+=V*ED[i_f];
            
        }
    }
    
    
    
    
//     set_zero_matr(RF_P3_JK, n_a*n_a*n_a*n_a*n_a*n_a*Nw);
    
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_P3_JK[(((((t*n_a+u)*n_a+v)*n_a+y)*n_a+x)*n_a+w)*Nw+i_f]+= RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+y)*n_a+w)*n_a+u)*Nw+i_f]
                                                                       -RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+y)*n_a+w)*n_a+t)*Nw+i_f]
                                                                       -RF_P3_VV[(((((t*n_a+v)*n_a+y)*n_a+x)*n_a+w)*n_a+u)*Nw+i_f]
                                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+y)*n_a+x)*n_a+w)*n_a+t)*Nw+i_f];
    
//     set_zero_matr(RF_PV_JK, n_a*n_a*n_a*n_a*Nw);
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PV_JK[(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+f0+i_f]+=-RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+u)*Nw+i_f]
                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+t)*Nw+i_f]
                                                       +RF_P3_VV[(((((t*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+u)*Nw+i_f]
                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+t)*Nw+i_f]
                                                       -RF_P3_VV[(((((t*n_a+x)*n_a+v)*n_a+w)*n_a+u)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((u*n_a+x)*n_a+v)*n_a+w)*n_a+t)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+w)*n_a+t)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((t*n_a+x)*n_a+w)*n_a+v)*n_a+u)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((t*n_a+w)*n_a+x)*n_a+v)*n_a+u)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((u*n_a+x)*n_a+w)*n_a+v)*n_a+t)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((u*n_a+v)*n_a+x)*n_a+w)*n_a+t)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((t*n_a+w)*n_a+x)*n_a+v)*n_a+u)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*Nw+i_f];
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int v=0; v<n_a; v++)
    for(int u=0; u<n_a; u++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PV_JK[(((u*n_a+t)*n_a+v)*n_a+w)*N_fit+f0+i_f]+=-RF_P3_VH[(((t*n_a+v)*n_a+w)*n_a+u)*Nw+i_f]
                                                       -RF_P3_HV[(((v*n_a+w)*n_a+u)*n_a+t)*Nw+i_f]
                                                       +RF_P3_VH[(((u*n_a+v)*n_a+w)*n_a+t)*Nw+i_f]
                                                       +RF_P3_HV[(((v*n_a+w)*n_a+t)*n_a+u)*Nw+i_f]
                                                       +RF_P3_VH[(((t*n_a+w)*n_a+v)*n_a+u)*Nw+i_f]
                                                       +RF_P3_HV[(((w*n_a+v)*n_a+u)*n_a+t)*Nw+i_f]
                                                       -RF_P3_VH[(((u*n_a+w)*n_a+v)*n_a+t)*Nw+i_f]
                                                       -RF_P3_HV[(((w*n_a+v)*n_a+t)*n_a+u)*Nw+i_f];

    
//     set_zero_matr(RF_P3_AB, n_a*n_a*n_a*n_a*n_a*n_a*Nw);
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_P3_AB[(((((t*n_a+u)*n_a+v)*n_a+w)*n_a+x)*n_a+y)*Nw+i_f]+=-RF_P3_VV[(((((t*n_a+v)*n_a+y)*n_a+w)*n_a+u)*n_a+x)*Nw+i_f]
                                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+y)*n_a+w)*n_a+t)*n_a+x)*Nw+i_f]
                                                                       +RF_P3_VV[(((((t*n_a+w)*n_a+y)*n_a+v)*n_a+u)*n_a+x)*Nw+i_f]
                                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+y)*n_a+v)*n_a+t)*n_a+x)*Nw+i_f]
                                                                       -RF_P3_VV[(((((t*n_a+v)*n_a+w)*n_a+y)*n_a+x)*n_a+u)*Nw+i_f]
                                                                       +RF_P3_VV[(((((u*n_a+v)*n_a+w)*n_a+y)*n_a+x)*n_a+t)*Nw+i_f]
                                                                       +RF_P3_VV[(((((t*n_a+w)*n_a+v)*n_a+y)*n_a+x)*n_a+u)*Nw+i_f]
                                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+v)*n_a+y)*n_a+x)*n_a+t)*Nw+i_f]
                                                                       -RF_P3_VV[(((((x*n_a+y)*n_a+v)*n_a+w)*n_a+u)*n_a+t)*Nw+i_f]
                                                                       +RF_P3_VV[(((((x*n_a+y)*n_a+v)*n_a+w)*n_a+t)*n_a+u)*Nw+i_f]
                                                                       +RF_P3_VV[(((((x*n_a+y)*n_a+w)*n_a+v)*n_a+u)*n_a+t)*Nw+i_f]
                                                                       -RF_P3_VV[(((((x*n_a+y)*n_a+w)*n_a+v)*n_a+t)*n_a+u)*Nw+i_f];

//     set_zero_matr(RF_PV_AB, n_a*n_a*n_a*n_a*Nw);
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int x=0; x<n_a; x++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+f0+i_f]+=-RF_P3_VV[(((((t*n_a+x)*n_a+w)*n_a+v)*n_a+x)*n_a+u)*Nw+i_f]
                                                       -RF_P3_VV[(((((t*n_a+x)*n_a+v)*n_a+w)*n_a+u)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+x)*n_a+t)*Nw+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((u*n_a+x)*n_a+v)*n_a+w)*n_a+x)*n_a+t)*Nw+i_f]
                                                       -RF_P3_VV[(((((u*n_a+x)*n_a+w)*n_a+v)*n_a+t)*n_a+x)*Nw+i_f]
                                                       +RF_P3_VV[(((((u*n_a+w)*n_a+x)*n_a+v)*n_a+t)*n_a+x)*Nw+i_f]
                                                       -RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+x)*n_a+u)*Nw+i_f]
                                                       +RF_P3_VV[(((((t*n_a+v)*n_a+x)*n_a+w)*n_a+u)*n_a+x)*Nw+i_f];
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int w=0; w<n_a; w++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PV_AB[(((t*n_a+u)*n_a+v)*n_a+w)*N_fit+f0+i_f]+=-RF_P3_VH[(((u*n_a+w)*n_a+v)*n_a+t)*Nw+i_f]
                                                       -RF_P3_HV[(((v*n_a+w)*n_a+u)*n_a+t)*Nw+i_f]
                                                       -RF_P3_VH[(((t*n_a+v)*n_a+w)*n_a+u)*Nw+i_f]
                                                       -RF_P3_HV[(((w*n_a+v)*n_a+t)*n_a+u)*Nw+i_f];
    
//     set_zero_matr(RF_PH, n_a*n_a*Nw);
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int v=0; v<n_a; v++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PH[(t*n_a+u)*N_fit+f0+i_f]+=-RF_P3_VV[(((((t*n_a+y)*n_a+v)*n_a+u)*n_a+v)*n_a+y)*Nw+i_f]
                                    +RF_P3_VV[(((((t*n_a+v)*n_a+y)*n_a+u)*n_a+v)*n_a+y)*Nw+i_f]*2;
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int y=0; y<n_a; y++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PH[(t*n_a+u)*N_fit+f0+i_f]+=-RF_P3_VH[(((t*n_a+y)*n_a+u)*n_a+y)*Nw+i_f]
                                    +RF_P3_VH[(((t*n_a+u)*n_a+y)*n_a+y)*Nw+i_f]*2
                                    -RF_P3_HV[(((y*n_a+u)*n_a+y)*n_a+t)*Nw+i_f]
                                    +RF_P3_HV[(((y*n_a+u)*n_a+t)*n_a+y)*Nw+i_f]*2;
#pragma omp parallel for collapse(2)
    for(int t=0; t<n_a; t++)
    for(int u=0; u<n_a; u++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PH[(t*n_a+u)*N_fit+f0+i_f]+=-RF_P3_HH[(u*n_a+t)*Nw+i_f];
    
//     set_zero_matr(RF_PS, Nw);
    for(int t=0; t<n_a; t++)
    for(int i_f=0; i_f<Nw; i_f++)
        RF_PS[f0+i_f]+=RF_P3_HH[(t*n_a+t)*Nw+i_f]*2;
    

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
    // For the virtual pairs a <= b, J_ab[t,u] = (ta|ub) and the denominator D_ab(t,u; f) = dE/(dE^2 + s), dE = e_t + e_u - e_a - e_b + E_fit[f]:
    //   RF_PV_AB[t,u,v,w,f] += sum_ab D_ab(t,u;f) [ J_ab[t,u] J_ab[v,w] + (a != b) J_ab[u,t] J_ab[w,v] ]
    //   RF_PV_JK[u,t,v,w,f] += sum_ab D_ab(t,u;f) K_ab[t,u] K_ab[v,w],   K_ab[t,u] = J_ab[t,u] - J_ab[u,t]
    // For a block of pairs: their J (one small product each, in parallel over the pairs), then for every output row (t,u) a matrix product over
    // the pairs: row[(v,w), f] += sum_q Y_q[(v,w)] Y_q[(t,u)] D_q(t,u;f). Every thread writes its own rows: no per-thread copies of the tables.
    const int n2 = n_a*n_a;
    const int naux = RI->aux_n_ao;
    std::vector<int> pa, pb;
    for(int a=0; a<n_v; a++) for(int b=a; b<n_v; b++){ pa.push_back(a); pb.push_back(b); }
    const int n_pair = (int)pa.size();
    const int P = 512;   // pairs per block
    std::vector<double> YAB((size_t)2*P*n2), YK((size_t)P*n2);
    set_blas_non_par();
    for(int p0=0; p0<n_pair; p0+=P){
        const int np = (n_pair-p0 < P) ? n_pair-p0 : P;
#pragma omp parallel
        {
            std::vector<double> Ja(n2);
#pragma omp for schedule(dynamic)
            for(int q=0; q<np; q++){
                const int a=pa[p0+q], b=pb[p0+q];
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n_a,n_a,naux,1.0,
                            RI->VA_RI_M+(size_t)a*n_a*naux,naux,RI->VA_RI_M+(size_t)b*n_a*naux,naux,0.0,Ja.data(),n_a);
                for(int t=0; t<n_a; t++) for(int u=0; u<n_a; u++){
                    YAB[(size_t)(2*q)*n2+t*n_a+u]   = Ja[t*n_a+u];
                    YAB[(size_t)(2*q+1)*n2+t*n_a+u] = (a!=b) ? Ja[u*n_a+t] : 0.0;
                    YK [(size_t)q*n2+t*n_a+u]       = Ja[t*n_a+u]-Ja[u*n_a+t];
                }
            }
        }
#pragma omp parallel
        {
            std::vector<double> GAB((size_t)2*np*N_fit), GK((size_t)np*N_fit);
#pragma omp for schedule(dynamic)
            for(int tu=0; tu<n2; tu++){
                const int t=tu/n_a, u=tu%n_a;
                for(int q=0; q<np; q++){
                    const int a=pa[p0+q], b=pb[p0+q];
                    const double y1=YAB[(size_t)(2*q)*n2+tu], y2=YAB[(size_t)(2*q+1)*n2+tu], yk=YK[(size_t)q*n2+tu];
                    const double dE0=e_a[t]+e_a[u]-e_v[a]-e_v[b];
                    for(int i_f=0; i_f<N_fit; i_f++){
                        const double dE=dE0+E_fit[i_f];
                        const double d=dE/(dE*dE+edshift);
                        GAB[(size_t)(2*q)*N_fit+i_f]=y1*d;
                        GAB[(size_t)(2*q+1)*N_fit+i_f]=y2*d;
                        GK[(size_t)q*N_fit+i_f]=yk*d;
                    }
                }
                cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,2*np,1.0,YAB.data(),n2,GAB.data(),N_fit,1.0,
                            RF_PV_AB+(size_t)tu*n2*N_fit,N_fit);
                cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n2,N_fit,np,1.0,YK.data(),n2,GK.data(),N_fit,1.0,
                            RF_PV_JK+(size_t)(u*n_a+t)*n2*N_fit,N_fit);
            }
        }
    }
    set_blas_par();
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
