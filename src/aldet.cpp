//standart
# include <stdio.h>
# include <omp.h>

//user
# include "blas_link.h"
# include "aldet.h"
# include "matr.h"
# include "timer.h"
# include  "from_hash.h"
# include "common_vars.h"
# include "version.h"
# include "defaults.h"
# include "etc.h"

# define max(a,b)  (((a)<(b))?(b):(a))
# define min(a,b)  (((a)>(b))?(b):(a))

#define NUM_AVX 1

#define TEST_ALDET

int sparsed_CI_vec::decompress(double * C, int a, int n_s){
    
    int l=c.size();
    for(int i=0; i<l;i++){
        C[n[i]*n_s+a]=c[i];
    }
    
    return 0;
}

int CI_sd_mult(double * O, int ld, sparsed_CI_vec * s, int n_s, double * d, int n_d, int ld_d){
    
    double * c;
    ci_int * a;
    int l;
    
    for(int i=0; i<n_s;i++){
        l=s[i].n.size();
        a=s[i].n.data();
        c=s[i].c.data();
        set_zero_matr(O+i*ld,n_d);
        for(int k=0;k<l;k++)
        for(int j=0;j<n_d;j++)
            O[i*ld+j]+=d[a[k]*ld_d+j]*c[k];
    }
    
    return 0;
    
}

int CI_sd_mult_tr(double * O, int ld, sparsed_CI_vec * s, int n_s, double * R, int n_d, int ld_d){
    
    double * c;
    ci_int * a;
    int l;
    
    for(int i=0; i<n_s;i++){
        l=s[i].n.size();
        a=s[i].n.data();
        c=s[i].c.data();
//         set_zero_matr(O+i*ld,ld);
        for(int k=0;k<l;k++)
        for(int j=0;j<n_d;j++)
            O[a[k]*n_d+j]+=R[j*ld_d+i]*c[k];
    }
    
    return 0;
    
}

int CI_ss_mult(double * O, int ld, sparsed_CI_vec * s1, int n_s1, sparsed_CI_vec * s2, int n_s2){
    
// #pragma omp parallel for        
    for(int i=0; i<n_s1;i++){
        int       l1=s1[i].n.size();
        ci_int *     a1=s1[i].n.data();
        double *  c1=s1[i].c.data();
        
        
        for(int j=0; j<n_s2;j++){
            int      l2=s2[j].n.size();
            ci_int *    a2=s2[j].n.data();
            double * c2=s2[j].c.data();

            for(int x=0;x<l1;x++)
            for(int y=0;y<l2;y++)
            if(a1[x]==a2[y]){
                O[i*ld+j]+=c1[x]*c2[y];
            }
//             exit(0);
        }
//             getchar();
    }
    
    return 0;
    
}


aldet_data::aldet_data(){
    
    import_done=0;
    
    n_act  = 0;
    na     = 0;
    nb     = 0;
    n_sets = 0;
    print_number = CAS_PRINT_NUMBER_DEFAULT;
    
    do_PT = 0;
    
    alloc_mode = ALDET_ALLOC_FULL;
    
    Na = 0;
    Nb = 0;    
    Nd = 0;
    fa = NULL;
    fb = NULL;
    vec_a = NULL;
    vec_b = NULL;
    
    bit_a = NULL;
    bit_b = NULL;
    buf = NULL;//more space for +a-b and -a+b ????
    
    
    n_states = NULL;
    coef     = NULL;
    coef_bas  = NULL;
    coef_asb  = NULL;
    coef_bsa  = NULL;
    
    E_states  = NULL;
    S2        = NULL;
    L2        = NULL;
    P         = NULL;
    E_act     = NULL;
    
    H_diag    = NULL;
    H_diag_appr=NULL;
    Hv        = NULL;
    Hv_buf    = NULL;
    
    Ia         = NULL;
    Ia_friends = NULL;
    Ib         = NULL;
    Ib_friends = NULL;

    J_act_a= NULL;
    J_act_b= NULL;
    K_act_a= NULL;
    K_act_b= NULL;
    
//     sym_ab = NULL;
    
    e1_ind_a  = NULL;
    e1_ind_b  = NULL;
    e1_orbs_a = NULL;
    e1_orbs_b = NULL;
    e1_sign_a = NULL;
    e1_sign_b = NULL;
//     e1_asm_ints_b = NULL;
    
    e2_orbs_a = NULL;
    e2_orbs_b = NULL;
    e2_sign_a = NULL;
    e2_sign_b = NULL;
    e2_ind_a = NULL;  
    e2_ind_b = NULL; 
    e2_V_a   = NULL;  
    e2_V_b   = NULL;  

    e3_ind_a = NULL;  
    e3_ind_b = NULL; 
    e3_V_a   = NULL;  
    e3_V_b   = NULL;  
    
    a_spin_sign = NULL;
    b_spin_sign = NULL;
    
    F_act_A     = NULL;
    F_act_B     = NULL;
    act_INTS_AA = NULL;
    act_INTS_AB = NULL;
    act_INTS_BB = NULL;
    T3_AAA     = NULL;
    T3_AAB     = NULL;
    T3_BBA     = NULL;
    T3_BBB     = NULL;
    T2_AA      = NULL;
    T2_AB      = NULL;
    T2_BB      = NULL;
    T1_A       = NULL;
    T1_B       = NULL;
    
    
}


int aldet_data::get_dim_meta(int ext_n_act, int ext_na, int ext_nb, int ext_n_sets, int ext_mult, int ext_print_number){
    
    int status =0;
    // 0 - normal
    // 1 - negative alpha
    // 2 - negative beta
    // 4 - excessive alpha
    // 8 - excessive beta
    
    n_act        = ext_n_act       ;
    na           = ext_na          ;
    nb           = ext_nb          ;
    n_sets       = ext_n_sets      ;
    mult         = ext_mult        ;
    print_number = ext_print_number;
    
    if(na<0    )status+=1;
    if(nb<0    )status+=2;
    if(na>n_act)status+=4;
    if(nb>n_act)status+=8;
    
    if(status){
        printf("ERROR: aldet_data::get_dim returned status %d which means:\n",status);
        if(status>7){printf("       nb(%d)>n_act(%d) -- status 8;\n",nb,n_act);status-=8;}
        if(status>3){printf("       na(%d)>n_act(%d) -- status 4;\n",na,n_act);status-=4;}
        if(status>1){printf("       nb(%d)>0         -- status 2;\n",nb,n_act);status-=2;}
        if(status>1){printf("       na(%d)>0         -- status 1;\n",na,n_act);status-=1;}
        exit(1);
    }
        
    Na = (int) std::lround(tgammal(n_act+1) / tgammal(na+1) / tgammal(n_act-na+1));
    Nb = (int) std::lround(tgammal(n_act+1) / tgammal(nb+1) / tgammal(n_act-nb+1));    
    Nd = (long)Na*Nb;
    
    
    if(print_number>Nd)
        print_number=Nd;
    
    n_states = new int     [n_sets];
    coef     = new double *[n_sets];for(int i=0;i<n_sets;i++)coef    [i]=NULL;
    coef_bas = new double *[n_sets];for(int i=0;i<n_sets;i++)coef_bas[i]=NULL;
    coef_asb = new double *[n_sets];for(int i=0;i<n_sets;i++)coef_asb[i]=NULL;
    coef_bsa = new double *[n_sets];for(int i=0;i<n_sets;i++)coef_bsa[i]=NULL;
    E_states = new double *[n_sets];for(int i=0;i<n_sets;i++)E_states[i]=NULL;
    S2       = new double *[n_sets];for(int i=0;i<n_sets;i++)S2      [i]=NULL;
    L2       = new double *[n_sets];for(int i=0;i<n_sets;i++)L2      [i]=NULL;
    P        = new double *[n_sets];for(int i=0;i<n_sets;i++)P       [i]=NULL;
    E_act    = new double *[n_sets];for(int i=0;i<n_sets;i++)E_act   [i]=NULL;
    
    sym_ab = 0;
    if(mult!=0){
        sym_ab=1;
        for(int i=mult;i>1;i-=2)sym_ab=-sym_ab;
    }
    
#ifdef TEST_ALDET
    n_e1_a = na*(n_act-na+1);
    n_e1_b = nb*(n_act-nb+1);
#endif
#ifndef TEST_ALDET    
    n_e1_a = na*(n_act-na);
    n_e1_b = nb*(n_act-nb);
#endif
    
    if(n_e1_a%NUM_AVX)n_e1_a = n_e1_a+NUM_AVX-n_e1_a%NUM_AVX;
    if(n_e1_b%NUM_AVX)n_e1_b = n_e1_b+NUM_AVX-n_e1_b%NUM_AVX;
    
#ifdef TEST_ALDET
    n_e2_a = na*(na-1)*(n_act-na+2)*(n_act-na+1)/4;
    n_e2_b = nb*(nb-1)*(n_act-nb+2)*(n_act-nb+1)/4;
#endif
#ifndef TEST_ALDET
    n_e2_a = na*(na-1)*(n_act-na)*(n_act-na-1)/4;
    n_e2_b = nb*(nb-1)*(n_act-nb)*(n_act-nb-1)/4;
#endif
    
    
    return 0;
}


int aldet_data::get_dim(int ext_n_act, int ext_na, int ext_nb, int ext_n_sets, int ext_mult, int ext_print_number, int ext_alloc){
    
    get_dim_meta(ext_n_act, ext_na, ext_nb, ext_n_sets, ext_mult, ext_print_number);
    
    alloc_mode = ext_alloc;
    if((alloc_mode!=ALDET_ALLOC_FULL)&&(alloc_mode!=ALDET_ALLOC_DIMS)){
        fprintf(out_stream,"ERROR: aldet_data::get_dim got unknown allocation mode %d\n", alloc_mode);
        exit(1);
    }
    if(alloc_mode==ALDET_ALLOC_DIMS)return 0;
    
//     if(na==0)if(nb==0)return 0;
    fa = new int [na * n_act];
    fb = new int [nb * n_act];
    vec_a = new int [Na*(na+1)];
    vec_b = new int [Nb*(nb+1)];
    get_factorials(na, n_act, fa);
    get_factorials(nb, n_act, fb);
    
    get_vec(na, n_act, Na, vec_a);
    get_vec(nb, n_act, Nb, vec_b);
    
    bit_a = new int [n_act];
    bit_b = new int [n_act];
    buf = new int [max(na+1,nb+1) + 1];//more space for +a-b and -a+b ????
    buf[0]=0;
    
    const int Na_zv = (int) std::lround(tgammal(n_act) / tgammal(na) / tgammal(n_act-na+1));
    const int Nb_zv = (int) std::lround(tgammal(n_act) / tgammal(nb) / tgammal(n_act-nb+1));
    Ia = new int [Na_zv];
    Ia_friends = new int [Na_zv * (n_act-na) * 3];
    Ib = new int [Nb_zv];
    Ib_friends = new int [Nb_zv * (n_act-nb) * 3];
    
    J_act_a = new double[n_act*n_act*Na];//????????
    J_act_b = new double[n_act*n_act*Nb];//????????
    K_act_a = new double[n_act*n_act*Na];//????????
    K_act_b = new double[n_act*n_act*Nb];//????????
#ifdef TEST_ALDET
#ifdef DEBUG_WARN
    fprintf(out_stream,"\n\nWARNING current version (%s) has non-optimal ALDet CI engine - diagonal elements are not skiped!\n\n\n", VERSION);
#endif
#endif
    
    act_INTS_AA = new double[n_act*n_act*n_act*n_act];
    act_INTS_AB = new double[n_act*n_act*n_act*n_act];
    act_INTS_BB = new double[n_act*n_act*n_act*n_act];
    F_act_A     = new double[n_act*n_act            ];
    F_act_B     = new double[n_act*n_act            ];
    
    e1_ind_a  = new ci_int[Na*n_e1_a];
    e1_ind_b  = new ci_int[Nb*n_e1_b];
    e1_orbs_a = new ci_int[Na*n_e1_a];
    e1_orbs_b = new ci_int[Nb*n_e1_b];
    e1_sign_a = new double[Na*n_e1_a];
    e1_sign_b = new double[Nb*n_e1_b];
//     e1_asm_ints_b = new int[2*Nb*n_e1_b];
    e2_ind_a  = new ci_int[Na*n_e2_a];
    e2_ind_b  = new ci_int[Nb*n_e2_b];
    e2_V_a    = new double[Na*n_e2_a];
    e2_V_b    = new double[Nb*n_e2_b];
    e2_sign_a = new double[Na*n_e2_a];
    e2_sign_b = new double[Nb*n_e2_b];
    e2_orbs_a = new ci_int[Na*n_e2_a];
    e2_orbs_b = new ci_int[Nb*n_e2_b];
    
    
    a_spin_sign = new double[Na*n_act];
    b_spin_sign = new double[Nb*n_act];
    
    for(int i_CI = 0; i_CI<Na; i_CI++){
        if(na>0)a_spin_sign[i_CI*n_act+vec_a[i_CI*(na+1)+na]-1]=1;
        for (int k=na-1; k>0; k--){
            a_spin_sign[i_CI*n_act+vec_a[i_CI*(na+1)+k]-1]=-a_spin_sign[i_CI*n_act+vec_a[i_CI*(na+1)+k+1]-1];
        }
        
    }
        
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        double sign=1;
        for(int t=0  ;t<n_act;t++){
            if(bit_b[t]==0){
                b_spin_sign[j_CI*n_act+t]=sign;
            }
            else
                sign=-sign;
        }
        
    }
    
    return 0;
}

int aldet_data::simple_import_data(double * ext_act_INTS,
                                   double * ext_act_INTS_AB,
                                   double * ext_F_act,
                                   double   ext_E_core){
    
    if(n_act==0){
        printf("ERROR: can not import arrays to aldet_data wth n_act=0\n");
        printf("       it means that you use corrupted version of NOPT\n");
        exit(0);
    }
    
    memcpy(act_INTS_AA, ext_act_INTS   , sizeof(double)*n_act*n_act*n_act*n_act);
    memcpy(act_INTS_AB, ext_act_INTS_AB, sizeof(double)*n_act*n_act*n_act*n_act);
    memcpy(act_INTS_BB, ext_act_INTS   , sizeof(double)*n_act*n_act*n_act*n_act);
    memcpy(F_act_A    , ext_F_act      , sizeof(double)*n_act*n_act            );
    memcpy(F_act_B    , ext_F_act      , sizeof(double)*n_act*n_act            );
    E_core      = ext_E_core    ;
    
    import_done=1;
    
    return 0;
    
}

int aldet_data::U_simple_import_data(double * ext_act_INTS_AA,
                                     double * ext_act_INTS_AB,
                                     double * ext_act_INTS_BB,
                                     double * ext_F_act_A,
                                     double * ext_F_act_B,
                                     double   ext_E_core){
    
    if(n_act==0){
        printf("ERROR: can not import arrays to aldet_data wth n_act=0\n");
        printf("       it means that you use corrupted version of NOPT\n");
        exit(0);
    }
    
    memcpy(act_INTS_AA, ext_act_INTS_AA, sizeof(double)*n_act*n_act*n_act*n_act);
    memcpy(act_INTS_AB, ext_act_INTS_AB, sizeof(double)*n_act*n_act*n_act*n_act);
    memcpy(act_INTS_BB, ext_act_INTS_BB, sizeof(double)*n_act*n_act*n_act*n_act);
    memcpy(F_act_A    , ext_F_act_A    , sizeof(double)*n_act*n_act            );
    memcpy(F_act_B    , ext_F_act_B    , sizeof(double)*n_act*n_act            );
    E_core      = ext_E_core    ;
    
    import_done=1;
    
    
    return 0;
    
}

int aldet_data::PT2_alloc(){
    
    if(T3_AAA==nullptr) T3_AAA = new double[n_act*n_act*n_act*n_act*n_act*n_act];
    if(T3_AAB==nullptr) T3_AAB = new double[n_act*n_act*n_act*n_act*n_act*n_act];
    if(T3_BBA==nullptr) T3_BBA = new double[n_act*n_act*n_act*n_act*n_act*n_act];
    if(T3_BBB==nullptr) T3_BBB = new double[n_act*n_act*n_act*n_act*n_act*n_act];
    if(T2_AA ==nullptr) T2_AA  = new double[n_act*n_act*n_act*n_act            ];
    if(T2_AB ==nullptr) T2_AB  = new double[n_act*n_act*n_act*n_act            ];
    if(T2_BB ==nullptr) T2_BB  = new double[n_act*n_act*n_act*n_act            ];
    if(T1_A  ==nullptr) T1_A   = new double[n_act*n_act                        ];
    if(T1_B  ==nullptr) T1_B   = new double[n_act*n_act                        ];
    
    return 0;
    
}



int aldet_data::PT2_import_data(double * ext_T3,
                                double * ext_T3_AB,
                                double * ext_T2,
                                double * ext_T2_AB,
                                double * ext_T1,
                                double   ext_T0){
    
    if(n_act==0){
        printf("ERROR: can not import arrays to aldet_data wth n_act=0\n");
        printf("       it means that you use corrupted version of NOPT\n");
        exit(0);
    }
    
    PT2_alloc();
    
    memcpy(T3_AAA, ext_T3   , sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T3_AAB, ext_T3_AB, sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T3_BBA, ext_T3_AB, sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T3_BBB, ext_T3   , sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T2_AA , ext_T2   , sizeof(double)*n_act*n_act*n_act*n_act            );
    memcpy(T2_AB , ext_T2_AB, sizeof(double)*n_act*n_act*n_act*n_act            );
    memcpy(T2_BB , ext_T2   , sizeof(double)*n_act*n_act*n_act*n_act            );
    memcpy(T1_A  , ext_T1   , sizeof(double)*n_act*n_act                        );
    memcpy(T1_B  , ext_T1   , sizeof(double)*n_act*n_act                        );
    T0    = ext_T0   ;
    
    do_PT = 1;
    
    return 0;
    
}

int aldet_data::UPT2_import_data(double * ext_T3_AAA,
                                 double * ext_T3_AAB,
                                 double * ext_T3_BBA,
                                 double * ext_T3_BBB,
                                 double * ext_T2_AA,
                                 double * ext_T2_AB,
                                 double * ext_T2_BB,
                                 double * ext_T1_A,
                                 double * ext_T1_B,
                                 double   ext_T0){
    
    if(n_act==0){
        printf("ERROR: can not import arrays to aldet_data wth n_act=0\n");
        printf("       it means that you use corrupted version of NOPT\n");
        exit(0);
    }
    
    PT2_alloc();
    
    memcpy(T3_AAA, ext_T3_AAA , sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T3_AAB, ext_T3_AAB , sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T3_BBA, ext_T3_BBA , sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T3_BBB, ext_T3_BBB , sizeof(double)*n_act*n_act*n_act*n_act*n_act*n_act);
    memcpy(T2_AA , ext_T2_AA  , sizeof(double)*n_act*n_act*n_act*n_act            );
    memcpy(T2_AB , ext_T2_AB  , sizeof(double)*n_act*n_act*n_act*n_act            );
    memcpy(T2_BB , ext_T2_BB  , sizeof(double)*n_act*n_act*n_act*n_act            );
    memcpy(T1_A  , ext_T1_A   , sizeof(double)*n_act*n_act                        );
    memcpy(T1_B  , ext_T1_B   , sizeof(double)*n_act*n_act                        );
    T0    = ext_T0      ;
    
    do_PT = 1;
    
    return 0;
    
}

int aldet_data::PT2_delete_data(){
    
    
    if(T3_AAA!=NULL) set_zero_matr(T3_AAA, n_act*n_act*n_act*n_act*n_act*n_act);
    if(T3_AAB!=NULL) set_zero_matr(T3_AAB, n_act*n_act*n_act*n_act*n_act*n_act);
    if(T3_BBA!=NULL) set_zero_matr(T3_BBA, n_act*n_act*n_act*n_act*n_act*n_act);
    if(T3_BBB!=NULL) set_zero_matr(T3_BBB, n_act*n_act*n_act*n_act*n_act*n_act);
    if(T2_AA !=NULL) set_zero_matr(T2_AA , n_act*n_act*n_act*n_act            );
    if(T2_AB !=NULL) set_zero_matr(T2_AB , n_act*n_act*n_act*n_act            );
    if(T2_BB !=NULL) set_zero_matr(T2_BB , n_act*n_act*n_act*n_act            );
    if(T1_A  !=NULL) set_zero_matr(T1_A  , n_act*n_act                        );
    if(T1_B  !=NULL) set_zero_matr(T1_B  , n_act*n_act                        );
    T0 = 0;
    
    do_PT = 0;
    
    return 0;
    
}


int aldet_data::gen_ext_ind(){
    
    if(na==0)if(nb==0)return 0;
    
    buf[0]=0;
    
    double sign1;
    double sign2;
    int i_ext;
    
    if(import_done==0){
        fprintf(out_stream,"ERROR: in function gen_ext_ind()\n");
        fprintf(out_stream,"       external references must be set by aldet_data impport function\n");
        exit(0);
    }
    
    set_zero_matr(J_act_a , n_act*n_act*Na);
    set_zero_matr(J_act_b , n_act*n_act*Nb);
    set_zero_matr(K_act_a , n_act*n_act*Na);
    set_zero_matr(K_act_b , n_act*n_act*Nb);
    
    //AA
    for(int i_CI =0; i_CI<Na; i_CI++){
        i_ext=0;
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        sign1=1.0;
        for(int t=0  ;t<n_act;t++)if(bit_a[t]!=0){
            sign2=1.0;
            bit_a[t]=0;
            for(int v=0  ;v<n_act;v++){
                if(bit_a[v]==0){
                    bit_a[v]=1;
                    e1_sign_a[i_CI*n_e1_a+i_ext] = sign1*sign2;
                    e1_ind_a [i_CI*n_e1_a+i_ext] = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                    e1_orbs_a[i_CI*n_e1_a+i_ext] = t*n_act+v;
                    i_ext++;
                    bit_a[v]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
        while(i_ext!=n_e1_a){
            e1_sign_a[i_CI*n_e1_a+i_ext] = 0;
            e1_ind_a [i_CI*n_e1_a+i_ext] = e1_ind_a[i_CI*n_e1_a+i_ext-1];
            e1_orbs_a[i_CI*n_e1_a+i_ext] = e1_orbs_a    [i_CI*n_e1_a+i_ext-1];
            i_ext++;
        }
    }
    
    //BB
    for(int i_CI =0; i_CI<Nb; i_CI++){
        
        i_ext=0;    
        
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[i_CI*(nb+1)+k]-1] = 1;
        
        //BB
        sign1=1.0;
        for(int t=0  ;t<n_act;t++)if(bit_b[t]!=0){
            sign2=1.0;
            bit_b[t]=0;
            for(int v=0  ;v<n_act;v++){
                if(bit_b[v]==0){
                    bit_b[v]=1;
                    e1_sign_b[i_CI*n_e1_b+i_ext] = sign1*sign2;
                    e1_ind_b [i_CI*n_e1_b+i_ext] = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                    e1_orbs_b[i_CI*n_e1_b+i_ext] = t*n_act+v;
//                     int i2 =i_ext/NUM_AVX;
//                     i2 = i2*NUM_AVX+i_ext;
//                     e1_asm_ints_b[2*i_CI*n_e1_b+i2        ]=e1_orbs_b[i_CI*n_e1_b+i_ext];
//                     e1_asm_ints_b[2*i_CI*n_e1_b+i2+NUM_AVX]=e1_ind_b [i_CI*n_e1_b+i_ext]*n_states[0];
                    i_ext++;
                    bit_b[v]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_b[t]=1;
            sign1=-sign1;
        }
        while(i_ext!=n_e1_b){
            e1_sign_b[i_CI*n_e1_b+i_ext] = 0;
            e1_ind_b [i_CI*n_e1_b+i_ext] = e1_ind_b[i_CI*n_e1_b+i_ext-1];
            e1_orbs_b[i_CI*n_e1_b+i_ext] = e1_orbs_b    [i_CI*n_e1_b+i_ext-1];
//             int i2 =i_ext/NUM_AVX;
//             i2 = i2*NUM_AVX+i_ext;
//             e1_asm_ints_b[2*i_CI*n_e1_b+i2        ]=e1_orbs_b[i_CI*n_e1_b+i_ext];
//             e1_asm_ints_b[2*i_CI*n_e1_b+i2+NUM_AVX]=e1_ind_b [i_CI*n_e1_b+i_ext]*n_states[0];
//             fprintf(out_stream,"%d %d %d %d\n",i_ext,i2,i2+NUM_AVX,n_states[0]);
//             getchar();
//             fprintf(out_stream,"b %d %d %d %d\n",i_ext,e1_sign_b   [i_CI*n_e1_b+i_ext],e1_ind_b[i_CI*n_e1_b+i_ext],e1_orbs_b[i_CI*n_e1_b+i_ext]);
            i_ext++;
        }
    }
//     getchar();
    //AAAA
    i_ext=0;
    for(int i_CI =0; i_CI<Na; i_CI++){
        
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        for(int t=0  ;t<n_act;t++)if(bit_a[t]!=0){
            sign1=1.0;
            bit_a[t]=0;
            for(int u=t+1;u<n_act;u++){
                if(bit_a[u]!=0){
                    bit_a[u]=0;
                    for(int v=0  ;v<n_act;v++){
                        if(bit_a[v]==0){
                            sign2=1.0;
                            bit_a[v]=1;
                            for(int w=v+1;w<n_act;w++){
                                if(bit_a[w]==0){
                                    bit_a[w]=1;
#ifndef TEST_ALDET
                                    if(t!=v)
                                    if(t!=w)
                                    if(u!=v)
                                    if(u!=w)
#endif
                                    {
                                        e2_orbs_a[i_ext] = ((t*n_act+u)*n_act+v)*n_act+w;
                                        e2_ind_a[i_ext] = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                                        e2_V_a  [i_ext] =(act_INTS_AA[((t*n_act+v)*n_act+u)*n_act+w]-
                                                          act_INTS_AA[((t*n_act+w)*n_act+u)*n_act+v])*
                                                          sign1*sign2;
                                        e2_sign_a[i_ext]= sign1*sign2;
                                        i_ext++;
                                    }
                                    bit_a[w]=0;
                                }
                                else
                                    sign2=-sign2;
                            }
                            bit_a[v]=0;
                        }
                    }
                    bit_a[u]=1;
                    sign1=-sign1;
                }
            }
            bit_a[t]=1;
        }
//         fprintf(out_stream,"%d %d\n", i_ext, n_e2_a);
//         getchar();
        
    }
    
//     fprintf(out_stream,"%d %d\n", i_ext, n_e2_a*Na);
//     getchar();
        
    //BBBB
    i_ext=0;
    for(int i_CI =0; i_CI<Nb; i_CI++){
        
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[i_CI*(nb+1)+k]-1] = 1;
        
        for(int t=0  ;t<n_act;t++)if(bit_b[t]!=0){
            sign1=1.0;
            bit_b[t]=0;
            for(int u=t+1;u<n_act;u++){
                if(bit_b[u]!=0){
                    bit_b[u]=0;
                    for(int v=0  ;v<n_act;v++){
                        if(bit_b[v]==0){
                            sign2=1.0;
                            bit_b[v]=1;
                            for(int w=v+1;w<n_act;w++){
                                if(bit_b[w]==0){
                                    bit_b[w]=1;
#ifndef TEST_ALDET
                                    if(t!=v)
                                    if(t!=w)
                                    if(u!=v)
                                    if(u!=w)
#endif
                                    {
                                        e2_orbs_b[i_ext] = ((t*n_act+u)*n_act+v)*n_act+w;
                                        e2_ind_b[i_ext] = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
//                                         fprintf(stderr,"BBBBB , %d\n",e2_ind_b[i_ext]);
                                        e2_V_b  [i_ext] =(act_INTS_BB[((t*n_act+v)*n_act+u)*n_act+w]-
                                                          act_INTS_BB[((t*n_act+w)*n_act+u)*n_act+v])*
                                                          sign1*sign2;
                                        e2_sign_b[i_ext]= sign1*sign2;
                                        i_ext++;
                                    }
                                    bit_b[w]=0;
                                }
                                else
                                    sign2=-sign2;
                            }
                            bit_b[v]=0;
                        }
                    }
                    bit_b[u]=1;
                    sign1=-sign1;
                }
            }
            bit_b[t]=1;
        }
    }
    
    
//     fprintf(out_stream,"b%d %d\n", i_ext, n_e2_b*Nb);
//     getchar();
    
    return 0;
    
}

int aldet_data::gen_ext_ind_PT(){
    
    buf[0]=0;
    
    double sign1;
    double sign2;
    double sign3;
    double sign4;
    double t_sign;
    
    int i_ext;
    
    
    set_zero_matr(J_act_a , n_act*n_act*Na);
    set_zero_matr(J_act_b , n_act*n_act*Nb);
    set_zero_matr(K_act_a , n_act*n_act*Na);
    set_zero_matr(K_act_b , n_act*n_act*Nb);
    
    //this will be optimized
    n_e1_a = na*(n_act-na+1);
    n_e1_b = nb*(n_act-nb+1);
    if(n_e1_a%NUM_AVX)n_e1_a = n_e1_a+NUM_AVX-n_e1_a%NUM_AVX;
    if(n_e1_b%NUM_AVX)n_e1_b = n_e1_b+NUM_AVX-n_e1_b%NUM_AVX;
    
    if(e1_ind_a  !=NULL) delete[] e1_ind_a ;
    if(e1_ind_b  !=NULL) delete[] e1_ind_b ;
    if(e1_orbs_a !=NULL) delete[] e1_orbs_a;
    if(e1_orbs_b !=NULL) delete[] e1_orbs_b;
    if(e1_sign_a !=NULL) delete[] e1_sign_a;
    if(e1_sign_b !=NULL) delete[] e1_sign_b;
    
    e1_ind_a  = new ci_int[Na*n_e1_a];
    e1_ind_b  = new ci_int[Nb*n_e1_b];
    e1_orbs_a = new ci_int[Na*n_e1_a];
    e1_orbs_b = new ci_int[Nb*n_e1_b];
    e1_sign_a = new double[Na*n_e1_a];
    e1_sign_b = new double[Nb*n_e1_b];
//     e1_asm_ints_b = new int[2*Nb*n_e1_b];
    
    n_e2_a = na*(na-1)*(n_act-na+2)*(n_act-na+1)/4;
    n_e2_b = nb*(nb-1)*(n_act-nb+2)*(n_act-nb+1)/4;
    if(n_e2_a%NUM_AVX)n_e2_a = n_e2_a+NUM_AVX-n_e2_a%NUM_AVX;
    if(n_e2_b%NUM_AVX)n_e2_b = n_e2_b+NUM_AVX-n_e2_b%NUM_AVX;
    
    if(e2_sign_a  !=NULL) delete[] e2_sign_a ;
    if(e2_sign_b  !=NULL) delete[] e2_sign_b ;
    if(e2_orbs_a  !=NULL) delete[] e2_orbs_a ;
    if(e2_orbs_b  !=NULL) delete[] e2_orbs_b ;
    if(e2_ind_a != NULL) delete[] e2_ind_a;
    if(e2_ind_b != NULL) delete[] e2_ind_b;
    if(e2_V_a   != NULL) delete[] e2_V_a  ;
    if(e2_V_b   != NULL) delete[] e2_V_b  ;
    
    
    e2_sign_a = new double[Na*n_e2_a];
    e2_sign_b = new double[Nb*n_e2_b];
    e2_orbs_a = new ci_int[Na*n_e2_a];
    e2_orbs_b = new ci_int[Nb*n_e2_b];
    e2_ind_a  = new ci_int[Na*n_e2_a];
    e2_ind_b  = new ci_int[Nb*n_e2_b];
    e2_V_a    = new double[Na*n_e2_a];
    e2_V_b    = new double[Nb*n_e2_b];
    
    n_e3_a = na*(na-1)*(na-2)*(n_act-na+3)*(n_act-na+2)*(n_act-na+1)/36;
    n_e3_b = nb*(nb-1)*(nb-2)*(n_act-nb+3)*(n_act-nb+2)*(n_act-nb+1)/36;
    
    if(e3_ind_a != NULL) delete[] e3_ind_a;
    if(e3_ind_b != NULL) delete[] e3_ind_b;
    if(e3_V_a   != NULL) delete[] e3_V_a  ;
    if(e3_V_b   != NULL) delete[] e3_V_b  ;
    
    
    e3_ind_a  = new ci_int[Na*n_e3_a];
    e3_ind_b  = new ci_int[Nb*n_e3_b];
    e3_V_a    = new double[Na*n_e3_a];
    e3_V_b    = new double[Nb*n_e3_b];
    
    //AA
    for(int i_CI =0; i_CI<Na; i_CI++){
        i_ext=0;
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        sign1=1.0;
        for(int t=0  ;t<n_act;t++)if(bit_a[t]!=0){
            sign2=1.0;
            bit_a[t]=0;
            for(int v=0  ;v<n_act;v++){
                if(bit_a[v]==0){
                    bit_a[v]=1;
//                     if(v!=t){
                    e1_sign_a[i_CI*n_e1_a+i_ext] = sign1*sign2;
                    e1_ind_a [i_CI*n_e1_a+i_ext] = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                    e1_orbs_a[i_CI*n_e1_a+i_ext] = t*n_act+v;
                    i_ext++;
//                     }
                    bit_a[v]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
        while(i_ext!=n_e1_a){
            e1_sign_a[i_CI*n_e1_a+i_ext] = 0;
            e1_ind_a [i_CI*n_e1_a+i_ext] = e1_ind_a[i_CI*n_e1_a+i_ext-1];
            e1_orbs_a[i_CI*n_e1_a+i_ext] = e1_orbs_a    [i_CI*n_e1_a+i_ext-1];
            i_ext++;
        }
    }
    
    
    
//     getchar();

    //BB
    for(int i_CI =0; i_CI<Nb; i_CI++){
        
        i_ext=0;    
        
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[i_CI*(nb+1)+k]-1] = 1;
        
        //BB
        sign1=1.0;
        for(int t=0  ;t<n_act;t++)if(bit_b[t]!=0){
            sign2=1.0;
            bit_b[t]=0;
            for(int v=0  ;v<n_act;v++){
                if(bit_b[v]==0){
                    bit_b[v]=1;
//                     if(v!=t){
                    e1_sign_b[i_CI*n_e1_b+i_ext] = sign1*sign2;
                    e1_ind_b [i_CI*n_e1_b+i_ext] = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                    e1_orbs_b[i_CI*n_e1_b+i_ext] = t*n_act+v;
                    i_ext++;
//                     }
                    bit_b[v]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_b[t]=1;
            sign1=-sign1;
        }
        while(i_ext!=n_e1_b){
            e1_sign_b[i_CI*n_e1_b+i_ext] = 0;
            e1_ind_b [i_CI*n_e1_b+i_ext] = e1_ind_b[i_CI*n_e1_b+i_ext-1];
            e1_orbs_b[i_CI*n_e1_b+i_ext] = e1_orbs_b    [i_CI*n_e1_b+i_ext-1];
            int i2 =i_ext/NUM_AVX;
            i2 = i2*NUM_AVX+i_ext;
            i_ext++;
        }
    }
//     getchar();
    //AAAA
    for(int i_CI =0; i_CI<Na; i_CI++){
        i_ext=0;
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        for(int t=0  ;t<n_act;t++)if(bit_a[t]!=0){
            sign1=1.0;
            bit_a[t]=0;
            for(int u=t+1;u<n_act;u++){
                if(bit_a[u]!=0){
                    bit_a[u]=0;
                    for(int v=0  ;v<n_act;v++){
                        if(bit_a[v]==0){
                            sign2=1.0;
                            bit_a[v]=1;
                            for(int w=v+1;w<n_act;w++){
                                if(bit_a[w]==0){
                                    bit_a[w]=1;
//                                     if(t!=v)
//                                     if(t!=w)
//                                     if(u!=v)
//                                     if(u!=w){
                                        e2_orbs_a[i_CI*n_e2_a+i_ext] = ((t*n_act+u)*n_act+v)*n_act+w;
                                        e2_sign_a[i_CI*n_e2_a+i_ext] = sign1*sign2;
                                        e2_ind_a [i_CI*n_e2_a+i_ext] = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                                        e2_V_a   [i_CI*n_e2_a+i_ext] =(act_INTS_AA[((t*n_act+v)*n_act+u)*n_act+w]-
                                                                       act_INTS_AA[((t*n_act+w)*n_act+u)*n_act+v])*
                                                                       sign1*sign2;
                                        i_ext++;
//                                     }
                                    bit_a[w]=0;
                                }
                                else
                                    sign2=-sign2;
                            }
                            bit_a[v]=0;
                        }
                    }
                    bit_a[u]=1;
                    sign1=-sign1;
                }
            }
            bit_a[t]=1;
        }
        while(i_ext!=n_e2_a){
            e2_sign_a[i_CI*n_e2_a+i_ext] = 0;
            e2_V_a   [i_CI*n_e2_a+i_ext] = 0;
            e2_ind_a [i_CI*n_e2_a+i_ext] = e2_ind_a [i_CI*n_e2_a+i_ext-1];
            e2_orbs_a[i_CI*n_e2_a+i_ext] = e2_orbs_a[i_CI*n_e2_a+i_ext-1];
            i_ext++;
        }

//         fprintf(out_stream,"%d %d\n", i_ext, n_e2_a);
//         getchar();
        
    }
    
//     fprintf(out_stream,"%d %d\n", i_ext, n_e2_a*Na);
//     getchar();
        
    //BBBB
    for(int i_CI =0; i_CI<Nb; i_CI++){
        i_ext=0;
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[i_CI*(nb+1)+k]-1] = 1;
        
        for(int t=0  ;t<n_act;t++)if(bit_b[t]!=0){
            sign1=1.0;
            bit_b[t]=0;
            for(int u=t+1;u<n_act;u++){
                if(bit_b[u]!=0){
                    bit_b[u]=0;
                    for(int v=0  ;v<n_act;v++){
                        if(bit_b[v]==0){
                            sign2=1.0;
                            bit_b[v]=1;
                            for(int w=v+1;w<n_act;w++){
                                if(bit_b[w]==0){
                                    bit_b[w]=1;
//                                     if(t!=v)
//                                     if(t!=w)
//                                     if(u!=v)
//                                     if(u!=w){
                                        e2_orbs_b[i_CI*n_e2_b+i_ext] = ((t*n_act+u)*n_act+v)*n_act+w;
                                        e2_sign_b[i_CI*n_e2_b+i_ext] = sign1*sign2;
                                        e2_ind_b [i_CI*n_e2_b+i_ext] = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                                        e2_V_b   [i_CI*n_e2_b+i_ext] =(act_INTS_BB[((t*n_act+v)*n_act+u)*n_act+w]-
                                                                       act_INTS_BB[((t*n_act+w)*n_act+u)*n_act+v])*
                                                                       sign1*sign2;
                                        i_ext++;
//                                     }
                                    bit_b[w]=0;
                                }
                                else
                                    sign2=-sign2;
                            }
                            bit_b[v]=0;
                        }
                    }
                    bit_b[u]=1;
                    sign1=-sign1;
                }
            }
            bit_b[t]=1;
        }
        while(i_ext!=n_e2_b){
            e2_sign_b[i_CI*n_e2_b+i_ext] = 0;
            e2_V_b   [i_CI*n_e2_b+i_ext] = 0;
            e2_ind_b [i_CI*n_e2_b+i_ext] = e2_ind_b [i_CI*n_e2_b+i_ext-1];
            e2_orbs_b[i_CI*n_e2_b+i_ext] = e2_orbs_b[i_CI*n_e2_b+i_ext-1];
            i_ext++;
        }

    }
    
    //AAAAAA
    i_ext=0;
    for(int i_CI =0; i_CI<Na; i_CI++){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        //check symmetrization!!!!!!!!!
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        double t_sign=1;
        for(int t=0  ;t<n_act;t++){
            if(bit_a[t]!=0){
                bit_a[t]=0;
                sign1=t_sign;
                for(int u=t+1;u<n_act;u++)if(bit_a[u]!=0){
                    bit_a[u]=0;
                            sign3=sign1;//1;
                            for(int w=u+1/*0*/;w<n_act;w++)if(bit_a[w]!=0){
                                bit_a[w]=0;
                                
                                
                    sign2=1;
                    for(int v=0;v<n_act;v++){
                        if(bit_a[v]==0){
                            bit_a[v]=1;
                                for(int x=v+1/*0*/;x<n_act;x++)if(bit_a[x]==0){
                                    sign4=1;
                                    for(int y=x+1;y<n_act;y++){
                                        if(bit_a[y]==0){
                                            bit_a[x]=1;
                                            bit_a[y]=1;
                                            // change order from a--+++- to ---+++
                                            // regular order a+(v)a+(x)a+(y)a(w)a(u)a(t);v>x>y,t>u>v
                                            e3_ind_a [i_ext] = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                                            e3_V_a   [i_ext] =(-T3_AAA[((((t*n_act+u)*n_act+v)*n_act+y)*n_act+x)*n_act+w]//;
                                                               +T3_AAA[((((t*n_act+w)*n_act+v)*n_act+y)*n_act+x)*n_act+u]
                                                               -T3_AAA[((((u*n_act+w)*n_act+v)*n_act+y)*n_act+x)*n_act+t]
                                                               +T3_AAA[((((t*n_act+u)*n_act+x)*n_act+y)*n_act+v)*n_act+w]
                                                               -T3_AAA[((((t*n_act+w)*n_act+x)*n_act+y)*n_act+v)*n_act+u]
                                                               +T3_AAA[((((u*n_act+w)*n_act+x)*n_act+y)*n_act+v)*n_act+t]
                                                               -T3_AAA[((((t*n_act+u)*n_act+y)*n_act+x)*n_act+v)*n_act+w]
                                                               +T3_AAA[((((t*n_act+w)*n_act+y)*n_act+x)*n_act+v)*n_act+u]
                                                               -T3_AAA[((((u*n_act+w)*n_act+y)*n_act+x)*n_act+v)*n_act+t])*
                                                               t_sign*sign1*sign2*sign3*sign4;
                                            i_ext++;
                                            bit_a[x]=0;
                                            bit_a[y]=0;    
                                        }
                                        else
                                            sign4=-sign4;
                                    }
                                }
                            bit_a[v]=0;                            }
                        else
                            sign2=-sign2;
                    }
                                bit_a[w]=1;
                                sign3=-sign3;                                }
                    bit_a[u]=1;
                    sign1=-sign1;
                }
                bit_a[t]=1;
                t_sign=-t_sign;
            }
        }        
    }
    
    //BBBBBB
    i_ext=0;
    for(int i_CI =0; i_CI<Nb; i_CI++){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        //check symmetrization!!!!!!!!!
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[i_CI*(nb+1)+k]-1] = 1;
        
        double t_sign=1;
        for(int t=0  ;t<n_act;t++){
            if(bit_b[t]!=0){
                bit_b[t]=0;
                sign1=t_sign;
                for(int u=t+1;u<n_act;u++)if(bit_b[u]!=0){
                    bit_b[u]=0;
                            sign3=sign1;//1;
                            for(int w=u+1/*0*/;w<n_act;w++)if(bit_b[w]!=0){
                                bit_b[w]=0;
                                
                                
                    sign2=1;
                    for(int v=0;v<n_act;v++){
                        if(bit_b[v]==0){
                            bit_b[v]=1;
                                for(int x=v+1/*0*/;x<n_act;x++)if(bit_b[x]==0){
                                    sign4=1;
                                    for(int y=x+1;y<n_act;y++){
                                        if(bit_b[y]==0){
                                            bit_b[x]=1;
                                            bit_b[y]=1;
                                            // change order from a--+-++ to ---+++
                                            // canonic order a+(v)a+(x)a+(y)a(w)a(u)a(t);v>x>y,t>u>v
                                            e3_ind_b [i_ext] = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                                            e3_V_b   [i_ext] =(-T3_BBB[((((t*n_act+u)*n_act+v)*n_act+y)*n_act+x)*n_act+w]//;
                                                               +T3_BBB[((((t*n_act+w)*n_act+v)*n_act+y)*n_act+x)*n_act+u]
                                                               -T3_BBB[((((u*n_act+w)*n_act+v)*n_act+y)*n_act+x)*n_act+t]
                                                               +T3_BBB[((((t*n_act+u)*n_act+x)*n_act+y)*n_act+v)*n_act+w]
                                                               -T3_BBB[((((t*n_act+w)*n_act+x)*n_act+y)*n_act+v)*n_act+u]
                                                               +T3_BBB[((((u*n_act+w)*n_act+x)*n_act+y)*n_act+v)*n_act+t]
                                                               -T3_BBB[((((t*n_act+u)*n_act+y)*n_act+x)*n_act+v)*n_act+w]
                                                               +T3_BBB[((((t*n_act+w)*n_act+y)*n_act+x)*n_act+v)*n_act+u]
                                                               -T3_BBB[((((u*n_act+w)*n_act+y)*n_act+x)*n_act+v)*n_act+t])*
                                                               t_sign*sign1*sign2*sign3*sign4;
                                            i_ext++;
                                            bit_b[x]=0;
                                            bit_b[y]=0;    
                                        }
                                        else
                                            sign4=-sign4;
                                    }
                                }
                            bit_b[v]=0;                            }
                        else
                            sign2=-sign2;
                    }
                                bit_b[w]=1;
                                sign3=-sign3;                                }
                    bit_b[u]=1;
                    sign1=-sign1;
                }
                bit_b[t]=1;
                t_sign=-t_sign;
            }
        }        
    }
    
    
//     fprintf(out_stream,"bbb%d %d\n", i_ext, n_e3_b*Nb);
//     getchar();
    
    return 0;
    
}


int aldet_data::init_zero_vec(int n_s, int i_set){
    
    n_states[i_set] = n_s;
    
    if(coef    [i_set]==NULL) coef    [i_set] = new double[Nd*n_s];
    if(coef_bas[i_set]==NULL) coef_bas[i_set] = new double[Nd*n_s];
    if(coef_asb[i_set]==NULL) coef_asb[i_set] = new double[Nd*n_s];
    if(coef_bsa[i_set]==NULL) coef_bsa[i_set] = new double[Nd*n_s];
    if(E_states[i_set]==NULL) E_states[i_set] = new double[   n_s];
    if(S2      [i_set]==NULL) S2      [i_set] = new double[   n_s];
    if(L2      [i_set]==NULL) L2      [i_set] = new double[   n_s];
    if(P       [i_set]==NULL) P       [i_set] = new double[   n_s];
//     if(sym_ab  [i_set]==NULL) sym_ab  [i_set] = new double[   n_s];
    
    
    set_zero_matr(coef    [i_set], Nd*n_s);
    set_zero_matr(coef_bas[i_set], Nd*n_s);
    set_zero_matr(coef_asb[i_set], Nd*n_s);
    set_zero_matr(coef_bsa[i_set], Nd*n_s);
    
    
    return 0;
}


int aldet_data::copy_coef(int i_set, aldet_data * C, int n_s, int i_set_inp, int do_transpose){
    
    if(i_set>=n_sets){
        fprintf(out_stream,"unable to run read_set_from_mol for i_set = %d and n_sets = %d\nfor correct run you nedd^ i < n_sets\n", i_set, n_sets);
        exit(1);
    }
    if((n_act!=C->n_act)||(na!=C->na)||(nb!=C->nb)){
        fprintf(out_stream,"unable to copy aldet data with different active space\n");
        exit(1);
    }
    
    n_states[i_set] = n_s;
    if(C->coef[i_set_inp]!=NULL){
        if(coef    [i_set]==NULL) coef    [i_set] = new double[Nd*n_s];
        if(E_states[i_set]==NULL) E_states[i_set] = new double[   n_s];
        if(S2      [i_set]==NULL) S2      [i_set] = new double[   n_s];
        if(L2      [i_set]==NULL) L2      [i_set] = new double[   n_s];
        if(P       [i_set]==NULL) P       [i_set] = new double[   n_s];
        memcpy(coef    [i_set],C->coef    [i_set_inp],Nd*n_s*sizeof(double));
        memcpy(E_states[i_set],C->E_states[i_set_inp],   n_s*sizeof(double));
        
        if(do_transpose){
            if(coef_bas[i_set]==NULL) coef_bas[i_set] = new double[Nd*n_s];
            if(coef_asb[i_set]==NULL) coef_asb[i_set] = new double[Nd*n_s];
            if(coef_bsa[i_set]==NULL) coef_bsa[i_set] = new double[Nd*n_s];
            transpose_3d_abc_to_bac(coef_bas[i_set], coef[i_set], Na, Nb, n_s);
            transpose_3d_abc_to_acb(coef_asb[i_set], coef[i_set], Na, Nb, n_s);
            transpose_3d_abc_to_bca(coef_bsa[i_set], coef[i_set], Na, Nb, n_s);
        }
    }
    
    return 0;
}


int aldet_data::read_set_from_ci_map_arr(ci_map_arr * ci, int n_s, int i_set){
    
    if(i_set>=n_sets){
        fprintf(out_stream,"unable to run read_set_from_ci_map_arr for i_set = %d and n_sets = %d\nfor correct run you nedd: i < n_sets\n", i_set, n_sets);
        exit(1);
    }
    
    n_states[i_set] = n_s;
    if(coef    [i_set]==NULL) coef    [i_set] = new double[Nd*n_s];
    if(coef_bas[i_set]==NULL) coef_bas[i_set] = new double[Nd*n_s];
    if(coef_asb[i_set]==NULL) coef_asb[i_set] = new double[Nd*n_s];
    if(coef_bsa[i_set]==NULL) coef_bsa[i_set] = new double[Nd*n_s];
    if(E_states[i_set]==NULL) E_states[i_set] = new double[   n_s];
    if(S2      [i_set]==NULL) S2      [i_set] = new double[   n_s];
    if(L2      [i_set]==NULL) L2      [i_set] = new double[   n_s];
    if(P       [i_set]==NULL) P       [i_set] = new double[   n_s];
//     if(sym_ab  [i_set]==NULL) sym_ab  [i_set] = new double[   n_s];
    
    
    set_zero_matr(coef[i_set],Nd*n_s);
    
//     double * ci_tmp;
    for (const auto& d: *ci){
        auto ind_a = get_index(d.first, na, n_act, fa, buf, 0);
        auto ind_b = get_index(d.first, nb, n_act, fb, buf, CI_MAX_SPACE);
        double * c = d.second;
        for(int i=0;i<n_s;i++)
            coef[i_set][(ind_a*Nb + ind_b)*n_s+i] = c[i];
    }
    
//     printf_timer("before t");
    transpose_3d_abc_to_bac(coef_bas[i_set], coef[i_set], Na, Nb, n_s);
    transpose_3d_abc_to_acb(coef_asb[i_set], coef[i_set], Na, Nb, n_s);
    transpose_3d_abc_to_bca(coef_bsa[i_set], coef[i_set], Na, Nb, n_s);
    
//     printf_timer("after t");
    
    return 0;
}

int aldet_data::write_civec(int i_s, char *out_name){
    
    FILE * out;
    out=fopen(out_name,"wb");
    
    int a=0;
    for(int i=0;i<8;i++)fwrite(&a,4,1,out);
    fwrite(&(n_states[i_s]),4,1,out);
    fwrite(&Nd,4,1,out);
    
    double * buf = new double[Na*Nb];
    
    for(int i=0;i<n_states[i_s];i++){
        for(int j=0;j<8;j++)fwrite(&a,4,1,out);
        for(int j=0;j<Nd;j++)buf[j]=coef[i_s][j*n_states[i_s]+i];
        
        fwrite(buf, 8, (size_t)(Nd), out);
        
    }
    
    fclose(out);
    
    delete[] buf;
    
    return 1;
}
int aldet_data::read_civec(int i_set, char * file_name, std::vector<int> n_st){
    
    FILE *f=fopen(file_name,"rb");
    if(f==NULL){
        fprintf(out_stream,"couldn't open file with CI wavefunctions %s \n",file_name);
        exit(1);
    }
    int Nst;
    int Ndet;

    fseek(f,32,SEEK_CUR);
    fread(&Nst, 4, 1, f);
    fread(&Ndet, 4, 1, f);
    n_states[i_set]=n_st.size();
//     fprintf(out_stream,"Nst = %d\n", Nst);
//     fprintf(out_stream,"Nst = %d\n",n_states[i_set]);
//     fprintf(out_stream,"Ndet = %d\n", Ndet);
//     fprintf(out_stream,"Ndet = %d\n", Nd);
    
//     getchar();
    
    if(coef    [i_set]==NULL)coef    [i_set]=new double[n_states[i_set]*Nd];
    if(coef_bas[i_set]==NULL)coef_bas[i_set]=new double[n_states[i_set]*Nd];
    if(coef_asb[i_set]==NULL)coef_asb[i_set]=new double[n_states[i_set]*Nd];
    if(coef_bsa[i_set]==NULL)coef_bsa[i_set]=new double[n_states[i_set]*Nd];
    
    double * data = new double [Ndet];
    int i_st=0;
    for(int i=0; i<Nst;i++){
        fseek(f,32,SEEK_CUR);
        fread(data, 8, (size_t)(Ndet), f);
        for(const auto &a:n_st)if(a==i){
            for(int i_CI=0;i_CI<Nd;i_CI++)
                coef[i_set][i_CI*n_states[i_set]+i_st]=data[i_CI];
            i_st++;
        }
    }
    
    delete[] data;

    return 0;
    
}



int aldet_data::transpose_ci(int i_set){

    if(i_set>=n_sets){
        fprintf(out_stream,"unable to run transpose_ci for i_set = %d and n_sets = %d\nfor correct run you nedd^ i < n_sets\n", i_set, n_sets);
        exit(1);
    }
    
    transpose_3d_abc_to_bac(coef_bas[i_set], coef[i_set], Na, Nb, n_states[i_set]);
    transpose_3d_abc_to_acb(coef_asb[i_set], coef[i_set], Na, Nb, n_states[i_set]);
    transpose_3d_abc_to_bca(coef_bsa[i_set], coef[i_set], Na, Nb, n_states[i_set]);
    
    return 0;
    
}


int aldet_data::calc_S(double * O, int a, int b){
    
//     fprintf(out_stream,"%d %d\n",n_states[a],n_states[b]);
//     getchar();
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n_states[a],n_states[b],Nd,1.0,
                         coef[a],n_states[a],
                         coef[b],n_states[b],0.0,
                         O      ,n_states[b]);//check it!!!!
    
    return 0;
}

int average_DM_aldet_diag(double * G_out, double * G, std::vector<double> avecoe,int na_p, int n_s){
    
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
        G_out[j]=G[j]/(norm);
    
//     fprintf(out_stream,"norm = %e\n",norm);
    return 0;
}

int aldet_data::calc_IPEA_single_AB(double * U_IP_A, double * H_IP_A, 
                                    double * U_EA_A, double * H_EA_A,
                                    double * U_IP_B, double * H_IP_B, 
                                    double * U_EA_B, double * H_EA_B,
                                    int a, std::vector<double> avecoe){
    
    int n_s = n_states[a];
    printf("WARNING: calc_IPEA_single_AB is not tested!!!!!!\n");
    //U_IP
    double * gamma = new double[n_s*n_act*n_act];
    set_zero_matr(gamma,n_act*n_act*n_s);
//     calc_DM(gamma, coef[a], coef[a], n_states[a], n_states[a], na, Na, Nb, fa, fb, vec_a, bit_a);
    calc_DMA_diag(gamma,a);
    average_DM_aldet_diag(U_IP_A,gamma, avecoe,n_act*n_act,n_s);
    
    set_zero_matr(gamma,n_act*n_act*n_s);
    calc_DMB_diag(gamma,a);
    average_DM_aldet_diag(U_IP_B,gamma, avecoe,n_act*n_act,n_s);
    
    //U_EA
    for(int i=0;i<n_act*n_act;i++)U_EA_A[i]=-U_IP_A[i];
    for(int i=0;i<n_act;i++)U_EA_A[i*(n_act+1)]=1.0+U_EA_A[i*(n_act+1)];

    for(int i=0;i<n_act*n_act;i++)U_EA_B[i]=-U_IP_B[i];
    for(int i=0;i<n_act;i++)U_EA_B[i*(n_act+1)]=1.0+U_EA_B[i*(n_act+1)];
    
    
    double * GAMMA_AA = new double[n_s*n_act*n_act*n_act*n_act];
    double * GAMMA_BB = new double[n_s*n_act*n_act*n_act*n_act];
    set_zero_matr(GAMMA_AA,n_s*n_act*n_act*n_act*n_act);
    set_zero_matr(GAMMA_BB,n_s*n_act*n_act*n_act*n_act);
    aldet_calc_DM_2body_AA_diag(GAMMA_AA, n_states[a], n_states[a], coef    [a], n_act, na, Na, Nb, fa, vec_a, 0, 1);
    aldet_calc_DM_2body_AA_diag(GAMMA_BB, n_states[a], n_states[a], coef_bas[0], n_act, nb, Nb, Na, fb, vec_b, 0, 1);

    average_DM_aldet_diag(GAMMA_AA,GAMMA_AA,avecoe,n_act*n_act*n_act*n_act,n_s);
    average_DM_aldet_diag(GAMMA_BB,GAMMA_BB,avecoe,n_act*n_act*n_act*n_act,n_s);
    
    double * GAMMA_EA_AA = new double[n_act*n_act*n_act*n_act];
    double * GAMMA_EA_BB = new double[n_act*n_act*n_act*n_act];
    
    for(int t=0;t<n_act;t++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++){
        GAMMA_EA_AA[((v*n_act+x)*n_act+w)*n_act+t]=GAMMA_AA[((v*n_act+x)*n_act+w)*n_act+t];
        if(t==v)GAMMA_EA_AA[((v*n_act+x)*n_act+w)*n_act+t]+=U_IP_A[x*n_act+w];
        if(t==w)GAMMA_EA_AA[((v*n_act+x)*n_act+w)*n_act+t]-=U_IP_A[x*n_act+v];
    }
    for(int t=0;t<n_act;t++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++){
        GAMMA_EA_BB[((v*n_act+x)*n_act+w)*n_act+t]=GAMMA_BB[((v*n_act+x)*n_act+w)*n_act+t];
        if(t==v)GAMMA_EA_BB[((v*n_act+x)*n_act+w)*n_act+t]+=U_IP_B[x*n_act+w];
        if(t==w)GAMMA_EA_BB[((v*n_act+x)*n_act+w)*n_act+t]-=U_IP_B[x*n_act+v];
    }
    
    double * GAMMA_AB = new double[n_s*n_act*n_act*n_act*n_act];
    set_zero_matr(GAMMA_AB,n_s*n_act*n_act*n_act*n_act);
    aldet_calc_DM_2body_AB_diag(GAMMA_AB, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
    average_DM_aldet_diag(GAMMA_AB,GAMMA_AB,avecoe,n_act*n_act*n_act*n_act,n_s);
    
    
    double * GAMMA_EA_AB = new double[n_act*n_act*n_act*n_act];
    
    for(int t=0;t<n_act;t++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++){
        GAMMA_EA_AB[((v*n_act+t)*n_act+x)*n_act+w]=-GAMMA_AB[((v*n_act+t)*n_act+x)*n_act+w];
        if(t==v)GAMMA_EA_AB[((v*n_act+t)*n_act+x)*n_act+w]+=U_IP_B[x*n_act+w]*2;//????
    }
    
    double * GAMMA_EA_BA = new double[n_act*n_act*n_act*n_act];
    
    for(int t=0;t<n_act;t++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++){
        GAMMA_EA_BA[((v*n_act+t)*n_act+x)*n_act+w]=-GAMMA_AB[((v*n_act+t)*n_act+x)*n_act+w];
        if(t==v)GAMMA_EA_BA[((v*n_act+t)*n_act+x)*n_act+w]+=U_IP_A[x*n_act+w]*2;//????
    }
    
    //H_IP
    set_zero_matr(H_IP_A,n_act*n_act);
    set_zero_matr(H_IP_B,n_act*n_act);

    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int w=0;w<n_act;w++)
        H_IP_A[t*n_act+u]+= U_IP_A[t*n_act+w]*F_act_A[u*n_act+w];
    
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int x=0;x<n_act;x++)
    for(int y=0;y<n_act;y++)
        H_IP_A[t*n_act+u]+=GAMMA_AA[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
                          (act_INTS_AA[((u*n_act+y)*n_act+x)*n_act+v]-
                           act_INTS_AA[((v*n_act+y)*n_act+x)*n_act+u]);
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int x=0;x<n_act;x++)
    for(int y=0;y<n_act;y++)
        H_IP_A[t*n_act+u]+= GAMMA_AB[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
                            act_INTS_AB[((u*n_act+y)*n_act+v)*n_act+x];
   
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int w=0;w<n_act;w++)
        H_IP_B[t*n_act+u]+= U_IP_B[t*n_act+w]*F_act_B[u*n_act+w];
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int x=0;x<n_act;x++)
    for(int y=0;y<n_act;y++)
        H_IP_B[t*n_act+u]+=GAMMA_BB[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
                          (act_INTS_BB[((u*n_act+y)*n_act+x)*n_act+v]-
                           act_INTS_BB[((v*n_act+y)*n_act+x)*n_act+u]);
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int x=0;x<n_act;x++)
    for(int y=0;y<n_act;y++)
        H_IP_B[t*n_act+u]+= GAMMA_AB[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
                            act_INTS_AB[((u*n_act+y)*n_act+v)*n_act+x];
   
    
// //     fprintf(out_stream,"H:\n");
// //     PrintMatr(H_IP,n_act,n_act,0);
//     
//     
    //H_EA
    set_zero_matr(H_EA_A,n_act*n_act);
    set_zero_matr(H_EA_B,n_act*n_act);

    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
        H_EA_A[t*n_act+u]+= U_EA_A[t*n_act+v]*F_act_A[u*n_act+v];

    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++)
        H_EA_A[t*n_act+u]+=GAMMA_EA_AA[((v*n_act+x)*n_act+w)*n_act+t]*0.5*
                           (act_INTS_AA[((v*n_act+u)*n_act+w)*n_act+x]-
                            act_INTS_AA[((v*n_act+x)*n_act+w)*n_act+u]);
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++)
        H_EA_A[t*n_act+u]+=GAMMA_EA_AB[((v*n_act+t)*n_act+x)*n_act+w]*0.5*
                            act_INTS_AB[((v*n_act+u)*n_act+w)*n_act+x];
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
        H_EA_B[t*n_act+u]+= U_EA_B[t*n_act+v]*F_act_B[u*n_act+v];
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++)
        H_EA_B[t*n_act+u]+=GAMMA_EA_BB[((v*n_act+x)*n_act+w)*n_act+t]*0.5*
                           (act_INTS_BB[((v*n_act+u)*n_act+w)*n_act+x]-
                            act_INTS_BB[((v*n_act+x)*n_act+w)*n_act+u]);
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++)
        H_EA_B[t*n_act+u]+=GAMMA_EA_BA[((v*n_act+t)*n_act+x)*n_act+w]*0.5*
                            act_INTS_AB[((v*n_act+u)*n_act+w)*n_act+x];///BA ?????
      
//     
// //     PrintMatr(H_EA,n_act,n_act,0);
//     
//     
//     printf_timer("calculation of IPEA matrices");
//     delete[] GAMMA;
//     delete[] GAMMA_EA;
//     delete[] GAMMA2;
//     delete[] GAMMA2_EA;
//     delete[] gamma;
    
    return 0;
}




int aldet_data::calc_IPEA_single(double * U_IP, double * H_IP, 
                                double * U_EA, double * H_EA,
                                int a, std::vector<double> avecoe){
    
    int n_s = n_states[a];
    
    //U_IP
    double * gamma = new double[n_s*n_act*n_act];
    set_zero_matr(gamma,n_act*n_act*n_s);
//     calc_DM(gamma, coef[a], coef[a], n_states[a], n_states[a], na, Na, Nb, fa, fb, vec_a, bit_a);
    calc_DM_diag(gamma,a);
    for(int i=0;i<n_s*n_act*n_act;i++)gamma[i]=gamma[i]*0.5;
    average_DM_aldet_diag(U_IP,gamma, avecoe,n_act*n_act,n_s);
    
    //U_EA
    for(int i=0;i<n_act*n_act;i++)U_EA[i]=-U_IP[i];
    for(int i=0;i<n_act;i++)U_EA[i*(n_act+1)]=1.0+U_EA[i*(n_act+1)];
    
    
    double * GAMMA = new double[n_s*n_act*n_act*n_act*n_act];
    set_zero_matr(GAMMA,n_s*n_act*n_act*n_act*n_act);
    aldet_calc_DM_2body_AA_diag(GAMMA, n_states[a], n_states[a], coef    [a], n_act, na, Na, Nb, fa, vec_a, 0, 1);
    aldet_calc_DM_2body_AA_diag(GAMMA, n_states[a], n_states[a], coef_bas[0], n_act, nb, Nb, Na, fb, vec_b, 0, 1);
    for(int i=0;i<n_s*n_act*n_act*n_act*n_act;i++)GAMMA[i]=GAMMA[i]*0.5;
    average_DM_aldet_diag(GAMMA,GAMMA,avecoe,n_act*n_act*n_act*n_act,n_s);
    
    
    double * GAMMA2 = new double[n_s*n_act*n_act*n_act*n_act];
    set_zero_matr(GAMMA2,n_s*n_act*n_act*n_act*n_act);
    aldet_calc_DM_2body_AB_diag(GAMMA2, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
    average_DM_aldet_diag(GAMMA2,GAMMA2,avecoe,n_act*n_act*n_act*n_act,n_s);
    
    
    // double * GAMMA3 = new double[n_s*n_act*n_act*n_act*n_act];
    // for(int i=0; i<n_act*n_act*n_act*n_act;i++)GAMMA3[i]=2*GAMMA[i]+GAMMA2[i];
    
    // PrintMatr(GAMMA3, n_act*n_act, n_act*n_act, 0);
    // exit(0);
    
    
    //H_IP
    set_zero_matr(H_IP,n_act*n_act);

    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int w=0;w<n_act;w++)
        H_IP[t*n_act+u]+= U_IP[t*n_act+w]*F_act_A[u*n_act+w];//unrestricted variant
    
    
    
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int x=0;x<n_act;x++)
    for(int y=0;y<n_act;y++)
        H_IP[t*n_act+u]+=GAMMA[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
                      (act_INTS_AA[((u*n_act+y)*n_act+x)*n_act+v]-
                       act_INTS_AA[((v*n_act+y)*n_act+x)*n_act+u]);//unrestricted variant
    
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int x=0;x<n_act;x++)
    for(int y=0;y<n_act;y++)
        H_IP[t*n_act+u]+= GAMMA2[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
                         act_INTS_AB[((u*n_act+y)*n_act+v)*n_act+x];//unrestricted variant
    
    // fprintf(out_stream,"H_IP:\n");
    // PrintMatr(H_IP, n_act, n_act, 0);
    
//     PrintMatr(H_IP,n_act,n_act,0);
    
    
    //H_EA
    set_zero_matr(H_EA,n_act*n_act);

    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
        H_EA[t*n_act+u]+= U_EA[t*n_act+v]*F_act_A[u*n_act+v];//unrestricted variant
    
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++){
        H_EA[t*n_act+u]+=GAMMA[((v*n_act+x)*n_act+w)*n_act+t]*0.5*
                      (act_INTS_AA[((v*n_act+u)*n_act+w)*n_act+x]-
                       act_INTS_AA[((v*n_act+x)*n_act+w)*n_act+u]);//unrestricted variant
    }
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int x=0;x<n_act;x++){
        H_EA[t*n_act+u]+=U_IP[x*n_act+v]*(2*act_INTS_AA[((t*n_act+u)*n_act+v)*n_act+x]-act_INTS_AA[((t*n_act+x)*n_act+v)*n_act+u]);
    }
    
    
    for(int t=0;t<n_act;t++)
    for(int u=0;u<n_act;u++)
    for(int v=0;v<n_act;v++)
    for(int w=0;w<n_act;w++)
    for(int x=0;x<n_act;x++){
        H_EA[t*n_act+u]-=GAMMA2[((v*n_act+t)*n_act+x)*n_act+w]*0.5*( act_INTS_AB[((v*n_act+u)*n_act+w)*n_act+x]);//unrestricted variant
    }
    
   
    // fprintf(out_stream,"H_EA:\n");
    // PrintMatr(H_EA,n_act,n_act,0);
    // exit(0);
    
    printf_timer("calculation of IPEA matrices");
    delete[] GAMMA;
    // delete[] GAMMA_EA;
    delete[] GAMMA2;
    delete[] gamma;
    
    return 0;
}

int aldet_data::calc_IPEA_single_CA(double * U_IP, double * H_IP, 
                                    double * U_EA, double * H_EA,
                                    int n_core, int n_mo, int ld,
                                    int a, std::vector<double> avecoe,
                                    double * H_core, double * aaag_ints){
    
    printf("calc_IPEA_single_CA is disabled since version 2.9.8\n");
    exit(0);

    
//     int n_s  = n_states[a];
//     int n_ext= n_mo-n_core-n_act;
//     int n_h  = n_ext+n_act;
//     
//     //U_IP
//     double * gamma = new double[n_s*n_act*n_s*n_act];
//     set_zero_matr(gamma,n_act*n_act*n_s*n_s);
// //     calc_DM(gamma, coef[a], coef[a], n_states[a], n_states[a], na, Na, Nb, fa, fb, vec_a, bit_a);
//     calc_DMA(gamma,a,a);
//     calc_DMB(gamma,a,a);
//     for(int i=0;i<n_s*n_act*n_s*n_act;i++)gamma[i]=gamma[i]*0.5;
//     average_DM_aldet(U_IP,gamma, avecoe,n_act*n_act,n_s);
//     
//     //U_EA
//     for(int i=0;i<n_act*n_act;i++)U_EA[i]=-U_IP[i];
//     for(int i=0;i<n_act;i++)U_EA[i*(n_act+1)]=1.0+U_EA[i*(n_act+1)];
//     
//     
//     double * GAMMA = new double[n_s*n_s*n_act*n_act*n_act*n_act];
//     set_zero_matr(GAMMA,n_s*n_s*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_2body_AA(GAMMA, n_states[a], n_states[a], coef    [a], n_act, na, Na, Nb, fa, vec_a, 0, 1);
//     aldet_calc_DM_2body_AA(GAMMA, n_states[a], n_states[a], coef_bas[0], n_act, nb, Nb, Na, fb, vec_b, 0, 1);
//     for(int i=0;i<n_s*n_s*n_act*n_act*n_act*n_act;i++)GAMMA[i]=GAMMA[i]*0.5;
//     average_DM_aldet(GAMMA,GAMMA,avecoe,n_act*n_act*n_act*n_act,n_s);
//     
//     double * GAMMA_EA = new double[n_act*n_act*n_act*n_act];
//     
//     for(int t=0;t<n_act;t++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++){
//         GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]=GAMMA[((v*n_act+x)*n_act+w)*n_act+t];
//         if(t==v)GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]+=U_IP[x*n_act+w];
//         if(t==w)GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]-=U_IP[x*n_act+v];
//     }
//     
//     double * GAMMA2 = new double[n_s*n_s*n_act*n_act*n_act*n_act];
//     set_zero_matr(GAMMA2,n_s*n_s*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_2body_AB(GAMMA2, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
//     average_DM_aldet(GAMMA2,GAMMA2,avecoe,n_act*n_act*n_act*n_act,n_s);
//     
//     
//     double * GAMMA2_EA = new double[n_act*n_act*n_act*n_act];
//     
//     for(int t=0;t<n_act;t++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++){
//         GAMMA2_EA[((v*n_act+t)*n_act+x)*n_act+w]=-GAMMA2[((v*n_act+t)*n_act+x)*n_act+w];
//         if(t==v)GAMMA2_EA[((v*n_act+t)*n_act+x)*n_act+w]+=U_IP[x*n_act+w]*2.0;
//     }
//     
//     //H_IP
//     set_zero_matr(H_IP,(n_act+n_core)*(n_act+n_core));
// 
//     for(int t=0;t<n_act       ;t++)
//     for(int u=0;u<n_act+n_core;u++)
//     for(int w=0;w<n_act       ;w++)
//         H_IP[t*(n_act+n_core)+u]+= U_IP[t*n_act+w]*H_core[u*ld+w+n_core];
//     
//     
//     for(int t=0;t<n_act       ;t++)
//     for(int u=0;u<n_act+n_core;u++)
//     for(int v=0;v<n_act       ;v++)
//     for(int x=0;x<n_act       ;x++)
//     for(int y=0;y<n_act       ;y++)
//         H_IP[t*(n_act+n_core)+u]+=GAMMA[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
//                       (aaag_ints[((v*n_act+x)*n_act+y)*ld+u]-
//                        aaag_ints[((v*n_act+y)*n_act+x)*ld+u]);
//     
//     
//     for(int t=0;t<n_act       ;t++)
//     for(int u=0;u<n_act+n_core;u++)
//     for(int v=0;v<n_act       ;v++)
//     for(int x=0;x<n_act       ;x++)
//     for(int y=0;y<n_act       ;y++)
//         H_IP[t*(n_act+n_core)+u]+= GAMMA2[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
//                          aaag_ints[((v*n_act+x)*n_act+y)*ld+u];
//    
// //     fprintf(out_stream,"H:\n");
// //     PrintMatr(H_IP,n_act,n_act,0);
//     
//     
//     //H_EA
//     set_zero_matr(H_EA,n_h*n_h);
// 
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_h  ;u++)
//     for(int v=0;v<n_act;v++)
//         H_EA[t*n_h+u]+= U_EA[t*n_act+v]*H_core[(u+n_core)*ld+v+n_core];;
//     
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_h  ;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//         H_EA[t*n_h+u]+=GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]*0.5*
//                       (aaag_ints[((x*n_act+w)*n_act+v)*ld+u+n_core]-
//                        aaag_ints[((x*n_act+v)*n_act+w)*ld+u+n_core]);
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_h  ;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//         H_EA[t*n_h+u]+=GAMMA2_EA[((v*n_act+t)*n_act+x)*n_act+w]*0.5*
//                       ( aaag_ints[((x*n_act+w)*n_act+v)*ld+u+n_core]);
//     
//    
//     
//     PrintMatr(H_EA,n_act,n_h,0);
//     
// //     for(int t=0;t<n_act;t++)
// //     for(int u=0;u<n_ext;u++)
// //         if(fabs(H_EA[t*n_h+u+n_act])>1e-4)printf("av[%d,%d] = %e\n",t,u,H_EA[t*n_h+u+n_act]);
//     
//     exit(0);
//     
//     printf_timer("IPEA");
//     delete[] GAMMA;
//     delete[] GAMMA_EA;
//     delete[] GAMMA2;
//     delete[] GAMMA2_EA;
//     delete[] gamma;
    
    return 0;
}



int aldet_data::calc_IPEA_double(double * U_IP, double * H_IP, 
                                 double * U_EA, double * H_EA,
                                 double * U_IP2, double * H_IP2,
                                 double * U_EA2, double * H_EA2,
                                 double * U_IP_AB, double * H_IP_AB,
                                 double * U_EA_AB, double * H_EA_AB,
                                 int a, std::vector<double> avecoe){
    
    printf("calc_IPEA_double is disabled since version 2.9.8\n");
    exit(0);
    
//     int n_s = n_states[a];
//     
//     //U_IP
//     double * gamma = new double[n_s*n_act*n_s*n_act];
//     set_zero_matr(gamma,n_act*n_act*n_s*n_s);
//     calc_DM_part(gamma, coef[a], coef[a], n_states[a], n_states[a], na, Na, Nb, fa, fb, vec_a, bit_a);
//     average_DM_aldet(U_IP,gamma, avecoe,n_act*n_act,n_s);
//     
//     //U_EA
//     for(int i=0;i<n_act*n_act;i++)U_EA[i]=-U_IP[i];
//     for(int i=0;i<n_act;i++)U_EA[i*(n_act+1)]=1.0+U_EA[i*(n_act+1)];
//     
//     
//     double * GAMMA = new double[n_s*n_s*n_act*n_act*n_act*n_act];
//     set_zero_matr(GAMMA,n_s*n_s*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_2body_AA(GAMMA, n_states[a], n_states[a], coef    [a], n_act, na, Na, Nb, fa, vec_a, 0, 1);
//     average_DM_aldet(GAMMA,GAMMA,avecoe,n_act*n_act*n_act*n_act,n_s);
//     
//     double * GAMMA_EA = new double[n_act*n_act*n_act*n_act];
//     
//     for(int t=0;t<n_act;t++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++){
//         GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]=GAMMA[((v*n_act+x)*n_act+w)*n_act+t];
//         if(t==v)GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]+=U_IP[x*n_act+w];
//         if(t==w)GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]-=U_IP[x*n_act+v];
//     }
//     
//     double * GAMMA2 = new double[n_s*n_s*n_act*n_act*n_act*n_act];
//     set_zero_matr(GAMMA2,n_s*n_s*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_2body_AB(GAMMA2, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
//     average_DM_aldet(GAMMA2,GAMMA2,avecoe,n_act*n_act*n_act*n_act,n_s);
//     
//     
//     double * GAMMA2_EA = new double[n_act*n_act*n_act*n_act];
//     
//     for(int t=0;t<n_act;t++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++){
//         GAMMA2_EA[((v*n_act+t)*n_act+x)*n_act+w]=-GAMMA2[((v*n_act+t)*n_act+x)*n_act+w];
//         if(t==v)GAMMA2_EA[((v*n_act+t)*n_act+x)*n_act+w]+=U_IP[x*n_act+w]*2.0;
//     }
//     
//     double * G3_AAA = new double[n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act];
//     set_zero_matr(G3_AAA,n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_3body_AAA(G3_AAA, n_states[a], n_states[a], coef    [a], n_act, na, Na, Nb, fa, vec_a, 0, 1);
//     average_DM_aldet(G3_AAA,G3_AAA,avecoe,n_act*n_act*n_act*n_act*n_act*n_act,n_s);
//     
//     
//     double * G3_AAB = new double[n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act];
//     set_zero_matr(G3_AAB,n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_3body_AAB(G3_AAB, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
//     average_DM_aldet(G3_AAB,G3_AAB,avecoe,n_act*n_act*n_act*n_act*n_act*n_act,n_s);
// 
//     //H_IP
//     set_zero_matr(H_IP,n_act*n_act);
// 
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int w=0;w<n_act;w++)
//         H_IP[t*n_act+u]+= U_IP[t*n_act+w]*F_act_A[u*n_act+w];//unrestricted variant
//     
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//         H_IP[t*n_act+u]+=GAMMA[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
//                       (act_INTS_AA[((u*n_act+y)*n_act+x)*n_act+v]-
//                        act_INTS_AA[((v*n_act+y)*n_act+x)*n_act+u]);//unrestricted variant
//     
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//         H_IP[t*n_act+u]+= GAMMA2[((t*n_act+y)*n_act+v)*n_act+x]*0.5*
//                          act_INTS_AB[((u*n_act+y)*n_act+v)*n_act+x];//unrestricted variant
//    
// //     fprintf(out_stream,"H:\n");
// //     PrintMatr(H_IP,n_act,n_act,0);
//     
//     
//     //H_EA
//     set_zero_matr(H_EA,n_act*n_act);
// 
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//         H_EA[t*n_act+u]+= U_EA[t*n_act+v]*F_act_A[u*n_act+v];//unrestricted variant
//     
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//         H_EA[t*n_act+u]+=GAMMA_EA[((v*n_act+x)*n_act+w)*n_act+t]*0.5*
//                       (act_INTS_AA[((v*n_act+u)*n_act+w)*n_act+x]-
//                        act_INTS_AA[((v*n_act+x)*n_act+w)*n_act+u]);//unrestricted variant
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//         H_EA[t*n_act+u]+=GAMMA2_EA[((v*n_act+t)*n_act+x)*n_act+w]*0.5*
//                       ( act_INTS_AB[((v*n_act+u)*n_act+w)*n_act+x]);//unrestricted variant
//     
//     //U_IP_2
//     for(int i=0, a=0;i<n_act;i++)
//     for(int j=i+1   ;j<n_act;j++,a++)
//     for(int k=0, b=0;k<n_act;k++)
//     for(int l=k+1   ;l<n_act;l++,b++){
//         U_IP2[a*n_act*(n_act-1)/2+b]=GAMMA[((i*n_act+k)*n_act+j)*n_act+l];
//     }
//     
//     //H_IP_2
//     set_zero_matr(H_IP2, n_act*n_act*(n_act-1)*(n_act-1)/4);
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int y=0;y<n_act;y++)
//         H_IP2[a*n_act*(n_act-1)/2+b]+= F_act_A[v*n_act+y]*GAMMA[((t*n_act+y)*n_act+u)*n_act+w]
//                                       -F_act_A[w*n_act+y]*GAMMA[((t*n_act+y)*n_act+u)*n_act+v];//unrestricted variant
//                                       
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int x=0;x<n_act;x++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP2[a*n_act*(n_act-1)/2+b]-=(act_INTS_AA[((v*n_act+g)*n_act+x)*n_act+z]
//                                       -act_INTS_AA[((x*n_act+g)*n_act+v)*n_act+z])*
//                                        G3_AAA[((((t*n_act+u)*n_act+x)*n_act+w)*n_act+z)*n_act+g]*0.5;//unrestricted variant
//         
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP2[a*n_act*(n_act-1)/2+b]+=(act_INTS_AA[((v*n_act+g)*n_act+w)*n_act+z]
//                                       -act_INTS_AA[((w*n_act+g)*n_act+v)*n_act+z])*
//                                        GAMMA[  ((t*n_act+g)*n_act+u)*n_act+z]*0.5;//unrestricted variant
//     
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int x=0;x<n_act;x++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP2[a*n_act*(n_act-1)/2+b]+=(act_INTS_AA[((w*n_act+g)*n_act+x)*n_act+z]
//                                       -act_INTS_AA[((x*n_act+g)*n_act+w)*n_act+z])*
//                                        G3_AAA[((((t*n_act+u)*n_act+x)*n_act+v)*n_act+z)*n_act+g]*0.5;//unrestricted variant
//         
// 
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP2[a*n_act*(n_act-1)/2+b]+= act_INTS_AB[((v*n_act+g)*n_act+y)*n_act+z]*
//                                        G3_AAB[((((t*n_act+u)*n_act+y)*n_act+w)*n_act+g)*n_act+z];//unrestricted variant
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP2[a*n_act*(n_act-1)/2+b]-= act_INTS_AA[((w*n_act+g)*n_act+y)*n_act+z]*
//                                        G3_AAB[((((t*n_act+u)*n_act+y)*n_act+v)*n_act+g)*n_act+z];//unrestricted variant
//                                        
//     //U_EA_2
//     for(int i=0, a=0;i<n_act;i++)
//     for(int j=i+1   ;j<n_act;j++,a++)
//     for(int k=0, b=0;k<n_act;k++)
//     for(int l=k+1   ;l<n_act;l++,b++){
//         U_EA2[a*n_act*(n_act-1)/2+b]=GAMMA[((k*n_act+i)*n_act+l)*n_act+j];
//         if(j==k)U_EA2[a*n_act*(n_act-1)/2+b]-=U_EA[i*n_act+l];
//         if(i==k)U_EA2[a*n_act*(n_act-1)/2+b]+=U_EA[j*n_act+l];
//         if(j==l)U_EA2[a*n_act*(n_act-1)/2+b]-=U_IP[k*n_act+i];
//         if(i==l)U_EA2[a*n_act*(n_act-1)/2+b]+=U_IP[k*n_act+j];
//     }    
//     
//     
//     //H_EA_2
//     double * GAMMA_EA_full = new double[n_act*n_act*n_act*n_act];
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++){
//         GAMMA_EA_full[((x*n_act+u)*n_act+w)*n_act+t]=GAMMA[((x*n_act+u)*n_act+w)*n_act+t];
//         if(x==u)GAMMA_EA_full[((x*n_act+u)*n_act+w)*n_act+t]+=U_EA[t*n_act+w];
//         if(x==t)GAMMA_EA_full[((x*n_act+u)*n_act+w)*n_act+t]-=U_EA[u*n_act+w];
//         if(w==u)GAMMA_EA_full[((x*n_act+u)*n_act+w)*n_act+t]+=U_IP[t*n_act+x];
//         if(w==t)GAMMA_EA_full[((x*n_act+u)*n_act+w)*n_act+t]-=U_IP[u*n_act+x];
//     }
//     
//     set_zero_matr(H_EA2, n_act*n_act*(n_act-1)*(n_act-1)/4);
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int x=0;x<n_act;x++)
//         H_EA2[a*n_act*(n_act-1)/2+b]+= F_act_A[w*n_act+x]*GAMMA_EA_full[((x*n_act+u)*n_act+v)*n_act+t]
//                                       -F_act_A[v*n_act+x]*GAMMA_EA_full[((x*n_act+u)*n_act+w)*n_act+t];//unrestricted variant
//                                       
//     set_zero_matr(G3_AAA,n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_3body_AAA_1(G3_AAA, n_states[a], n_states[a], coef    [a], n_act, na, Na, Nb, fa, vec_a, 0, 1);
//     average_DM_aldet(G3_AAA,G3_AAA,avecoe,n_act*n_act*n_act*n_act*n_act*n_act,n_s);
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//         H_EA2[a*n_act*(n_act-1)/2+b]+=(act_INTS_AA[((x*n_act+w)*n_act+y)*n_act+z]
//                                       -act_INTS_AA[((x*n_act+z)*n_act+y)*n_act+w])*
//                                        G3_AAA[((((t*n_act+u)*n_act+x)*n_act+y)*n_act+z)*n_act+v]*0.5;//unrestricted variant
//                                        
//                                        
//     set_zero_matr(G3_AAA,n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_3body_AAA_2(G3_AAA, n_states[a], n_states[a], coef    [a], n_act, na, Na, Nb, fa, vec_a, 0, 1);
//     average_DM_aldet(G3_AAA,G3_AAA,avecoe,n_act*n_act*n_act*n_act*n_act*n_act,n_s);
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//         H_EA2[a*n_act*(n_act-1)/2+b]+=(act_INTS_AA[((x*n_act+v)*n_act+y)*n_act+z]
//                                       -act_INTS_AA[((x*n_act+z)*n_act+y)*n_act+v])*
//                                        G3_AAA[((((t*n_act+u)*n_act+x)*n_act+y)*n_act+w)*n_act+z]*0.5;//unrestricted variant
//     
//                                        
//     set_zero_matr(G3_AAB,n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_3body_AAB_2(G3_AAB, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
//     average_DM_aldet(G3_AAB,G3_AAB,avecoe,n_act*n_act*n_act*n_act*n_act*n_act,n_s);
//     
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//         H_EA2[a*n_act*(n_act-1)/2+b]+= act_INTS_AB[((x*n_act+w)*n_act+y)*n_act+z]*
//                                        G3_AAB[((((t*n_act+u)*n_act+x)*n_act+y)*n_act+z)*n_act+v];//unrestricted variant
// 
//     for(int t=0, a=0;t<n_act;t++)
//     for(int u=t+1   ;u<n_act;u++,a++)
//     for(int v=0, b=0;v<n_act;v++)
//     for(int w=v+1   ;w<n_act;w++,b++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//         H_EA2[a*n_act*(n_act-1)/2+b]-= act_INTS_AB[((x*n_act+v)*n_act+y)*n_act+z]*
//                                        G3_AAB[((((t*n_act+u)*n_act+x)*n_act+y)*n_act+z)*n_act+w];//unrestricted variant
// 
//     
//     
//     //U_IP_AB
//     for(int i=0;i<n_act;i++)
//     for(int j=0;j<n_act;j++)
//     for(int k=0;k<n_act;k++)
//     for(int l=0;l<n_act;l++){
//         U_IP_AB[((i*n_act+j)*n_act+k)*n_act+l]=GAMMA2[((i*n_act+k)*n_act+j)*n_act+l]/2;
//     }
//     
//     
//     //H_IP_AB
//     set_zero_matr(H_IP_AB,n_act*n_act*n_act*n_act);
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int y=0;y<n_act;y++)
//         H_IP_AB[((t*n_act+u)*n_act+v)*n_act+w]+= F_act_A[v*n_act+y]*U_IP_AB[((t*n_act+u)*n_act+y)*n_act+w]
//                                                 +F_act_A[w*n_act+y]*U_IP_AB[((t*n_act+u)*n_act+v)*n_act+y];//unrestricted variant
//     
//     set_zero_matr(G3_AAB,n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_3body_AAB(G3_AAB, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
//     average_DM_aldet(G3_AAB,G3_AAB,avecoe,n_act*n_act*n_act*n_act*n_act*n_act,n_s);
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP_AB[((t*n_act+u)*n_act+v)*n_act+w]+=(act_INTS_AA[((v*n_act+g)*n_act+y)*n_act+z]-
//                                                  act_INTS_AA[((y*n_act+g)*n_act+v)*n_act+z])*
//                                                  G3_AAB[((((t*n_act+y)*n_act+u)*n_act+z)*n_act+g)*n_act+w]*0.5;//unrestricted variant
// 
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP_AB[((t*n_act+u)*n_act+v)*n_act+w]+=(act_INTS_AA[((w*n_act+g)*n_act+y)*n_act+z]-
//                                                  act_INTS_AA[((y*n_act+g)*n_act+w)*n_act+z])*
//                                                  G3_AAB[((((u*n_act+y)*n_act+t)*n_act+z)*n_act+g)*n_act+v]*0.5;//unrestricted variant
// 
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP_AB[((t*n_act+u)*n_act+v)*n_act+w]-= act_INTS_AB[((v*n_act+g)*n_act+y)*n_act+z]*
//                                                  G3_AAB[((((u*n_act+y)*n_act+t)*n_act+w)*n_act+z)*n_act+g];//*0.5;//unrestricted variant
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++)
//         H_IP_AB[((t*n_act+u)*n_act+v)*n_act+w]+= act_INTS_AB[((v*n_act+g)*n_act+w)*n_act+z]*
//                                                  U_IP_AB[((t*n_act+u)*n_act+g)*n_act+z];//*0.5; //unrestricted variant
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//     for(int z=0;z<n_act;z++)
//     for(int g=0;g<n_act;g++){
//         H_IP_AB[((t*n_act+u)*n_act+v)*n_act+w]+= act_INTS_AB[((x*n_act+g)*n_act+w)*n_act+z]*
//                                                  G3_AAB[((((t*n_act+x)*n_act+u)*n_act+g)*n_act+v)*n_act+z];//*0.5;//unrestricted variant
//     }
//     
//     
//     //U_EA_AB
//     for(int i=0;i<n_act;i++)
//     for(int j=0;j<n_act;j++)
//     for(int k=0;k<n_act;k++)
//     for(int l=0;l<n_act;l++){
//         U_EA_AB[((i*n_act+j)*n_act+k)*n_act+l]=GAMMA2[((i*n_act+k)*n_act+j)*n_act+l]/2;
//         if(i==k)U_EA_AB[((i*n_act+j)*n_act+k)*n_act+l]+=U_EA[j*n_act+l];
//         if(j==l)U_EA_AB[((i*n_act+j)*n_act+k)*n_act+l]-=U_IP[k*n_act+i];
//     }    
//     
//     //H_EA_AB
//     set_zero_matr(H_EA_AB,n_act*n_act*n_act*n_act);
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//         H_EA_AB[((t*n_act+u)*n_act+v)*n_act+w]+= F_act_A[v*n_act+x]*U_EA_AB[((t*n_act+u)*n_act+x)*n_act+w]
//                                                 +F_act_A[w*n_act+x]*U_EA_AB[((t*n_act+u)*n_act+v)*n_act+x];//unrestricted variant
//     
//     set_zero_matr(G3_AAB,n_s*n_s*n_act*n_act*n_act*n_act*n_act*n_act);
//     aldet_calc_DM_3body_AAB_3(G3_AAB, n_states[a], n_states[a], coef    [a], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, 0, 1);
//     average_DM_aldet(G3_AAB,G3_AAB,avecoe,n_act*n_act*n_act*n_act*n_act*n_act,n_s);
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//         H_EA_AB[((t*n_act+u)*n_act+v)*n_act+w]+=(act_INTS_AA[((x*n_act+v)*n_act+y)*n_act+z]-
//                                                  act_INTS_AA[((x*n_act+z)*n_act+y)*n_act+v])*
//                                                  G3_AAB[((((t*n_act+u)*n_act+w)*n_act+x)*n_act+y)*n_act+z]*0.5;//unrestricted variant
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//         H_EA_AB[((t*n_act+u)*n_act+v)*n_act+w]+=(act_INTS_AA[((x*n_act+w)*n_act+y)*n_act+z]-
//                                                  act_INTS_AA[((x*n_act+z)*n_act+y)*n_act+w])*
//                                                  G3_AAB[((((u*n_act+t)*n_act+v)*n_act+x)*n_act+y)*n_act+z]*0.5;//unrestricted variant
// 
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int g=0;g<n_act;g++)
//         H_EA_AB[((t*n_act+u)*n_act+v)*n_act+w]-=(act_INTS_AB[((x*n_act+g)*n_act+y)*n_act+w])*
//                                                  G3_AAB[((((t*n_act+u)*n_act+y)*n_act+x)*n_act+v)*n_act+g];//*0.5;//unrestricted variant
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//     for(int z=0;z<n_act;z++)
//         H_EA_AB[((t*n_act+u)*n_act+v)*n_act+w]+=(act_INTS_AB[((x*n_act+v)*n_act+y)*n_act+z])*
//                                                  G3_AAB[((((u*n_act+t)*n_act+x)*n_act+w)*n_act+y)*n_act+z];//*0.5;//unrestricted variant
//     
//     for(int t=0;t<n_act;t++)
//     for(int u=0;u<n_act;u++)
//     for(int v=0;v<n_act;v++)
//     for(int w=0;w<n_act;w++)
//     for(int x=0;x<n_act;x++)
//     for(int y=0;y<n_act;y++)
//         H_EA_AB[((t*n_act+u)*n_act+v)*n_act+w]+=(act_INTS_AB[((x*n_act+v)*n_act+y)*n_act+w])*
//                                                  U_EA_AB[((t*n_act+u)*n_act+x)*n_act+y];//*0.5;//unrestricted variant
//     
//     
//     printf_timer("IPEA");
//     delete[] GAMMA;
//     delete[] GAMMA_EA;
//     delete[] GAMMA2;
//     delete[] GAMMA2_EA;
//     delete[] G3_AAA;
//     delete[] G3_AAB;
//     delete[] GAMMA_EA_full;
//     delete[] gamma;
    
    return 0;
}



int aldet_data::calc_DM(double * O, int a, int b){
    
    calc_DMA(O,a,b);
    calc_DMB(O,a,b);
    
    return 0;
}

int aldet_data::calc_DMA(double * O, int a, int b){
    
    calc_DM_part(O, coef[a], coef[b], n_states[a], n_states[b], na, Na, Nb, fa, fb, vec_a, bit_a);
    
    return 0;
}

int aldet_data::calc_DMB(double * O, int a, int b){
    
    calc_DM_part(O, coef_bas[a], coef_bas[b], n_states[a], n_states[b], nb, Nb, Na, fb, fa, vec_b, bit_b);
    
    return 0;
}

int aldet_data::calc_DM_part(double *O, double *B, double *K, int nsB, int nsK, int n_e, int N1, int N2, int * f1, int * f2, int * vec, int * bit){
    
    double * coef_B;// local array of bra-state coef
    double * coef_K;// local array of ket-state coef
    double sign_A;
    double sign_B;
    
    double * BUF = new double[nsB*nsK*n_act*n_act];
    set_zero_matr(BUF,nsB*nsK*n_act*n_act);
    double * BUF1;
    
    
    for(int i_CI = 0; i_CI<N1; i_CI++){
        memset(bit,0,n_act*sizeof(int));
        for (int k=1; k<n_e+1; k++) bit[vec[i_CI*(n_e+1)+k]-1] = 1;
        
        
        sign_A=1;
        coef_B=B+i_CI*N2*nsB;
        for(int i=0;i<n_act;i++)
            if(bit[i]){
                bit[i]=0;
                sign_B=1;
                    for(int j=0;j<n_act;j++)
                    if(bit[j]==0){
                        BUF1 = BUF + (i*n_act+j)*nsB*nsK;
                        bit[j]=1;
                        
                        auto i_CIext = get_ind_from_ON(bit, n_act, n_e, f1, buf);
                        coef_K=K+i_CIext*N2*nsK;
                        
//                         for(int i_s=0;i_s<nsB;i_s++)
//                         for(int j_s=0;j_s<nsK;j_s++)
//                         for(int j_CI = 0; j_CI<N2; j_CI++)
//                             BUF1[i_s*nsK+j_s]+=coef_B[j_CI*nsB+i_s]*coef_K[j_CI*nsK+j_s]*sign_A*sign_B;
                        
                        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                                    nsB,nsK,N2,sign_A*sign_B,
                                    coef_B,nsB,
                                    coef_K,nsK,1.0,
                                    BUF1,nsK);
                        
                        
                        bit[j]=0;
                    }
                    else
                        sign_B=-sign_B;

                sign_A=-sign_A;
                bit[i]=1;
            }
    }
//     printf_timer("transpose");
    
    for(int i_s=0;i_s<nsB*nsK;i_s++)
    for(int i=0;i<n_act*n_act;i++)
//     for(int j=0;j<n_act;j++)
//     for(int j_s=0;j_s<nsK;j_s++)
        O[i_s*n_act*n_act+i]+=BUF[i*nsB*nsK+i_s];
                        
    delete[] BUF;
    
   
    return 0;
}

int aldet_data::calc_DM_diag(double * O, int a){
    
    calc_DMA_diag(O,a);
    calc_DMB_diag(O,a);
    
    return 0;
}

int aldet_data::calc_DMA_diag(double * O, int a){
    
    calc_DM_diag_part(O, coef[a], n_states[a], na, Na, Nb, fa, fb, vec_a, bit_a);
    
    return 0;
}

int aldet_data::calc_DMB_diag(double * O, int a){
    
    calc_DM_diag_part(O, coef_bas[a], n_states[a], nb, Nb, Na, fb, fa, vec_b, bit_b);
    
    return 0;
}


int aldet_data::calc_DM_diag_part(double *O, double *C, int ns, int n_e, int N1, int N2, int * f1, int * f2, int * vec, int * bit){
    
    double * coef_0;// local array of bra-state coef
    double * coef_e;// local array of bra-state coef
    double sign_A;
    double sign_B;
    
    double * BUF = new double[ns*n_act*n_act];
    set_zero_matr(BUF,ns*n_act*n_act);
    double * BUF1;
    
    
    for(int i_CI = 0; i_CI<N1; i_CI++){
        memset(bit,0,n_act*sizeof(int));
        for (int k=1; k<n_e+1; k++) bit[vec[i_CI*(n_e+1)+k]-1] = 1;
        
        
        sign_A=1;
        coef_0=C+i_CI*N2*ns;
        for(int i=0;i<n_act;i++)
            if(bit[i]){
                bit[i]=0;
                sign_B=1;
                    for(int j=0;j<n_act;j++)
                    if(bit[j]==0){
                        BUF1 = BUF + (i*n_act+j)*ns;
                        bit[j]=1;
                        
                        auto i_CIext = get_ind_from_ON(bit, n_act, n_e, f1, buf);
                        coef_e=C+i_CIext*N2*ns;
                        
                        for(int j_CI = 0; j_CI<N2; j_CI++)
                        for(int i_s=0;i_s<ns;i_s++)
                            BUF1[i_s]+=coef_0[j_CI*ns+i_s]*coef_e[j_CI*ns+i_s]*sign_A*sign_B;
                        
                        
                        bit[j]=0;
                    }
                    else
                        sign_B=-sign_B;

                sign_A=-sign_A;
                bit[i]=1;
            }
    }
//     printf_timer("transpose");
    
    for(int i_s=0;i_s<ns;i_s++)
    for(int i=0;i<n_act*n_act;i++)
//     for(int j=0;j<n_act;j++)
//     for(int j_s=0;j_s<nsK;j_s++)
        O[i_s*n_act*n_act+i]+=BUF[i*ns+i_s];
                        
    delete[] BUF;
    
   
    return 0;
}

int calc_P_T(int * P, double * T, const int& size, double * A){
    LU(size, A, P);
//     fprintf(out_stream,"A\n");
//     PrintMatr(A,size,size,1);
    for (int i = 0; i<size; i++)
        for (int j = 0; j<i; j++){
            T[i*size+j]=-1.0*A[i*size+j];
            A[i*size+j]=0.0;
        }
    
    InvU(size, A);
    for (int i = 0; i<size; i++)
        for (int j = i; j<size; j++) T[i*size+j]=A[i*size+j];
//     fprintf(out_stream,"T\n");
//     PrintMatr(T,size,size,1);
    return 0;
}

int change_ci_with_P_1(double * ci_1, const int& N, const int& na, const int& Na, int * vec_a, int * fa, const int& nb, const int& Nb, int * vec_b, int * fb, int * P_1, int n_s){
    change_ci_with_P_one_dim(ci_1, N, na, Na, vec_a, fa, Nb*n_s, P_1);
    double * buf = new double[Na*Nb*n_s];
    transpose_3d_abc_to_bac(buf, ci_1, Na, Nb, n_s);
    memcpy(ci_1, buf, sizeof(double)*Na*Nb*n_s);
//     transpose_one_matrix(ci_1, Na, Nb);
    
    change_ci_with_P_one_dim(ci_1, N, nb, Nb, vec_b, fb, Na*n_s, P_1);
    transpose_3d_abc_to_bac(buf, ci_1, Nb, Na, n_s);
    memcpy(ci_1, buf, sizeof(double)*Na*Nb*n_s);
    delete[] buf;
    
    return 0;
}

int aldet_data::malmqvist(int i_set, double * U){
    
    if(n_act==0) return 0;
    
    double * U_copy = new double[n_act*n_act];
    memcpy(U_copy, U, sizeof(double)*n_act*n_act);
    
    double * ci_buf = new double[Na*Nb*n_states[i_set]];
    int * P_l = new int[n_act];
    double * T_l = new double[n_act*n_act];
    calc_P_T(P_l, T_l, n_act, U_copy);
    
    change_ci_with_P_1(coef[i_set], n_act, na, Na, vec_a, fa, nb, Nb, vec_b, fb, P_l, n_states[i_set]);

    change_ci_with_T_one_dim(n_act, na, Na, vec_a, fa, Nb*n_states[i_set], coef[i_set], ci_buf, T_l);
    transpose_3d_abc_to_bac(ci_buf, coef[i_set], Na, Nb, n_states[i_set]);
    memcpy(coef[i_set], ci_buf, sizeof(double)*Na*Nb*n_states[i_set]);

    change_ci_with_T_one_dim(n_act, nb, Nb, vec_b, fb, Na*n_states[i_set], coef[i_set], ci_buf, T_l);
    transpose_3d_abc_to_bac(ci_buf, coef[i_set], Nb, Na, n_states[i_set]);
    memcpy(coef[i_set], ci_buf, sizeof(double)*Na*Nb*n_states[i_set]);
    
    transpose_ci(i_set);
    
    delete[] ci_buf;
    delete[] T_l;
    delete[] P_l;
    delete[] U_copy;
    
    return 0;
}

int aldet_data::E_act_calc(double * E, int i_set){
    
    double Ea;
    
    E_act[i_set] = new double[Na*Nb];
    
    for(int i_CI = 0; i_CI<Na; i_CI++){
        Ea = 0;
        for (int k=1; k<na+1; k++) Ea += E[vec_a[i_CI*(na+1)+k]-1];
        
        for(int j_CI = 0; j_CI<Nb; j_CI++){
            E_act[i_set][i_CI*Nb+j_CI] = Ea;
            for (int k=1; k<nb+1; k++) E_act[i_set][i_CI*Nb+j_CI]+= E[vec_b[j_CI*(nb+1)+k]-1];
        }
    }
    
    return 0;
    
}

int aldet_data::F_calc(double * F, int i_set){
    
    double * C = coef[i_set];
    int n_s = n_states[i_set];
    
    
    for(int i_CI = 0; i_CI<Na; i_CI++){
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        for(int i=0;i<n_s;i++)
        for(int j=0;j<n_s;j++)
            F[i*n_s+j]+=C[(i_CI*Nb+j_CI)*n_s+i]*C[(i_CI*Nb+j_CI)*n_s+j]*E_act[i_set][i_CI*Nb+j_CI];
    }
    }
    
    return 0;
}

int aldet_data::H_calc(double * H, int n_s){
    
    
    set_zero_matr(H,n_s*n_s);
    
    aldet_calc_1body   (H, n_s, n_states[0], coef    [0], n_act, na, Na, Nb, fa, vec_a, F_act_A, 0, 1);
    aldet_calc_1body   (H, n_s, n_states[0], coef_bas[0], n_act, nb, Nb, Na, fb, vec_b, F_act_B, 0, 1);
    aldet_calc_2body_AA(H, n_s, n_states[0], coef    [0], n_act, na, Na, Nb, fa, vec_a, act_INTS_AA, 0, 1);
    aldet_calc_2body_AA(H, n_s, n_states[0], coef_bas[0], n_act, nb, Nb, Na, fb, vec_b, act_INTS_BB, 0, 1);
    aldet_calc_2body_AB(H, n_s, n_states[0], coef    [0], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, Ia, Ib, Ia_friends, Ib_friends,act_INTS_AB, 0, 1);
    
    for(int i=0;i<n_s;i++)H[i*n_s+i]+=E_core; 
    
    return 0;
    
}

int aldet_data::H_full_calc(double *H){
    
    sparsed_CI_vec *   lb = new sparsed_CI_vec[Nd]; 
    sparsed_CI_vec * H_lb = new sparsed_CI_vec[Nd]; 
    
    set_zero_matr(H,Nd*Nd);
    H_diag_calc();
    gen_bf(lb,Nd);
    
    double * H_lb_d = nullptr;
    
    if(do_PT==0)
        H_mult_sparsed_to_sparsed(H_lb, lb, Nd);
    else{
        H_lb_d = new_double_w_check(1LL*Nd*Nd,"dense form of {H x bf} products");
        H_mult_sparsed_to_dense(H_lb_d, lb, Nd);
    }
    
    if(do_PT==0)
        CI_ss_mult(H, Nd, lb, Nd, H_lb, Nd);  
    else
        CI_sd_mult(H, Nd, lb, Nd, H_lb_d, Nd, Nd);

    
    return 0;
        
}



int aldet_data::PT_update(){
//     if(na==0)if(nb==0)return 0;
#ifndef TEST_ALDET
    exit(0);
#endif

    for(int i=0; i<n_act;i++)
    for(int j=0; j<n_act;j++)
    for(int k=0; k<n_act;k++)
    for(int l=0; l<n_act;l++)
    for(int m=0; m<n_act;m++)
        T2_AA [((  i*n_act+j         )*n_act+k)*n_act+l]+=
        T3_AAA[((((j*n_act+i)*n_act+m)*n_act+l)*n_act+k)*n_act+m];

    if(T2_BB!=T2_AA)
    for(int i=0; i<n_act;i++)
    for(int j=0; j<n_act;j++)
    for(int k=0; k<n_act;k++)
    for(int l=0; l<n_act;l++)
    for(int m=0; m<n_act;m++)
        T2_BB [((  i*n_act+j         )*n_act+k)*n_act+l]+=
        T3_BBB[((((j*n_act+i)*n_act+m)*n_act+l)*n_act+k)*n_act+m];

    
    if(act_INTS_AB==act_INTS_AA){
        printf("ERROR: use of the external references in aldet_data is deprecated check act_INT_AA and act_INTS_AB\n");
        exit(0);
//         act_INTS_AB_alloced_inside = new double[n_act*n_act*n_act*n_act];
//         act_INTS_AB = act_INTS_AB_alloced_inside;
//         memcpy(act_INTS_AB,act_INTS_AA, n_act*n_act*n_act*n_act*sizeof(double));
    }
    for(int i=0; i<n_act;i++)
    for(int j=0; j<n_act;j++)
    for(int k=0; k<n_act;k++)
    for(int l=0; l<n_act;l++)
        act_INTS_AB[((i*n_act+j)*n_act+k)*n_act+l]=//AABB
        act_INTS_AB[((i*n_act+j)*n_act+k)*n_act+l]+//AABB
        T2_AB     [((i*n_act+k)*n_act+j)*n_act+l];//ABAB
    //AAAA etc. are updated in gen_ext_ind_PT()

    memcpy(act_INTS_AA,act_INTS_AB, n_act*n_act*n_act*n_act*sizeof(double));
    memcpy(act_INTS_BB,act_INTS_AB, n_act*n_act*n_act*n_act*sizeof(double));
    
    gen_ext_ind_PT();
    
    
    
    for(int i=0; i<n_act*n_act;i++)
        F_act_A[i]+=T1_A[i];
    
    for(int i=0; i<n_act*n_act;i++)
        F_act_B[i]+=T1_B[i];
    
    H_diag_calc_PT();
    
    for(int i=0;i<Nd;i++)
        H_diag[i]+=T0;
    
    E_core+=T0;
    
    return 0;
}

int aldet_data::H_mult(int n0, int n_s){
    
    if(Hv==NULL) {
        Hv     = new double[n_states[0]*Na*Nb];
        Hv_buf = new double[n_states[0]*Na*Nb];
        set_zero_matr(Hv,n_states[0]*Na*Nb);
    }
    transpose_ci(0);
    
    for(int i=0 ;i<Na ;i++)
    for(int j=0 ;j<Nb ;j++)
    for(int k=n0;k<n_s;k++)
        Hv[(i*Nb+j)*n_states[0]+k]=0;
    
    
    set_zero_matr(Hv_buf,n_states[0]*Na*Nb);
#pragma omp parallel
    {
        int nt = omp_get_thread_num();
        H_mult_A  (Hv_buf, n0, n_s, n_states[0], coef_asb[0], nt, num_threads);
        H_mult_AA (Hv_buf, n0, n_s, n_states[0], coef_asb[0], nt, num_threads);    
        if(do_PT)
        H_mult_AAA(Hv_buf, n0, n_s, n_states[0], coef_asb[0], nt, num_threads);    
    }
    transpose_3d_abc_to_acb_part_b_sum(Hv, Hv_buf, Na, n0, n_s, n_states[0], Nb);
    set_zero_matr(Hv_buf,n_states[0]*Na*Nb);
    
#pragma omp parallel
    {
        int nt = omp_get_thread_num();
        H_mult_B  (Hv_buf, n0, n_s, n_states[0], coef_bsa[0], nt, num_threads);
        H_mult_BB (Hv_buf, n0, n_s, n_states[0], coef_bsa[0], nt, num_threads);
        if(do_PT)
        H_mult_BBB(Hv_buf, n0, n_s, n_states[0], coef_bsa[0], nt, num_threads);
    }
    transpose_3d_abc_to_cab_part_b_sum(Hv, Hv_buf, Nb, n0, n_s, n_states[0], Na);
    
#pragma omp parallel
    {
        int nt = omp_get_thread_num();
        H_mult_AB(Hv, n0, n_s, n_states[0], coef[0], nt, num_threads); 
    }

#pragma omp parallel for
    for(int i=0;i<Nd;i++)
        for(int j=n0;j<n_s;j++)
#ifdef TEST_ALDET
            Hv[i*n_states[0]+j]+=E_core*coef[0][i*n_states[0]+j];
#endif            
#ifndef TEST_ALDET
            Hv[i*n_states[0]+j]+=H_diag[i]*coef[0][i*n_states[0]+j];
#endif                
    
    if(do_PT){
#pragma omp parallel
        {
            int nt = omp_get_thread_num();
            H_mult_AAB(Hv, n0, n_s, n_states[0], coef[0], nt, num_threads); 
        }
#pragma omp parallel
        {
            int nt = omp_get_thread_num();
            H_mult_ABB(Hv, n0, n_s, n_states[0], coef[0], nt, num_threads); 
        }

        
    }
    return 0;
    
}


int aldet_data::H_mult_sparsed_to_dense(double * ext_Hc, sparsed_CI_vec * c, ci_int n_s){
    
    set_zero_matr(ext_Hc,1LL*n_s*Na*Nb);
    
    ci_int l,i,j,i1,k1,ind;
    double * K;
    double   h,s,coef;
    
    for(ci_int i_s=0; i_s<n_s; i_s++){
        
        l=c[i_s].c.size();
        for(ci_int a=0; a<l;a++){
            ind = c[i_s].n[a];
#ifdef TEST_ALDET
            ext_Hc[ind*n_s+i_s]+=/*H_diag[ind]*/E_core*c[i_s].c[a];
#endif
#ifndef TEST_ALDET
            ext_Hc[ind*n_s+i_s]+=H_diag[ind]*c[i_s].c[a];
#endif
            i=ind/Nb;
            j=ind%Nb;
            for(ci_int k = 0;k<n_e1_a;k++){
                s = e1_sign_a[i*n_e1_a+k];
                h = (F_act_A[e1_orbs_a[i*n_e1_a+k]]/*+J_act_a[i*n_act*n_act+e1_orbs_a[i*n_e1_a+k]]*/)*s;
//                 K = K_act_b+e1_orbs_a[i*n_e1_a+k]*Nb;
                
                ext_Hc[(e1_ind_a[i*n_e1_a+k]*Nb+j)*n_s+i_s]+=(h/*+s*K[j]*/)*c[i_s].c[a];
            }
            for(ci_int k = 0;k<n_e2_a;k++){
                h = e2_V_a[i*n_e2_a+k];
                ext_Hc[(e2_ind_a[i*n_e2_a+k]*Nb+j)*n_s+i_s]+=h*c[i_s].c[a];
                
            }
            if(do_PT)
            for(ci_int k = 0;k<n_e3_a;k++){
                h = e3_V_a[i*n_e3_a+k];
                ext_Hc[(e3_ind_a[i*n_e3_a+k]*Nb+j)*n_s+i_s]+=h*c[i_s].c[a];
                
            }
            for(ci_int k = 0;k<n_e1_b;k++){
                s = e1_sign_b[j*n_e1_b+k];
                h = (F_act_B[e1_orbs_b[j*n_e1_b+k]]/*+J_act_b[j*n_act*n_act+e1_orbs_b[j*n_e1_b+k]]*/)*s;
//                 K = K_act_b+e1_orbs_b[j*n_e1_b+k]*Nb;
                
                ext_Hc[(i*Nb+e1_ind_b[j*n_e1_b+k])*n_s+i_s]+=(h/*+s*K[i]*/)*c[i_s].c[a];
            }
            for(ci_int k = 0;k<n_e2_b;k++){
                h = e2_V_b[j*n_e2_b+k];
                ext_Hc[(i*Nb+e2_ind_b[j*n_e2_b+k])*n_s+i_s]+=h*c[i_s].c[a];
                
            }
            if(do_PT)
            for(ci_int k = 0;k<n_e3_b;k++){
                h = e3_V_b[j*n_e3_b+k];
                ext_Hc[(i*Nb+e3_ind_b[j*n_e3_b+k])*n_s+i_s]+=h*c[i_s].c[a];
                
            }
            for(ci_int k = 0;k<n_e1_a;k++){
                k1 = e1_orbs_a[i*n_e1_a+k]*n_act*n_act;
                s = e1_sign_a[i*n_e1_a+k];
                i1 = e1_ind_a[i*n_e1_a+k]*Nb;
                for(ci_int l = 0;l<n_e1_b;l++){
                    h = act_INTS_AB[k1+e1_orbs_b[j*n_e1_b+l]]*
                        e1_sign_b[j*n_e1_b+l]*s;
            
                    ext_Hc[(i1+e1_ind_b[j*n_e1_b+l])*n_s+i_s]+=h*c[i_s].c[a];
                }
            }
            if(do_PT)
            for(ci_int k = 0;k<n_e2_a;k++){
                k1 = e2_orbs_a[i*n_e2_a+k]*n_act*n_act;
                s  = e2_sign_a[i*n_e2_a+k];
                i1 = e2_ind_a [i*n_e2_a+k]*Nb;
                for(ci_int l = 0;l<n_e1_b;l++){
                    h = T3_AAB[k1+e1_orbs_b[j*n_e1_b+l]]*
                        e1_sign_b[j*n_e1_b+l]*s;
            
                    ext_Hc[(i1+e1_ind_b[j*n_e1_b+l])*n_s+i_s]+=h*c[i_s].c[a];
                }
            }
            if(do_PT)
            for(ci_int k = 0;k<n_e2_b;k++){
                k1 = e2_orbs_b[j*n_e2_b+k]*n_act*n_act;
                s  = e2_sign_b[j*n_e2_b+k];
                i1 = e2_ind_b [j*n_e2_b+k];
                for(ci_int l = 0;l<n_e1_a;l++){
                    h = T3_BBA[k1+e1_orbs_a[i*n_e1_a+l]]*
                        e1_sign_a[i*n_e1_a+l]*s;
            
                    ext_Hc[(i1+e1_ind_a[i*n_e1_a+l]*Nb)*n_s+i_s]+=h*c[i_s].c[a];
                }
            }
                
//             if(do_PT){
//                 fprintf(out_stream,"H_mult_sparsed_to_dense is not written for the PT variant\n");
//                 exit(0);
//             }
        }
            
    }

    return 0;
    
}

int aldet_data::H_mult_sparsed_to_sparsed(sparsed_CI_vec * ext_Hc, sparsed_CI_vec * c, ci_int n_s){
    
    ci_int l,i,j,i1,k1,ind;
    double * K;
    double   h,s,coef;
    
    for(ci_int i_s=0; i_s<n_s; i_s++){
        
        l=c[i_s].c.size();
        ext_Hc[i_s].c.resize((n_e1_a+n_e2_a+n_e1_b+n_e2_b+n_e1_a*n_e1_b+1)*l);
        ext_Hc[i_s].n.resize((n_e1_a+n_e2_a+n_e1_b+n_e2_b+n_e1_a*n_e1_b+1)*l);
//         fprintf(out_stream,"%d (%d)\n",ext_Hc[i_s].c.size(),l);
//         getchar();
        ci_int Hi=0;
        for(ci_int a=0; a<l;a++){
            ind = c[i_s].n[a];
            ext_Hc[i_s].n[Hi]=ind;
#ifndef TEST_ALDET
            ext_Hc[i_s].c[Hi]=H_diag[ind]*c[i_s].c[a];
#endif
#ifdef TEST_ALDET
            ext_Hc[i_s].c[Hi]=E_core*c[i_s].c[a];
#endif
            Hi++;
            i=ind/Nb;
            j=ind%Nb;
            for(ci_int k = 0;k<n_e1_a;k++){
                s = e1_sign_a[i*n_e1_a+k];
                h = (F_act_A[e1_orbs_a[i*n_e1_a+k]]
#ifndef TEST_ALDET
                +J_act_a[i*n_act*n_act+e1_orbs_a[i*n_e1_a+k]]
#endif                
                )*s;
                K = K_act_b+e1_orbs_a[i*n_e1_a+k]*Nb;
                
                ext_Hc[i_s].n[Hi]=e1_ind_a[i*n_e1_a+k]*Nb+j;
                ext_Hc[i_s].c[Hi]=(h
#ifndef TEST_ALDET
                +s*K[j]
#endif
                )*c[i_s].c[a];
//                 fprintf(stderr,"A %d\n",ext_Hc[i_s].n[Hi]);
                Hi++;
            
                
            }
            for(ci_int k = 0;k<n_e2_a;k++){
                h = e2_V_a[i*n_e2_a+k];
                ext_Hc[i_s].n[Hi]=e2_ind_a[i*n_e2_a+k]*Nb+j;
                ext_Hc[i_s].c[Hi]=h*c[i_s].c[a];
//                 fprintf(stderr,"AA %d\n",ext_Hc[i_s].n[Hi]);
                Hi++;
            
                
            }
            for(ci_int k = 0;k<n_e1_b;k++){
                s = e1_sign_b[j*n_e1_b+k];
                h = (F_act_B[e1_orbs_b[j*n_e1_b+k]]
#ifndef TEST_ALDET
                +J_act_b[j*n_act*n_act+e1_orbs_b[j*n_e1_b+k]]
#endif
                )*s;
                K = K_act_b+e1_orbs_b[j*n_e1_b+k]*Nb;
                
                ext_Hc[i_s].n[Hi]=i*Nb+e1_ind_b[j*n_e1_b+k];
                ext_Hc[i_s].c[Hi]=(h
#ifndef TEST_ALDET
                +s*K[i]
#endif
                )*c[i_s].c[a];
//                 fprintf(stderr,"B %d\n",ext_Hc[i_s].n[Hi]);
                Hi++;
            
            }
            for(ci_int k = 0;k<n_e2_b;k++){
                h = e2_V_b[j*n_e2_b+k];
                ext_Hc[i_s].n[Hi]=i*Nb+e2_ind_b[j*n_e2_b+k];
                ext_Hc[i_s].c[Hi]=h*c[i_s].c[a];
//                 fprintf(stderr,"BB %d\n",ext_Hc[i_s].n[Hi]);
                Hi++;
            
                
            }
            for(ci_int k = 0;k<n_e1_a;k++){
                k1 = e1_orbs_a[i*n_e1_a+k]*n_act*n_act;
                s = e1_sign_a[i*n_e1_a+k];
                i1 = e1_ind_a[i*n_e1_a+k]*Nb;
                for(ci_int l = 0;l<n_e1_b;l++){
                    h = act_INTS_AB[k1+e1_orbs_b[j*n_e1_b+l]]*
                        e1_sign_b[j*n_e1_b+l]*s;
                    ext_Hc[i_s].n[Hi]=i1+e1_ind_b[j*n_e1_b+l];
                    ext_Hc[i_s].c[Hi]=h*c[i_s].c[a];
//                     fprintf(stderr,"AB %d\n",ext_Hc[i_s].n[Hi]);
                    Hi++;
            
                }
            }
            if(do_PT){
                fprintf(out_stream,"H_mult_sparsed_to_sparsed is not written for the PT variant\n");
                exit(0);
            }
            
        }
//         fprintf(out_stream,"%d\n",Hi);
//         getchar();            
    }
//     exit(0);

    return 0;
    
}


int aldet_data::H_mult_A(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th){
   
    double * c;
    double * K;
    double   h;
    double   s;
    for(int i = i_th; i < Na; i+=n_th)
    for(int k = 0;k<n_e1_a;k++){
        c = ci_I+e1_ind_a[i*n_e1_a+k]*Nb*ld;
        s = e1_sign_a[i*n_e1_a+k];
        h = (F_act_A[e1_orbs_a[i*n_e1_a+k]]/*+J_act_a[i*n_act*n_act+e1_orbs_a[i*n_e1_a+k]]*/)*s;
//         K = K_act_b+e1_orbs_a[i*n_e1_a+k]*Nb;
        
        for(int i_s=n0; i_s<n_s; i_s++){
        for(int j = 0 ; j  < Nb; j++)
            ci_O[i*Nb*ld+i_s*Nb+j]+=(h/*+s*K[j]*/)*c[i_s*Nb+j];
        }
            
        
    }
    
    return 0;
}

int aldet_data::H_mult_B(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th){
   
    double * c;
    double * K;
    double   h;
    double   s;
    
    for(int i = i_th; i < Nb; i+=n_th)
    for(int k = 0;k<n_e1_b;k++){
        c = ci_I+e1_ind_b[i*n_e1_b+k]*Na*ld;
        s = e1_sign_b[i*n_e1_b+k];
        h = (F_act_B[e1_orbs_b[i*n_e1_b+k]]/*+J_act_b[i*n_act*n_act+e1_orbs_b[i*n_e1_b+k]]*/)*s;
//         K = K_act_a+e1_orbs_b[i*n_e1_b+k]*Na;
        
        for(int i_s=n0; i_s<n_s; i_s++){
        for(int j = 0 ; j  < Na; j++)
                ci_O[i*Na*ld+i_s*Na+j]+=(h/*+s*K[j]*/)*c[i_s*Na+j];
        }
            
        
    }
    
    return 0;
}


int aldet_data::H_mult_AA(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th){
    double * c;
    double   h;
    
    for(int i = i_th; i < Na; i+=n_th)
    for(int k = 0;k<n_e2_a;k++){
        c =ci_I+e2_ind_a[i*n_e2_a+k]*Nb*ld;
        h = e2_V_a[i*n_e2_a+k];
//         if(i!=e2_ind_a[i*n_e2_a+k])
        for(int i_s=n0; i_s<n_s; i_s++){
        for(int j =  0; j  < Nb; j++)
                ci_O[i*Nb*ld+i_s*Nb+j]+=h*c[i_s*Nb+j];
        }
            
        
    }
    return 0;
}

int aldet_data::G_AA(double * __restrict__ G){
    double * c;
    double * c2;
    double * R;
    double s;
    
    int N=n_act;
    int n_s=n_states[0];
    double * BUF = new double[N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*n_s*n_s);
    
    for(int i = 0; i < Na; i++)
    for(int k = 0;k<n_e2_a;k++){
        c  = coef_asb[0]+i*Nb*n_s;
        c2 = coef_asb[0]+e2_ind_a[i*n_e2_a+k]*Nb*n_s;
        R = BUF+e2_orbs_a[i*n_e2_a+k]*n_s*n_s;
        s = e2_sign_a[i*n_e2_a+k];
        for(int i_s=0; i_s<n_s; i_s++)
        for(int j_s=0; j_s<n_s; j_s++){
        for(int j =  0; j  < Nb; j++)
                R[i_s*n_s+j_s]+=-c[i_s*Nb+j]*c2[j_s*Nb+j]*s;
        }
    }
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    for(int k=i;k<N;k++)
    for(int l=j;l<N;l++){
        G[i_s*N*N*N*N+i*N*N*N+j*N*N+k*N+l]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+j*N*N+i*N+l]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+i*N*N*N+l*N*N+k*N+j]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+l*N*N+i*N+j]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
    }

    delete[] BUF;
    
    return 0;
}

int aldet_data::G_AB(double * __restrict__ G){
    double * c;
    double * c2;
    double * R_AB;
    double * R_BA;
    double s;
    double s2;
    int c1;
    
    int N=n_act;
    int n_s=n_states[0];
    double * BUF = new double[N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*n_s*n_s);
    
    for(int i = 0; i < Na; i++)
    for(int k = 0;k<n_e1_a;k++){
        int a = e1_orbs_a[i*n_e1_a+k];
        double s = e1_sign_a[i*n_e1_a+k];
        int c1 = e1_ind_a[i*n_e1_a+k]*Nb;
        
        for(int j = 0; j < Nb; j++)
        for(int l = 0;l<n_e1_b;l++)
        {
            c  = coef[0]+(i*Nb+j)*n_s;
            c2 = coef[0]+(c1+e1_ind_b[j*n_e1_b+l])*n_s;
            int b=e1_orbs_b[j*n_e1_b+l];
            R_AB = BUF+(a*n_act*n_act+b)*n_s*n_s;
            R_BA = BUF+(b*n_act*n_act+a)*n_s*n_s;
            s2 = e1_sign_b[j*n_e1_b+l]*s;
            for(int i_s=0; i_s<n_s; i_s++)
            for(int j_s=0; j_s<n_s; j_s++){
                double c3=c[i_s]*c2[j_s]*s2;
                R_AB[i_s*n_s+j_s]+=c3;
                R_BA[i_s*n_s+j_s]+=c3;
            }
        }
    }
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    for(int k=0;k<N;k++)
    for(int l=0;l<N;l++){
        G[i_s*N*N*N*N+i*N*N*N+j*N*N+k*N+l]+= BUF[(i*N*N*N+j*N*N+k*N+l)*n_s*n_s+i_s];
    }

    delete[] BUF;
    
    return 0;
}

int aldet_data::G_BB(double * __restrict__ G){
    double * c;
    double * c2;
    double * R;
    double s;
    
    int N=n_act;
    int n_s=n_states[0];
    double * BUF = new double[N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*n_s*n_s);
    
    for(int i = 0; i < Nb; i++)
    for(int k = 0;k<n_e2_b;k++){
        c  = coef_bsa[0]+i*Na*n_s;
        c2 = coef_bsa[0]+e2_ind_b[i*n_e2_b+k]*Na*n_s;
        R = BUF+e2_orbs_b[i*n_e2_b+k]*n_s*n_s;
        s = e2_sign_b[i*n_e2_b+k];
        for(int i_s=0; i_s<n_s; i_s++)
        for(int j_s=0; j_s<n_s; j_s++){
        for(int j =  0; j  < Na; j++)
                R[i_s*n_s+j_s]+=-c[i_s*Na+j]*c2[j_s*Na+j]*s;
        }
    }
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    for(int k=i;k<N;k++)
    for(int l=j;l<N;l++){
        G[i_s*N*N*N*N+i*N*N*N+j*N*N+k*N+l]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+j*N*N+i*N+l]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+i*N*N*N+l*N*N+k*N+j]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+l*N*N+i*N+j]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
    }

    delete[] BUF;
    
    return 0;
}



int aldet_data::H_mult_BB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th){
    double * c;
    double   h;
    
    for(int i = i_th; i < Nb; i+=n_th)
    for(int k = 0;k<n_e2_b;k++){
        c =ci_I+e2_ind_b[i*n_e2_b+k]*Na*ld;
        h = e2_V_b[i*n_e2_b+k];
//         if(i!=e2_ind_b[i*n_e2_b+k])
        for(int i_s=n0; i_s<n_s; i_s++){
        for(int j  = 0; j  < Na; j++)
                ci_O[i*Na*ld+i_s*Na+j]+=h*c[i_s*Na+j];
        }
            
        
    }
    return 0;
}


int aldet_data::H_mult_AB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                          double * __restrict__ ci_I,
                          int i_th, int n_th){
    
    double t[NUM_AVX];
    double c[NUM_AVX];
    double V[NUM_AVX];
    
//     printf_timer("generating sp matrix");
    
    for(int i = i_th; i < Na; i+=n_th)
    for(int k = 0;k<n_e1_a;k++){
        int a = e1_orbs_a[i*n_e1_a+k]*n_act*n_act;
        double s = e1_sign_a[i*n_e1_a+k];
        int c1 = e1_ind_a[i*n_e1_a+k]*Nb;
        
        for(int j = 0; j < Nb; j++)
        for(int l = 0;l<n_e1_b;l+=NUM_AVX)
        {
            
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] = act_INTS_AB[a+e1_orbs_b[j*n_e1_b+l+ii]];
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] *= e1_sign_b[j*n_e1_b+l+ii];
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] *= s;
            for(int i_s=n0;i_s<n_s; i_s++){
                for(int ii=0;ii<NUM_AVX;ii++)
                    c[ii] = ci_I[(c1+e1_ind_b[j*n_e1_b+l+ii])*ld+i_s];
                for(int ii=0;ii<NUM_AVX;ii++)
                    t[ii] = V[ii] * c[ii];
                for(int ii=0;ii<NUM_AVX;ii++)
                    ci_O[(i*Nb+j)*ld+i_s]+=t[ii];
            }
        }
    }
    
    
//     printf_timer("mat mult");
    
    
    return 0;
}

int aldet_data::H_mult_AAA(double * __restrict__ ci_O, int n0, int n_s, int ld,
                           double * __restrict__ ci_I,
                           int i_th, int n_th){
    double * c;
    double   h;
    
    for(int i = i_th; i < Na; i+=n_th)
    for(int k = 0;k<n_e3_a;k++){
        c =ci_I+e3_ind_a[i*n_e3_a+k]*Nb*ld;
        h = e3_V_a[i*n_e3_a+k];
//         if(i!=e3_ind_a[i*n_e3_a+k])
        for(int i_s=n0; i_s<n_s; i_s++){
        for(int j =  0; j  < Nb; j++)
                ci_O[i*Nb*ld+i_s*Nb+j]+=h*c[i_s*Nb+j];
        }
            
        
    }
    return 0;
}

int aldet_data::H_mult_BBB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                           double * __restrict__ ci_I,
                           int i_th, int n_th){
    double * c;
    double   h;
    
    for(int i = i_th; i < Nb; i+=n_th)
    for(int k = 0;k<n_e3_b;k++){
        c =ci_I+e3_ind_b[i*n_e3_b+k]*Na*ld;
        h = e3_V_b[i*n_e3_b+k];
//         if(i!=e3_ind_b[i*n_e3_b+k])
        for(int i_s=n0; i_s<n_s; i_s++){
        for(int j  = 0; j  < Na; j++)
                ci_O[i*Na*ld+i_s*Na+j]+=h*c[i_s*Na+j];
        }
            
        
    }
    return 0;
}



int aldet_data::H_mult_AAB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                          double * __restrict__ ci_I,
                          int i_th, int n_th){
    
    double t[NUM_AVX];
    double c[NUM_AVX];
    double V[NUM_AVX];
    
//     printf_timer("generating sp matrix");
    
    for(int i = i_th; i < Na; i+=n_th)
    for(int k = 0;k<n_e2_a;k++){
        int a = e2_orbs_a[i*n_e2_a+k]*n_act*n_act;
        double s = e2_sign_a[i*n_e2_a+k];
        int c1 = e2_ind_a[i*n_e2_a+k]*Nb;
        
        for(int j = 0; j < Nb; j++)
        for(int l = 0;l<n_e1_b;l+=NUM_AVX)
        {
            
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] = T3_AAB[a+e1_orbs_b[j*n_e1_b+l+ii]];
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] *= e1_sign_b[j*n_e1_b+l+ii];
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] *= s;
            for(int i_s=n0;i_s<n_s; i_s++){
                for(int ii=0;ii<NUM_AVX;ii++)
                    c[ii] = ci_I[(c1+e1_ind_b[j*n_e1_b+l+ii])*ld+i_s];
                for(int ii=0;ii<NUM_AVX;ii++)
                    t[ii] = V[ii] * c[ii];
                for(int ii=0;ii<NUM_AVX;ii++)
                    ci_O[(i*Nb+j)*ld+i_s]+=t[ii];
            }
        }
    }
    
    
//     printf_timer("mat mult");
    
    
    return 0;
}

int aldet_data::H_mult_ABB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                          double * __restrict__ ci_I,
                          int i_th, int n_th){
    
    double t[NUM_AVX];
    double c[NUM_AVX];
    double V[NUM_AVX];
    
//     printf_timer("generating sp matrix");
    
    for(int i = i_th; i < Nb; i+=n_th)
    for(int k = 0;k<n_e2_b;k++){
        int a = e2_orbs_b[i*n_e2_b+k]*n_act*n_act;
        double s = e2_sign_b[i*n_e2_b+k];
        int c1 = e2_ind_b[i*n_e2_b+k];
        
        for(int j = 0; j < Na; j++)
        for(int l = 0;l<n_e1_a;l+=NUM_AVX)
        {
            
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] = T3_BBA[a+e1_orbs_a[j*n_e1_a+l+ii]];
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] *= e1_sign_a[j*n_e1_a+l+ii];
            for(int ii=0;ii<NUM_AVX;ii++)
                V[ii] *= s;
            for(int i_s=n0;i_s<n_s; i_s++){
                for(int ii=0;ii<NUM_AVX;ii++)
                    c[ii] = ci_I[(c1+e1_ind_a[j*n_e1_a+l+ii]*Nb)*ld+i_s];
                for(int ii=0;ii<NUM_AVX;ii++)
                    t[ii] = V[ii] * c[ii];
                for(int ii=0;ii<NUM_AVX;ii++)
                    ci_O[(j*Nb+i)*ld+i_s]+=t[ii];
            }
        }
    }
    
    
//     printf_timer("mat mult");
    
    
    return 0;
}

int aldet_data::G_calc(double * G){
    
    double ** G_th =new double *[num_threads];
    G_th[0] = G;
    for(int i=1;i<num_threads;i++)G_th[i]=new double[n_act*n_act*n_act*n_act*n_states[0]];
    
    #pragma omp parallel
    {
        int nt = omp_get_thread_num();
        set_zero_matr(G_th[nt],n_states[0]*n_act*n_act*n_act*n_act);
        aldet_calc_DM_2body_AA_diag(G_th[nt], n_states[0], n_states[0], coef    [0], n_act, na, Na, Nb, fa, vec_a, nt, num_threads);
        aldet_calc_DM_2body_AA_diag(G_th[nt], n_states[0], n_states[0], coef_bas[0], n_act, nb, Nb, Na, fb, vec_b, nt, num_threads);
        aldet_calc_DM_2body_AB_diag(G_th[nt], n_states[0], n_states[0], coef    [0], n_act, na, nb, Na, Nb, fa, fb, vec_a, vec_b, nt, num_threads);
    
    }
    
    for(int i=1;i<num_threads;i++)
    for(int j=0;j<n_act*n_act*n_act*n_act*n_states[0];j++)
        G_th[0][j]+=G_th[i][j];
    for(int i=1;i<num_threads;i++)delete[] G_th[i];
    delete[] G_th;
    
    
    return 0;
    
}


// int aldet_data::G_calc_AA(double * __restrict__ ci_O, int n0, int n_s, int ld,
//                           double * __restrict__ ci_I,
//                           int i_th, int n_th){
//     double * c;
//     double   h;
//     
//     for(int i = i_th; i < Na; i+=n_th)
//     for(int k = 0;k<n_e2_a;k++){
//         c =ci_I+e2_ind_a[i*n_e2_a+k]*Nb*ld;
//         h = e2_V_a[i*n_e2_a+k];
// //         if(i!=e2_ind_a[i*n_e2_a+k])
//         for(int i_s=n0; i_s<n_s; i_s++){
//         for(int j =  0; j  < Nb; j++)
//                 ci_O[i*Nb*ld+i_s*Nb+j]+=h*c[i_s*Nb+j];
//         }
//             
//         
//     }
//     return 0;
// }


int aldet_data::H_diag_calc(){
    
    if(H_diag     ==NULL) H_diag      = new double[/*n_states[0]**/Nd];
    if(H_diag_appr==NULL) H_diag_appr = new double[/*n_states[0]**/Nd];
    
    set_zero_matr(H_diag,/*n_states[0]**/Nd);
        
    int N=n_act;
    
    for(int i_CI =0/*i_th*/; i_CI<Na; i_CI++/*+=n_th*/)
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        H_diag[i_CI*Nb+j_CI]=E_core;
        
        memset(bit_a,0,N*sizeof(int));
        memset(bit_b,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            H_diag[i_CI*Nb+j_CI]+=F_act_A[t*N+t];
            
            for(int u=t+1;u<N;u++)if(bit_a[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS_AA[((t*N+t)*N+u)*N+u]-act_INTS_AA[((t*N+u)*N+u)*N+t];
            for(int u=0  ;u<N;u++)if(bit_b[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS_AB[((t*N+t)*N+u)*N+u];
        }
        for(int t=0  ;t<N;t++)if(bit_b[t]!=0){
            H_diag[i_CI*Nb+j_CI]+=F_act_B[t*N+t];
//             
            for(int u=t+1;u<N;u++)if(bit_b[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS_BB[((t*N+t)*N+u)*N+u]-act_INTS_BB[((t*N+u)*N+u)*N+t];
//             for(int u=0  ;u<N;u++)if(bit_b[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS[((t*N+t)*N+u)*N+u];
        }
        
//         fprintf(out_stream,"H[%d,%d]=%e\n",i_CI, j_CI, H_diag[i_CI*Nb+j_CI]);
//         if(j_CI==4) return 0;
    }
    memcpy(H_diag_appr,H_diag,Nd*sizeof(double));
    if(mult!=0){
        int * ind_done = new int[Nd];
        set_zero_matr_int(ind_done,Nd);
        double E_min;
        int i_CI,j_CI;
        int n_ext;
        int e_a, e_b, Ne;
        int * vec_e = new int [Na*(na+1)];
        int * bit_e = new int [n_act];
        int * i_ext = new int[n_act];

        for(int i=0;i<Nd;i++)if(ind_done[i]==0){
            
            i_CI=i/Nb;
            j_CI=i%Nb;
            
            
            memset(bit_a,0,n_act*sizeof(int));
            memset(bit_b,0,n_act*sizeof(int));
            for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
            for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
            n_ext=0;
            for(int t=0; t<n_act;t++)if(bit_a[t]!=bit_b[t]){
                i_ext[n_ext]=t;
                n_ext++;
                bit_a[t]=0;
                bit_b[t]=1;
            }
            if((na==nb)&&(n_ext==2)){
                ind_done[j_CI*Nb+i_CI]=1;
                
            }
            else{
                E_min=H_diag_appr[i];
                e_a=(n_ext+na-nb)/2;
                e_b=(n_ext-na+nb)/2;
                Ne = (int) std::lround(tgammal(n_ext+1) / tgammal(e_a+1) / tgammal(e_b+1));
                get_vec(e_a, n_ext, Ne, vec_e);
                

                //may be i will make Hamiltonian and correct e_act for CSF but it is not really needed
                //here all H_appr[i,i] are set to minimal actue it is not bad
                for(int i_e=0;i_e<Ne;i_e++){
                    memset(bit_e,0,n_ext*sizeof(int));
                    for (int k=1; k<e_a+1; k++) bit_e[vec_e[i_e*(e_a+1)+k]-1] = 1;
                    
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=1;
                        bit_b[i_ext[t]]=0;
                    }
                    auto i_CIext = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                    auto j_CIext = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                    ind_done[i_CIext*Nb+j_CIext]=1;
                    if(E_min>H_diag_appr[i_CIext*Nb+j_CIext])E_min=H_diag_appr[i_CIext*Nb+j_CIext];
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=0;
                        bit_b[i_ext[t]]=1;
                    }
                    
                }
                for(int i_e=0;i_e<Ne;i_e++){
                    memset(bit_e,0,n_ext*sizeof(int));
                    for (int k=1; k<e_a+1; k++) bit_e[vec_e[i_e*(e_a+1)+k]-1] = 1;
                    
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=1;
                        bit_b[i_ext[t]]=0;
                    }
                    auto i_CIext = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                    auto j_CIext = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                    H_diag_appr[i_CIext*Nb+j_CIext]=E_min;
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=0;
                        bit_b[i_ext[t]]=1;
                    }
                    
                }
//                 fprintf(out_stream,"q\n");
            }
            
        }
        delete[] ind_done;
        delete[] vec_e   ;
        delete[] bit_e   ;
        delete[] i_ext   ;

    }
    
    return 0;
    
}

int aldet_data::H_diag_calc_PT(){
    
    if(H_diag     ==NULL) H_diag      = new double[/*n_states[0]**/Nd];
    if(H_diag_appr==NULL) H_diag_appr = new double[/*n_states[0]**/Nd];
    
    set_zero_matr(H_diag,/*n_states[0]**/Nd);
        
    int N=n_act;
    
    for(int i_CI =0/*i_th*/; i_CI<Na; i_CI++/*+=n_th*/)
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        H_diag[i_CI*Nb+j_CI]=E_core;
        
        memset(bit_a,0,N*sizeof(int));
        memset(bit_b,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            H_diag[i_CI*Nb+j_CI]+=F_act_A[t*N+t];
            
            for(int u=t+1;u<N;u++)if(bit_a[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS_AA[((t*N+t)*N+u)*N+u]-act_INTS_AA[((t*N+u)*N+u)*N+t]
                                                                      + T2_AA     [((u*N+t)*N+t)*N+u];///check it!!!!!!!
            for(int u=0  ;u<N;u++)if(bit_b[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS_AB[((t*N+t)*N+u)*N+u];
        }
        for(int t=0  ;t<N;t++)if(bit_b[t]!=0){
            H_diag[i_CI*Nb+j_CI]+=F_act_B[t*N+t];
//             
            for(int u=t+1;u<N;u++)if(bit_b[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS_BB[((t*N+t)*N+u)*N+u]-act_INTS_BB[((t*N+u)*N+u)*N+t]
                                                                      + T2_BB     [((u*N+t)*N+t)*N+u];
//             for(int u=0  ;u<N;u++)if(bit_b[u]!=0) H_diag[i_CI*Nb+j_CI]+=act_INTS[((t*N+t)*N+u)*N+u];
        }
        
//         fprintf(out_stream,"H[%d,%d]=%e\n",i_CI, j_CI, H_diag[i_CI*Nb+j_CI]);
//         if(j_CI==4) return 0;
    }
    memcpy(H_diag_appr,H_diag,Nd*sizeof(double));
    if(mult!=0){
        int * ind_done = new int[Nd];
        set_zero_matr_int(ind_done,Nd);
        double E_min;
        int i_CI,j_CI;
        int n_ext;
        int e_a, e_b, Ne;
        int * vec_e = new int [Na*(na+1)];
        int * bit_e = new int [n_act];
        int * i_ext = new int[n_act];

        for(int i=0;i<Nd;i++)if(ind_done[i]==0){
            
            i_CI=i/Nb;
            j_CI=i%Nb;
            
            
            memset(bit_a,0,n_act*sizeof(int));
            memset(bit_b,0,n_act*sizeof(int));
            for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
            for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
            n_ext=0;
            for(int t=0; t<n_act;t++)if(bit_a[t]!=bit_b[t]){
                i_ext[n_ext]=t;
                n_ext++;
                bit_a[t]=0;
                bit_b[t]=1;
            }
            if((na==nb)&&(n_ext==2)){
                ind_done[j_CI*Nb+i_CI]=1;
                
            }
            else{
                E_min=H_diag_appr[i];
                e_a=(n_ext+na-nb)/2;
                e_b=(n_ext-na+nb)/2;
                Ne = (int) std::lround(tgammal(n_ext+1) / tgammal(e_a+1) / tgammal(e_b+1));
                get_vec(e_a, n_ext, Ne, vec_e);
                

                //may be i will make Hamiltonian and correct e_act for CSF but it is not really needed
                //here all H_appr[i,i] are set to minimal actue it is not bad
                for(int i_e=0;i_e<Ne;i_e++){
                    memset(bit_e,0,n_ext*sizeof(int));
                    for (int k=1; k<e_a+1; k++) bit_e[vec_e[i_e*(e_a+1)+k]-1] = 1;
                    
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=1;
                        bit_b[i_ext[t]]=0;
                    }
                    auto i_CIext = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                    auto j_CIext = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                    ind_done[i_CIext*Nb+j_CIext]=1;
                    if(E_min>H_diag_appr[i_CIext*Nb+j_CIext])E_min=H_diag_appr[i_CIext*Nb+j_CIext];
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=0;
                        bit_b[i_ext[t]]=1;
                    }
                    
                }
                for(int i_e=0;i_e<Ne;i_e++){
                    memset(bit_e,0,n_ext*sizeof(int));
                    for (int k=1; k<e_a+1; k++) bit_e[vec_e[i_e*(e_a+1)+k]-1] = 1;
                    
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=1;
                        bit_b[i_ext[t]]=0;
                    }
                    auto i_CIext = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                    auto j_CIext = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                    H_diag_appr[i_CIext*Nb+j_CIext]=E_min;
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=0;
                        bit_b[i_ext[t]]=1;
                    }
                    
                }
//                 fprintf(out_stream,"q\n");
            }
            
        }
        delete[] ind_done;
        delete[] vec_e   ;
        delete[] bit_e   ;
        delete[] i_ext   ;

    }
    
    return 0;
    
}

        
int aldet_data::H_calc_PT2(double * H, int n_s){
    
    
    
    return 0;
}
        


int aldet_data::calc_spin_matr(double * O, double * C,int n_s, int ld){
    
    
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n_s,n_s,Nd,1.0,
                         C,ld,
                         C,ld,0.0,
                         O      ,n_s);
    
    double Sz =(na-nb)/2.0;
    for(int i=0;i<n_s*n_s;i++)O[i]=O[i]*(Sz*Sz-Sz);
    
    if(na==0    )return 0;
    if(nb==n_act)return 0;
    
    int NaM = (int) std::lround(tgammal(n_act+1) / tgammal(na  ) / tgammal(n_act-na+2));
    int NbM = (int) std::lround(tgammal(n_act+1) / tgammal(nb+2) / tgammal(n_act-nb  ));    
    
    int * faM = new int [(na-1) * n_act];
    int * fbM = new int [(nb+1) * n_act];
    get_factorials(na-1, n_act, faM);
    get_factorials(nb+1, n_act, fbM);
    
    double * S_M_c = new double[NaM*NbM*n_s];
    set_zero_matr(S_M_c, NaM*NbM*n_s);
    
    
    for(int i_CI = 0; i_CI<Na; i_CI++)
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        memset(bit_a,0,n_act*sizeof(int));
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        for(int t=0  ;t<n_act;t++)if(bit_a[t]!=0)if(bit_b[t]==0){
            bit_a[t]=0;
            bit_b[t]=1;
            auto i_CIext = get_ind_from_ON(bit_a, n_act, na-1, faM, buf);
            auto j_CIext = get_ind_from_ON(bit_b, n_act, nb+1, fbM, buf);
            for(int i_s=0;i_s<n_s;i_s++)
                S_M_c[(i_CIext*NbM+j_CIext)*n_s+i_s]+= C[(i_CI*Nb+j_CI)*ld+i_s]*
                                                       a_spin_sign[i_CI*n_act+t]*
                                                       b_spin_sign[j_CI*n_act+t];
            
            bit_a[t]=1;
            bit_b[t]=0;
            
        }
            
    }
    
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n_s,n_s,NaM*NbM,1.0,
                         S_M_c,n_s,
                         S_M_c,n_s,1.0,
                         O    ,n_s);
    
//     PrintMatr(O, n_s,n_s,0);
//     for(int i_s=0;i_s<n_s;i_s++)
//         fprintf(out_stream,"%d - %e\n",i_s,O[i_s*(n_s+1)]);
//     getchar();
    delete[] faM   ;
    delete[] fbM   ;
    delete[] S_M_c ;
    
    return 0;
}

int aldet_data::calc_L2_matr(double * O, int i_s,int n_s, int ld){
    
       
    
    if(Hv==NULL) {
        Hv     = new double[n_states[i_s]*Na*Nb];
        Hv_buf = new double[n_states[i_s]*Na*Nb];
        set_zero_matr(Hv,n_states[i_s]*Na*Nb);
    }
    transpose_ci(i_s);
    
    for(int i=0 ;i<Na ;i++)
    for(int j=0 ;j<Nb ;j++)
    for(int k=0;k<n_s;k++)
        Hv[(i*Nb+j)*n_states[i_s]+k]=0;
    
    
    set_zero_matr(Hv_buf,n_states[i_s]*Na*Nb);
    
    double * backup;
    backup=F_act_A;
    F_act_A=Lambda_act;
    
#pragma omp parallel
    {
        int nt = omp_get_thread_num();
        H_mult_A  (Hv_buf, 0, n_s, n_states[i_s], coef_asb[i_s], nt, num_threads);
    }
    
    F_act_A=backup;
    transpose_3d_abc_to_acb_part_b_sum(Hv, Hv_buf, Na, 0, n_s, n_states[i_s], Nb);
    set_zero_matr(Hv_buf,n_states[i_s]*Na*Nb);
    
    backup=F_act_B;
    F_act_B=Lambda_act;
#pragma omp parallel
    {
        int nt = omp_get_thread_num();
        H_mult_B  (Hv_buf, 0, n_s, n_states[i_s], coef_bsa[i_s], nt, num_threads);
    }
    F_act_B=backup;
    transpose_3d_abc_to_cab_part_b_sum(Hv, Hv_buf, Nb, 0, n_s, n_states[i_s], Na);

#pragma omp parallel for    
    for(int i=0;i<Nd;i++)
        for(int j=0;j<n_s;j++)
            Hv[i*n_states[i_s]+j]+=Lambda_core*coef[i_s][i*n_states[i_s]+j];
    
     


    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n_s,n_s,Na*Nb,1.0,
                         Hv,n_states[i_s],
                         Hv,n_states[i_s],0.0,
                         O    ,n_s);
    
    
    return 0;
}

int aldet_data::calc_P_matr(double * O, int i_s,int n_s, int ld){
    
    if(Hv==NULL) {
        Hv     = new double[n_states[i_s]*Na*Nb];
    }
    
    
    double * sign_rep_act = new double[n_act];
    
    for(int i=0; i< n_act;i++){
        if(act_rep_num[i]== 0)sign_rep_act[i]= 1.0;
        if(act_rep_num[i]== 1)sign_rep_act[i]=-1.0;
        if(act_rep_num[i]== 2)sign_rep_act[i]= 1.0;
        if(act_rep_num[i]== 3)sign_rep_act[i]=-1.0;
        if(act_rep_num[i]== 4)sign_rep_act[i]= 1.0;
        if(act_rep_num[i]== 5)sign_rep_act[i]=-1.0;
        if(act_rep_num[i]== 6)sign_rep_act[i]= 1.0;
        if(act_rep_num[i]== 7)sign_rep_act[i]=-1.0;
        if(act_rep_num[i]== 8)sign_rep_act[i]= 1.0;
        if(act_rep_num[i]== 9)sign_rep_act[i]=-1.0;
        if(act_rep_num[i]==10)sign_rep_act[i]= 1.0;
        
//         printf("r[%d]=%e\n",i,sign_rep_act[i]);
    }
    
    double * sign_a = new double[Na];
    double * sign_b = new double[Nb];
    
    for(int i_CI = 0; i_CI<Na; i_CI++){
        
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        sign_a[i_CI]=1.0;
        for(int t=0  ;t<n_act;t++)if(bit_a[t]!=0)sign_a[i_CI]=sign_a[i_CI]*sign_rep_act[t];
            
    }
    
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        sign_b[j_CI]=1.0;
        for(int t=0  ;t<n_act;t++)if(bit_b[t]!=0)sign_b[j_CI]=sign_b[j_CI]*sign_rep_act[t];
        
            
    }
    delete[]sign_rep_act;
    
    for(int i_CI = 0; i_CI<Na ; i_CI++)
    for(int j_CI = 0; j_CI<Nb ; j_CI++)
    for(int k_s  = 0; k_s <n_s; k_s ++)
        Hv[(i_CI*Nb+j_CI)*n_states[i_s]+k_s]=coef[i_s][(i_CI*Nb+j_CI)*ld+k_s]*sign_a[i_CI]*sign_b[j_CI];
            
    delete[]sign_a;
    delete[]sign_b;
    

    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n_s,n_s,Na*Nb,1.0,
                         coef[i_s],ld,
                         Hv,n_states[i_s],0.0,
                         O    ,n_s);
    
    return 0;
}


int aldet_data::print_states(int i_s, int n_s, int print){
    
    double * S2_loc = new double[n_s*n_s];
    double * L2_loc;
    double * P_loc;
    int * max_coef = new int[10];
    calc_spin_matr(S2_loc,coef[i_s],n_s,n_states[i_s]);
    if(LINEAR){
        L2_loc = new double[n_s*n_s];
        P_loc  = new double[n_s*n_s];
        calc_L2_matr(L2_loc,i_s,n_s,n_states[i_s]);
        calc_P_matr ( P_loc,i_s,n_s,n_states[i_s]);
    }
        
    
    for(int i=0; i<n_s; i++){
        S2[i_s][i]=S2_loc[i*n_s+i];
        if(print)fprintf(out_stream,"State %d  E  = % 18.10f S^2 = %.2f",i,E_states[i_s][i],S2_loc[i*n_s+i]);
        if(LINEAR){
            L2[i_s][i]=L2_loc[i*n_s+i];
            P [i_s][i]= P_loc[i*n_s+i];
            if(print)fprintf(out_stream," L^2 = %.2f",L2_loc[i*n_s+i]);
            if(print)if(L2_loc[i*n_s+i]<1e-3){
                if     (fabs(P_loc[i*n_s+i]-1.0)<1e-8)fprintf(out_stream," (+)");
                else if(fabs(P_loc[i*n_s+i]+1.0)<1e-8)fprintf(out_stream," (-)");
                else                                  fprintf(out_stream," (?)");
            }
        }
        
        if(print){
            fprintf(out_stream,":\n");
            find_max_coef(max_coef, print_number, coef[i_s], Nd, i, n_states[i_s]);
            for(int j=0;j<print_number;j++){
                fprintf(out_stream,"% .10e  | ",coef[i_s][max_coef[j]*n_states[i_s]+i]);
                printf_occ_a(max_coef[j]/Nb);
                fprintf(out_stream," | ");
                printf_occ_b(max_coef[j]%Nb);
                fprintf(out_stream,"\n");
            }
        }
    }
    
    delete[] max_coef;
    delete[] S2_loc;
    if(LINEAR)delete[] L2_loc;
    if(LINEAR)delete[]  P_loc;
    
    return 0;
}





int aldet_data::gen_bf(sparsed_CI_vec * s, int n_s){
    
    int * gues_num = new int[Nd];
    int * i_ext = new int[n_act];
    int i_CI,j_CI;
    int i,j;
    int i_s;
    int n_ext;
    int e_a, e_b, Ne;
//     fprintf(out_stream,"Na= %d\n",Na);
//     fprintf(out_stream,"Nb= %d\n",Nb);
//     fprintf(out_stream,"na= %d\n",na);
//     fprintf(out_stream,"nb= %d\n",nb);
//     exit(0);
    int * vec_e = new int [max(Na*(na+1),Nb*(nb+1))];
    int * bit_e = new int [n_act];
    int bf_found;
    
    int n1=0;
    int n2=0;
    int n3=0;
    
    double sign=1;
    if(mult==3)sign=-1;
    
    find_N_min_els(gues_num, Nd,H_diag,Nd);
//     for(i=0,i_s=0;i<Nd;i++)gues_num[i]=i;
    
    int diag=1;
    if(mult==0)  diag=0;
    if(na==0)    diag=0;
    if(nb==0)    diag=0;
    if(na==n_act)diag=0;
    if(nb==n_act)diag=0;
    
    if(diag==0)
    for(i=0,i_s=0;i<Nd;i++){
        bf_found=0;
        for(int j=0;j<i_s;j++)if(gues_num[i]==s[j].n[0])bf_found=1;
        if(bf_found)continue;
        i_CI=gues_num[i]/Nb;
        j_CI=gues_num[i]%Nb;
        
        
        memset(bit_a,0,n_act*sizeof(int));
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        n_ext=0;
        for(int t=0; t<n_act;t++)if(bit_a[t]!=bit_b[t]){
            i_ext[n_ext]=t;
            n_ext++;
            bit_a[t]=0;
            bit_b[t]=1;
        }
        if(n_ext==0){
            s[i_s].c.push_back(1.0);
            s[i_s].n.push_back(gues_num[i]);
            i_s++;
            n1++;
        }
        else if((na==nb)&&(n_ext==2)){
            s[i_s].c.push_back(1.0);
            s[i_s].n.push_back(gues_num[i]);
            i_s++;
            s[i_s].c.push_back(1.0);
            s[i_s].n.push_back(j_CI*Nb+i_CI);
            i_s++;
//             fprintf(out_stream,"t\n");
            n2++;
        }
        else{
            n3++;
            e_a=(n_ext+na-nb)/2;
            e_b=(n_ext-na+nb)/2;
            Ne = (int) std::lround(tgammal(n_ext+1) / tgammal(e_a+1) / tgammal(e_b+1));
            get_vec(e_a, n_ext, Ne, vec_e);
            for(int i_e=0;i_e<Ne;i_e++){
                memset(bit_e,0,n_ext*sizeof(int));
                for (int k=1; k<e_a+1; k++) bit_e[vec_e[i_e*(e_a+1)+k]-1] = 1;
                
                
                for(int t=0; t<n_ext;t++)if(bit_e[t]){
                    bit_a[i_ext[t]]=1;
                    bit_b[i_ext[t]]=0;
                }
                auto i_CIext = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                auto j_CIext = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
                s[i_s].c.push_back(1.0);
                s[i_s].n.push_back(i_CIext*Nb+j_CIext);
                i_s++;
                for(int t=0; t<n_ext;t++)if(bit_e[t]){
                    bit_a[i_ext[t]]=0;
                    bit_b[i_ext[t]]=1;
                }
                
            }
//             fprintf(out_stream,"q\n");
        }
        if(i_s>n_s)break;
//         
    }
    else{
        
        double Sz =(na-nb)/2.0;
//         for(int i=0;i<n_s*n_s;i++)O[i]=O[i]*(Sz*Sz-Sz);
        
        int NaM = (int) std::lround(tgammal(n_act+1) / tgammal(na  ) / tgammal(n_act-na+2));
        int NbM = (int) std::lround(tgammal(n_act+1) / tgammal(nb+2) / tgammal(n_act-nb  ));    
        
        int * faM = new int [(na-1) * n_act];
        int * fbM = new int [(nb+1) * n_act];
        get_factorials(na-1, n_act, faM);
        get_factorials(nb+1, n_act, fbM);
        
        double ** S = new double*[n_act];
        for(int i=0;i<n_act;i++)S[i]=NULL;
        
        double ** S_eval = new double*[n_act];
        for(int i=0;i<n_act;i++)S_eval[i]=NULL;
        
        double ** Sc = new double*[n_act];
        for(int i=0;i<n_act;i++)Sc[i]=NULL;
        
        int ** Sn = new int*[n_act];
        for(int i=0;i<n_act;i++)Sn[i]=NULL;
        
        
        for(i=0,i_s=0;i<Nd;i++){
            
            bf_found=0;
            for(int j=0;j<i_s;j++)
            for(int k=0;k<s[j].n.size();k++)
                if(gues_num[i]==s[j].n[k]){
                    bf_found=1;
                    break;
                }
                    
            if(bf_found)continue;
            i_CI=gues_num[i]/Nb;
            j_CI=gues_num[i]%Nb;
            
            
            memset(bit_a,0,n_act*sizeof(int));
            memset(bit_b,0,n_act*sizeof(int));
            for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
            for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
            n_ext=0;
            for(int t=0; t<n_act;t++)if(bit_a[t]!=bit_b[t]){
                i_ext[n_ext]=t;
                n_ext++;
                bit_a[t]=0;
                bit_b[t]=1;
            }
            if(n_ext==0){
                if(mult==1){
                    s[i_s].c.push_back(1.0);
                    s[i_s].n.push_back(gues_num[i]);
                    i_s++;
//                     fprintf(out_stream,"s\n");
                    n1++;
                }
            }
            else if((na==nb)&&(n_ext==2)){
                if(mult<4){
                    s[i_s].c.push_back(0.5*sqrt(2));
                    s[i_s].n.push_back(gues_num[i]);
//                     i_s++;
                    s[i_s].c.push_back(0.5*sqrt(2)*sign);
                    s[i_s].n.push_back(j_CI*Nb+i_CI);
                    i_s++;
//                     fprintf(out_stream,"t\n");
                    n2++;
                }
            }
            else if(n_ext>=(mult-1)){
                n3++;
                e_a=(n_ext+na-nb)/2;
                e_b=(n_ext-na+nb)/2;
                Ne = (int) std::lround(tgammal(n_ext+1) / tgammal(e_a+1) / tgammal(e_b+1));
                get_vec(e_a, n_ext, Ne, vec_e);
                if(Sn    [n_ext-1]==NULL) Sn    [n_ext-1] = new int   [Ne*n_act];
                if(Sc    [n_ext-1]==NULL) Sc    [n_ext-1] = new double[Ne*n_act];
                if(S     [n_ext-1]==NULL) S     [n_ext-1] = new double[Ne*Ne   ];
                if(S_eval[n_ext-1]==NULL) S_eval[n_ext-1] = new double[Ne      ];
                set_zero_matr(Sc[n_ext-1],Ne*n_act);
                for(int i_e=0;i_e<Ne;i_e++){
                    memset(bit_e,0,n_ext*sizeof(int));
                    for (int k=1; k<e_a+1; k++) bit_e[vec_e[i_e*(e_a+1)+k]-1] = 1;
                    
                    
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=1;
                        bit_b[i_ext[t]]=0;
                    }
                    auto i_CIext = get_ind_from_ON(bit_a, n_act, na, fa, buf);
                    auto j_CIext = get_ind_from_ON(bit_b, n_act, nb, fb, buf);
//                     s[i_s].c.push_back(1.0);
                    s[i_s].n.push_back(i_CIext*Nb+j_CIext);
                    
                    for(int t=0;t<n_act;t++){
                        Sn[n_ext-1][i_e*n_act+t]=-1;
                        if(bit_a[t]==1)if(bit_b[t]==0){
                            Sc[n_ext-1][i_e*n_act+t]=a_spin_sign[i_CIext*n_act+t]*
                                            b_spin_sign[j_CIext*n_act+t];
                            bit_a[t]=0;
                            bit_b[t]=1;
                            Sn[n_ext-1][i_e*n_act+t] = get_ind_from_ON(bit_a, n_act, na-1, faM, buf)*NbM+
                                              get_ind_from_ON(bit_b, n_act, nb+1, fbM, buf);
                            
                            bit_a[t]=1;
                            bit_b[t]=0;
                        }
                    }
                    
                    for(int t=0; t<n_ext;t++)if(bit_e[t]){
                        bit_a[i_ext[t]]=0;
                        bit_b[i_ext[t]]=1;
                    }
                    
                }
                for(int i_e=0;i_e<Ne;i_e++)
                for(int j_e=0;j_e<Ne;j_e++){
                    S[n_ext-1][i_e*Ne+j_e]=0;
                    for(int t=0;t<n_act;t++)
                    for(int u=0;u<n_act;u++)
                        if(Sn[n_ext-1][i_e*n_act+t]!=-1)
                        if(Sn[n_ext-1][i_e*n_act+t]==Sn[n_ext-1][j_e*n_act+u])
                            S[n_ext-1][i_e*Ne+j_e]+=Sc[n_ext-1][i_e*n_act+t]*Sc[n_ext-1][j_e*n_act+u];
                
                }
                for(int i_e=0;i_e<Ne;i_e++)S[n_ext-1][i_e*Ne+i_e]+=Sz*Sz-Sz;
                
                
                
                lapack_diag(S[n_ext-1],S_eval[n_ext-1],Ne);
//                 PrintMatr(S_eval[n_ext-1],1,Ne,0);
//                 PrintMatr(S[n_ext-1],Ne,Ne,0);
                
                for(int i_e=0,w=0;i_e<Ne;i_e++)
                    if(fabs(S_eval[n_ext-1][i_e]-(mult*mult-1.0)/4.0)<1e-8){// S2 =S*(S+1),S=(mult-1)/2
                        for(int j_e=0;j_e<Ne;j_e++)
                            s[i_s].c.push_back(S[n_ext-1][i_e*Ne+j_e]);
                        if(w)
                            s[i_s].n=s[i_s-1].n;
                        w=1;
//                         for(int j_e=0;j_e<Ne;j_e++)
//                             fprintf(out_stream,"%e(%d)\t",s[i_s].c[j_e],s[i_s].n[j_e]);
//                         fprintf(out_stream,"\n");
                        i_s++;
                    }
                
//                 getchar();
            }
//             fprintf(stderr,"%d %d\n",i_s,i);
            if(i_s>n_s)break;
//             
        }
        for(int i=0;i<n_act;i++)if(S_eval[i]!=NULL)delete[] S_eval[i];
        for(int i=0;i<n_act;i++)if(S     [i]!=NULL)delete[] S     [i];
        for(int i=0;i<n_act;i++)if(Sc    [i]!=NULL)delete[] Sc    [i];
        for(int i=0;i<n_act;i++)if(Sn    [i]!=NULL)delete[] Sn    [i];
        
        delete[] S;
        delete[] S_eval;
        delete[] Sc;
        delete[] Sn;
        delete[] faM;
        delete[] fbM;
    }
    
    delete[] gues_num;
    delete[] i_ext   ;
    delete[] vec_e   ;
    delete[] bit_e   ;

//     fprintf(out_stream,"%d %d %d %d\n",n1,n2,n3,n1+2*n2+6*n3);
//     fprintf(out_stream,"i_s=%d\n",i_s);
//     getchar();
    
    if (i_s==0){
        printf("ERROR: could not find CSF with mult=%d\n",mult);
        printf("       check your active space parameters\n");
        exit(0);
    }
    
    return i_s;
}

int aldet_data::symmetrization(int i_set){
    
    if(i_set>=n_sets){
        fprintf(out_stream,"unable to run symmetrization for i_set = %d and n_sets = %d\nfor correct run you nedd^ i < n_sets\n", i_set, n_sets);
        exit(1);
    }

    return 0;
    
//     if(mult==0)
//         return 0;
//     if(na!=nb)
//         return 0;
//     
//     for(int i=0; i<Na;i++)
//     for(int j=i; j<Na;j++)
//     for(int s=0; s<n_states[i_set];s++){
//         coef[i_set][(i*Na+j)*n_states[i_set]+s]=0.5*(coef[i_set][(i*Na+j)*n_states[i_set]+s]+
//                                                      coef[i_set][(j*Na+i)*n_states[i_set]+s]*sym_ab);
//         coef[i_set][(j*Na+i)*n_states[i_set]+s]=     coef[i_set][(i*Na+j)*n_states[i_set]+s]*sym_ab;
//     }
//     
    
    return 0;
}


int aldet_data::printf_occ_a(int i_CI){
    
    memset(bit_a,0,n_act*sizeof(int));
    
    for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
    for (int t=0; t<n_act;t++)
        if(bit_a[t])fprintf(out_stream,"1");
        else        fprintf(out_stream,"0");
    
        return 0;
}

int aldet_data::printf_occ_b(int i_CI){
    
    memset(bit_b,0,n_act*sizeof(int));
    
    for (int k=1; k<nb+1; k++) bit_b[vec_b[i_CI*(nb+1)+k]-1] = 1;
    for (int t=0; t<n_act;t++)
        if(bit_b[t])fprintf(out_stream,"1");
        else        fprintf(out_stream,"0");
    
        return 0;
}

int aldet_data::printf_occ_sum(int i_CI, int j_CI){
    
    memset(bit_a,0,n_act*sizeof(int));
    
    for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1]++;
    for (int k=1; k<nb+1; k++) bit_a[vec_b[j_CI*(nb+1)+k]-1]++;
    for (int t=0; t<n_act;t++)
        fprintf(out_stream,"%d",bit_a[t]);
    
        return 0;
}



aldet_data::~aldet_data(){
    
    if(fa    != NULL)delete[] fa;
    if(fb    != NULL)delete[] fb;
    if(vec_a != NULL)delete[] vec_a;
    if(vec_b != NULL)delete[] vec_b;

    if(bit_a != NULL)delete[] bit_a;
    if(bit_b != NULL)delete[] bit_b;
    if(buf   != NULL)delete[] buf;//more space for +a-b and -a+b ????
    
    if(n_states != NULL)delete[] n_states;

    if(coef    !=NULL)for(int i=0;i<n_sets;i++)if(coef    [i]!=NULL)delete[] coef    [i];delete[] coef    ;
    if(coef_bas!=NULL)for(int i=0;i<n_sets;i++)if(coef_bas[i]!=NULL)delete[] coef_bas[i];delete[] coef_bas;
    if(coef_asb!=NULL)for(int i=0;i<n_sets;i++)if(coef_asb[i]!=NULL)delete[] coef_asb[i];delete[] coef_asb;
    if(coef_bsa!=NULL)for(int i=0;i<n_sets;i++)if(coef_bsa[i]!=NULL)delete[] coef_bsa[i];delete[] coef_bsa;
    if(E_states!=NULL)for(int i=0;i<n_sets;i++)if(E_states[i]!=NULL)delete[] E_states[i];delete[] E_states;
    if(S2      !=NULL)for(int i=0;i<n_sets;i++)if(S2      [i]!=NULL)delete[] S2      [i];delete[] S2      ;
    if(L2      !=NULL)for(int i=0;i<n_sets;i++)if(L2      [i]!=NULL)delete[] L2      [i];delete[] L2      ;
    if(P       !=NULL)for(int i=0;i<n_sets;i++)if(P       [i]!=NULL)delete[] P       [i];delete[] P       ;
    if(E_act   !=NULL)for(int i=0;i<n_sets;i++)if(E_act   [i]!=NULL)delete[] E_act   [i];delete[] E_act   ;
    
//     if(sym_ab  !=NULL)delete[] sym_ab  ;
    
    if(H_diag      !=NULL) delete[] H_diag     ;
    if(H_diag_appr !=NULL) delete[] H_diag_appr;
    
    if(Hv        != NULL) delete[] Hv        ;
    if(Hv_buf    != NULL) delete[] Hv_buf    ;
    
    if(Ia         != NULL) delete[] Ia        ;
    if(Ia_friends != NULL) delete[] Ia_friends;
    if(Ib         != NULL) delete[] Ib        ;
    if(Ib_friends != NULL) delete[] Ib_friends;
    
    if(J_act_a!= NULL) delete[] J_act_a;
    if(J_act_b!= NULL) delete[] J_act_b;
    if(K_act_a!= NULL) delete[] K_act_a;
    if(K_act_b!= NULL) delete[] K_act_b;
    
    if(e1_ind_a      != NULL) delete[]e1_ind_a      ;
    if(e1_ind_b      != NULL) delete[]e1_ind_b      ;
    if(e1_orbs_a     != NULL) delete[]e1_orbs_a     ;
    if(e1_orbs_b     != NULL) delete[]e1_orbs_b     ;
    if(e1_sign_a     != NULL) delete[]e1_sign_a     ;
    if(e1_sign_b     != NULL) delete[]e1_sign_b     ;
//     if(e1_asm_ints_b != NULL) delete[]e1_asm_ints_b ;
    
    if(e2_sign_a  !=NULL) delete[] e2_sign_a ;
    if(e2_sign_b  !=NULL) delete[] e2_sign_b ;
    if(e2_orbs_a  !=NULL) delete[] e2_orbs_a ;
    if(e2_orbs_b  !=NULL) delete[] e2_orbs_b ;
    if(e2_ind_a != NULL) delete[] e2_ind_a;  
    if(e2_ind_b != NULL) delete[] e2_ind_b; 
    if(e2_V_a   != NULL) delete[] e2_V_a  ;  
    if(e2_V_b   != NULL) delete[] e2_V_b  ;  
    
    if(e3_ind_a != NULL) delete[] e3_ind_a;
    if(e3_ind_b != NULL) delete[] e3_ind_b;
    if(e3_V_a   != NULL) delete[] e3_V_a  ;
    if(e3_V_b   != NULL) delete[] e3_V_b  ;
    
    if(a_spin_sign != NULL) delete[] a_spin_sign;
    if(b_spin_sign != NULL) delete[] b_spin_sign;
    
    if(F_act_A    !=NULL) delete[] F_act_A    ;
    if(F_act_B    !=NULL) delete[] F_act_B    ;
    if(act_INTS_AA!=NULL) delete[] act_INTS_AA;
    if(act_INTS_AB!=NULL) delete[] act_INTS_AB;
    if(act_INTS_BB!=NULL) delete[] act_INTS_BB;
    if(T3_AAA    !=NULL) delete[] T3_AAA    ;
    if(T3_AAB    !=NULL) delete[] T3_AAB    ;
    if(T3_BBA    !=NULL) delete[] T3_BBA    ;
    if(T3_BBB    !=NULL) delete[] T3_BBB    ;
    if(T2_AA     !=NULL) delete[] T2_AA     ;
    if(T2_AB     !=NULL) delete[] T2_AB     ;
    if(T2_BB     !=NULL) delete[] T2_BB     ;
    if(T1_A      !=NULL) delete[] T1_A      ;
    if(T1_B      !=NULL) delete[] T1_B      ;

    
}

int ci_from_ci(const int& N, const int& na, const int& nb, aldet_data * CI, int s, int f, const int& i_state, double *& ci){
    int Na = (int) std::lround(tgammal(N+1) / tgammal(na+1) / tgammal(N-na+1));
    int Nb = (int) std::lround(tgammal(N+1) / tgammal(nb+1) / tgammal(N-nb+1)); 
    int * fa = new int [na * N];
    int * fb = new int [nb * N];
    get_factorials(na, N, fa);
    get_factorials(nb, N, fb);
    int * buf_a = new int [na+1];
    int * buf_b = new int [nb+1];
    buf_a[0]=0;
    buf_b[0]=0;
    ci = new double [Na*Nb];
    
    int * bit_a = new int [N+1];
    int * bit_b = new int [N+1];
    
    int n_vt = N-CI->n_act;
    
    for (auto i = 0u; i<Na*Nb; i++) ci[i]=0.0;
    
    for(int i_CI=0;i_CI<CI->Na;i_CI++){
        memset(bit_a,0,(N+1)*sizeof(int));
        for (int k=1; k<CI->na+1; k++) bit_a[CI->vec_a[i_CI*(CI->na+1)+k]-1+n_vt] = 1;
        auto ind_a = get_index_from_array(bit_a, na, N, fa, buf_a, 0);
        
        for(int j_CI=0;j_CI<CI->Nb;j_CI++){
            memset(bit_b,0,(N+1)*sizeof(int));
            for (int k=1; k<CI->nb+1; k++) bit_b[CI->vec_b[j_CI*(CI->nb+1)+k]-1+n_vt] = 1;
            auto ind_b = get_index_from_array(bit_b, nb, N, fb, buf_b, 0);
            
            ci[ind_a*Nb + ind_b] = CI->coef[0][(i_CI*CI->Nb+j_CI)*CI->n_states[0]+i_state];
            
        }
        
    }
    
//     printf("S_cfh = %.10e\n",cblas_ddot(Na*Nb, ci, 1, ci, 1));
    
    delete[] fa;
    delete[] fb;
    delete[] buf_a;
    delete[] buf_b;
    
    return 0;
}

int aldet_copy(aldet_data * O, aldet_data * I){
    
    //check mol_link.cpp:396
    O->get_dim(I->n_act, I->na, I->nb, I->n_sets, I->mult, I->print_number, I->alloc_mode);
    O->n_sets=I->n_sets;
    for(int i=0;i<O->n_sets;i++){
        O->n_states[i]=I->n_states[i];
        O->copy_coef(i, I, O->n_states[i], i, 1);
        
    }
    
    return 0;
}

    
int ci_pr_aldet_first_order(ci_pr_vec_aldet * O, aldet_data * I, int n_orb, double ** S, int nA, int nB, int n_st){
    
//     fprintf(out_stream,"ci_1st: %d %d\n", nA,nB);
//     getchar();
    
    (*O).resize(2*n_orb+1);
    O->at(0).n_tr=0;
    O->at(0).n_A=nA;
    O->at(0).n_B=nB;
    O->at(0).order=0;
//     O->at(0).B.set_empty_key(-1);
    O->at(0).key.reset();
    
        
    O->at(0).B.get_dim((*I).n_act, (*I).na, (*I).nb, (*I).n_sets, (*I).mult, (*I).print_number);
    for(int i=0;i<(*I).n_sets;i++){
        O->at(0).B.n_states[i]=(*I).n_states[i];
        O->at(0).B.coef   [i]=new double[O->at(0).B.Nd*O->at(0).B.n_states[i]];
        O->at(0).B.coef_bas[i]=new double[O->at(0).B.Nd*O->at(0).B.n_states[i]];
        for(int j=0;j<(*I).Na*(*I).Nb*(*I).n_states[i];j++)O->at(0).B.coef[i][j]=(*I).coef[i][j];
        transpose_3d_abc_to_bac(O->at(0).B.coef_bas[i], O->at(0).B.coef[i], O->at(0).B.Na, O->at(0).B.Nb, O->at(0).B.n_states[i]);
    }
        
        
    ci_key tmp_key;
    double sign;
    for(int i_e=0;i_e<n_orb;i_e++){
        O->at(i_e+1).n_tr=1;
        O->at(i_e+1).n_A=nA+1;
        O->at(i_e+1).n_B=nB;
        O->at(i_e+1).order=1;
        O->at(i_e+1).key.reset();
        O->at(i_e+1).key.set(i_e);
//         O->at(i+1).B.set_empty_key(-1);
        aldet_data * B = &(O->at(i_e+1).B);
        
        (*B).get_dim((*I).n_act, (*I).na-1, (*I).nb, (*I).n_sets, (*I).mult, (*I).print_number);
        for(int i=0;i<(*I).n_sets;i++){
            (*B).n_states[i]=(*I).n_states[i];
            (*B).coef   [i]=new double[(*B).Nd*(*B).n_states[i]];
            (*B).coef_bas[i]=new double[(*B).Nd*(*B).n_states[i]];
            set_zero_matr((*B).coef[i],(*B).Na*(*B).Nb*(*B).n_states[i]);
            
            for(int i_CI = 0; i_CI<(*I).Na; i_CI++){
                
                memset((*B).bit_a,0,(*B).n_act*sizeof(int));
                for (int k=1; k<(*I).na+1; k++) (*B).bit_a[(*I).vec_a[i_CI*((*I).na+1)+k]-1] = 1;
                
                sign=1;
                for(int j=0;j<n_orb;j++)
                if((*B).bit_a[j]){
                    (*B).bit_a[j]=0;
                    auto i_CIext = get_ind_from_ON((*B).bit_a, (*B).n_act, (*B).na, (*B).fa, (*B).buf);
                    for(int j_CI = 0; j_CI<(*B).Nb; j_CI++)
                    for(int i_s = 0; i_s<(*B).n_states[i]; i_s++){
                        (*B).coef[i][(i_CIext*(*B).Nb+j_CI)*(*B).n_states[i]+i_s]+=
                        (*I).coef[i][(i_CI   *(*I).Nb+j_CI)*(*I).n_states[i]+i_s]*S[i][j*n_orb+i_e]*sign;
                    }
                    (*B).bit_a[j]=1;
                    sign =-sign;
                }
            }
            transpose_3d_abc_to_bac((*B).coef_bas[i], (*B).coef[i], (*B).Na, (*B).Nb, (*B).n_states[i]);
        }
//         fprintf(out_stream,"\n\nelement %d:\n",i+1);
//         print_ci_map_arr_el(&(O->at(i+1).B),n_orb,0);
//         fprintf(out_stream,"\n\nelement %d size = %d\n",i+1,O->at(i+1).B.size());
//         getchar();
    }
    for(int i_e=0;i_e<n_orb;i_e++){
        O->at(i_e+n_orb+1).n_tr=1;
        O->at(i_e+n_orb+1).n_A=nA;
        O->at(i_e+n_orb+1).n_B=nB+1;
        O->at(i_e+n_orb+1).order=1;
        O->at(i_e+n_orb+1).key.reset();
        O->at(i_e+n_orb+1).key.set(i_e+CI_MAX_SPACE);
        aldet_data * B = &(O->at(i_e+n_orb+1).B);
        
        (*B).get_dim((*I).n_act, (*I).na, (*I).nb-1, (*I).n_sets, (*I).mult, (*I).print_number);
        for(int i=0;i<(*I).n_sets;i++){
            (*B).n_states[i]=(*I).n_states[i];
            (*B).coef   [i]=new double[(*B).Nd*(*B).n_states[i]];
            (*B).coef_bas[i]=new double[(*B).Nd*(*B).n_states[i]];
            set_zero_matr((*B).coef[i],(*B).Na*(*B).Nb*(*B).n_states[i]);
            
            for(int j_CI = 0; j_CI<(*I).Nb; j_CI++){
                
                memset((*B).bit_b,0,(*B).n_act*sizeof(int));
                for (int k=1; k<(*I).nb+1; k++) (*B).bit_b[(*I).vec_b[j_CI*((*I).nb+1)+k]-1] = 1;
                
                sign=1;
                for(int j=0;j<n_orb;j++)
                if((*B).bit_b[j]){
                    (*B).bit_b[j]=0;
                    auto j_CIext = get_ind_from_ON((*B).bit_b, (*B).n_act, (*B).nb, (*B).fb, (*B).buf);
                    for(int i_CI = 0; i_CI<(*B).Na; i_CI++)
                    for(int i_s = 0; i_s<(*B).n_states[i]; i_s++){
                        (*B).coef[i][(i_CI*(*B).Nb+j_CIext)*(*B).n_states[i]+i_s]+=
                        (*I).coef[i][(i_CI*(*I).Nb+j_CI   )*(*I).n_states[i]+i_s]*S[i][j*n_orb+i_e]*sign;
                    }
                    (*B).bit_b[j]=1;
                    sign =-sign;
                }
            }
            transpose_3d_abc_to_bac((*B).coef_bas[i], (*B).coef[i], (*B).Na, (*B).Nb, (*B).n_states[i]);
        }
//         fprintf(out_stream,"\n\nelement %d:\n",i+1);
//         print_ci_map_arr_el(&(O->at(i+1).B),n_orb,0);
//         fprintf(out_stream,"\n\nelement %d size = %d\n",i+1,O->at(i+1).B.size());
//         getchar();
    }
    
    
    
    fprintf(out_stream,"M - %ld\n",O[0].size());
//     getchar();
//     print_ci_map(O+1,max);
    
    return 0;
}

int ci_pr_aldet::add_A(aldet_data * I, int n_si, int n_so){
    
    ci_key tmp_key;
    
    int dna=0;
    int dnb=0;
    for(int i=0; i<(*I).n_act; i++)if(key[i             ])dna++;
    for(int i=0; i<(*I).n_act; i++)if(key[i+CI_MAX_SPACE])dnb++;
    
//     fprintf(out_stream,"dn %d %d\n",dna,dnb);
//     getchar();
    
    if(A.n_act==0)A.get_dim((*I).n_act, (*I).na+dna, (*I).nb+dnb, (*I).n_sets, (*I).mult, (*I).print_number);
    A.n_states[n_so]=(*I).n_states[n_si];
    A.coef   [n_so]=new double[A.Nd*A.n_states[n_so]];
    A.coef_bas[n_so]=new double[A.Nd*A.n_states[n_so]];
    set_zero_matr(A.coef[n_so],A.Nd*A.n_states[n_so]);
    for(int i_CI=0;i_CI<(*I).Na;i_CI++)
    for(int j_CI=0;j_CI<(*I).Nb;j_CI++){
        tmp_key.reset();
        for (int k=1; k<(*I).na+1; k++) tmp_key.set((*I).vec_a[i_CI*((*I).na+1)+k]-1             );
        for (int k=1; k<(*I).nb+1; k++) tmp_key.set((*I).vec_b[j_CI*((*I).nb+1)+k]-1+CI_MAX_SPACE);
        
        if(no_eq_occ(tmp_key,key,CI_MAX_SPACE,CI_MAX_SPACE)){
            memset(A.bit_a,0,A.n_act*sizeof(int));
            memset(A.bit_b,0,A.n_act*sizeof(int));
            for(int i=0; i<A.n_act; i++)if(tmp_key[i             ]|key[i             ])A.bit_a[i]=1;
            for(int i=0; i<A.n_act; i++)if(tmp_key[i+CI_MAX_SPACE]|key[i+CI_MAX_SPACE])A.bit_b[i]=1;
            auto i_CIext = get_ind_from_ON(A.bit_a, A.n_act, A.na, A.fa, A.buf);
            auto j_CIext = get_ind_from_ON(A.bit_b, A.n_act, A.nb, A.fb, A.buf);
            
            for(int i_s=0; i_s<A.n_states[n_so];i_s++)
                A   .coef[n_so][(i_CIext*   A.Nb+j_CIext)*A.n_states[n_so]+i_s]=
                (*I).coef[n_si][(i_CI   *(*I).Nb+j_CI   )*A.n_states[n_so]+i_s]*ci_link_sign(tmp_key,key,CI_MAX_SPACE,CI_MAX_SPACE);
        }
    }
    transpose_3d_abc_to_bac(A.coef_bas[n_so], A.coef[n_so], A.Na, A.Nb, A.n_states[n_so]);
    
//     fprintf(out_stream,"s = %ld\n",A.size());
//     getchar();
    return 0;
}

int ci_pr_aldet::clear(){
    
//     delete A;
//     delete B;
    
    return 1;
}

int aldet_S_calc(double *O, aldet_data * A, int a, aldet_data * B, int b){
    
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
             A->n_states[a],B->n_states[b]/*check here*/,A->Nd,1.0,
             A->coef[a],A->n_states[a],
             B->coef[b],B->n_states[b],0.0,
             O,B->n_states[b]/*check here*/);

    return 0;
}

int aldet_DMA_calc(double *O, aldet_data * A, int a, aldet_data * B, int b, int ld){
    
    double * T =new double[A->n_act*A->n_act*A->n_states[a]*B->n_states[b]];
    set_zero_matr(T,A->n_act*A->n_act*A->n_states[a]*B->n_states[b]);
    
    A->calc_DM_part(T, A->coef[a], B->coef[b], A->n_states[a], B->n_states[b], A->na, A->Na, A->Nb, A->fa, A->fb, A->vec_a, A->bit_a);
    
    set_zero_matr(O,ld*ld*A->n_states[a]*B->n_states[b]);
    for(int s=0; s<A->n_states[a]*B->n_states[b];s++)
    for(int i=0; i<A->n_act;i++)
    for(int j=0; j<A->n_act;j++)
        O[(s*ld      +i)*ld      +j]=
        T[(s*A->n_act+i)*A->n_act+j];
    
    delete[] T;
    
    return 1;
}

int aldet_DMB_calc(double *O, aldet_data * A, int a, aldet_data * B, int b, int ld){
    
    double * T =new double[A->n_act*A->n_act*A->n_states[a]*B->n_states[b]];
    set_zero_matr(T,A->n_act*A->n_act*A->n_states[a]*B->n_states[b]);
    
    A->calc_DM_part(T, A->coef_bas[a], B->coef_bas[b], A->n_states[a], B->n_states[b], A->nb, A->Nb, A->Na, A->fb, A->fa, A->vec_b, A->bit_b);
    
    set_zero_matr(O,ld*ld*A->n_states[a]*B->n_states[b]);
    for(int s=0; s<A->n_states[a]*B->n_states[b];s++)
    for(int i=0; i<A->n_act;i++)
    for(int j=0; j<A->n_act;j++)
        O[(s*ld      +i)*ld      +j]=
        T[(s*A->n_act+i)*A->n_act+j];
    
    delete[] T;
    
    return 1;
}

int aldet_calc_CI_2el_A0AA(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f){
    
    omp_set_num_threads(num_threads);
    
    double **O_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)O_th[i]=new double[s1*s2*ld];
    for(int i=0;i<num_threads;i++)set_zero_matr(O_th[i],s1*s2*ld);

    #pragma omp parallel
    {
        double * c;
        int count=0;
        double sign_A1;
        double sign_B1;
        double sign_B2;
        double sign;
        double  * cV;
        double * c2;
        int i_s;
        double c_max;
        cV = new double[ld];
        
        int Na1,Nb1,Na2,Nb2;
        double * C1, * C2;
        int * fa, * vec;
        int na, na2;
        if(f==0){
            Na1=I1->Na;
            Nb1=I1->Nb;
            Na2=I2->Na;
            Nb2=I2->Nb;
            C1=I1->coef[a];
            C2=I2->coef[b];
            na=I1->na;
            na2=I2->na;
            vec=I1->vec_a;
            fa=I2->fa;
        }
        else{
            Na1=I1->Nb;
            Nb1=I1->Na;
            Na2=I2->Nb;
            Nb2=I2->Na;
            C1=I1->coef_bas[a];
            C2=I2->coef_bas[b];
            na=I1->nb;
            na2=I2->nb;
            vec=I1->vec_b;
            fa=I2->fb;
        }
        
        
        
        int * bit = new int [n_f-n_i];
        int * buf = new int [na2 + 1];
        buf[0]=0;
    
        int nt = omp_get_thread_num();
//         fprintf(out_stream,"nt = %d\n",nt);
        
        double * O_loc=new double[s1*s2*ld];
        set_zero_matr(O_loc,s1*s2*ld);
    
        for(int i_CI=nt;i_CI<Na1;i_CI+=num_threads){
            
            memset(bit,0,(n_f-n_i)*sizeof(int));
            for (int k=1; k<na+1; k++) bit[vec[i_CI*(na+1)+k]-1] = 1;
            
        
            sign_A1=1;
            for(int i=n_i;i<n_f;i++)if(bit[i-n_i]==1){
//                 id_T=id_I;
                bit[i-n_i]=0;
                    sign_B1=1;
                    for(int k=n_i;k<n_f-1;k++)if(bit[k-n_i]==0){
                        
                        bit[k-n_i]=1;
                        sign_B2=sign_B1;
                        for(int l=k+1;l<n_f;l++)if(bit[l-n_i]==0){
                            bit[l-n_i]=1;
                            sign = sign_A1*/*sign_A2**/sign_B1*sign_B2;
                            c_max=0;
//                             fprintf(out_stream,"c_max = %e\n",c_max);
                            for(int j=0;j<ld;j++)//j=0 *0,5?????
//                             for(i_s=0;i_s<s1;i_s++){
                                cV[j]=/*c[i_s]**/(V[((i*ld+k)*ld+j)*ld+l]-V[((i*ld+l)*ld+j)*ld+k])*sign;
//                                 if(c_max<fabs(cV[i_s*ld+j]))c_max=fabs(cV[i_s*ld+j]);
//                             }
                            
                            auto i_CIext = get_ind_from_ON(bit, n_f-n_i, na2, fa, buf);
                            
                            for(int j_CI=0;j_CI<Nb1;j_CI++){
                                c = C1+(i_CI*Nb1+j_CI)*s1;
                                c2 = C2+(i_CIext*Nb2+j_CI)*s2;
                                for(int i_s=0;i_s<s1;i_s++)
                                for(int j_s=0;j_s<s2;j_s++)
                                for(int j  =0;j  <ld;  j++)
                                    O_loc[(i_s*s2+j_s)*ld+j]+=cV[j]*c[i_s]*c2[j_s];
                            }
                            
                            bit[l-n_i]=0;
                        }
                        else
                            sign_B2=-sign_B2;
                        bit[k-n_i]=0;
                    }
                    else
                         sign_B1=-sign_B1;
                    
//                     id_I.set(j+f);
//                     sign_A2=-sign_A2;
                
//                 }
                bit[i-n_i]=1;
                sign_A1=-sign_A1;
            }
        }
        for(int i_s=0;i_s<s1*s2*ld;i_s++)
            O_th[nt][i_s]=O_loc[i_s];
        
        delete[] O_loc;
        delete[] bit;
        delete[] buf;
        delete[] cV;
        
    }
    
    for(int i_s=0;i_s<s1*s2*ld;i_s++)
    for(int i=0;i<num_threads;i++)
        O[i_s]+=O_th[i][i_s];
    for(int i=0;i<num_threads;i++)delete[] O_th[i];
    delete[] O_th;


        

    
//     fprintf(out_stream,"v1:\n");
//     PrintMatr(O,n,1,1);

    return 1;
}


int aldet_calc_CI_2el_AAA0(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f){
    
    omp_set_num_threads(num_threads);
    
    double **O_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)O_th[i]=new double[s1*s2*ld];
    for(int i=0;i<num_threads;i++)set_zero_matr(O_th[i],s1*s2*ld);

    #pragma omp parallel
    {
        double * c;
        int count=0;
        double sign_A1;
        double sign_A2;
        double sign_B1;
        double sign;
        double  * cV;
        double * c2;
        int i_s;
        double c_max;
        cV = new double[ld];
        
        int Na1,Nb1,Na2,Nb2;
        double * C1, * C2;
        int * fa, * vec;
        int na, na2;
        if(f==0){
            Na1=I1->Na;
            Nb1=I1->Nb;
            Na2=I2->Na;
            Nb2=I2->Nb;
            C1=I1->coef[a];
            C2=I2->coef[b];
            na=I1->na;
            na2=I2->na;
            vec=I1->vec_a;
            fa=I2->fa;
        }
        else{
            Na1=I1->Nb;
            Nb1=I1->Na;
            Na2=I2->Nb;
            Nb2=I2->Na;
            C1=I1->coef_bas[a];
            C2=I2->coef_bas[b];
            na=I1->nb;
            na2=I2->nb;
            vec=I1->vec_b;
            fa=I2->fb;
        }
        
        int * bit = new int [n_f-n_i];
        int * buf = new int [na2 + 1];
        buf[0]=0;
        
        int nt = omp_get_thread_num();
        
        double * O_loc=new double[s1*s2*ld];
        set_zero_matr(O_loc,s1*s2*ld);
    
        for(int i_CI=nt;i_CI<Na1;i_CI+=num_threads){
            
            memset(bit,0,(n_f-n_i)*sizeof(int));
            for (int k=1; k<na+1; k++) bit[vec[i_CI*(na+1)+k]-1] = 1;
            
            sign_A1=1;
            for(int i=n_i;i<n_f-1;i++)if(bit[i-n_i]==1){
//                 id_T=bit;
                bit[i-n_i]=0;
                sign_A2=sign_A1;
                for(int j=i+1;j<n_f;j++)if(bit[j-n_i]==1){
                    bit[j-n_i]=0;
                    sign_B1=1;
                    for(int k=n_i;k<n_f;k++)if(bit[k-n_i]==0){
//                         id_O=id_T;
                        bit[k-n_i]=1;
//                         sign_B2=sign_B1;
//                         for(int l=k+1;l<n;l++)if(!bit[l]){
//                             bit.set(l);
                            sign = sign_A1*sign_A2*sign_B1/**sign_B2*/;
                            c_max=0;
//                             fprintf(out_stream,"c_max = %e\n",c_max);
                            for(int l=0;l<ld;l++)//l=0 *0,5?????
//                             for(i_s=0;i_s<s1;i_s++){
        
                                cV[l]=(V[((i*ld+k)*ld+j)*ld+l]-V[((i*ld+l)*ld+j)*ld+k])*sign;
//                                 if(c_max<fabs(cV[i_s*ld+l]))c_max=fabs(cV[i_s*ld+l]);
//                             }
//                             fprintf(out_stream,"c_max = %e\n",c_max);
//                             getchar();
//                             if(c_max>ci_v_cutoff){
                            auto i_CIext = get_ind_from_ON(bit, n_f-n_i, na2, fa, buf);
                            
                            for(int j_CI=0;j_CI<Nb1;j_CI++){
                                c = C1+(i_CI*Nb1+j_CI)*s1;
                                c2 = C2+(i_CIext*Nb2+j_CI)*s2;
                                
                                for(int i_s=0;i_s<s1;i_s++)
                                for(int j_s=0;j_s<s2;j_s++)
                                for(int l  =0;l  <ld;  l++)
                                    O_loc[(i_s*s2+j_s)*ld+l]+=cV[l]*c[i_s]*c2[j_s];
                            }    
                                                        
//                             }
//                             bit.reset(l);
//                         }
//                         else
//                             sign_B2=-sign_B2;
                        bit[k-n_i]=0;
                    }
                    else
                         sign_B1=-sign_B1;
                    
                    bit[j-n_i]=1;
                    sign_A2=-sign_A2;
                
                }
                bit[i-n_i]=1;
                sign_A1=-sign_A1;
            }
        }
        
        for(int i_s=0;i_s<s1*s2*ld;i_s++)
            O_th[nt][i_s]=O_loc[i_s];
        
        delete[] O_loc;
        delete[] cV;
        delete[] bit;
        delete[] buf;
    }
    
    for(int i_s=0;i_s<s1*s2*ld;i_s++)
    for(int i=0;i<num_threads;i++)
        O[i_s]+=O_th[i][i_s];
    for(int i=0;i<num_threads;i++)delete[] O_th[i];
    delete[] O_th;

//     fprintf(out_stream,"v1:\n");
//     PrintMatr(O,n,1,1);
    
    return 1;
}

int aldet_calc_CI_2el_AB0B(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2,int f){
    
    omp_set_num_threads(num_threads);
    
    double **O_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)O_th[i]=new double[s1*s2*ld];
    for(int i=0;i<num_threads;i++)set_zero_matr(O_th[i],s1*s2*ld);

    #pragma omp parallel
    {
        double * c;
        int count=0;
        double sign_A1;
        double sign_A2;
        double sign_B2;
        double sign;
        double  * cV;
        double * c2;
        int i_s;
        double c_max;
        cV = new double[s1*ld];
        
        int Na1,Nb1,Na2,Nb2;
        double * C1, * C2;
        int * fa, *fb, * vec_a, * vec_b;
        int na, nb, na2, nb2;
        if(f==0){
            Na1=I1->Na;
            Nb1=I1->Nb;
            Na2=I2->Na;
            Nb2=I2->Nb;
            C1=I1->coef[a];
            C2=I2->coef[b];
            na=I1->na;
            nb=I1->nb;
            na2=I2->na;
            nb2=I2->nb;
            vec_a=I1->vec_a;
            vec_b=I1->vec_b;
            fa=I2->fa;
            fb=I2->fb;
        }
        else{
            Na1=I1->Nb;
            Nb1=I1->Na;
            Na2=I2->Nb;
            Nb2=I2->Na;
            C1=I1->coef_bas[a];
            C2=I2->coef_bas[b];
            na=I1->nb;
            nb=I1->na;
            na2=I2->nb;
            nb2=I2->na;
            vec_a=I1->vec_b;
            vec_b=I1->vec_a;
            fa=I2->fb;
            fb=I2->fa;
        }
        
        int * bit_a = new int [n_f-n_i];
        int * bit_b = new int [n_f-n_i];
        int * buf = new int [max(na2,nb2) + 1];
        buf[0]=0;
        
        int nt = omp_get_thread_num();
        
        double * O_loc=new double[s1*s2*ld];
        set_zero_matr(O_loc,s1*s2*ld);
    
        for(int i_CI=nt;i_CI<Na1;i_CI+=num_threads)
        for(int j_CI=0;j_CI<Nb1;j_CI++){
            
            c = C1+(i_CI*Nb1+j_CI)*s1;
            memset(bit_a,0,(n_f-n_i)*sizeof(int));
            memset(bit_b,0,(n_f-n_i)*sizeof(int));
            for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
            for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
            
            sign_A1=1;
            for(int i=n_i;i<n_f;i++)if(bit_a[i-n_i]==1){
//                 id_T=id_I;
                bit_a[i-n_i]=0;
                sign_A2=1;
                for(int j=n_i;j<n_f;j++)if(bit_b[j-n_i]==1){
                    bit_b[j-n_i]=0;
//                     sign_B1=1;
//                     for(int k=0;k<n;k++)if(!id_I[k+fa]){
//                         id_O=id_T;
//                         id_I.set(k+fa);
                        sign_B2=1;
                        for(int l=n_i;l<n_f;l++)if(bit_b[l-n_i]==0){
                            bit_b[l-n_i]=1;
                            sign =sign_A1*sign_A2*/*sign_B1**/sign_B2;
                            c_max=0;
                            for(int k=0;k<ld;k++)
                            for(i_s=0;i_s<s1;i_s++){
                                cV[i_s*ld+k]=c[i_s]*V[((i*ld+k)*ld+j)*ld+l]*sign;
                                if(c_max<fabs(cV[i_s*ld+k]))c_max=fabs(cV[i_s*ld+k]);
                            }
                            
//                             if(c_max>ci_v_cutoff){
                            auto i_CIext = get_ind_from_ON(bit_a, n_f-n_i, na2, fa, buf);
                            auto j_CIext = get_ind_from_ON(bit_b, n_f-n_i, nb2, fb, buf);
                            
                            c2 = C2+(i_CIext*Nb2+j_CIext)*s2;
                            for(int k=0;k<ld;k++)
                             for(int i_s=0;i_s<s1;i_s++)
                             for(int j_s=0;j_s<s2;j_s++)
                                 O_loc[(i_s*s2+j_s)*ld+k]+=cV[i_s*ld+k]*c2[j_s];
                                     
                             
//                             }
                            
                            bit_b[l-n_i]=0;
                        }
                        else
                            sign_B2=-sign_B2;
//                         id_I.reset(k+fa);
//                     }
//                     else
//                         sign_B1=-sign_B1;
                    bit_b[j-n_i]=1;
                    sign_A2=-sign_A2;
                }
                bit_a[i-n_i]=1;
                sign_A1=-sign_A1;
            }
        }
        for(int i_s=0;i_s<s1*s2*ld;i_s++)
            O_th[nt][i_s]=O_loc[i_s];
        
        delete[] O_loc;
        delete[] cV;
        delete[] bit_a;
        delete[] bit_b;
        delete[] buf;
    }
    
    for(int i_s=0;i_s<s1*s2*ld;i_s++)
    for(int i=0;i<num_threads;i++)
        O[i_s]+=O_th[i][i_s];
    for(int i=0;i<num_threads;i++)delete[] O_th[i];
    delete[] O_th;

//         fprintf(out_stream,"v1_AB0B:\n");
//         PrintMatr(O,s1*s2*ld,ld,1);
        
    return 1;
}

int aldet_calc_CI_2el_0BAB(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2,int f){
    
    omp_set_num_threads(num_threads);
    
    double **O_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)O_th[i]=new double[s1*s2*ld];
    for(int i=0;i<num_threads;i++)set_zero_matr(O_th[i],s1*s2*ld);

    #pragma omp parallel
    {
        double * c;
        int count=0;
        double sign_A1;
        double sign_A2;
        double sign_B1;
        double sign_B2;
        double sign;
        double  * cV;
        double * c2;
        int i_s;
        double c_max;
        cV = new double[s1*ld];
        
        int Na1,Nb1,Na2,Nb2;
        double * C1, * C2;
        int * fa, *fb, * vec_a, * vec_b;
        int na, nb, na2, nb2;
        if(f==0){
            Na1=I1->Na;
            Nb1=I1->Nb;
            Na2=I2->Na;
            Nb2=I2->Nb;
            C1=I1->coef[a];
            C2=I2->coef[b];
            na=I1->na;
            nb=I1->nb;
            na2=I2->na;
            nb2=I2->nb;
            vec_a=I1->vec_a;
            vec_b=I1->vec_b;
            fa=I2->fa;
            fb=I2->fb;
        }
        else{
            Na1=I1->Nb;
            Nb1=I1->Na;
            Na2=I2->Nb;
            Nb2=I2->Na;
            C1=I1->coef_bas[a];
            C2=I2->coef_bas[b];
            na=I1->nb;
            nb=I1->na;
            na2=I2->nb;
            nb2=I2->na;
            vec_a=I1->vec_b;
            vec_b=I1->vec_a;
            fa=I2->fb;
            fb=I2->fa;
        }
        
        int * bit_a = new int [n_f-n_i];
        int * bit_b = new int [n_f-n_i];
        int * buf = new int [max(na2,nb2) + 1];
        buf[0]=0;
        
        int nt = omp_get_thread_num();
        
        double * O_loc=new double[s1*s2*ld];
        set_zero_matr(O_loc,s1*s2*ld);
    
        for(int i_CI=nt;i_CI<Na1;i_CI+=num_threads)
        for(int j_CI=0;j_CI<Nb1;j_CI++){
            c = C1+(i_CI*Nb1+j_CI)*s1;
            memset(bit_a,0,(n_f-n_i)*sizeof(int));
            memset(bit_b,0,(n_f-n_i)*sizeof(int));
            for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
            for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
            
//             sign_A1=1;
//             for(int i=0;i<n;i++)if(id_I[i+fa]){
//                 id_T=id_I;
//                 id_I.reset(i+fa);
                sign_A2=1;
                for(int j=n_i;j<n_f;j++)if(bit_b[j-n_i]==1){
                    bit_b[j-n_i]=0;
                    sign_B1=1;
                    for(int k=n_i;k<n_f;k++)if(bit_a[k-n_i]==0){
                        bit_a[k-n_i]=1;
                        sign_B2=1;
                        for(int l=n_i;l<n_f;l++)if(bit_b[l-n_i]==0){
                            bit_b[l-n_i]=1;
                            sign =/*sign_A1**/sign_A2*sign_B1*sign_B2;
                            c_max=0;
                            for(int i=0;i<ld;i++)
                            for(i_s=0;i_s<s1;i_s++){
                                cV[i_s*ld+i]=c[i_s]*V[((i*ld+k)*ld+j)*ld+l]*sign;
                                if(c_max<fabs(cV[i_s*ld+i]))c_max=fabs(cV[i_s*ld+i]);
                            }
                            
//                             if(c_max>ci_v_cutoff){
                            auto i_CIext = get_ind_from_ON(bit_a, n_f-n_i, na2, fa, buf);
                            auto j_CIext = get_ind_from_ON(bit_b, n_f-n_i, nb2, fb, buf);
                            
                            c2 = C2+(i_CIext*Nb2+j_CIext)*s2;
                            
                            for(int i=0;i<ld;i++)
                            for(int i_s=0;i_s<s1;i_s++)
                            for(int j_s=0;j_s<s2;j_s++)
                                O_loc[(i_s*s2+j_s)*ld+i]+=cV[i_s*ld+i]*c2[j_s];
                                    
                            
//                             }
                            
                            bit_b[l-n_i]=0;
                        }
                        else
                            sign_B2=-sign_B2;
                        bit_a[k-n_i]=0;
                    }
                    else
                        sign_B1=-sign_B1;
                    bit_b[j-n_i]=1;
                    sign_A2=-sign_A2;
                }
//                 id_I.set(i+fa);
//                 sign_A1=-sign_A1;
//             }
        }
        
        for(int i_s=0;i_s<s1*s2*ld;i_s++)
            O_th[nt][i_s]=O_loc[i_s];
        
        delete[] O_loc;
        delete[] cV;
        delete[] bit_a;
        delete[] bit_b;
        delete[] buf;
    }
    
    for(int i_s=0;i_s<s1*s2*ld;i_s++)
    for(int i=0;i<num_threads;i++)
        O[i_s]+=O_th[i][i_s];
    for(int i=0;i<num_threads;i++)delete[] O_th[i];
    delete[] O_th;

        
//     fprintf(out_stream,"v1 B:\n");
//     PrintMatr(O,n,1,1);
    
    return 1;
}

int aldet_calc_CI_2el_0A0B(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f){
    
    omp_set_num_threads(num_threads);
    
    double **O_th= new double * [num_threads];
    for(int i=0;i<num_threads;i++)O_th[i]=new double[s1*s2*ld*ld];
    for(int i=0;i<num_threads;i++)set_zero_matr(O_th[i],s1*s2*ld*ld);

    #pragma omp parallel
    {
        double * c;
        int count=0;
        double sign_A2;
        double sign_B2;
        double sign;
        double  * c1;
        double * c2;
        int i_s;
        double c_max;
        
        int Na1,Nb1,Na2,Nb2;
        double * C1, * C2;
        int * fa, *fb, * vec_a, * vec_b;
        int na, nb, na2, nb2;
        if(f==0){
            Na1=I1->Na;
            Nb1=I1->Nb;
            Na2=I2->Na;
            Nb2=I2->Nb;
            C1=I1->coef[a];
            C2=I2->coef[b];
            na=I1->na;
            nb=I1->nb;
            na2=I2->na;
            nb2=I2->nb;
            vec_a=I1->vec_a;
            vec_b=I1->vec_b;
            fa=I2->fa;
            fb=I2->fb;
        }
        else{
            Na1=I1->Nb;
            Nb1=I1->Na;
            Na2=I2->Nb;
            Nb2=I2->Na;
            C1=I1->coef_bas[a];
            C2=I2->coef_bas[b];
            na=I1->nb;
            nb=I1->na;
            na2=I2->nb;
            nb2=I2->na;
            vec_a=I1->vec_b;
            vec_b=I1->vec_a;
            fa=I2->fb;
            fb=I2->fa;
        }
        
        int * bit_a = new int [n_f-n_i];
        int * bit_b = new int [n_f-n_i];
        int * buf = new int [max(na2,nb2) + 1];
        buf[0]=0;
        
        int nt = omp_get_thread_num();
        
        double * O_loc=new double[s1*s2*ld*ld];
        set_zero_matr(O_loc,s1*s2*ld*ld);
    
        for(int i_CI=nt;i_CI<Na1;i_CI+=num_threads)
        for(int j_CI=0;j_CI<Nb1;j_CI++){
            c = C1+(i_CI*Nb1+j_CI)*s1;
            memset(bit_a,0,(n_f-n_i)*sizeof(int));
            memset(bit_b,0,(n_f-n_i)*sizeof(int));
            for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
            for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
//             sign_A1=1;
//             for(int i=0;i<n-1;i++)if(id_I[i+f]){
//                 id_T=id_I;
//                 id_I.reset(i+f);
                sign_A2=1;
                for(int j=n_i;j<n_f;j++)if(bit_a[j-n_i]==1){
                    bit_a[j-n_i]=0;
//                     sign_B1=1;
//                     for(int k=0;k<k_end;k++)if(!id_I[k+f1]){
//                         id_O=id_T;
//                         id_I.set(k+f1);
                        sign_B2=1;
                        for(int l=n_i;l<n_f;l++)if(bit_b[l-n_i]==0){
                            bit_b[l-n_i]=1;
                            sign = /*sign_A1**/sign_A2*/*sign_B1**/sign_B2;
                            
                            auto i_CIext = get_ind_from_ON(bit_a, n_f-n_i, na2, fa, buf);
                            auto j_CIext = get_ind_from_ON(bit_b, n_f-n_i, nb2, fb, buf);
                            
                            c2 = C2+(i_CIext*Nb2+j_CIext)*s2;
                            
                            for(int i_s=0;i_s<s1;i_s++)
                            for(int j_s=0;j_s<s2;j_s++)
                                O_loc[((i_s*s2+j_s)*ld+j)*ld+l]+=c[i_s]*c2[j_s]*sign;
                                    
//                             }
                            
                            
                            bit_b[l-n_i]=0;
                        }
                        else
                            sign_B2=-sign_B2;
//                         id_I.reset(k+f1);
//                     }
//                     else
//                          sign_B1=-sign_B1;
                    
                    bit_a[j-n_i]=1;
                    sign_A2=-sign_A2;
                
                }
//                 id_I.set(i+f);
//                 sign_A1=-sign_A1;
//             }
        }
        
        for(int i_s=0;i_s<s1*s2*ld*ld;i_s++)
            O_th[nt][i_s]=O_loc[i_s];
        
        delete[] O_loc;
        
        delete[] bit_a;
        delete[] bit_b;
        delete[] buf;
    }
    
    for(int i_s=0;i_s<s1*s2*ld*ld;i_s++)
    for(int i=0;i<num_threads;i++)
        O[i_s]+=O_th[i][i_s];
    for(int i=0;i<num_threads;i++)delete[] O_th[i];
    delete[] O_th;

        
    
    return 1;
}

int aldet_calc_CI_2el_00AA(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f1, int f2){
    
    ci_key id_I;
    double * c;
    int count=0;
    double sign_B1;
    double sign_B2;
    double sign;
    double  * c1;
    double * c2;
    int i_s;
    double c_max;
    int l0;
    int k_end=n_f;
    if(f1==f2)k_end=n_f-1;
    
    int Na1,Nb1,Na2,Nb2;
    double * C1, * C2;
    int * fa, *fb, * vec_a, * vec_b;
    int na, nb, na2, nb2;
    Na1=I1->Na;
    Nb1=I1->Nb;
    Na2=I2->Na;
    Nb2=I2->Nb;
    C1=I1->coef[a];
    C2=I2->coef[b];
    na=I1->na;
    nb=I1->nb;
    na2=I2->na;
    nb2=I2->nb;
    vec_a=I1->vec_a;
    vec_b=I1->vec_b;
    
    fa=I2->fa;
    fb=I2->fb;
    
    int * bit_a = new int [n_f-n_i];
    int * bit_b = new int [n_f-n_i];
    int * buf = new int [max(na2,nb2) + 1];
    buf[0]=0;
    int * bit1=bit_a;
    int * bit2=bit_a;
    
    if(f1)bit1=bit_b;
    if(f2)bit2=bit_b;
    
    

    for(int i_CI=0;i_CI<Na1;i_CI++)
    for(int j_CI=0;j_CI<Nb1;j_CI++){
        c = C1+(i_CI*Nb1+j_CI)*s1;
        memset(bit_a,0,(n_f-n_i)*sizeof(int));
        memset(bit_b,0,(n_f-n_i)*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;

//         sign_A1=1;
//         for(int i=0;i<n-1;i++)if(id_I[i+f]){
//             id_T=id_I;
//             id_I.reset(i+f);
//             sign_A2=sign_A1;
//             for(int j=i+1;j<n;j++)if(id_I[j+f]){
//                 id_I.reset(j+f);
                sign_B1=1;
                for(int k=n_i;k<k_end;k++)if(bit1[k]==0){
//                     id_O=id_T;
                    bit1[k]=1;
                    if(f1==f2){
                        sign_B2=sign_B1;
                        l0=k+1;
                    }
                    else{
                        sign_B2=1;
                        l0=n_i;
                    }
                    for(int l=l0;l<n_f;l++)if(bit2[l]==0){
                        bit2[l]=1;
                        sign = /*sign_A1*sign_A2**/sign_B1*sign_B2;
                        
                        auto i_CIext = get_ind_from_ON(bit_a, n_f-n_i, na2, fa, buf);
                        auto j_CIext = get_ind_from_ON(bit_b, n_f-n_i, nb2, fb, buf);
                        
                        c2 = C2+(i_CIext*Nb2+j_CIext)*s2;
                        
                        for(int i_s=0;i_s<s1;i_s++)
                        for(int j_s=0;j_s<s2;j_s++)
                            O[((i_s*s2+j_s)*ld+k)*ld+l]+=c[i_s]*c2[j_s]*sign;
                            
                        bit2[l]=0;
                    }
                    else
                        sign_B2=-sign_B2;
                    bit1[k]=0;
                }
                else
                     sign_B1=-sign_B1;
                
//                 id_I.set(j+f);
//                 sign_A2=-sign_A2;
            
//             }
//             id_I.set(i+f);
//             sign_A1=-sign_A1;
//         }
    }
    
    
    return 1;
}

int aldet_calc_CI_2el_AA00(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f1, int f2){
    
    ci_key id_I;
    double * c;
    int count=0;
    double sign_A1;
    double sign_A2;
    double sign;
    double  * c1;
    double * c2;
    int i_s;
    double c_max;
    int j0;
    int i_end=n_f;
    if(f1==f2)i_end=n_f-1;
    
    int Na1,Nb1,Na2,Nb2;
    double * C1, * C2;
    int * fa, *fb, * vec_a, * vec_b;
    int na, nb, na2, nb2;
    Na1=I1->Na;
    Nb1=I1->Nb;
    Na2=I2->Na;
    Nb2=I2->Nb;
    C1=I1->coef[a];
    C2=I2->coef[b];
    na=I1->na;
    nb=I1->nb;
    na2=I2->na;
    nb2=I2->nb;
    vec_a=I1->vec_a;
    vec_b=I1->vec_b;
    
    fa=I2->fa;
    fb=I2->fb;
    
    int * bit_a = new int [n_f-n_i];
    int * bit_b = new int [n_f-n_i];
    int * buf = new int [max(na2,nb2) + 1];
    buf[0]=0;
    int * bit1=bit_a;
    int * bit2=bit_a;
    
    if(f1)bit1=bit_b;
    if(f2)bit2=bit_b;
    
    

    for(int i_CI=0;i_CI<Na1;i_CI++)
    for(int j_CI=0;j_CI<Nb1;j_CI++){
        c = C1+(i_CI*Nb1+j_CI)*s1;
        memset(bit_a,0,(n_f-n_i)*sizeof(int));
        memset(bit_b,0,(n_f-n_i)*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        sign_A1=1;
        for(int i=n_i;i<i_end;i++)if(bit1[i]==1){
//             id_T=id_I;
            bit1[i]=0;
            if(f1==f2){
                sign_A2=sign_A1;
                j0=i+1;
            }
            else{
                sign_A2=1;
                j0=n_i;
            }
            for(int j=j0;j<n_f;j++)if(bit2[j]==1){
                bit2[j]=0;
//                 sign_B1=1;
//                 for(int k=0;k<k_end;k++)if(!id_I[k+f1]){
//                     id_O=id_T;
//                     id_I.set(k+f1);
//                     if(f1==f2){
//                         j0=k+1;
//                         sign_B2=sign_B1;
//                     }
//                     else{
//                         l0=0;
//                         sign_B2=1;
//                     }
//                     for(int l=l0;l<n;l++)if(!id_I[l+f2]){
//                         id_I.set(l+f2);
                        sign = sign_A1*sign_A2/**sign_B1*sign_B2*/;
                        
                        auto i_CIext = get_ind_from_ON(bit_a, n_f-n_i, na2, fa, buf);
                        auto j_CIext = get_ind_from_ON(bit_b, n_f-n_i, nb2, fb, buf);
                        
                        c2 = C2+(i_CIext*Nb2+j_CIext)*s2;
                        for(int i_s=0;i_s<s1;i_s++)
                        for(int j_s=0;j_s<s2;j_s++)
                            O[((i_s*s2+j_s)*ld+i)*ld+j]+=c[i_s]*c2[j_s]*sign;
                                
                        
                            
//                         }
//                         id_I.reset(l+f2);
//                     }
//                     else
//                         sign_B2=-sign_B2;
//                     id_I.reset(k+f1);
//                 }
//                 else
//                      sign_B1=-sign_B1;
                
                bit2[j]=1;
                sign_A2=-sign_A2;
            
            }
            bit1[i]=1;
            sign_A1=-sign_A1;
        }
    }
    
    return 1;
}


int aldet_gen_1_el_vec_arr(double * O,aldet_data * I1, int a, int nA, aldet_data * I2, int b, int nB, int n_i, int n_f, int ld, int f){
    
    
    double * coef_A;
    double * coef_B;
    double sign_A;
    double sign_B;
    
    set_zero_matr(O,nA*nB*ld);
    
    int Na1,Nb1,Na2,Nb2;
    double * C1, * C2;
    int * fa, * vec;
    int na, na2;
    if(f==0){
        Na1=I1->Na;
        Nb1=I1->Nb;
        Na2=I2->Na;
        Nb2=I2->Nb;
        C1=I1->coef[a];
        C2=I2->coef[b];
        na=I1->na;
        na2=I2->na;
        vec=I1->vec_a;
        fa=I2->fa;
    }
    else{
        Na1=I1->Nb;
        Nb1=I1->Na;
        Na2=I2->Nb;
        Nb2=I2->Na;
        C1=I1->coef_bas[a];
        C2=I2->coef_bas[b];
        na=I1->nb;
        na2=I2->nb;
        vec=I1->vec_b;
        fa=I2->fb;
    }
    
    
    
    int * bit = new int [n_f-n_i];
    int * buf = new int [na2 + 1];
    buf[0]=0;
    
    for(int i_CI=0;i_CI<Na1;i_CI++){
        
        memset(bit,0,(n_f-n_i)*sizeof(int));
        for (int k=1; k<na+1; k++) bit[vec[i_CI*(na+1)+k]-1] = 1;
        
        sign_A=1;
        
        for(int i=n_i;i<n_f;i++)
        if(bit[i-n_i]==1){
            bit[i-n_i]=0;
            auto i_CIext = get_ind_from_ON(bit, n_f-n_i, na2, fa, buf);
            
            for(int j_CI=0;j_CI<Nb1;j_CI++){
                coef_A = C1+(i_CI   *Nb1+j_CI)*nA;
                coef_B = C2+(i_CIext*Nb2+j_CI)*nB;
        
                for(int i_s=0;i_s<nA;i_s++)
                for(int j_s=0;j_s<nB;j_s++)
                    O[(i_s*nB+j_s)*ld+i]+=coef_A[i_s]*coef_B[j_s]*sign_A;
            }
            bit[i-n_i]=1;
            sign_A=-sign_A;
        }
    }
    
//     fprintf(out_stream,"1_el_vec result:\n");
//     PrintMatr(O,ld,1,1);
    
    delete[] bit;
    delete[] buf;
   
    return 0;
}

int aldet_gen_1_el_vec_m_arr(double * O,aldet_data * I1, int a, int nA, aldet_data * I2, int b, int nB, int n_i, int n_f, int ld, int f){
    
    double * coef_A;
    double * coef_B;
    double sign_A;
    double sign_B;
    
    set_zero_matr(O,nA*nB*ld);
    
    int Na1,Nb1,Na2,Nb2;
    double * C1, * C2;
    int * fa, * vec;
    int na, na2;
    if(f==0){
        Na1=I1->Na;
        Nb1=I1->Nb;
        Na2=I2->Na;
        Nb2=I2->Nb;
        C1=I1->coef[a];
        C2=I2->coef[b];
        na=I1->na;
        na2=I2->na;
        vec=I1->vec_a;
        fa=I2->fa;
    }
    else{
        Na1=I1->Nb;
        Nb1=I1->Na;
        Na2=I2->Nb;
        Nb2=I2->Na;
        C1=I1->coef_bas[a];
        C2=I2->coef_bas[b];
        na=I1->nb;
        na2=I2->nb;
        vec=I1->vec_b;
        fa=I2->fb;
    }
    
    
    
    int * bit = new int [n_f-n_i];
    int * buf = new int [na2 + 1];
    buf[0]=0;
    
    for(int i_CI=0;i_CI<Na1;i_CI++){
        
        memset(bit,0,(n_f-n_i)*sizeof(int));
        for (int k=1; k<na+1; k++) bit[vec[i_CI*(na+1)+k]-1] = 1;

        sign_B=1;
        for(int j=n_i;j<n_f;j++)
        if(bit[j-n_i]==0){
            bit[j-n_i]=1;
            
            auto i_CIext = get_ind_from_ON(bit, n_f-n_i, na2, fa, buf);
            
            for(int j_CI=0;j_CI<Nb1;j_CI++){
    
                coef_A = C1+(i_CI*Nb1+j_CI)*nA;
                coef_B = C2+(i_CIext*Nb2+j_CI)*nB;
                        
                for(int i_s=0;i_s<nA;i_s++)
                for(int j_s=0;j_s<nB;j_s++)
                    O[(i_s*nB+j_s)*ld+j]+=coef_A[i_s]*coef_B[j_s]*sign_B;
            }
            bit[j-n_i]=0;
        }
        else
            sign_B=-sign_B;

    }
    
//     fprintf(out_stream,"1_el_vec result:\n");
//     PrintMatr(O,ld,1,1);

    delete[] bit;
    delete[] buf;
   
    return 0;
}

int aldet_calc_2body_AA(double * P, int n_s, int ld,
                        double * ci,
                        int N,int na, int Na, int Nb, int * fa, int * vec_a,
                        double * act_INTS,
                        int i_th, int n_th){
    
    int * bit_a = new int [N];
    int * buf = new int [na + 1];//more space for +a-b and -a+b
    buf[0]=0;
    
    double sign ;
    double sign1;
    double sign2;
    double sign3;
    
    double Rc;
    
    double R;
    double * CI_0;
    double * CI_e;
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //AAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign1=1;
            bit_a[t]=0;
            for(int u=t+1;u<N;u++){
                if(bit_a[u]!=0){
//                     sign2=1;
                    bit_a[u]=0;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]==0){
                            sign3=1;
                            bit_a[v]=1;
                            for(int w=v+1;w<N;w++){
                                if(bit_a[w]==0){
                                    bit_a[w]=1;
                                    sign=sign1*sign3;
                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                    R = act_INTS[((t*N+v)*N+u)*N+w]-act_INTS[((t*N+w)*N+u)*N+v];
                                    for(int j_CI = 0; j_CI<Nb; j_CI++)
                                    for(int i_s=0;i_s<n_s;i_s++){
//                                         Rc =0;
//                                         L = L_fit[(i_CI*Nb+j_CI)*n_s+i_s];
//                                         S = S_fit[(i_CI*Nb+j_CI)*n_s+i_s];
//                                         C = C_fit+(2*n_fit_pol+1)*((i_CI*Nb+j_CI)*n_s+i_s);
                                        //<tu|wa>*P[a,v]-<tu|va>*P[a,w] to be checked
//                                         Rc+=ddot(L,R+S, 1, C, 1);
                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                        Rc =R*CI_0[i_s]*sign;
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

int aldet_calc_2body_AB(double * P, int n_s, int ld,
                        double * ci,
                        int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                        int * Ia, int * Ib, int * Ia_friends, int * Ib_friends,
                        double * act_INTS,
                        int i_th, int n_th){
    
    
    const int Na_zv = (int) std::lround(tgammal(N) / tgammal(na) / tgammal(N-na+1));
    const int Nb_zv = (int) std::lround(tgammal(N) / tgammal(nb) / tgammal(N-nb+1));
    
//     
    double Rc;
    double sign;
    double R;
    double * CI_0;
    double * CI_e;

    
    
    for(int t=0  ;t<N;t++){
        get_I_I_friends(na, N, Na, fa, t, vec_a, Ia, Ia_friends);//if(nt){fprintf(out_stream,"nt>1\n");exit(0);}
        for(int u=0  ;u<N;u++){//fprintf(stderr,"t=%d u=%d  (thread %d)        \r",t,u, i_th);
            get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib, Ib_friends);
            for (int i_CI = i_th; i_CI<Na_zv; i_CI+=n_th){
                int k_CI = Ia[i_CI];
                for (int j_CI = 0; j_CI<Nb_zv; j_CI++){
                    int l_CI = Ib[j_CI];
                    CI_0=ci+(k_CI*Nb+l_CI  )*ld;
                    R=act_INTS[((t*N+t)*N+u)*N+u];
                    for(int i_s=0;i_s<n_s;i_s++){
                        Rc =0;
                        Rc =R*CI_0[i_s];
                        for(int j_s=0;j_s<n_s;j_s++)
                            P[i_s*n_s+j_s]+=Rc*CI_0[j_s];
                    }
                    for (int n = 0; n<(N-nb); n++){
                    int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                        CI_e = ci+(k_CI*Nb+ind_n[0])*ld;
                        R=act_INTS[((t*N+t)*N+u)*N+ind_n[1]];
                        for(int i_s=0;i_s<n_s;i_s++){
                            Rc =R*CI_0[i_s]*ind_n[2];
                            for(int j_s=0;j_s<n_s;j_s++)
                                P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                        }
                    }
                    for (int m = 0; m<(N-na); m++){
                        int * ind_m = Ia_friends + i_CI*(N-na)*3 + m*3;
                        CI_e = ci+(ind_m[0]*Nb+l_CI)*ld;
                        R=act_INTS[((t*N+ind_m[1])*N+u)*N+u];
                        for(int i_s=0;i_s<n_s;i_s++){
                            Rc =R*CI_0[i_s]*ind_m[2];
                            for(int j_s=0;j_s<n_s;j_s++)
                                P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                        }
                        for (int n = 0; n<(N-nb); n++){
                            int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                            CI_e = ci+(ind_m[0]*Nb+ind_n[0])*ld;
                            R=act_INTS[((t*N+ind_m[1])*N+u)*N+ind_n[1]];;
                            sign = ind_m[2]*ind_n[2];
                            for(int i_s=0;i_s<n_s;i_s++){
                                Rc =R*CI_0[i_s]*sign;
                                for(int j_s=0;j_s<n_s;j_s++)
                                    P[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                            }
                        }
                    }
                }
            }
        }
    }
    
//     delete[] Ia        ;
//     delete[] Ia_friends;
//     delete[] Ib        ;
//     delete[] Ib_friends;
    
    return 0;
       
}


int aldet_calc_1body(double * P, int n_s, int ld,
                     double * ci,
                     int N,int na, int Na, int Nb, int * fa, int * vec_a,
                     double * H,
                     int i_th, int n_th){
    int * bit_a = new int [N];
//     int * bit_b = new int [N];
    int * buf = new int [na + 1];//more space for +a-b and -a+b
    buf[0]=0;
    
    double sign ;
    double sign1;
    double sign2;
    double Rc,R;
        
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //A00A
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            bit_a[t]=0;
            sign2=1;
            for(int w=0;w<N;w++){
                if(bit_a[w]==0){
                    bit_a[w]=1;
                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                    sign=sign1*sign2;
                    R = H[t*N+w];
                    for(int j_CI = 0; j_CI<Nb; j_CI++)
                    for(int i_s=0;i_s<n_s;i_s++){
                        
                        Rc =R*ci[(i_CI*Nb+j_CI  )*ld+i_s]*sign;
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


int aldet_H_mult_2body_AB_3(double * ci_O, int n_s, int ld,
                          double * ci_I,
                          int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                          int * Ia, int * Ib, int * Ia_friends, int * Ib_friends,
                          double * act_INTS,
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
    
    double Rc;
    
    double R;
    double * CI_0;
    double * CI_e;
    
    int num_ext = na*nb*(N-na+1)*(N-nb+1);
    double * __restrict__ H_s       = new double[Na*Nb*num_ext];
    int    * __restrict__ col_idx   = new    int[Na*Nb*num_ext];
    int    * __restrict__ row_start = new    int[Na*Nb];
    for(int i=0; i<Na*Nb;i++)
        row_start[i]=i*num_ext;
    
    
    int i_ext=0;
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d                                    (thread %d)        \r",i_CI, i_th);
    for(int j_CI =0   ; j_CI<Nb; j_CI++    ){
        
        memset(bit_a,0,N*sizeof(int));
        memset(bit_b,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        //AAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign2=1;
            bit_a[t]=0;
            for(int u=0;u<N;u++){
                if(bit_b[u]!=0){
                    sign3=1;
                    bit_b[u]=0;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]==0){
                            sign4=1;
                            bit_a[v]=1;
                            for(int w=0;w<N;w++){
                                if(bit_b[w]==0){
                                    bit_b[w]=1;
                                    sign=sign1*sign2*sign3*sign4;
                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                    auto j_CIext = get_ind_from_ON(bit_b, N, nb, fb, buf);
                                    R = act_INTS[((t*N+v)*N+u)*N+w]*sign;
                                    CI_0 = ci_O+(i_CI   *Nb+j_CI   )*ld;
                                    CI_e = ci_I+(i_CIext*Nb+j_CIext)*ld;
                                    H_s[i_ext]=R;
                                    col_idx[i_ext] = i_CIext*Nb+j_CIext;
                                    i_ext++;
//                                     for(int i_s=0;i_s<n_s;i_s++)
//                                         CI_0[i_s]+=R*CI_e[i_s];
                                    
                                    bit_b[w]=0;
                                }
                                else
                                    sign4=-sign4;
                            }
                            bit_a[v]=0;
                        }
                        else
                            sign3=-sign3;
                    }
                    bit_b[u]=1;
                    sign2=-sign2;
                }
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
    }
    }
    
    fprintf(out_stream,"i_ext = %d\n",i_ext);
    
    
    double * __restrict__ ci_ti = new double [Na*Nb*ld];
    double * __restrict__ ci_to = new double [Na*Nb*ld];
    transpose_3d_abc_to_bac(ci_ti, ci_I, Na*Nb, ld,1);
    transpose_3d_abc_to_bac(ci_to, ci_O, Na*Nb, ld,1);
    printf_timer("generating sp matrix");
    double t[4];
    int * __restrict__ ind;
    double * __restrict__ H;
    
    double * __restrict__ Ci;
    double * __restrict__ Co;
//     double * tmp = new double[n_s];
    Co=ci_to;
    Ci=ci_ti;
    for(int i_s=0; i_s<n_s; i_s++){
        ind = col_idx;
        H   = H_s;
        for(int i = 0; i < Na*Nb; i++){
            t[0]=0;
            t[1]=0;
            t[2]=0;
            t[3]=0;
            for(int j = 0;j<num_ext;j+=4,H+=4,ind+=4){
                t[0] += H[0] * Ci[ind[0]];
                t[1] += H[1] * Ci[ind[1]];
                t[2] += H[2] * Ci[ind[2]];
                t[3] += H[3] * Ci[ind[3]];
            }
            
            Co[i]+=t[0]+t[1]+t[2]+t[3];
        }
        Ci+=Na*Nb;
        Co+=Na*Nb;
    }
    
    printf_timer("mat mult");
    transpose_3d_abc_to_bac(ci_O, ci_to,ld, Na*Nb, 1);
    printf_timer("mat mult");    
                    getchar();

    delete[] bit_a;
    delete[] bit_b;
    delete[] buf;
    
    
    return 0;
}


int aldet_H_mult_2body_AB_2(double * __restrict__ ci_O, int n_s, int ld,
                          double * __restrict__ ci_I,
                          int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                          int * Ia, int * Ib, int * Ia_friends, int * Ib_friends,
                          double * act_INTS,
                          int i_th, int n_th){
    
    
    const int Na_zv = (int) std::lround(tgammal(N) / tgammal(na) / tgammal(N-na+1));
    const int Nb_zv = (int) std::lround(tgammal(N) / tgammal(nb) / tgammal(N-nb+1));
    
//     fprintf(out_stream,"%d\n", n_s);
//     fprintf(out_stream,"%d %d %d %d\n", Na, Nb, Na_zv, Nb_zv);
//     getchar();
    
//     int * Ia = new int [Na_zv];
//     int * Ia_friends = new int [Na_zv * (N-na) * 3];
//     int * Ib = new int [Nb_zv];
//     int * Ib_friends = new int [Nb_zv * (N-nb) * 3];
//     
    double Rc;
    double sign;
    double R;
    double * CI_0;
    double * CI_e;

    int count =0;
    
//     n_s=5;
    
//     for(int t=0  ;t<N;t++){
//         get_I_I_friends(na, N, Na, fa, t, vec_a, Ia, Ia_friends);//if(nt){fprintf(out_stream,"nt>1\n");exit(0);}
//         for(int u=0  ;u<N;u++){//fprintf(stderr,"t=%d u=%d  (thread %d)        \r",t,u, i_th);
//             get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib, Ib_friends);
//             for (int i_CI = i_th; i_CI<Na_zv; i_CI+=n_th){
//                 int k_CI = Ia[i_CI];
//                 for (int j_CI = 0; j_CI<Nb_zv; j_CI++){
//                     int l_CI = Ib[j_CI];
//                     CI_0=ci_O+(k_CI*Nb+l_CI  )*ld;
//                     CI_e=ci_I+(k_CI*Nb+l_CI  )*ld;
//                     R=act_INTS[((t*N+t)*N+u)*N+u];
// //                     count++;
//                     //if(fabs(R)>1e-8)count++;
//                     for(int i_s=0;i_s<n_s;i_s++){
//                         CI_0[i_s]+=R*CI_e[i_s];
//                     }
//                     for (int n = 0; n<(N-nb); n++){
//                     int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
//                         CI_e = ci_I+(k_CI*Nb+ind_n[0])*ld;
//                         R=act_INTS[((t*N+t)*N+u)*N+ind_n[1]]*ind_n[2];
// //                         count++;
//                         //if(fabs(R)>1e-8)count++;
//                         for(int i_s=0;i_s<n_s;i_s++){
//                             CI_0[i_s]+=R*CI_e[i_s];
//                         }
//                     }
//                     for (int m = 0; m<(N-nb); m++){
//                         int * ind_m = Ia_friends + i_CI*(N-na)*3 + m*3;
//                         CI_e = ci_I+(ind_m[0]*Nb+l_CI)*ld;
//                         R=act_INTS[((t*N+ind_m[1])*N+u)*N+u]*ind_m[2];
// //                         count++;
//                         //if(fabs(R)>1e-8)count++;
//                         for(int i_s=0;i_s<n_s;i_s++){
//                             CI_0[i_s]+=R*CI_e[i_s];
//                         }
//                         for (int n = 0; n<(N-nb); n++){
//                             int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
//                             CI_e = ci_I+(ind_m[0]*Nb+ind_n[0])*ld;
//                             sign = ind_m[2]*ind_n[2];
//                             R=act_INTS[((t*N+ind_m[1])*N+u)*N+ind_n[1]]*sign;
// //                             count++;
//                             //if(fabs(R)>1e-8)count++;
//                             for(int i_s=0;i_s<n_s;i_s++){
//                                 CI_0[i_s]+=R*CI_e[i_s];
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
    
    
    double * __restrict__ H_s = new double[Na*Nb*na*nb*(N-na+1)*(N-nb+1)];
    int    * __restrict__ isp = new    int[Na*Nb*na*nb*(N-na+1)*(N-nb+1)];
    int    * __restrict__ jsp = new    int[Na*Nb*na*nb*(N-na+1)*(N-nb+1)];
    
    set_zero_matr    (H_s,Na*Nb*na*nb*(N-na+1)*(N-nb+1));
    set_zero_matr_int(isp,Na*Nb*na*nb*(N-na+1)*(N-nb+1));
    set_zero_matr_int(jsp,Na*Nb*na*nb*(N-na+1)*(N-nb+1));
    printf_timer("alloc");
    
//     n_s=5;
    
    int ind_sp=0;
    for(int t=0  ;t<N;t++){
        get_I_I_friends(na, N, Na, fa, t, vec_a, Ia, Ia_friends);//if(nt){fprintf(out_stream,"nt>1\n");exit(0);}
        for(int u=0  ;u<N;u++){//fprintf(stderr,"t=%d u=%d  (thread %d)        \r",t,u, i_th);
            get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib, Ib_friends);
            for (int i_CI = i_th; i_CI<Na_zv; i_CI+=n_th){
                int k_CI = Ia[i_CI];
                for (int j_CI = 0; j_CI<Nb_zv; j_CI++){
                    int l_CI = Ib[j_CI];
                    
                    isp[ind_sp]=(k_CI*Nb+l_CI)*ld;
                    jsp[ind_sp]=(k_CI*Nb+l_CI)*ld;
                    H_s[ind_sp]=act_INTS[((t*N+t)*N+u)*N+u];
                    ind_sp++;
                    
                    for (int n = 0; n<(N-nb); n++){
                        int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                        isp[ind_sp]=(k_CI*Nb+ind_n[0])*ld;
                        jsp[ind_sp]=(k_CI*Nb+l_CI    )*ld;
                        H_s[ind_sp]=act_INTS[((t*N+t)*N+u)*N+ind_n[1]]*ind_n[2];
                        ind_sp++;
                    }
                    for (int m = 0; m<(N-nb); m++){
                        int * ind_m = Ia_friends + i_CI*(N-na)*3 + m*3;
                        isp[ind_sp]=(ind_m[0]*Nb+l_CI)*ld;
                        jsp[ind_sp]=(k_CI*Nb+l_CI    )*ld;
                        H_s[ind_sp]=act_INTS[((t*N+ind_m[1])*N+u)*N+u]*ind_m[2];
                        ind_sp++;
                        for (int n = 0; n<(N-nb); n++){
                            int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                            isp[ind_sp]=(ind_m[0]*Nb+ind_n[0])*ld;
                            jsp[ind_sp]=(k_CI*Nb+l_CI        )*ld;
                            H_s[ind_sp]=act_INTS[((t*N+ind_m[1])*N+u)*N+ind_n[1]]*ind_m[2]*ind_n[2];
                            ind_sp++;
                        }
                    }
                }
            }
        }
    }
                    fprintf(out_stream,"i_sp = %d\n",ind_sp);
    printf_timer("generating sp matrix");
    
//     blas_sparse_matrix A;
//     A = BLAS_duscr_begin( Na*Nb, Na*Nb );

    
//     int i_CI;
//     int j_CI;
        CI_e=ci_O;
        double * CI_t=ci_I;
        double * CI_o=ci_O;
//         int * ind =isp;
//         int * ind2=jsp;
    for(int i=0;i<ind_sp;i++){
        double R2   =H_s[i];
        CI_t=ci_I+isp[i]/**ld*/;
        CI_o=ci_O+jsp[i]/**ld*/;
       for(int i_s=0; i_s<n_s;i_s ++)
           CI_o[i_s]+=CI_t[i_s]*R2;
    }
        
    
//     double *Rt;
//     Rt= new double[n_s];
//     for(int i_s =0; i_s <n_s                    ;i_s ++)
//         R=0;
    
//     int k_CI;
//     
//     for(int i_CI=0; i_CI<Na*Nb                  ;i_CI++)
//     for(int i_tr=0; i_tr<na*nb*(N-na+1)*(N-nb+1);i_tr++){
//         k_CI=isp[i_CI*na*nb*(N-na+1)*(N-nb+1)+i_tr];
//         for(int i_s =0; i_s <n_s                    ;i_s ++)
//             ci_O[i_CI*ld+i_s]+=ci_I[i*ld+i_s]*H_s[i_CI*na*nb*(N-na+1)*(N-nb+1)+i_tr];
//     }
//         Rt[i_s]+=2.0;
    printf_timer("mat mult");
                    getchar();
    
//     for(int i_s =0; i_s <n_s                    ;i_s ++)
//         R+=Rt[i_s];
//     
//     fprintf(out_stream,"%e\n",R);

//     delete[] Ia        ;
//     delete[] Ia_friends;
//     delete[] Ib        ;
//     delete[] Ib_friends;
    
    return 0;
       
}


int aldet_calc_DM_2body_AA(double * G, int n_s, int ld,
                           double * ci,
                           int N,int na, int Na, int Nb, int * fa, int * vec_a,
                           int i_th, int n_th){
    
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*n_s*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //AAAA
//         sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign2=1;
            bit_a[t]=0;
            for(int u=t;u<N;u++){
                if(bit_a[u]!=0){
                    sign3=1;
                    bit_a[u]=0;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]==0){
                            sign4=1;
                            bit_a[v]=1;
                            for(int w=v;w<N;w++){
                                if(bit_a[w]==0){
                                    bit_a[w]=1;
                                    sign=sign2*sign4;
                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                    R = BUF+(((t*N+u)*N+v)*N+w)*n_s*n_s;
                                    for(int j_CI = 0; j_CI<Nb; j_CI++)
                                    for(int i_s=0;i_s<n_s;i_s++){
                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                        Rc = CI_0[i_s]*sign;
                                        CI_e = ci+(i_CIext*Nb+j_CI)*ld;
                                        for(int j_s=0;j_s<n_s;j_s++)
                                            R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                    }
//                                     R = BUF+(((t*N+w)*N+u)*N+v)*n_s*n_s;
//                                     for(int j_CI = 0; j_CI<Nb; j_CI++)
//                                     for(int i_s=0;i_s<n_s;i_s++){
//                                         CI_0 = ci+(i_CI*Nb+j_CI)*ld;
//                                         Rc = -CI_0[i_s]*sign;
//                                         CI_e = ci+(i_CIext*Nb+j_CI)*ld;
//                                         for(int j_s=0;j_s<n_s;j_s++)
//                                             R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
//                                     }
                                    bit_a[w]=0;
                                }
                                else
                                    sign4=-sign4;
                            }
                            bit_a[v]=0;
                        }
                        else
                            sign3=-sign3;
                    }
                    bit_a[u]=1;
                    sign2=-sign2;
                }
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
    }
    
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    for(int k=i;k<N;k++)
    for(int l=j;l<N;l++){
        G[i_s*N*N*N*N+i*N*N*N+j*N*N+k*N+l]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+j*N*N+i*N+l]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+i*N*N*N+l*N*N+k*N+j]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+l*N*N+i*N+j]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s*n_s+i_s];
    }

    
    
    delete[] bit_a;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}

int aldet_calc_DM_2body_AA_diag(double * G, int n_s, int ld,
                                double * ci,
                                int N,int na, int Na, int Nb, int * fa, int * vec_a,
                                int i_th, int n_th){
    
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*n_s];
    set_zero_matr(BUF, N*N*N*N*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //AAAA
//         sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign2=1;
            bit_a[t]=0;
            for(int u=t;u<N;u++){
                if(bit_a[u]!=0){
                    sign3=1;
                    bit_a[u]=0;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]==0){
                            sign4=1;
                            bit_a[v]=1;
                            for(int w=v;w<N;w++){
                                if(bit_a[w]==0){
                                    bit_a[w]=1;
                                    sign=sign2*sign4;
                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                    R = BUF+(((t*N+u)*N+v)*N+w)*n_s;
                                    for(int j_CI = 0; j_CI<Nb; j_CI++){
                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                        CI_e = ci+(i_CIext*Nb+j_CI)*ld;
                                        for(int i_s=0;i_s<n_s;i_s++)
                                            R[i_s]+=CI_0[i_s]*CI_e[i_s]*sign;
                                    }
                                    bit_a[w]=0;
                                }
                                else
                                    sign4=-sign4;
                            }
                            bit_a[v]=0;
                        }
                        else
                            sign3=-sign3;
                    }
                    bit_a[u]=1;
                    sign2=-sign2;
                }
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
    }
    
    
    for(int i_s=0;i_s<n_s;i_s++)
    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    for(int k=i;k<N;k++)
    for(int l=j;l<N;l++){
        G[i_s*N*N*N*N+i*N*N*N+j*N*N+k*N+l]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+j*N*N+i*N+l]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s+i_s];
        G[i_s*N*N*N*N+i*N*N*N+l*N*N+k*N+j]+= BUF[(i*N*N*N+k*N*N+j*N+l)*n_s+i_s];
        G[i_s*N*N*N*N+k*N*N*N+l*N*N+i*N+j]+=-BUF[(i*N*N*N+k*N*N+j*N+l)*n_s+i_s];
    }

    
    
    delete[] bit_a;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}


int aldet_calc_DM_2body_AB(double * G, int n_s, int ld,
                           double * ci,
                           int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
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
    double * R_AB;
    double * R_BA;
    double * CI_0;
    double * CI_e;

    double * BUF = new double[N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*n_s*n_s);

    
    for(int t=0  ;t<N;t++){
        get_I_I_friends(na, N, Na, fa, t, vec_a, Ia, Ia_friends);//if(nt){fprintf(out_stream,"nt>1\n");exit(0);}
        for(int u=0  ;u<N;u++){//fprintf(stderr,"t=%d u=%d  (thread %d)        \r",t,u, i_th);
            get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib, Ib_friends);
            for (int i_CI = i_th; i_CI<Na_zv; i_CI+=n_th){
                int k_CI = Ia[i_CI];
                for (int j_CI = 0; j_CI<Nb_zv; j_CI++){
                    int l_CI = Ib[j_CI];
                    CI_0=ci+(k_CI*Nb+l_CI  )*ld;
                    R_AB=BUF+(((t*N+t)*N+u)*N+u)*n_s*n_s;
                    R_BA=BUF+(((u*N+u)*N+t)*N+t)*n_s*n_s;
                    for(int i_s=0;i_s<n_s;i_s++){
                        Rc =0;
                        Rc =CI_0[i_s];
                        for(int j_s=0;j_s<n_s;j_s++)
                            R_AB[i_s*n_s+j_s]+=Rc*CI_0[j_s];
                        for(int j_s=0;j_s<n_s;j_s++)
                            R_BA[i_s*n_s+j_s]+=Rc*CI_0[j_s];
                    }
                    for (int n = 0; n<(N-nb); n++){
                    int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                        CI_e = ci+(k_CI*Nb+ind_n[0])*ld;
                        R_AB=BUF+(((t*N+t)*N+u)*N+ind_n[1])*n_s*n_s;
                        R_BA=BUF+(((u*N+ind_n[1])*N+t)*N+t)*n_s*n_s;
                        for(int i_s=0;i_s<n_s;i_s++){
                            Rc =CI_0[i_s]*ind_n[2];
                            for(int j_s=0;j_s<n_s;j_s++)
                                R_AB[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                            for(int j_s=0;j_s<n_s;j_s++)
                                R_BA[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                        }
                    }
                    for (int m = 0; m<(N-na); m++){
                        int * ind_m = Ia_friends + i_CI*(N-na)*3 + m*3;
                        CI_e = ci+(ind_m[0]*Nb+l_CI)*ld;
                        R_AB=BUF+(((t*N+ind_m[1])*N+u)*N+u)*n_s*n_s;
                        R_BA=BUF+(((u*N+u)*N+t)*N+ind_m[1])*n_s*n_s;
                        for(int i_s=0;i_s<n_s;i_s++){
                            Rc =CI_0[i_s]*ind_m[2];
                            for(int j_s=0;j_s<n_s;j_s++)
                                R_AB[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                            for(int j_s=0;j_s<n_s;j_s++)
                                R_BA[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                        }
                        for (int n = 0; n<(N-nb); n++){
                            int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                            CI_e = ci+(ind_m[0]*Nb+ind_n[0])*ld;
                            R_AB=BUF+(((t*N+ind_m[1])*N+u)*N+ind_n[1])*n_s*n_s;
                            R_BA=BUF+(((u*N+ind_n[1])*N+t)*N+ind_m[1])*n_s*n_s;
                            sign = ind_m[2]*ind_n[2];
                            for(int i_s=0;i_s<n_s;i_s++){
                                Rc =CI_0[i_s]*sign;
                                for(int j_s=0;j_s<n_s;j_s++)
                                    R_AB[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                for(int j_s=0;j_s<n_s;j_s++)
                                    R_BA[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                            }
                        }
                    }
                }
            }
        }
    }
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N*N*N*N;i++)
//     for(int j=0;j<n_act;j++)
//     for(int j_s=0;j_s<n_sK;j_s++)
        G[i_s*N*N*N*N+i]+=BUF[i*n_s*n_s+i_s];
    
    
    delete[] BUF;
    delete[] Ia        ;
    delete[] Ia_friends;
    delete[] Ib        ;
    delete[] Ib_friends;
    
    return 0;
       
}

int aldet_calc_DM_2body_AB_diag(double * G, int n_s, int ld,
                                double * ci,
                                int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
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
    double * R_AB;
    double * R_BA;
    double * CI_0;
    double * CI_e;

    double * BUF = new double[N*N*N*N*n_s];
    set_zero_matr(BUF, N*N*N*N*n_s);

    
    for(int t=0  ;t<N;t++){
        get_I_I_friends(na, N, Na, fa, t, vec_a, Ia, Ia_friends);//if(nt){fprintf(out_stream,"nt>1\n");exit(0);}
        for(int u=0  ;u<N;u++){//fprintf(stderr,"t=%d u=%d  (thread %d)        \r",t,u, i_th);
            get_I_I_friends(nb, N, Nb, fb, u, vec_b, Ib, Ib_friends);
            for (int i_CI = i_th; i_CI<Na_zv; i_CI+=n_th){
                int k_CI = Ia[i_CI];
                for (int j_CI = 0; j_CI<Nb_zv; j_CI++){
                    int l_CI = Ib[j_CI];
                    CI_0=ci+(k_CI*Nb+l_CI  )*ld;
                    R_AB=BUF+(((t*N+t)*N+u)*N+u)*n_s;
                    R_BA=BUF+(((u*N+u)*N+t)*N+t)*n_s;
                    for(int i_s=0;i_s<n_s;i_s++){
                        Rc =0;
                        Rc =CI_0[i_s];
                        R_AB[i_s]+=Rc*CI_0[i_s];
                        R_BA[i_s]+=Rc*CI_0[i_s];
                    }
                    for (int n = 0; n<(N-nb); n++){
                    int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                        CI_e = ci+(k_CI*Nb+ind_n[0])*ld;
                        R_AB=BUF+(((t*N+t)*N+u)*N+ind_n[1])*n_s;
                        R_BA=BUF+(((u*N+ind_n[1])*N+t)*N+t)*n_s;
                        for(int i_s=0;i_s<n_s;i_s++){
                            Rc =CI_0[i_s]*ind_n[2];
                            R_AB[i_s]+=Rc*CI_e[i_s];
                            R_BA[i_s]+=Rc*CI_e[i_s];
                        }
                    }
                    for (int m = 0; m<(N-na); m++){
                        int * ind_m = Ia_friends + i_CI*(N-na)*3 + m*3;
                        CI_e = ci+(ind_m[0]*Nb+l_CI)*ld;
                        R_AB=BUF+(((t*N+ind_m[1])*N+u)*N+u)*n_s;
                        R_BA=BUF+(((u*N+u)*N+t)*N+ind_m[1])*n_s;
                        for(int i_s=0;i_s<n_s;i_s++){
                            Rc =CI_0[i_s]*ind_m[2];
                            R_AB[i_s]+=Rc*CI_e[i_s];
                            R_BA[i_s]+=Rc*CI_e[i_s];
                        }
                        for (int n = 0; n<(N-nb); n++){
                            int * ind_n = Ib_friends + j_CI*(N-nb)*3 + n*3;
                            CI_e = ci+(ind_m[0]*Nb+ind_n[0])*ld;
                            R_AB=BUF+(((t*N+ind_m[1])*N+u)*N+ind_n[1])*n_s;
                            R_BA=BUF+(((u*N+ind_n[1])*N+t)*N+ind_m[1])*n_s;
                            sign = ind_m[2]*ind_n[2];
                            for(int i_s=0;i_s<n_s;i_s++){
                                Rc =CI_0[i_s]*sign;
                                R_AB[i_s]+=Rc*CI_e[i_s];
                                R_BA[i_s]+=Rc*CI_e[i_s];
                            }
                        }
                    }
                }
            }
        }
    }
    
    for(int i_s=0;i_s<n_s;i_s++)
    for(int i=0;i<N*N*N*N;i++)
//     for(int j=0;j<n_act;j++)
//     for(int j_s=0;j_s<n_sK;j_s++)
        G[i_s*N*N*N*N+i]+=BUF[i*n_s+i_s];
    
    
    delete[] BUF;
    delete[] Ia        ;
    delete[] Ia_friends;
    delete[] Ib        ;
    delete[] Ib_friends;
    
    return 0;
       
}


int aldet_calc_DM_3body_AAA(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int Na, int Nb, int * fa, int * vec_a,
                            int i_th, int n_th){
    
    int * bit_a = new int [N];
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*N*N*n_s*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //AAAAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign2=1;
            bit_a[t]=0;
            for(int u=0;u<N;u++){
                if(bit_a[u]!=0){
                    sign3=1;
                    bit_a[u]=0;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]!=0){
                            sign4=1;
                            bit_a[v]=0;
                            for(int w=0;w<N;w++){
                                if(bit_a[w]==0){
                                    sign5=1;
                                    bit_a[w]=1;
                                    for(int x=0;x<N;x++){
                                        if(bit_a[x]==0){
                                            sign6=1;
                                            bit_a[x]=1;
                                            for(int y=0;y<N;y++){
                                                if(bit_a[y]==0){
                                                    bit_a[y]=1;
                                                                        
                                                    sign=sign1*sign2*sign3*sign4*sign5*sign6;
                                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                    R = BUF+(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*n_s*n_s;
                                                    for(int j_CI = 0; j_CI<Nb; j_CI++)
                                                    for(int i_s=0;i_s<n_s;i_s++){
                                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                        Rc = CI_0[i_s]*sign;
                                                        CI_e = ci+(i_CIext*Nb+j_CI)*ld;
                                                        for(int j_s=0;j_s<n_s;j_s++)
                                                            R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                                    }
                                                    bit_a[y]=0;
                                                }
                                                else
                                                    sign6=-sign6;
                                            }
                                            bit_a[x]=0;
                                        }
                                        else
                                            sign5=-sign5;
                                    }
                                    bit_a[w]=0;
                                }
                                else
                                    sign4=-sign4;
                            }
                            bit_a[v]=1;
                            sign3=-sign3;
                        }
                        
                    }
                    bit_a[u]=1;
                    sign2=-sign2;
                }
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
    }
    
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N*N*N*N*N*N;i++)
        G[i_s*N*N*N*N*N*N+i]+=BUF[i*n_s*n_s+i_s];

    
    
    delete[] bit_a;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}

int aldet_calc_DM_3body_AAA_1(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int Na, int Nb, int * fa, int * vec_a,
                            int i_th, int n_th){
    
    int * bit_a = new int [N];
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*N*N*n_s*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //AAAAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]==0){
            sign2=1;
            bit_a[t]=1;
            for(int u=0;u<N;u++){
                if(bit_a[u]==0){
                    sign3=1;
                    bit_a[u]=1;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]!=0){
                            sign4=1;
                            bit_a[v]=0;
                            for(int w=0;w<N;w++){
                                if(bit_a[w]!=0){
                                    sign5=1;
                                    bit_a[w]=0;
                                    for(int x=0;x<N;x++){
                                        if(bit_a[x]==0){
                                            sign6=1;
                                            bit_a[x]=1;
                                            for(int y=0;y<N;y++){
                                                if(bit_a[y]!=0){
                                                    bit_a[y]=0;
                                                                        
                                                    sign=sign1*sign2*sign3*sign4*sign5*sign6;
                                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                    R = BUF+(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*n_s*n_s;
                                                    for(int j_CI = 0; j_CI<Nb; j_CI++)
                                                    for(int i_s=0;i_s<n_s;i_s++){
                                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                        Rc = CI_0[i_s]*sign;
                                                        CI_e = ci+(i_CIext*Nb+j_CI)*ld;
                                                        for(int j_s=0;j_s<n_s;j_s++)
                                                            R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                                    }
                                                    bit_a[y]=1;
                                                    sign6=-sign6;
                                                }
                                            }
                                            bit_a[x]=0;
                                        }
                                        else
                                            sign5=-sign5;
                                    }
                                    bit_a[w]=1;
                                    sign4=-sign4;
                                }
                            }
                            bit_a[v]=1;
                            sign3=-sign3;
                        }
                        
                    }
                    bit_a[u]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_a[t]=0;
        }
        else
            sign1=-sign1;
    }
    
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N*N*N*N*N*N;i++)
        G[i_s*N*N*N*N*N*N+i]+=BUF[i*n_s*n_s+i_s];

    
    
    delete[] bit_a;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}

int aldet_calc_DM_3body_AAA_2(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int Na, int Nb, int * fa, int * vec_a,
                            int i_th, int n_th){
    
    int * bit_a = new int [N];
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*N*N*n_s*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        //AAAAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]==0){
            sign2=1;
            bit_a[t]=1;
            for(int u=0;u<N;u++){
                if(bit_a[u]==0){
                    sign3=1;
                    bit_a[u]=1;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]!=0){
                            sign4=1;
                            bit_a[v]=0;
                            for(int w=0;w<N;w++){
                                if(bit_a[w]!=0){
                                    sign5=1;
                                    bit_a[w]=0;
                                    for(int x=0;x<N;x++){
                                        if(bit_a[x]!=0){
                                            sign6=1;
                                            bit_a[x]=0;
                                            for(int y=0;y<N;y++){
                                                if(bit_a[y]==0){
                                                    bit_a[y]=1;
                                                                        
                                                    sign=sign1*sign2*sign3*sign4*sign5*sign6;
                                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                    R = BUF+(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*n_s*n_s;
                                                    for(int j_CI = 0; j_CI<Nb; j_CI++)
                                                    for(int i_s=0;i_s<n_s;i_s++){
                                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                        Rc = CI_0[i_s]*sign;
                                                        CI_e = ci+(i_CIext*Nb+j_CI)*ld;
                                                        for(int j_s=0;j_s<n_s;j_s++)
                                                            R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                                    }
                                                    bit_a[y]=0;
                                                }
                                                else
                                                    sign6=-sign6;
                                            }
                                            bit_a[x]=1;
                                            sign5=-sign5;
                                        }
                                    }
                                    bit_a[w]=1;
                                    sign4=-sign4;
                                }
                            }
                            bit_a[v]=1;
                            sign3=-sign3;
                        }
                        
                    }
                    bit_a[u]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_a[t]=0;
        }
        else
            sign1=-sign1;
    }
    
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N*N*N*N*N*N;i++)
        G[i_s*N*N*N*N*N*N+i]+=BUF[i*n_s*n_s+i_s];

    
    
    delete[] bit_a;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}



int aldet_calc_DM_3body_AAB(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                            int i_th, int n_th){
    
    int * bit_a = new int [N];
    int * bit_b = new int [N];
    int * buf = new int [max(na+1,nb+1) + 1];//more space for +a-b and -a+b
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*N*N*n_s*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        memset(bit_b,0,N*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        //AAAAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]!=0){
            sign2=1;
            bit_a[t]=0;
            for(int u=0;u<N;u++){
                if(bit_a[u]!=0){
                    sign3=1;
                    bit_a[u]=0;
                    for(int v=0  ;v<N;v++){
                        if(bit_b[v]!=0){
                            sign4=1;
                            bit_b[v]=0;
                            for(int w=0;w<N;w++){
                                if(bit_a[w]==0){
                                    sign5=1;
                                    bit_a[w]=1;
                                    for(int x=0;x<N;x++){
                                        if(bit_a[x]==0){
                                            sign6=1;
                                            bit_a[x]=1;
                                            for(int y=0;y<N;y++){
                                                if(bit_b[y]==0){
                                                    bit_b[y]=1;
                                                                        
                                                    sign=sign1*sign2*sign3*sign4*sign5*sign6;
                                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                    auto j_CIext = get_ind_from_ON(bit_b, N, nb, fb, buf);
                                                    R = BUF+(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*n_s*n_s;
//                                                     for(int j_CI = 0; j_CI<Nb; j_CI++)
                                                    for(int i_s=0;i_s<n_s;i_s++){
                                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                        Rc = CI_0[i_s]*sign;
                                                        CI_e = ci+(i_CIext*Nb+j_CIext)*ld;
                                                        for(int j_s=0;j_s<n_s;j_s++)
                                                            R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                                    }
                                                    bit_b[y]=0;
                                                }
                                                else
                                                    sign6=-sign6;
                                            }
                                            bit_a[x]=0;
                                        }
                                        else
                                            sign5=-sign5;
                                    }
                                    bit_a[w]=0;
                                }
                                else
                                    sign4=-sign4;
                            }
                            bit_b[v]=1;
                            sign3=-sign3;
                        }
                        
                    }
                    bit_a[u]=1;
                    sign2=-sign2;
                }
            }
            bit_a[t]=1;
            sign1=-sign1;
        }
    }
    }
    
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N*N*N*N*N*N;i++)
        G[i_s*N*N*N*N*N*N+i]+=BUF[i*n_s*n_s+i_s];

    
    
    delete[] bit_a;
    delete[] bit_b;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}


int aldet_calc_DM_3body_AAB_2(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                            int i_th, int n_th){
    
    int * bit_a = new int [N];
    int * bit_b = new int [N];
    int * buf = new int [max(na+1,nb+1) + 1];//more space for +a-b and -a+b
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*N*N*n_s*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        memset(bit_b,0,N*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        //AAAAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]==0){
            sign2=1;
            bit_a[t]=1;
            for(int u=0;u<N;u++){
                if(bit_a[u]==0){
                    sign3=1;
                    bit_a[u]=1;
                    for(int v=0  ;v<N;v++){
                        if(bit_a[v]!=0){
                            sign4=1;
                            bit_a[v]=0;
                            for(int w=0;w<N;w++){
                                if(bit_b[w]!=0){
                                    sign5=1;
                                    bit_b[w]=0;
                                    for(int x=0;x<N;x++){
                                        if(bit_b[x]==0){
                                            sign6=1;
                                            bit_b[x]=1;
                                            for(int y=0;y<N;y++){
                                                if(bit_a[y]!=0){
                                                    bit_a[y]=0;
                                                                        
                                                    sign=sign1*sign2*sign3*sign4*sign5*sign6;
                                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                    auto j_CIext = get_ind_from_ON(bit_b, N, nb, fb, buf);
                                                    R = BUF+(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*n_s*n_s;
//                                                     for(int j_CI = 0; j_CI<Nb; j_CI++)
                                                    for(int i_s=0;i_s<n_s;i_s++){
                                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                        Rc = CI_0[i_s]*sign;
                                                        CI_e = ci+(i_CIext*Nb+j_CIext)*ld;
                                                        for(int j_s=0;j_s<n_s;j_s++)
                                                            R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                                    }
                                                    bit_a[y]=1;
                                                    sign6=-sign6;
                                                }
                                            }
                                            bit_b[x]=0;
                                        }
                                        else
                                            sign5=-sign5;
                                    }
                                    bit_b[w]=1;
                                    sign4=-sign4;
                                }
                            }
                            bit_a[v]=1;
                            sign3=-sign3;
                        }
                        
                    }
                    bit_a[u]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_a[t]=0;
        }
        else
            sign1=-sign1;
    }
    }
    
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N*N*N*N*N*N;i++)
        G[i_s*N*N*N*N*N*N+i]+=BUF[i*n_s*n_s+i_s];

    
    
    delete[] bit_a;
    delete[] bit_b;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}

int aldet_calc_DM_3body_AAB_3(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                            int i_th, int n_th){
    
    int * bit_a = new int [N];
    int * bit_b = new int [N];
    int * buf = new int [max(na+1,nb+1) + 1];//more space for +a-b and -a+b
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
    double * CI_0;
    double * CI_e;
    
    double * BUF = new double[N*N*N*N*N*N*n_s*n_s];
    set_zero_matr(BUF, N*N*N*N*N*N*n_s*n_s);
    
    for(int i_CI =i_th; i_CI<Na; i_CI+=n_th){//fprintf(stderr,"i_CI=%d  (thread %d)        \r",i_CI, i_th);
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        memset(bit_a,0,N*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        memset(bit_b,0,N*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        //AAAAAA
        sign1=1;
        for(int t=0  ;t<N;t++)if(bit_a[t]==0){
            sign2=1;
            bit_a[t]=1;
            for(int u=0;u<N;u++){
                if(bit_b[u]==0){
                    sign3=1;
                    bit_b[u]=1;
                    for(int v=0  ;v<N;v++){
                        if(bit_b[v]!=0){
                            sign4=1;
                            bit_b[v]=0;
                            for(int w=0;w<N;w++){
                                if(bit_a[w]!=0){
                                    sign5=1;
                                    bit_a[w]=0;
                                    for(int x=0;x<N;x++){
                                        if(bit_a[x]!=0){
                                            sign6=1;
                                            bit_a[x]=0;
                                            for(int y=0;y<N;y++){
                                                if(bit_a[y]==0){
                                                    bit_a[y]=1;
                                                                        
                                                    sign=sign1*sign2*sign3*sign4*sign5*sign6;
                                                    auto i_CIext = get_ind_from_ON(bit_a, N, na, fa, buf);
                                                    auto j_CIext = get_ind_from_ON(bit_b, N, nb, fb, buf);
                                                    R = BUF+(((((t*N+u)*N+v)*N+w)*N+x)*N+y)*n_s*n_s;
//                                                     for(int j_CI = 0; j_CI<Nb; j_CI++)
                                                    for(int i_s=0;i_s<n_s;i_s++){
                                                        CI_0 = ci+(i_CI*Nb+j_CI)*ld;
                                                        Rc = CI_0[i_s]*sign;
                                                        CI_e = ci+(i_CIext*Nb+j_CIext)*ld;
                                                        for(int j_s=0;j_s<n_s;j_s++)
                                                            R[i_s*n_s+j_s]+=Rc*CI_e[j_s];
                                                    }
                                                    bit_a[y]=0;
                                                }
                                                else
                                                    sign6=-sign6;
                                            }
                                            bit_a[x]=1;
                                            sign5=-sign5;
                                        }
                                    }
                                    bit_a[w]=1;
                                    sign4=-sign4;
                                }
                            }
                            bit_b[v]=1;
                            sign3=-sign3;
                        }
                        
                    }
                    bit_b[u]=0;
                }
                else
                    sign2=-sign2;
            }
            bit_a[t]=0;
        }
        else
            sign1=-sign1;
    }
    }
    
    
    for(int i_s=0;i_s<n_s*n_s;i_s++)
    for(int i=0;i<N*N*N*N*N*N;i++)
        G[i_s*N*N*N*N*N*N+i]+=BUF[i*n_s*n_s+i_s];

    
    
    delete[] bit_a;
    delete[] bit_b;
    delete[] buf;
    delete[] BUF;
    
    return 0;
}


int ci_ext(aldet_data * O, int dir, aldet_data * I, int i_set){
    
    
    O->get_dim(I->n_act, I->na+dir, I->nb, 1, -1, I->print_number);
    
    O->U_simple_import_data(I->act_INTS_AA,
                            I->act_INTS_AB,
                            I->act_INTS_BB,
                            I->F_act_A,
                            I->F_act_B,
                            I->E_core);
    
    double sign;
    int n_s =I->n_states[i_set];
    int Na =I->Na;
    int NaE=O->Na;
    int Nb =I->Nb;
    int na = I->na;
    int naE= I->na+dir;
    int n_act = I->n_act;
    int * bit_a = I->bit_a;
    int * vec_a = I->vec_a;
    int * bufE = O->buf;
    int * faE = O->fa;
    O->init_zero_vec(n_s*n_act, 0);
    O->n_states[0]=n_s*n_act;
    if(dir==-1)
    for(int i_CI = 0; i_CI<Na; i_CI++){
        
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        sign=1;
        for(int t=0  ;t<n_act;t++)if(bit_a[t]==1){
            bit_a[t]=0;
            
            auto i_CIext = get_ind_from_ON(bit_a, n_act, naE, faE, bufE);
            
            for(int j_CI = 0; j_CI<Nb; j_CI++)
            for(int i_s=0;i_s<n_s;i_s++)
                O->coef[0    ][(i_CIext*Nb+j_CI)*n_s*n_act+i_s*n_act+t]+=
                I->coef[i_set][(i_CI   *Nb+j_CI)*n_s+i_s]*sign;
            
            bit_a[t]=1;
            sign=-sign;
            
        }
    }
    if(dir==1)
    for(int i_CI = 0; i_CI<Na; i_CI++){
        memset(bit_a,0,n_act*sizeof(int));
        for (int k=1; k<na+1; k++) bit_a[vec_a[i_CI*(na+1)+k]-1] = 1;
        
        sign=1;
        for(int t=0  ;t<n_act;t++)if(bit_a[t]==0){
            bit_a[t]=1;
            
            auto i_CIext = get_ind_from_ON(bit_a, n_act, naE, faE, bufE);
            
            for(int j_CI = 0; j_CI<Nb; j_CI++)
            for(int i_s=0;i_s<n_s;i_s++)
                O->coef[0    ][(i_CIext*Nb+j_CI)*n_s*n_act+i_s*n_act+t]+=
                I->coef[i_set][(i_CI   *Nb+j_CI)*n_s+i_s]*sign;
            
            bit_a[t]=0;
        }
        else
            sign=-sign;
            
    }

    
    O->transpose_ci(0);
    
    
    
    
    return 0;
}


int ci_ext_B(aldet_data * O, int dir, aldet_data * I, int i_set){
    
    
    O->get_dim(I->n_act, I->na, I->nb+dir, 1, -1, I->print_number);
    
    O->U_simple_import_data(I->act_INTS_AA,
                            I->act_INTS_AB,
                            I->act_INTS_BB,
                            I->F_act_A,
                            I->F_act_B,
                            I->E_core);
    
    double sign;
    int n_s =I->n_states[i_set];
    int Na =I->Na;
    int Nb =I->Nb;
    int NbE=O->Nb;
    int nb = I->nb;
    int nbE= I->nb+dir;
    int n_act = I->n_act;
    int * bit_b = O->bit_b;
    int * vec_b = O->vec_b;
    int * bufE = O->buf;
    int * fbE = O->fb;
    O->init_zero_vec(n_s*n_act, 0);
    O->n_states[0]=n_s*n_act;
    if(dir==-1)
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        sign=1;
        for(int t=0  ;t<n_act;t++)if(bit_b[t]==1){
            bit_b[t]=0;
            
            auto j_CIext = get_ind_from_ON(bit_b, n_act, nbE, fbE, bufE);
            
            for(int i_CI = 0; i_CI<Na; i_CI++)
            for(int i_s=0;i_s<n_s;i_s++)
                O->coef[0    ][(i_CI*NbE+j_CIext)*n_s*n_act+i_s*n_act+t]+=
                I->coef[i_set][(i_CI*Nb +j_CI   )*n_s+i_s]*sign;
            
            bit_b[t]=1;
            sign=-sign;
            
        }
    }
    if(dir==1)
    for(int j_CI = 0; j_CI<Nb; j_CI++){
        memset(bit_b,0,n_act*sizeof(int));
        for (int k=1; k<nb+1; k++) bit_b[vec_b[j_CI*(nb+1)+k]-1] = 1;
        
        sign=1;
        for(int t=0  ;t<n_act;t++)if(bit_b[t]==0){
            bit_b[t]=1;
            
            auto j_CIext = get_ind_from_ON(bit_b, n_act, nbE, fbE, bufE);
            
            for(int i_CI = 0; i_CI<Na; i_CI++)
            for(int i_s=0;i_s<n_s;i_s++)
                O->coef[0    ][(i_CI*NbE+j_CIext)*n_s*n_act+i_s*n_act+t]+=
                I->coef[i_set][(i_CI*Nb +j_CI   )*n_s+i_s]*sign;
            
            bit_b[t]=0;
        }
        else
            sign=-sign;
            
    }

    
    O->transpose_ci(0);
    
//     I->print_states(0,2);
//     getchar();
//     O->print_states(0,12);
//     getchar();
    
    
    return 0;
}

int ci_rotate_Pi_pair(aldet_data * CI, int i_set, double sin_for_phi, double cos_for_phi, int number_of_Pi_pair, int * ind_pi_states){
    
    int n_s = CI->n_states[i_set]; 
    int Na =  CI->Na;
    int Nb =  CI->Nb;
    double old_coef_first_in_pi_pair, old_coef_second_in_pi_pair, new_coef_first_in_pi_pair, new_coef_second_in_pi_pair;
    for(int i_CI = 0; i_CI<Na*Nb; i_CI++){
        old_coef_first_in_pi_pair =CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2  ]];
        old_coef_second_in_pi_pair=CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2+1]];
        
       //// printf("%e %e\n", sin_for_phi, sqrt(1-sin_for_phi*sin_for_phi));
       //// getchar();
        
        new_coef_first_in_pi_pair = sin_for_phi*old_coef_first_in_pi_pair + cos_for_phi*old_coef_second_in_pi_pair;
        new_coef_second_in_pi_pair= cos_for_phi*old_coef_first_in_pi_pair - sin_for_phi*old_coef_second_in_pi_pair;
        
        CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2]]   = new_coef_first_in_pi_pair ;
        CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2+1]] = new_coef_second_in_pi_pair;
    }

    CI->transpose_ci(0);
    return 0;
}

//// int ci_diabatize_Pi(aldet_data * CI, int i_set, double sin_for_phi, double cos_for_phi, int number_of_Pi_pair, int * ind_pi_states){
////     
////     int n_s = CI->n_states[i_set]; 
////     int Na =  CI->Na;
////     int Nb =  CI->Nb;
////     double old_coef_second_pi, old_coef_third_pi, new_coef_second_pi, new_coef_third_pi;
////     for(int i_CI = 0; i_CI<Na*Nb; i_CI++){
////         old_coef_second_pi =CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2  ]];
////         old_coef_third_pi=CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2+1]]  ;
////     
////     //// printf("%e %e\n", sin_for_phi, sqrt(1-sin_for_phi*sin_for_phi));
////         getchar();
////         
////         new_coef_second_pi = sin_for_phi*old_coef_second_pi + cos_for_phi*old_coef_third_pi;
////         new_coef_third_pi  = cos_for_phi*old_coef_second_pi - sin_for_phi*old_coef_third_pi;
////         
////         CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2  ]] = new_coef_second_pi;
////         CI->coef[i_set][i_CI*n_s+ind_pi_states[number_of_Pi_pair*2+1]] = new_coef_third_pi ;
//// 
//// 
//// 
//// 
//// 
////     }
//// 
////     CI->transpose_ci(0);
////     return 0;
//// }
