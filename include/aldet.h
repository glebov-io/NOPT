#ifndef ALDET_H
#define ALDET_H
// #include "molecule.h"
#include "CI.h"
#include "nopt_typedef.h"

class sparsed_CI_vec;

class sparsed_CI_vec{
    public:
        std::vector<double> c;
        std::vector<ci_int> n;
        
        int decompress(double * C, int a, int n_s);
    
};

int CI_sd_mult(double * O, int ld, sparsed_CI_vec * s, int n_s, double * d, int n_d, int ld_d);

int CI_sd_mult_tr(double * O, int ld, sparsed_CI_vec * s, int n_s, double * R, int n_d, int ld_d);

int CI_ss_mult(double * O, int ld, sparsed_CI_vec * s1, int n_s1, sparsed_CI_vec * s2, int n_s2);

class aldet_data;

// get_dim build modes. DIMS stops after the active-space dimensions and the per-set state
// tables: nothing sized by Na/Nb is allocated, so the CI engine cannot run on the object.
enum aldet_alloc_kind { ALDET_ALLOC_FULL = 0, ALDET_ALLOC_DIMS = 1 };

class aldet_data 
{
    public:
        //CI coefficients
        double ** coef;//abs -- alpha-beta-state
        double ** coef_bas;//transpositions
        double ** coef_asb;//transpositions
        double ** coef_bsa;//transpositions
        
        //for storing multiple data
        int n_sets;
        int * n_states;
        
        //active space
        int n_act;
        int na;
        int nb;
        int mult;
        int print_number;
        
        //calculated variables
        ci_int Na;//number of alpha configurations
        ci_int Nb;//number of beta configurations
        ci_int Nd;//number of all configurations
        int * fa;
        int * fb;
        int * vec_a;
        int * vec_b;
        int * bit_a;
        int * bit_b;
        int * buf;
        
        //bufer for energy
        double ** E_states;
        double ** S2;
        double ** L2;
        double ** P ;
        //sum of active orbital energies
        double ** E_act;
        
        // arrays for storing results of multiplication of H and CI vectors
        double * Hv;    //actual
        double * Hv_buf;//bufer
        
        // diagonal elements of H
        double * H_diag;
        double * H_diag_appr;
        
        int import_done;
        double   E_core;    
        double * F_act_A;    
        double * F_act_B;    
        double * act_INTS_AA;
        double * act_INTS_AB;
        double * act_INTS_BB;
        
        int do_PT;
        double * T3_AAA;
        double * T3_AAB;
        double * T3_BBA;
        double * T3_BBB;
        double * T2_AA ;
        double * T2_AB ;
        double * T2_BB ;
        double * T1_A  ;
        double * T1_B  ;
        double   T0    ;
        
        double   Lambda_core;
        double * Lambda_act;
        int * act_rep_num;
        
        
        double * J_act_a;
        double * J_act_b;
        double * K_act_a;
        double * K_act_b;
        
        
        
        int * Ia;
        int * Ib;
        int * Ia_friends;
        int * Ib_friends;
        
        double sym_ab;
        
        int n_e1_a;
        int n_e1_b;
        ci_int * __restrict__ e1_ind_a ;  
        ci_int * __restrict__ e1_ind_b ; 
        ci_int * __restrict__ e1_orbs_a;  
        ci_int * __restrict__ e1_orbs_b;  
//         int * __restrict__ e1_asm_ints_b; //may be not needed  
        
        double * __restrict__ e1_sign_a;  
        double * __restrict__ e1_sign_b;  
        
        int n_e2_a;
        int n_e2_b;
        ci_int * __restrict__ e2_orbs_a;  
        ci_int * __restrict__ e2_orbs_b;  
        ci_int * __restrict__ e2_ind_a ;  
        ci_int * __restrict__ e2_ind_b ; 
        double * __restrict__ e2_V_a;  
        double * __restrict__ e2_V_b;  
        double * __restrict__ e2_sign_a;  
        double * __restrict__ e2_sign_b;  
        
        
        int n_e3_a;
        int n_e3_b;
        ci_int * __restrict__ e3_ind_a ;  
        ci_int * __restrict__ e3_ind_b ; 
        double * __restrict__ e3_V_a;  
        double * __restrict__ e3_V_b;  
        
        
        double* a_spin_sign;
        double* b_spin_sign;
        
        int alloc_mode;// aldet_alloc_kind: FULL (default) | DIMS
        
        aldet_data(); 
        // Active-space dimensions and the per-set state tables. Allocates nothing sized by
        // Na/Nb -- the determinant space itself is built by get_dim.
        int get_dim_meta(int ext_n_act, int ext_na, int ext_nb, int ext_n_sets, int ext_mult, int ext_print_number);
        int get_dim(int ext_n_act, int ext_na, int ext_nb, int ext_n_sets, int ext_mult, int ext_print_number, int ext_alloc = ALDET_ALLOC_FULL);
        
        int simple_import_data(double * ext_act_INTS,
                               double * ext_act_INTS_AB,
                               double * ext_F_act,
                               double   ext_E_core);
        
        int U_simple_import_data(double * ext_act_INTS_AA,
                                 double * ext_act_INTS_AB,
                                 double * ext_act_INTS_BB,
                                 double * ext_F_act_A,
                                 double * ext_F_act_B,
                                 double   ext_E_core);
        
        int PT2_alloc();
        
        int PT2_import_data(double * ext_T3,
                            double * ext_T3_AB,
                            double * ext_T2,
                            double * ext_T2_AB,
                            double * ext_T1,
                            double   ext_T0);
        
        int UPT2_import_data(double * ext_T3_AAA,
                             double * ext_T3_AAB,
                             double * ext_T3_BBA,
                             double * ext_T3_BBB,
                             double * ext_T2_AA,
                             double * ext_T2_AB,
                             double * ext_T2_BB,
                             double * ext_T1_A,
                             double * ext_T1_B,
                             double   ext_T0);
        
        int PT2_delete_data();
        
        int gen_ext_ind();
        int gen_ext_ind_PT();
        int init_zero_vec(int n_s, int i_set);
//         int read_set_from_mol(molecule * A, int n, int f, int n_s, int i);
        int read_set_from_ci_map_arr(ci_map_arr * ci, int n_s, int i_set);
        int write_civec(int i_s, char *out_name);
        int read_civec(int i_set, char * file_name, std::vector<int> n_st);
        int transpose_ci(int i_set);
        int copy_coef(int i_set, aldet_data * C, int n_s, int i_set_inp, int do_transpose);
//         int get_ci_data
        int calc_S(double * O, int a, int b);
        int calc_IPEA_single(double * U_IP, double * H_IP, 
                             double * U_EA, double * H_EA,
                             int a, std::vector<double> avecoe);
        
        int calc_IPEA_single_AB(double * U_IP_A, double * H_IP_A, 
                                double * U_EA_A, double * H_EA_A,
                                double * U_IP_B, double * H_IP_B, 
                                double * U_EA_B, double * H_EA_B,
                                int a, std::vector<double> avecoe);
        
        int calc_IPEA_single_CA(double * U_IP, double * H_IP, 
                                double * U_EA, double * H_EA,
                                int n_core, int n_mo, int ld,
                                int a, std::vector<double> avecoe,
                                double * F_ga, double * aaag_ints);
        
        int calc_IPEA_double(double * U_IP, double * H_IP, 
                             double * U_EA, double * H_EA,
                             double * U_IP2, double * H_IP2,
                             double * U_EA2, double * H_EA2,
                             double * U_IP_AB, double * H_IP_AB,
                             double * U_EA_AB, double * H_EA_AB,
                             int a, std::vector<double> avecoe);
        
//         int calc_av_DMB(double * O, int a, std::vector<double> avecoe);
        
        int calc_DM (double * O, int a, int b);
        int calc_DMA(double * O, int a, int b);
        int calc_DMB(double * O, int a, int b);
        int calc_DM_part(double *O, double *B, double *K, int nsB, int nsK, int n_e, int N1, int N2, int * f1, int * f2, int * vec, int * bit);
        
        int calc_DM_diag (double * O, int a);
        int calc_DMA_diag(double * O, int a);
        int calc_DMB_diag(double * O, int a);
        int calc_DM_diag_part(double *O, double *C, int ns, int n_e, int N1, int N2, int * f1, int * f2, int * vec, int * bit);
        
        
        int malmqvist(int i_set, double * U);
        int E_act_calc(double * E, int i_set);
        int F_calc(double * F, int i_set);
        
        
        int H_calc(double * H, int n_s);
        int H_full_calc(double *H);
        
        
        int PT_update();
        int H_mult(int n0, int n_s);
        
        int H_mult_sparsed_to_dense(double * ext_Hc, sparsed_CI_vec * c, ci_int n_s);
        
        int H_mult_sparsed_to_sparsed(sparsed_CI_vec * ext_Hc, sparsed_CI_vec * c, ci_int n_s);
        
        int H_mult_A(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th);
        
        int H_mult_B(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th);
        
        int H_mult_AA(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th);
        
        int H_mult_BB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th);
        
        int H_mult_AB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                          double * __restrict__ ci_I,
                          int i_th, int n_th);
        
        int H_mult_AAA(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th);
        
        int H_mult_BBB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                       double * __restrict__ ci_I,
                       int i_th, int n_th);
        
        int H_mult_AAB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                          double * __restrict__ ci_I,
                          int i_th, int n_th);
        
        int H_mult_ABB(double * __restrict__ ci_O, int n0, int n_s, int ld,
                          double * __restrict__ ci_I,
                          int i_th, int n_th);
        
        int G_AA(double * __restrict__ G);
        int G_AB(double * __restrict__ G);
        int G_BB(double * __restrict__ G);
        
        
        
        int G_calc(double * G);
        
        int H_diag_calc();
        int H_diag_calc_PT();
        
        int H_calc_PT2(double * H, int n_s);
        
        
        int calc_spin_matr(double * O, double * C,int n_s, int ld);
        int calc_L2_matr  (double * O, int i_s   ,int n_s, int ld);
        int calc_P_matr  (double * O, int i_s   ,int n_s, int ld);
        int print_states(int a, int n_s, int print);
        
        int gen_bf(sparsed_CI_vec * c, int n_s);
        int symmetrization(int i_set);
        
        int printf_occ_a(int i_CI);
        int printf_occ_b(int i_CI);
        int printf_occ_sum(int i_CI, int j_CI);
        
        ~aldet_data();
    private:
};

int aldet_copy(aldet_data * O, aldet_data * I); 

int ci_from_ci(const int& N, const int& na, const int& nb, aldet_data * CI, int s, int f, const int& i_state, double *& ci);

int ci_ext(aldet_data * O, int dir, aldet_data * I, int i_set);

int ci_ext_2(aldet_data * O, int dir, aldet_data * I, int i_set);

int ci_ext_AB(aldet_data * O, int dir, aldet_data * I, int i_set);

int F_IPEA(double * eps, double * U, double * Um, double * Up, int dir, aldet_data * I, int i_set,std::vector<double> avecoe);

int F_IPEA_double(double * eps, double * U, double * Um, double * Up, int dir, int spin_eq, aldet_data * I, int i_set,std::vector<double> avecoe);

class ci_pr_aldet;

class ci_pr_aldet{
    
public:
    
    aldet_data A;
    aldet_data B;
    int n_tr;
    int n_A;
    int n_B;
    int order;
    
    ci_key key;
    
//     ci_pr_aldet();
    int add_A(aldet_data * I, int n_si, int n_so);
    int clear();
};

// #ifdef GOOGLE_D_M
// using ci_ind_map = google::dense_hash_map< ci_key,long>;
// #endif
// 
// #ifdef TSL_R_M
// using ci_ind_map = tsl::robin_map< ci_key,long>;
// #endif
// 
// #ifdef STD_U_M
// using ci_ind_map = std::unordered_map< ci_key,long>;
// #endif
// 
using ci_pr_vec_aldet = std::vector<ci_pr_aldet>;
// 
// int ci_subspace_dec(ci_pr_vec * O, ci_map * I, int n_el, int min, int max);
// 
int ci_pr_aldet_first_order(ci_pr_vec_aldet * O, aldet_data * I, int n_orb, double ** S, int nA, int nB, int n_st);
// 
// int print_ci_map_arr_el(ci_map_arr * ci, int n,int el);


int aldet_S_calc(double *O, aldet_data * A, int a, aldet_data * B, int b);

int aldet_DMA_calc(double *O, aldet_data * A, int a, aldet_data * B, int b, int ld);

int aldet_DMB_calc(double *O, aldet_data * A, int a, aldet_data * B, int b, int ld);

int aldet_calc_CI_2el_A0AA(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f);

int aldet_calc_CI_2el_AAA0(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f);

int aldet_calc_CI_2el_AB0B(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2,int f);

int aldet_calc_CI_2el_0BAB(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2,int f);

int aldet_calc_CI_2el_0A0B(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f);

int aldet_calc_CI_2el_00AA(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f1, int f2);

int aldet_calc_CI_2el_AA00(double * O, aldet_data * I1, int a, aldet_data * I2, int b, double * V, int n_i, int n_f, int ld, int s1, int s2, int f1, int f2);

int aldet_gen_1_el_vec_arr(double * O,aldet_data * I1, int a, int nA, aldet_data * I2, int b, int nB, int n_i, int n_f, int ld, int f);

int aldet_gen_1_el_vec_m_arr(double * O,aldet_data * I1, int a, int nA, aldet_data * I2, int b, int nB, int n_i, int n_f, int ld, int f);

int aldet_calc_2body_AA(double * P, int n_s, int ld,
                          double * ci,
                          int N,int na, int Na, int Nb, int * fa, int * vec_a,
                          double * act_INTS,
                          int i_th, int n_th);

int aldet_calc_2body_AB(double * P, int n_s, int ld,
                        double * ci,
                        int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                        int * Ia, int * Ib, int * Ia_friends, int * Ib_friends,
                        double * act_INTS,
                        int i_th, int n_th);

int aldet_calc_1body(double * P, int n_s, int ld,
                     double * ci,
                     int N,int na, int Na, int Nb, int * fa, int * vec_a,
                     double * H,
                     int i_th, int n_th);

int aldet_H_mult_1body(double * ci_O, int n_s, int ld,
                       double * ci_I,
                       int N,int na, int Na, int Nb, int * fa, int * vec_a,
                       double * H,
                       int i_th, int n_th, int is_b);

int aldet_H_mult_2body_AA(double * ci_O, int n_s, int ld,
                          double * ci_I,
                          int N,int na, int Na, int Nb, int * fa, int * vec_a,
                          double * act_INTS,
                          int i_th, int n_th, int is_b);

int aldet_H_mult_2body_AB(double * ci_O, int n_s, int ld,
                          double * ci_I,
                          int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                          int * Ia, int * Ib, int * Ia_friends, int * Ib_friends,
                          double * act_INTS,
                          int i_th, int n_th);

int aldet_calc_DM_2body_AA(double * G, int n_s, int ld,
                           double * ci,
                           int N,int na, int Na, int Nb, int * fa, int * vec_a,
                           int i_th, int n_th);

int aldet_calc_DM_2body_AA_diag(double * G, int n_s, int ld,
                                double * ci,
                                int N,int na, int Na, int Nb, int * fa, int * vec_a,
                                int i_th, int n_th);

int aldet_calc_DM_2body_AB(double * G, int n_s, int ld,
                           double * ci,
                           int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                           int i_th, int n_th);

int aldet_calc_DM_2body_AB_diag(double * G, int n_s, int ld,
                                double * ci,
                                int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                                int i_th, int n_th);

int aldet_calc_DM_3body_AAA(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int Na, int Nb, int * fa, int * vec_a,
                            int i_th, int n_th);

int aldet_calc_DM_3body_AAA_1(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int Na, int Nb, int * fa, int * vec_a,
                            int i_th, int n_th);

int aldet_calc_DM_3body_AAA_2(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int Na, int Nb, int * fa, int * vec_a,
                            int i_th, int n_th);


int aldet_calc_DM_3body_AAB(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                            int i_th, int n_th);


int aldet_calc_DM_3body_AAB_2(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                            int i_th, int n_th);

int aldet_calc_DM_3body_AAB_3(double * G, int n_s, int ld,
                            double * ci,
                            int N,int na, int nb, int Na, int Nb, int * fa, int * fb, int * vec_a, int * vec_b,
                            int i_th, int n_th);


int ci_ext(aldet_data * O, int dir, aldet_data * I, int i_set);

int ci_ext_B(aldet_data * O, int dir, aldet_data * I, int i_set);

int ci_rotate_Pi_pair(aldet_data * CI, int i_set, double sin_for_phi, double cos_for_phi, int number_of_Pi_pair, int * ind_pi_states);

#endif

 
