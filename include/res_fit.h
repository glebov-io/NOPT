#ifndef RES_FIT_H
#define RES_FIT_H

# include "RI.h"
# include "aldet.h"
# include <vector>

class res_fit_data;

// Replacement lists of the strings of one spin (CSR over the strings): for every string the strings E_tu E_vw (two electrons, t < u removed,
// v < w added) or E_xy (one electron) lead to, with their sign and the orbital part of the table index, in the order the contraction loops of
// res_fit.cpp enumerate them.
struct rf_rep_list{
    std::vector<long> off;      // entries of string i: off[i] .. off[i+1]-1
    std::vector<int> ext;       // the new string
    std::vector<int> orb;       // ((t N + u) N + v) N + w for the doubles, x N + y for the singles
    std::vector<signed char> sgn;
};

class res_fit_data 
{
    public:
        
        //external parameters
        aldet_data * ci; 
        double * E0; 
        int n_s; 
        RI_data * RI; 
        double * e_c; 
        double * e_a; 
        double * e_v;
        int n_c; 
        int n_a;
        int n_v;
        double * H_AV;
        double * H_CA;
        double * H_CV;
        double edshift;
        int N_fit;
//         int n_fit_pol;
        
        
        //grid data
        double * E_fit   ;
        int    * S_fit   ;
        int    * L_fit   ;
        double * C_fit   ;
        int    * S_fit_tr;
        int    * L_fit_tr;
        double * C_fit_tr; 
        
        // RF operator tables
        double * RF_PV   ;
        double * RF_PH   ;

        double * RF_PS;
        double * RF_PV_JK;
        double * RF_PV_AB;
        
        double * RF_P3_JK;
        double * RF_P3_AB;
        
        // The three-body tables in windows of the energy grid: RF_P3_JK/AB hold the grid points win_f0 .. win_f0 + win_nf - 1 (stride win_nf).
        // full_tables (set before gen_grid): one window over the whole grid, kept after E2_calc (the analytic gradient and the self-tests read the
        // tables); otherwise the windows are built, contracted and dropped one after the other, NOPT_XMC_WINDOW_MB bounding their memory.
        int full_tables = 1;
        int win_f0 = 0, win_nf = 0;
        // strings ordered so that the start of the interpolation stencil grows along them for every other string and state (E_act = Ea + Eb)
        std::vector<int> order_b, order_a;
        // replacement lists for the alpha-beta three-body terms: alpha doubles / beta singles (AAB) and beta doubles / alpha singles (ABB)
        rf_rep_list dbl_a, sgl_b, dbl_b, sgl_a;
        int build_rep_lists();
        // the windowed part of E2_calc: AV, CA and the three-body contraction for every window
        int E2_windows(double ** E2_th, bool av, bool ca);
        // grid points per window for the memory bound NOPT_XMC_WINDOW_MB (default 4096 MB for the five window-sized arrays)
        int window_points() const;
        // NOPT_XMC_BLOCKS (test switch): is this PT class used
        static bool block_used(const char * b);
        // the three-body tables at the grid points f0 .. f0 + nw - 1 into JK, AB (n_a^6 x nw each), as E2_calc builds them; the contributions of AV and CA
        // to the four-index tables, which E2_calc already added, go to scratch copies (the analytic gradient rebuilds the windows it needs)
        int build_P3_window(int f0, int nw, double * JK, double * AB);
        
        
        //functions
        int set_par(aldet_data * ext_ci, double * ext_E0, int ext_n_s, RI_data * ext_RI, 
                    double * ext_eps, int ext_n_c, int ext_n_a, int ext_n_v, 
                    double * ext_H_AV, double * ext_H_CA, double * ext_H_CV,
                    double ext_edshift);
        
        int gen_grid(int ext_N_fit, int ext_n_fit_pol);
        
        //1-el properties table generators
        int calc_RF_1_AV(double * P_AV);
        int calc_RF_1_CA(double * P_CA);
        int calc_RF_1_CV(double * P_CV);

        //energy table generators
        int calc_RF_2_CV();        
        int calc_RF_2_AV();        
        int calc_RF_2_CA();
        int calc_RF_2_CCVV();
        int calc_RF_2_CAVV();
        int calc_RF_2_AAVV();
        int calc_RF_2_CCAV();
        int calc_RF_2_CCAA();
        
        //N-body operators
        int calc_0body(double * E2, int i_th, int n_th);
        int calc_1body_A(double * E2, int i_th, int n_th);
        int calc_1body_B(double * E2, int i_th, int n_th);
        int calc_2body_AA(double * E2, int i_th, int n_th);
        int calc_2body_AB(double * E2, int i_th, int n_th);
        int calc_2body_BB(double * E2, int i_th, int n_th);
        int calc_3body_AAA(double * E2, int i_th, int n_th);
        int calc_3body_AAB(double * E2, int i_th, int n_th);
        int calc_3body_ABB(double * E2, int i_th, int n_th);
        int calc_3body_BBB(double * E2, int i_th, int n_th);
        
        
        //main functions
        int P1_calc(double * P1, double * P_AV, double * P_CA, double * P_CV, int AV, int CV, int CA);
        
        int E2_calc(double * E2);
        
        ~res_fit_data();
       
       
};




#endif
