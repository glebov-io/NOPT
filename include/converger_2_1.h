#ifndef __CONV_H
#define __CONV_H

# include "molecule.h"
# include "l-bfgs_2_1.h"

class soscf_engine;

class soscf_engine 
{
    public:
        l_bfgs_engine BFGS;
        double * orb_grad;
        double * orb_shift;
        double * app_orb_hess;
        double * rot_matr;
        double * alp_orb_energy;
        double grad_max_soscf;
        double grad_min_soscf;
        double x_max_soscf;
        int n_ao;
        int n_mo;
        int n_el;
        int N_LBFGS_VECTORS;
        
//         molecule * MOL;

//         int soscf_rhf();
        
        soscf_engine();
        
        ~soscf_engine();
        
        int init(int n, int n_el_inp,int n_mo_inp, int n_ao_a_inp);
        
//         int soscf_rhf_iter();
        
        int orb_grad_calc(double *, int, int);
        
        int orb_hess_calc(int n, int, double * B);
        
        double calc(double * F);
        
        int step(double * VEC, double * BUF);
//         int OHF_step(double * VEC, double * O_B, double * M, double * BUF, int n_ort);
//         int step_OHF(double * VEC, double * O_B, double * M, double * B,int n_ort);
        
//         int f_transformed_diag();
};

class soscf_engine_ROHF;

class soscf_engine_ROHF 
{
    public:
        l_bfgs_engine BFGS;
        double * orb_grad  ;
        double * orb_grad_a;
        double * orb_grad_b;
        double * orb_shift  ;
        double * orb_shift_a;
        double * orb_shift_b;
        double * app_orb_hess  ;
        double * app_orb_hess_a;
        double * app_orb_hess_b;
        double * rot_matr;
        double * alp_orb_energy;
        double grad_max_soscf;
        double grad_min_soscf;
        double x_max_soscf;
        int n_ao;
        int n_d;
        int n_a;
        int n_b;
        int N_LBFGS_VECTORS;
        
//         molecule * MOL;

//         int soscf_rhf();
        
        int init(int,int,int,int,int);
        
        int finalize();
        
//         int soscf_rhf_iter();
        int orb_grad_part_add(double * g, double *F, int n, int n_v, int n_g, double k);
        
        int orb_grad_calc(double *, double *, double *);
        
        int orb_hess_part_add(double * h, double *F, int n, int n_v, int n_h, int n_s, double k);
        
        int orb_hess_calc(double *, double *, double *);
        
        double calc(double *, double *, double *);
        
        int rot_matr_add(double * R, double *S, int n, int n_v, int n_g);
        
        int step(double * VEC, double * BUF);
        
        int OHF_step(double * VEC, double * O_B, double * M, double * BUF, int n_ort);
//         int step_OHF(double * VEC, double * O_B, double * M, double * B,int n_ort);
        
//         int f_transformed_diag();
};

class soscf_engine_UHF;

class soscf_engine_UHF 
{
    public:
        l_bfgs_engine BFGS;
        double * orb_grad_a;
        double * orb_grad_b;
        double * orb_shift_a;
        double * orb_shift_b;
        double * app_orb_hess_a;
        double * app_orb_hess_b;
        double * rot_matr_a;
        double * rot_matr_b;
        double * alp_orb_energy;
        double grad_max_soscf;
        double grad_min_soscf;
        double x_max_soscf;
        int n_ao_a;
        int n_ao_b;
        int n_el_a;
        int n_el_b;
        int N_LBFGS_VECTORS;
        
//         molecule * MOL;

//         int soscf_rhf();
        
        int init(int,int,int,int,int/*, molecule **/);
        
        int finalize();
        
//         int soscf_rhf_iter();
        
        int orb_grad_calc_a(double *);
        int orb_grad_calc_b(double *);
        
        int orb_hess_calc_a(double * B);
        int orb_hess_calc_b(double * B);
        
        double calc_a(double * F);
        double calc_b(double * F);
        
        int rot_matr_calc();
        int vec_update(double * VEC, double * BUF, char ab);
        int vec_update_OHF(double * VEC, double * O_B, double * M, double * B,int n_ort, char ab);
        
//         int f_transformed_diag();
};

class soscf_engine_MCSCF;

class soscf_engine_MCSCF 
{
    public:
        l_bfgs_engine BFGS;
        double * orb_grad;
        double * orb_shift;
        double * app_orb_hess;
        double * rot_matr;
//         double * alp_orb_energy;
        double grad_max_soscf;
        double grad_min_soscf;
        double x_max_soscf;
        int n_ao;
        int n_mo;
        int n_c;
        int n_a;
        int n_v;
        int N_LBFGS_VECTORS;
        
//         molecule * MOL;

//         int soscf_rhf();
        
        soscf_engine_MCSCF();
        
        int init(int n, int ext_n_c, int ext_n_a, int ext_n_v, int ext_n_ao, double ext_x_max_soscf);
        
        int orb_grad_calc(double *G);
        
        int orb_hess_calc(double *B);
        
        double calc(double * G, double * B);
        
        double step(double * VEC, double * BUF);
        
        void reset_history();   // drop the L-BFGS history

        ~soscf_engine_MCSCF();
        
};


#endif
 
