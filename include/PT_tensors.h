#ifndef RES_FIT_H
#define RES_FIT_H

# include "RI.h"
# include "casci_solver.h"

class PT_tensors;

class PT_tensors 
{
    public:
        
        RI_data * integrals; 

        double * e_c;
        double * e_a;
        double * e_IP;
        double * e_EA;
        double * e_v;
        double * IP_U;
        double * EA_U;
        double * IP_Um;
        double * EA_Um;
        double * IP_Up;
        double * EA_Up;

        double * e_IP_2;
        double * e_EA_2;
        double * IP_2_U;
        double * EA_2_U;
        double * IP_2_Um;
        double * EA_2_Um;
        double * IP_2_Up;
        double * EA_2_Up;
        
        double * e_IP_AB;
        double * e_EA_AB;
        double * IP_AB_U;
        double * EA_AB_U;
        double * IP_AB_Um;
        double * EA_AB_Um;
        double * IP_AB_Up;
        double * EA_AB_Up;
        
        
        int n_c; 
        int n_a;
        int n_v;
        double * H_AV;
        double * H_CA;
        double * H_CV;
        double edshift;
        
        
        
        // RF operator tables
        double * RF_PV   ;
        double * RF_PH   ;

        double RF_PS;
        double * RF_PV_JK;
        double * RF_PV_AB;
        
        double * RF_P3_JK;
        double * RF_P3_AB;
        
        int n_active_el_alp;
        int n_active_el_bet;
        //// ADDED
        
        //functions
        
        //constructor
        PT_tensors();
        
        //initialization
        int set_par(RI_data * ext_integrals, 
                    double * ext_eps_c,
                    int ext_n_c, int ext_n_a, int ext_n_v, 
                    double * ext_H_AV, double * ext_H_CA, double * ext_H_CV,
                    double ext_edshift, molecule * A); //// ADDED - get number of alpha, beta electrons
        int IPEA(casci_solver * I, std::vector<double> avecoe);
        //the same matrices from an ensemble already in this frame, without asking the solver
        int IPEA(const double * g1, const double * g2, const double * gamma, const double * GAMMA);
        int MPPT(aldet_data * I, int i_set, std::vector<double> avecoe);
        int set_zero();
        int symm();
        
        //1-el properties table generators
        int calc_IPEA_1_AV(double * P_AV);
        int calc_IPEA_1_CA(double * P_CA);
        int calc_IPEA_1_CV(double * P_CV);

        int calc_EE_1_AV(double * P_AV);
        int calc_EE_1_CA(double * P_CA);
        int calc_EE_1_CV(double * P_CV);

        //energy table generators
        int calc_IPEA_2_CV();        
        int calc_IPEA_2_AV();        
        int calc_IPEA_2_CA();
        int calc_IPEA_2_CCVV();
        int calc_IPEA_2_CAVV();
        int calc_IPEA_2_AAVV();
        int calc_IPEA_2_CCAV();
        int calc_IPEA_2_CCAA();
        
        int calc_EE_2_CV();        
        int calc_EE_2_AV();        
        int calc_EE_2_CA();
        int calc_EE_2_CCVV();
        int calc_EE_2_CAVV();
        int calc_EE_2_AAVV();
        int calc_EE_2_CCAV();
        int calc_EE_2_CCAA();
        int calc_EE_2_VVVV();

        int calc_2_AV_bc();        
        int calc_2_CA_bc();
        int calc_2_CAVV_bc();
        int calc_2_AAVV_bc();
        int calc_2_CCAV_bc();
        int calc_2_CCAA_bc();
        
        
        //main functions
        int P1_calc_IPEA(double * P1, aldet_data * CI, double * P_AV, double * P_CA, double * P_CV, int AV, int CV, int CA);
        int P1_calc_EE  (double * P1, aldet_data * CI, double * P_AV, double * P_CA, double * P_CV, int AV, int CV, int CA);
        int E2_calc_IPEA();
        int E2_calc_EE  ();
        
        int E2_calc_bc();
        int E2_calc_orb_der(int orb);
        
        //destructor
        ~PT_tensors();
       
       
    private:
       
        //IP/EA matrices -> the eigenpairs the dressing reads; the buffers stay the caller's
        int IPEA_solve(double * IP_H, double * EA_H);
       
};

int sym_JK_T(double *RF_PV_JK, double * RF_PV,int n_a);

int sym_JK_CA_T(double *RF_PV_JK, double * RF_PV,int n_a);

int sym_AB_T(double * RF_PV_AB, double * RF_PV, int n_a);

int sym_AB_CA_T(double * RF_PV_AB, double * RF_PV, int n_a);

#endif
