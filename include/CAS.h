#ifndef __cas
#define __cas

class molecule;

# include "inp_par_read.h"
# include "davidson.h"
# include "casci_solver.h"
# include <memory>

class CAS_engine;
class orbital_localizer;   // active-space localizer

class CAS_engine{
    public:
        // active space size
        int n_ao  ;
        int n_act ;
        int n_core;
        int n_vac ;
        int rotate_orbs;
        
        int * rep_num;
        
        //number of states
        int n_s;
        //optimized states
        int n_s_opt;
        std::vector<int> i_s_opt;
        
        //for CI
        int n_CI;
        int ci_solver;                            // CISOLVER_ALDET / CISOLVER_DMRG (cas is not kept)
        casci_solver * CI;                        // drives any CI backend (aldet, DMRG, …)
        std::unique_ptr<casci_solver> CI_owner;   // owns the concrete backend instance
        dav_par dav;

        // active-space localization (DMRG, localize=pm)
        std::unique_ptr<orbital_localizer> localizer_;
        std::vector<double> U_loc;                // n_act^2, [a*n_act+p]; C_loc = ACT_CVEC * U_loc

        // Active-basis rotation the CI backend's RDMs still need: the canonicalization of a backend
        // that cannot rotate its own wavefunction. gamma' = U gamma U^T, U[p*n_act+i] (p canonical,
        // i solve). Empty means nothing is pending; a solve returns RDMs in the current basis.
        // Cleared at the end of the run: post-SCF modules re-solve in the canonical basis.
        std::vector<double> U_pending;

        // warm-start (localization-rotation MPS reuse) state
        int  warm_start_cfg = 0;                  // cached cas->dmrg.warm_start (off|on)
        int  warm_after_cfg = 0;                  // cached cas->dmrg.warm_start_after (freeze gate)
        int  warm_ci_calls  = 0;                  // solve invocations so far (freeze/warm gate)
        bool warm_frozen    = false;              // has the localized frame been pinned?
        std::vector<double> U_loc_frozen;         // frozen localization (n_act^2)
        std::vector<double> C_loc_prev;           // previous frame's active orbitals, localized or
                                                  // canonical (n_ao x n_act)

        // Exit code of the SCF loop, as the line printed after it reads: 0 not converged,
        // 1 energy, 2 gradient, 3 step. -1 = no SCF run.
        int scf_exit = -1;

        molecule * M;
        
        //H_calc data
        double E_core;
        double * F_core_AO;
        double * F_act;
        double * act_INTS;
        
        //for linear
        double * Lambda_act;
        
        //state averaging
        std::vector<int> non_zero_w;
        std::vector<double> wstate;
        std::vector<double> S_rep;
        std::vector<int> Lambda_rep;
        std::vector<int> P_rep;
        std::vector<std::vector<double>> wstate_rep;
        std::vector<double> wstate_actual;
        
        //tracking
        int track;
        double * S_track;
        std::vector<int> w_num;
        std::vector<int> w_renum;
        std::vector<double> wstate_init;
        
        //molecular orbitals
        double * MO_VEC;  // all orbitals written as rows
        double * MO_BUF;
        double * GEN_CVEC; // all orbitals written as columns (copy from MO_VEC)
        double * ACT_CVEC; // active orbitals written as columns (copy from MO_VEC)
        
        //Hamiltonian
        double * H;
        
        double * F_core_MO;
        double * F_tot;//F_act+F_core
        double * J;
        double * K;
        
        //density matrices
        double * DM_C;//AO-basis
        double * DM_A;//AO-basis
        double * gamma;
        double * GAMMA;
        
        //2el tensor
        double * aaag_ints;//(tu|vp)?
        double * aaaa_ints;//(tu|vw)?
        double * aa_J_ints;//(pp|tu)
        double * aa_K_ints;//(pt|pu)
        double * cv_J_ints;//(aa|ii)
        double * cv_K_ints;//(ai|ai)
        
        //grad and hess
        double * G_ga;
        double * G;
        double * B;
        
        //properites
        int n_prop;
        double ** Prop_AO; //external
        double * Prop_Act;
        double * Prop_Core;
        double * Prop_value;
        double * Prop_nuc;
        
        
        
        CAS_engine();
        int init(cas_par * cas, molecule * ext_M);
        int SCF_alloc();
        int tensors_recalc(int n);
        int update_ACT_CVEC();
        int rotate_pending_gamma(double * g, int n_blocks);
        int rotate_pending_GAMMA(double * G);
        int CI_calc(int primary, int create_track_data,int read);
        int calc_DM_C();
        
        int AO_to_MO(double *M_out, double *M_in);
        int calc_gamma();
        double calc_E(double *g, double *G);
        int calc_F(double * F, double *g);
        int calc_G(double * G_ga_state, double * gamma_state, double *GAMMA_state);
        int calc_grad(double * G_state, double * F_state, double * G_ga_state);
        int scale_grad(double * G_state, double * gamma_state);
        int calc_hess(double * B_state, double * gamma_state, double * F_state, double * G_ga_state);
        
        int make_canonical(bool with_active);
        int av_DM_and_F_calc();
        double SA_grad_hess_calc();
        double SM_grad_hess_calc();
        
//         int F_vac();
        int Prop_calc();
        int Prop_calc_with_num(int n_calc_prop);
        int TrC_calc(int a, int b);
        int TrDM(int a, int b);
        int print_properties(const char* name);
        int rotate();
        int print_av_table(const char* type_of_calc);
        int print_av_table_with_prop(const char* type_of_calc, int num_prop, double ** P, char ** names);
        ~CAS_engine();
        
        int counter_first_sigma;
        int counter_for_number_of_Pi_states;
        int * ind_pi_states;
    
    
};

int average_DM(double * G, std::vector<double> avecoe,int na_p, int n_s);

// int CAS_CI(molecule * A, molecule * EF, int have_efrag); 

int CAS_SCF(molecule * A, cas_par * cas, char * job_name);


#endif
