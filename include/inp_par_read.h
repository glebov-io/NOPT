#ifndef IPR_H
#define IPR_H

# include <vector>
# include <string>

class backup_par // read/write/calc
{
    public:
        int h1;// 0 = read, 1 = calc, 2 = calc and write
//         int h2;
//         int ri;
        
        char * prefix;
        
        backup_par();
        int read_group(char * inp); 
        int disable(); 
//         int read_line(char * inp); 
        ~backup_par();
};

class rhf_par
{
    public:
        int y;
        //guess
        int guess;     // guess_kind: HUCKEL (default) | SAD
//         int huckel_guess;
//         int h_core_guess;
//         int read_guess;
        //convergence
        int max_it;
        double e_conv;
        double g_conv;
        
        rhf_par();
        int read_group(char * inp); 
        int read_line(char * inp); 
        int write_info(int n_cor);
        ~rhf_par();
    
};

class dav_par
{
    public:
        int n_s;
        int n_bf;
        int max_it;
        double e_conv;
        double r_conv;
        double se_min;
        double edshift;
        int sparsed_Hc;

        
        
        dav_par();
        int read_group(char * inp); 
        int read_line(char * inp); 
        int write_info();
        ~dav_par();
    
};


// Starting orbitals when the input carries no $VEC group.
enum guess_kind { GUESS_HUCKEL = 0, GUESS_SAD = 1 };

// CI backend driving the CAS-SCF active-space solve.
enum cisolver_kind { CISOLVER_ALDET = 0, CISOLVER_DMRG = 1 };
enum converger_kind { CONVERGER_SOSCF = 0, CONVERGER_SXPT = 1 };

// $DMRG group — value sets validated against keyword lists (dmrg_par::read_line).
enum dmrg_hf_occ_kind   { DMRG_HF_OCC_UNKNOWN = -1, DMRG_HF_OCC_INTEGRAL = 0 };
enum dmrg_schedule_kind { DMRG_SCHED_UNKNOWN  = -1, DMRG_SCHED_DEFAULT   = 0 };
enum dmrg_localize_kind { DMRG_LOC_UNKNOWN = -1, DMRG_LOC_OFF = 0, DMRG_LOC_PM = 1, DMRG_LOC_BOYS = 2 };
enum dmrg_locorder_kind { DMRG_LOCORDER_UNKNOWN = -1, DMRG_LOCORDER_FIEDLER = 0, DMRG_LOCORDER_GAOPT = 1, DMRG_LOCORDER_NONE = 2 };
enum dmrg_warm_kind     { DMRG_WARM_UNKNOWN = -1, DMRG_WARM_OFF = 0, DMRG_WARM_ON = 1 };
enum dmrg_lowm_kind     { DMRG_LOW_M_UNKNOWN = -1, DMRG_LOW_M_OFF = 0, DMRG_LOW_M_ON = 1, DMRG_LOW_M_AUTO = 2 };

// $DSRG group — CCVV source dressing (dsrg_par::read_line).
enum dsrg_ccvv_src_kind { DSRG_CCVV_SRC_UNKNOWN = -1, DSRG_CCVV_SRC_NORMAL = 0, DSRG_CCVV_SRC_ZERO = 1 };
// $DSRG group — reference relaxation level (dsrg_par::read_line).
enum dsrg_relax_kind { DSRG_RELAX_UNKNOWN = -1, DSRG_RELAX_NONE = 0, DSRG_RELAX_ONCE = 1 };

// $CDAS group — CDAS flow the run takes (cdas_par::read_line).
enum cdas_mode_kind { CDAS_MODE_UNKNOWN = -1, CDAS_MODE_NATIVE = 0, CDAS_MODE_TRUNC_GNO = 1, CDAS_MODE_DELTA_GNO = 2 };

class dmrg_par // settings for the DMRG (block2) CI backend; see $DMRG group
{
    public:
        int    m;          // bond dimension (required, > 0)
        int    sweeps;     // maximum number of DMRG sweeps
        double sweep_tol;  // sweep energy convergence tolerance
        int    hf_occ;     // initial occupancy scheme (dmrg_hf_occ_kind)
        int    schedule;   // sweep schedule (dmrg_schedule_kind)
        int    localize;       // active-space localization (dmrg_localize_kind): off | pm | boys
        int    dump_loc_orbs;  // dump localized orbitals (GAMESS .out) at iteration 0, then continue
        int    loc_order;      // DMRG orbital ordering (dmrg_locorder_kind): fiedler | gaopt | none
        std::string save_dir;  // block2 scratch root (renormalized ops / MPS)
        double memory;         // block2 double-stack size, GB (> 0)
        double main_stack;     // main double stack, GB (0 = the default share of memory)
        int    rdm_passes;     // RDM sweeps the operator set is split over (>= 1)
        int    warm_start;       // MPS warm-start across macro-iterations (dmrg_warm_kind): off | on
        int    warm_sweeps;      // clean-sweep budget of the warm re-solve; 0 = auto (sweeps/2); +2 noisy sweeps when warm_noise_scale > 0
        double warm_noise_scale; // warm-schedule noise as a multiple of the last solve's discarded weight
        int    rot_m;            // MPS-rotation time-evolution bond dim (0 = use m)
        int    rot_steps;        // MPS-rotation TE steps (dt = 1/rot_steps; total time 1)
        int    warm_start_after; // cold macro-iterations before freezing the localized frame
        int    warm_rotate;      // rotate the reused MPS into the new basis (dmrg_warm_kind): off = reuse-only | on
        int    print_dets;       // report leading determinants after convergence (dmrg_warm_kind): off | on
        int    det_rot_m;        // bond dim for the localized->canonical read-out rotation (0 = auto: min(2m,1500))
        int    det_rot_steps;    // TE steps for the read-out rotation
        int    extract_m;        // bond dim the canonical MPS is compressed to before extraction (0 = none)
        double extract_cutoff;   // determinant magnitude cutoff for the extraction search
        int    h2caa_m;          // compressed-intermediate bond dim for the DSRG h2caa overlap (0 = auto: 2m)
        int    low_m_opt;        // MPO simplification rule (dmrg_lowm_kind): on = store AD/full B explicitly
                                 //   (faster solve, ~+40% operator stack) | off = transpose-lean | auto by K^2*m^2

        dmrg_par();
        int read_group(char * inp);
        int read_line(char * inp);
        int validate();        // enforces the value checks; exits loudly on a bad value
        int write_info();
        ~dmrg_par();

};

// $AVAS group -- atoms= and shells= are mandatory (avas_par::validate).
class avas_par
{
    public:
        int y;
        std::vector<int> atoms;    // 1-based indices of the atoms carrying the target shells
        std::vector<int> shell_n;  // principal number of each target nl shell
        std::vector<int> shell_l;  // angular momentum of each target nl shell
        std::vector<int> shell_atom; // 1-based atom each target label is bound to, 0 = every atom in atoms=
        std::string ref_basis;     // reference minimal basis the target shells are taken from

        avas_par();
        int read_group(char * inp);
        int read_line(char * inp);
        int validate();            // enforces the mandatory keywords; exits loudly
        int write_info() const;
        ~avas_par();

};

class cas_par
{
    public:
        int y;
        //CI backend
        int ci_solver;     // cisolver_kind: ALDET (default) | DMRG
        //orbital converger
        int converger;     // converger_kind: SOSCF (default) | SXPT
        int lbfgs;         // orbital L-BFGS history depth, 0 = bare step
        //convergence
        int max_it;
        double e_conv;
        double g_conv;
        double s_conv;
        
        double x_max;
        bool x_max_set;   // user gave x_max, so the converger default does not apply
        bool lbfgs_set;   // user gave lbfgs, which only converger=sxpt reads
        
        int method;
        int SA;
        int SM;
        int DA;
        
        //states
        int n_s;
        
        std::vector<double> w_state;
        int w_state_type;// 0 - read, 1 - all eq to 1, 2 - separated by symmetry ir.rep.
        std::vector<std::vector<double>> w_state_by_rep;
        std::vector<int> rep_num;
        std::vector<int> rep_lambda;
        std::vector<int> rep_sign;
        std::vector<double> rep_spin;
        
        //tracking
        int track;
        
        //rotate orbitals before cas_par
        int rotate_orbs;
        
        //davidson
        dav_par dav;
        //dmrg (when ci_solver == CISOLVER_DMRG)
        dmrg_par dmrg;

        cas_par();
        int read_group(char * inp); 
        int read_line(char * inp); 
        int w_linear_read_line(char * line);
        int w_state_by_rep_read(char * inp);
        int write_info(int n_a, int n_b, int n_o, int n_c, int mult);
        ~cas_par();
    
};

class cis_par
{
    public:
        int y;
        //convergence
        
        //states
        int n_s;
	int n_f;
	int method;
	int nat_orb;
	
        std::vector<double> w_state;
        
        //tracking
        int track;
        
        //davidson
        dav_par dav;
        
        cis_par();
        int read_group(char * inp);
        int read_line(char * inp);
        int write_info();
        ~cis_par();

};

class mp2_par
{
    public:
        int y;
        int n_f;
        int nat_orb;

        mp2_par();
        int read_group(char * inp);
        int read_line(char * inp);
        int write_info();
        ~mp2_par();

};

class xmc_par
{
    public:
        int y;
//         int n_s;
        double edshift;
//         std::vector<double> avecoe;
        cas_par * cas;
        int have_ifitd;
        std::vector<double> ifitd_energy;
        int d_only;
        char* gamess_file_name;
        int n_fit;
        int n_fit_pol;
        int pt1_d;
        
        xmc_par();
        int read_group(char * inp, cas_par * ext_cas); 
        int read_line(char * inp); 
        int write_info(int n_a, int n_b, int n_o, int mult);
//         int write_ci_info(int n_a, int n_b, int n_o, int mult);
        ~xmc_par();
        
    
};

class gno_par             // settings of the GNO modes (trunc_gno, delta_gno); read in the $CDAS group
{
    public:
        int mode;          // cdas_mode_kind, set by $CDAS cdas_mode=
        int skip_scalar;   // set by $CDAS skip_gno_scalar=; 1 leaves F0 out, -1 = the mode's default
        int m_delta;       // set by $CDAS m_delta=; branch bond dimension of delta_gno
        int sweeps_branch; // branch sweep budget, resolved from $DMRG sweeps
        double dav_branch; // branch Davidson threshold, resolved from $DMRG sweep_tol

        gno_par();
        bool on() const;       // true in the two GNO modes; exits on a mode it does not know
        int write_info();
        ~gno_par();

};

class cdas_par
{
    public:
        int y;
//         int n_s;
        double edshift;
//         std::vector<double> avecoe;
        cas_par * cas;
        int have_eps;
        std::vector<double> eps;
        char* gamess_file_name;
//         int mult;
        
        int IPEA;
        int MPPT;
        int HOMO;
        int actual;
        int sing_e;
        int mult_e;
        int orb_e;
        int fit_e;
        int n_orb;
        int rotate_orbs;
        int pt1_d;
        int m_cdas;        // set by $CDAS m_cdas=; PT-stage bond dimension in cdas_mode=native
        bool m_cdas_set;   // user gave m_cdas; otherwise the PT stage runs at $DMRG m

        gno_par gno;

        cdas_par();
        int read_group(char * inp, cas_par * ext_cas);
        int read_line(char * inp);
        int write_info(int n_a, int n_b, int n_o, int mult);
//         int write_ci_info(int n_a, int n_b, int n_o, int mult);
        ~cdas_par();


};

class dsrg_par            // settings for DSRG-PT2 (state-specific or SA reference); see $DSRG group
{
    public:
        int y;
        cas_par * cas;
        double s;          // flow parameter (> 0)
        int ccvv_source;   // dsrg_ccvv_src_kind: normal | zero
        int root;          // state-specific root (>= 0)
        int print;         // print verbosity
        int sa;            // state-averaged ensemble reference (0 | 1)
        int relax;         // dsrg_relax_kind: none | once

        dsrg_par();
        int read_group(char * inp, cas_par * ext_cas);
        int read_line(char * inp);
        int validate();        // enforces the value checks; exits loudly on a bad value
        int write_info(int n_a, int n_b, int n_o, int mult);
        ~dsrg_par();

};

# include "molecule.h"

class inp_par
{
    public:
        char* inp_name;
        char* job_name;
        std::vector<int> mol_line;
        std::vector<int> act_space_line;
        
        char* out_folder_name;
        int doci_dec;
        
        char* point_group;
        
        rhf_par rhf;
        avas_par avas;
        cas_par cas;
        cis_par cis;
        mp2_par mp2;
        xmc_par xmc;
        cdas_par cdas;
        dsrg_par dsrg;
        
//         int charge;
//         int parsing_done;
        
//         int non_diag=0;
        int ecp;
        int num_threads;
        int out_folder_defined;
        
        inp_par();
        int write_info();
        int read(char * inp_name);
        int read_p_group(char * inp);
        int read_p_line(char * inp); 
        std::vector<int> count_groups(char * inp, std::vector<const char *>start_kw, std::vector<const char *>stop_kw);
        int read_m_line(char * inp, int n); 
        int read_symm_group(char * inp);
        int read_symm_line(char * inp); 
        ~inp_par();
        
};

std::vector<int> molecule_read_by_inp_par(molecule * M, inp_par * P);



#endif
