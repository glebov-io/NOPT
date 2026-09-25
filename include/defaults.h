#define DEFAULT_JOB_NAME "NOPT_JOB"

#define BUF_LINE_LENGTH 1000

#define RHF_MAX_IT_DEFAULT 50
#define RHF_EN_CON_DEFAULT 1e-10
#define RHF_GR_CON_DEFAULT 1e-7

#define PRINT_DIPOLE 1
#define PRINT_DISP 0
#define PRINT_QUADRUPOLE 0
#define PRINT_MULLIKEN 0


#define CAS_MAX_IT_DEFAULT 50
#define CAS_EN_CON_DEFAULT 1e-10
#define CAS_GR_CON_DEFAULT 1e-7
#define CAS_STEP_CON_DEFAULT 1e-7
#define CAS_X_MAX_DEFAULT 0.1
// Converger restart floor: fraction of the CI solve's truncation energy an energy rise must exceed;
// 3x the largest solve-to-solve scatter measured against it (Cr2 (12,28) DMRG m=500, cold restarts)
#define CAS_RESET_TRUNC_FRAC 0.05
#define CAS_PRINT_NUMBER_DEFAULT 10
#define CAS_METHOD_DEFAULT 1

#define DAV_MAX_IT_DEFAULT 250
#define DAV_EN_CON_DEFAULT 1e-10
#define DAV_R_CON_DEFAULT 5e-8
#define DAV_N_STATES_DEFAULT 10
#define DAV_N_STATES_MULT_DEFAULT 3
#define DAV_N_BF_DEFAULT 300
#define DAV_SE_MIN_DEFAULT 1e-8
#define DAV_ED_SHIFT_DEFAULT 0.03

#define XMC_N_FIT 300
#define XMC_N_FIT_POL 3

#define DMRG_M_DEFAULT 0            // bond dimension: 0 = unset (required when CISOLVER=dmrg)
#define DMRG_SWEEPS_DEFAULT 40
#define DMRG_SWEEP_TOL_DEFAULT 1e-8
#define DMRG_SAVE_DIR_DEFAULT "/dev/shm"   // block2 scratch root (RAM-backed by default)
#define DMRG_MEMORY_DEFAULT 1.0            // block2 double-stack size, GB
#define DMRG_RDM_PASSES_DEFAULT 1          // RDM sweeps the operator set is split over
#define DMRG_MAIN_STACK_DEFAULT 0.0        // main double stack, GB; 0 = the ratio below
#define DMRG_DMAIN_RATIO_DEFAULT 0.2       // main share of the double stacks
#define DMRG_FP_CODEC_CUTOFF 1e-16         // partition-file compression precision (pyblock2 driver values)
#define DMRG_FP_CODEC_CHUNK 1024           // partition-file compression chunk length, in elements
#define DMRG_WARM_START_DEFAULT 1          // MPS warm-start across macro-iterations: on by default
#define DMRG_WARM_SWEEPS_DEFAULT 0         // max sweeps for the warm re-solve; 0 = auto (sweeps/2)
#define DMRG_WARM_NOISE_SCALE_DEFAULT 0.0  // warm-schedule noise as a multiple of the last solve's discarded weight (0 = off)
#define DMRG_WARM_NOISE_MIN 1e-10          // scaled noise below this: the warm re-solve stays noise-free
#define DMRG_ORD_DRIFT_TOL 1.005           // lattice counts as optimal below this ordering-cost drift
#define DMRG_ROT_M_DEFAULT 0               // MPS-rotation time-evolution bond dim (0 = use m)
#define DMRG_ROT_STEPS_DEFAULT 1           // MPS-rotation TE steps (dt = 1/rot_steps; total time 1)
#define DMRG_WARM_START_AFTER_DEFAULT 0    // CI solves run cold before the localized frame is frozen
#define DMRG_WARM_ROTATE_DEFAULT 1         // rotate the reused MPS into the new basis: on by default (off = reuse-only)
#define DMRG_PRINT_DETS_DEFAULT 1          // report leading determinants after convergence: on by default
#define DMRG_DET_ROT_M_DEFAULT 0           // bond dim for the localized->canonical read-out rotation (0 = auto: min(2m,1500))
#define DMRG_DET_ROT_STEPS_DEFAULT 10      // TE steps for the read-out rotation (larger than warm-start's: this rotation is big)
#define DMRG_EXTRACT_M_DEFAULT 0           // bond dim the canonical MPS is compressed to before determinant extraction (0 = none)
#define DMRG_EXTRACT_CUTOFF_DEFAULT 1e-3   // determinant magnitude cutoff for the TRIE extraction search
#define DMRG_H2CAA_M_DEFAULT 0             // compressed-intermediate bond dim for the DSRG h2caa overlap (0 = auto: 2m)
#define DMRG_LOW_M_OPT_DEFAULT 2           // MPO simplification rule: 0 = off, 1 = on, 2 = auto (threshold below)
// Auto threshold on K^2*m^2 (K active orbitals, m bond dimension): above it the transpose-lean MPO
// rule is kept, the explicit AD/full-B store growing as K^2*m^2 (boundary near K=30, m=3000).
#define DMRG_LOW_M_OPT_KK_MM_MAX 8.0e9

#define DSRG_S_DEFAULT 0.5                 // DSRG flow parameter
#define DSRG_ROOT_DEFAULT 0                // state-specific root (unrelaxed)
#define DSRG_PRINT_DEFAULT 1               // print verbosity (>=2 adds per-class rows)
#define DSRG_SA_DEFAULT 0                  // state-averaged ensemble reference (0 = state-specific root)
#define DSRG_RELAX_DEFAULT DSRG_RELAX_NONE // reference relaxation level (none = unrelaxed)

#define AVAS_REF_BASIS_DEFAULT "cc-pvtz-minao"   // $AVAS reference minimal basis (H-Kr)
