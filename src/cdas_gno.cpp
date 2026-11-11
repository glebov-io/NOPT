// GNO stage: the CAS-SCF ensemble inherited into the PT frame, the dressed and folded active-space
// totals built on it, the cold folded reference solve, then the truncated checkpoint, the folded
// and dressed branches taken from it, and read-out A.

# include <cstdio>
# include <cstdlib>
# include <cmath>
# include <vector>

# include "common_vars.h"
# include "inp_out.h"
# include "timer.h"
# include "blas_link.h"
# include "inp_par_read.h"
# include "CAS.h"
# include "PT_tensors.h"
# include "tensor_rotate.h"
# include "cdas_fold.h"
# include "cdas_gno.h"

namespace {

// Slot marker: the largest |S| in row i of a cross-set overlap does not sit on the diagonal, so
// that branch state is not the reference's slot i. No threshold enters and no root is relabeled.
bool row_peak_off_diagonal(int n_s, const double * S, int i){
    int jm = 0;
    for(int j=1;j<n_s;j++)
        if(std::fabs(S[(size_t)i*n_s+j]) > std::fabs(S[(size_t)i*n_s+jm]))jm = j;
    return jm != i;
}

// The dressed handle, from the totals the fold was taken of.
void import_dressed_handle(CAS_engine * CAS, const gno_operators & ops){
    CAS->CI->import_named_operator(OP_DRESSED, ops.g1_d.data(), ops.g2_d.data(),
                                   ops.g3.data(), ops.c_d);
}

} // namespace

void gno_ensemble(CAS_engine * CAS, cdas_par * cdas, int loc_realized, const double * U_loc,
                  bool want_g3, gno_ensemble_data & ens){

    std::vector<double>().swap(ens.G3);

    const int n = CAS->n_act;
    const size_t n2=(size_t)n*n, n4=n2*n2;

    if(CAS->ci_solver!=CISOLVER_DMRG){
        fprintf(out_stream,"ERROR: the GNO stage inherits its ensemble from a DMRG CAS"
                           " stage; use cisolver=dmrg\n");
        exit(EXIT_FAILURE);
    }
    casci_solver * CI = CAS->CI_owner.get();
    if(CI==nullptr){
        fprintf(out_stream,"ERROR: the GNO stage found no CAS-stage CI engine to inherit"
                           " the ensemble from\n");
        exit(EXIT_FAILURE);
    }

    const int n_s = CI->n_states();
    if((int)cdas->cas->w_state.size()<n_s){
        fprintf(out_stream,"ERROR: the GNO stage found %d state weights for %d states\n",
                (int)cdas->cas->w_state.size(), n_s);
        exit(EXIT_FAILURE);
    }
    double wsum=0.0;
    for(int i=0;i<n_s;i++)wsum += cdas->cas->w_state[i];
    if(wsum<=0.0){
        fprintf(out_stream,"ERROR: the GNO stage has zero total state weight\n");
        exit(EXIT_FAILURE);
    }
    ens.w_ens.resize(n_s);
    for(int i=0;i<n_s;i++)ens.w_ens[i] = cdas->cas->w_state[i]/wsum;
    const std::vector<double> & w_ens = ens.w_ens;

    // the last CAS solve's densities, in the basis that solve ran in: per state, and the block the
    // backend already averaged with the same weights
    std::vector<double> d1((size_t)n_s*n2, 0.0);
    CI->calc_DM_diag(d1.data(), 0);
    std::vector<double> g_solve(n2, 0.0);
    for(int i=0;i<n_s;i++)
    for(size_t k=0;k<n2;k++)
        g_solve[k] += w_ens[i]*d1[(size_t)i*n2+k];
    std::vector<double> G_solve(n4, 0.0);
    CI->G_calc(G_solve.data());

    std::vector<double> Uc;
    if(!CI->report_rotation(Uc) || Uc.size()!=n2){
        fprintf(out_stream,"ERROR: no canonicalization stored on the CAS engine; the active frame"
                           " is unknown\n");
        exit(EXIT_FAILURE);
    }

    // V[i*n+l]: the CAS solve's active basis -> the PT frame, canonicalization then localization
    std::vector<double> V(n2, 0.0);
    if(loc_realized && U_loc!=nullptr)
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n,n,n,1.0,
                    Uc.data(),n, U_loc,n, 0.0, V.data(),n);
    else
        for(int p=0;p<n;p++)
        for(int i=0;i<n;i++)
            V[(size_t)i*n+p] = Uc[(size_t)p*n+i];

    ens.gamma.assign(n2, 0.0);
    ens.GAMMA.assign(n4, 0.0);
    rotate1(g_solve.data(), V.data(), n, ens.gamma.data(), /*forward=*/true);
    rotate2(G_solve.data(), V.data(), n, ens.GAMMA.data(), true);
    printf_timer("GNO ensemble densities");

    // the ensemble three-body moment of the GNO scalar, gathered on transient per-root extracts
    if(want_g3){
        if(!CI->supports_g3_diag()){
            fprintf(out_stream,"ERROR: the GNO scalar needs the per-state 3-body moment via"
                               " G3_calc_diag, which this CI backend does not provide;"
                               " use $CDAS skip_gno_scalar=1\n");
            exit(EXIT_FAILURE);
        }
        const size_t n6 = n4*n2;
        std::vector<double> G3_acc(n6, 0.0);
        {
            std::vector<double> G3_i(n6, 0.0);
            for(int i=0;i<n_s;i++){
                CI->G3_calc_diag(G3_i.data(), i);
                cblas_daxpy((int)n6, w_ens[i], G3_i.data(),1, G3_acc.data(),1);
            }
        }
        ens.G3.assign(n6, 0.0);
        rotate3(G3_acc.data(), V.data(), n, ens.G3.data(), /*forward=*/true);
        std::vector<double>().swap(G3_acc);
        printf_timer("GNO ensemble three-body moment");
    }

    const int exit_flag = CAS->scf_exit;
    if(exit_flag==0)
        fprintf(out_stream,"NOTE: CASSCF did not converge; the ensemble is the last iteration's\n");
    fflush(out_stream);
}

gno_operators prepare_gno_operators(int n, const PT_tensors & T, const double * H_AA,
                                    const double * act_INTS, double E_core,
                                    const double * gamma, const double * GAMMA,
                                    const double * G3_ens){

    const size_t n2=(size_t)n*n, n4=n2*n2, n6=n4*n2;

    gno_operators ops;
    ops.n = n;

    ops.g1_d.assign(H_AA, H_AA+n2);
    for(size_t i=0;i<n2;i++)ops.g1_d[i] += T.RF_PH[i];

    // the dressed importer's pair permutation of the two-body table
    ops.g2_d.assign(act_INTS, act_INTS+n4);
    for(int a=0;a<n;a++)
    for(int c=0;c<n;c++)
    for(int b=0;b<n;b++)
    for(int d=0;d<n;d++)
        ops.g2_d[(((size_t)a*n+c)*n+b)*n+d] += T.RF_PV_AB[(((size_t)a*n+b)*n+c)*n+d];

    ops.g3.resize(n6);
    assemble_g3(n, T.RF_P3_AB, ops.g3.data());

    ops.c_d = E_core + T.RF_PS;

    ops.F1.assign(n2, 0.0);
    ops.F2.assign(n4, 0.0);
    build_fold(n, ops.g3.data(), gamma, GAMMA, ops.F1.data(), ops.F2.data());

    ops.F0 = G3_ens ? fold_scalar(n, ops.g3.data(), G3_ens, ops.F1.data(), ops.F2.data(),
                                  gamma, GAMMA)
                    : 0.0;

    ops.g1_f = ops.g1_d;
    for(size_t i=0;i<n2;i++)ops.g1_f[i] += ops.F1[i];
    ops.g2_f = ops.g2_d;
    for(size_t i=0;i<n4;i++)ops.g2_f[i] += ops.F2[i];
    ops.c_f = ops.c_d + ops.F0;

    return ops;
}

int cdas_gno_run(CAS_engine * CAS, const PT_tensors & T, cdas_par * cdas,
                 gno_ensemble_data & ens, const double * H_AA, const double * act_INTS,
                 double E_core){

    gno_par & fp = cdas->gno;

    const int n   = CAS->n_act;
    const int n_s = CAS->CI->n_states();
    const std::vector<double> & gamma = ens.gamma;
    const std::vector<double> & GAMMA = ens.GAMMA;

    // trunc_gno stops at the folded reference; delta_gno continues into the two branches
    bool branches;
    if     (fp.mode==CDAS_MODE_TRUNC_GNO) branches = false;
    else if(fp.mode==CDAS_MODE_DELTA_GNO) branches = true;
    else{
        fprintf(out_stream,"ERROR: the folded-reference stage runs in cdas_mode=trunc_gno or"
                           " cdas_mode=delta_gno only\n");
        exit(EXIT_FAILURE);
    }

    if(!CAS->CI->supports_operator_handles()){
        fprintf(out_stream,"ERROR: the GNO stage needs a backend with named operator handles;"
                           " use cisolver=dmrg\n");
        exit(EXIT_FAILURE);
    }
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"___________________Starting_GNO_folded_reference_stage_________________\n\n");
    fflush(out_stream);

    // ---- dressed and folded totals ----
    const bool scalar_on = !ens.G3.empty();
    gno_operators ops = prepare_gno_operators(n, T, H_AA, act_INTS, E_core,
                                              gamma.data(), GAMMA.data(),
                                              scalar_on?ens.G3.data():nullptr);
    std::vector<double>().swap(ens.G3);
    printf_timer("GNO operator preparation");

    if(scalar_on)
        fprintf(out_stream,"GNO scalar F0                    % .10f\n",ops.F0);
    else if(!branches)
        fprintf(out_stream,"NOTE: GNO scalar skipped -- energies carry a state-independent,"
                           " geometry-dependent shift; excitation energies are unaffected"
                           " (skip_gno_scalar=0 restores it)\n");

    CAS->CI->import_named_operator(OP_FOLDED, ops.g1_f.data(), ops.g2_f.data(), nullptr, ops.c_f);
    printf_timer("GNO folded operator import");

    // ---- cold folded reference solve ----
    CAS->CI->select_operator(OP_FOLDED);
    CAS->CI->solve(1,0,false);

    // sweep energies of the cold solve
    std::vector<double> eps_hi(n_s);
    for(int i=0;i<n_s;i++)eps_hi[i] = CAS->CI->E_state(i);
    const double hi_resid  = CAS->CI->last_solve_resid();
    const double hi_dw     = CAS->CI->last_solve_dw();
    const bool   hi_conv   = (hi_resid < cdas->cas->dmrg.sweep_tol);
    fprintf(out_stream,"\nFolded reference solve: residual |dE| = %.3e, discarded weight %.3e,"
                       " converged: %s\n",hi_resid,hi_dw,hi_conv?"yes":"no");
    fflush(out_stream);

    // the reference set and the state a branch starts from, both taken before any read-out
    CAS->CI->save_state_set("hi");
    CAS->CI->save_checkpoint("hi");
    printf_timer("GNO folded reference solve");

    // without F0 the absolute folded energies are shifted, so they are reported only where they
    // are the stage's result; read-out A is free of the shift either way
    if(scalar_on || !branches){
        if(!branches) fprintf(out_stream,"\nCDAS-PT2 (trunc_gno) Energy summary:\n");
        else          fprintf(out_stream,"\nFolded reference energy summary:\n");
        PrintEnergy(eps_hi.data(), n_s, 0);
    }
    fflush(out_stream);

    // ---- truncated checkpoint and the two branches ----
    if(branches){
        // the branches start from the reference as it was stored
        CAS->CI->load_checkpoint("hi");
        import_dressed_handle(CAS, ops);

        // checkpoint t: the compression both branches start from
        CAS->CI->solve_branch(OP_FOLDED, fp.m_delta, 1, fp.dav_branch, false);
        const double dw_entry_t = CAS->CI->last_entry_dw();
        const int    m_t_actual = CAS->CI->last_max_bond_dim();
        CAS->CI->save_checkpoint("t");
        fprintf(out_stream,"\nCheckpoint (m=%d, actual %d): discarded weight %.3e\n",
                fp.m_delta, m_t_actual, dw_entry_t);
        fflush(out_stream);
        printf_timer("GNO checkpoint t");

        // folded branch f, continued on the live state t
        const int sweeps_two_site_f = CAS->CI->solve_branch(OP_FOLDED, fp.m_delta, fp.sweeps_branch,
                                                            fp.dav_branch, true);
        const int    sweeps_tail_f = CAS->CI->last_tail_sweeps();
        const double dw_two_site_f = CAS->CI->last_solve_dw();
        const double dw_tail_f     = CAS->CI->last_tail_dw();
        const bool   converged_f   = CAS->CI->last_solve_converged();
        CAS->CI->save_state_set("f");
        // the discarded weight of the phase the branch closed on
        fprintf(out_stream,"Folded branch (m=%d): %d sweeps, discarded weight %.3e, converged: %s\n",
                fp.m_delta, sweeps_two_site_f+sweeps_tail_f,
                sweeps_tail_f>0?dw_tail_f:dw_two_site_f, converged_f?"yes":"no");
        fflush(out_stream);
        printf_timer("GNO folded branch");

        // dressed branch d: the same schedule on the same start, only the operator differs
        CAS->CI->load_checkpoint("t");
        const int sweeps_two_site_d = CAS->CI->solve_branch(OP_DRESSED, fp.m_delta, fp.sweeps_branch,
                                                            fp.dav_branch, true);
        const int    sweeps_tail_d = CAS->CI->last_tail_sweeps();
        const double dw_two_site_d = CAS->CI->last_solve_dw();
        const double dw_tail_d     = CAS->CI->last_tail_dw();
        const bool   converged_d   = CAS->CI->last_solve_converged();
        CAS->CI->save_state_set("d");
        fprintf(out_stream,"Dressed branch (m=%d): %d sweeps, discarded weight %.3e, converged: %s\n",
                fp.m_delta, sweeps_two_site_d+sweeps_tail_d,
                sweeps_tail_d>0?dw_tail_d:dw_two_site_d, converged_d?"yes":"no");
        fflush(out_stream);
        printf_timer("GNO dressed branch");

        // stored-state expectations, the two cross-set overlaps and read-out A
        const size_t ns2 = (size_t)n_s*n_s;
        std::vector<double> E_f_expect(n_s, 0.0), E_d_expect(n_s, 0.0);
        CAS->CI->expect_set("f", OP_FOLDED , E_f_expect.data());
        CAS->CI->expect_set("d", OP_DRESSED, E_d_expect.data());
        std::vector<double> S_hi_f(ns2, 0.0), S_hi_d(ns2, 0.0);
        CAS->CI->overlap_sets("hi","f",S_hi_f.data());
        CAS->CI->overlap_sets("hi","d",S_hi_d.data());

        std::vector<double> E_A(n_s, 0.0);
        std::vector<int> slot_flag_f(n_s, 0), slot_flag_d(n_s, 0);
        for(int i=0;i<n_s;i++){
            E_A[i] = eps_hi[i] + (E_d_expect[i] - E_f_expect[i]);
            slot_flag_f[i] = row_peak_off_diagonal(n_s, S_hi_f.data(), i)?1:0;
            slot_flag_d[i] = row_peak_off_diagonal(n_s, S_hi_d.data(), i)?1:0;
        }
        printf_timer("GNO branch read-out");

        for(int i=0;i<n_s;i++){
            if(slot_flag_f[i])
                fprintf(out_stream,"NOTE: slot %d changed identity between the reference"
                                   " and the folded branch\n",i);
            if(slot_flag_d[i])
                fprintf(out_stream,"NOTE: slot %d changed identity between the reference"
                                   " and the dressed branch\n",i);
        }

        fprintf(out_stream,"\nCDAS-PT2 (delta_gno) Energy summary:\n");
        PrintEnergy(E_A.data(), n_s, 0);
        fflush(out_stream);
    }

    CAS->CI->release_named_states();
    printf_timer("GNO reference stage");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    fflush(out_stream);

    return 0;
}
