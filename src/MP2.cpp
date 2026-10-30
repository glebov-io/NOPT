//standart

//user
# include "blas_link.h"
# include "molecule.h"
# include "inp_par_read.h"
# include "SCF.h"
# include "RI.h"
# include "PT_tensors.h"
# include "matr.h"
# include "doCI_matr.h"
# include "common_vars.h"
# include "timer.h"
# include "defaults.h"
# include "mp2.h"
# include "grabbers.h"

# include <vector>
# include <cmath>

//------------------------------------------------------------------------------------------------------------------------
// Unrelaxed MP2 1-RDM (canonical MO basis) + natural orbitals.
//
// Occupied i,j,k; virtual a,b,c; canonical spatial orbitals. NOPT sign conventions:
//   t_ik^ab   = (ia|kb) / (eps_i + eps_k - eps_a - eps_b)
//   tbar_ik^ab = 2 t_ik^ab - t_ik^ba
// Density blocks (spin-summed):
//   dP_ij = -2 sum_{k,ab} t_ik^ab tbar_jk^ab      (occ-occ, P_ij = 2 delta_ij + dP_ij)
//   P_ab  =  2 sum_{ij,c} t_ij^ac tbar_ij^bc      (virt-virt)
// The OV block is exactly zero, so the two blocks diagonalize independently.
//
// The elementwise loops parallelize the outer virtual index: n_v >> n_c gives even
// chunks, and T is laid out with a-major stride so each thread owns a contiguous block.
//------------------------------------------------------------------------------------------------------------------------
static int mp2_nat_orb(molecule * M, RI_data & R, double * eps,
                       int n_c, int n_v, int n_f, int n_occ, int n_ao,
                       double E_corr, char * job_name)
{
    long naux = R.aux_n_ao;
    double * VC = R.VC_RI_M;              // [(a*n_c+i)*naux + P] = (ia|P)
    double * e_c = eps;                   // occupied (correlated) orbital energies
    double * e_v = eps + n_c;             // virtual orbital energies

    std::vector<double> T (long(n_v)*n_c*n_v);   // t_ik^ab   for fixed k, [(a*n_c+i)*n_v+b]
    std::vector<double> Tb(long(n_v)*n_c*n_v);   // tbar_ik^ab for fixed k
    std::vector<double> dP_oo(long(n_c)*n_c, 0.0);
    std::vector<double> P_vv (long(n_v)*n_v, 0.0);

    for(int k=0; k<n_c; k++){

        // T[a,i,b] = (ia|kb): one big dgemm of the (n_v*n_c) x naux tensor against the fixed-k slice.
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                       n_v*n_c, n_v, naux, 1.0,
                       VC, naux,
                       VC + long(k)*naux, n_c*naux, 0.0,
                       T.data(), n_v);

        // divide by the energy denominator -> t_ik^ab (in place)
#pragma omp parallel for
        for(int a=0; a<n_v; a++)
        for(int i=0; i<n_c; i++)
        for(int b=0; b<n_v; b++){
            long idx = (long(a)*n_c+i)*n_v+b;
            T[idx] /= (e_c[i]+e_c[k]-e_v[a]-e_v[b]);
        }

        // Tb[a,i,b] = 2 t_ik^ab - t_ik^ba
#pragma omp parallel for
        for(int a=0; a<n_v; a++)
        for(int i=0; i<n_c; i++)
        for(int b=0; b<n_v; b++){
            long idx  = (long(a)*n_c+i)*n_v+b;
            long idxt = (long(b)*n_c+i)*n_v+a;
            Tb[idx] = 2.0*T[idx] - T[idxt];
        }

        // P_ab += 2 sum_{i,c} t_ik^ac tbar_ik^bc  (row a of T is the contiguous [i][c] block)
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                       n_v, n_v, n_c*n_v, 2.0,
                       T.data(),  n_c*n_v,
                       Tb.data(), n_c*n_v, 1.0,
                       P_vv.data(), n_v);

        // dP_ij -= 2 sum_{a,b} t_ik^ab tbar_jk^ab  (per-a n_c x n_c contractions, beta=1)
        for(int a=0; a<n_v; a++)
            nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                           n_c, n_c, n_v, -2.0,
                           T.data() +long(a)*n_c*n_v, n_v,
                           Tb.data()+long(a)*n_c*n_v, n_v, 1.0,
                           dP_oo.data(), n_c);
    }

    // exact internal identities (gates against the independently coded CCVV energy)
    double tr_dP = 0.0, tr_Pvv = 0.0, e_dP = 0.0, e_Pvv = 0.0;
    for(int i=0; i<n_c; i++){ tr_dP  += dP_oo[i*n_c+i]; e_dP  += e_c[i]*dP_oo[i*n_c+i]; }
    for(int a=0; a<n_v; a++){ tr_Pvv += P_vv [a*n_v+a]; e_Pvv += e_v[a]*P_vv [a*n_v+a]; }

    double resid_N = std::fabs(tr_dP + tr_Pvv);
    double resid_E = std::fabs(E_corr + e_dP + e_Pvv);

    if(resid_N > 1e-8)
        fprintf(out_stream,"WARNING: MP2 density particle-conservation residual = %e\n",resid_N);
    if(resid_E > 1e-8)
        fprintf(out_stream,"WARNING: MP2 density energy-consistency residual   = %e\n",resid_E);

    // symmetrize before diagonalization
    for(int i=0; i<n_c; i++)
    for(int j=i+1; j<n_c; j++){
        double s = 0.5*(dP_oo[i*n_c+j]+dP_oo[j*n_c+i]);
        dP_oo[i*n_c+j] = dP_oo[j*n_c+i] = s;
    }
    for(int a=0; a<n_v; a++)
    for(int b=a+1; b<n_v; b++){
        double s = 0.5*(P_vv[a*n_v+b]+P_vv[b*n_v+a]);
        P_vv[a*n_v+b] = P_vv[b*n_v+a] = s;
    }

    // full occupied block: P_oo = 2*I + dP_oo
    for(int i=0; i<n_c; i++) dP_oo[i*n_c+i] += 2.0;

    std::vector<double> occ_o(n_c), occ_v(n_v);
    lapack_diag(dP_oo.data(), occ_o.data(), n_c);   // eigenvectors in rows, ascending occupations
    lapack_diag(P_vv.data(),  occ_v.data(), n_v);

    // occupations -> orb_energy field (raw, no clamping)
    for(int i=0; i<n_f; i++) M->orb_energy[i]       = 2.0;
    for(int i=0; i<n_c; i++) M->orb_energy[n_f+i]   = occ_o[i];
    for(int a=0; a<n_v; a++) M->orb_energy[n_occ+a] = occ_v[a];

    // rotate MO_VEC blockwise: C_NO = U . C_MO (frozen rows untouched)
    std::vector<double> B_occ (long(n_c)*n_ao);
    std::vector<double> B_virt(long(n_v)*n_ao);
    cblas_dcopy(n_c*n_ao, M->MO_VEC + long(n_f)  *n_ao, 1, B_occ.data(),  1);
    cblas_dcopy(n_v*n_ao, M->MO_VEC + long(n_occ)*n_ao, 1, B_virt.data(), 1);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                n_c, n_ao, n_c, 1.0, dP_oo.data(), n_c, B_occ.data(),  n_ao, 0.0,
                M->MO_VEC + long(n_f)  *n_ao, n_ao);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                n_v, n_ao, n_v, 1.0, P_vv.data(),  n_v, B_virt.data(), n_ao, 0.0,
                M->MO_VEC + long(n_occ)*n_ao, n_ao);

    M->sort_orbs('d');
    M->check_orb_symmetry();
    M->MO_gamess_format();

    // write the natural orbitals in the standard punch formats
    char name[BUF_LINE_LENGTH];

    fprintf(out_stream,"\n\n");
    fprintf(out_stream,"Writing MP2 natural orbitals:\n");

    sprintf(name,"%s_MP2_NatOrb.out",job_name);
    M->GAMESS_type_out_print(name, -1);
    fprintf(out_stream,"visualization file: %s\n",name);

    sprintf(name,"%s_MP2_NatOrb.orb",job_name);
    M->MO_print(name);
    fprintf(out_stream,"data file         : %s\n",name);

    // terse summary
    double tr_P = 2.0*n_f;
    for(int i=0; i<n_c; i++) tr_P += occ_o[i];
    for(int a=0; a<n_v; a++) tr_P += occ_v[a];

    fprintf(out_stream,"\n");
    fprintf(out_stream,"Occupied-block NO occupations (descending):\n");
    for(int i=n_c-1; i>=0; i--)
        fprintf(out_stream," % .8f\n",occ_o[i]);
    fprintf(out_stream,"Virtual-block NO occupations, largest %d (descending):\n",
            n_v<15?n_v:15);
    for(int a=n_v-1, m=0; a>=0 && m<15; a--, m++)
        fprintf(out_stream," % .8f\n",occ_v[a]);

    fprintf(out_stream,"\n");
    fprintf(out_stream,"Tr[P]                             = % .8f  (n_el = %d)\n",tr_P,M->n_el_calc);
    fprintf(out_stream,"particle-conservation residual    = %e\n",resid_N);
    fprintf(out_stream,"energy-consistency  residual      = %e\n",resid_E);
    fprintf(out_stream,"\n");

    return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int MP2(molecule * M, mp2_par * mp2, char * job_name)
{
    mp2->write_info();

    if(RI==0){
        fprintf(out_stream,"ERROR: MP2 requires RI=1\n");
        exit(EXIT_FAILURE);
    }
    if(M->n_el_calc % 2 != 0){
        fprintf(out_stream,"ERROR: MP2 requires a closed-shell reference (even electron count, got %d)\n",
                M->n_el_calc);
        exit(EXIT_FAILURE);
    }

    int n_occ = M->n_el_calc/2;
    int n_f   = mp2->n_f;
    int n_c   = n_occ - n_f;
    int n_v   = M->n_mo - n_occ;
    int n_ao  = M->n_ao;

    if(n_f < 0 || n_f >= n_occ){
        fprintf(out_stream,"ERROR: MP2 n_f=%d out of range [0, %d)\n",n_f,n_occ);
        exit(EXIT_FAILURE);
    }

    if(RI)gen_RI_AA(M);

    // reference RHF energy, recomputed from the converged orbitals (same formula as SCF)
    std::vector<double> DM(long(n_ao)*n_ao), F_AO(long(n_ao)*n_ao);
    gen_HF_DM(DM.data(), M->MO_VEC, n_ao, n_occ);
    M->calc_F_AO(F_AO.data(), DM.data(), 1.0);
    double E_RHF = E_1el_calc(F_AO.data(), DM.data(), n_ao, n_ao)
                 + E_1el_calc(M->H_AO,     DM.data(), n_ao, n_ao) + M->V_nuc;

    // correlated-space RI B-tensor; frozen core enters via a pointer shift into MO_VEC/orb_energy
    RI_data R;
    R.set_par(M, n_c, 0, n_v);
    R.MO_calc(M->MO_VEC + long(n_f)*n_ao, n_ao);
    double * eps = M->orb_energy + n_f;   // [core(n_c) | virt(n_v)]

    // MP2 correlation energy: reuse the validated closed-shell CCVV code (n_a = 0)
    PT_tensors T;
    T.set_par(&R, eps, n_c, 0, n_v, NULL, NULL, NULL, 0.0, M);
    T.RF_PS = 0;
    T.calc_EE_2_CCVV();
    double E_corr = T.RF_PS;

    fprintf(out_stream,"\n");
    fprintf(out_stream,"E(RHF)                            = % .10f\n",E_RHF);
    fprintf(out_stream,"E(MP2) correlation                = % .10f\n",E_corr);
    fprintf(out_stream,"E(MP2) total                      = % .10f\n",E_RHF+E_corr);
    fprintf(out_stream,"\n");

    if(grab_E==1){
        E_grabbed.resize(1);
        E_grabbed[0]=E_RHF+E_corr;
    }

    if(mp2->nat_orb)
        mp2_nat_orb(M, R, eps, n_c, n_v, n_f, n_occ, n_ao, E_corr, job_name);

    printf_timer("MP2");

    return 0;
}
