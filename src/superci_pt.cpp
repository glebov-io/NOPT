// Super-CI-PT orbital converger: the first-order amplitudes of the Dyall H0 over the
// singly excited space, taken as the orbital rotation parameters.
# include <cmath>
# include <cstdlib>
# include <cstdio>
# include <cstring>
# include <algorithm>
# include <limits>

# include "blas_link.h"
# include "matr.h"
# include "molecule.h"
# include "CAS.h"
# include "superci_pt.h"
# include "common_vars.h"

namespace {

// A metric direction below TAU_DROP is a linearly dependent SX state and carries no
// rotation. TAU_ADMIT re-admits it only well clear of the drop, so the kept set cannot
// flip between macro-iterations and make the amplitude jump.
const double TAU_DROP  = 1e-10;
const double TAU_ADMIT = 1e-9;

// max|offdiag| the canonicalization is allowed to leave, relative to the block's spread,
// and max|V^T V - I| it is allowed to depart from a rotation. The second is what sees a
// block that was never filled: its off-diagonal residual is exactly zero.
const double CANON_TOL = 1e-10;
const double ORTHO_TOL = 1e-12;

double max_abs(const double * x, long n){
    double m=0.0;
    for(long i=0;i<n;i++){
        if(std::isnan(x[i])) return x[i];
        m = std::max(m, std::fabs(x[i]));
    }
    return m;
}

}   // namespace


int superci_pt_engine::init(int ext_n_c, int ext_n_a, int ext_n_v, int ext_n_ao,
                            const int * ext_rep_num, int ext_n_rep, bool ext_sym, double ext_x_max,
                            int ext_lbfgs){

    n_c   = ext_n_c;
    n_a   = ext_n_a;
    n_v   = ext_n_v;
    n_ao  = ext_n_ao;
    n_mo  = n_c+n_a+n_v;
    n_rep = ext_n_rep;
    rep_num = ext_rep_num;
    sym   = ext_sym;
    x_max = ext_x_max;
    app_max = 0.0;
    pp.reported = false;
    ph.reported = false;
    kept_changed  = false;

    pp.keep.assign(std::max(n_rep,1), -1);
    ph.keep.assign(std::max(n_rep,1), -1);
    pp.nd = 0;
    ph.nd = 0;

    lbfgs.init(ext_lbfgs, (size_t)n_c*n_a + (size_t)n_c*n_v + (size_t)n_a*n_v, x_max);

    return 0;
}

// pp.keep/ph.keep ratchet the kept metric rank against the previous macro-iteration, which assumes
// the occupation spectrum moves smoothly; -1 is init()'s "no previous iteration" state.
void superci_pt_engine::reset_history(){

    lbfgs.restart();
    pp.keep.assign(std::max(n_rep,1), -1);
    ph.keep.assign(std::max(n_rep,1), -1);
}

double superci_pt_engine::calc(const double * G){

    return max_abs(G, (long)n_c*n_a + (long)n_c*n_v + (long)n_a*n_v);
}

// Window-local indices carrying irrep i_r. With symmetry off every rep_num is -1, so the
// caller asks for i_r=-1 and gets the whole window in one block.
void superci_pt_engine::members(int base, int dim, int i_r, std::vector<int>& mem) const{

    mem.clear();
    for(int p=0;p<dim;p++) if(rep_num[base+p]==i_r) mem.push_back(p);
}

// Per-irrep block diagonalization of F[n0..n0+dim), touching nothing: F, MO_VEC,
// rep_num and orb_energy are all left as they were.
void superci_pt_engine::canonicalize_block(const double * F, int n0, int dim,
                                           std::vector<double>& V, std::vector<double>& eps){

    V.assign((size_t)dim*dim, 0.0);
    eps.assign(std::max(dim,1), 0.0);
    if(dim==0) return;

    std::vector<int> reps;
    if(!sym)      reps.push_back(-1);
    else          for(int r=0;r<n_rep;r++) reps.push_back(r);

    std::vector<int> mem;
    std::vector<double> blk, ev;
    for(size_t ir=0; ir<reps.size(); ir++){
        members(n0, dim, reps[ir], mem);
        const int m = mem.size();
        if(m==0) continue;
        blk.assign((size_t)m*m, 0.0);
        ev .assign(m, 0.0);
        for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
            blk[(size_t)i*m+j] = F[(size_t)(n0+mem[i])*n_ao + n0+mem[j]];

        lapack_diag(blk.data(), ev.data(), m);   // rows of blk are the eigenvectors

        for(int mu=0;mu<m;mu++){
            eps[mem[mu]] = ev[mu];
            for(int j=0;j<m;j++)
                V[(size_t)mem[j]*dim + mem[mu]] = blk[(size_t)mu*m + j];
        }
    }
}

// max|offdiag(V^T F V)| within each irrep block: V is block-diagonal, so the inter-irrep
// part of F passes through it untouched and no amplitude ever sees it. Everything
// downstream assumes the blocks are diagonal, and the transformation is silent when
// they are not.
double superci_pt_engine::canonical_residual(const double * F, int n0, int dim,
                                             const std::vector<double>& V){

    if(dim<2) return 0.0;

    std::vector<int> reps;
    if(!sym)      reps.push_back(-1);
    else          for(int r=0;r<n_rep;r++) reps.push_back(r);

    std::vector<int> mem;
    std::vector<double> V_b;
    double off=0.0;
    for(size_t ir=0; ir<reps.size(); ir++){
        members(n0, dim, reps[ir], mem);
        const int m = mem.size();
        if(m<2) continue;

        buf1.assign((size_t)m*m, 0.0);
        buf2.assign((size_t)m*m, 0.0);
        V_b .assign((size_t)m*m, 0.0);
        for(int i=0;i<m;i++)
        for(int j=0;j<m;j++){
            buf1[(size_t)i*m+j] = F[(size_t)(n0+mem[i])*n_ao + n0+mem[j]];
            V_b [(size_t)i*m+j] = V[(size_t)mem[i]*dim + mem[j]];
        }

        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, m,m,m, 1.0,
                       buf1.data(),m, V_b.data(),m, 0.0, buf2.data(),m);
        nopt_par_dgemm(CblasRowMajor,CblasTrans  ,CblasNoTrans, m,m,m, 1.0,
                       V_b.data(),m, buf2.data(),m, 0.0, buf1.data(),m);

        for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
            if(i!=j) off = std::max(off, std::fabs(buf1[(size_t)i*m+j]));
    }

    return off;
}

// max|V^T V - I|. A window the blocking loop never reached leaves V = 0, which is
// invisible to the off-diagonal residual and O(1) here.
double superci_pt_engine::orthogonality_defect(int dim, const std::vector<double>& V){

    if(dim<1) return 0.0;

    buf1.assign((size_t)dim*dim, 0.0);
    nopt_par_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, dim,dim,dim, 1.0,
                   V.data(),dim, V.data(),dim, 0.0, buf1.data(),dim);

    double d=0.0;
    for(int i=0;i<dim;i++)
    for(int j=0;j<dim;j++)
        d = std::max(d, std::fabs(buf1[(size_t)i*dim+j] - (i==j?1.0:0.0)));

    return d;
}

// K[t,u] = -sum_w F^I_uw gamma_tw - sum_wxy (uw|xy) GAMMA_tw,xy, and Ktilde = K + 2 F_tot,
// both on the active-active block and both from the state-averaged RDMs.
void superci_pt_engine::build_koopmans(CAS_engine * CAS){

    const long na2 = (long)n_a*n_a;
    const long na3 = na2*n_a;

    K  .assign((size_t)na2, 0.0);
    K_t.assign((size_t)na2, 0.0);
    if(n_a==0) return;

    // F^I on the active block, laid out as [u*n_a+w] so the GEMM sees a contiguous operand
    buf1.assign((size_t)na2, 0.0);
    for(int u=0;u<n_a;u++)
    for(int w=0;w<n_a;w++)
        buf1[(size_t)u*n_a+w] = CAS->F_core_MO[(size_t)(n_c+u)*n_ao + n_c+w];

    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans, n_a,n_a,n_a, -1.0,
                   CAS->gamma,n_a, buf1.data(),n_a, 0.0, K.data(),n_a);
    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans, n_a,n_a,na3, -1.0,
                   CAS->GAMMA,na3, CAS->aaaa_ints,na3, 1.0, K.data(),n_a);

    // K is symmetric only through the 2-RDM's permutational symmetry, which a truncated
    // MPS satisfies to truncation accuracy; lapack_diag keeps one triangle, so without
    // this the pencil sees an arbitrary half of the asymmetry rather than its mean.
    for(int t=0;t<n_a;t++)
    for(int u=0;u<t;u++){
        const double s = 0.5*(K[(size_t)t*n_a+u] + K[(size_t)u*n_a+t]);
        K[(size_t)t*n_a+u] = s;
        K[(size_t)u*n_a+t] = s;
    }

    for(int t=0;t<n_a;t++)
    for(int u=0;u<n_a;u++)
        K_t[(size_t)t*n_a+u] = K[(size_t)t*n_a+u] + 2.0*CAS->F_tot[(size_t)(n_c+t)*n_ao + n_c+u];
}

// M C_mu = sign * eig_mu * metric * C_mu with C^T metric C = 1, solved per active irrep
// through a canonical orthogonalization of the metric. Rows of C are the C_mu.
void superci_pt_engine::solve_pencil(const double * M_in, const double * metric, double sign,
                                     sx_pencil & P, const char * what){

    P.C.clear();
    P.eig.clear();
    P.rep.clear();
    P.D.clear();
    P.nd = 0;
    if(n_a==0) return;

    std::vector<int> reps;
    if(!sym)      reps.push_back(-1);
    else          for(int r=0;r<n_rep;r++) reps.push_back(r);

    std::vector<int> mem;
    std::vector<double> g_r, n_r, X, sub, red, lam;
    for(size_t ir=0; ir<reps.size(); ir++){
        members(n_c, n_a, reps[ir], mem);
        const int m = mem.size();
        if(m==0) continue;

        g_r.assign((size_t)m*m, 0.0);
        n_r.assign(m, 0.0);
        for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
            g_r[(size_t)i*m+j] = metric[(size_t)mem[i]*n_a + mem[j]];
        lapack_diag(g_r.data(), n_r.data(), m);   // ascending

        int n_lo=0, n_hi=0;
        for(int k=0;k<m;k++){
            if(n_r[k]>TAU_DROP ) n_lo++;
            if(n_r[k]>TAU_ADMIT) n_hi++;
        }
        int & prev = P.keep[!sym ? 0 : reps[ir]];
        int nk = n_lo;
        if(prev>=0 && n_lo>prev) nk = std::max(prev, n_hi);
        if(prev>=0 && nk!=prev) kept_changed = true;
        prev = nk;
        if(nk<m && !P.reported){
            P.reported = true;
            fprintf(out_stream,"NOTE: super-CI-PT drops %d %s direction(s) of the active metric"
                               " (smallest %s occupation %.2e); those rotations are frozen\n",
                               m-nk, what, what, n_r[0]);
        }

        // D rows = the dropped metric eigenvectors, each in its irrep's slots
        const size_t base_d = P.D.size();
        P.D.resize(base_d + (size_t)(m-nk)*n_a, 0.0);
        for(int k=0;k<m-nk;k++){
            double * row = P.D.data() + base_d + (size_t)k*n_a;
            for(int p=0;p<m;p++) row[mem[p]] = g_r[(size_t)k*m+p];
        }
        P.nd += m-nk;
        if(nk==0) continue;

        // X[p*nk+j] = U_keep n_keep^-1/2, the kept metric eigenvectors as columns
        X.assign((size_t)m*nk, 0.0);
        for(int j=0;j<nk;j++){
            const int k = m-nk+j;
            const double s = 1.0/std::sqrt(n_r[k]);
            for(int p=0;p<m;p++) X[(size_t)p*nk+j] = g_r[(size_t)k*m+p]*s;
        }

        sub.assign((size_t)m*m, 0.0);
        for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
            sub[(size_t)i*m+j] = M_in[(size_t)mem[i]*n_a + mem[j]];

        std::vector<double> tmp((size_t)m*nk, 0.0);
        red.assign((size_t)nk*nk, 0.0);
        lam.assign(nk, 0.0);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, m,nk,m, 1.0,
                       sub.data(),m, X.data(),nk, 0.0, tmp.data(),nk);
        nopt_par_dgemm(CblasRowMajor,CblasTrans  ,CblasNoTrans, nk,nk,m, 1.0,
                       X.data(),nk, tmp.data(),nk, 0.0, red.data(),nk);
        lapack_diag(red.data(), lam.data(), nk);   // rows of red are the eigenvectors

        // tmp2[mu*m+p] = sum_j red[mu*nk+j] X[p*nk+j], the pencil vectors in the irrep's slots
        std::vector<double> tmp2((size_t)nk*m, 0.0);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans  , nk,m,nk, 1.0,
                       red.data(),nk, X.data(),nk, 0.0, tmp2.data(),m);

        const size_t base = P.C.size();
        P.C.resize(base + (size_t)nk*n_a, 0.0);
        for(int mu=0;mu<nk;mu++){
            P.eig.push_back(sign*lam[mu]);
            P.rep.push_back(reps[ir]);
            double * row = P.C.data() + base + (size_t)mu*n_a;
            for(int p=0;p<m;p++) row[mem[p]] = tmp2[(size_t)mu*m+p];
        }
    }
}

// Cayley: U = (I - kappa/2)^-1 (I + kappa/2), exactly orthogonal for antisymmetric kappa.
// MO_VEC holds orbitals as rows, so C_new = U C_old; only the n_mo optimized rows exist.
void superci_pt_engine::apply_rotation(CAS_engine * CAS){

    const size_t nn = (size_t)n_mo*n_mo;
    buf1.assign(nn, 0.0);
    buf2.assign(nn, 0.0);
    for(size_t i=0;i<nn;i++){
        buf1[i] = -0.5*kappa[i];
        buf2[i] =  0.5*kappa[i];
    }
    for(int i=0;i<n_mo;i++){
        buf1[(size_t)i*n_mo+i] += 1.0;
        buf2[(size_t)i*n_mo+i] += 1.0;
    }

    // A row-major buffer read column-major is its own transpose, and inversion commutes
    // with that flip, so buf1 comes back holding (I - kappa/2)^-1 row-major.
    lapack_int N = n_mo, info = 0, lwork = 2*N*N+6*N+1;
    std::vector<lapack_int> piv(std::max(n_mo,1));
    std::vector<double> work(lwork);
#ifdef _OPENBLAS
    LAPACK_dgetrf(&N,&N,buf1.data(),&N,piv.data(),&info);
    if(info==0) LAPACK_dgetri(&N,buf1.data(),&N,piv.data(),work.data(),&lwork,&info);
#endif
#ifdef _MKL
    DGETRF(&N,&N,buf1.data(),&N,piv.data(),&info);
    if(info==0) DGETRI(&N,buf1.data(),&N,piv.data(),work.data(),&lwork,&info);
#endif
    if(info!=0){
        fprintf(out_stream,"ERROR: super-CI-PT Cayley transform failed (LAPACK info=%d)\n",(int)info);
        exit(EXIT_FAILURE);
    }

    std::vector<double> U(nn, 0.0);
    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_mo,n_mo,n_mo, 1.0,
                   buf1.data(),n_mo, buf2.data(),n_mo, 0.0, U.data(),n_mo);

    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_mo,n_ao,n_mo, 1.0,
                   U.data(),n_mo, CAS->MO_VEC,n_ao, 0.0, CAS->MO_BUF,n_ao);
    memcpy(CAS->MO_VEC, CAS->MO_BUF, (size_t)n_mo*n_ao*sizeof(double));
}

// T = -M^-1 q on a full [it|ia|ta] vector: into the canonical frame, the three PT divisions,
// back out, the symmetry masks. Uses the frame and the pencils step() built this macro-iteration.
void superci_pt_engine::amplitudes(const double * q, std::vector<double>& T){

    const size_t o_ia  = (size_t)n_c*n_a;
    const size_t o_ta  = o_ia + (size_t)n_c*n_v;
    const size_t n_rot = o_ta + (size_t)n_a*n_v;
    const double * g_it = q;
    const double * g_ia = q + o_ia;
    const double * g_ta = q + o_ta;

    // --- gradient into the canonical frame -------------------------------------------
    gc.assign(n_rot, 0.0);
    double * gc_it = gc.data();
    double * gc_ia = gc.data() + o_ia;
    double * gc_ta = gc.data() + o_ta;
    if(n_c&&n_a)
        nopt_par_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, n_c,n_a,n_c, 1.0,
                       V_c.data(),n_c, g_it,n_a, 0.0, gc_it,n_a);
    if(n_a&&n_v)
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_a,n_v,n_v, 1.0,
                       g_ta,n_v, V_v.data(),n_v, 0.0, gc_ta,n_v);
    if(n_c&&n_v){
        buf1.assign((size_t)n_c*n_v, 0.0);
        nopt_par_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, n_c,n_v,n_c, 1.0,
                       V_c.data(),n_c, g_ia,n_v, 0.0, buf1.data(),n_v);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_c,n_v,n_v, 1.0,
                       buf1.data(),n_v, V_v.data(),n_v, 0.0, gc_ia,n_v);
    }

    const int nk_p = pp.eig.size();
    const int nk_h = ph.eig.size();

    // --- amplitudes ------------------------------------------------------------------
    T .assign(n_rot, 0.0);
    Tc.assign(n_rot, 0.0);
    double * T_it  = T.data();
    double * T_ia  = T.data() + o_ia;
    double * T_ta  = T.data() + o_ta;
    double * Tc_it = Tc.data();
    double * Tc_ia = Tc.data() + o_ia;
    double * Tc_ta = Tc.data() + o_ta;

    // symmetry-forbidden pairs are skipped, not divided: their gradient and denominator can both be 0
    for(int i=0;i<n_c;i++)
    for(int a=0;a<n_v;a++){
        if(rep_num[i]!=rep_num[n_c+n_a+a]) continue;
        Tc_ia[(size_t)i*n_v+a] = -gc_ia[(size_t)i*n_v+a]/(4.0*(eps_v[a]-eps_c[i]));
    }

    if(nk_p&&n_v){
        std::vector<double> Y((size_t)nk_p*n_v, 0.0);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, nk_p,n_v,n_a, 1.0,
                       pp.C.data(),n_a, gc_ta,n_v, 0.0, Y.data(),n_v);
        for(int mu=0;mu<nk_p;mu++)
        for(int a=0;a<n_v;a++){
            if(pp.rep[mu]!=rep_num[n_c+n_a+a]) continue;
            Y[(size_t)mu*n_v+a] *= -0.5/(eps_v[a]-pp.eig[mu]);
        }
        nopt_par_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, n_a,n_v,nk_p, 1.0,
                       pp.C.data(),n_a, Y.data(),n_v, 0.0, Tc_ta,n_v);
    }

    if(nk_h&&n_c){
        std::vector<double> Y((size_t)nk_h*n_c, 0.0);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans, nk_h,n_c,n_a, 1.0,
                       ph.C.data(),n_a, gc_it,n_a, 0.0, Y.data(),n_c);
        for(int mu=0;mu<nk_h;mu++)
        for(int i=0;i<n_c;i++){
            if(ph.rep[mu]!=rep_num[i]) continue;
            Y[(size_t)mu*n_c+i] *= -0.5/(ph.eig[mu]-eps_c[i]);
        }
        nopt_par_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, n_c,n_a,nk_h, 1.0,
                       Y.data(),n_c, ph.C.data(),n_a, 0.0, Tc_it,n_a);
    }

    // --- back out of the canonical frame ---------------------------------------------
    if(n_c&&n_a)
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_c,n_a,n_c, 1.0,
                       V_c.data(),n_c, Tc_it,n_a, 0.0, T_it,n_a);
    if(n_a&&n_v)
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans, n_a,n_v,n_v, 1.0,
                       Tc_ta,n_v, V_v.data(),n_v, 0.0, T_ta,n_v);
    if(n_c&&n_v){
        buf1.assign((size_t)n_c*n_v, 0.0);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_c,n_v,n_c, 1.0,
                       V_c.data(),n_c, Tc_ia,n_v, 0.0, buf1.data(),n_v);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans, n_c,n_v,n_v, 1.0,
                       buf1.data(),n_v, V_v.data(),n_v, 0.0, T_ia,n_v);
    }

    // symmetry-forbidden amplitudes, mirroring calc_grad's masks
    for(int i=0;i<n_c;i++)
    for(int t=0;t<n_a;t++)
        if(rep_num[i]!=rep_num[n_c+t]) T_it[(size_t)i*n_a+t]=0.0;
    for(int i=0;i<n_c;i++)
    for(int a=0;a<n_v;a++)
        if(rep_num[i]!=rep_num[n_c+n_a+a]) T_ia[(size_t)i*n_v+a]=0.0;
    for(int t=0;t<n_a;t++)
    for(int a=0;a<n_v;a++)
        if(rep_num[n_c+t]!=rep_num[n_c+n_a+a]) T_ta[(size_t)t*n_v+a]=0.0;
}

// Smallest PT denominator behind T over the symmetry-allowed pairs.
double superci_pt_engine::min_denominator() const{

    const int nk_p = pp.eig.size();
    const int nk_h = ph.eig.size();
    double den = std::numeric_limits<double>::infinity();
    for(int i=0;i<n_c;i++)
    for(int a=0;a<n_v;a++)
        if(rep_num[i]==rep_num[n_c+n_a+a]) den = std::min(den, 4.0*(eps_v[a]-eps_c[i]));
    for(int mu=0;mu<nk_p;mu++)
    for(int a=0;a<n_v;a++)
        if(pp.rep[mu]==rep_num[n_c+n_a+a]) den = std::min(den, eps_v[a]-pp.eig[mu]);
    for(int mu=0;mu<nk_h;mu++)
    for(int i=0;i<n_c;i++)
        if(ph.rep[mu]==rep_num[i]) den = std::min(den, ph.eig[mu]-eps_c[i]);
    return den;
}

double superci_pt_engine::step(CAS_engine * CAS, double s_conv){

    // --- canonical frame -------------------------------------------------------------
    canonicalize_block(CAS->F_tot, 0       , n_c, V_c, eps_c);
    canonicalize_block(CAS->F_tot, n_c+n_a , n_v, V_v, eps_v);

    const double off_c = canonical_residual(CAS->F_tot, 0      , n_c, V_c);
    const double off_v = canonical_residual(CAS->F_tot, n_c+n_a, n_v, V_v);
    const double span  = std::max(1.0, std::max(max_abs(eps_c.data(),n_c),
                                                max_abs(eps_v.data(),n_v)));
    const double orth_c = orthogonality_defect(n_c, V_c);
    const double orth_v = orthogonality_defect(n_v, V_v);
    if(std::max(off_c,off_v) > CANON_TOL*span){
        fprintf(out_stream,"ERROR: super-CI-PT canonicalization left the Fock blocks non-diagonal"
                           " (core %.3e, virtual %.3e, tolerance %.3e)\n",
                           off_c, off_v, CANON_TOL*span);
        exit(EXIT_FAILURE);
    }

    if(std::max(orth_c,orth_v) > ORTHO_TOL){
        fprintf(out_stream,"ERROR: super-CI-PT canonicalization is not a rotation"
                           " (core %.3e, virtual %.3e, tolerance %.3e)\n",
                           orth_c, orth_v, ORTHO_TOL);
        exit(EXIT_FAILURE);
    }

    // --- Koopmans matrices and the two pencils ---------------------------------------
    build_koopmans(CAS);

    // both metrics go through lapack_diag too, and CAS->gamma is shared and must not be
    // touched, so the symmetric part goes into local copies
    std::vector<double> gam_p((size_t)n_a*n_a, 0.0), gam_h((size_t)n_a*n_a, 0.0);
    for(int t=0;t<n_a;t++)
    for(int u=0;u<n_a;u++){
        const double g = 0.5*(CAS->gamma[(size_t)t*n_a+u] + CAS->gamma[(size_t)u*n_a+t]);
        gam_p[(size_t)t*n_a+u] = g;
        gam_h[(size_t)t*n_a+u] = (t==u?2.0:0.0) - g;
    }

    solve_pencil(K  .data(), gam_p.data(), -1.0, pp, "particle");
    solve_pencil(K_t.data(), gam_h.data(), +1.0, ph, "hole");

    // --- the bare step ---------------------------------------------------------------
    const size_t o_ia  = (size_t)n_c*n_a;
    const size_t o_ta  = o_ia + (size_t)n_c*n_v;
    const size_t n_rot = o_ta + (size_t)n_a*n_v;
    amplitudes(CAS->G, T_vec);

    double mx = max_abs(T_vec.data(), (long)n_rot);
    // converged in these orbitals: they stay where the amplitude was evaluated, no step applied
    if(mx<s_conv){
        app_max = 0.0;
        fprintf(out_stream," SX-PT: max|T| %.1e below s_conv, orbitals kept as they are\n",mx);
        return mx;
    }

    if(lbfgs.traced()) fprintf(out_stream," SX-PT den %.2e\n", min_denominator());

    // --- L-BFGS, then the trust cap, then the rotation --------------------------------
    const double * a_it = T_vec.data();
    const double * a_ia = T_vec.data()+o_ia;
    const double * a_ta = T_vec.data()+o_ta;
    double pre_cap = mx;                // |rotation| the trust cap was handed
    // The history spans rotations the pencils now freeze or newly admit: its pairs cannot describe them.
    if(kept_changed){
        kept_changed = false;
        if(lbfgs.active()){            // the flag needs a previous iteration, so there is a previous point
            lbfgs.restart();
            fprintf(out_stream,"NOTE: super-CI-PT L-BFGS history dropped: the active metric directions the"
                               " pencils keep changed this macro-iteration\n");
        }
    }
    if(lbfgs.active()){
        g_vec.assign(CAS->G, CAS->G + n_rot);
        // the L-BFGS step keeps no component along the metric directions the pencils drop
        auto project = [this, o_ta](std::vector<double>& v){
            if(pp.nd>0 && n_v>0){
                std::vector<double> Y((size_t)pp.nd*n_v, 0.0);
                nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, pp.nd,n_v,n_a, 1.0,
                               pp.D.data(),n_a, v.data()+o_ta,n_v, 0.0, Y.data(),n_v);
                nopt_par_dgemm(CblasRowMajor,CblasTrans  ,CblasNoTrans, n_a,n_v,pp.nd, -1.0,
                               pp.D.data(),n_a, Y.data(),n_v, 1.0, v.data()+o_ta,n_v);
            }
            if(ph.nd>0 && n_c>0){
                std::vector<double> Y((size_t)n_c*ph.nd, 0.0);
                nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans  , n_c,ph.nd,n_a, 1.0,
                               v.data(),n_a, ph.D.data(),n_a, 0.0, Y.data(),ph.nd);
                nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, n_c,n_a,ph.nd, -1.0,
                               Y.data(),ph.nd, ph.D.data(),n_a, 1.0, v.data(),n_a);
            }
        };
        // r = M^-1 v, the positive map at the centre of the recursion
        auto h0inv = [this](const std::vector<double>& v, std::vector<double>& r){
            amplitudes(v.data(), r);
            cblas_dscal((lapack_int)r.size(), -1.0, r.data(),1);
        };
        pre_cap = lbfgs.direction(g_vec, h0inv, project, app_vec, mx);
        a_it = app_vec.data();
        a_ia = app_vec.data()+o_ia;
        a_ta = app_vec.data()+o_ta;
    }
    else if(mx>x_max){
        const double s = x_max/mx;
        for(size_t i=0;i<n_rot;i++) T_vec[i]*=s;
    }
    if(pre_cap>x_max)
        fprintf(out_stream," SX-PT is scaling rotation angle matrix Xmax=%.5e\n", pre_cap);

    app_max = max_abs(a_it, (long)n_rot);

    kappa.assign((size_t)n_mo*n_mo, 0.0);
    for(int i=0;i<n_c;i++)
    for(int t=0;t<n_a;t++){
        kappa[(size_t)i*n_mo + n_c+t] =  a_it[(size_t)i*n_a+t];
        kappa[(size_t)(n_c+t)*n_mo+i] = -a_it[(size_t)i*n_a+t];
    }
    for(int i=0;i<n_c;i++)
    for(int a=0;a<n_v;a++){
        kappa[(size_t)i*n_mo + n_c+n_a+a] =  a_ia[(size_t)i*n_v+a];
        kappa[(size_t)(n_c+n_a+a)*n_mo+i] = -a_ia[(size_t)i*n_v+a];
    }
    for(int t=0;t<n_a;t++)
    for(int a=0;a<n_v;a++){
        kappa[(size_t)(n_c+t)*n_mo + n_c+n_a+a] =  a_ta[(size_t)t*n_v+a];
        kappa[(size_t)(n_c+n_a+a)*n_mo + n_c+t] = -a_ta[(size_t)t*n_v+a];
    }

    apply_rotation(CAS);

    return mx;
}
