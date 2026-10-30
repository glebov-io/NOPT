# include "blas_link.h"

# include "avas.h"
# include "matr.h"
# include "libint_link.h"
# include "basis_lib_read.h"
# include "common_vars.h"
# include "defaults.h"
# include "SCF.h"
# include "RI.h"
# include "timer.h"

# include <vector>
# include <cstdio>
# include <algorithm>
# include <numeric>
# include <string>

//------------------------------------------------------------------------------------------------------------------------
// sigma spectra beyond the selection boundary are printed only this far
#define AVAS_PRINT_TAIL 20

static const char * avas_l_labels = "spdfghik";

// Reference shells of the requested atoms carrying a requested nl label; a label bound to an
// atom applies to that atom only. Within one atom the k-th reference shell of angular momentum
// l is the principal number n = k+l+1.
static std::vector<Shell> avas_ref_shells(molecule * M, const avas_par & A,
                                          const std::vector<Shell> & all,
                                          const std::vector<int> & center)
{
    std::vector<Shell> out;
    int n_kw = int(A.shell_n.size());

    for(int i_sel=0; i_sel<int(A.atoms.size()); i_sel++){

        int i_a = A.atoms[i_sel]-1;

        std::vector<int> found(n_kw,0);
        std::vector<int> apply(n_kw);
        int n_l[8]={0,0,0,0,0,0,0,0};

        for(int k=0;k<n_kw;k++)
            apply[k]=((A.shell_atom[k]==0)||(A.shell_atom[k]==i_a+1));

        for(int i=0;i<int(all.size());i++){
            if(center[i]!=i_a)continue;
            int l = all[i].contr[0].l;
            if(l>7)continue;
            n_l[l]++;
            for(int k=0;k<n_kw;k++)
                if(apply[k])
                if((A.shell_l[k]==l)&&(A.shell_n[k]==n_l[l]+l)){
                    out.push_back(all[i]);
                    found[k]=1;
                }
        }

        for(int k=0;k<n_kw;k++)
            if(apply[k])
            if(found[k]==0){
                fprintf(out_stream,"ERROR: $AVAS reference shell %d%c is absent for atom %d (%s)\n",
                                    A.shell_n[k],avas_l_labels[A.shell_l[k]],i_a+1,M->atom_names[i_a]);
                fprintf(out_stream,"       basis %s, library %s\n",A.ref_basis.c_str(),NOPT_LIB);
                exit(EXIT_FAILURE);
            }
    }

    return out;
}

// (n,l) labels that apply to atom i_a (0-based), sorted
static std::vector<int> avas_atom_labels(const avas_par & A, int i_a)
{
    std::vector<int> nl;
    for(int k=0;k<int(A.shell_n.size());k++)
        if((A.shell_atom[k]==0)||(A.shell_atom[k]==i_a+1))nl.push_back(A.shell_n[k]*8+A.shell_l[k]);
    std::sort(nl.begin(),nl.end());
    return nl;
}

static std::string avas_rep_name(molecule * M, int r)
{
    std::string s(M->S.rep_name[r]);
    s.erase(0,s.find_first_not_of(' '));
    s.erase(s.find_last_not_of(' ')+1);
    return s;
}

// A = (C S12) S22^-1 (C S12)^T for one MO row block; U <- eigenvectors in rows, sigma ascending.
// With lab (the irrep of each block row, all valid) A is diagonalized per irrep, its off-irrep part
// never read, and lab returns the irrep of each eigenvector row.
static void avas_block_projector(const double * C, int n_b, int n_ao, int n_ref,
                                 const double * S12, const double * S22i,
                                 double * U, double * sigma, int * lab, int n_rep)
{
    if(n_b<=0)return;

    std::vector<double> Mb(size_t(n_b)*n_ref);
    std::vector<double> P (size_t(n_b)*n_ref);

    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                   n_b,n_ref,n_ao,1.0,
                   C,n_ao,
                   S12,n_ref,0.0,
                   Mb.data(),n_ref);

    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                   n_b,n_ref,n_ref,1.0,
                   Mb.data(),n_ref,
                   S22i,n_ref,0.0,
                   P.data(),n_ref);

    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                   n_b,n_b,n_ref,1.0,
                   P.data(),n_ref,
                   Mb.data(),n_ref,0.0,
                   U,n_b);

    symmetrization(U,n_b);
    if(lab==nullptr){
        lapack_diag(U,sigma,n_b);
        return;
    }

    // irrep-major eigenvectors scattered into full-length rows, then one stable ascending sort
    std::vector<double> Ab(U,U+size_t(n_b)*n_b), V(size_t(n_b)*n_b,0.0), s(n_b);
    std::vector<int>    l(n_b);
    int k=0;
    for(int r=0;r<n_rep;r++){
        std::vector<int> idx;
        for(int i=0;i<n_b;i++)if(lab[i]==r)idx.push_back(i);
        int m=int(idx.size());
        if(m==0)continue;

        std::vector<double> B(size_t(m)*m), e(m);
        for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
            B[size_t(i)*m+j]=Ab[size_t(idx[i])*n_b+idx[j]];
        lapack_diag(B.data(),e.data(),m);

        for(int i=0;i<m;i++,k++){
            for(int j=0;j<m;j++)V[size_t(k)*n_b+idx[j]]=B[size_t(i)*m+j];
            s[k]=e[i];
            l[k]=r;
        }
    }

    std::vector<int> ord(n_b);
    std::iota(ord.begin(),ord.end(),0);
    std::stable_sort(ord.begin(),ord.end(),[&](int a,int b){ return s[a]<s[b]; });
    for(int i=0;i<n_b;i++){
        cblas_dcopy(n_b,V.data()+size_t(ord[i])*n_b,1,U+size_t(i)*n_b,1);
        sigma[i]=s[ord[i]];
        lab  [i]=l[ord[i]];
    }
}

// index of the largest drop in a descending spectrum
static int avas_max_gap(const double * sigma, int n)
{
    int best=-1;
    double g=-1.0;

    for(int i=0;i+1<n;i++)
        if(sigma[i]-sigma[i+1]>g){
            g=sigma[i]-sigma[i+1];
            best=i;
        }

    return best;
}

static void avas_print_sigma(const char * title, const double * sigma, int n, int n_sel)
{
    int n_p = n_sel+AVAS_PRINT_TAIL;
    if(n_p>n)n_p=n;

    fprintf(out_stream,"%s (%d of %d selected)\n",title,n_sel,n);
    for(int i=0,c=0;i<n_p;i++,c++){
        if((i==n_sel)&&(i)){ fprintf(out_stream,"\n  ---- selection boundary ----\n"); c=0; }
        else if((i)&&(c==6)){ fprintf(out_stream,"\n"); c=0; }
        fprintf(out_stream," %11.8f",sigma[i]);
    }
    if(n_p<n)fprintf(out_stream,"\n  ... %d smaller values not printed",n-n_p);
    fprintf(out_stream,"\n\n");
}

// Rotates each row block [0,n_frz) [n_frz,n_cor) [n_cor,n_cor+n_act) [n_cor+n_act,n_mo) into the
// eigenvectors of F = h + 2J[D] - K[D], D over the n_occ occupied rows (invariant under the AVAS
// rotations); eigenvalues ascending into orb_energy. Under $SYMM per irrep, rows regrouped by label.
static void avas_canonicalize(molecule * M, int n_frz)
{
    int n_ao  = M->n_ao;
    int n_mo  = M->n_mo;
    int n_occ = M->n_el_calc/2;
    int n_cor = M->n_cor_orb;
    int n_act = M->n_act_orb[0];

    std::vector<double> DM(size_t(n_ao)*n_ao), F(size_t(n_ao)*n_ao), B(size_t(n_ao)*n_ao);

    if(RI)gen_RI_AA(M);
    M->mc=0;
    gen_HF_DM(DM.data(), M->MO_VEC, n_ao, n_occ);
    M->calc_F_AO(F.data(), DM.data(), 1.0);

    // MO-basis Fock over the n_mo rows, row stride n_ao as diag_X_MO_block reads it
    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                   n_ao,n_mo,n_ao,1.0,
                   F.data(),n_ao,
                   M->MO_VEC,n_ao,0.0,
                   B.data(),n_ao);
    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                   n_mo,n_mo,n_ao,1.0,
                   M->MO_VEC,n_ao,
                   B.data(),n_ao,0.0,
                   F.data(),n_ao);

    M->diag_X_MO_block(F.data(), 0          , n_frz            , nullptr);
    M->diag_X_MO_block(F.data(), n_frz      , n_cor-n_frz      , nullptr);
    M->diag_X_MO_block(F.data(), n_cor      , n_act            , nullptr);
    M->diag_X_MO_block(F.data(), n_cor+n_act, n_mo-n_cor-n_act , nullptr);
}

//------------------------------------------------------------------------------------------------------------------------
int avas_steer(molecule * M, const avas_par & A, char * job_name)
{
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"_____________________Starting_AVAS_orbital_steering____________________\n\n");

    A.write_info();

    int n_ao  = M->n_ao;
    int n_occ = M->n_el_calc/2;
    int n_vir = M->n_mo-n_occ;
    int k_o   = n_occ-M->n_cor_orb;
    int k_v   = M->n_act_orb[0]-k_o;

    if((k_o<0)||(k_v<0)||(k_v>n_vir)){
        fprintf(out_stream,"ERROR: AVAS cannot fill the active window: %d occupied and %d virtual orbitals\n",k_o,k_v);
        fprintf(out_stream,"       are needed, %d and %d are available\n",n_occ,n_vir);
        exit(EXIT_FAILURE);
    }
    // several fragments interleave several active blocks, which is not one window to fill
    if(M->n_frag>1){
        fprintf(out_stream,"ERROR: AVAS supports a single active space (got %d fragments; N_MOL must be 1)\n",M->n_frag);
        exit(EXIT_FAILURE);
    }
    int n_frz = A.ncore;
    if((n_frz<0)||(n_frz>M->n_cor_orb)){
        fprintf(out_stream,"ERROR: $AVAS ncore=%d is out of range; accepted: 0 to %d (the CAS core size)\n",
                            n_frz,M->n_cor_orb);
        exit(EXIT_FAILURE);
    }
    int n_ob = n_occ-n_frz;

    // the reference basis is read for the selected atoms only, so an element it does not
    // carry is an error only when that atom is a target
    std::vector<int> atom_sel(M->n_atoms,0);
    for(int i_sel=0; i_sel<int(A.atoms.size()); i_sel++){
        if(A.atoms[i_sel]>M->n_atoms){
            fprintf(out_stream,"ERROR: $AVAS atom %d is out of range (the molecule has %d atoms)\n",
                                A.atoms[i_sel],M->n_atoms);
            exit(EXIT_FAILURE);
        }
        atom_sel[A.atoms[i_sel]-1]=1;
    }

    // Under $SYMM every row needs a label: the rows past ncore rotate per irrep, and every row is
    // later canonicalized per irrep, which drops an unlabelled one. The reference projector commutes
    // with the group only if each image of a target atom is a target carrying the same labels.
    if(IS_SYM){
        for(int i=0;i<M->n_mo;i++)
            if((M->rep_num[i]<0)||(M->rep_num[i]>=M->S.n_rep)){
                fprintf(out_stream,"ERROR: $AVAS under $SYMM needs an irrep label on every orbital, but MO %d has none "
                                   "(it is not symmetry-adapted); run AVAS in C1\n",i);
                exit(EXIT_FAILURE);
            }
        for(int i_sel=0; i_sel<int(A.atoms.size()); i_sel++){
            int i_a = A.atoms[i_sel]-1;
            for(int i_op=0;i_op<M->S.n_op;i_op++){
                int i_b = M->S.at_refl[i_a*M->S.n_op+i_op];
                if((atom_sel[i_b]==0)||(avas_atom_labels(A,i_a)!=avas_atom_labels(A,i_b))){
                    fprintf(out_stream,"ERROR: $AVAS atoms= / shells= are not closed under group %s: atom %d (%s) is the "
                                       "image of atom %d; select it with the same shells, or run in C1\n",
                                       M->S.group,i_b+1,M->atom_names[i_b],i_a+1);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    std::vector<int>   ref_center;
    std::vector<Shell> ref_all = basis_lib_read_gbs(M,A.ref_basis.c_str(),1,0,
                                                    nullptr,&ref_center,true,nullptr,nullptr,&atom_sel);
    std::vector<Shell> ref_s   = avas_ref_shells(M,A,ref_all,ref_center);

    int n_ref=0;
    for(auto &sh: ref_s)n_ref+=sh.size();

    fprintf(out_stream,"Reference functions:              %d\n",n_ref);
    fprintf(out_stream,"Active window:                    %d occupied + %d virtual\n\n",k_o,k_v);

    // AO_1el_from_2shells accumulates, so both overlaps start at zero
    std::vector<double> S22 (size_t(n_ref)*n_ref,0.0);
    std::vector<double> S12 (size_t(n_ao )*n_ref,0.0);
    AO_1el_from_2shells(S22.data(),ref_s,ref_s,n_ref,n_ref,'s',0);
    AO_1el_from_2shells(S12.data(),M->s ,ref_s,n_ao ,n_ref,'s',0);

    std::vector<double> S22i(S22);
    inv_matr_constr(S22i.data(),n_ref);

    std::vector<double> U_o(size_t(n_ob )*n_ob ), sig_o(n_ob );
    std::vector<double> U_v(size_t(n_vir)*n_vir), sig_v(n_vir);

    double * C_ob = M->MO_VEC+size_t(n_frz)*n_ao;

    // block labels in, eigenvector labels out (per-irrep rotation); C1 passes none
    std::vector<int> lab_o, lab_v;
    if(IS_SYM){
        lab_o.assign(M->rep_num+n_frz,M->rep_num+n_occ   );
        lab_v.assign(M->rep_num+n_occ,M->rep_num+M->n_mo);
    }
    int * l_o = IS_SYM?lab_o.data():nullptr;
    int * l_v = IS_SYM?lab_v.data():nullptr;

    avas_block_projector(C_ob                         ,n_ob ,n_ao,n_ref,S12.data(),S22i.data(),U_o.data(),sig_o.data(),l_o,M->S.n_rep);
    avas_block_projector(M->MO_VEC+size_t(n_occ)*n_ao ,n_vir,n_ao,n_ref,S12.data(),S22i.data(),U_v.data(),sig_v.data(),l_v,M->S.n_rep);

    // rotate the two row blocks in place; ascending sigma leaves the k_o selected occupieds as
    // the last occupied rows, and the reversal below leaves the k_v selected virtuals first
    std::vector<double> B(size_t(n_ob>n_vir?n_ob:n_vir)*n_ao);

    if(n_ob>0){
        cblas_dcopy(n_ob*n_ao,C_ob,1,B.data(),1);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                       n_ob,n_ao,n_ob,1.0,
                       U_o.data(),n_ob,
                       B.data(),n_ao,0.0,
                       C_ob,n_ao);
    }

    if(n_vir>0){
        cblas_dcopy(n_vir*n_ao,M->MO_VEC+size_t(n_occ)*n_ao,1,B.data(),1);
        nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                       n_vir,n_ao,n_vir,1.0,
                       U_v.data(),n_vir,
                       B.data(),n_ao,0.0,
                       M->MO_VEC+size_t(n_occ)*n_ao,n_ao);
    }

    for(int i=0;i<n_vir/2;i++){
        cblas_dswap(n_ao,M->MO_VEC+size_t(n_occ+i)*n_ao,1,M->MO_VEC+size_t(n_occ+n_vir-1-i)*n_ao,1);
        double t=sig_v[i]; sig_v[i]=sig_v[n_vir-1-i]; sig_v[n_vir-1-i]=t;
        if(IS_SYM){ int lt=lab_v[i]; lab_v[i]=lab_v[n_vir-1-i]; lab_v[n_vir-1-i]=lt; }
    }

    for(int i=0;i<n_ob ;i++)M->orb_energy[n_frz+i] = sig_o[i];
    for(int a=0;a<n_vir;a++)M->orb_energy[n_occ+a] = sig_v[a];
    if(IS_SYM){
        for(int i=0;i<n_ob ;i++)M->rep_num[n_frz+i] = lab_o[i];
        for(int a=0;a<n_vir;a++)M->rep_num[n_occ+a] = lab_v[a];
    }

    std::vector<double> sig_o_d(n_ob);
    for(int i=0;i<n_ob;i++)sig_o_d[i]=sig_o[n_ob-1-i];

    avas_print_sigma("AVAS occupied-block sigma (descending)",sig_o_d.data(),n_ob ,k_o);
    avas_print_sigma("AVAS virtual-block  sigma (descending)",sig_v  .data(),n_vir,k_v);

    if(IS_SYM){
        std::vector<int> n_so(M->S.n_rep,0), n_sv(M->S.n_rep,0);
        for(int i=0;i<k_o;i++)n_so[M->rep_num[n_occ-k_o+i]]++;
        for(int a=0;a<k_v;a++)n_sv[M->rep_num[n_occ    +a]]++;

        fprintf(out_stream,"AVAS selected orbitals by irrep\n            ");
        for(int r=0;r<M->S.n_rep;r++)fprintf(out_stream," %4s",avas_rep_name(M,r).c_str());
        fprintf(out_stream,"\n  occupied  ");
        for(int r=0;r<M->S.n_rep;r++)fprintf(out_stream," %4d",n_so[r]);
        fprintf(out_stream,"\n  virtual   ");
        for(int r=0;r<M->S.n_rep;r++)fprintf(out_stream," %4d",n_sv[r]);
        fprintf(out_stream,"\n\n");
    }

    if((k_o>0)&&(k_o<n_ob)&&(avas_max_gap(sig_o_d.data(),n_ob)!=k_o-1))
        fprintf(out_stream,"NOTE: the occupied boundary after %d orbitals is not the largest gap of its spectrum\n",k_o);
    if((k_v>0)&&(k_v<n_vir)&&(avas_max_gap(sig_v.data(),n_vir)!=k_v-1))
        fprintf(out_stream,"NOTE: the virtual boundary after %d orbitals is not the largest gap of its spectrum\n",k_v);

    // a selected orbital with sigma ~ 0 lies in the projector kernel: the reference span is
    // exhausted and its identity is arbitrary within the block
    int nz_o=0,nz_v=0;
    for(int i=0;i<k_o;i++)if(sig_o_d[i]>1e-8)nz_o++;
    for(int a=0;a<k_v;a++)if(sig_v  [a]>1e-8)nz_v++;
    if(nz_o<k_o)
        fprintf(out_stream,"NOTE: only %d of the %d selected occupied orbitals overlap the reference "
                           "(%d functions); the rest are arbitrary within the occupied space\n",nz_o,k_o,n_ref);
    if(nz_v<k_v)
        fprintf(out_stream,"NOTE: only %d of the %d selected virtual orbitals overlap the reference "
                           "(%d functions); the rest are arbitrary within the virtual space\n",nz_v,k_v,n_ref);

    if(A.canonicalize){
        avas_canonicalize(M, n_frz);
        fprintf(out_stream,"\n");
        printf_timer("AVAS Fock canonicalization");
    }

    // A rotation within one irrep keeps the block's off-irrep weight but can gather it in one row,
    // so every label is re-certified by the orbital reader's own test, judged on rep_num.
    if(IS_SYM){
        std::vector<int> lab(M->rep_num,M->rep_num+M->n_mo);
        M->check_orb_symmetry();
        for(int i=0;i<M->n_mo;i++)
            if(M->rep_num[i]!=lab[i]){
                fprintf(out_stream,"ERROR: AVAS orbital %d lost its irrep label (%s -> %s); the input orbitals are not "
                                   "adapted tightly enough, run AVAS in C1\n",i,avas_rep_name(M,lab[i]).c_str(),
                                   M->rep_num[i]<0?"none":avas_rep_name(M,M->rep_num[i]).c_str());
                exit(EXIT_FAILURE);
            }
    }

    M->MO_gamess_format();

    char name[BUF_LINE_LENGTH];

    fprintf(out_stream,"\n");
    fprintf(out_stream,"Writing AVAS orbitals:\n");

    sprintf(name,"%s_AVAS.out",job_name);
    M->GAMESS_type_out_print(name,-1);
    fprintf(out_stream,"visualization file: %s\n",name);

    sprintf(name,"%s_AVAS.orb",job_name);
    M->MO_print(name);
    fprintf(out_stream,"data file         : %s\n",name);

    fprintf(out_stream,"\n");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");

    return 0;
}
