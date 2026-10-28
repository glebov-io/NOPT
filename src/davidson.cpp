
//doCI
# include "blas_link.h"
# include "davidson.h"
# include "matr.h"
# include "timer.h"
# include "common_vars.h"
# include "etc.h"
# include "spin_projector.h"
# include <memory>



inline double ED_with_shift(double E, double edshift){
    
        return E/(E*E+edshift);
        
}


davidson_solver::davidson_solver(){
    
    H     = NULL;
    E     = NULL;
    E_old = NULL;
    x     = NULL;
    r     = NULL;
    Hx    = NULL;
    V_m   = NULL;
    spin_proj = 0;
    
}

int davidson_solver::set_par(aldet_data * ext_C, dav_par dav){
    
    // 1. copy external variables
    C         = ext_C        ;
    max_n_vec = dav.n_s;
    n_bf      = dav.n_bf;
    max_it    = dav.max_it;
    e_conv    = dav.e_conv;
    r_conv    = dav.r_conv;
    se_min    = dav.se_min;
    edshift   = dav.edshift;
    sparsed_Hc = dav.sparsed_Hc;
    spin_proj  = dav.spin_proj;
    
    max_dim   = max_n_vec+n_bf+200;
    
    if(n_bf==0){
        fprintf(out_stream,"write proper davidson parameters!!!!!!!!!\n");
        exit(1);
    }
    
    
    // 2. initialize internal aldet data
    // 2.a. initialize
    V.get_dim(C->n_act, C->na, C->nb, 1, C->mult, C->print_number);
    V.init_zero_vec(max_n_vec,0);
    // 2.b set references
    V.simple_import_data(C->act_INTS_AA, C->act_INTS_AB, C->F_act_A, C->E_core);
    V.U_simple_import_data(C->act_INTS_AA,
                           C->act_INTS_AB,
                           C->act_INTS_BB,
                           C->F_act_A,
                           C->F_act_B,
                           C->E_core);
    
    if(C->do_PT){
        V.UPT2_import_data(C->T3_AAA,
                           C->T3_AAB,
                           C->T3_BBA,
                           C->T3_BBB,
                           C->T2_AA,
                           C->T2_AB,
                           C->T2_BB,
                           C->T1_A,
                           C->T1_B,
                           C->T0);
    }
    
    if(LINEAR){
        V.Lambda_act  =C->Lambda_act;
        V.Lambda_core=C->Lambda_core;
        V.act_rep_num =C->act_rep_num;
    }
    
    
    // 2.c. calc diagonal
//     V.H_diag_calc();
    
    // 3. allocate memory
    H     = new_double_w_check(1LL*max_dim*max_dim    ,"CI effective matrix");
    E     = new_double_w_check(1LL*max_dim            ,"CI eigenvalue");
    E_old = new_double_w_check(1LL*max_dim            ,"CI eigenvalue (backup)");
    x     = new_double_w_check(1LL*V.Nd*C->n_states[0],"CI vector");
    r     = new_double_w_check(1LL*V.Nd*C->n_states[0],"CI residue vector");
    Hx    = new_double_w_check(1LL*V.Nd*C->n_states[0],"CI {H x vector} product");
    
    
    return 0;
    
}


int davidson_solver::set_par_m(aldet_data * ext_C, dav_par dav, int ext_n_CI){
    
    // 1. copy external variables
    C         = ext_C        ;
    max_n_vec = dav.n_s;
    n_bf      = dav.n_bf;
    max_it    = dav.max_it;
    e_conv    = dav.e_conv;
    r_conv    = dav.r_conv;
    se_min    = dav.se_min;
    edshift   = dav.edshift;
    
    max_dim   = max_n_vec;
    
    if(n_bf==0){
        fprintf(out_stream,"write proper davidson parameters!!!!!!!!!\n");
        exit(1);
    }
    
    
    // 2. initialize internal aldet data
    // 2.a. initialize
    n_CI=ext_n_CI;
    V_m= new aldet_data[n_CI];
    for(int i=0;i<n_CI;i++){
        V_m[i].get_dim(C[i].n_act, C[i].na, C[i].nb, 1, C[i].mult, C[i].print_number);
        V_m[i].init_zero_vec(max_n_vec,0);
        // 2.b set references
        V_m[i].simple_import_data(C[i].act_INTS_AA, C[i].act_INTS_AB, C[i].F_act_A, C[i].E_core);
        V_m[i].U_simple_import_data(C[i].act_INTS_AA,
                               C[i].act_INTS_AB,
                               C[i].act_INTS_BB,
                               C[i].F_act_A,
                               C[i].F_act_B,
                               C[i].E_core);
        
        if(C[i].do_PT){
            V_m[i].UPT2_import_data(C[i].T3_AAA,
                               C[i].T3_AAB,
                               C[i].T3_BBA,
                               C[i].T3_BBB,
                               C[i].T2_AA,
                               C[i].T2_AB,
                               C[i].T2_BB,
                               C[i].T1_A,
                               C[i].T1_B,
                               C[i].T0);
        }
        
        if(LINEAR){
            V_m[i].Lambda_act  =C[i].Lambda_act;
            V_m[i].Lambda_core=C[i].Lambda_core;
            V_m[i].act_rep_num =C[i].act_rep_num;
        }
        
        
        // 2.c. calc diagonal
        V_m[i].H_diag_calc();
    }
    // 3. allocate memory
    H     = new double[1LL*max_dim*max_dim*n_CI];
    E     = new double[1LL*max_dim*n_CI];
    E_old = new double[1LL*max_dim*n_CI];
    
    int N=0;
    for(int i=0;i<n_CI;i++)N+=V_m[i].Nd*C[i].n_states[0];
    
    x     = new_double_w_check(1LL*N,"CI vector");
    r     = new_double_w_check(1LL*N,"CI residue vector");
    Hx    = new_double_w_check(1LL*N,"CI {H x vector} product");
    
    
    return 0;
    
}

int davidson_solver::set_par_cis(CIS_engine * ext_cis, dav_par dav){
    
    // 1. copy external variables
    C         = NULL        ;
    cis       = ext_cis;
    max_n_vec = 50;//dav.n_s;
    n_bf      = dav.n_bf;
    max_it    = dav.max_it;
    e_conv    = dav.e_conv;
    r_conv    = dav.r_conv;
    se_min    = dav.se_min;
    edshift   = dav.edshift;
    
    max_dim   = max_n_vec+n_bf+200;
    
    if(n_bf==0){
        fprintf(out_stream,"write proper davidson parameters!!!!!!!!!\n");
        exit(1);
    }
    
    
    // 3. allocate memory
    H     = new double[1LL*max_dim*max_dim];
    E     = new double[1LL*max_dim];
    E_old = new double[1LL*max_dim];
    x     = new double[1LL*cis->n_c*cis->n_v*cis->n_s];
    r     = new double[1LL*cis->n_c*cis->n_v*cis->n_s];
    Hx    = new double[1LL*cis->n_c*cis->n_v*cis->n_s];
    
    
    return 0;
    
}


int davidson_solver::read_guess(){
    
    int n_s = C->n_states[0];
    
    int * gues_num;
    gues_num = new int[n_s];
    
    set_zero_matr(V.coef[0],max_n_vec*V.Nd);
    for(int i=0;i<V.Nd;i++)
    for(int j=0;j<n_s ;j++)
        V.coef[0][i*max_n_vec+j]=C->coef[0][i*n_s+j];
    V.transpose_ci(0);
    
    delete[]gues_num;
    
    return 0;
    
}

extern double * H_ci;
int davidson_solver::run(int print, int read_states){
    
//     print=1;
    if(print){
        fprintf(out_stream,"\n\n");
        fprintf(out_stream,"____________________Starting_Davidson_diagonalization__________________\n");
    }
    int t_H_lb=1;
    if(sparsed_Hc)if(V.do_PT==0)t_H_lb=0;
    
#ifdef _OPENBLAS
    openblas_set_num_threads(num_threads);
#endif    
    sparsed_CI_vec *   lb = new sparsed_CI_vec[n_bf+200]; 
    sparsed_CI_vec * H_lb = new sparsed_CI_vec[n_bf+200]; 

    V.gen_ext_ind();
    if(V.do_PT)V.PT_update();
    
    std::unique_ptr<spin_projector> sp;
    if(spin_proj){
        sp.reset(new spin_projector(V));
        if(!sp->ok){
            fprintf(out_stream,"ERROR: spin_proj: mult=%d (S=%.1f) does not exist for n_alp=%d, n_bet=%d in %d orbitals\n",
                    V.mult, sp->S, V.na, V.nb, V.n_act);
            exit(1);
        }
        if(print){
            fprintf(out_stream,"Spin projection onto S = %.1f (Ms = %.1f):", sp->S, (V.na - V.nb) / 2.0);
            if(sp->parity) fprintf(out_stream," alpha-beta parity %+.0f,", sp->sym_ab);
            if(sp->remove.empty()) fprintf(out_stream," no other spin left\n");
            else{
                fprintf(out_stream," removing S' =");
                for(double r : sp->remove) fprintf(out_stream," %.1f", 0.5 * (sqrt(1 + 4 * r) - 1));
                fprintf(out_stream,"\n");
            }
        }
    }
    
    int n = V.gen_bf(lb,n_bf);
    double * H_lb_d = nullptr;
    
    if(t_H_lb==0)
        V.H_mult_sparsed_to_sparsed(H_lb, lb, n);
    else{
        H_lb_d = new_double_w_check(1LL*n*V.Na*V.Nb,"dense form of {H x bf} products");
        V.H_mult_sparsed_to_dense(H_lb_d, lb, n);
    }
    int n_s = C->n_states[0];
    if(sp && n_s > sp->n_states){
        fprintf(out_stream,"ERROR: spin_proj: %d roots requested, but %d orbitals with n_alp=%d, n_bet=%d hold only %.0f states with S=%.1f\n",
                n_s, V.n_act, V.na, V.nb, sp->n_states, sp->S);
        exit(1);
    }
    int dim=n;    // current dimension (k)
    int n0=0;
    
    
    
    max_dim=max_n_vec+n;
    
    set_zero_matr(H,max_dim*max_dim);
    set_zero_matr(E_old,max_dim);
    
    int n_step=-1; // step number
    int conv_num;  // number of converged roots
    int conv_num_prev=0;  // number of converged roots
    

    double * S = new double[1LL*max_dim*max_dim];
    double * B = new double[1LL*max_dim*max_dim];
    double * c = new double[1LL*max_dim*max_dim];
    double * norm = new double [n_s];
    
    
    
    double * H_bf = new double[1LL*n*n];
    double * S_bf = new double[1LL*n*n];
    set_zero_matr(H_bf, n*n);
    set_zero_matr(S_bf, n*n);
    


    if(t_H_lb==0)
        CI_ss_mult(H_bf, n, lb, n, H_lb, n);
    else
        CI_sd_mult(H_bf, n, lb, n, H_lb_d, n, n);

    for(int i=0;i<n;i++)
        S_bf[i*n+i]=1.0;
    
    
//     if(V.do_PT){
//         printf("H_CI:\n");
//         PrintMatr10(H_bf,n,n,0);
//     }
 
    if(print==-1){
        H_ci = new double[1LL*n*n];
        memcpy(H_ci, H_bf, n*n*sizeof(double));
        return n;
    }
    
    double max_asym=0;
    double     asym  ;
    
    if(read_states)/*if((V.Na==V.Nb)||(V.mult!=0))*/{
        
        for(int i=0;i<V.Nd;i++)
        for(int j=0;j<n_s;j++)
            V.coef[0][i*max_n_vec+j]=C->coef[0][i*n_s+j];
        if(sp) sp->project(V.coef[0], n_s, max_n_vec);
        
        dim+=n_s;
        V.transpose_ci(0);
    }
    
    double d_max;
    
    while(true){
        
        n_step++;
        
        
        if(n_step==max_it){
            fprintf(out_stream,"ERROR: davidson diagonalization didn't converge after %d iterations\n", max_it);
            fprintf(out_stream,"       this means there are some problems with your CI\n");
            fprintf(out_stream,"       if you believe it is not true try to increase \"dav_it\" parameter\n");
            exit(1);
        }
        
        set_zero_matr(H,dim*dim);
        set_zero_matr(S,dim*dim);
        
        for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            H[i*dim+j]=H_bf[i*n+j];
        
        for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            S[i*dim+j]=S_bf[i*n+j];
        
//         printf_timer("copy H_S");
    
        if(dim>n){
            V.H_mult(n0,dim-n);
            
//             printf_timer("H_mult");
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                                 dim-n,dim-n,V.Nd,1.0,
                                 V.coef[0],max_n_vec,
                                 V.Hv,max_n_vec,0.0,
                                 H + (dim+1)*n ,dim);
            
            CI_sd_mult(H + n, dim, lb, n, V.Hv, dim-n, max_n_vec);
            
            transpose_A_to_B(H + dim*n, H + n, dim-n, n,dim,dim);
            
            
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                                 dim-n,dim-n,V.Nd,1.0,
                                 V.coef[0],max_n_vec,
                                 V.coef[0],max_n_vec,0.0,
                                 S + (dim+1)*n ,dim);
            
            CI_sd_mult(S + n, dim, lb, n, V.coef[0], dim-n, max_n_vec);
            transpose_A_to_B(S + dim*n, S + n, dim-n, n,dim,dim);
        }

        HC_SCE_p(H,S,c,B,E,dim,se_min);
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                             V.Nd,n_s,dim-n,1.0,
                             V.coef[0],max_n_vec,
                             c+n,dim,0.0,
                             x,n_s);
        
        CI_sd_mult_tr(x, n_s, lb, n, c, n_s, dim);
        
        
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                             V.Nd,n_s,dim-n,1.0,
                             V.Hv,max_n_vec,
                             c+n,dim,0.0,
                             Hx,n_s);
        if(t_H_lb==0)
            CI_sd_mult_tr(Hx, n_s, H_lb, n, c, n_s, dim);
        else
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                             V.Nd,n_s,n,1.0,
                             H_lb_d,n,
                             c,dim,1.0,
                             Hx,n_s);
        
            
        for(int i=0;i<V.Nd;i++)
        for(int j=0;j<n_s;j++)
            r[i*n_s+j]=E[j]*x[i*n_s+j]-Hx[i*n_s+j];
        
            
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                             n_s,n_s,V.Nd,1.0,
                             r,n_s,
                             r,n_s,0.0,
                             H ,n_s);

        if(print){
            fprintf(out_stream,"\n\nStep %d\n",n_step);
            fprintf(out_stream,"_________________________________________________\n");
            fprintf(out_stream,"  n | E                 | dE         | norm(r)   |\n");
            fprintf(out_stream,"____|___________________|____________|___________|\n");
            for(int i=0;i<n_s;i++) fprintf(out_stream,"%3d |% 18.10f | % .3e | %.3e |\n",i,E[i],E[i]-E_old[i],H[i*n_s+i]);
            fprintf(out_stream,"____|___________________|____________|___________|\n");
        
            fprintf(out_stream,"number of corr vectors=%d\n", dim);
        }
        
        conv_num=0;
        for(int i=0;i<n_s;i++)if((H[i*n_s+i]<r_conv)||(fabs(E[i]-E_old[i])<e_conv))conv_num++;
        if(print)fprintf(out_stream,"%d roots converged at step %d\n",conv_num,n_step);
        if(conv_num==n_s){
            for(int i=0;i<V.Nd;i++)
                for(int j=0;j<n_s;j++)
                    V.coef[0][i*max_n_vec+j]=x[i*n_s+j];
                
            V.transpose_ci(0);
            break;
        }
        
        n0=dim-n;
        if(n_step==0)/*if(read_states==0)*/{
            for(int i=0;i<V.Nd;i++)
                 for(int j=0;j<n_s;j++)
                     V.coef[0][i*max_n_vec+j]=x[i*n_s+j];
            n0=0;
            dim=n_s+n;
        }
        
        for(int i=0;i<V.Nd;i++)
        for(int j=0;j<n_s ;j++){
            r[i*n_s+j]=r[i*n_s+j]*ED_with_shift(E[j]-V.H_diag_appr[i], edshift);
//             norm[j]+=r[i*n_s+j]*r[i*n_s+j];
        }
        
        for(int j=0;j<n_s ;j++)norm[j]=0;
        for(int i=0;i<V.Nd;i++)
        for(int j=0;j<n_s ;j++){
//             r[i*n_s+j]=r[i*n_s+j]*ED_with_shift(E[j]-V.H_diag_appr[i], edshift);
            norm[j]+=r[i*n_s+j]*r[i*n_s+j];
        }

        for(int j=0;j<n_s ;j++)norm[j]=sqrt(norm[j]);
        
        
        
        int new0 = dim-n; // first new expansion vector
        if(dim+n_s-conv_num<max_dim+1){
            for(int i=0;i<V.Nd;i++)
                for(int j=0, d=dim-n;j<n_s;j++)
                    if((H[j*n_s+j]>r_conv)&&(fabs(E[j]-E_old[j])>e_conv)){
                        V.coef[0][i*max_n_vec+d]=r[i*n_s+j]/norm[j];;
                        d++;
                    }
            dim+=(n_s-conv_num);
        }
        else{
            for(int i=0;i<V.Nd;i++)
                for(int j=0;j<n_s;j++)
                    V.coef[0][i*max_n_vec+j]=x[i*n_s+j];
            
            for(int i=0;i<V.Nd;i++)
                for(int j=0, d=n_s;j<n_s;j++)
                    if((H[j*n_s+j]>r_conv)&&(fabs(E[j]-E_old[j])>e_conv)){
                        V.coef[0][i*max_n_vec+d]=r[i*n_s+j]/norm[j];;
                        d++;
                    }
            n0=0;
            dim=2*n_s-conv_num+n;
            new0=n_s;
        }
//         printf_timer("new V");
        if(sp && dim-n > new0){
            sp->project(V.coef[0] + new0, dim-n-new0, max_n_vec);
            for(int j=new0;j<dim-n;j++){
                double nn=0;
                for(int i=0;i<V.Nd;i++) nn+=V.coef[0][i*max_n_vec+j]*V.coef[0][i*max_n_vec+j];
                nn = nn > 1e-16 ? 1.0/sqrt(nn) : 0.0; // a residual of the other spins only: drop it (S05 skips it)
                for(int i=0;i<V.Nd;i++) V.coef[0][i*max_n_vec+j]*=nn;
            }
        }
        V.transpose_ci(0);
        if(print){
            if(V.Na==V.Nb)if(V.mult!=0){
                d_max=sym_check_for_tensor(V.coef[0], V.Na,dim-n,V.n_states[0], -V.sym_ab);
                fprintf(out_stream,"maximum deviation from symmetric form - %e\n",d_max);
                
            }
        }
        else{
            d_max=0;
            for(int i=0;i<n_s;i++)
                if(d_max<fabs(E[i]-E_old[i]))d_max=fabs(E[i]-E_old[i]);
            // fprintf(stderr,"\rdavidson iter %3d, dE=% .3e",n_step,d_max);
            d_max=0;
            for(int i=0;i<n_s;i++)
                if(d_max<H[i*n_s+i])d_max=H[i*n_s+i];
            // fprintf(stderr,", dR=%.3e",d_max);
            if(V.Na==V.Nb)if(V.mult!=0){
                d_max=sym_check_for_tensor(V.coef[0], V.Na,dim-n,V.n_states[0], -V.sym_ab);
                // fprintf(stderr,", m.d.s.=%.3e",d_max);
                
            }
            
        }
        memcpy(E_old,E,sizeof(double)*n_s);
        fflush(out_stream);
        
    }
    // fprintf(stderr,"\r                                                                \r");
    
    
    if(print){
        for(int i=0; i<n_s; i++)V.E_states[0][i]=E[i];
        fprintf(out_stream,"\n\nCAS_CI WaveFunctions:\n\n");
    }
    V.print_states(0,n_s,print);
    
    fflush(out_stream);
    
    delete[] lb;
    delete[] H_lb;
    if(H_lb_d!=nullptr)delete[] H_lb_d;
    
    delete[] norm;
    delete[] S    ;
    delete[] B    ;
    delete[] c    ;
    delete[] H_bf ;
    delete[] S_bf ;
    
    //import roots to the external storage
    if(sp){
        sp->project(x, n_s, n_s);
        if(print) fprintf(out_stream,"Spin projection: max |(S^2 - S(S+1)) c| over the roots = %.1e\n", sp->residual(x, n_s, n_s));
    }
    for(int i=0;i<V.Nd;i++)
        for(int j=0;j<n_s;j++)
            C->coef[0][i*n_s+j]=x[i*n_s+j];
    
    for(int i=0;i<n_s;i++)
        C->E_states[0][i]=E[i];
    for(int i=0;i<n_s;i++)
        C->S2[0][i]=V.S2[0][i];
    if(LINEAR)
    for(int i=0;i<n_s;i++)
        C->L2[0][i]=V.L2[0][i];
    if(LINEAR)
    for(int i=0;i<n_s;i++)
        C->P[0][i]=V.P[0][i];
        

    // C->symmetrization(0);
    C->transpose_ci(0);
    
    
    if(print){
        fprintf(out_stream,"\n\n");
        printf_timer("Davidson diagonalization");
        fprintf(out_stream,"_______________________________________________________________________\n");
    }
//     exit(0);
    
    return n_step;
}

int davidson_solver::run_cis(int print, int read_states){
    
//     print=1;
    if(print){
        fprintf(out_stream,"\n\n");
        fprintf(out_stream,"____________________Starting_Davidson_diagonalization__________________\n");
    }
    
#ifdef _OPENBLAS
    openblas_set_num_threads(num_threads);
#endif    

    int n =0;//n_bf;
    cis->gen_bf();
    cis->calc_H_CIS();

    int n_s = cis->n_s;// full number of roots
    int dim=n_s;//n;    // current dimension (k)
    int n_ci=cis->n_c*cis->n_v;
    
    
    max_dim=max_n_vec+n;
    
    set_zero_matr(H,max_dim*max_dim);
    set_zero_matr(E_old,max_dim);
    
    int n_step=-1; // step number
    int conv_num;  // number of converged roots
    int conv_num_prev=0;  // number of converged roots
    

    double * S = new double[1LL*max_dim*max_dim];
    double * B = new double[1LL*max_dim*max_dim];
    double * c = new double[1LL*max_dim*max_dim];
    double * norm = new double [n_s];
    
    
    cis->gen_H();
        
    double d_max;
    
    cis->calc_evec();
    
    while(true){
        
        n_step++;
        
        cis->H_mult(dim);
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                             dim,dim,n_ci,1.0,
                             cis->C,n_ci,
                             cis->d,n_ci,0.0,
                             H,dim);
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                             dim,dim,n_ci,1.0,
                             cis->C,n_ci,
                             cis->C,n_ci,0.0,
                             S,dim);
        
//         PrintMatr(H,dim,dim,0);
        
        HC_SCE_p(H,S,c,B,E,dim,se_min);
        
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                             n_s,n_ci,dim,1.0,
                             c,dim,
                             cis->C,n_ci,0.0,
                             x,n_ci);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                             n_s,n_ci,dim,1.0,
                             c,dim,
                             cis->d,n_ci,0.0,
                             Hx,n_ci);
        
        for(int i=0;i<n_ci;i++)
            for(int j=0;j<n_s;j++)
                r[i*n_s+j]=E[j]*x[j*n_ci+i]-Hx[j*n_ci+i];
        
            
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                             n_s,n_s,n_ci,1.0,
                             r,n_s,
                             r,n_s,0.0,
                             H ,n_s);

        if(print){
            fprintf(out_stream,"\n\nStep %d\n",n_step);
            fprintf(out_stream,"_________________________________________________\n");
            fprintf(out_stream,"  n | E                 | dE         | norm(r)   |\n");
            fprintf(out_stream,"____|___________________|____________|___________|\n");
            for(int i=0;i<n_s;i++) fprintf(out_stream,"%3d |% 18.10f | % .3e | %.3e |\n",i,E[i],E[i]-E_old[i],H[i*n_s+i]);
            fprintf(out_stream,"____|___________________|____________|___________|\n");
        
            fprintf(out_stream,"number of corr vectors=%d\n", dim);
        }
        
        conv_num=0;
        
        for(int i=0;i<n_s;i++)if((H[i*n_s+i]<r_conv)/*||(fabs(E[i]-E_old[i])<e_conv)*/)conv_num++;
        
        if(print)fprintf(out_stream,"%d roots converged at step %d\n",conv_num,n_step);
        if(conv_num==n_s){
            for(int i=0;i<n_s;i++)
            for(int j=0;j<n_ci ;j++)
                cis->C[i*n_ci+j]=x[i*n_ci+j];
                
            break;
        }
        
        for(int i=0;i<n_ci;i++)
        for(int j=0;j<n_s ;j++){
            r[i*n_s+j]=r[i*n_s+j]*ED_with_shift(E[j]-cis->E_app[i], edshift);
        }
        
        for(int j=0;j<n_s ;j++)norm[j]=0;
        
        for(int i=0;i<n_ci;i++)
        for(int j=0;j<n_s ;j++){
            norm[j]+=r[i*n_s+j]*r[i*n_s+j];
        }

        for(int j=0;j<n_s ;j++)norm[j]=sqrt(norm[j]);
        
        
        
        if(dim+n_s-conv_num<max_dim+1){
            for(int i=0;i<n_ci;i++)
                for(int j=0, d=dim/*-n*/;j<n_s;j++)
                    if((H[j*n_s+j]>r_conv)&&(fabs(E[j]-E_old[j])>e_conv)){
                        cis->C[d*n_ci+i]=r[i*n_s+j]/norm[j];
                        d++;
                    }
            dim+=(n_s-conv_num);
        }
        else{
            for(int i=0;i<n_s;i++)
            for(int j=0;j<n_ci ;j++)
                cis->C[i*n_ci+j]=x[i*n_ci+j];
            
            for(int i=0;i<n_ci;i++)
                for(int j=0, d=n_s;j<n_s;j++)
                    if((H[j*n_s+j]>r_conv)&&(fabs(E[j]-E_old[j])>e_conv)){
                        cis->C[d*n_ci+i]=r[i*n_s+j]/norm[j];;
                        d++;
                    }
            dim=2*n_s-conv_num+n;
        }

        if(print==0){
            d_max=0;
            for(int i=0;i<n_s;i++)
                if(d_max<fabs(E[i]-E_old[i]))d_max=fabs(E[i]-E_old[i]);
            // fprintf(stderr,"\rdavidson iter %3d, dE=% .3e",n_step,d_max);
            d_max=0;
            for(int i=0;i<n_s;i++)
                if(d_max<H[i*n_s+i])d_max=H[i*n_s+i];
            // fprintf(stderr,", dR=%.3e",d_max);
                        
        }
        memcpy(E_old,E,sizeof(double)*n_s);
        fflush(out_stream);
        
    }
    // fprintf(stderr,"\r                                                                \r");
    
    fflush(out_stream);
    
    delete[] norm;
    delete[] S    ;
    delete[] B    ;
    delete[] c    ;
    
    
    if(print){
        fprintf(out_stream,"\n\n");
        printf_timer("Davidson diagonalization");
        fprintf(out_stream,"_______________________________________________________________________\n");
    }

    for (int i=0; i<n_s; i++) // Maksim
    {
        cis->E_app[i] = E[i];
    }
    
    return 0;//n_step;
}




int davidson_solver::run_m(int print, int read_states){
    
// //     print=1;
    if(print){
        fprintf(out_stream,"\n\n");
        fprintf(out_stream,"_______________Starting_Davidson_multiple_diagonalization______________\n");
    }
    int n_s =1;// C->n_states[0];
    int * dim=new int[n_CI];
    for(int i=0;i<n_CI;i++)dim[i]=n_s;
    
    int * n0=new int[n_CI];
    for(int i=0;i<n_CI;i++)n0[i]=0;
    
    int Nd=V_m[0].Nd;
    
    double ** H_m     = new double*[n_CI];
    double ** E_m     = new double*[n_CI];
    double ** E_old_m = new double*[n_CI];
    double ** x_m     = new double*[n_CI];
    double ** r_m     = new double*[n_CI];
    double ** Hx_m    = new double*[n_CI];
    
    H_m    [0] = H    ;for(int i=0;i<n_CI-1;i++) H_m    [i+1]=H_m    [i]+max_dim*max_dim;
    E_m    [0] = E    ;for(int i=0;i<n_CI-1;i++) E_m    [i+1]=E_m    [i]+max_dim;
    E_old_m[0] = E_old;for(int i=0;i<n_CI-1;i++) E_old_m[i+1]=E_old_m[i]+max_dim;
    x_m    [0] = x    ;for(int i=0;i<n_CI-1;i++) x_m    [i+1]=x_m    [i]+Nd*n_s;
    r_m    [0] = r    ;for(int i=0;i<n_CI-1;i++) r_m    [i+1]=r_m    [i]+Nd*n_s;
    Hx_m   [0] = Hx   ;for(int i=0;i<n_CI-1;i++) Hx_m   [i+1]=Hx_m   [i]+Nd*n_s;


//     set_zero_matr(H,max_dim*max_dim*n_CI);
    set_zero_matr(E_old,max_dim*n_CI);
    
    int n_step=-1; // step number
    int conv_num;  // number of converged roots
    int conv_num_prev=0;  // number of converged roots
    int conv_CI=0;
    for(int i=0;i<n_CI;i++){
        printf("\n\nset %d:\n", i);
        C[i].print_states(0,n_s,1);
    }
    
    double * S = new double[1LL*max_dim*max_dim];//delete HC_SCE_p!!
    double * B = new double[1LL*max_dim*max_dim];
    double * c = new double[1LL*max_dim*max_dim];
    double * v_ort = new double[1LL*V_m[0].Nd*n_CI];
    double * norm = new double [max_dim];
//  
    for(int i=0;i<n_CI;i++)V_m[i].gen_ext_ind();;
    for(int i=0;i<n_CI;i++)if(V_m[i].do_PT)V_m[i].PT_update();
    
    
    for(int i=0;i<n_CI            ;i++)
    for(int j=0;j<Nd       ;j++)
    for(int k=0;k<n_s;k++)
        V_m[i].coef[0][j*max_n_vec+k]=C[i].coef[0][j*n_s+k];
    
    for(int i=0;i<n_CI;i++)
    for(int j=0;j<Nd  ;j++)
        v_ort         [i*Nd+j]=C[i].coef[0][j];
   

//     for(int i=0;i<n_CI;i++){
//         printf("\n\nset %d:\n", i);
//         V_m[i].print_states(0,n_s,1);
//     }

    
    for(int i=0;i<n_CI;i++)V_m[i].transpose_ci(0);
    
    double d_max;
    double d_max_tmp;
    double E_max;
    double dR_max;
    
    while(true){
        
        d_max=0;
        E_max=0;
        dR_max=0;
        conv_CI=0;
        n_step++;
        
        
        if(n_step==max_it){
            fprintf(out_stream,"ERROR: davidson diagonalization didn't converge after %d iterations\n", max_it);
        
            fprintf(out_stream,"       this means there are some problems with your CI\n");
            fprintf(out_stream,"       if you believe it is not true try to increase \"dav_it\" parameter\n");
            exit(1);
        }
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_CI,n_CI,Nd,1.0,
                        v_ort,Nd,
                        v_ort,Nd,0.0,
                        S,n_CI);
        
        if(print){
            fprintf(out_stream,"\n\nStep %d\n",n_step);
            fprintf(out_stream,"___________________________________________________________________\n");
            fprintf(out_stream,"  n | E                 | dE         | norm(r)   | n_conv | n_corr |\n");
            fprintf(out_stream,"____|___________________|____________|___________|________|________|\n");
        }
        
        for(int i_CI=0;i_CI<n_CI;i_CI++){
            
            
            for(int j=0;j<dim[i_CI];j++)
            for(int i=0;i<n_CI;i++)if(i!=i_CI)
            {
                double c=0;
                for(int k=0;k<Nd;k++)
                    c+=V_m[i_CI].coef[0][k*max_n_vec+j]*v_ort[i*Nd+k];
                
                for(int k=0;k<Nd;k++)
                    V_m[i_CI].coef[0][k*max_n_vec+j]-=c*v_ort[i*Nd+k];
            }
            
            for(int j=0;j<dim[i_CI];j++)norm[j]=0;
            for(int i=0;i<Nd;i++)
            for(int j=0;j<dim[i_CI] ;j++){
                norm[j]+=V_m[i_CI].coef[0][i*max_n_vec+j]*V_m[i_CI].coef[0][i*max_n_vec+j];
            }
//             for(int j=0;j<dim[i_CI];j++)printf("n=%e\n",norm[j]);
        
            for(int j=0;j<dim[i_CI] ;j++)norm[j]=sqrt(norm[j]);
            
            for(int i=0;i<Nd;i++)
            for(int j=0;j<dim[i_CI] ;j++){
                V_m[i_CI].coef[0][i*max_n_vec+j]=V_m[i_CI].coef[0][i*max_n_vec+j]/norm[j];
            }
            V_m[i_CI].transpose_ci(0);
            
            
            
            set_zero_matr(H_m[i_CI],dim[i_CI]*dim[i_CI]);
            set_zero_matr(S        ,dim[i_CI]*dim[i_CI]);
//             for(int i=0;i<dim[i_CI];i++)S[i*(dim[i_CI]+1)]=1.0;//delete HC_SCE_p!!
            
            V_m[i_CI].H_mult(0,dim[i_CI]);
            
//             printf_timer("H_mult");
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                        dim[i_CI],dim[i_CI],Nd,1.0,
                        V_m[i_CI].coef[0],max_n_vec,
                        V_m[i_CI].Hv,max_n_vec,0.0,
                        H_m[i_CI],dim[i_CI]);
            
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                        dim[i_CI],dim[i_CI],Nd,1.0,
                        V_m[i_CI].coef[0],max_n_vec,
                        V_m[i_CI].coef[0],max_n_vec,0.0,
                        S,dim[i_CI]);
            
           
            HC_SCE_p(H_m[i_CI],S,c,B,E_m[i_CI],dim[i_CI],se_min);//delete HC_SCE_p!!
                        
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                             Nd,n_s,dim[i_CI],1.0,
                             V_m[i_CI].coef[0],max_n_vec,
                             c,dim[i_CI],0.0,
                             x_m[i_CI],n_s);
            
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                             Nd,n_s,dim[i_CI],1.0,
                             V_m[i_CI].Hv,max_n_vec,
                             c,dim[i_CI],0.0,
                             Hx_m[i_CI],n_s);
            
            
            for(int i=0;i<Nd  ;i++)
                v_ort         [i_CI*Nd+i]=x_m[i_CI][i/**n_s+j*/];
        
            for(int i=0;i<Nd;i++)
            for(int j=0;j<n_s;j++)
                r_m[i_CI][i*n_s+j]=E_m[i_CI][j]*x_m[i_CI][i*n_s+j]-Hx_m[i_CI][i*n_s+j];
            
            
            for(int j=0;j<n_s;j++)
            for(int i=0;i<n_CI;i++)if(i!=i_CI)
            {
                double c=0;
                for(int k=0;k<Nd;k++)
                    c+=r_m[i_CI][k*n_s+j]*v_ort[i*Nd+k];
                
                for(int k=0;k<Nd;k++)
                    r_m[i_CI][k*n_s+j]-=c*v_ort[i*Nd+k];
            }
            
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                             n_s,n_s,Nd,1.0,
                             r_m[i_CI],n_s,
                             r_m[i_CI],n_s,0.0,
                             H_m[i_CI],n_s);
            
            conv_num=0;
            for(int i=0;i<n_s;i++)if((H_m[i_CI][i*n_s+i]<r_conv)||(fabs(E_m[i_CI][i]-E_old_m[i_CI][i])<e_conv))conv_num++;
            if(print)
            
            if(print){
                for(int i=0;i<n_s;i++) fprintf(out_stream,"%3d |% 18.10f | % .3e | %.3e |        |        |\n",i,E_m[i_CI][i],E_m[i_CI][i]-E_old_m[i_CI][i],H_m[i_CI][i*n_s+i]);
                fprintf(out_stream,"----|-------------------|------------|-----------|  %4d  |  %4d  |\n",conv_num,dim[i_CI]);
            }
            
            if(conv_num==n_s){
                for(int i=0;i<Nd;i++)
                    for(int j=0;j<n_s;j++)
                        V_m[i_CI].coef[0][i*max_n_vec+j]=x_m[i_CI][i*n_s+j];
                    
                V_m[i_CI].transpose_ci(0);
                conv_CI++;
            }
            else{
                n0[i_CI]=dim[i_CI];
                
                for(int i=0;i<Nd;i++)
                for(int j=0;j<n_s ;j++){
                    r_m[i_CI][i*n_s+j]=r_m[i_CI][i*n_s+j]*ED_with_shift(E_m[i_CI][j]-V_m[i_CI].H_diag_appr[i], edshift);
                }
                        
                if(dim[i_CI]+n_s-conv_num<max_dim+1){
                    for(int i=0;i<Nd;i++)
                        for(int j=0, d=dim[i_CI];j<n_s;j++)
                            if((H_m[i_CI][j*n_s+j]>r_conv)&&(fabs(E_m[i_CI][j]-E_old_m[i_CI][j])>e_conv)){
                                V_m[i_CI].coef[0][i*max_n_vec+d]=r_m[i_CI][i*n_s+j];// /norm[j];
                                d++;
                            }
                    dim[i_CI]+=(n_s-conv_num);
                }
                else{
                    for(int i=0;i<Nd;i++)
                        for(int j=0;j<n_s;j++)
                            V_m[i_CI].coef[0][i*max_n_vec+j]=x_m[i_CI][i*n_s+j];
                    
                    for(int i=0;i<Nd;i++)
                        for(int j=0, d=n_s;j<n_s;j++)
                            if((H_m[i_CI][j*n_s+j]>r_conv)&&(fabs(E_m[i_CI][j]-E_old_m[i_CI][j])>e_conv)){
                                V_m[i_CI].coef[0][i*max_n_vec+d]=r_m[i_CI][i*n_s+j];
                                d++;
                            }
                    n0[i_CI]=0;
                    dim[i_CI]=2*n_s-conv_num;
                }
                                        
                
            }
        
            d_max_tmp=sym_check_for_tensor(V_m[i_CI].coef[0], V_m[i_CI].Na,dim[i_CI],V_m[i_CI].n_states[0], -V_m[i_CI].sym_ab);
            if(d_max<d_max_tmp)d_max=d_max_tmp;
            for(int i=0;i<n_s;i++)
                if(dR_max<H[i*n_s+i])dR_max=H[i*n_s+i];
            for(int i=0;i<n_s;i++)
                if(E_max<fabs(E[i]-E_old[i]))E_max=fabs(E[i]-E_old[i]);
            
            V_m[i_CI].transpose_ci(0);
            memcpy(E_old,E,sizeof(double)*max_dim*n_CI);
            
        }
        
        if(print){
            fprintf(out_stream,"____|___________________|____________|___________|________|________|\n");
        }
        
        if(print){
            if(V_m[0].Na==V_m[0].Nb)if(V_m[0].mult!=0){
                fprintf(out_stream,"maximum deviation from symmetric form - %e\n",d_max);
                
            }
        }
        else{
            // fprintf(stderr,"\rdavidson iter %3d, dE=% .3e",n_step,E_max);
            // fprintf(stderr,", dR=%.3e",dR_max);
            if(V_m[0].Na==V_m[0].Nb)if(V_m[0].mult!=0){
                // fprintf(stderr,", m.d.s.=%.3e",d_max);
                
            }
            
        }
        printf("conv_CI=%d\n",conv_CI);
//         getchar();
        fflush(out_stream);
        if(conv_CI==n_CI)break;
    }
    
    for(int i=0;i<n_CI;i++){
        printf("\n\nset %d:\n", i);
        V_m[i].print_states(0,n_s,1);
    }
    
    exit(0);
    
    
//     fprintf(stderr,"\r                                                                \r");
//     
//     
//     if(print){
//         for(int i=0; i<n_s; i++)V_m[i_CI].E_states[0][i]=E[i];
//         fprintf(out_stream,"\n\nCAS_CI WaveFunctions:\n\n");
//     }
//     V_m[i_CI].print_states(0,n_s,print);
//     
//     fflush(out_stream);
//     
//     delete[] lb;
//     delete[] H_lb;
//     delete[] H_lb_d;
//     
//     delete[] norm;
//     delete[] S    ;
//     delete[] B    ;
//     delete[] c    ;
//     delete[] H_bf ;
//     delete[] S_bf ;
//     
//     //import roots to the external storage
//     for(int i=0;i<Nd;i++)
//         for(int j=0;j<n_s;j++)
//             C->coef[0][i*n_s+j]=x[i*n_s+j];
//     
//     for(int i=0;i<n_s;i++)
//         C->E_states[0][i]=E[i];
//     for(int i=0;i<n_s;i++)
//         C->S2[0][i]=V_m[i_CI].S2[0][i];
//     if(LINEAR)
//     for(int i=0;i<n_s;i++)
//         C->L2[0][i]=V_m[i_CI].L2[0][i];
//     if(LINEAR)
//     for(int i=0;i<n_s;i++)
//         C->P[0][i]=V_m[i_CI].P[0][i];
//         
// 
//     C->symmetrization(0);
//     C->transpose_ci(0);
//     
//     
//     if(print){
//         fprintf(out_stream,"\n\n");
//         printf_timer("Davidson diagonalization");
//         fprintf(out_stream,"_______________________________________________________________________\n");
//     }
// //     exit(0);
    
    delete[] H_m    ;
    delete[] E_m    ;
    delete[] E_old_m;
    delete[] x_m    ;
    delete[] r_m    ;
    delete[] Hx_m   ;
    
    delete[] v_ort;
    
    delete[] dim;
    delete[] n0;
    
    
    return 0;//n_step;
}


davidson_solver::~davidson_solver(){
    
    if(H     != NULL) delete[] H     ;
    if(E     != NULL) delete[] E     ;
    if(E_old != NULL) delete[] E_old ;
    if(x     != NULL) delete[] x     ;
    if(r     != NULL) delete[] r     ;
    if(Hx    != NULL) delete[] Hx    ;
    if(V_m   != NULL) delete[] V_m   ;
    
}
