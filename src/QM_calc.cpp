# include "molecule.h"
# include "SCF.h"
# include "CAS.h"
# include "XMCQDPT.h"
# include "CDAS_PT.h"
# include "CDAS_PT_rel.h"
# include "dsrg_pt.h"
# include "inp_par_read.h"
# include "geom.h"
# include "nopt_libint_engines.h"
# include "z_matrix.h"
# include "RI.h"
# include "l-bfgs_2_1.h"
# include "matr.h"
# include "defaults.h"
#include "timer.h"
#include <cstring>
# include "grabbers.h"
# include "mp2.h"
# include "avas.h"

extern coord_list C;

int single_point_calc( inp_par * P, molecule * Qm){ 
        
    // XMCQDPT needs the determinant CI object the DMRG backend does not provide; the relativistic
    // (SO) CDAS path is likewise determinant-only. Non-relativistic CDAS runs through CDAS_PT2
    // below (SO is a global fixed at parse time, so it is already valid here).
    if(P->cas.y && P->cas.ci_solver==CISOLVER_DMRG){
        if(P->xmc.y){
            fprintf(out_stream,"ERROR: CISOLVER=dmrg cannot be combined with XMCQDPT "
                               "(it needs the determinant CI object, which the DMRG backend does not provide)\n");
            exit(EXIT_FAILURE);
        }
        if(P->cdas.y && SO==1){
            fprintf(out_stream,"ERROR: CISOLVER=dmrg cannot be combined with relativistic CDAS-PT "
                               "(the SO/GRPP path is determinant-only)\n");
            exit(EXIT_FAILURE);
        }
    }

    // AVAS rewrites the reference orbitals so that the active window holds the target atomic
    // valence span. It fills only the window CAS defines; under $SYMM it rotates per irrep.
    if(P->avas.y){
        if(!P->cas.y){
            fprintf(out_stream,"ERROR: $AVAS needs CAS=1 (nothing else consumes the steered active window)\n");
            exit(EXIT_FAILURE);
        }
        if(P->mp2.y || P->cis.y){
            fprintf(out_stream,"ERROR: $AVAS cannot be combined with MP2 or CIS in one run "
                               "(both need canonical orbitals)\n");
            exit(EXIT_FAILURE);
        }
        if(Qm->reorder){
            fprintf(out_stream,"ERROR: $AVAS cannot be combined with $ACT_SPACE reorder=1 "
                               "(two active-space steering mechanisms)\n");
            exit(EXIT_FAILURE);
        }
        if(IS_SYM && Qm->S.n_rep!=Qm->S.n_op){
            fprintf(out_stream,"ERROR: $AVAS supports groups with one-dimensional irreps only (Cs, Ci, C2, C2v, C2h, "
                               "D2, D2h); use one of them or C1\n");
            exit(EXIT_FAILURE);
        }
        if(Qm->PP.size()){
            fprintf(out_stream,"ERROR: $AVAS supports all-electron calculations only "
                               "(an ECP removes the reference shells from the calculation basis)\n");
            exit(EXIT_FAILURE);
        }
    }

    // DSRG-PT2 runs without point-group symmetry only (IS_SYM is 0 for group=C1).
    if(P->dsrg.y && IS_SYM){
        fprintf(out_stream,"ERROR: DSRG-PT2 does not support point-group symmetry; "
                           "run without $SYMM (or with group=C1)\n");
        exit(EXIT_FAILURE);
    }

    Qm->gen_1el_data();
    fprintf(out_stream,"\n\n");
    printf_timer("Calculation of 1-el matrices");

    
    if(P->rhf.y)
        RHF    (Qm,&(P->rhf), P->job_name);
    else
        Qm->MO_orth();
    
    if(P->avas.y)avas_steer(Qm, P->avas, P->job_name);

    if(P->cis.y)CIS      (Qm, &(P->cis), P->job_name);

    // MP2 natural-orbital generation permanently overwrites MO_VEC/orb_energy; the
    // two-step workflow goes through the written .orb file, so forbid one-shot mixing.
    if(P->mp2.y && (P->cas.y || P->xmc.y || P->cdas.y || P->cis.y)){
        fprintf(out_stream,"ERROR: MP2 cannot be combined with CAS/XMCQDPT/CDAS/CIS in one run "
                           "(natural-orbital generation overwrites the reference orbitals)\n");
        exit(EXIT_FAILURE);
    }
    if(P->mp2.y && !P->rhf.y){
        fprintf(out_stream,"ERROR: MP2 requires a converged RHF reference (set RHF=1)\n");
        exit(EXIT_FAILURE);
    }
    if(P->mp2.y)MP2      (Qm, &(P->mp2), P->job_name);


    if(P->cas.y)CAS_SCF  (Qm,&(P->cas), P->job_name);
//     Qm->MO_backup();
    
    if(P->xmc.y)QDPT2    (Qm,&(P->xmc), P->job_name);
    
    if(SO==0)if(P->cdas.y){
        CDAS_PT2    (Qm,&(P->cdas), P->job_name);
    }
    if(SO==1)if(P->cdas.y)CDAS_PT2_rel(Qm,&(P->cdas), P->job_name);

    // DSRG-PT2 drives through the casci_solver seam (not as_aldet()), so it is not caught by the
    // CISOLVER_DMRG reject above and works for both the aldet and DMRG backends.
    if(P->dsrg.y)SA_DSRG_PT2(Qm,&(P->dsrg), P->job_name);


//     Qm->MO_restore();
    
    return 0;

}

int single_point_calc_with_refresh(inp_par * P, molecule * Qm){
    
    Qm->sync_shell_to_geom();
    nopt_engines_link_mol(Qm);
    if(RI)regen_RI_AA(Qm);
    single_point_calc(P,Qm);
    
    return 0;
}


int num_grad_calc_Z( inp_par * P, molecule * Qm, molecule * Mm, int u){
        
//     Z_matrix z;
//     Qm->z.init(Qm->n_atoms);
//     Qm->z.reorder();
//     Qm->z.calc_Z  (Qm->atom_coord);
    Qm->z.fprint("init.zm");
    Qm->z.calc_xyz(Qm->atom_coord);
//     Qm->move_to_mcs(1,1);
    Qm->GAMESS_geom_print("init.out");
    Qm->sync_shell_to_geom();
    
    nopt_engines_link_mol(Qm);
    grab_E=1;
    
//     FILE * calc=fopen("calc.log","w");
//     out_stream=calc;
    
    single_point_calc(P,Qm);
    
    double E0, Ef, Eb, E_old;
    E0=E_grabbed[1];
    E_old = E0+100.0;
    
    FILE * grad=fopen("grad.log","w");

    fprintf(grad,"E0= %.10f\n",E0);
    
    
    Qm->z.set_deriv_step(0.01,0.01,0.01);
    
    int n_c=3*Qm->z.n_at-6;
    double * s = new double[n_c];
    double * G = new double[n_c];
    double * H = new double[n_c];
    for(int i=0;i<n_c;i++)H[i]=1.0/4.0;
    
    l_bfgs_engine BFGS;
    BFGS.init(n_c,10);
    while(fabs(E0-E_old)>0.000001){
        E_old=E0;
        for(int i=0;i<n_c;i++){
            fprintf(stderr,"move c%d forward\n",i);
            Qm->z.p[i]+=Qm->z.dP[i];
            Qm->z.calc_xyz(Qm->atom_coord);
//             Qm->move_to_mcs(0,1);
            Qm->sync_shell_to_geom();
            nopt_engines_link_mol(Qm);
            
            if(RI)regen_RI_AA(Qm);
            
//             Qm->GAMESS_geom_print("0f.out");
            single_point_calc(P,Qm);
            Ef=E_grabbed[1];
//             fprintf(out_stream,"E_0f= %.10f\n",Ef);
            
            Qm->z.p[i]-=Qm->z.dP[i];
            
            fprintf(stderr,"move c%d backward\n",i);
            Qm->z.p[i]-=Qm->z.dP[i];
            Qm->z.calc_xyz(Qm->atom_coord);
//             Qm->move_to_mcs(0,1);
            Qm->sync_shell_to_geom();
            nopt_engines_link_mol(Qm);
            
            if(RI)regen_RI_AA(Qm);
            
//             Qm->GAMESS_geom_print("0b.out");
            single_point_calc(P,Qm);
            Eb=E_grabbed[1];
            Qm->z.p[i]+=Qm->z.dP[i];
            
            G[i]=0.5*(Ef-Eb)/Qm->z.dP[i];
        }
        fprintf(grad,"G:\n");
        fPrintMatr(grad,G,1,n_c,0);
        BFGS.step(s, G, H, do_nothing, do_nothing, n_c,100);
        fprintf(grad,"s:\n");
        fPrintMatr(grad,s,1,n_c,0);        
        
        for(int i=0;i<n_c;i++)Qm->z.p[i]+=s[i];
        
        Qm->z.calc_xyz(Qm->atom_coord);
        Qm->z.fprint("current.zm");
//         Qm->move_to_mcs(0,1);
        Qm->GAMESS_geom_print("current.out");
        Qm->sync_shell_to_geom();
        nopt_engines_link_mol(Qm);
        if(RI)regen_RI_AA(Qm);
        
        fprintf(stderr,"step1 done\n");
//         getchar();
        single_point_calc(P,Qm);
        E0=E_grabbed[1];
        fprintf(grad,"E0= %.10f\n",E0);
        fflush(grad);
    }    
    
    
    return 0;
}

int hessian_calc(inp_par * P, molecule * Qm){
    
    grab_E=1;
    
    FILE * hess=fopen("hess.log","w");
    FILE * hess_steps=fopen("hess_steps.log","w");
    
    int n_c=3*Qm->n_atoms;
    size_t num_states = P->cas.n_s;
    // size_t num_states = 1;
    double Hessian[num_states][n_c][n_c];
    memset(Hessian, 0, sizeof(Hessian)); // a VLA cannot take an initializer (clang rejects it)
    double Ea[num_states], Eb[num_states], Ec[num_states], Ed[num_states];

    fprintf(stderr,"num of sates: %d \n",num_states);

    for(int i=0;i<n_c;i++){
        fprintf(hess_steps,"0 \n");
        for(int j=0;j<n_c;j++){
            if(j>=i){
                fprintf(stderr,"hess move %d forward and %d forward\n",i,j);
                Qm->atom_coord[i]+=0.01;
                Qm->atom_coord[j]+=0.01;
                single_point_calc_with_refresh(P,Qm);
                
                for(int k=0; k<num_states; k++){
                    Ea[k]=E_grabbed[k];
                }

                Qm->atom_coord[i]-=0.01;
                Qm->atom_coord[j]-=0.01;

                fprintf(stderr,"hess move %d forward and %d backward\n",i,j);
                Qm->atom_coord[i]+=0.01;
                Qm->atom_coord[j]-=0.01;
                
                single_point_calc_with_refresh(P,Qm);

                for(int k=0; k<num_states; k++){
                    Eb[k]=E_grabbed[k];
                }

                Qm->atom_coord[i]-=0.01;
                Qm->atom_coord[j]+=0.01;

                fprintf(stderr,"hess move %d backward and %d forward\n",i,j);
                Qm->atom_coord[i]-=0.01;
                Qm->atom_coord[j]+=0.01;
                
                single_point_calc_with_refresh(P,Qm);

                for(int k=0; k<num_states; k++){
                    Ec[k]=E_grabbed[k];
                }

                Qm->atom_coord[i]+=0.01;
                Qm->atom_coord[j]-=0.01;

                fprintf(stderr,"hess move %d backward and %d backward\n",i,j);
                Qm->atom_coord[i]-=0.01;
                Qm->atom_coord[j]-=0.01;
                
                single_point_calc_with_refresh(P,Qm);

                for(int k=0; k<num_states; k++){
                    Ed[k]=E_grabbed[k];
                }

                Qm->atom_coord[i]+=0.01;
                Qm->atom_coord[j]+=0.01;

                for(int k=0; k<num_states; k++){
                    Hessian[k][i][j]=0.25*(Ea[k]-Eb[k]-Ec[k]+Ed[k])/0.01/0.01;
                    Hessian[k][j][i]=0.25*(Ea[k]-Eb[k]-Ec[k]+Ed[k])/0.01/0.01;
                }
                
                fprintf(hess_steps,"%.10f    ", Hessian[0][i][j]);
                
            }
            else
                fprintf(hess_steps,"%.10f    ", Hessian[0][i][j]);
        }

        fprintf(hess_steps,"\n\n");

        for (int k=0; k<num_states; k++){
            fprintf(hess_steps,"%d \n", k);
            for(int i=0;i<n_c;i++){
                for(int j=0;j<n_c;j++){
                    fprintf(hess_steps,"%.10f    ", Hessian[k][i][j]);
                }
                fprintf(hess_steps,"\n");
                fflush(hess_steps);
            }
            fprintf(hess_steps,"\n\n");
            fflush(hess_steps);
        }

    fprintf(hess_steps,"\n");
    fflush(hess_steps);

    }

    
    for(int k=0; k<num_states; k++){
        fprintf(hess,"State No %d \n", k);

        for(int i=0;i<n_c;i++){
            for(int j=0;j<n_c;j++){
                fprintf(hess,"%.10f    ", Hessian[k][i][j]);
            }

            fprintf(hess,"\n");
            fflush(hess);
        }

        fprintf(hess,"\n\n");
    }
    
    
    return 0;
}

int optimization_w_num_grad_calc( inp_par * P, molecule * Qm, int n_s){
    
        
    nopt_engines_link_mol(Qm);
    grab_E=1;
    
//     FILE * calc=fopen("calc.log","w");
//     out_stream=calc;
    
    single_point_calc(P,Qm);
    
    double E0, Ef, Eb, E_old;
    E0=E_grabbed[n_s];
    E_old = E0+100.0;
    
    FILE * grad=fopen("grad.log","w");

    fprintf(grad,"E0= %.10f\n",E0);
    
    Qm->GAMESS_geom_print("init.out");
    
    int n_c=3*Qm->n_atoms;
    double * s = new double[n_c];
    double * G = new double[n_c];
    double * H = new double[n_c];
    for(int i=0;i<n_c;i++)H[i]=1.0/3.0;
    
    l_bfgs_engine BFGS;
    BFGS.init(n_c,10);
    int i_r=0;
    double s_max, g_max;
    while(true){
        E_old=E0;
        for(int i=0;i<n_c;i++){
            fprintf(stderr,"move c%d forward\n",i);
            Qm->atom_coord[i]+=0.01;
            Qm->sync_shell_to_geom();
            nopt_engines_link_mol(Qm);
            if(RI)regen_RI_AA(Qm);
            
//             Qm->GAMESS_geom_print("0f.out");
            single_point_calc(P,Qm);
            Ef=E_grabbed[n_s];
//             fprintf(out_stream,"E_0f= %.10f\n",Ef);
            
            Qm->atom_coord[i]-=0.01;
            
            fprintf(stderr,"move c%d backward\n",i);
            Qm->atom_coord[i]-=0.01;
            Qm->sync_shell_to_geom();
            nopt_engines_link_mol(Qm);
            if(RI)regen_RI_AA(Qm);
            
//             Qm->GAMESS_geom_print("0b.out");
            single_point_calc(P,Qm);
            Eb=E_grabbed[n_s];
            Qm->atom_coord[i]+=0.01;
            
            G[i]=0.5*(Ef-Eb)/0.01;
        }
        fprintf(grad,"G:\n");
        fPrintMatr(grad,G,1,n_c,0);
        BFGS.step(s, G, H, do_nothing, do_nothing, n_c,0.1);
        fprintf(grad,"s:\n");
        fPrintMatr(grad,s,1,n_c,0);        
        for(int i=0;i<n_c;i++)Qm->atom_coord[i]+=s[i];
        
        Qm->sync_shell_to_geom();
        nopt_engines_link_mol(Qm);
        if(RI)regen_RI_AA(Qm);
        
        Qm->GAMESS_geom_print("current.out");
        single_point_calc(P,Qm);
        E0=E_grabbed[n_s];
        s_max=0;
        g_max=0;
        for(int j=0;j<n_c;j++){
            if(s_max<fabs(s[j]))s_max=fabs(s[j]);
            if(g_max<fabs(G[j]))g_max=fabs(G[j]);
        }
        fprintf(grad,"E= %.10f\n",E0);
        fprintf(grad,"dE= %e s=%e, g=%e\n",E0-E_old, s_max, g_max);
        fflush(grad);
        
        if((fabs(E0-E_old)<1E-7)&&(g_max<3e-5)){
            fprintf(out_stream,"Optimization converged\n");
            break;
        }
        if((s_max<1E-4)&&(g_max<3e-5)){
            fprintf(out_stream,"Optimization converged\n");
            break;
        }
        i_r++;
        if(i_r>100){
            fprintf(out_stream,"Optimization did not converge\n");
            break;
        }
        
    }    
    
    
    return 0;
}

int num_grad_calc_P( inp_par * P, molecule * Qm, molecule * Mm, int u){
    
//     Z_matrix z;
//     Qm->z.init(Qm->n_atoms);
//     Qm->z.reorder();
//     Qm->z.calc_Z  (Qm->atom_coord);
    Qm->z.fprint("init.zm");
    Qm->z.calc_xyz(Qm->atom_coord);
//     Qm->move_to_mcs(1,1);
    Qm->GAMESS_geom_print("init.out");
    Qm->sync_shell_to_geom();
    nopt_engines_link_mol(Qm);
    grab_E=1;
    
//     FILE * calc=fopen("calc.log","w");
//     out_stream=calc;
    
    single_point_calc(P,Qm);
    
    double E0, Ef, Eb, E_old;
    E0=E_grabbed[1];
    E_old = E0+100.0;
    
    FILE * grad=fopen("grad.log","w");

    fprintf(grad,"E0= %.10f\n",E0);
    
    int n_c=C.value.size();
//     printf("n_c = %d\n",n_c);
//     getchar();
    double * s = new double[n_c];
    double * G = new double[n_c];
    double * H = new double[n_c];
    for(int i=0;i<n_c;i++)H[i]=1.0/4.0;
    
    double s_max=10;
    double g_max=10;
    
    l_bfgs_engine BFGS;
    BFGS.init(n_c,10);
    set_std_delta();
    while(true){
        E_old=E0;
        for(int i=0;i<n_c;i++){
            fprintf(stderr,"move c%d forward\n",i);
            C.value[i]+=C.value_delta[i];
            coord_sync(Qm->z.p);
            Qm->z.calc_xyz(Qm->atom_coord);
//             Qm->move_to_mcs(0,1);
            Qm->sync_shell_to_geom();
            nopt_engines_link_mol(Qm);
            if(RI)regen_RI_AA(Qm);
            
            Qm->z.fprint("current.zm");
            Qm->GAMESS_geom_print("current.out");
        
//             Qm->GAMESS_geom_print("0f.out");
            single_point_calc(P,Qm);
            Ef=E_grabbed[1];
//             fprintf(out_stream,"E_0f= %.10f\n",Ef);
            
            C.value[i]-=C.value_delta[i];
            
            fprintf(stderr,"move c%d backward\n",i);
            C.value[i]-=C.value_delta[i];
            coord_sync(Qm->z.p);
            Qm->z.calc_xyz(Qm->atom_coord);
//             Qm->move_to_mcs(0,1);
            Qm->sync_shell_to_geom();
            nopt_engines_link_mol(Qm);
            if(RI)regen_RI_AA(Qm);
            
//             Qm->GAMESS_geom_print("0b.out");
            single_point_calc(P,Qm);
            Eb=E_grabbed[1];
            C.value[i]+=C.value_delta[i];
            
            G[i]=0.5*(Ef-Eb)/C.value_delta[i];
        }
        fprintf(grad,"G:\n");
        fPrintMatr(grad,G,1,n_c,0);
        BFGS.step(s, G, H, do_nothing, do_nothing, n_c,100);
        fprintf(grad,"s:\n");
        fPrintMatr(grad,s,1,n_c,0);        
        
        for(int i=0;i<n_c;i++)C.value[i]+=s[i];
        coord_sync(Qm->z.p);
        Qm->z.calc_xyz(Qm->atom_coord);
        Qm->z.fprint("current.zm");
        fprint_coord_list("current.par");
//         Qm->move_to_mcs(0,1);
        Qm->GAMESS_geom_print("current.out");
        Qm->sync_shell_to_geom();
        nopt_engines_link_mol(Qm);
        if(RI)regen_RI_AA(Qm);
        
        fprintf(stderr,"step1 done\n");
//         getchar();
        single_point_calc(P,Qm);
        E0=E_grabbed[1];
        fprintf(grad,"E0= %.10f\n",E0);
        s_max=0;
        g_max=0;
        for(int j=0;j<n_c;j++){
            if(s_max<fabs(s[j]))s_max=fabs(s[j]);
            if(g_max<fabs(G[j]))g_max=fabs(G[j]);
        }
        
        fprintf(grad,"dE= %e s=%e, g=%e\n",E0-E_old, s_max, g_max);
        
        fflush(grad);
        
        if((fabs(E0-E_old)>1E-8)&&(s_max<1e-5))break;
        if((         g_max>1E-8)&&(s_max<1e-5))break;
        
    }    
    
    
    return 0;
}

int optimization_P( inp_par * P, molecule * Qm, molecule * Mm, int u){
    
    
    int n_s=2;
    
   
    char * folder = new char[BUF_LINE_LENGTH];
    FILE * scan_file = fopen("scan.log","w");
    
    
    nopt_engines_link_mol(Qm);
    int n_c=C.value.size();

    double * s = new double[n_c];
    double * G = new double[n_c];
    double * H = new double[n_c];
//     for(int i_p=0;i_p<72;i_p++){
        
//         sprintf(folder,"step%d",i_p);
        sprintf(folder,"opt");
        
        struct stat st = {0};
        if (stat (folder, &st) == -1) {
            mkdir(folder, 0755);
        }
        chdir(folder);
        out_stream = fopen("calc.log","w");
        
        
//         Qm->z.p[32]=(180.0-5.0*i_p)/180.0*M_PI;
        
        
        Qm->z.calc_xyz(Qm->atom_coord);
//         if(i_p==0)
//             Qm->move_to_mcs(1,1);
//         else      Qm->move_to_mcs(0,1);
        Qm->z.fprint("init.zm");
        Qm->GAMESS_geom_print("init.out");
        Qm->sync_shell_to_geom();
        nopt_engines_link_mol(Qm);
        if(RI)regen_RI_AA(Qm);
            

	grab_E=1;
        
        single_point_calc(P,Qm);
        Qm->MO_backup();
        
        double E0, Ef, Eb, E_old;
        E0=E_grabbed[n_s];
        E_old = E0+100.0;
        
        FILE * grad=fopen("grad.log","w");
        
        fprintf(grad,"E0= %.10f\n",E0);
        
//         printf("n_c = %d\n",n_c);
//         getchar();
        
        double s_max=10;
        double g_max=10;
        
        bfgs_engine BFGS;
        BFGS.init(n_c);
        BFGS.set_eq_diag(0.3333);
        set_std_delta();
        int i_r=0;
        while(true){
            E_old=E0;
            for(int i=0;i<n_c;i++){
                fprintf(out_stream,"Optimization step %d: moving coord %d forward\n",i_r,i);
                C.value[i]+=C.value_delta[i];
                coord_sync(Qm->z.p);
                Qm->z.calc_xyz(Qm->atom_coord);
//                 Qm->move_to_mcs(0,1);
                Qm->sync_shell_to_geom();
                nopt_engines_link_mol(Qm);
                if(RI)regen_RI_AA(Qm);
                
//                 Qm->z.fprint("current.zm");
//                 Qm->GAMESS_geom_print("current.out");
            
//                 Qm->GAMESS_geom_print("0f.out");
                Qm->MO_restore();
                single_point_calc(P,Qm);
                Ef=E_grabbed[n_s];
//                 fprintf(out_stream,"E_0f= %.10f\n",Ef);
                
                C.value[i]-=C.value_delta[i];
                
                fprintf(out_stream,"Optimization step %d: moving coord %d backward\n",i_r,i);
                C.value[i]-=C.value_delta[i];
                coord_sync(Qm->z.p);
                Qm->z.calc_xyz(Qm->atom_coord);
//                 Qm->move_to_mcs(0,1);
                Qm->sync_shell_to_geom();
                nopt_engines_link_mol(Qm);
                if(RI)regen_RI_AA(Qm);
                
//                 Qm->GAMESS_geom_print("0b.out");
                single_point_calc(P,Qm);
                Eb=E_grabbed[n_s];
                C.value[i]+=C.value_delta[i];
                
                G[i]=0.5*(Ef-Eb)/C.value_delta[i];
//                 H[i]=fabs((Ef+Eb-2.0*E0)/(C.value_delta[i]*C.value_delta[i]));
//                 H[i]=H[i]+0.1/H[i];
//                 if(i_r==0)BFGS.set_diag(H);
            }
            fprintf(grad,"G:\n");
            fPrintMatr(grad,G,1,n_c,0);
            BFGS.step(s, G, do_nothing);
//             BFGS.step(s, G, H, do_nothing, do_nothing, n_c,0.1);
            fprintf(grad,"s:\n");
            fPrintMatr(grad,s,1,n_c,0);        
            
            for(int i=0;i<n_c;i++)C.value[i]+=s[i];
            int n_l=0;
            double a=0.5;
            while(true){
                coord_sync(Qm->z.p);
                Qm->z.calc_xyz(Qm->atom_coord);
                Qm->z.fprint("current.zm");
                fprint_coord_list("current.par");
//                 Qm->move_to_mcs(0,1);
                Qm->GAMESS_geom_print("current.out");
                fprintf(out_stream,"Optimization step %d - result:\n",i_r);
                Qm->sync_shell_to_geom();
                nopt_engines_link_mol(Qm);
                if(RI)regen_RI_AA(Qm);
                
	            //write_ci=1;
	            write_orbs=1;
                Qm->MO_restore();
                single_point_calc(P,Qm);
                E0=E_grabbed[n_s];
	            if(E0<=E_old)break;
                if(n_l>5)break;
                fprintf(grad,"positive dE scaling step\n");
                for(int i=0;i<n_c;i++)C.value[i]-=a*s[i];
                a=a/2.0;
                for(int i=0;i<n_c;i++)BFGS.prev_step[i]=BFGS.prev_step[i]/2.0;
//                 if (BFGS.lbfgs_vec_num>0)
//                 for(int i=0;i<n_c;i++)
//                     BFGS.prev_step[i+(BFGS.lbfgs_step_num-1)%BFGS.lbfgs_vec_num*n_c]=
//                     BFGS.prev_step[i+(BFGS.lbfgs_step_num-1)%BFGS.lbfgs_vec_num*n_c]/2.0;
                n_l++;
            }
            Qm->MO_backup();
                
            //write_ci=0;
	        write_orbs=0;
            fprintf(grad,"E0= %.10f\n",E0);
            s_max=0;
            g_max=0;
            for(int j=0;j<n_c;j++){
                if(s_max<fabs(s[j]))s_max=fabs(s[j]);
                if(g_max<fabs(G[j]))g_max=fabs(G[j]);
            }
            
            fprintf(grad,"dE= %e s=%e, g=%e\n",E0-E_old, s_max, g_max);
            
            fflush(grad);
            
            if((fabs(E0-E_old)<1E-8)&&(g_max<1e-4)){
                fprintf(out_stream,"Optimization converged\n");
                break;
            }
            if((         s_max<3E-4)&&(g_max<1e-4)){
                fprintf(out_stream,"Optimization converged\n");
                break;
            }
            i_r++;
            if(i_r>100){
                fprintf(out_stream,"Optimization did not converge\n");
                break;
            }
        }
        fclose(grad);
        fclose(out_stream);
        out_stream = stdout;
        
//         fprintf(scan_file,"a = %d  E0= %.10f\n",180-5*i_p,E0);
        chdir("../");
        fflush(scan_file);
        fclose(out_stream);
        out_stream=stdout;

//     }
    
    delete[] s;
    delete[] G;
    delete[] H;
    delete[] folder;
        
    
    return 0;
}


int relaxed_scan_P( inp_par * P, molecule * Qm, molecule * Mm, int u){
    
    
    int n_s=2;
        
    char * folder = new char[BUF_LINE_LENGTH];
    FILE * scan_file = fopen("scan.log","w");
    
    
    nopt_engines_link_mol(Qm);
    int n_c=C.value.size();

    double * s = new double[n_c];
    double * G = new double[n_c];
    double * H = new double[n_c];
//     for(int i_p=0;i_p<5;i_p++){
//     for(int i_p=0;i_p<8;i_p++){
    for(int i_p=0;i_p<36;i_p++){
        
        sprintf(folder,"step%d",i_p);
//         sprintf(folder,"opt");
        
        struct stat st = {0};
        if (stat (folder, &st) == -1) {
            mkdir(folder, 0755);
        }
        chdir(folder);
        out_stream = fopen("calc.log","w");
        
        
//         Qm->z.p[37]=(130.0+5.0*i_p)/180.0*M_PI;
//         Qm->z.p[37]=(125.0-5.0*i_p)/180.0*M_PI;
//         Qm->z.p[32]=(-5.0+5.0*i_p)/180.0*M_PI;
        Qm->z.p[32]=(-10.0-5.0*i_p)/180.0*M_PI;
        
        
        Qm->z.calc_xyz(Qm->atom_coord);
//         if(i_p==0)
//             Qm->move_to_mcs(1,1);
//         else      Qm->move_to_mcs(0,1);
        Qm->z.fprint("init.zm");
        Qm->GAMESS_geom_print("init.out");
        Qm->sync_shell_to_geom();
        nopt_engines_link_mol(Qm);
        if(RI)regen_RI_AA(Qm);
        

	grab_E=1;
        
        single_point_calc(P,Qm);
        Qm->MO_backup();
        
        double E0, Ef, Eb, E_old;
        E0=E_grabbed[n_s];
        E_old = E0+100.0;
        
        FILE * grad=fopen("grad.log","w");
        
        fprintf(grad,"E0= %.10f\n",E0);
        
//         printf("n_c = %d\n",n_c);
//         getchar();
        
        double s_max=10;
        double g_max=10;
        
        bfgs_engine BFGS;
        BFGS.init(n_c);
        BFGS.set_eq_diag(0.3333);
        set_std_delta();
        int i_r=0;
        while(true){
            E_old=E0;
            for(int i=0;i<n_c;i++){
                fprintf(out_stream,"Optimization step %d: moving coord %d forward\n",i_r,i);
                C.value[i]+=C.value_delta[i];
                coord_sync(Qm->z.p);
                Qm->z.calc_xyz(Qm->atom_coord);
//                 Qm->move_to_mcs(0,1);
                Qm->sync_shell_to_geom();
                nopt_engines_link_mol(Qm);
                if(RI)regen_RI_AA(Qm);
                
//                 Qm->z.fprint("current.zm");
//                 Qm->GAMESS_geom_print("current.out");
            
//                 Qm->GAMESS_geom_print("0f.out");
                Qm->MO_restore();
                single_point_calc(P,Qm);
                Ef=E_grabbed[n_s];
//                 fprintf(out_stream,"E_0f= %.10f\n",Ef);
                
                C.value[i]-=C.value_delta[i];
                
                fprintf(out_stream,"Optimization step %d: moving coord %d backward\n",i_r,i);
                C.value[i]-=C.value_delta[i];
                coord_sync(Qm->z.p);
                Qm->z.calc_xyz(Qm->atom_coord);
//                 Qm->move_to_mcs(0,1);
                Qm->sync_shell_to_geom();
                nopt_engines_link_mol(Qm);
                if(RI)regen_RI_AA(Qm);
                
//                 Qm->GAMESS_geom_print("0b.out");
                single_point_calc(P,Qm);
                Eb=E_grabbed[n_s];
                C.value[i]+=C.value_delta[i];
                
                G[i]=0.5*(Ef-Eb)/C.value_delta[i];
//                 H[i]=fabs((Ef+Eb-2.0*E0)/(C.value_delta[i]*C.value_delta[i]));
//                 H[i]=H[i]+0.1/H[i];
//                 if(i_r==0)BFGS.set_diag(H);
            }
            fprintf(grad,"G:\n");
            fPrintMatr(grad,G,1,n_c,0);
            BFGS.step(s, G, do_nothing);
//             BFGS.step(s, G, H, do_nothing, do_nothing, n_c,0.1);
            fprintf(grad,"s:\n");
            fPrintMatr(grad,s,1,n_c,0);        
            
            for(int i=0;i<n_c;i++)C.value[i]+=s[i];
            
            
            s_max=0;
            g_max=0;
            for(int j=0;j<n_c;j++){
                if(s_max<fabs(s[j]))s_max=fabs(s[j]);
                if(g_max<fabs(G[j]))g_max=fabs(G[j]);
            }
            
//             if(g_max>1.0){
//                 fprintf(out_stream,"Can not continue optimization, gradient is too large\n");
//                 exit(0);
//             }
            
            int n_l=0;
            double a=0.5;
            while(true){
                coord_sync(Qm->z.p);
                Qm->z.calc_xyz(Qm->atom_coord);
                Qm->z.fprint("current.zm");
                fprint_coord_list("current.par");
//                 Qm->move_to_mcs(0,1);
                Qm->GAMESS_geom_print("current.out");
                fprintf(out_stream,"Optimization step %d - result:\n",i_r);
                Qm->sync_shell_to_geom();
                nopt_engines_link_mol(Qm);
                if(RI)regen_RI_AA(Qm);
                
	            //write_ci=1;
	            write_orbs=1;
                Qm->MO_restore();
                single_point_calc(P,Qm);
                E0=E_grabbed[n_s];
	            if(E0<=E_old)break;
                if(n_l>5)break;
                fprintf(grad,"positive dE, scaling step\n");
                for(int i=0;i<n_c;i++)C.value[i]-=a*s[i];
                a=a/2.0;
                for(int i=0;i<n_c;i++)BFGS.prev_step[i]=BFGS.prev_step[i]/2.0;
//                 if (BFGS.lbfgs_vec_num>0)
//                 for(int i=0;i<n_c;i++)
//                     BFGS.prev_step[i+(BFGS.lbfgs_step_num-1)%BFGS.lbfgs_vec_num*n_c]=
//                     BFGS.prev_step[i+(BFGS.lbfgs_step_num-1)%BFGS.lbfgs_vec_num*n_c]/2.0;
                n_l++;
            }
            Qm->MO_backup();
                
            //write_ci=0;
	        write_orbs=0;
            fprintf(grad,"E0= %.10f\n",E0);
            
            fprintf(grad,"dE= %e s=%e, g=%e\n",E0-E_old, s_max, g_max);
            
            fflush(grad);
            
            if((fabs(E0-E_old)<1E-8)&&(g_max<1e-4)){
                fprintf(out_stream,"Optimization converged\n");
                break;
            }
            if((         s_max<3E-4)&&(g_max<1e-4)){
                fprintf(out_stream,"Optimization converged\n");
                break;
            }
            i_r++;
            if(i_r>100){
                fprintf(out_stream,"Optimization did not converge\n");
                break;
            }
        }
        fclose(grad);
        fclose(out_stream);
        out_stream = stdout;
        
//         fprintf(scan_file,"a = %d  E0= %.10f\n",130+5*i_p,E0);
//         fprintf(scan_file,"a = %d  E0= %.10f\n",125-5*i_p,E0);
//         fprintf(scan_file,"a = %d  E0= %.10f\n",-5+5*i_p,E0);
        fprintf(scan_file,"a = %d  E0= %.10f\n",-10-5*i_p,E0);
        chdir("../");
        fflush(scan_file);
        fclose(out_stream);
        out_stream=stdout;

    }   
    
    delete[] s;
    delete[] G;
    delete[] H;
    delete[] folder;
        
    
    return 0;
}
