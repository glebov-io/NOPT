
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cctype>

#include "inp_par_read.h"
#include "keywords.h"
#include "mol_link.h"
#include "etc.h"
#include "xmc_read.h"
#include "basis_lib_read.h"
#include "defaults.h"
#include "common_vars.h"
#include "inp_out.h"


const char*ny{"no yes"};

//RWC
backup_par::backup_par(){
    
    h1=1;
//     h2=1;
//     ri=1;
    prefix=NULL;
}

int backup_par::disable(){
    
    if(h1!=1)fprintf(out_stream,"\n\n\nWARNING: storing and reading 1-electron Hamiltonian is disabled\n\n\n");
    
    h1=1;
//     h2=1;
//     ri=1;
    return 0;
    
}


int backup_par::read_group(char * inp){
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, backup_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, backup_group_start)==0){
        return 0;
    }
    
    while(!P.r_eof()){
        if(key_word_comp(line, h1_backup_kw))
            h1 = kw_to_i(line, h1_backup_kw, 1);
//     if(key_word_comp(line, h2_backup_kw))
//         h2 = kw_to_i(line, h2_backup_kw, 1);

//     if(key_word_comp(line, ri_backup_kw))
//         ri = kw_to_i(line, ri_backup_kw, 1);
    
        if(key_word_comp(    line, prefix_backup_kw))
            kw_to_s(&prefix, line, prefix_backup_kw);
        
        if(key_word_comp(line, backup_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    
    
    
    
    return 0;
}

backup_par::~backup_par(){
    
    if(prefix!=NULL) delete[] prefix;
}

int backup_code_print(int a){
    
    if(a==0) fprintf(out_stream,"read (%d)\n",a);
    if(a==1) fprintf(out_stream,"calc, but not write (%d)\n",a);
    if(a==2) fprintf(out_stream,"calc and write (%d)\n",a);
    return 0;
}

//RHF
rhf_par::rhf_par(){
    
    y=0;
    guess = GUESS_HUCKEL;
//     huckel_guess=0;
//     h_core_guess=0;
//     read_guess  =0;
    max_it = RHF_MAX_IT_DEFAULT;
    e_conv = RHF_EN_CON_DEFAULT;
    g_conv = RHF_GR_CON_DEFAULT;
    
};

int rhf_par::read_group(char * inp){
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, rhf_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, rhf_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $RHF group in file %s\n",inp);
        exit(1);
    }
    
    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, rhf_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
//     if((huckel_guess+h_core_guess+read_guess)==0){
//         huckel_guess=1;
//         h_core_guess=0;
//         read_guess  =0;
//     }
        
//     if((huckel_guess+h_core_guess+read_guess)!=1){
//         fprintf(out_stream,"ERROR: multiple choice of starting guess\n");
//         fprintf(out_stream,"       HUCKEL - %c%c%c\n",ny[3*huckel_guess],ny[3*huckel_guess+1],ny[3*huckel_guess+2]);
//         fprintf(out_stream,"       H_CORE - %c%c%c\n",ny[3*h_core_guess],ny[3*h_core_guess+1],ny[3*h_core_guess+2]);
//         fprintf(out_stream,"       READ   - %c%c%c\n",ny[3*  read_guess],ny[3*  read_guess+1],ny[3*  read_guess+2]);
//         fprintf(out_stream,"       you must choose only one.\n");
//         exit(1);
//     }
    
    
    
    
    return 0;
}

int rhf_par::read_line(char * inp){
    
    // GUESS=HUCKEL carries the retired bare keyword as its value
    if(key_word_comp(inp, rhf_huckel)&&(key_word_comp(inp, guess_kw)==0)){
        fprintf(out_stream,"ERROR: while parsing $RHF group keyword HUCKEL is deprecated\n");
        exit(0);
    }

    if(key_word_comp(inp, rhf_h_core)){
        fprintf(out_stream,"ERROR: while parsing $RHF group keyword H_CORE is deprecated\n");
        exit(0);
    }
    
    if(key_word_comp(inp, guess_kw)){
        if      (kw_to_kw(inp, guess_kw, guess_huckel_kw)) guess = GUESS_HUCKEL;
        else if (kw_to_kw(inp, guess_kw, guess_sad_kw   )) guess = GUESS_SAD;
        else{
            fprintf(out_stream,"ERROR: unknown GUESS value; accepted: huckel, sad\n");
            exit(1);
        }
    }

//     if(key_word_comp(inp, rhf_read))
//         read_guess=1;
    
    if(key_word_comp(inp, max_it_kw)){
        max_it = kw_to_i(inp, max_it_kw, RHF_MAX_IT_DEFAULT);
    }
    
    if(key_word_comp(inp, e_conv_kw)){
        e_conv = kw_to_f(inp, e_conv_kw, RHF_EN_CON_DEFAULT);
    }
    if(key_word_comp(inp, g_conv_kw)){
        g_conv = kw_to_f(inp, g_conv_kw, RHF_GR_CON_DEFAULT);
    }
    
    return 0;
}

int rhf_par::write_info(int n_cor){
    
    int s;
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"_______________________Starting_RHF_calculations_______________________\n\n");
    fprintf(out_stream,"Energy convergence:               %e\n",e_conv);
    fprintf(out_stream,"Orbital gradient convergence:     %e\n",g_conv);
    fprintf(out_stream,"Maximum number of iterations:     %d\n",max_it);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Number of core orbitals:          %d\n",n_cor);
    
//     fprintf(out_stream,"Starting guess:                   ");
//     if(huckel_guess)fprintf(out_stream,"HUCKEL\n");
//     if(h_core_guess)fprintf(out_stream,"H CORE\n");
//     if(  read_guess)fprintf(out_stream,"Read from file\n");
    
    fprintf(out_stream,"\n");
//     fprintf(out_stream,"Folder for output files: %s\n",out_folder_name);
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
    
    return 0;
    
}

rhf_par::~rhf_par(){
    
}

//CAS
cas_par::cas_par(){
    y=0;
    ci_solver = CISOLVER_ALDET;
    converger = CONVERGER_SOSCF;
    lbfgs     = CAS_LBFGS_DEFAULT;
    //convergence
    max_it = CAS_MAX_IT_DEFAULT;
    e_conv = CAS_EN_CON_DEFAULT;
    g_conv = CAS_GR_CON_DEFAULT;
    s_conv = CAS_STEP_CON_DEFAULT;
    x_max  = CAS_X_MAX_DEFAULT;
    x_max_set = false;
    lbfgs_set = false;
    method=0;
    
    //states
    n_s=1;
    w_state.resize(1);
    w_state[0]=1.0;
    
    track=0;
    rotate_orbs=1;
    w_state_type=-1;
    // Lambda=-1;

        
}

int cas_par::w_linear_read_line(char * inp){
    int found_L=0;
    int found_S=0;
    int found_W=0;
    int found_M=0;
    int L,S;
    double M;
    std::vector<double>W;
    
    
    if(key_word_comp(inp, w_linear_L_kw)){
        L = kw_to_i (inp, w_linear_L_kw, -18);
        if(L!=-18)found_L=1;
    }
    
    if(key_word_comp(inp, w_linear_sign_kw)){
        S = kw_to_i (inp, w_linear_sign_kw, 0);
        found_S=1;
    }
    
    if(key_word_comp(inp, mult_kw)){
        M = kw_to_f (inp, mult_kw, 0);
        found_M=1;
    }
    
    if(key_word_comp(inp, w_state_kw)){
        int n=kw_count(inp, w_state_kw,';');
        W.resize(n);
        kw_to_d_v(&W, inp, w_state_kw, n);
        cut_zero(&W);
        if(n) found_W=1;
    }
    
    if((found_L+found_M+found_S+found_W)==0)return 0;
    
    if(found_L*found_M*found_W*(found_S+L)==0){
        fprintf(out_stream,"ERROR: could not find w_state, multiplicity");
        if(L==0)fprintf(out_stream,", sign");
        fprintf(out_stream," and/or lambda while pharsing $W_STATE group. Line:\n");
        fprintf(out_stream,"       - %s\n",inp);
        exit(0);
    }
    if(L==0)if((S!=1)&&(S!=-1)){
        fprintf(out_stream,"ERROR: wrong value -- sign=%d\n",S);
        fprintf(out_stream,"       sign can be only +1 or -1 for Lambda=0\n");
        fprintf(out_stream,"       error is in line:\n%s",inp);
        exit(1);
    }
    
    
    w_state_by_rep.push_back(W);
    rep_lambda.push_back(L);
    rep_sign.push_back(S);
    rep_spin.push_back((M-1.0)/2.0);
    
    
    
    return 0;
}

int cas_par::w_state_by_rep_read(char * inp){
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    if(LINEAR==0){
        nopt_printf("ERROR: IRREP_W_STATE is written only for group=LINEAR inp_par_read.cpp:225");exit(1);
    }
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, w_state_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, w_state_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $W_STATE group in file %s\n",inp);
        fprintf(out_stream,"       $W_STATE group is needed because w_state=ir.rep is set in $CAS group\n");
        exit(1);
    }
    P.r_gets(line,BUF_LINE_LENGTH);
    while(!P.r_eof()){
        w_linear_read_line(line);
        P.r_gets(line,BUF_LINE_LENGTH);;
        if(key_word_comp(line, w_state_group_end))break;
    }
    return 0;


}
int cas_par::read_group(char * inp){

    dav.read_group(inp);
    dmrg.read_group(inp);

    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, cas_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, cas_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $CAS group in file %s\n",inp);
        exit(1);
    }
    
    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, cas_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    
    if(dav.n_s==-1)
        dav.n_s = max(DAV_N_STATES_MULT_DEFAULT*n_s, DAV_N_STATES_DEFAULT);
    
    if(w_state_type==1){
        w_state.resize(n_s);
        for(int i=0;i<n_s;i++)w_state[i]=1.0;
    }
    if(w_state_type==2)w_state_by_rep_read(inp);
    
    if(n_s<w_state.size()){
        n_s=w_state.size();
        fprintf(out_stream,"\n\nWARNING: n_s was changed to %d due to large w_state length\n\n",n_s);
    }
    
    if(method>3 ){
        nopt_printf("ERROR: duplicate definition of CASSCF method\n");
        nopt_printf("       use SA or SM, but not both\n");
        exit(1);
    }
    method--;
    if(method==-1)method=CAS_METHOD_DEFAULT;

    // super-CI-PT is state-averaged by construction: its Koopmans matrices come from the
    // SA RDMs and it solves one common amplitude set.
    if(converger==CONVERGER_SXPT){
        if(method!=1){
            nopt_printf("ERROR: converger=sxpt is state-averaged only; use SA\n");
            exit(1);
        }
        if(lbfgs<0){
            nopt_printf("ERROR: lbfgs must be a non-negative history depth (0 = bare step)\n");
            exit(1);
        }
        if(!x_max_set) x_max = CAS_X_MAX_SXPT_DEFAULT;
    }
    if(converger==CONVERGER_SOSCF && lbfgs_set)
        nopt_printf("NOTE: lbfgs applies to converger=sxpt only; ignored\n");

    if(ci_solver==CISOLVER_DMRG) dmrg.validate();

    return 0;
}

int cas_par::read_line(char * inp){
    
    if(key_word_comp(inp, max_it_kw)){
        max_it = kw_to_i(inp, max_it_kw, CAS_MAX_IT_DEFAULT);
    }
    
    
    
    if(key_word_comp(inp, cisolver_kw)){
        if      (kw_to_kw(inp, cisolver_kw, cisolver_aldet_kw)) ci_solver = CISOLVER_ALDET;
        else if (kw_to_kw(inp, cisolver_kw, cisolver_dmrg_kw )) ci_solver = CISOLVER_DMRG;
        else{
            fprintf(out_stream,"ERROR: unknown CISOLVER value; accepted: aldet, dmrg\n");
            exit(1);
        }
    }

    if(key_word_comp(inp, converger_kw)){
        if      (kw_to_kw_exact(inp, converger_kw, converger_soscf_kw)) converger = CONVERGER_SOSCF;
        else if (kw_to_kw_exact(inp, converger_kw, converger_sxpt_kw )) converger = CONVERGER_SXPT;
        else{
            fprintf(out_stream,"ERROR: unknown CONVERGER value; accepted: soscf, sxpt\n");
            exit(1);
        }
    }

    if(key_word_comp(inp, cas_track_kw)){
        track=1;
    }
    
    if(key_word_comp(inp, cas_rotate_orbs_kw)){
        rotate_orbs = kw_to_i(inp, cas_rotate_orbs_kw, 1);
    }
    
    if(key_word_comp(inp, e_conv_kw)){
        e_conv = kw_to_f(inp, e_conv_kw, CAS_EN_CON_DEFAULT);
    }
    
    if(key_word_comp(inp, g_conv_kw)){
        g_conv = kw_to_f(inp, g_conv_kw, CAS_GR_CON_DEFAULT);
    }
    
    if(key_word_comp(inp, s_conv_kw)){
        s_conv = kw_to_f(inp, s_conv_kw, CAS_STEP_CON_DEFAULT);
    }
    
    if(key_word_comp(inp, x_max_kw)){
        x_max = kw_to_f(inp, x_max_kw, CAS_X_MAX_DEFAULT);
        x_max_set = true;
    }
    
    if(key_word_comp(inp, cas_lbfgs_kw)){
        lbfgs = kw_to_i(inp, cas_lbfgs_kw, CAS_LBFGS_DEFAULT);
        lbfgs_set = true;
    }

    if(key_word_comp(inp, cas_SA_kw)){
        method+=2;
    }
    if(key_word_comp(inp, cas_SM_kw)){
        method+=3;
    }
        
    
    if(key_word_comp(inp, num_state_kw)){
        n_s = kw_to_i(inp, num_state_kw, 1);
    }
    
    if(key_word_comp(inp, w_state_kw)){
        if(w_state_type!=-1){
            fprintf(out_stream,"ERROR: multiple definition of w_state\n");
            exit(1);
        }
        if(kw_to_kw(inp,w_state_kw,w_state_eq_kw)   ){w_state_type=1;goto ws_done;}
        if(kw_to_kw(inp,w_state_kw,w_state_irrep_kw)){w_state_type=2;goto ws_done;}
        w_state_type=0;
        int n=kw_count(inp, w_state_kw,';');
        w_state.resize(n);
        kw_to_d_v(&w_state, inp, w_state_kw, n);
        cut_zero(&w_state);
    }
    ws_done:
        
    return 0;
}

int cas_par::write_info(int n_a, int n_b, int n_o, int n_c, int mult){
    
    int s;
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"______________________Starting_CASSCF_calculations_____________________\n\n");
    
    fprintf(out_stream,"Number of states                  %d\n",n_s );
    fprintf(out_stream,"State multiplicity                %d " ,mult);
    if(mult==1) fprintf(out_stream,"(singlet)\n");
    if(mult==2) fprintf(out_stream,"(doublet)\n");
    if(mult==3) fprintf(out_stream,"(triplet)\n");
    if(mult==4) fprintf(out_stream,"(quadruplet)\n");
    if(mult >4) fprintf(out_stream,"(high-spin)\n");
    if(mult==0) fprintf(out_stream,"(all)\n");
    // if(LINEAR){
    // fprintf(out_stream,"State angle moment                ");
    // if(Lambda==-1)fprintf(out_stream,"(all)\n");
    // else          fprintf(out_stream,"%d\n",Lambda);
    // }
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Active space:\n");
    fprintf(out_stream,"Number of alpha electrons         %d\n",n_a);
    fprintf(out_stream,"Number of beta  electrons         %d\n",n_b);
    fprintf(out_stream,"Number of active orbitals         %d\n",n_o);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Number of core orbitals:          %d\n",n_c);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Preparation of virtual orbitals:  %c%c%c\n",ny[3*rotate_orbs],ny[3*rotate_orbs+1],ny[3*rotate_orbs+2]);
    fprintf(out_stream,"State tracking:                   %c%c%c\n",ny[3*track      ],ny[3*track      +1],ny[3*track      +2]);
    if(method==1)fprintf(out_stream,"State averaging:\n");
    if(method==2)fprintf(out_stream,"Separate minimization:\n");
    if(w_state_type==0){
        fprintf(out_stream,"Weights:\t");
        for(auto &w: w_state) fprintf(out_stream,"% .2e ",w);
        fprintf(out_stream,"\n");
    }
    if(w_state_type==1)fprintf(out_stream,"Weights:\tall equal to 1\n");
    if(w_state_type==2){
        for(int i=0;i<w_state_by_rep.size();i++){
            fprintf(out_stream,"%4d) ",i);
            fprintf(out_stream,"L =%3d ",rep_lambda[i]);
            if(rep_lambda[i]==0)if(rep_sign[i]== 1)fprintf(out_stream,"(+), ");
            if(rep_lambda[i]==0)if(rep_sign[i]==-1)fprintf(out_stream,"(-), ");
            if(rep_lambda[i]!=0)                   fprintf(out_stream,"   , ");
            fprintf(out_stream,"S=%4.1f, ",rep_spin[i]);
            fprintf(out_stream,"Weights:\t");
            for(auto &w: w_state_by_rep[i]) fprintf(out_stream,"% .2e ",w);
            fprintf(out_stream,";\n");
        }
    }
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"SCF settings:\n");
    fprintf(out_stream,"Energy convergence:               %e\n",e_conv);
    fprintf(out_stream,"Orbital gradient convergence:     %e\n",g_conv);
    fprintf(out_stream,"Rotation matrix convergence :     %e\n",s_conv);
    fprintf(out_stream,"Maximum number of iterations:     %d\n",max_it);
    if(converger==CONVERGER_SXPT){
        fprintf(out_stream,"Orbital converger:                super-CI-PT (sxpt)\n");
        fprintf(out_stream,"Orbital L-BFGS depth        :     %d\n",lbfgs);
        fprintf(out_stream,"Maximum SX-PT step          :     %e\n",x_max);
    }
    else fprintf(out_stream,"Maximum SOSCF step          :     %e\n",x_max);
    fprintf(out_stream,"\n");
    if      (ci_solver==CISOLVER_ALDET){
        fprintf(out_stream,"CI solver:                        determinant CI (aldet)\n");
        dav.write_info();
    }
    else if (ci_solver==CISOLVER_DMRG){
        fprintf(out_stream,"CI solver:                        DMRG\n");
        dmrg.write_info();
    }
    else{
        fprintf(out_stream,"CI solver:                        unknown (%d)\n",ci_solver);
    }
    
//     fprintf(out_stream,"Folder for output files: %s\n",out_folder_name);
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
    
    return 0;
    
}

cas_par::~cas_par(){
    
}

//-----------------------------------------------------------------------------------------------------

//CIS
cis_par::cis_par(){
    y=0;

    n_s = 1;
    n_f = 0;
    method = 1;
    nat_orb = 0;		// новое

}

int cis_par::read_group(char * inp){
    
    dav.read_group(inp);
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, cis_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, cis_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $CIS group in file %s\n",inp);
        exit(1);
    }
    
    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, cis_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }

    if ( (method!=1)&&(method!=2) )
    {
    fprintf(out_stream, "WARNING: Method = %d is not available. \n", method);
    fprintf(out_stream, "         Method = 1 is used \n");
    method = 1;
    }

    if(dav.n_s==-1)
        dav.n_s = max(DAV_N_STATES_MULT_DEFAULT*n_s, DAV_N_STATES_DEFAULT);
    
    
    
    if(n_s<w_state.size())
    {
        n_s=w_state.size();
        fprintf(out_stream,"\n\n WARNING: n_s was changed to %d due to large w_state length\n\n", n_s);
    }
        
    return 0;
}

int cis_par::read_line(char * inp){
    
    if(key_word_comp(inp, num_frozen_orb_kw)){
        n_f = kw_to_i(inp, num_frozen_orb_kw, 0);
    }
    
    if(key_word_comp(inp, num_state_kw)){
        n_s = kw_to_i(inp, num_state_kw, 1);
    }
    
    if(key_word_comp(inp, method_kw)){
        method = kw_to_i(inp, method_kw, 1);
    }

    if(key_word_comp(inp, nat_orb_kw)){
        nat_orb = kw_to_i(inp, nat_orb_kw, 0);
    }

    if(key_word_comp(inp, w_state_kw)){
        int n=kw_count(inp, w_state_kw,';');
        w_state.resize(n);
        kw_to_d_v(&w_state, inp, w_state_kw, n);
        cut_zero(&w_state);
    }
            
    return 0;
}

int cis_par::write_info(){
    
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"______________________Starting_CIS_calculations________________________\n\n");
    
    fprintf(out_stream,"Number of states                  %d\n",n_s );
    fprintf(out_stream,"Number of frozen orbitals         %d\n",n_f );
    fprintf(out_stream,"Method used                       CIS_%d\n",method);
    if (nat_orb){fprintf(out_stream,"Make Natural Orbitals             yes");}
    else{fprintf(out_stream,"Make Natural Orbitals             no");}
    
    fprintf(out_stream,"\n");
    dav.write_info();
    
//     fprintf(out_stream,"Folder for output files: %s\n",out_folder_name);
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
    
    return 0;
    
}

cis_par::~cis_par(){

}

//-----------------------------------------------------------------------------------------------------

//MP2
mp2_par::mp2_par(){
    y=0;

    n_f = 0;
    nat_orb = 0;

}

int mp2_par::read_group(char * inp){

    recursive_file P;
    char line[BUF_LINE_LENGTH];

    P.r_open(inp);

    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, mp2_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, mp2_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $MP2 group in file %s\n",inp);
        exit(1);
    }

    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, mp2_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }

    return 0;
}

int mp2_par::read_line(char * inp){

    if(key_word_comp(inp, num_frozen_orb_kw)){
        n_f = kw_to_i(inp, num_frozen_orb_kw, 0);
    }

    if(key_word_comp(inp, nat_orb_kw)){
        nat_orb = kw_to_i(inp, nat_orb_kw, 0);
    }

    return 0;
}

int mp2_par::write_info(){

    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"______________________Starting_MP2_calculations________________________\n\n");

    fprintf(out_stream,"Number of frozen orbitals         %d\n",n_f );
    if (nat_orb){fprintf(out_stream,"Make Natural Orbitals             yes");}
    else{fprintf(out_stream,"Make Natural Orbitals             no");}

    fprintf(out_stream,"\n");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");


    return 0;

}

mp2_par::~mp2_par(){

}


//-----------------------------------------------------------------------------------------------------

dav_par::dav_par(){
    
    n_s    = -1;
    n_bf   = DAV_N_BF_DEFAULT;
    max_it = DAV_MAX_IT_DEFAULT;
    e_conv = DAV_EN_CON_DEFAULT;
    r_conv = DAV_R_CON_DEFAULT;
    se_min = DAV_SE_MIN_DEFAULT;
    edshift= DAV_ED_SHIFT_DEFAULT;
    sparsed_Hc = 0;
    
}

int dav_par::read_group(char * inp){
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, dav_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, dav_group_start)==0){
        return 0;
    }
    
    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, dav_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    
    
    
    return 0;
}

int dav_par::read_line(char * inp){
    
    if(key_word_comp(inp, num_state_kw)){
        n_s = kw_to_f(inp, num_state_kw, -1);
    }
    
    if(key_word_comp(inp, dav_n_bf_kw)){
        n_bf = kw_to_i(inp, dav_n_bf_kw, DAV_N_BF_DEFAULT);
    }
    
    if(key_word_comp(inp, max_it_kw)){
        max_it = kw_to_i(inp, max_it_kw, DAV_MAX_IT_DEFAULT);
    }
    
    if(key_word_comp(inp, e_conv_kw)){
        e_conv = kw_to_f(inp, e_conv_kw, DAV_EN_CON_DEFAULT);
    }
    
    if(key_word_comp(inp, r_conv_kw)){
        r_conv = kw_to_f(inp, r_conv_kw, DAV_R_CON_DEFAULT);
    }
    
    if(key_word_comp(inp, dav_se_kw)){
        se_min = kw_to_f(inp, dav_se_kw, DAV_SE_MIN_DEFAULT);
    }
    
    if(key_word_comp(inp, edshift_kw)){
        edshift = kw_to_f(inp, edshift_kw, DAV_ED_SHIFT_DEFAULT);
    }

    if(key_word_comp(inp, dav_sparsed_HC_kw)){
        sparsed_Hc = kw_to_i(inp, dav_sparsed_HC_kw, 0);
    }

    
    
    return 0;
}

int dav_par::write_info(){
    
    fprintf(out_stream,"Davidson diagonalization settings:\n");
    fprintf(out_stream,"Number of correlation vectors:    %d\n",n_s    );
    fprintf(out_stream,"Number of basis functions:        %d\n",n_bf   );
    fprintf(out_stream,"Maximum number of iterations:     %d\n",max_it );
    fprintf(out_stream,"Energy convergence:               %e\n",e_conv );
    fprintf(out_stream,"Residue convergence:              %e\n",r_conv );
    fprintf(out_stream,"Orthogonality cut-off:            %e\n",se_min );
    fprintf(out_stream,"Energy denominator shift:         %e\n",edshift);
    fprintf(out_stream,"Sparsed multiplication:           %s\n",sparsed_Hc?"yes":"no");
    fprintf(out_stream,"\n");
    
    return 0;
}

dav_par::~dav_par(){

}

//-----------------------------------------------------------------------------------------------------

//DMRG
dmrg_par::dmrg_par(){

    m         = DMRG_M_DEFAULT;          // 0 = unset; required when CISOLVER=dmrg
    sweeps    = DMRG_SWEEPS_DEFAULT;
    sweep_tol = DMRG_SWEEP_TOL_DEFAULT;
    hf_occ    = DMRG_HF_OCC_INTEGRAL;
    schedule  = DMRG_SCHED_DEFAULT;
    save_dir  = DMRG_SAVE_DIR_DEFAULT;
    memory    = DMRG_MEMORY_DEFAULT;
    main_stack = DMRG_MAIN_STACK_DEFAULT;
    rdm_passes = DMRG_RDM_PASSES_DEFAULT;
    localize  = DMRG_LOC_OFF;
    dump_loc_orbs = 0;
    loc_order = DMRG_LOCORDER_FIEDLER;
    warm_start       = DMRG_WARM_START_DEFAULT;
    warm_sweeps      = DMRG_WARM_SWEEPS_DEFAULT;
    warm_noise_scale = DMRG_WARM_NOISE_SCALE_DEFAULT;
    rot_m            = DMRG_ROT_M_DEFAULT;
    rot_steps        = DMRG_ROT_STEPS_DEFAULT;
    warm_start_after = DMRG_WARM_START_AFTER_DEFAULT;
    warm_rotate      = DMRG_WARM_ROTATE_DEFAULT;
    print_dets       = DMRG_PRINT_DETS_DEFAULT;
    det_rot_m        = DMRG_DET_ROT_M_DEFAULT;
    det_rot_steps    = DMRG_DET_ROT_STEPS_DEFAULT;
    extract_m        = DMRG_EXTRACT_M_DEFAULT;
    extract_cutoff   = DMRG_EXTRACT_CUTOFF_DEFAULT;
    h2caa_m          = DMRG_H2CAA_M_DEFAULT;
    low_m_opt        = DMRG_LOW_M_OPT_DEFAULT;

}

int dmrg_par::read_group(char * inp){

    recursive_file P;
    char line[BUF_LINE_LENGTH];

    P.r_open(inp);

    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, dmrg_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, dmrg_group_start)==0){
        return 0;
    }

    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, dmrg_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }

    return 0;
}

int dmrg_par::read_line(char * inp){

    if(key_word_comp(inp, dmrg_m_kw)){
        m = kw_to_i(inp, dmrg_m_kw, DMRG_M_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_sweeps_kw)){
        sweeps = kw_to_i(inp, dmrg_sweeps_kw, DMRG_SWEEPS_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_sweep_tol_kw)){
        sweep_tol = kw_to_f(inp, dmrg_sweep_tol_kw, DMRG_SWEEP_TOL_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_hf_occ_kw)){
        if(kw_to_kw(inp, dmrg_hf_occ_kw, dmrg_hf_occ_integral_kw)) hf_occ = DMRG_HF_OCC_INTEGRAL;
        else                                                       hf_occ = DMRG_HF_OCC_UNKNOWN;
    }

    if(key_word_comp(inp, dmrg_schedule_kw)){
        if(kw_to_kw(inp, dmrg_schedule_kw, dmrg_schedule_default_kw)) schedule = DMRG_SCHED_DEFAULT;
        else                                                          schedule = DMRG_SCHED_UNKNOWN;
    }

    if(key_word_comp(inp, dmrg_save_dir_kw)){
        char* tmp=nullptr;
        kw_to_s(&tmp, inp, dmrg_save_dir_kw);
        if(tmp){ save_dir=tmp; delete[] tmp; }
    }

    if(key_word_comp(inp, dmrg_memory_kw)){
        memory = kw_to_f(inp, dmrg_memory_kw, DMRG_MEMORY_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_main_stack_kw)){
        main_stack = kw_to_f(inp, dmrg_main_stack_kw, DMRG_MAIN_STACK_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_rdm_passes_kw)){
        rdm_passes = kw_to_i(inp, dmrg_rdm_passes_kw, DMRG_RDM_PASSES_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_localize_kw)){
        if     (kw_to_kw(inp, dmrg_localize_kw, dmrg_localize_off_kw))  localize = DMRG_LOC_OFF;
        else if(kw_to_kw(inp, dmrg_localize_kw, dmrg_localize_pm_kw))   localize = DMRG_LOC_PM;
        else if(kw_to_kw(inp, dmrg_localize_kw, dmrg_localize_boys_kw)) localize = DMRG_LOC_BOYS;
        else                                                           localize = DMRG_LOC_UNKNOWN;
    }

    if(key_word_comp(inp, dmrg_dump_loc_kw)){
        dump_loc_orbs = 1;
    }

    if(key_word_comp(inp, dmrg_loc_order_kw)){
        if     (kw_to_kw(inp, dmrg_loc_order_kw, dmrg_loc_order_fiedler_kw)) loc_order = DMRG_LOCORDER_FIEDLER;
        else if(kw_to_kw(inp, dmrg_loc_order_kw, dmrg_loc_order_gaopt_kw))   loc_order = DMRG_LOCORDER_GAOPT;
        else if(kw_to_kw(inp, dmrg_loc_order_kw, dmrg_loc_order_none_kw))    loc_order = DMRG_LOCORDER_NONE;
        else                                                                loc_order = DMRG_LOCORDER_UNKNOWN;
    }

    if(key_word_comp(inp, dmrg_warm_start_kw)){
        if     (kw_to_kw(inp, dmrg_warm_start_kw, dmrg_warm_off_kw)) warm_start = DMRG_WARM_OFF;
        else if(kw_to_kw(inp, dmrg_warm_start_kw, dmrg_warm_on_kw))  warm_start = DMRG_WARM_ON;
        else                                                         warm_start = DMRG_WARM_UNKNOWN;
    }

    if(key_word_comp(inp, dmrg_warm_sweeps_kw)){
        warm_sweeps = kw_to_i(inp, dmrg_warm_sweeps_kw, DMRG_WARM_SWEEPS_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_warm_noise_scale_kw)){
        warm_noise_scale = kw_to_f(inp, dmrg_warm_noise_scale_kw, DMRG_WARM_NOISE_SCALE_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_rot_m_kw)){
        rot_m = kw_to_i(inp, dmrg_rot_m_kw, DMRG_ROT_M_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_rot_steps_kw)){
        rot_steps = kw_to_i(inp, dmrg_rot_steps_kw, DMRG_ROT_STEPS_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_warm_start_after_kw)){
        warm_start_after = kw_to_i(inp, dmrg_warm_start_after_kw, DMRG_WARM_START_AFTER_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_warm_rotate_kw)){
        if     (kw_to_kw(inp, dmrg_warm_rotate_kw, dmrg_warm_off_kw)) warm_rotate = DMRG_WARM_OFF;
        else if(kw_to_kw(inp, dmrg_warm_rotate_kw, dmrg_warm_on_kw))  warm_rotate = DMRG_WARM_ON;
        else                                                          warm_rotate = DMRG_WARM_UNKNOWN;
    }

    if(key_word_comp(inp, dmrg_print_dets_kw)){
        if     (kw_to_kw(inp, dmrg_print_dets_kw, dmrg_warm_off_kw)) print_dets = DMRG_WARM_OFF;
        else if(kw_to_kw(inp, dmrg_print_dets_kw, dmrg_warm_on_kw))  print_dets = DMRG_WARM_ON;
        else                                                         print_dets = DMRG_WARM_UNKNOWN;
    }

    if(key_word_comp(inp, dmrg_det_rot_m_kw)){
        det_rot_m = kw_to_i(inp, dmrg_det_rot_m_kw, DMRG_DET_ROT_M_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_det_rot_steps_kw)){
        det_rot_steps = kw_to_i(inp, dmrg_det_rot_steps_kw, DMRG_DET_ROT_STEPS_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_extract_m_kw)){
        extract_m = kw_to_i(inp, dmrg_extract_m_kw, DMRG_EXTRACT_M_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_extract_cutoff_kw)){
        extract_cutoff = kw_to_f(inp, dmrg_extract_cutoff_kw, DMRG_EXTRACT_CUTOFF_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_h2caa_m_kw)){
        h2caa_m = kw_to_i(inp, dmrg_h2caa_m_kw, DMRG_H2CAA_M_DEFAULT);
    }

    if(key_word_comp(inp, dmrg_low_m_opt_kw)){
        if     (kw_to_kw(inp, dmrg_low_m_opt_kw, dmrg_warm_off_kw)) low_m_opt = DMRG_LOW_M_OFF;
        else if(kw_to_kw(inp, dmrg_low_m_opt_kw, dmrg_warm_on_kw))  low_m_opt = DMRG_LOW_M_ON;
        else                                                        low_m_opt = DMRG_LOW_M_UNKNOWN;
    }

    return 0;
}

int dmrg_par::validate(){

    int ok=1;

    if(m<=0){
        fprintf(out_stream,"ERROR: $DMRG bond dimension m=%d must be > 0 (set 'm' in $DMRG)\n",m);
        ok=0;
    }
    if(sweeps<=0){
        fprintf(out_stream,"ERROR: $DMRG sweeps=%d must be > 0\n",sweeps);
        ok=0;
    }
    if(warm_sweeps<=0){ warm_sweeps = sweeps/2; if(warm_sweeps<1) warm_sweeps = 1; } // auto = sweeps/2
    if(sweep_tol<=0){
        fprintf(out_stream,"ERROR: $DMRG sweep_tol=%e must be > 0\n",sweep_tol);
        ok=0;
    }
    if(hf_occ==DMRG_HF_OCC_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown hf_occ value; accepted: integral\n");
        ok=0;
    }
    if(schedule==DMRG_SCHED_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown schedule value; accepted: default\n");
        ok=0;
    }
    if(localize==DMRG_LOC_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown localize value; accepted: off, pm, boys\n");
        ok=0;
    }
    if(localize==DMRG_LOC_BOYS){
        fprintf(out_stream,"ERROR: $DMRG localize=boys not implemented yet; accepted: off, pm\n");
        ok=0;
    }
    if(loc_order==DMRG_LOCORDER_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown loc_order value; accepted: fiedler, gaopt, none\n");
        ok=0;
    }
    if(save_dir.empty()){
        fprintf(out_stream,"ERROR: $DMRG save_dir must not be empty\n");
        ok=0;
    }
    if(memory<=0){
        fprintf(out_stream,"ERROR: $DMRG memory=%g must be > 0 (block2 double-stack size in GB)\n",memory);
        ok=0;
    }
    if(rdm_passes<1){
        fprintf(out_stream,"ERROR: $DMRG rdm_passes=%d must be >= 1\n",rdm_passes);
        ok=0;
    }
    if(main_stack!=0 && (main_stack<=0 || main_stack>=memory)){
        fprintf(out_stream,"ERROR: $DMRG main_stack=%g must be > 0 and < memory=%g GB\n",
                main_stack,memory);
        ok=0;
    }
    if(warm_start==DMRG_WARM_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown warm_start value; accepted: off, on\n");
        ok=0;
    }
    if(warm_rotate==DMRG_WARM_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown warm_rotate value; accepted: off, on\n");
        ok=0;
    }
    if(print_dets==DMRG_WARM_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown print_dets value; accepted: off, on\n");
        ok=0;
    }
    if(det_rot_m<=0){ det_rot_m = 2*m < 1500 ? 2*m : 1500; } // auto = min(2m, 1500)
    if(det_rot_steps<=0){
        fprintf(out_stream,"ERROR: $DMRG det_rot_steps=%d must be > 0\n",det_rot_steps);
        ok=0;
    }
    if(extract_m<0){
        fprintf(out_stream,"ERROR: $DMRG extract_m=%d must be >= 0 (0 = no compression)\n",extract_m);
        ok=0;
    }
    if(extract_cutoff<=0){
        fprintf(out_stream,"ERROR: $DMRG extract_cutoff=%e must be > 0\n",extract_cutoff);
        ok=0;
    }
    if(h2caa_m<0){
        fprintf(out_stream,"ERROR: $DMRG h2caa_m=%d must be >= 0 (0 = auto: 2m)\n",h2caa_m);
        ok=0;
    }
    if(low_m_opt==DMRG_LOW_M_UNKNOWN){
        fprintf(out_stream,"ERROR: $DMRG unknown low_m_opt value; accepted: off, on\n");
        ok=0;
    }
    if(warm_start==DMRG_WARM_ON){
        if(warm_noise_scale<0){
            fprintf(out_stream,"ERROR: $DMRG warm_noise_scale=%g must be >= 0 (0 = noise-free warm re-solve)\n",warm_noise_scale);
            ok=0;
        }
        if(rot_m<0){
            fprintf(out_stream,"ERROR: $DMRG rot_m=%d must be >= 0 (0 = use m)\n",rot_m);
            ok=0;
        }
        if(rot_steps<=0){
            fprintf(out_stream,"ERROR: $DMRG rot_steps=%d must be > 0\n",rot_steps);
            ok=0;
        }
        if(warm_start_after<0){
            fprintf(out_stream,"ERROR: $DMRG warm_start_after=%d must be >= 0\n",warm_start_after);
            ok=0;
        }
    }

    if(!ok)exit(1);

    return 0;
}

int dmrg_par::write_info(){

    fprintf(out_stream,"DMRG settings:\n");
    fprintf(out_stream,"Bond dimension (m):               %d\n",m);
    fprintf(out_stream,"Maximum number of sweeps:         %d\n",sweeps);
    fprintf(out_stream,"Sweep energy convergence:         %e\n",sweep_tol);
    if(hf_occ==DMRG_HF_OCC_INTEGRAL)
        fprintf(out_stream,"Initial occupancy:                integral\n");
    if(schedule==DMRG_SCHED_DEFAULT)
        fprintf(out_stream,"Sweep schedule:                   default\n");
    if(localize==DMRG_LOC_OFF)
        fprintf(out_stream,"Active-space localization:        off\n");
    if(localize==DMRG_LOC_PM)
        fprintf(out_stream,"Active-space localization:        Pipek-Mezey\n");
    if(dump_loc_orbs)
        fprintf(out_stream,"Dump localized orbitals:          yes\n");
    if(loc_order==DMRG_LOCORDER_FIEDLER)
        fprintf(out_stream,"DMRG orbital ordering:            Fiedler\n");
    if(loc_order==DMRG_LOCORDER_GAOPT)
        fprintf(out_stream,"DMRG orbital ordering:            GAopt (genetic, seeded)\n");
    if(loc_order==DMRG_LOCORDER_NONE)
        fprintf(out_stream,"DMRG orbital ordering:            none (input order)\n");
    fprintf(out_stream,"Scratch directory (save_dir):     %s\n",save_dir.c_str());
    fprintf(out_stream,"Memory (block2 double stack):     %g GB\n",memory);
    if(main_stack>0)
        fprintf(out_stream,"Main double stack:                %g GB\n",main_stack);
    fprintf(out_stream,"RDM sweep passes:                 %d\n",rdm_passes);
    fprintf(out_stream,"Partition files:                  fp_codec cutoff %g, chunk %d\n",
            (double)DMRG_FP_CODEC_CUTOFF,(int)DMRG_FP_CODEC_CHUNK);
    if(low_m_opt==DMRG_LOW_M_AUTO)
        fprintf(out_stream,"Low-m MPO optimization:           auto\n");
    if(low_m_opt==DMRG_LOW_M_ON)
        fprintf(out_stream,"Low-m MPO optimization:           on\n");
    if(low_m_opt==DMRG_LOW_M_OFF)
        fprintf(out_stream,"Low-m MPO optimization:           off\n");
    if(warm_start==DMRG_WARM_ON){
        fprintf(out_stream,"MPS warm-start:                   on (after %d cold iter)\n",warm_start_after);
        fprintf(out_stream,"Warm re-solve sweeps:             %d\n",warm_sweeps);
        fprintf(out_stream,"Warm noise scale (x discarded w): %g\n",warm_noise_scale);
        fprintf(out_stream,"Rotate reused MPS:                %s\n",warm_rotate==DMRG_WARM_ON?"yes":"no (reuse-only)");
        fprintf(out_stream,"MPS-rotation bond dim (rot_m):    %d\n",rot_m==0?m:rot_m);
        if(warm_rotate==DMRG_WARM_ON)
            fprintf(out_stream,"MPS-rotation TE steps (rot_steps):%d\n",rot_steps);
    }
    else
        fprintf(out_stream,"MPS warm-start:                   off\n");
    if(print_dets==DMRG_WARM_ON){
        fprintf(out_stream,"Leading determinants:             on\n");
        fprintf(out_stream,"  read-out rotation bond dim:     %d (%d TE steps)\n",det_rot_m,det_rot_steps);
        if(extract_m>0)
            fprintf(out_stream,"  extraction compression:         m=%d, cutoff=%e\n",extract_m,extract_cutoff);
        else
            fprintf(out_stream,"  extraction:                     no compression, cutoff=%e\n",extract_cutoff);
    }
    else
        fprintf(out_stream,"Leading determinants:             off\n");
    fprintf(out_stream,"\n");

    return 0;
}

dmrg_par::~dmrg_par(){

}

//-----------------------------------------------------------------------------------------------------

static const char * avas_l_labels = "spdfghik";

// "4s" -> n=4,l=0. Non-zero return = not an nl label.
static int avas_nl_from_label(const char * lab, int * n, int * l){

    if(lab==nullptr)                       return 1;
    if(strlen(lab)!=2)                     return 1;
    if((lab[0]<'1')||(lab[0]>'9'))         return 1;

    const char * p = strchr(avas_l_labels,tolower((unsigned char)lab[1]));
    if(p==nullptr)                         return 1;

    *n = lab[0]-'0';
    *l = int(p-avas_l_labels);
    if(*n <= *l)                           return 1;

    return 0;
}

// "2:5p" -> the "5p" label bound to atom 2; a bare label is bound to atom 0.
static int avas_target_from_label(const char * lab, int * n, int * l, int * atom){

    if(lab==nullptr)                       return 1;

    const char * c = strchr(lab,':');
    if(c==nullptr){
        *atom=0;
        return avas_nl_from_label(lab,n,l);
    }

    for(const char * p=lab;p<c;p++)
        if((p[0]<'0')||(p[0]>'9'))         return 1;

    *atom = atoi(lab);
    if(*atom<1)                            return 1;

    return avas_nl_from_label(c+1,n,l);
}

avas_par::avas_par(){

    y=0;
    ref_basis = AVAS_REF_BASIS_DEFAULT;

}

int avas_par::read_group(char * inp){

    recursive_file P;
    char line[BUF_LINE_LENGTH];

    P.r_open(inp);

    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, avas_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, avas_group_start)==0){
        return 0;
    }
    y=1;

    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, avas_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }

    return 0;
}

int avas_par::read_line(char * inp){

    if(key_word_comp(inp, avas_atoms_kw)){
        int n = kw_count(inp, avas_atoms_kw, ';');
        atoms.resize(n);
        kw_to_i_v(&atoms, inp, avas_atoms_kw, n);
    }

    if(key_word_comp(inp, avas_shells_kw)){
        int n = kw_count(inp, avas_shells_kw, ';');
        std::vector<char*> lab;
        lab.resize(n);
        kw_to_s_v(&lab, inp, avas_shells_kw, n);
        shell_n.resize(n);
        shell_l.resize(n);
        shell_atom.resize(n);
        for(int i=0;i<n;i++){
            if(avas_target_from_label(lab[i],&shell_n[i],&shell_l[i],&shell_atom[i])){
                fprintf(out_stream,"ERROR: $AVAS shells= got \"%s\"; expected nl labels like 4s or 3d,"
                                   " or atom-qualified ones like 2:5p\n",lab[i]);
                exit(1);
            }
            delete[] lab[i];
        }
    }

    if(key_word_comp(inp, avas_ref_basis_kw)){
        char* tmp=nullptr;
        kw_to_s(&tmp, inp, avas_ref_basis_kw);
        if(tmp){ ref_basis=tmp; delete[] tmp; }
    }

    return 0;
}

int avas_par::validate(){

    int ok=1;

    if(atoms.size()==0){
        fprintf(out_stream,"ERROR: $AVAS needs atoms=<1-based atom list>; (no default)\n");
        ok=0;
    }
    if(shell_n.size()==0){
        fprintf(out_stream,"ERROR: $AVAS needs shells=<nl label list>; e.g. shells=4s 3d; (no default)\n");
        ok=0;
    }
    if(ref_basis.empty()){
        fprintf(out_stream,"ERROR: $AVAS ref_basis must not be empty\n");
        ok=0;
    }
    // duplicates would repeat reference functions and make the reference overlap singular
    for(int i=0;i<int(atoms.size());i++){
        if(atoms[i]<1){
            fprintf(out_stream,"ERROR: $AVAS atom index %d must be >= 1\n",atoms[i]);
            ok=0;
        }
        for(int j=i+1;j<int(atoms.size());j++)
            if(atoms[i]==atoms[j]){
                fprintf(out_stream,"ERROR: $AVAS atom %d is listed twice\n",atoms[i]);
                ok=0;
            }
    }
    for(int i=0;i<int(shell_n.size());i++)
    for(int j=i+1;j<int(shell_n.size());j++)
        if((shell_n[i]==shell_n[j])&&(shell_l[i]==shell_l[j]))
        if((shell_atom[i]==shell_atom[j])||(shell_atom[i]==0)||(shell_atom[j]==0)){
            fprintf(out_stream,"ERROR: $AVAS shell %d%c is listed twice\n",shell_n[i],avas_l_labels[shell_l[i]]);
            ok=0;
        }
    for(int i=0;i<int(shell_atom.size());i++){
        if(shell_atom[i]==0)continue;
        int listed=0;
        for(int j=0;j<int(atoms.size());j++)
            if(atoms[j]==shell_atom[i])listed=1;
        if(listed==0){
            fprintf(out_stream,"ERROR: $AVAS shell %d:%d%c names atom %d, which is not in atoms=\n",
                                shell_atom[i],shell_n[i],avas_l_labels[shell_l[i]],shell_atom[i]);
            ok=0;
        }
    }
    // a bare label applies to every listed atom, a qualified one to its own atom only
    if(shell_atom.size())
    for(int i=0;i<int(atoms.size());i++){
        int covered=0;
        for(int k=0;k<int(shell_atom.size());k++)
            if((shell_atom[k]==0)||(shell_atom[k]==atoms[i]))covered=1;
        if(covered==0){
            fprintf(out_stream,"ERROR: $AVAS atom %d has no target shell\n",atoms[i]);
            ok=0;
        }
    }

    if(!ok)exit(1);

    return 0;
}

int avas_par::write_info() const {

    fprintf(out_stream,"AVAS settings:\n");
    fprintf(out_stream,"Reference basis:                  %s\n",ref_basis.c_str());
    fprintf(out_stream,"Target atoms:                    ");
    for(int i=0;i<int(atoms.size());i++)
        fprintf(out_stream," %d",atoms[i]);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Target shells:                   ");
    for(int i=0;i<int(shell_n.size());i++){
        if(shell_atom[i])fprintf(out_stream," %d:%d%c",shell_atom[i],shell_n[i],avas_l_labels[shell_l[i]]);
        else             fprintf(out_stream," %d%c",shell_n[i],avas_l_labels[shell_l[i]]);
    }
    fprintf(out_stream,"\n\n");

    return 0;
}

avas_par::~avas_par(){

}

//XMC
xmc_par::xmc_par(){
    y=0;
    edshift=0.02;
    have_ifitd=0;
    d_only=0;
    pt1_d=1;
//     mult=0;
    gamess_file_name = NULL;
    
    n_fit    =XMC_N_FIT    ;
    n_fit_pol=XMC_N_FIT_POL;
    
}

int xmc_par::read_group(char * inp, cas_par * ext_cas){
    
    cas = ext_cas;
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, xmc_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, xmc_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $XMC group in file %s\n",inp);
        exit(1);
    }
    
    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, xmc_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    if(d_only)if(gamess_file_name==NULL){
        fprintf(out_stream,"ERROR: you must use INP=... with D_ONLY=1\n");
        exit(0);
    }
    if(d_only)if(gamess_file_name!=NULL){
        FILE * XMCFile=fopen(gamess_file_name,"r");
        if(XMCFile==NULL){
            fprintf(out_stream,"ERROR: xmc_par::read_group couldn't open file with xmc data for reading energy\n");
            fprintf(out_stream,"       file: %s\n",gamess_file_name);
            exit(1);
        }
        fclose(XMCFile);
    
    }
    

    if(have_ifitd){
//         fprintf(out_stream,"%s\n",gamess_file_name);
//         getchar();
        if(gamess_file_name==NULL){
            fprintf(out_stream,"ERROR: you must use INP=... with IFITD=1\n");
            exit(0);
        }
        ifitd_energy = xmc_ifitd_read(gamess_file_name);
    }
    
    
        
    return 0;
}

int xmc_par::read_line(char * inp){
    
    if(key_word_comp(inp, edshift_kw))
        edshift = kw_to_f(inp, edshift_kw, 0.02);
    
    if(key_word_comp(inp, xmc_d_only_kw)){
        d_only = kw_to_i(inp, xmc_d_only_kw, 0);
    }
    
    if(key_word_comp(inp, xmc_n_fit_pol_kw)){
        n_fit_pol = kw_to_i(inp, xmc_n_fit_pol_kw, XMC_N_FIT_POL);
    }
    
    if(key_word_comp(inp, xmc_n_fit_kw)){
        n_fit = kw_to_i(inp, xmc_n_fit_kw, XMC_N_FIT);
    }
        
    if(key_word_comp(inp, inp_kw)){
        kw_to_s(&gamess_file_name, inp, inp_kw);
//         fprintf(out_stream,"%s\n",gamess_file_name);
    }
    
    
//     fprintf(out_stream,"ed = %e\n",edshift);
//     getchar();
    
    if(key_word_comp(inp, ifitd_kw)){
        have_ifitd=kw_to_i(inp, ifitd_kw, 0);
    }
    
    if(key_word_comp(inp, pt1_dipole_kw))
        pt1_d = kw_to_i(inp, pt1_dipole_kw,1);

//     getchar();



    return 0;
}

int xmc_par::write_info(int n_a, int n_b, int n_o, int mult){
    
    int s;
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"_____________________Starting_XMCQDPT2_calculations____________________\n\n");
    
    fprintf(out_stream,"Number of states                  %d\n",cas->n_s );
    fprintf(out_stream,"State multiplicity                %d " ,mult);
    if(mult==1) fprintf(out_stream,"(singlet)\n");
    if(mult==2) fprintf(out_stream,"(doublet)\n");
    if(mult==3) fprintf(out_stream,"(triplet)\n");
    if(mult==4) fprintf(out_stream,"(quadruplet)\n");
    if(mult >4) fprintf(out_stream,"(high-spin)\n");
    if(mult==0) fprintf(out_stream,"(all)\n");
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Active space:\n");
    fprintf(out_stream,"Number of alpha electrons         %d\n",n_a);
    fprintf(out_stream,"Number of beta  electrons         %d\n",n_b);
    fprintf(out_stream,"Number of active orbitals         %d\n",n_o);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"State averaging:\n");
    fprintf(out_stream,"Weights:\t");
    for(auto &w: cas->w_state) fprintf(out_stream,"% .2e ",w);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Energy Denominator SHIFT          %e (a.u.)\n", edshift);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Resolvent fitting:\n");
    fprintf(out_stream,"grid size                         %d\n", n_fit);
    fprintf(out_stream,"polynomial degree                 %d\n", n_fit_pol*2+1);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Additional options:\n");
    fprintf(out_stream,"skip energy calculation an read from file        -     %c%c%c\n",ny[3*d_only]
                                                                            ,ny[3*d_only+1]
                                                                            ,ny[3*d_only+2]);
    fprintf(out_stream,"read fitting energies from file                  -     %c%c%c\n",ny[3*have_ifitd]
                                                                            ,ny[3*have_ifitd+1]
                                                                            ,ny[3*have_ifitd+2]);
    if((have_ifitd!=0)||(d_only!=0))
        fprintf(out_stream,"File with external data: %s\n",gamess_file_name);
//     dav.write_info();
    fprintf(out_stream,"\nPT first order term: %s\n",pt1_d?"yes":"no");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
    
    return 0;
    
}


xmc_par::~xmc_par(){
    
    if(gamess_file_name != NULL)
        delete[] gamess_file_name;
        
}

gno_par::gno_par(){

    mode=CDAS_MODE_NATIVE;
    skip_scalar=-1;
    m_delta=0;
    sweeps_branch=0;
    dav_branch=0.0;

}

bool gno_par::on() const{

    if     (mode==CDAS_MODE_NATIVE)    return false;
    else if(mode==CDAS_MODE_TRUNC_GNO) return true;
    else if(mode==CDAS_MODE_DELTA_GNO) return true;
    else{
        fprintf(out_stream,"ERROR: unknown cdas_mode; accepted: native, trunc_gno, delta_gno\n");
        exit(1);
    }

}

int gno_par::write_info(){

    fprintf(out_stream,"\nCDAS mode                         ");
    if     (mode==CDAS_MODE_TRUNC_GNO) fprintf(out_stream,"trunc_gno\n");
    else if(mode==CDAS_MODE_DELTA_GNO) fprintf(out_stream,"delta_gno\n");
    else                               fprintf(out_stream,"unknown\n");
    fprintf(out_stream,"GNO scalar                        %s\n",skip_scalar?"skipped":"computed");
    if(mode==CDAS_MODE_DELTA_GNO)
        fprintf(out_stream,"Branch bond dimension             %d\n",m_delta);

    return 0;
}

gno_par::~gno_par(){

}

cdas_par::cdas_par(){
    
    y=0;
    edshift=0.00;
    have_eps=0;
    gamess_file_name=NULL;
    IPEA=0;
    MPPT=0;
    HOMO=0;
    actual=0;
    sing_e=0;
    mult_e=0;
    orb_e=0;
    fit_e=0;
    rotate_orbs=1;//default - perform rotation
    pt1_d=1;
    m_cdas=0;
    m_cdas_set=false;

}

int cdas_par::read_group(char * inp, cas_par * ext_cas){
    
    cas=ext_cas;
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, cdas_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, cdas_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $CDAS group in file %s\n",inp);
        exit(1);
    }
    
    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, cdas_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    
    if(gno.mode==CDAS_MODE_UNKNOWN){
        fprintf(out_stream,"ERROR: $CDAS unknown cdas_mode value; accepted: native, trunc_gno, delta_gno\n");
        exit(1);
    }

    if((IPEA+MPPT+HOMO+actual+sing_e+mult_e+orb_e+fit_e)!=1){
        fprintf(out_stream,"ERROR: incorrect choice of energy scheme\n");
        fprintf(out_stream,"       IPEA                    - %c%c%c\n",ny[3*IPEA  ],ny[3*IPEA  +1],ny[3*IPEA  +2]);
        fprintf(out_stream,"       MPPT                    - %c%c%c\n",ny[3*MPPT  ],ny[3*MPPT  +1],ny[3*MPPT  +2]);
        fprintf(out_stream,"       HOMO                    - %c%c%c\n",ny[3*HOMO  ],ny[3*HOMO  +1],ny[3*HOMO  +2]);
        fprintf(out_stream,"       ACTUAL                  - %c%c%c\n",ny[3*actual],ny[3*actual+1],ny[3*actual+2]);
        fprintf(out_stream,"       ENERGY                  - %c%c%c\n",ny[3*sing_e],ny[3*sing_e+1],ny[3*sing_e+2]);
        fprintf(out_stream,"       ENERGIES                - %c%c%c\n",ny[3*mult_e],ny[3*mult_e+1],ny[3*mult_e+2]);
        fprintf(out_stream,"       USE_ORB_FOR_ENERGY      - %c%c%c\n",ny[3*orb_e ],ny[3*orb_e +1],ny[3*orb_e +2]);
        fprintf(out_stream,"       USE_FIREFLY_FIT_ENERGY  - %c%c%c\n",ny[3*fit_e ],ny[3*fit_e +1],ny[3*fit_e +2]);
        fprintf(out_stream,"       you must choose only one.\n");
        exit(1);
    }

    if(ext_cas->ci_solver==CISOLVER_DMRG){
        if(MPPT){
            fprintf(out_stream,"ERROR: MPPT is supported by the aldet solver only; use cisolver=aldet\n");
            exit(1);
        }
        if(actual||mult_e||fit_e){
            fprintf(out_stream,"ERROR: per-orbital active energies are not supported with the dmrg solver\n");
            fprintf(out_stream,"       (active orbitals are not canonical)\n");
            fprintf(out_stream,"       use HOMO, ENERGY, USE_ORB_FOR_ENERGY or IPEA\n");
            exit(1);
        }
    }

    //the default the mode asks for: trunc_gno carries F0 in its energies, delta_gno cancels it
    if(gno.skip_scalar==-1) gno.skip_scalar = (gno.mode==CDAS_MODE_TRUNC_GNO) ? 0 : 1;

    if(gno.skip_scalar!=0&&gno.skip_scalar!=1){
        fprintf(out_stream,"ERROR: $CDAS skip_gno_scalar accepts 0|1, off|on, false|true\n");
        exit(1);
    }

    if(gno.on()&&MPPT){
        fprintf(out_stream,"ERROR: cdas_mode=trunc_gno|delta_gno run the EE and IPEA schemes only;"
                           " drop MPPT or use cdas_mode=native\n");
        exit(1);
    }

    if(gno.on()&&ext_cas->ci_solver!=CISOLVER_DMRG){
        fprintf(out_stream,"ERROR: cdas_mode=trunc_gno|delta_gno needs a CI backend with named operator handles;"
                           " use cisolver=dmrg\n");
        exit(1);
    }

    if(gno.on()){
        //the branch follows the reference solve's budget and threshold
        gno.sweeps_branch = ext_cas->dmrg.sweeps;
        gno.dav_branch    = ext_cas->dmrg.sweep_tol/10.0;

        if(gno.mode==CDAS_MODE_DELTA_GNO){
            if(gno.m_delta<=0){
                fprintf(out_stream,"ERROR: $CDAS cdas_mode=delta_gno needs m_delta=<branch bond dimension> (> 0)\n");
                exit(1);
            }
            if(gno.m_delta>ext_cas->dmrg.m){
                fprintf(out_stream,"ERROR: $CDAS m_delta=%d is above the reference bond dimension $DMRG m=%d;\n",
                        gno.m_delta,ext_cas->dmrg.m);
                fprintf(out_stream,"       m_delta above the reference bond dimension is not a compression\n");
                exit(1);
            }
        }
    }

    if(m_cdas_set&&m_cdas<=0){
        fprintf(out_stream,"ERROR: $CDAS m_cdas=<PT-stage bond dimension> must be > 0\n");
        exit(1);
    }

    if(m_cdas_set&&gno.on()){
        fprintf(out_stream,"ERROR: $CDAS m_cdas applies to cdas_mode=native only;\n");
        fprintf(out_stream,"       the GNO reference follows $DMRG m, and in delta_gno the branches follow m_delta\n");
        exit(1);
    }

    if(m_cdas_set&&ext_cas->ci_solver!=CISOLVER_DMRG){
        fprintf(out_stream,"ERROR: $CDAS m_cdas is a DMRG bond dimension; use cisolver=dmrg\n");
        exit(1);
    }

    return 0;
}

int is_splitter(char a);

//exact-token value match: case-insensitive over val, which must be closed by a splitter
//or the end of the line, so a longer value is not a hit. The "=" binds to the matched
//keyword: only blanks may stand between them, anything else is no match.
static int cdas_kw_value_is(char * inp, vector<const char *> keywords, const char * val){

    char * f = key_word_find(inp, keywords);
    if(f==nullptr) return 0;
    char * s = f;
    while(s[0]!='\0'&&!is_splitter(s[0]))s++;
    while(s[0]==' ')s++;
    if(s[0]!='=') return 0;
    while(is_splitter(s[0]))s++;
    int i=0;
    while(val[i]!='\0'&&tolower((unsigned char)s[i])==tolower((unsigned char)val[i]))i++;
    if(val[i]!='\0') return 0;
    return (s[i]=='\0'||is_splitter(s[i]));
}

int cdas_par::read_line(char * inp){
        
    if(key_word_comp(inp, edshift_kw))
        edshift = kw_to_f(inp, edshift_kw, 0.02);
    
    if(key_word_comp(inp, cdas_ipea_kw)){
        IPEA=1;
    }
    
    if(key_word_comp(inp, cdas_mppt_kw)){
        MPPT=1;
    }
    
    if(key_word_comp(inp, cdas_homo_kw)){
        HOMO=1;
    }
    
    if(key_word_comp(inp, cdas_actual_kw)){
        actual=1;
    }
    
    if(key_word_comp(inp, cdas_sing_en_kw)){
        sing_e=1;
        have_eps=0;
        eps.resize(1);
        eps[0]=kw_to_f(inp, cdas_sing_en_kw, 0.0);
    }
    
    if(key_word_comp(inp, cdas_mult_en_kw)){
        mult_e=1;
        have_eps=1;
        int n=kw_count(inp, cdas_mult_en_kw,';');
        kw_to_d_v(&eps, inp, cdas_mult_en_kw, n);
    }
    if(key_word_comp(inp, cdas_orb_en_kw)){
        orb_e=1;
        n_orb=kw_to_i(inp, cdas_orb_en_kw, -1);
    }
    
    if(key_word_comp(inp, cdas_fit_en_kw)){
        fit_e=1;
        have_eps=1;
        kw_to_s(&gamess_file_name, inp, cdas_fit_en_kw);
        eps=xmc_ifitd_read(gamess_file_name);
    }
    
    if(key_word_comp(inp, cdas_rot_orbs))
        rotate_orbs = kw_to_i(inp, cdas_rot_orbs,1);
    
    if(key_word_comp(inp, pt1_dipole_kw))
        pt1_d = kw_to_i(inp, pt1_dipole_kw,1);

    if(key_word_comp(inp, cdas_m_cdas_kw)){
        m_cdas = kw_to_i(inp, cdas_m_cdas_kw,0);
        m_cdas_set = true;
    }

    if(key_word_comp(inp, cdas_mode_kw)){
        if     (cdas_kw_value_is(inp, cdas_mode_kw, "native"   )) gno.mode = CDAS_MODE_NATIVE;
        else if(cdas_kw_value_is(inp, cdas_mode_kw, "trunc_gno")) gno.mode = CDAS_MODE_TRUNC_GNO;
        else if(cdas_kw_value_is(inp, cdas_mode_kw, "delta_gno")) gno.mode = CDAS_MODE_DELTA_GNO;
        else                                                      gno.mode = CDAS_MODE_UNKNOWN;
    }

    if(key_word_comp(inp, cdas_m_delta_kw))
        gno.m_delta = kw_to_i(inp, cdas_m_delta_kw,0);

    //a value the keyword does not accept sticks: a later line of the group cannot overwrite it
    if(key_word_comp(inp, cdas_skip_gno_scalar_kw)&&gno.skip_scalar!=2){
        if     (cdas_kw_value_is(inp, cdas_skip_gno_scalar_kw, "0"    )
              ||cdas_kw_value_is(inp, cdas_skip_gno_scalar_kw, "off"  )
              ||cdas_kw_value_is(inp, cdas_skip_gno_scalar_kw, "false")) gno.skip_scalar = 0;
        else if(cdas_kw_value_is(inp, cdas_skip_gno_scalar_kw, "1"    )
              ||cdas_kw_value_is(inp, cdas_skip_gno_scalar_kw, "on"   )
              ||cdas_kw_value_is(inp, cdas_skip_gno_scalar_kw, "true" )) gno.skip_scalar = 1;
        else                                                             gno.skip_scalar = 2;
    }

    return 0;
}

int cdas_par::write_info(int n_a, int n_b, int n_o, int mult){
    
    int s;
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"_____________________Starting_CDAS_PT2_calculations____________________\n\n");
    
    fprintf(out_stream,"Number of states                  %d\n",cas->n_s );
    fprintf(out_stream,"State multiplicity                %d " ,mult);
    if(mult==1) fprintf(out_stream,"(singlet)\n");
    if(mult==2) fprintf(out_stream,"(doublet)\n");
    if(mult==3) fprintf(out_stream,"(triplet)\n");
    if(mult==4) fprintf(out_stream,"(quadruplet)\n");
    if(mult >4) fprintf(out_stream,"(high-spin)\n");
    if(mult==0) fprintf(out_stream,"(all)\n");
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Active space:\n");
    fprintf(out_stream,"Number of alpha electrons         %d\n",n_a);
    fprintf(out_stream,"Number of beta  electrons         %d\n",n_b);
    fprintf(out_stream,"Number of active orbitals         %d\n",n_o);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"State averaging:\n");
    fprintf(out_stream,"Weights:\t");
    for(auto &w: cas->w_state) fprintf(out_stream,"% .2e ",w);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Energy Denominator SHIFT          %e (a.u.)\n", edshift);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Energy scheme                                    -     ");
    if(IPEA  )fprintf(out_stream,"IPEA\n");
    if(MPPT  )fprintf(out_stream,"MPPT\n");
    if(HOMO  )fprintf(out_stream,"HOMO\n");
    if(actual)fprintf(out_stream,"ACTUAL\n");
    if(sing_e)fprintf(out_stream,"ENERGY\n");
    if(mult_e)fprintf(out_stream,"ENERGIES\n");
    if(orb_e )fprintf(out_stream,"USE_ORB_FOR_ENERGY\n");
    if(fit_e )fprintf(out_stream,"USE_FIREFLY_FIT_ENERGY\n");
    if(fit_e!=0)
        fprintf(out_stream,"File with external data: %s\n",gamess_file_name);
    // fprintf(out_stream,"\nOrbital rotation: ");
         // if(rotate_orbs==1)fprintf(out_stream,"1 - use semi-canonical orbitals\n");
    // else if(rotate_orbs==0)fprintf(out_stream,"0 - use initial orbitals\n");
    // else if(rotate_orbs==2)fprintf(out_stream,"2 - use semi-canonic inactive and initial active orbitals\n");
    // else{
        // nopt_printf("\n\nERROR: invalid value of cdas.rotate_orbs=%d\n", rotate_orbs);
        // exit(0);
    // }
    fprintf(out_stream,"\nPT first order term: %s\n",pt1_d?"yes":"no");
    if(m_cdas_set)
        fprintf(out_stream,"PT-stage bond dimension (m_cdas)  %d\n",m_cdas);
    
    if(gno.on()) gno.write_info();

    
//     dav.write_info();
    
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
    
    return 0;
    
}


cdas_par::~cdas_par(){

    if(gamess_file_name!=NULL)delete[] gamess_file_name;

}

dsrg_par::dsrg_par(){

    y=0;
    cas=NULL;
    s=DSRG_S_DEFAULT;
    ccvv_source=DSRG_CCVV_SRC_NORMAL;
    root=DSRG_ROOT_DEFAULT;
    print=DSRG_PRINT_DEFAULT;
    sa=DSRG_SA_DEFAULT;
    relax=DSRG_RELAX_DEFAULT;

}

int dsrg_par::read_group(char * inp, cas_par * ext_cas){

    cas=ext_cas;

    recursive_file P;
    char line[BUF_LINE_LENGTH];

    P.r_open(inp);

    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, dsrg_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, dsrg_group_start)==0){
        fprintf(out_stream,"ERROR: could not find $DSRG group in file %s\n",inp);
        exit(1);
    }

    while(!P.r_eof()){
        read_line(line);
        if(key_word_comp(line, dsrg_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }

    validate();

    return 0;
}

int dsrg_par::read_line(char * inp){

    if(key_word_comp(inp, dsrg_s_kw))
        s = kw_to_f(inp, dsrg_s_kw, DSRG_S_DEFAULT);

    if(key_word_comp(inp, dsrg_ccvv_source_kw)){
        if     (kw_to_kw(inp, dsrg_ccvv_source_kw, dsrg_ccvv_normal_kw)) ccvv_source = DSRG_CCVV_SRC_NORMAL;
        else if(kw_to_kw(inp, dsrg_ccvv_source_kw, dsrg_ccvv_zero_kw))   ccvv_source = DSRG_CCVV_SRC_ZERO;
        else                                                            ccvv_source = DSRG_CCVV_SRC_UNKNOWN;
    }

    if(key_word_comp(inp, dsrg_root_kw))
        root = kw_to_i(inp, dsrg_root_kw, DSRG_ROOT_DEFAULT);

    if(key_word_comp(inp, dsrg_print_kw))
        print = kw_to_i(inp, dsrg_print_kw, DSRG_PRINT_DEFAULT);

    if(key_word_comp(inp, dsrg_sa_kw))
        sa = kw_to_i(inp, dsrg_sa_kw, DSRG_SA_DEFAULT);

    if(key_word_comp(inp, dsrg_relax_kw)){
        if     (kw_to_kw(inp, dsrg_relax_kw, dsrg_relax_none_kw)) relax = DSRG_RELAX_NONE;
        else if(kw_to_kw(inp, dsrg_relax_kw, dsrg_relax_once_kw)) relax = DSRG_RELAX_ONCE;
        else                                                      relax = DSRG_RELAX_UNKNOWN;
    }
    // sa=1 on its own yields an ensemble average only, so the multi-state step is its default;
    // an explicit relax= still wins.
    else if(sa==1)
        relax = DSRG_RELAX_ONCE;

    return 0;
}

int dsrg_par::validate(){

    int ok=1;

    if(s<=0){
        fprintf(out_stream,"ERROR: $DSRG s=%g must be > 0 (the regularizer uses sqrt(s))\n",s);
        ok=0;
    }
    if(ccvv_source==DSRG_CCVV_SRC_UNKNOWN){
        fprintf(out_stream,"ERROR: $DSRG unknown ccvv_source value; accepted: normal, zero\n");
        ok=0;
    }
    if(root<0){
        fprintf(out_stream,"ERROR: $DSRG root=%d must be >= 0\n",root);
        ok=0;
    }
    if(sa!=0 && sa!=1){
        fprintf(out_stream,"ERROR: $DSRG sa=%d must be 0 or 1\n",sa);
        ok=0;
    }
    if(relax==DSRG_RELAX_UNKNOWN){
        fprintf(out_stream,"ERROR: $DSRG unknown relax value; accepted: none, once\n");
        ok=0;
    }
    if(sa==1 && root!=DSRG_ROOT_DEFAULT){
        fprintf(out_stream,"ERROR: $DSRG sa=1 averages all states; do not also set root=%d\n",root);
        ok=0;
    }

    if(!ok) exit(1);

    return 0;
}

int dsrg_par::write_info(int n_a, int n_b, int n_o, int mult){

    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"_____________________Starting_DSRG_PT2_calculation_____________________\n\n");

    fprintf(out_stream,"Number of states                  %d\n",cas->n_s);
    if(!sa) fprintf(out_stream,"Selected root (state-specific)    %d\n",root);
    fprintf(out_stream,"State multiplicity                %d\n",mult);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Active space:\n");
    fprintf(out_stream,"Number of alpha electrons         %d\n",n_a);
    fprintf(out_stream,"Number of beta  electrons         %d\n",n_b);
    fprintf(out_stream,"Number of active orbitals         %d\n",n_o);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Flow parameter s                  %g\n",s);
    fprintf(out_stream,"CCVV source                       %s\n",ccvv_source==DSRG_CCVV_SRC_ZERO?"zero":"normal");
    fprintf(out_stream,"State averaging (sa)              %s\n",sa?"on (ensemble reference)":"off (state-specific)");
    fprintf(out_stream,"Reference relaxation              %s\n",relax==DSRG_RELAX_ONCE?"once":"none");
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");

    return 0;
}

dsrg_par::~dsrg_par(){

}

int print_dipole     = PRINT_DIPOLE;
int print_dispersion = PRINT_DISP;
int print_quadrupole = PRINT_QUADRUPOLE;
int print_mulliken   = PRINT_MULLIKEN;

int read_prop_line(char * inp){
    
    if(key_word_comp(inp, print_dipole_kw))
        print_dipole = kw_to_i(inp, print_dipole_kw, 1);
    
    if(key_word_comp(inp, print_dispersion_kw))
        print_dispersion = kw_to_i(inp, print_dispersion_kw, 0);
    
    if(key_word_comp(inp, print_quadrupole_kw))
        print_quadrupole = kw_to_i(inp, print_quadrupole_kw, 0);
    
    if(key_word_comp(inp, print_mulliken_kw))
        print_mulliken = kw_to_i(inp, print_mulliken_kw, 0);
    
    return 0;
}

int read_prop_group(char * inp){
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
    if(P.file[0]==NULL){
        fprintf(out_stream,"ERROR: could not open %s\n",inp);
        exit(1);
    }
//     fprintf(out_stream,"could open %s\n",inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, prop_group)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(key_word_comp(line, prop_group)==0){
        return 0;
    }
    
    
    while(true){
        read_prop_line(line);
        if(key_word_comp(line, prop_group_end))break;
        if(P.r_eof()) break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    
//     
//     if(P.r_eof()){
//         return 0;
//     }
    
    
    
    return 0;
}

inp_par::inp_par(){
    inp_name = new char[BUF_LINE_LENGTH];
    ecp=0;
    num_threads=1;
    doci_dec=0;
    out_folder_defined=0;
    out_folder_name=NULL;
    point_group=NULL;
    
    job_name = NULL;
    
    
}

int inp_par::write_info(){
    
    int s;
    fprintf(out_stream,"\n\n\n");
    fprintf(out_stream,"_____________________________Input_summary_____________________________\n\n");
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Number of threads: % d\n",num_threads);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Job name: %s\n",job_name);
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Methods:\n");
    fprintf(out_stream,"Restricted Hartree-Fock                          -     %c%c%c\n",ny[3*rhf.y],ny[3*rhf.y+1],ny[3*rhf.y+2]);
    fprintf(out_stream,"Complete Active Space SCF                        -     %c%c%c\n",ny[3*cas.y],ny[3*cas.y+1],ny[3*cas.y+2]);
    fprintf(out_stream,"eXtended MultiConfiguration QuasiDegenerate PT   -     %c%c%c\n",ny[3*xmc.y],ny[3*xmc.y+1],ny[3*xmc.y+2]);
    fprintf(out_stream,"Complete Degenerate Active Space PT              -     %c%c%c\n",ny[3*cdas.y],ny[3*cdas.y+1],ny[3*cdas.y+2]);
    fprintf(out_stream,"Second order Moller-Plesset PT                   -     %c%c%c\n",ny[3*mp2.y],ny[3*mp2.y+1],ny[3*mp2.y+2]);
    fprintf(out_stream,"State-specific DSRG-PT2                          -     %c%c%c\n",ny[3*dsrg.y],ny[3*dsrg.y+1],ny[3*dsrg.y+2]);
    fprintf(out_stream,"Settings:\n");
    fprintf(out_stream,"D5                                               -     %c%c%c\n",ny[3*D5],ny[3*D5+1],ny[3*D5+2]);
    fprintf(out_stream,"Resolution of Identity                           -     %c%c%c\n",ny[3*RI],ny[3*RI+1],ny[3*RI+2]);
    fprintf(out_stream,"Spin-orbit                                       -     %c%c%c\n",ny[3*SO],ny[3*SO+1],ny[3*SO+2]);
    fprintf(out_stream,"Symmetry point group                             -     ");if(IS_SYM)fprintf(out_stream,"%s\n",point_group);
                                                                      else      fprintf(out_stream,"no(C1)\n");
    fprintf(out_stream,"Printing:\n");
    fprintf(out_stream,"dipole moment                                    -     %c%c%c\n",ny[3*print_dipole    ],ny[3*print_dipole    +1],ny[3*print_dipole    +2]);
    fprintf(out_stream,"electron dispersion                              -     %c%c%c\n",ny[3*print_dispersion],ny[3*print_dispersion+1],ny[3*print_dispersion+2]);
    fprintf(out_stream,"quadrupole moment                                -     %c%c%c\n",ny[3*print_quadrupole],ny[3*print_quadrupole+1],ny[3*print_quadrupole+2]);
    fprintf(out_stream,"Mulliken charges                                 -     %c%c%c\n",ny[3*print_mulliken  ],ny[3*print_mulliken  +1],ny[3*print_mulliken  +2]);
    
    if(backup.h1!=1){
    fprintf(out_stream,"\n");
    fprintf(out_stream,"Backup settings:\n");
    fprintf(out_stream,"- 1el data: "); backup_code_print(backup.h1);
//     fprintf(out_stream,"- 2el data: "); backup_code_print(backup.h2);
//     fprintf(out_stream,"- RI  data: "); backup_code_print(backup.ri);
    fprintf(out_stream,"- prefix  : %s\n", backup.prefix);
    }

    
    fprintf(out_stream,"_______________________________________________________________________\n\n\n");
    
//     exit(0);
    return 0;
    
}

int inp_par::read_p_group(char * inp){
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
    if(P.file[0]==NULL){
        fprintf(out_stream,"ERROR: could not open %s\n",inp);
        exit(1);
    }
//     fprintf(out_stream,"could open %s\n",inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);
    while((key_word_comp(line, par_group)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(P.r_eof()){
        fprintf(out_stream,"ERROR: could not find $PAR group in file %s\n",inp);
        exit(1);
    }
    
    while(!P.r_eof()){
        read_p_line(line);
        if(key_word_comp(line, par_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    
    if(job_name==NULL){
        job_name = new char[BUF_LINE_LENGTH];
        sprintf(job_name,"%s\0",DEFAULT_JOB_NAME);
    }
    
    
    return 0;
}

std::vector<int> inp_par::count_groups(char * inp, std::vector<const char *>start_kw, std::vector<const char *>stop_kw){
    
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    std::vector<int> out;
    out.resize(0);
    
    P.r_open(inp);
    if(P.file[0]==NULL){
        fprintf(out_stream,"ERROR: could not open %s\n",inp);
        exit(1);
    }
    int line_num=0;
    P.r_gets(line,BUF_LINE_LENGTH);
    while(!P.r_eof()){
        if(key_word_comp(line, start_kw)){
            out.push_back(line_num);
            while(key_word_comp(line, stop_kw)==0){
                P.r_gets(line,BUF_LINE_LENGTH);line_num++;//printf("A:%d - %s\n", line_num,line);
                if(P.r_eof()){
                    fprintf(out_stream,"ERROR: could not find end of the group started at line %d\n",out.back());
                    exit(1);
                }
            }
            P.r_gets(line,BUF_LINE_LENGTH);line_num++;//printf("B:%d - %s\n", line_num,line);
        }
        else{
            P.r_gets(line,BUF_LINE_LENGTH);line_num++;//printf("C:%d - %s\n", line_num,line);
        }
    }
    
//     for(int i=0;i<out.size();i++)
//         nopt_printf("%d\n",out[i]);
    
    
    return out;
}

int inp_par::read_symm_group(char * inp){
    
    recursive_file P;
    char line[BUF_LINE_LENGTH];
    
    P.r_open(inp);
    if(P.file[0]==NULL){
        fprintf(out_stream,"ERROR: could not open %s\n",inp);
        exit(1);
    }
//     fprintf(out_stream,"could open %s\n",inp);
        
    P.r_gets(line,BUF_LINE_LENGTH);;
    while((key_word_comp(line, symm_group_start)==0)&&(!P.r_eof()))P.r_gets(line,BUF_LINE_LENGTH);;
    if(P.r_eof()){
        IS_SYM=0;
        return 0;
    }
    
    while(!P.r_eof()){
        read_symm_line(line);
        if(key_word_comp(line, symm_group_end))break;
        P.r_gets(line,BUF_LINE_LENGTH);;
    }
    
    
    
    
    return 0;
}



int inp_par::read(char * ext_inp){
    
    sprintf(inp_name,"%s",ext_inp);
    
    read_p_group(inp_name);
    
    mol_line       = count_groups(inp_name,      mol_group      ,      mol_group_end);
    
    int mc=0;
    if(cas.y)  mc=1;
    if(xmc.y)  mc=1;
    if(cdas.y) mc=1;
    if(dsrg.y) mc=1;
    if((cas.y+xmc.y+cdas.y+dsrg.y+rhf.y)==0){
        mc=1;
        nopt_printf("WARNING: no QM method chosen\n");
        nopt_printf("         probably nothing to do\n");
    }
    
    
    if(mc){
        act_space_line = count_groups(inp_name,act_space_group_start,act_space_group_end);
    
        if(mol_line.size()!=act_space_line.size()){
            nopt_printf("ERROR: inconsistent number of\n");
            nopt_printf("       $MOL       groups (%3d) and\n",      mol_line.size());
            nopt_printf("       $ACT_SPACE groups (%3d) \n"   ,act_space_line.size());
            exit(0);
        }
    }
    
    read_prop_group(inp_name);
    
    read_symm_group(inp_name);
    
    backup.read_group(inp_name);
    
    if(out_folder_defined==0){
        out_folder_name=new char[3];
        out_folder_name[0]='.';
        out_folder_name[1]='/';
        out_folder_name[2]='\0';
    }
    
    if(xmc.y) cas.y=1;
    if(cdas.y)cas.y=1;
    if(dsrg.y)cas.y=1;

    if(rhf.y)  rhf.read_group(inp_name);
    if(cas.y)  cas.read_group(inp_name);
    if(xmc.y)  xmc.read_group(inp_name,&cas);
    if(cdas.y)cdas.read_group(inp_name,&cas);
    if(dsrg.y)dsrg.read_group(inp_name,&cas);
    if(cis.y)  cis.read_group(inp_name);
    if(mp2.y)  mp2.read_group(inp_name);

    avas.read_group(inp_name);
    if(avas.y) avas.validate();

    
    
    
    return 0;
    
}

int inp_par::read_p_line(char * inp){
    
    if(key_word_comp(inp, num_threads_kw))
        num_threads = kw_to_i(inp, num_threads_kw, 1);
    
    
    if(key_word_comp(inp, doci_dec_kw))
        doci_dec=kw_to_i(inp, doci_dec_kw,0);
    
    if(key_word_comp(inp, out_folder_kw)){
        out_folder_defined=1;
        kw_to_s(&out_folder_name, inp, out_folder_kw);
    }
    
    if(key_word_comp(inp, d5_kw))
        D5 = kw_to_i(inp, d5_kw, 0);

    if(key_word_comp(inp, ri_kw)){
        RI = kw_to_i(inp, ri_kw, 0);
        if(RI_flag!=-1){
            if(RI_flag!=RI){
                fprintf(stdout,"ERROR: redeclaration of RI flag\n");
                fprintf(stdout,"       one or more of your .inp files containes both\n");
                fprintf(stdout,"       RI=0 and RI=1\n");
                exit(0);
            }
        }
        else RI_flag=RI;
    }
    
    if(key_word_comp(inp, so_kw))
        SO = kw_to_i(inp, so_kw, 0);
    
    
    if(key_word_comp(inp, rhf_kw))
        rhf.y=kw_to_i(inp, rhf_kw,0);
    
    if(key_word_comp(inp, cas_kw))
        cas.y=kw_to_i(inp, cas_kw,0);
    
    if(key_word_comp(inp, xmc_kw))
        xmc.y=kw_to_i(inp, xmc_kw,0);

    if(key_word_comp(inp, cdas_kw))
        cdas.y=kw_to_i(inp, cdas_kw,0);

    if(key_word_comp(inp, dsrg_kw))
        dsrg.y=kw_to_i(inp, dsrg_kw,0);

    if(key_word_comp(inp, cis_kw))
        cis.y=kw_to_i(inp, cis_kw,0);

    if(key_word_comp(inp, mp2_kw))
        mp2.y=kw_to_i(inp, mp2_kw,0);

    if(key_word_comp(inp, name_kw)){
//         out_folder_defined=1;
        kw_to_s(&(job_name), inp, name_kw);
    }
    
//     fprintf(out_stream,"nt=%d\n",num_threads);
//     char* nt;
//     nt = key_word_find(inp,num_threads_kw);
//     if(nt==NULL)fprintf(out_stream,"not found\n");
//     if(nt!=NULL)fprintf(out_stream,"%d\n",atoi(strstr(nt,"=")+1));
    
    
    return 0;
}

int inp_par::read_symm_line(char * inp){
    
    if(key_word_comp(inp, symm_point_group_kw)){
        kw_to_s(&point_group, inp, symm_point_group_kw);
        IS_SYM=1;
        if(strstr(point_group,"C1"))IS_SYM=0;
        if(key_word_comp(point_group,linear_kw))LINEAR=1;
    }
    
    
    return 0;
}

inp_par::~inp_par(){
    
    if(out_folder_name!=NULL)delete[] out_folder_name;
    if(point_group    !=NULL)delete[] point_group    ;
    if(inp_name       !=NULL)delete[] inp_name       ;
    if(job_name       !=NULL)delete[] job_name       ;

}





std::vector<int> molecule_read_by_inp_par(molecule * M, inp_par * P){
    
    char * report_name = new char[BUF_LINE_LENGTH];
    // sprintf(report_name,"%s_out_stream.txt\0",P->job_name);
    // mol_report=stdout;//fopen(report_name,"w");
    
    std::vector<int> return_info;
    
    int s = P->mol_line.size();
    
    int mc=0;
    if( P->cas.y||P->xmc.y||P->cdas.y||P->dsrg.y)mc=1;
    if((P->cas.y +P->xmc.y +P->cdas.y+P->dsrg.y+P->rhf.y)==0)mc=1;
    
    // Linking several molecules rebuilds the active space with mult=0, which the DMRG
    // backend rejects -- and the rebuild is a full determinant build, so stop before it.
    if((s>1)&&(P->cas.ci_solver==CISOLVER_DMRG)){
        fprintf(out_stream,"ERROR: DMRG does not support N_MOL>1 (got %d molecules); use CISOLVER=aldet\n",s);
        exit(EXIT_FAILURE);
    }
    
    // Linking reorders the merged orbitals, which detaches them from the atomic
    // occupations the SAD density is carried by.
    if((s>1)&&(P->rhf.guess==GUESS_SAD)){
        fprintf(out_stream,"ERROR: SAD guess does not support N_MOL>1 (got %d molecules); use GUESS=huckel\n",s);
        exit(EXIT_FAILURE);
    }

    // DMRG reads only the active-space dimensions off molecule::CI -- never the
    // determinant space, which is unbuildable at the sizes DMRG is used for.
    int ci_alloc = ALDET_ALLOC_FULL;
    if      (P->cas.ci_solver==CISOLVER_ALDET) ci_alloc = ALDET_ALLOC_FULL;
    else if (P->cas.ci_solver==CISOLVER_DMRG ) ci_alloc = ALDET_ALLOC_DIMS;
    else{
        fprintf(out_stream,"ERROR: unknown CISOLVER (%d); accepted: aldet, dmrg\n",P->cas.ci_solver);
        exit(EXIT_FAILURE);
    }
        
        
    molecule *Mf;
    Mf = new molecule[s];
        
    fprintf(out_stream,"Reading molecule:\n");
    
    
    for(int i=0;i<s;i++){

        fprintf(out_stream,"-----------------------------------------------------------------------\n");
        
        sprintf(Mf[i].source_name,"%s",P->inp_name);
        Mf[i].inp_mol_line       = P->mol_line      [i];
        if(mc) Mf[i].inp_act_space_line = P->act_space_line[i];
        
        
        Mf[i].geom_read();
//         Mf[i].geom_print();
        
        fprintf(out_stream,"                      charge   = %d\n",Mf[i].mol_charge);
        
        Mf[i].basis_name_read();
        fprintf(out_stream,"                      basis    = %s",Mf[i].basis_name);
        if(Mf[i].basis_in_lib)fprintf(out_stream," (library)\n");
        else                  fprintf(out_stream," (current folder)\n");
        
        if((Mf[i].ecp_name!=Mf[i].basis_name)||SO){
            fprintf(out_stream,"                      ecp      = %s",Mf[i].ecp_name);
            if(Mf[i].ecp_in_lib)fprintf(out_stream," (library)\n");
            else                fprintf(out_stream," (current folder)\n");
        }
        
        if(SO){
            fprintf(out_stream,"                      so       = %s",Mf[i].so_name);
            if(Mf[i].so_in_lib)fprintf(out_stream," (library)\n");
            else               fprintf(out_stream," (current folder)\n");
        }
        
        
        if(Mf[i].basis_format==0)
            Mf[i].s=basis_lib_read_gbs(Mf+i,Mf[i].basis_name,
                                       Mf[i].basis_in_lib,Mf[i].basis_start_line,
                                       &(Mf[i].lib_coef),&(Mf[i].shell_center),false, nullptr, nullptr);
        if(Mf[i].basis_format==1)
            Mf[i].s=basis_lib_read_exp(Mf+i,Mf[i].basis_name,
                                       Mf[i].basis_in_lib,Mf[i].basis_start_line,
                                       &(Mf[i].lib_coef),&(Mf[i].shell_center),false);
            
            
        if(Mf[i].ecp_format==0)ecp_lib_read_gbs (Mf+i,Mf[i].ecp_in_lib,Mf[i].ecp_start_line,Mf[i].ecp_name,"ECP");
        if(Mf[i].ecp_format==1)ecp_lib_read_exp (Mf+i,Mf[i].ecp_in_lib,Mf[i].ecp_start_line,Mf[i].ecp_name,"ECP");
        if(Mf[i].ecp_format==2)ecp_lib_read_grpp(Mf+i,Mf[i].ecp_in_lib,Mf[i].ecp_start_line,Mf[i].ecp_name,"ECP");
        if(SO){
            if(Mf[i].so_format==0)ecp_lib_read_gbs (Mf+i,Mf[i].so_in_lib,Mf[i].so_start_line,Mf[i].so_name,"SO");
            if(Mf[i].so_format==1)ecp_lib_read_exp (Mf+i,Mf[i].so_in_lib,Mf[i].so_start_line,Mf[i].so_name,"SO");
            if(Mf[i].so_format==2)ecp_lib_read_grpp(Mf+i,Mf[i].so_in_lib,Mf[i].so_start_line,Mf[i].so_name,"SO");
        }
        
        
        Mf[i].n_ao=0;
        for(auto &s: Mf[i].s)
            Mf[i].n_ao+=s.size();
        Mf[i].n_mo=Mf[i].n_ao;
        if(D5){
            Mf[i].n_mo=0;
            for(auto &s: Mf[i].s)
                Mf[i].n_mo+=s.contr[0].l*2+1;
        }

        return_info.push_back(Mf[i].s.size());
        
        if(RI){
            if(Mf[i].ri_basis_format==0)
                Mf[i].ri_s=basis_lib_read_gbs(Mf+i,Mf[i].ri_basis_name,
                                              Mf[i].ri_basis_in_lib,Mf[i].ri_basis_start_line,
                                              &(Mf[i].lib_coef_ri),&(Mf[i].ri_shell_center), true, nullptr, nullptr);
                                              
            if(Mf[i].ri_basis_format==1)
                Mf[i].ri_s=basis_lib_read_exp(Mf+i,Mf[i].ri_basis_name,
                                              Mf[i].ri_basis_in_lib,Mf[i].ri_basis_start_line,
                                              &(Mf[i].lib_coef_ri),&(Mf[i].ri_shell_center), true);
            fprintf(out_stream,"                      RI basis = %s",Mf[i].ri_basis_name);
            if(Mf[i].ri_basis_in_lib)fprintf(out_stream," (library)\n");
            else                     fprintf(out_stream," (current folder)\n");
        
        }
        
        Mf[i].MO_gen(P->rhf.guess);
        
        if(mc)Mf[i].active_space_read(P->rhf.y,1-P->cas.y,ci_alloc);
        else
            Mf[i].STATES_set_zero();
        
    }
    fprintf(out_stream,"_______________________________________________________________________\n\n");

    mol_array_link(M, Mf, s, 1);
    
    if(IS_SYM)M->S.init(P->point_group);
    
    delete[] Mf;
    
    if(LINEAR)M->lin_prepare();
    //fclose(mol_report);
    // mol_report=stdout;
    
    
    M->geom_print();
    
    M->basis_print(0);
    if(RI)M->basis_print(1);
    M->PP_print();
    
    
    
    
    fflush(out_stream);
    
    return return_info;
    
}


