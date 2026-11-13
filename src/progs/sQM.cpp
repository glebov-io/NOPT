//standart
# include <stdio.h>
# include <omp.h>
// # include <libint2/engine.h>

//user
# include "molecule.h"
# include "timer.h"
# include "defaults.h"
# include "doCI_matr.h"
# include "matr.h"
# include "version.h"
# include "git_stamp.h"
# include "inp_par_read.h"
# include "inp_out.h"
# include "common_vars.h"
# include "nopt_libint_engines.h"
# include "geom.h"
# include "z_matrix.h"
// # include "CIS.h"
# include "QM_calc.h"
#include "integral_derivatives.h"


int print_welcome_table(char * P_name){
    
        
    fprintf(out_stream,"\n%s\n",   FULL_NAME);
    fprintf(out_stream,"                  %s version %s\n\n", PROJ_NAME, VERSION);
    fprintf(out_stream,"\nsQM - standart QM module.\n\n");
    fprintf(out_stream,"Compilation options:\n");
    fprintf(out_stream,"1) linear algebra package - ");
#ifdef _OPENBLAS
    fprintf(out_stream,"OpenBLAS\n");
#endif
#ifdef _MKL
    fprintf(out_stream,"MKL\n");
#endif
#ifdef _NOPT_BLAS_PAR
    fprintf(out_stream,"2) non-parallel BLAS matrix multiplication in the OMP parallel cycles");
#endif
#ifndef _NOPT_BLAS_PAR
    fprintf(out_stream,"2) parallel BLAS matrix multiplication");
#endif    
    fprintf(out_stream,"\n3) build git commit %s",NOPT_GIT_COMMIT);
    fprintf(out_stream,"\n\n");

    
    
    fprintf(out_stream,"selected command  line options:\n");
    fprintf(out_stream,"-i           %s\t input file\n\n",P_name);
    
    print_current_time("Calculation started at: ");
    
    fprintf(out_stream,"\n");
    fprintf(out_stream,"_______________________________________________________________________\n");
    print_file_content(P_name);

    return 0;
}


int main ( int argc , char ** argv ) 
{ 
    
    get_start_time();
    out_stream =stdout;// fopen("log.log","w");
    

    nQM=0;
    molecule M;
    char* P_name;
    inp_par P;
    
    for(int i=1;i<argc;i++){
            if(strstr(argv[i],"-i")!=NULL){ P_name = argv[i+1];i++;}
    };
    
    print_welcome_table(P_name);
    P.read(P_name);
    
    P.write_info();
    
    nopt_initialize(P.num_threads);
    molecule_read_by_inp_par(&M,&P);
    
    nopt_engines_link_mol(&M);
    
    // M.gen_1el_data();
    // fprintf(out_stream,"\n\n");
    // printf_timer("Calculation of 1-el matrices");

    // M.check_orb_symmetry();
    // M.MO_gamess_format();
    // M.GAMESS_type_out_print("orb.out",-1);
    single_point_calc(&P, &M);
    
    if(P.opt.y){
        nopt_deriv_engines_initialize(&M);
        optimization(&P, &M,0);
    }

        
   
    nopt_finalize();
    
    
    print_current_time("\n\nCalculation finished at: ");
    
    return 0;

}
 
