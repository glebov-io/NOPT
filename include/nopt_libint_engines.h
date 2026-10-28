#ifndef _nopt_libint_engines
#define _nopt_libint_engines

# include <vector>
# include <libint2/engine.h>

# include "common_vars.h"
# include "blas_link.h"
# include "molecule.h"
# include "ecp.h"

// extern "C" void libgrpp_init();
#ifdef _USE_GRPP
extern "C"{
    #include "libgrpp.h"
}
#endif


using libint2::Engine;
using libint2::Operator;

extern std::vector<Engine> s_g_engines;//overlap
extern std::vector<Engine> s_h_engines;//overlap secondary
extern std::vector<Engine> t_g_engines;//kinetic
extern std::vector<Engine> v_g_engines;//NAI
extern std::vector<Engine> e_g_engines;//ERI
extern std::vector<Engine> d_g_engines;//dipole
extern std::vector<Engine> q_g_engines;//quadrupole
extern std::vector<Engine> r_g_engines;//resolution of identity 3-center
extern std::vector<Engine> r2g_engines;//resolution of identity 2-center

#ifdef _USE_GRPP
extern nopt_grpp grpp_engine;//ECP
#endif


inline int nopt_engines_initialize(){
    
    libint2::initialize();
    
    s_g_engines.resize(num_threads);
    s_h_engines.resize(num_threads);
    d_g_engines.resize(num_threads);
    q_g_engines.resize(num_threads);
    t_g_engines.resize(num_threads);
    v_g_engines.resize(num_threads);
    if(RI==0)
        e_g_engines.resize(num_threads);
    if(RI==1){
        r_g_engines.resize(num_threads);
        r2g_engines.resize(num_threads);
    }
        
    
//     q_g_engines[0].set_params(std::array<double,3>{0.0, 0.0, 0.0});
    
#pragma omp parallel for
    for(int i=0; i<num_threads;i++){
        s_g_engines[i] = Engine(Operator::overlap,GTO_MAX,L_MAX);
        s_h_engines[i] = Engine(Operator::overlap,GTO_MAX,L_MAX);
        t_g_engines[i] = Engine(Operator::kinetic,GTO_MAX,L_MAX);
        v_g_engines[i] = Engine(Operator::nuclear,GTO_MAX,L_MAX);
        if(RI==0)
            e_g_engines[i] = Engine(Operator::coulomb,GTO_MAX,L_MAX);
        d_g_engines[i] = Engine(Operator::emultipole1,GTO_MAX,L_MAX);
        d_g_engines[i].set_params(std::array<double,3>{0.0, 0.0, 0.0});
        q_g_engines[i] = Engine(Operator::emultipole2,GTO_MAX,L_MAX);
        q_g_engines[i].set_params(std::array<double,3>{0.0, 0.0, 0.0});
        if(RI==1){
#ifdef _OLD_LIBINT
            r_g_engines[i] = Engine(Operator::coulomb,GTO_MAX,RI_L_MAX);
            r2g_engines[i] = Engine(Operator::coulomb,GTO_MAX,RI_L_MAX);
            r_g_engines[i].set_braket(libint2::BraKet::xs_xx);//for libint 2.4
            r2g_engines[i].set_braket(libint2::BraKet::xs_xs);//for libint 2.4
#else
            // built with their braket: set() after a default (xx_xx) build keeps GTO_MAX^4 scratch
            r_g_engines[i] = Engine(Operator::coulomb,GTO_MAX,RI_L_MAX,0,std::numeric_limits<double>::epsilon(),
                                    libint2::default_params(Operator::coulomb),libint2::BraKet::xs_xx);
            r2g_engines[i] = Engine(Operator::coulomb,GTO_MAX,RI_L_MAX,0,std::numeric_limits<double>::epsilon(),
                                    libint2::default_params(Operator::coulomb),libint2::BraKet::xs_xs);
#endif
        }
    }
    
    
#ifdef _USE_GRPP
//     grpp_engine.init(A);
    libgrpp_init();
#endif
    
    engines_initialized=1;
    
    return 0;
}

inline int nopt_engines_link_mol(molecule * A){
    
    
    if(engines_initialized==0){
        printf("nopt_engines_initialize() must be launched before start of calculation\n");
        exit(0);
    }
    
    for(int i=0; i<num_threads;i++){
        v_g_engines[i].set_params( A->libint_point_charges());
    }
#ifdef _USE_GRPP
    grpp_engine.set_mol(A);
#endif
    
    return 0;
}

inline int nopt_initialize(int ext_num_threads){
    
    omp_set_dynamic(0);
    // No NOPT region nests; an MKL libblock2 turns nesting on and MKL dynamic off when it loads.
    omp_set_max_active_levels(1);
#ifdef _MKL
    mkl_set_dynamic(1);
#endif
    nopt_parallel_init(ext_num_threads);
    //num_threads is set by nopt_parallel_init
    omp_set_num_threads(num_threads);
    set_blas_par();
    
    nopt_engines_initialize();
    
    return 0;
}

inline int nopt_finalize(){
    
    nopt_parallel_finalize();
    
    libint2::finalize();
#ifdef _USE_GRPP
    libgrpp_finalize();
#endif
    
    return 0;
}

#endif
