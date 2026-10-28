#ifndef dav_h
#define dav_h
# include "inp_par_read.h"
# include "aldet.h"
# include "CIS.h"

// written based on  SIAM J. ScI. COMPUT. Vol. 15, No. 1, pp. 62-76, January 1994
// see p. 65

class davidson_solver;

class davidson_solver 
{
    public:
        aldet_data * C;// reference to external aldet for input/output 
        CIS_engine * cis;
        int max_n_vec; // maximal number of vectors (m)
        int n_bf;      // number of basis functions
        int max_it;    // maximum number of iterations
        int max_dim  ; // maximal dimension of H and S
        double r_conv; // residue convergence criteria
        double e_conv; // energy convergence criteria
        double se_min; // orthogonality criterion for S eigenvalue
        double edshift;// energy denominator shift for r[a,i]/(E[a]-H[i,i])
        
        aldet_data V;  // inner aldet workspace
        int n_CI;
        aldet_data * V_m;  // inner aldet workspace multiple task
        
        
        
        double * x;    // H eigenvector
        double * r;    // residue (r)
        double * Hx;   // H*x
        
        double * H;
        double * E;
        double * E_old;
        
        int sparsed_Hc;
        int spin_proj; // project the vectors onto S = (mult-1)/2 (dav_par::spin_proj)
        
        
        davidson_solver();
        int set_par(aldet_data * ext_C, dav_par dav);
        int set_par_m(aldet_data * ext_C, dav_par dav,int ext_n_CI);
        int set_par_cis(CIS_engine * ext_cis, dav_par dav);
        int unity_guess(sparsed_CI_vec * s);
        int read_guess();
        int run  (int print, int read_states);
	int run_cis(int print, int read_states);
        int run_m(int print, int read_states);
        ~davidson_solver();
    
        
        
    private:
};

#endif
 
