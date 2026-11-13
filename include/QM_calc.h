# include "molecule.h"
# include "inp_par_read.h"


int single_point_calc( inp_par * P, molecule * Qm);
 
int hessian_calc( inp_par * P, molecule * Qm);
int optimization( inp_par * P, molecule * Qm, int recalc);
int num_grad_calc_Z( inp_par * P, molecule * Qm, molecule * Mm, int u);
int num_grad_calc_P( inp_par * P, molecule * Qm, molecule * Mm, int u);
int relaxed_scan_P( inp_par * P, molecule * Qm, molecule * Mm, int u);

