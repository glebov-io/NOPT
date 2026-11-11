#ifndef NOPT_GRADIENT_RHF_H
#define NOPT_GRADIENT_RHF_H

#include "molecule.h"

// Analytic nuclear gradient of the converged closed-shell RHF energy, for the branch the energy used: exact 4-center
// ERI (RI=0) or density-fitted J and K (RI!=0, full-rank Coulomb metric). Needs a molecule on which
// RHF() has converged (MO_VEC, orb_energy) and nopt_deriv_engines_initialize().
//
// g (3*n_atoms, bohr, dE/dR) is overwritten. Returns 0 on success, nonzero if the molecule is outside
// the supported set (open shell, ECP, field, symmetry, rank-deficient RI metric): then g is untouched.
int rhf_gradient(molecule * M, double * g);

#endif
