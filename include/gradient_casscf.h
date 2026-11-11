#ifndef NOPT_GRADIENT_CASSCF_H
#define NOPT_GRADIENT_CASSCF_H

#include "molecule.h"

// Analytic nuclear gradient of the state-averaged CAS-SCF energy  E_SA = sum_K w_K E_K / sum_K w_K,
// at a converged CAS-SCF (state-specific is the case of a single state): the energy is variational in the orbitals
// and in the CI vectors, so no response is needed, only the Hellmann-Feynman and Pulay terms,
//   g = V_NN' + D:h' + 1/2 J-K type RI terms + active-space term - W:S',
// with W the energy-weighted density of the unconstrained orbital gradient. RI branch (the energy CAS uses
// for its two-electron integrals); C1, all-electron, no field. Individual-root gradients are NOT provided here.
//
// g (3*n_atoms, bohr, dE/dR) is overwritten; *E_out receives the SA energy rebuilt from the densities and
// the integrals (the "functional identity": it must equal the cas_scf energy of the run). Returns 0 on success.
int casscf_sa_gradient(molecule * M, double * g, double * E_out);

#endif
