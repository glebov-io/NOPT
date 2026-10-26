#ifndef NOPT_GRADIENT_DRIVER_H
#define NOPT_GRADIENT_DRIVER_H

// One entry point for the analytic nuclear gradient: runs the energy calculation the input selects at the CURRENT geometry of M and
// differentiates the energy of that method. Callable repeatedly in one process (geometry optimization): every request is reset before the energy run.
//   RHF (exact, RI).
// Anything else is refused (no numerical substitute). On success: g (3 n_atoms, Eh/bohr, dE/dR in the coordinates of M->atom_coord, bohr),
// *E the energy that was differentiated, *method a short name; returns 0. The gradient is refused when it violates translational or
// rotational invariance beyond 1e-5 (every supported path is C1 without external fields).
#include "molecule.h"
#include "inp_par_read.h"

// Which analytic method the input selects (nullptr: none supported, a message is printed).
const char * analytic_gradient_method(const inp_par * P);

int analytic_gradient(inp_par * P, molecule * M, int root, double * g, double * E, const char ** method);

// Current and peak resident memory of the process, MB (for the timing lines of the gradients).
double grad_rss_mb();
double grad_peak_rss_mb();
#endif
