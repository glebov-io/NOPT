#ifndef __avas
#define __avas

#include "molecule.h"
#include "inp_par_read.h"

// Atomic Valence Active Space. Rotates the occupied rows past the first ncore and the virtual
// rows of MO_VEC so that the orbitals overlapping the requested atomic reference shells fill the
// $ACT_SPACE window [n_cor_orb, n_cor_orb+n_act_orb) by construction: never re-sort them. All-electron
// only; under $SYMM the rotation is per irrep and atoms=/shells= must be closed under the group.
int avas_steer(molecule * M, const avas_par & avas, char * job_name);

#endif
