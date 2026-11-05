#ifndef NOPT_GRADIENT_RESULT_H
#define NOPT_GRADIENT_RESULT_H

// Typed result of an analytic-gradient run: one record per computed gradient,
// written as a JSON line to the file named by NOPT_GRAD_JSON (like energy_result.h for energies).
//
//   {"method":"rhf","variant":"ri","root":-1,"units":"Eh/bohr","natoms":3,"g":[...3N...]}
//
// g[3*atom + xyz] is dE/dR (a force has the opposite sign), atoms in the order of the molecule.
// variant states the energy branch that was differentiated ("exact" or "ri" for RHF).

#include <cstdio>
#include <cstdlib>

inline void gradient_result_record(const char * method, const char * variant, int root, int natoms, const double * g){
    const char * name = getenv("NOPT_GRAD_JSON");
    if(!name || !*name) return;
    FILE * f = fopen(name, "a");
    if(!f) return;
    fprintf(f, "{\"method\":\"%s\",\"variant\":\"%s\",\"root\":%d,\"units\":\"Eh/bohr\",\"natoms\":%d,\"g\":[",
            method, variant, root, natoms);
    for(int i = 0; i < 3 * natoms; i++) fprintf(f, "%s%.17g", i ? "," : "", g[i]);
    fprintf(f, "]}\n");
    fclose(f);
}

#endif
