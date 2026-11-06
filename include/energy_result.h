#ifndef NOPT_ENERGY_RESULT_H
#define NOPT_ENERGY_RESULT_H

// Machine-readable record of the final energies of a run (regression tests, analytic-gradient checks).
//
// Off by default and silent: nothing is written, and no output of the program changes, unless the
// environment variable NOPT_ENERGY_JSON names a file. Then every method appends one JSON object per
// line, with the energies in %.17g (the log prints 10 digits):
//
//   {"seq":3,"method":"xmcqdpt2","variant":"","root":-1,"n":3,"e":[...],"w":[...]}
//
//   method   rhf | cis | cas_scf | xmc_cas | xmcqdpt2 | dsrg_pt2
//   variant  qualifier within the method ("excitation" for cis, "sa" / "root" for dsrg_pt2, ...)
//   root     DSRG target root, otherwise -1
//   e        per-root energies in the order the log prints them (not an ensemble average, unless the
//            variant says so)
//   w        state weights the method used for its reference (wstate_actual), empty if none
//
// The file is opened in append mode and flushed after every record, so a run that aborts later
// still leaves the records of the methods that finished.

#include <cstdio>
#include <cstdlib>

namespace nopt_energy_result{

inline FILE * sink(){
    static FILE * f = nullptr;
    static bool tried = false;
    if(!tried){
        tried = true;
        const char * name = getenv("NOPT_ENERGY_JSON");
        if(name && *name) f = fopen(name, "a");
    }
    return f;
}

inline void write_array(FILE * f, const char * key, const double * a, int n){
    fprintf(f, ",\"%s\":[", key);
    for(int i=0; i<n; i++) fprintf(f, "%s%.17g", i ? "," : "", a[i]);
    fprintf(f, "]");
}

}

inline void energy_result_record(const char * method, const char * variant, int root,
                                 const double * e, int n, const double * w = nullptr, int nw = 0){
    FILE * f = nopt_energy_result::sink();
    if(!f) return;
    static int seq = 0;
    fprintf(f, "{\"seq\":%d,\"method\":\"%s\",\"variant\":\"%s\",\"root\":%d,\"n\":%d",
            seq++, method, variant, root, n);
    nopt_energy_result::write_array(f, "e", e, n);
    nopt_energy_result::write_array(f, "w", w, w ? nw : 0);
    fprintf(f, "}\n");
    fflush(f);
}

inline void energy_result_record(const char * method, const char * variant, int root, double e){
    energy_result_record(method, variant, root, &e, 1);
}

#endif
