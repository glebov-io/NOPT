// opt: geometry optimization with the ANALYTIC nuclear gradient.
//   run_opt -i input.inp [-root N] [-maxit 100] [-gtol 3e-5] [-nofollow]
// Same L-BFGS steps and convergence rule as optimization_w_num_grad_calc (QM_calc.cpp: |dE| < 1e-7 and max|g| < gtol, or max|step| < 1e-4 and
// max|g| < gtol), but the gradient of every geometry comes from analytic_gradient() (gradient_driver.h): the energy calculation the input selects
// and its analytic derivative, in one call. CAS-based methods follow one solution: from the second geometry on the CAS-SCF starts from the
// orbitals of the previous one (RHF is skipped; the orbitals are re-orthonormalized in the new metric) -- restarting from RHF orbitals at every
// geometry lets the CAS-SCF land on different solutions and the optimization oscillate between them. -nofollow restores the restart. A method without an analytic gradient is refused; the numerical optimizer (run_optimization) stays
// available as the independent reference. Output: one line per iteration; the trajectory (Angstrom) in <NAME>_opt.xyz.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "molecule.h"
#include "inp_par_read.h"
#include "inp_out.h"
#include "common_vars.h"
#include "timer.h"
#include "nopt_libint_engines.h"
#include "QM_calc.h"
#include "RI.h"
#include "l-bfgs_2_1.h"
#include "gradient_driver.h"

static void write_xyz(FILE * f, molecule * M, double E, int it){
    const double b2a = 1.0 / 1.889725989;
    fprintf(f, "%d\niteration %d  E = %.12f\n", M->n_atoms, it, E);
    for(int a = 0; a < M->n_atoms; a++)
        fprintf(f, "%-3s % .10f % .10f % .10f\n", M->atom_names[a], M->atom_coord[3 * a] * b2a, M->atom_coord[3 * a + 1] * b2a, M->atom_coord[3 * a + 2] * b2a);
    fflush(f);
}

int main(int argc, char ** argv){
    out_stream = stdout;
    get_start_time();     // the "Full time" of the timers, as run_sQM sets it
    nQM = 0;
    molecule M;
    char * P_name = nullptr;
    inp_par P;
    int root = -2, maxit = 100;
    bool follow = true;
    double gtol = 3e-5;
    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "-i") && i + 1 < argc){ P_name = argv[++i]; continue; }
        if(!strcmp(argv[i], "-root") && i + 1 < argc){ root = atoi(argv[++i]); continue; }
        if(!strcmp(argv[i], "-maxit") && i + 1 < argc){ maxit = atoi(argv[++i]); continue; }
        if(!strcmp(argv[i], "-gtol") && i + 1 < argc){ gtol = atof(argv[++i]); continue; }
        if(!strcmp(argv[i], "-nofollow")){ follow = false; continue; }
    }
    if(!P_name){ fprintf(stderr, "usage: run_opt -i input [-root N] [-maxit 100] [-gtol 3e-5]\n"); return 2; }
    P.read(P_name);
    nopt_initialize(P.num_threads);
    molecule_read_by_inp_par(&M, &P);
    nopt_engines_link_mol(&M);
    const char * method = analytic_gradient_method(&P);
    if(!method) return 3;
    if(root == -2) root = -1;   // SA-CASSCF: the state-averaged energy unless -root N

    char name[1024]; snprintf(name, sizeof name, "%s_opt.xyz", P.job_name);
    FILE * traj = fopen(name, "w");
    const int n_c = 3 * M.n_atoms;
    std::vector<double> g(n_c), s(n_c), H(n_c, 1.0 / 3.0);
    l_bfgs_engine BFGS;
    BFGS.init(n_c, 10);
    double E = 0.0, E_old = 0.0;
    if(analytic_gradient(&P, &M, root, g.data(), &E, &method)) return 3;
    write_xyz(traj, &M, E, 0);
    if(follow && strcmp(method, "rhf")) P.rhf.y = 0;     // follow the CAS-SCF solution from geometry to geometry
    fprintf(out_stream, "\nOPT %3d  E = % .12f  max|g| = %.3e\n", 0, E, [&]{ double m = 0; for(double v : g) m = std::max(m, std::fabs(v)); return m; }());
    int converged = 0;
    for(int it = 1; it <= maxit; it++){
        E_old = E;
        BFGS.step(s.data(), g.data(), H.data(), do_nothing, do_nothing, n_c, 0.1);
        for(int i = 0; i < n_c; i++) M.atom_coord[i] += s[i];
        M.sync_shell_to_geom();
        nopt_engines_link_mol(&M);
        if(RI) regen_RI_AA(&M);
        if(analytic_gradient(&P, &M, root, g.data(), &E, &method)){ fprintf(out_stream, "ERROR: the gradient failed at iteration %d\n", it); fclose(traj); return 3; }
        double smax = 0, gmax = 0;
        for(int i = 0; i < n_c; i++){ smax = std::max(smax, std::fabs(s[i])); gmax = std::max(gmax, std::fabs(g[i])); }
        write_xyz(traj, &M, E, it);
        fprintf(out_stream, "\nOPT %3d  E = % .12f  dE = % .3e  max|g| = %.3e  max|step| = %.3e\n", it, E, E - E_old, gmax, smax);
        if((std::fabs(E - E_old) < 1e-7 && gmax < gtol) || (smax < 1e-4 && gmax < gtol)){ converged = 1; break; }
    }
    fclose(traj);
    fprintf(out_stream, "\nOptimization %s (%s, analytic gradient); final E = % .12f; trajectory %s\n", converged ? "converged" : "did NOT converge", method, E, name);
    nopt_finalize();
    return converged ? 0 : 5;
}
