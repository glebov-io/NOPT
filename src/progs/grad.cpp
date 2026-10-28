// grad: single point plus analytic nuclear gradient.
//   run_grad -i input.inp [-root N] [-json FILE] [-numgrad h]
// Runs the energy calculation exactly as run_sQM does, then differentiates the energy of the method the input selects:
//   RHF (exact and RI);  SA-CASSCF (RI, determinant CI): the state-averaged energy, or with -root N the energy of root N;
//   DSRG-PT2 (RI, determinant CI, unrelaxed or relax=once): the energy the $DSRG group defines (root=, or the ensemble with sa=1).
// Anything else is refused, not approximated. Weights: any nonnegative SA weights (zero-weight roots are spectators).
// The gradient is printed (Eh/bohr, dE/dR) and, with -json FILE (or NOPT_GRAD_JSON=FILE), appended as a JSON line.
// -root N overrides NOPT_GRAD_ROOT. NOPT_GRAD_REPEAT=1: a second call in the same process must reproduce the first (exit code 6 otherwise).
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

#include "molecule.h"
#include "inp_par_read.h"
#include "inp_out.h"
#include "common_vars.h"
#include "nopt_libint_engines.h"
#include "QM_calc.h"
#include "integral_derivatives.h"
#include "gradient_rhf.h"
#include "gradient_casscf.h"
#include "casscf_resp.h"
#include "gradient_result.h"
#include "gradient_driver.h"
#include "dsrg_grad.h"
#include "grabbers.h"
#include "RI.h"
#include <omp.h>

int main(int argc, char ** argv){
    out_stream = stdout;
    nQM = 0;
    molecule M;
    char * P_name = nullptr;
    inp_par P;
    int root_cli = -1;
    double numgrad_h = 0.0;   // -numgrad h: also the numerical gradient NOPT's optimizer uses (central differences, step h bohr, in-process), timed
    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "-root") && i + 1 < argc){ root_cli = atoi(argv[++i]); continue; }
        if(!strcmp(argv[i], "-numgrad") && i + 1 < argc){ numgrad_h = atof(argv[++i]); continue; }
        if(!strcmp(argv[i], "-json") && i + 1 < argc){ setenv("NOPT_GRAD_JSON", argv[++i], 1); continue; }
        if(!strcmp(argv[i], "-i") && i + 1 < argc){ P_name = argv[++i]; continue; }
    }
    if(!P_name){ fprintf(stderr, "usage: run_grad -i input [-root N] [-json FILE]\n"); return 2; }
    if(root_cli >= 0){ char b[32]; snprintf(b, sizeof b, "%d", root_cli); setenv("NOPT_GRAD_ROOT", b, 1); }
    P.read(P_name);
    nopt_initialize(P.num_threads);
    molecule_read_by_inp_par(&M, &P);
    nopt_engines_link_mol(&M);

    const char * method = analytic_gradient_method(&P);
    if(!method) return 3;
    const char * root_env = getenv("NOPT_GRAD_ROOT");
    const bool is_rhf = !strcmp(method, "rhf"), is_dsrg = !strcmp(method, "dsrg_pt2");
    int root = (root_env && *root_env) ? atoi(root_env) : -1;
    if(is_rhf) root = -1;
    std::vector<double> g(3 * M.n_atoms);
    double E = 0.0;
    const double t_start = omp_get_wtime();
    const int rc = analytic_gradient(&P, &M, root, g.data(), &E, &method);
    const double t_analytic = omp_get_wtime() - t_start;
    if(rc) return rc;

    const char * title = is_rhf ? "RHF" : is_dsrg ? (P.dsrg.relax ? "DSRG-PT2 (relax=once)" : "DSRG-PT2 (unrelaxed)") : (root >= 0 ? "CAS-SCF root" : "SA-CASSCF");
    fprintf(out_stream, "\n\nAnalytic %s gradient (%s), E = % .12f, Eh/bohr, dE/dR:\n", title, RI ? "RI" : "exact", E);
    for(int a = 0; a < M.n_atoms; a++)
        fprintf(out_stream, "  %-3s % .12f % .12f % .12f\n", M.atom_names[a], g[3 * a], g[3 * a + 1], g[3 * a + 2]);
    gradient_result_record(method, RI ? "ri" : "exact", root, M.n_atoms, g.data());

    if(numgrad_h > 0.0){
        // the numerical gradient as optimization_w_num_grad_calc computes it: 6 n_atoms energies, displaced +-h one coordinate at a time, in this process
        const double ta = t_analytic;
        g_dsrg_grad.want = false;      // energies only from here on
        const int idx = root >= 0 ? root : 0;
        grab_E = 1;
        single_point_calc(&P, &M);
        const double Ec = E_grabbed.size() > (size_t)idx ? E_grabbed[idx] : 0.0;
        if(std::fabs(Ec - E) > 1e-7)
            fprintf(out_stream, "\nNumerical gradient: the printed energy %d (% .10f) is not the differentiated one (% .10f) -- skipped\n", idx, Ec, E);
        else{
            const double t0 = omp_get_wtime();
            std::vector<double> gn(3 * M.n_atoms);
            for(int i = 0; i < 3 * M.n_atoms; i++){
                double Ep = 0, Em = 0;
                for(int sgn : {1, -1}){
                    M.atom_coord[i] += sgn * numgrad_h; M.sync_shell_to_geom(); nopt_engines_link_mol(&M); if(RI) regen_RI_AA(&M);
                    single_point_calc(&P, &M);
                    (sgn > 0 ? Ep : Em) = E_grabbed[idx];
                    M.atom_coord[i] -= sgn * numgrad_h;
                }
                gn[i] = (Ep - Em) / (2.0 * numgrad_h);
            }
            M.sync_shell_to_geom(); nopt_engines_link_mol(&M); if(RI) regen_RI_AA(&M);
            const double tn = omp_get_wtime() - t0;
            double d = 0; for(int i = 0; i < 3 * M.n_atoms; i++) d = std::max(d, std::fabs(gn[i] - g[i]));
            fprintf(out_stream, "\nGradient timing: analytic %.2f s (with the energy), numerical (%d energies, step %.3g bohr) %.2f s, ratio %.1f; max |numerical - analytic| = %.2e\n",
                    ta, 6 * M.n_atoms, numgrad_h, tn, tn / std::max(ta, 1e-9), d);
        }
    }

    if(getenv("NOPT_GRAD_REPEAT") && *getenv("NOPT_GRAD_REPEAT")){
        // a second call in the same process (the state the first one left behind) must give the same energy and gradient. The CAS-SCF
        // restarts from the orbitals the first call ended with, so the agreement is to the convergence of the calculation, not bitwise.
        std::vector<double> g2(3 * M.n_atoms);
        double E2 = 0.0;
        const int rc2 = analytic_gradient(&P, &M, root, g2.data(), &E2, &method);
        double d = 0; for(int i = 0; i < 3 * M.n_atoms; i++) d = std::max(d, std::fabs(g2[i] - g[i]));
        fprintf(out_stream, "\nRepeated call in the same process: rc %d, |dE| = %.2e, max |dg| = %.2e %s\n", rc2, std::fabs(E2 - E), d,
                (rc2 == 0 && std::fabs(E2 - E) < 1e-9 && d < 1e-7) ? "ok" : "MISMATCH");
        if(rc2 || std::fabs(E2 - E) >= 1e-9 || d >= 1e-7){ nopt_finalize(); return 6; }
    }

    nopt_finalize();
    return 0;
}
