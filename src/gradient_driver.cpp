#include "gradient_driver.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>

#include "common_vars.h"
#include "grabbers.h"
#include "QM_calc.h"
#include "integral_derivatives.h"
#include "gradient_rhf.h"
#include "gradient_casscf.h"
#include "casscf_resp.h"
#include "dsrg_grad.h"

double grad_rss_mb(){
    FILE * f = fopen("/proc/self/statm", "r"); if(!f) return 0;
    long pages = 0, res = 0; if(fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0; fclose(f);
    return res * (sysconf(_SC_PAGESIZE) / 1048576.0);
}
double grad_peak_rss_mb(){
    FILE * f = fopen("/proc/self/status", "r"); if(!f) return 0;
    char line[256]; double kb = 0;
    while(fgets(line, sizeof line, f)) if(!strncmp(line, "VmHWM:", 6)){ kb = atof(line + 6); break; }
    fclose(f); return kb / 1024.0;
}

const char * analytic_gradient_method(const inp_par * P){
    const bool others = P->dsrg.y || P->mp2.y || P->cis.y || P->avas.y;
    if(P->dsrg.y && P->cas.y && P->cas.ci_solver == CISOLVER_ALDET && !P->xmc.y && !P->cdas.y && !P->mp2.y && !P->cis.y && !P->avas.y) return "dsrg_pt2";
    if(P->rhf.y && !P->cas.y && !P->xmc.y && !P->cdas.y && !others) return "rhf";
    const bool aldet = P->cas.y && P->cas.ci_solver == CISOLVER_ALDET;
    if(aldet && !P->xmc.y && !P->cdas.y && !others) return "cas_scf";
    fprintf(out_stream, "ERROR: no analytic gradient for this combination of methods (supported: RHF; SA-CASSCF and DSRG-PT2 with the "
                        "determinant CI); not substituting a numerical one\n");
    return nullptr;
}

int analytic_gradient(inp_par * P, molecule * M, int root, double * g, double * E, const char ** method_out){
    const char * method = analytic_gradient_method(P);
    if(!method) return 3;
    if(method_out) *method_out = method;
    const bool rhf = !strcmp(method, "rhf");
    const bool cas = !strcmp(method, "cas_scf");
    const bool dsrg = !strcmp(method, "dsrg_pt2");
    // DSRG-PT2 computes its gradient inside the energy run, on request; reset the requests (a previous geometry may have left results behind)
    g_dsrg_grad.want = dsrg; g_dsrg_grad.status = 0; g_dsrg_grad.done = false;
    if(dsrg && root >= 0 && !P->dsrg.sa && root != P->dsrg.root){
        fprintf(out_stream, "ERROR: the DSRG-PT2 target root is set in the input ($DSRG root=%d); -root %d differs\n", P->dsrg.root, root); return 3; }
    grab_E = 1;
    const double t0 = omp_get_wtime();
    single_point_calc(P, M);
    const double t1 = omp_get_wtime();
    if(deriv_check_supported(M)) return 3;
    nopt_deriv_engines_initialize(M);
    const int nat = M->n_atoms;
    std::vector<double> gr(3 * nat, 0.0);
    double e = 0.0;
    int rc = 0;
    if(rhf){
        if(rhf_gradient(M, gr.data())) rc = 3;
        else e = E_grabbed.empty() ? 0.0 : E_grabbed[0];
    }
    else if(cas){
        if(root >= 0){ if(casscf_root_gradient(M, root, gr.data(), &e)) rc = 3; }
        else if(casscf_sa_gradient(M, gr.data(), &e)) rc = 3;
    }
    else if(dsrg){
        if(!g_dsrg_grad.done){ fprintf(out_stream, "ERROR: the DSRG-PT2 gradient was not computed (see the messages above)\n"); rc = 3; }
        else if(dsrg_grad_contract(M)) rc = 3;
        else { gr = g_dsrg_grad.g; e = g_dsrg_grad.E; }
    }
    nopt_deriv_engines_finalize();
    if(rc) return rc;
    fprintf(out_stream, "\nAnalytic gradient timing: energy calculation with the gradient hooks %.2f s, derivative-integral contraction %.2f s; peak memory %.0f MB\n",
            t1 - t0, omp_get_wtime() - t1, grad_peak_rss_mb());
    // every supported path is C1 without external fields: the gradient is invariant under translations and rotations
    double tr[3] = {0, 0, 0}, tq[3] = {0, 0, 0}, gmax = 0;
    for(int a = 0; a < nat; a++) for(int q = 0; q < 3; q++){
        tr[q] += gr[3 * a + q]; gmax = std::max(gmax, std::fabs(gr[3 * a + q]));
        const double * x = M->atom_coord + 3 * a;
        tq[q] += x[(q + 1) % 3] * gr[3 * a + (q + 2) % 3] - x[(q + 2) % 3] * gr[3 * a + (q + 1) % 3];
    }
    double bad = 0; for(int q = 0; q < 3; q++) bad = std::max(bad, std::max(std::fabs(tr[q]), std::fabs(tq[q])));
    fprintf(out_stream, "\nAnalytic gradient invariance: max |sum of forces|, |torque| = %.2e (gradient scale %.2e)\n", bad, gmax);
    if(bad > 1e-5){ fprintf(out_stream, "ERROR: the analytic gradient violates translational/rotational invariance (%.2e): it is wrong; not reported\n", bad); return 4; }
    if(bad > 1e-7) fprintf(out_stream, "WARNING: the analytic gradient violates translational/rotational invariance by %.2e\n", bad);
    for(int i = 0; i < 3 * nat; i++) g[i] = gr[i];
    if(E) *E = e;
    return 0;
}
