#include "gradient_rhf.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "blas_link.h"
#include "common_vars.h"
#include "integral_derivatives.h"
#include "libint_link.h"
#include "matr.h"
#include "ri_gradient.h"

int rhf_gradient(molecule * M, double * g){
    if(deriv_check_supported(M)) return 1;
    const int n = M->n_ao, nocc = M->n_el_calc / 2, na = M->n_atoms;
    if(M->n_el_calc != 2 * nocc){
        fprintf(out_stream, "ERROR: the RHF gradient needs an even number of electrons (n_el = %d)\n", M->n_el_calc);
        return 1;
    }
    const size_t n2 = (size_t)n * n;

    // spin-summed density P = 2 C_occ C_occ^T and energy-weighted density W = 2 C_occ eps C_occ^T
    // (MO_VEC rows are the orbitals, in the Cartesian AO basis)
    std::vector<double> P(n2, 0.0), mW(n2, 0.0);
    for(int k = 0; k < nocc; k++){
        const double * c = M->MO_VEC + (size_t)k * n;
        const double e = M->orb_energy[k];
        for(int a = 0; a < n; a++) for(int b = 0; b < n; b++){
            P[(size_t)a * n + b]  += 2.0 * c[a] * c[b];
            mW[(size_t)a * n + b] -= 2.0 * e * c[a] * c[b];       // -W: the overlap enters with a minus sign
        }
    }

    {   // the analytic gradient assumes a stationary SCF: max |F_ia| in the MO basis (the RHF loop also stops on the energy criterion alone)
        std::vector<double> F(n2, 0.0), FC(n2, 0.0);
        M->calc_F_AO(F.data(), P.data(), 0.5);
        const int nmo = M->n_mo;
        double gmax = 0.0;
        for(int a = nocc; a < nmo; a++){
            for(int x = 0; x < n; x++){ double t = 0; for(int y = 0; y < n; y++) t += F[(size_t)x * n + y] * M->MO_VEC[(size_t)a * n + y]; FC[x] = t; }
            for(int i = 0; i < nocc; i++){ double t = 0; for(int x = 0; x < n; x++) t += M->MO_VEC[(size_t)i * n + x] * FC[x]; gmax = std::max(gmax, std::fabs(t)); }
        }
        fprintf(out_stream, "RHF gradient: stationarity of the reference, max |F_ia| = %.2e\n", gmax);
        if(gmax > 1e-7) fprintf(out_stream, "WARNING: the RHF reference is not converged to stationarity (max |F_ia| = %.2e); the analytic gradient carries an error of "
                                            "about that order. The SCF also stops on the energy criterion: tighten it (e.g. dE=-1 disables it) together with grad\n", gmax);
    }

    std::vector<double> grad(3 * na, 0.0);
    deriv_nuclear_repulsion(M, grad.data());
    deriv_kinetic_contract(M, P.data(), grad.data());
    deriv_nuclear_contract(M, P.data(), grad.data());
    deriv_overlap_contract(M, mW.data(), grad.data());

    if(RI == 0){
        // E2 = 1/2 sum P P [(ab|cd) - 1/2 (ac|bd)]
        deriv_eri_JK_contract(M, P.data(), 0.5, -0.25, grad.data());
    }else{
        long naux = 0;
        for(auto & s : M->ri_s) naux += s.size();
        std::vector<double> L(n2 * naux), Vm((size_t)naux * naux, 0.0);
        calc_2el_AO_INTS_RI(M->s, n, M->ri_s, (int)naux, L.data());
        RI_V_calc(M->ri_s, (int)naux, Vm.data());
        double lmin, lmax;
        ri_eigenvalue_range((int)naux, Vm.data(), &lmin, &lmax);
        fprintf(out_stream, "RI Coulomb metric: %ld auxiliary functions, eigenvalues %.3e .. %.3e (condition number %.2e)\n",
                naux, lmin, lmax, lmax / lmin);
        if(!(lmin > 1e-12)){
            fprintf(out_stream, "ERROR: the RI Coulomb metric is (nearly) rank deficient (lowest eigenvalue %.3e); "
                                "the analytic RI gradient assumes full rank\n", lmin);
            return 1;
        }
        std::vector<double> Lbar(n2 * naux), Vbar((size_t)naux * naux);
        ri_jk_vjp(n, (int)naux, L.data(), Vm.data(), P.data(), 0.5, -0.25, Lbar.data(), Vbar.data());
        deriv_ri3_contract(M, [&](const int *, const int * o, const int * nn, double * w){
            for(int p = 0; p < nn[0]; p++) for(int a = 0; a < nn[1]; a++) for(int b = 0; b < nn[2]; b++)
                w[(p * nn[1] + a) * nn[2] + b] = Lbar[((size_t)(o[1] + a) * n + o[2] + b) * naux + o[0] + p];
        }, grad.data());
        deriv_ri2_contract(M, [&](const int *, const int * o, const int * nn, double * w){
            for(int p = 0; p < nn[0]; p++) for(int q = 0; q < nn[1]; q++)
                w[p * nn[1] + q] = Vbar[(size_t)(o[0] + p) * naux + o[1] + q];
        }, grad.data());
    }
    for(int i = 0; i < 3 * na; i++) g[i] = grad[i];
    return 0;
}
