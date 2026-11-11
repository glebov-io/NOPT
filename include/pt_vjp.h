#ifndef NOPT_PT_VJP_H
#define NOPT_PT_VJP_H

// Reverse-mode (adjoint) rules for the eight PT blocks of E2_calc_EE.
// The forward kernels are not touched: each pt_vjp_EE_*
// mirrors one calc_EE_2_* and maps seeds on the RAW outputs of that block (RF_* before E2_calc_EE's
// symmetrization and permutation) to adjoints of its inputs. Inputs are the MO-basis RI tensors of
// RI_data (layouts as in RI.h), the orbital energies the block reads, and H_AV/H_CA/H_CV.
//
// Everything is ACCUMULATED (+=) into PTInputAdj, which the caller zeroes. e_IP and e_EA are copies of
// the active orbital energies made by set_par, so by default the adjoints of e_IP[t] / e_EA[t] go to e_a[t];
// ask for them separately (A.e_IP, A.e_EA) when they are not copies.
#include <functional>
#include <vector>
#include "PT_tensors.h"
#include "pt_input_adj.h"   // PTInputAdj: the adjoints of the block inputs

// Test hook: if set, pt_vjp_selftest runs this instead of T.calc_EE_2_<block>() as the forward pass; it must leave the raw block outputs in T.RF_* (EE layout).
extern std::function<void(PT_tensors &, const std::string &)> pt_vjp_forward_override;

struct PTRawSeeds{
    double PS = 0.0;
    const double * PH = nullptr, * PV_JK = nullptr, * PV_AB = nullptr, * P3_JK = nullptr, * P3_AB = nullptr;
};

// Uniform shift added to every denominator dE of the block adjoints below (for XMC the energy-grid value E_fit[f], with e_IP = e_EA = e_a; 0 for the plain EE kernels): the
// XMC resolvent tables at grid point f are the EE tables with every dE shifted by E_fit[f]. Set before calling pt_vjp_EE_*, reset after.
extern double pt_vjp_shift;


// Fused over an energy grid (XMC): K seed sets Sg[f] at denominator shifts shg[f], the integrals of each fat index pair built once for all of them;
// the shift adjoints go to shadj[f] (+=). The single-point functions below are these with K = 1, shift pt_vjp_shift, shadj = &A.shift_adj.
int pt_vjp_EE_CCVV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj);
int pt_vjp_EE_CAVV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj);
int pt_vjp_EE_AAVV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj);
int pt_vjp_EE_CCAV_grid(PT_tensors & T, const PTRawSeeds * Sg, const double * shg, int K, PTInputAdj & A, double * shadj);

int pt_vjp_EE_CCVV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);
int pt_vjp_EE_CAVV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);
int pt_vjp_EE_AAVV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);
int pt_vjp_EE_CCAV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);
int pt_vjp_EE_CCAA(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);
int pt_vjp_EE_CV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);
int pt_vjp_EE_AV(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);
int pt_vjp_EE_CA(PT_tensors & T, const PTRawSeeds & S, PTInputAdj & A);

// The end of E2_calc_EE: symmetrization of RF_PH / RF_PV_* / RF_P3_* as matrices (A_ij <- (A_ij + A_ji)/2 for
// i < j, a self-adjoint projector) and the permutation RF_P3_AB[t,u,v,w,x,y] <- raw[t,u,x,v,w,y]. This maps seeds on the
// FINAL arrays (what PT2_import_data receives) to seeds on the RAW block outputs that
// the pt_vjp_EE_* functions take. Output vectors are resized; PS passes through unchanged.
struct PTRawSeedStore{
    double PS = 0.0;
    std::vector<double> PH, PV_JK, PV_AB, P3_JK, P3_AB;
    PTRawSeeds view() const;
};
int pt_vjp_EE_final(int n_a, double PS, const std::vector<double> & PH, const std::vector<double> & PV_JK, const std::vector<double> & PV_AB,
                    const std::vector<double> & P3_JK, const std::vector<double> & P3_AB, PTRawSeedStore * raw);

// The whole of E2_calc_EE: seeds on the FINAL arrays -> adjoints of all inputs (final step, then the eight blocks).
int pt_vjp_EE_all(PT_tensors & T, double PS, const std::vector<double> & PH, const std::vector<double> & PV_JK, const std::vector<double> & PV_AB,
                  const std::vector<double> & P3_JK, const std::vector<double> & P3_AB, PTInputAdj & A);

// Dot-product test of the adjoints against central finite differences of <ybar, f(x)>, f = one block
// run in isolation (outputs zeroed first). blocks: "CCVV,..."; driven by xmc_vjp_test (NOPT_XMC_VJP_TEST in QDPT2).
int pt_vjp_selftest(PT_tensors & T, const char * blocks);
int pt_vjp_selftest_final(PT_tensors & T);

#endif
