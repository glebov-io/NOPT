#ifndef NOPT_XMC_VJP_TEST_H
#define NOPT_XMC_VJP_TEST_H

// Dot-product test of the PT block adjoints against the XMC resolvent tables (res_fit_data::calc_RF_2_*): the table of a block at energy-grid point f is
// the EE table with every denominator shifted by E_fit[f]; pt_vjp_shift carries the shift. NOPT_XMC_VJP_TEST=<blocks> [NOPT_XMC_VJP_GRID=f] in QDPT2.
// Known issue: on a three-state input with n_fit=40 it reports a mismatch while the end-to-end finite-difference checks of the gradient pass;
// it is therefore not part of tests/gradients (the contraction adjoints are covered by NOPT_XMC_ADJ_TEST and the gradient checks).
class PT_tensors;
class res_fit_data;
int xmc_vjp_selftest(PT_tensors & T, res_fit_data & RF, const char * blocks, int f);
#endif
