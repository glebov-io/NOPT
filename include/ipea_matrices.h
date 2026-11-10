#ifndef IPEA_MATRICES_H
#define IPEA_MATRICES_H

// The IP and EA metric and Hamiltonian matrices of an n-orbital active space, built from an
// ensemble: gamma = spin-summed 1-RDM (n^2), GAMMA = 2-RDM in the G_calc layout (n^4),
// g1 = embedded bare one-electron matrix, g2 = real Coulomb integrals (tu|vw) carrying the full
// eightfold symmetry. The four n^2 outputs are overwritten and must not alias the inputs.
void ipea_matrices(int n, const double * g1, const double * g2,
                   const double * gamma, const double * GAMMA,
                   double * U_IP, double * H_IP, double * U_EA, double * H_EA);

#endif
