#ifndef CDAS_FOLD_H
#define CDAS_FOLD_H

// The three-body dressing g3 and its at-most-two-body fold against a frozen ensemble.
// Flat row-major NOPT layouts: g3[((((t*n+u)*n+v)*n+w)*n+x)*n+y] pairs (t,u),(v,w),(x,y);
// gamma[t*n+u]; GAMMA[((t*n+u)*n+v)*n+w] = <e_{tu,vw}> (pairs with (tu|vw)); F2 in the g2
// layout [((v*n+w)*n+x)*n+y]; F1[a*n+b]. No output may alias an input.

// Twelve-term recombination of the imported T3_AB block into the pair-S3 symmetric,
// Hermitian g3 (n^6 doubles, caller-allocated).
void assemble_g3(int n, const double * T3_AB, double * g3);

// Generalized normal ordering of (1/6) sum g3 e_{tu,vw,xy} against (gamma, GAMMA) with every
// scalar dropped: F = 1/2 sum F2 e_{vw,xy} + sum F1 E_ab. F1 (n^2) and F2 (n^4) are overwritten.
// fold_scalar supplies the dropped scalar.
void build_fold(int n, const double * g3, const double * gamma, const double * GAMMA,
                double * F1, double * F2);

// Ensemble energies of the one-, two- and three-body pieces: sum h1 gamma; 1/2 sum h2 GAMMA;
// 1/6 sum g3[t,u,v,w,x,y] G3[t,v,x,u,w,y], with G3 in the G3_calc_diag layout
// G3[(((((p*n+q)*n+r)*n+i)*n+j)*n+k] = <a+_p a+_q a+_r a_k a_j a_i>.
double fold_e1(int n, const double * h1, const double * gamma);
double fold_e2(int n, const double * h2, const double * GAMMA);
double fold_e3(int n, const double * g3, const double * G3);

// The scalar build_fold drops, against the same ensemble: fold_e3(g3, G3_ens)
// - fold_e1(F1, gamma) - fold_e2(F2, GAMMA). G3_ens carries fold_e3's layout. Nothing is written.
double fold_scalar(int n, const double * g3, const double * G3_ens, const double * F1,
                   const double * F2, const double * gamma, const double * GAMMA);

#endif
