#ifndef CDAS_GNO_H
#define CDAS_GNO_H

#include <vector>

class CAS_engine;
class PT_tensors;
class cdas_par;

// Active-space totals of the GNO stage, in the PT-frame (native) active basis: the dressed
// operator (bare plus the PT tables, three-body group g3) and its at-most-two-body folded
// image. Layouts are the dressed import's; filled once and never mutated afterwards.
struct gno_operators {
    int n;
    double c_d, c_f, F0;
    std::vector<double> g1_d, g2_d, g3, F1, F2, g1_f, g2_f;
};

// The CAS-SCF ensemble in the PT frame: the state-averaged 1- and 2-RDMs, the three-body moment
// (empty when the GNO scalar is skipped) and the normalized state weights.
struct gno_ensemble_data {
    std::vector<double> gamma, GAMMA, G3, w_ens;
};

// The ensemble inherited from the retained CAS-stage engine, pulled before the PT stage builds
// its own: the RDMs rotated into the PT frame, and with want_g3 the weighted three-body moment.
// loc_realized/U_loc are the localization the PT stage applied and its n x n rotation
// (non-owning, nullptr when none).
void gno_ensemble(CAS_engine * CAS, cdas_par * cdas, int loc_realized, const double * U_loc,
                  bool want_g3, gno_ensemble_data & ens);

// Dressed totals from H_AA, act_INTS, E_core and the PT tables (RF_PH, RF_PV_AB, RF_P3_AB,
// RF_PS), then the fold of g3 against the ensemble (gamma, GAMMA): g1_f = g1_d + F1,
// g2_f = g2_d + F2, c_f = c_d + F0. A null G3_ens skips the scalar, F0 = 0.
gno_operators prepare_gno_operators(int n, const PT_tensors & T, const double * H_AA,
                                    const double * act_INTS, double E_core,
                                    const double * gamma, const double * GAMMA,
                                    const double * G3_ens);

// Folded-reference stage of the GNO modes, entered from CDAS_PT2 when $CDAS cdas_mode is
// trunc_gno or delta_gno: it completes the approximated calculation in place of the native PT2
// solve and returns for the caller's cleanup. H_AA (embedded active 1-e) and act_INTS (active
// (tu|vw)) are the PT-frame integrals of the dressed totals; ens.G3 is released once they exist.
int cdas_gno_run(CAS_engine * CAS, const PT_tensors & T, cdas_par * cdas,
                 gno_ensemble_data & ens, const double * H_AA, const double * act_INTS,
                 double E_core);

#endif
