#ifndef NOPT_DSRG_GRAD_H
#define NOPT_DSRG_GRAD_H

// Analytic gradient of the spin-free DSRG-PT2 energy, unrelaxed and relax=once.
// The energy of dsrg_sf_tensors + dsrg_sf_batch (Eref + [F,T1] + [F,T2] + [V,T1] + in-core [V,T2] L1/L2 + lambda3 + DF CCVV/CAVV/CCAV) is
// transcribed onto a reverse-mode tensor tape (tensor_tape.h) as a function of the quantities SA_DSRG_PT2 hands to the algebra core, all in
// the semicanonical basis: the RI B-tensors VC, VA, CA, AA (RI_data layouts), eps (core, active, virtual), the Fock cross blocks, the
// spin-summed 1-RDM, the seam 2-RDM, the 3-body moment, the bare h (core diagonal, active block) and the scalar. dsrg_tape_energy returns
// every ledger term (gate against the production ledger) and, optionally, the adjoints of all inputs.
// relax=once: the dressed active operator (build_hbar, the batched CAVV/CCAV Hbar1 folds, degno) is transcribed onto the same tape; the relaxed
// energy sum_d w_d E_d of the dressed roots is differentiated by Hellmann-Feynman on them (seeds e0, gamma_d, Gamma_d/2 on e0_d, h1_d, h2_d).
#include <vector>

struct DsrgTapeIn{
    int nc = 0, na = 0, nv = 0; long naux = 0;
    double s = 0.5; bool ccvv_zero = false;
    const double * VC = nullptr, * VA = nullptr, * CA = nullptr, * AA = nullptr;
    const double * ec = nullptr, * ea = nullptr, * ev = nullptr;
    const double * fca = nullptr, * fcv = nullptr, * fav = nullptr;    // [i*na+t], [i*nv+a], [t*nv+a]
    const double * L1 = nullptr, * GAMMA = nullptr, * G3 = nullptr;    // na^2, na^4 (seam layout), na^6 (moment <p+q+r+ k j i>, BEFORE the cumulant)
    const double * hcd = nullptr, * hact = nullptr;                     // bare h: core diagonal (nc), active block (na^2)
    double escalar = 0.0;
};

struct DsrgTapeLedger{
    double E_FT1 = 0, E_FT2 = 0, E_VT1 = 0, aavv_L1 = 0, aavv_L2 = 0, ccaa_L1 = 0, ccaa_L2 = 0, caav_L1 = 0, caav_L2 = 0, caaa_L1 = 0, caaa_L2 = 0,
           aaav_L1 = 0, aaav_L2 = 0, E3v = 0, E3c = 0, Eref = 0, ccvv = 0, cavv = 0, ccav = 0, total = 0;
};

struct DsrgTapeAdj{
    std::vector<double> VC, VA, CA, AA, ec, ea, ev, fca, fcv, fav, L1, GAMMA, G3, hcd, hact;
};

// relax=once: the dressed active operator (build_hbar + the driver's batched CAVV/CCAV Hbar1 folds, then degno) on the same tape.
struct DsrgRelaxOut{ double e0 = 0; std::vector<double> h1, h2_chem, Hbar1, Hbar2; };   // e0 with the scalar; Hbar1/Hbar2 before degno
struct DsrgRelaxSeeds{ double e0b = 0; const double * h1b = nullptr, * h2b_chem = nullptr; };   // adjoints of e0_d, h1_d (na^2), h2_d chemist (na^4)

// The batched DF terms (dsrg_batch_vjp.cpp), never materialized: CCVV returns its energy and, for seed != 0, accumulates seed x its adjoints;
// CAVV / CCAV return the n_a x n_a intermediate C (Cb == nullptr) or accumulate the adjoints for the seed Cb on C.
double dsrg_batch_CCVV(const DsrgTapeIn & in, double seed, std::vector<double> * VCb, std::vector<double> * ecb, std::vector<double> * evb);
void dsrg_batch_CAVV(const DsrgTapeIn & in, std::vector<double> * C, const double * Cb, std::vector<double> * VCb, std::vector<double> * VAb,
                     std::vector<double> * ecb, std::vector<double> * eab, std::vector<double> * evb);
void dsrg_batch_AAVV(const DsrgTapeIn & in, std::vector<double> * MT, std::vector<double> * MS, const double * MTb, const double * MSb,
                     std::vector<double> * VAb, std::vector<double> * eab, std::vector<double> * evb);
void dsrg_batch_CCAV(const DsrgTapeIn & in, std::vector<double> * C, const double * Cb, std::vector<double> * VCb, std::vector<double> * CAb,
                     std::vector<double> * ecb, std::vector<double> * eab, std::vector<double> * evb);

// Forward (and, with adj != nullptr, backward: seed 1 on the total unrelaxed energy, or with rs the seeds on the dressed operator).
int dsrg_tape_energy(const DsrgTapeIn & in, DsrgTapeLedger * out, DsrgTapeAdj * adj, const DsrgRelaxSeeds * rs = nullptr, DsrgRelaxOut * ro = nullptr);

// Dot-product test of the tape adjoints against central differences of the tape energy, one input group at a time. Returns the failures.
int dsrg_tape_selftest(const DsrgTapeIn & in);

// ---- the nuclear gradient (run_grad) ----
#include "casscf_resp.h"
class dsrg_par;
struct DsrgGradRequest{
    bool want = false;               // set by analytic_gradient before the energy calculation
    int status = 0;                  // nonzero: refused or failed (message printed)
    bool done = false;
    CasRef * ref = nullptr;          // SA-CASSCF snapshot taken before SA_DSRG_PT2 semicanonicalizes the orbitals
    CasObjects total;                // objects for the derivative integrals (direct + response)
    std::vector<double> g;           // dE/dR after dsrg_grad_contract
    double E = 0.0;
};
extern DsrgGradRequest g_dsrg_grad;

// At the start of SA_DSRG_PT2 (orbitals still those of the CAS-SCF). Refuses non-determinant CI.
int dsrg_grad_snapshot(molecule * M, dsrg_par * dsrg);
// After the unrelaxed energy. in: the tape inputs (semicanonical); Csc: semicanonical orbitals (n_mo x n_ao rows); eps: their orbital energies (n_mo);
// Uc, Ua, Uv: the block rotations of SA_DSRG_PT2 (column i = semicanonical orbital i in the CAS-SCF orbitals, [k*m+i]); roots/wts: the reference roots
// whose densities were averaged (one root with weight 1 for the state-specific variant).
int dsrg_gradient(molecule * M, const DsrgTapeIn & in, const double * Csc, const double * eps, const double * Uc, const double * Ua, const double * Uv,
                  const std::vector<int> & roots, const std::vector<double> & wts, const DsrgRelaxSeeds * rs = nullptr, double E_relax = 0.0);
// relax=once: after the dressed solve. dressed: the CI object holding the dressed roots (semicanonical active basis); wd: the weights SA_DSRG_PT2
// gave the dressed roots (one-hot on the matched root for the state-specific variant); the relaxed energy sum_d wd E_d is differentiated by
// Hellmann-Feynman on the dressed roots, its operator (e0_d, h1_d, h2_d) back through the tape. Checks that the tape operator reproduces E_d.
class aldet_data;
int dsrg_relax_gradient(molecule * M, const DsrgTapeIn & in, const double * Csc, const double * eps, const double * Uc, const double * Ua, const double * Uv,
                        const std::vector<int> & roots, const std::vector<double> & wts, aldet_data * dressed, const std::vector<double> & wd,
                        const std::vector<double> & E_dressed);
int dsrg_grad_contract(molecule * M);

#endif
