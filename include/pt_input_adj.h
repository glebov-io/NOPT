#ifndef NOPT_PT_INPUT_ADJ_H
#define NOPT_PT_INPUT_ADJ_H

// Adjoints of the inputs of the second-order PT blocks (E2_calc_EE): the MO-basis RI tensors of RI_data, the orbital energies
// and H_AV/H_CA/H_CV. The block adjoints accumulate into it (+=); pt_chain_B / pt_chain_H take it back to C, L, V and h.
struct PTInputAdj{
    double * VC = nullptr, * VA = nullptr, * CA = nullptr, * CC = nullptr, * AA = nullptr;   // like RI_data::*_RI_M
    double * e_c = nullptr, * e_a = nullptr, * e_v = nullptr;
    double * e_IP = nullptr, * e_EA = nullptr;   // optional: if null, their adjoints are added to e_a (e_IP, e_EA are copies of e_a)
    double * H_AV = nullptr, * H_CA = nullptr, * H_CV = nullptr;
    double shift_adj = 0.0;   // adjoint of the uniform denominator shift of the block adjoints (XMC energy grid)
};

#endif
