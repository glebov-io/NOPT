#ifndef NOPT_CASSCF_RESP_H
#define NOPT_CASSCF_RESP_H

// Response of the state-averaged CAS-SCF reference (Z-vector of the analytic gradients).
//
// At a converged SA-CASSCF the stationarity conditions are r(x) = grad_x E_SA = 0 in the variables
//   x = ( kappa_pq  for the non-redundant orbital rotations (core-active, core-virtual, active-virtual) ,
//         delta_I   for the CI vectors of the WEIGHTED states, each in the complement of the weighted space {c_J} ,
//         rho_IJ    rotations between weighted states of different weight (redundant for equal weights) ).
// Zero-weight roots are not variables: they do not enter E_SA, and a target energy that depends on one only as an eigenvector (a root energy) needs
// no multiplier for it (Hellmann-Feynman).
// the Hessian J = d r / d x is symmetric, and the gradient of a root energy E_n (or of any energy E(x, R) that is a smooth function of the
// reference) is the Lagrangian derivative
//   dE/dR = dE/dR|_x  +  z . dr/dR|_x ,         J z = - dE/dx .
// Because r = grad E_SA,   z . dr/dR = d/d eps [ dE_SA/dR ] (x + eps z): the response term is the eps-derivative of the same
// geometric objects (densities D, auxiliary-basis adjoints Lbar and Vbar, energy-weighted W) that the SA-CASSCF gradient contracts with
// derivative integrals. Both d/d eps (the response term) and J p (the Hessian action) are EXACT directional derivatives (cas_jvp: the
// tangent-linear code of the residual and of the objects; the CI densities by the polarization identity of bilinear forms). The 4-point
// central differences of the analytic residual/objects that were used first are kept as test oracles only (cas_hessian_vec_fd).
// Convention for the orbital variable: dC_p = sum_q kappa_pq C_q with kappa antisymmetric, kappa_pq = x_pq for the listed p < q.
#include <vector>
#include <memory>
#include "molecule.h"
#include "aldet.h"
#include "ri_gradient.h"

struct CasBase;   // reference SA quantities for the tangent-linear code (casscf_resp.cpp)

struct CasObjects{
    double E = 0.0;                 // energy of the weighted density (without the response)
    std::vector<double> D;          // n x n  total density (contracted with T', V', kinetic and nuclear derivative integrals)
    std::vector<double> Lbar, Vbar; // n*n*naux, naux*naux: adjoints of the RI quantities L=(ab|P), V=(P|Q), held in the EIGENBASIS of V (Lbt, Vbt:
                                    // Lbar = Lbt U, Vbar = U^T Vbt U, ri_gradient.h); cas_geo_contract converts them in place
    std::vector<double> GC;         // n x nmo, [a*nmo+p]: unconstrained derivative of E with respect to the MO coefficients C[p][a]
    std::vector<double> Wao;        // n x n  energy-weighted density of the symmetric orthonormality constraint (contracted with -S')
};

class CasRef{
public:
    int n = 0, nc = 0, na = 0, nv = 0, nmo = 0, naux = 0;
    int ns = 0;                     // number of states with a nonzero weight: the SA functional and its response live on these
    int ns_all = 0;                 // number of CI roots computed (zero-weight roots included)
    std::vector<int> states;        // absolute root index of each weighted state (ns entries)
    size_t Nd = 0;
    double Enuc = 0.0;
    std::vector<double> C0;         // nmo x n (rows = MOs), converged orbitals
    std::vector<double> h, V;       // AO core Hamiltonian, RI metric
    std::vector<double> L_own;      // RI (ab|P) [(a*n+b)*naux+P] when the gradient computes its own copy
    const double * Lp = nullptr;    // the RI three-index array used: RI_AA of the energy calculation (same integrals) or L_own
    const double * L() const { return Lp; }
    std::vector<double> w;          // normalized weights of the weighted states
    std::vector<double> c0;         // ns x Nd reference CI vectors of the weighted states, state-major [I*Nd + d]
    std::vector<double> c_all;      // ns_all x Nd CI vectors of all roots (absolute order), state-major
    std::vector<double> root_vector(int root) const;   // CI vector (Nd) of absolute root `root` (any weight)
    int slot(int root) const;       // index of absolute root `root` among the weighted states, -1 for a zero-weight root
    std::vector<std::pair<int,int>> pairs;   // non-redundant rotation pairs (p < q)
    std::vector<std::pair<int,int>> rho;     // rotations between weighted states I < J with DIFFERENT weights (non-redundant then): c_I += r c_J, c_J -= r c_I
    aldet_data * ci_template = nullptr;
    molecule * M = nullptr;
    mutable RIEig ri_;                   // cached eigendecomposition of V and L U^T (n*n*naux)
    const RIEig & ri() const;
    mutable std::shared_ptr<CasBase> base_;

    size_t nk() const { return pairs.size(); }
    size_t nr() const { return rho.size(); }
    size_t ro() const { return pairs.size() + (size_t)ns * Nd; }              // offset of the rho block in x
    size_t nx() const { return pairs.size() + (size_t)ns * Nd + rho.size(); }

    int build(molecule * M);        // from a molecule that has just run a converged determinant-CI CAS-SCF; returns 0 on success
    // Re-express the reference in rotated active orbitals: new_i = sum_k U[k*na + i] old_k (the convention of three_block_diagonalization in XMCQDPT.cpp)
    // and every root vector carried along by aldet::malmqvist(U). E_SA and its stationarity are invariant; the cached base is dropped.
    int rotate_active(const double * U);
};

// Densities and the objects of the energy of a weighted set of CI vectors (weights w_i, vectors vecs[i*Nd+d], normalized inside) in the
// orbitals C (nmo x n rows). With want_LV = false only E, D, GC, Wao are filled.
int cas_objects(const CasRef & R, const double * C, const double * vecs, int nvec, const double * wts, bool want_LV, CasObjects * out);

// Active-space integrals of the orbitals C (RI): HAA = active block of the core Fock h + 2J[D_core] - K[D_core] (na x na), act = (tu|vw) (na^4,
// [((t*na+u)*na+v)*na+w]), the core energy with V_NN. Any output may be null.
void cas_active_integrals(const CasRef & R, const double * C, std::vector<double> * HAA, std::vector<double> * act, double * Ecore);
// Diagonal of C F_SA C^T for every MO, F_SA = h + J[D] - 1/2 K[D], D = 2 sum_core c c^T + sum_tu gamma_tu c_t c_u^T: the orbital energies of
// canonical orbitals when a method overwrites them (XMC ifitd).
void cas_sa_fock_diag(const CasRef & R, const double * C, const double * gamma, std::vector<double> * d);
// Objects (D, Lbar, Vbar, GC, Wao) of the linear functional E = sum_tu HAAb_tu HAA_tu + sum_tuvw actb_tuvw (tu|vw) of the orbitals C (no core
// energy, no V_NN). Checks out->E against the functional evaluated directly; returns 1 on a mismatch.
int cas_objects_active_linear(const CasRef & R, const double * C, const double * HAAb, const double * actb, bool want_LV, CasObjects * out);

// The SA-CASSCF residual r(x) at displaced orbitals C and CI vectors c (ns x Nd): r_kappa (nk) and r_delta (ns*Nd, projected on the complement
// of the REFERENCE SA space and scaled by 2 w_I). Optionally returns the SA objects. Also returns the SA energy.
int cas_residual(const CasRef & R, const double * C, const double * c, std::vector<double> * r, CasObjects * sa_obj, double * E_sa);

// Displaced state C(x), c(x) = c0 + P_perp delta for x = (kappa, delta) times `eps`.
void cas_displace(const CasRef & R, const double * x, double eps, std::vector<double> * C, std::vector<double> * c);

// Exact directional derivative at the reference along x = (kappa, delta) (delta projected on the complement):
//   Jx (may be null)   = J x, the Hessian action (derivative of the residual of cas_residual);
//   dobj (may be null) = derivative of the SA objects D, GC, Wao and (if want_LV) Lbar, Vbar of cas_residual(..., sa_obj).
int cas_jvp(const CasRef & R, const double * x, std::vector<double> * Jx, CasObjects * dobj, bool want_LV);
// Hessian action J p (= cas_jvp); the CI part is projected onto the complement.
void cas_hessian_vec(const CasRef & R, const std::vector<double> & p, std::vector<double> * Jp);
// Oracle: the same by 4-point central differences of the analytic residual (step NOPT_RESP_HEPS / max|p|, default 2e-4).
void cas_hessian_vec_fd(const CasRef & R, const std::vector<double> & p, std::vector<double> * Jp);

extern double g_cas_z_residual;   // final relative residual of the last cas_solve_z (accepted below 1e-6)

// Solves J z = rhs (conjugate gradient on the symmetric Hessian), z and rhs projected onto the complement for the CI part.
// Returns the number of iterations (negative if not converged).
int cas_solve_z(const CasRef & R, const std::vector<double> & rhs, std::vector<double> * z, double tol, int maxit);

// State-averaged 1-RDM of the reference vectors (na x na).
void cas_sa_gamma(const CasRef & R, std::vector<double> * gamma_sa);

// H c for arbitrary vectors (nvec x Nd, state-major) with the active-space integrals built from the orbitals C.
void cas_H_apply(const CasRef & R, const double * C, const double * vecs, int nvec, double * Hv);

// Derivative of <cbar|c(theta)> with respect to the rotation of the active orbitals t -> u carried by the CI vectors (aldet::malmqvist), for every pair t < u:
// g[t*na+u] = d/dtheta sum_s <cbar_s | malmqvist(U(theta)) c_s>,  g[u*na+t] = -g[t*na+u],  U(theta) = 1 + theta (e_tu - e_ut) + ... (U[t*na+u] = sin) or its
// transpose (transpose_U). Oracle: 4-point central difference of the linear map through malmqvist. c, cbar: nvec x Nd, state-major.
void cas_ci_rotation_g_fd(const CasRef & R, const double * c, const double * cbar, int nvec, double * g, bool transpose_U);
// Transition one-particle density T[t*na+u] = sum_s <bra_s| E_tu |ket_s> (spin-summed, E_tu = a+_t a_u), exact.
void cas_transition_rdm(const CasRef & R, const double * bra, const double * ket, int nvec, double * T);

// W (energy-weighted density of the symmetric orthonormality constraint) from an unconstrained orbital gradient GC (n x nmo):
// W_ao = C (1/2 sym(C^T GC)) C^T; optionally returns C^T GC.
void cas_w_from_gc(const CasRef & R, const double * C, const std::vector<double> & GC, std::vector<double> * Wao, std::vector<double> * CtG_out);

// Ghat v for vectors v (nvec x Nd), Ghat = sum_tu M_tu E_tu with M symmetrized (no projection).
void cas_onebody_apply(const CasRef & R, const double * M, const double * vecs, int nvec, double * out);

// dE/dc_I = 2 w_I P_perp Ghat c_I for an energy that depends on the reference CI vectors only through gamma_SA with the adjoint gbar; with unequal
// weights also the rho block, dE/drho_IJ = 2 (w_I - w_J) <c_I|Ghat|c_J>. b_delta: ns*Nd + nr.
void cas_gamma_rhs(const CasRef & R, const double * gbar, std::vector<double> * b_delta);

// The Lagrangian gradient of an energy whose direct (fixed-reference) derivatives are D, Lbar, Vbar and GC (all contributions summed by
// the caller) and whose derivative with respect to the reference CI vectors is b_delta: solves the Z-vector, adds the response, contracts.
int cas_lagrangian_gradient(molecule * M, const CasRef & R, CasObjects & direct, const std::vector<double> & b_delta, double * g);
// The same without the contraction with derivative integrals (they need the derivative engines): returns the total objects (direct + response).
// `direct` is consumed (its arrays are moved into tot).
int cas_lagrangian_objects(molecule * M, const CasRef & R, CasObjects & direct, const std::vector<double> & b_delta, CasObjects * tot);

// General Lagrangian for a target that depends on the reference CI vectors of ANY roots: cbar_all (ns_all x Nd, state-major, absolute roots; may be
// empty) is dE/dc_I at fixed orbitals for every root, b_extra (ns*Nd + nr, may be empty) further derivatives on the weighted-state variables
// (e.g. cas_gamma_rhs). Zero-weight roots are not SA variables: each is an eigenvector of the CAS Hamiltonian, and its derivative is carried by a
// multiplier of its own eigen-equation (P_perp (H - E_Z) z_Z = -P_perp cbar_Z on the complement of all roots, and the rotations among zero-weight roots);
// the multiplier terms, sum_Z <t_Z|H|c_Z>, join `direct` as a fixed-reference energy, and rotations between a weighted and a zero-weight root are SA
// variables (part of delta). A dependence on the rotation between weighted roots of EQUAL weight (redundant for E_SA) is carried by a multiplier of
// <c_I|H|c_J> = 0 (the roots are eigenvectors of the CAS Hamiltonian).
int cas_lagrangian_objects_cbar(molecule * M, const CasRef & R, CasObjects & direct, const std::vector<double> & cbar_all, const std::vector<double> & b_extra,
                                CasObjects * tot);

// Gradient (dE/dR, bohr) of the energy of root `root` of the SA-CASSCF (bare CAS energy, no PT). Equal weights only.
int casscf_root_gradient(molecule * M, int root, double * g, double * E_out);

// Contracts the objects with the derivative integrals (T', V_nuc', V_NN', (P|ab)', (P|Q)', -S'), accumulating into g (3 n_atoms).
int cas_geo_contract(molecule * M, const CasRef & R, CasObjects & obj, double * g);   // obj.Lbar/Vbar are converted in place

#endif
