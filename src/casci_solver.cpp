#include "casci_solver.h"

#include <cstdio>
#include <cstdlib>

#include "common_vars.h"   // out_stream

// Out-of-line anchor: gives casci_solver a key function so its vtable/RTTI is emitted in
// exactly one translation unit instead of every includer.
casci_solver::~casci_solver() {}

// Default: the extended-Koopmans IP/EA blocks have no backend-independent construction, and a
// backend reaching here would leave all four output buffers untouched.
int casci_solver::calc_IPEA_single(double *, double *, double *, double *, int, std::vector<double>) {
    fprintf(out_stream, "ERROR: this CI backend does not implement calc_IPEA_single"
                        " (extended-Koopmans IP/EA blocks)\n");
    exit(EXIT_FAILURE);
}

// Default: root following compares CI vectors across an active-basis change; a backend without
// index-comparable vectors would leave S_track holding the caller's stale buffer.
void casci_solver::calc_S(double*, int, int) {
    fprintf(out_stream, "ERROR: this CI backend does not implement calc_S"
                        " (state tracking needs CI-vector overlaps)\n");
    exit(EXIT_FAILURE);
}

// Default: both shipped backends snapshot their state set for the dressed re-solve overlap, so
// reaching here is a driver mis-dispatch (our own contract) -- abort loudly.
void casci_solver::snapshot_states(int) {
    fprintf(out_stream, "ERROR: this CI backend cannot snapshot its wavefunction set for the"
                        " dressed re-solve overlap (aldet and DMRG/block2 backends only)\n");
    exit(EXIT_FAILURE);
}

// Default: only the DMRG/block2 backend encodes a dressed general MPO. Any other backend reaching
// here is a driver mis-dispatch (our own contract), so abort loudly naming the supported backend.
void casci_solver::import_dressed_operator(const double*, const double*, const double*, double) {
    fprintf(out_stream, "ERROR: this CI backend does not support dressed-operator import"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

// Defaults: only the DMRG/block2 backend holds named operator handles, so any other backend
// reaching one of these is a driver mis-dispatch (our own contract) -- abort loudly.
void casci_solver::import_named_operator(int, const double*, const double*, const double*, double) {
    fprintf(out_stream, "ERROR: this CI backend does not import named operators"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::select_operator(int) {
    fprintf(out_stream, "ERROR: this CI backend cannot select a named operator"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

// Defaults: branch solves, named state sets and named checkpoints are the DMRG/block2 backend's
// alone; any other backend reaching one of these is a driver mis-dispatch (our own contract).
int casci_solver::solve_branch(int, int, int, double, bool) {
    fprintf(out_stream, "ERROR: this CI backend cannot continue its state set on a named operator"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::save_checkpoint(const char*) {
    fprintf(out_stream, "ERROR: this CI backend cannot checkpoint its wavefunction set"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::load_checkpoint(const char*) {
    fprintf(out_stream, "ERROR: this CI backend cannot restore a checkpointed wavefunction set"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::save_state_set(const char*) {
    fprintf(out_stream, "ERROR: this CI backend cannot store a named state set"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::overlap_sets(const char*, const char*, double*) {
    fprintf(out_stream, "ERROR: this CI backend does not overlap two named state sets"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::expect_set(const char*, int, double*) {
    fprintf(out_stream, "ERROR: this CI backend does not provide named-operator expectation values"
                        " on a named state set (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::release_named_states() {
    fprintf(out_stream, "ERROR: this CI backend holds no named state sets to release"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

// Defaults: the branch-solve records exist only where a branch solve does -- returning a placeholder
// would be reported as a measurement, so abort loudly instead.
double casci_solver::last_entry_dw() const {
    fprintf(out_stream, "ERROR: this CI backend records no branch entry discarded weight"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

double casci_solver::last_tail_dw() const {
    fprintf(out_stream, "ERROR: this CI backend records no one-site tail discarded weight"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

int casci_solver::last_tail_sweeps() const {
    fprintf(out_stream, "ERROR: this CI backend records no one-site tail sweep count"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

bool casci_solver::last_solve_converged() const {
    fprintf(out_stream, "ERROR: this CI backend records no per-solve convergence flag"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

int casci_solver::last_max_bond_dim() const {
    fprintf(out_stream, "ERROR: this CI backend has no bond dimension to measure"
                        " (DMRG/block2 backend only)\n");
    exit(EXIT_FAILURE);
}

// Defaults: both shipped backends implement the transition-density read-outs; reaching one
// of these is a driver mis-dispatch (our own contract), so abort loudly, never return silence.
#if 0  // full transition 2-RDM: no consumer, the driver reads G2_calc_diag. Revive for first-order
       // properties -- the 2-body Mbar needs bra != ket densities between the dressed roots.
void casci_solver::G_calc_full(double*) {
    fprintf(out_stream, "ERROR: this CI backend does not provide the full transition 2-RDM read-out\n");
    exit(EXIT_FAILURE);
}
#endif

void casci_solver::G3_calc_diag(double*, int) {
    fprintf(out_stream, "ERROR: this CI backend does not provide the per-state 3-body moment"
                        " read-out (aldet and DMRG/block2 backends only)\n");
    exit(EXIT_FAILURE);
}

void casci_solver::G2_calc_diag(double*) {
    fprintf(out_stream, "ERROR: this CI backend does not provide the per-state 2-RDM"
                        " read-out (aldet and DMRG/block2 backends only)\n");
    exit(EXIT_FAILURE);
}

#if 0  // DIRECT lambda3 path (superseded by the explicit lattice-3RDM route; revive for nact >~ 30)
void casci_solver::h2caa_overlap(const double*, const double*, int, double*) {
    fprintf(out_stream, "ERROR: this CI backend does not provide the complementary-overlap read-out\n");
    exit(EXIT_FAILURE);
}
#endif
