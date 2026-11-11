# Regression and analytic-gradient tests

Docker only; no host toolchain is needed.

```sh
tests/gradients/build.sh                 # build from this tree inside the builder image
tests/gradients/regress.py               # energy regression, all cases
tests/gradients/regress.py --threads 4 1 10 --vs-image pupeza/nopt:gto27-dmrg-x86-64-v4
tests/gradients/regress.py --update      # rewrite reference/*.json from this run (review the diff!)
```

`build.sh` takes its dependencies (libint, libgrpp, block2, OpenBLAS) from the `builder` stage of
`dockerbuild/Dockerfile` (Docker caches them) and compiles NOPT from the working tree, so make
rebuilds incrementally. Knobs: `ARCH`, `GTO_MAX`, `DMRG`, `JOBS`, `TARGETS`, `LIBINT_DERIV`. It writes a
gitignored `Makefile.vars` with the container paths if there is none. The programs run only inside
that image; `[PROG=<program>] run.sh <dir> <input> [args]` starts one (default `run_sQM`).

## Full-precision energies

`include/energy_result.h`: with `NOPT_ENERGY_JSON=<file>` set, every method appends one JSON line with
its final energies in `%.17g` (the log has 10 digits). Without the variable nothing is written and no
output changes. Records: `rhf`, `cis`, `cas_scf`, `xmc_cas`, `xmcqdpt2`, `dsrg_pt2` (fields in the header).

## What `regress.py` checks

| check | meaning |
|---|---|
| golden | records equal `reference/<case>.json` |
| log | the printed log agrees with the records to the printed digits |
| example | the log agrees with the example's own benchmark log (1e-6 Eh) |
| variants | dropping the `$PROP` block does not change the energies |
| threads | other thread counts give the same energies |
| image | the printed energies equal those of a published `pupeza/nopt` image |

Cases (`cases.json`): `c4h6_xmc`, `cis`, `dsrg4ene_aldet` (all from `examples/`) and `h2o_rhf`
(`fixtures/h2o`, RHF). The golden files come from GCC 13, x86-64-v4, OpenBLAS
0.3.26 (OpenMP); another CPU or BLAS can differ in the last digits, hence the 1e-9 Eh tolerance.

## Derivative libint

The analytic gradients need a libint with first derivatives. The libint of the published images is
energy-only; with it the gradient code refuses with a message. `libint-deriv.sh gen|compile` builds one
with first derivatives of the 1-body, 4-, 3- and 2-center integrals at `MAX_AM=7` (the pinned commit of
the images), in a host directory (`~/nopt-libint-deriv`), not in an image; `build.sh` uses it when it
exists (`LIBINT_DERIV=` forces the image libint). Generation takes about 30 min (one thread, <3 GB,
14 675 files / 1.9 GB of source); compilation needs about 2 GB of RAM per job (12 jobs in 26 GB);
the install is 236 MB. `ERI4=0` skips the 4-center derivatives (11 127 files / 1.3 GB), enough for RI.

`libint_deriv_smoke.cpp` checks the result against finite differences and prints the buffer
ordering:

    g++-13 -O1 -std=c++17 -I<install>/include -I/usr/include/eigen3 libint_deriv_smoke.cpp \
        -L<install>/lib -lint2 -Wl,-rpath,<install>/lib && ./a.out

## Component checks

| script / program | checks |
|---|---|
| `derivtest.sh` | every derivative-integral contraction against 4th-order finite differences of the integrals |
| `PROG=run_ritest run.sh <dir> <any.inp>` | density-fitting adjoints on random data: L-only, V-only and both, a negative control, the J/K seeds, the adjoint of V^-1/2 |
| `PROG=run_chaintest run.sh <dir> <any.inp>` | the chains of the PT inputs back to C, L, V, h (`pt_chain_B`, `pt_chain_H`) and the adjoint of the semicanonicalization (`pt_canon_vjp`, with and without the Fock-block seeds), against central finite differences of the forward definitions on random data, with the L/V adjoints in the auxiliary basis and in the eigenbasis of V; two negative controls |
| `PROG=run_tapetest run.sh <dir> <any.inp>` | the reverse-mode tape: einsum (products, diagonals, scalar operands, accumulation, a threaded contraction), every DSRG regularizer with the orbital-energy adjoints and the Taylor branch of R, auxgemm, lin, constant leaves |
| `PROG=run_resptest run.sh <dir> <cas.inp>` | SA-CASSCF response: CI kernel against the production CAS, zero residual, symmetry of the finite-difference Hessian, dense against iterative solve |

`run_ritest`, `run_chaintest` and `run_tapetest` need no molecule and no derivative libint (the input file is ignored); each prints PASS or FAIL and
returns 0 or 1.

## Gradient checks (`run_grad`)

`gradcheck.py <input> [--root N] [--follow]` compares the analytic gradient with 4-point central
differences of the same energy; every displaced energy is a separate process, read from the 17-digit
energy record. The scripts below run it on the fixtures:

| script | method |
|---|---|
| `rhf_gradient.sh` | RHF, exact and RI; invariances, thread independence |
| `casscf_gradient.sh`, `casscf_root_gradient.sh` | SA-CASSCF energy; individual roots, zero and unequal weights |
| `opt_check.sh` | `run_opt` converges (RHF, SA-CASSCF energy, a CAS-SCF root) |
| `regress_all.sh` | all of the above, the energy regression and the component checks, one PASS/FAIL line per step |

Converge the SCF tightly for gradient tests (`dE=1e-16 grad=1e-10` in `$RHF`, `grad=1e-10 step=1e-11
dE=1e-15` in `$CAS`).
