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
rebuilds incrementally. Knobs: `ARCH`, `GTO_MAX`, `DMRG`, `JOBS`, `TARGETS`. It writes a
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

## Component checks

| script / program | checks |
|---|---|
| `PROG=run_ritest run.sh <dir> <any.inp>` | density-fitting adjoints on random data: L-only, V-only and both, a negative control, the J/K seeds, the adjoint of V^-1/2 |
