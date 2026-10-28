# Analytic nuclear gradients

`run_grad` computes the energy of the input (exactly as `run_sQM` does) and its analytic gradient dE/dR.
`run_opt` optimizes the geometry with it (L-BFGS). No method falls back to a numerical gradient: unsupported input is refused.

```
run_grad -i input.inp [-root N] [-json g.jsonl] [-numgrad h]
run_opt  -i input.inp [-root N] [-maxit 100] [-gtol 3e-5] [-nofollow]
```

| option | meaning |
|---|---|
| `-root N` | CAS-SCF root to differentiate. Without it: the state-averaged energy. DSRG-PT2: the root is set by `$DSRG root=` |
| `-json FILE` / `NOPT_GRAD_JSON=FILE` | append the gradient as a JSON line (`NOPT_ENERGY_JSON=FILE` records the energies) |
| `-numgrad h` | also compute the numerical gradient the old optimizer uses (central differences, step `h` bohr) and print both timings |
| `NOPT_GRAD_REPEAT=1` | repeat the calculation in the same process and compare (exit code 6 on a mismatch) |
| `run_opt -nofollow` | restart every CAS-SCF from RHF instead of following the previous solution |

Output: `Analytic <method> gradient ..., E = ...` followed by one line per atom (Eh/bohr). Exit codes: 3 unsupported input or failure,
4 the gradient violates translational/rotational invariance by more than 1e-5 (not printed), 5 `run_opt` did not converge.

The programs need a libint with first derivatives (`tests/gradients/libint-deriv.sh`); with the energy-only libint of the
published images they stop with a message.

## Supported

| method | input | what is differentiated |
|---|---|---|
| RHF | `RHF=1` | energy (RI or exact integrals) |
| SA-CASSCF | `CAS=1`, `RI=1`, determinant CI (`cisolver=aldet`, the default) | SA energy, or root `-root N` (orbital and CI response through the Z-vector) |
| DSRG-PT2 | `DSRG=1` on such a CAS-SCF, `$dsrg root=` or `sa=1`, `relax=none / once` | unrelaxed or relaxed energy defined by `$dsrg` |

Any non-negative SA weights; roots with zero weight can be differentiated too. Symmetry is not used (C1).
Refused: `cisolver=dmrg`, ECP, spin-orbit, external fields, and every other method (XMCQDPT2 comes separately).

## Examples

Every input here was checked against 4-point central differences of its energy (`tests/gradients/gradcheck.py`); errors are in Eh/bohr.

| directory / file | shows | E (Eh) | check vs FD |
|---|---|---|---|
| `01_rhf/c2h4_rhf.inp` | exact-integral RHF | -78.038188 | 2.6e-10 |
| `02_casscf/h2o_sa2.inp` | SA-2 CAS(4,4): SA energy; `-root 1`: excited root | -75.894452 / -75.748093 | 9.5e-10 / 2.5e-9 |
| `02_casscf/h2o_zero_weight.inp` | 4 roots, weights `1 1`: `-root 2` has zero weight | -75.425512 | 8.4e-8 |
| `05_dsrg/h2o_dsrg_ss.inp`, `_ss_relax.inp` | DSRG-PT2 state-specific, unrelaxed / `relax=once` | -76.221927 / -76.224045 | 1.0e-9 / 1.3e-9 |
| `05_dsrg/h2o_dsrg_s2_r1.inp`, `_s2_sa.inp`, `_cas66_r1.inp` | root of an SA-2 reference; SA ensemble; CAS(6,6) root 1 | -75.918186 / -76.075078 / -75.918739 | 3.6e-9 / 2.4e-9 / 1.3e-9 |
| `07_optimization/h2o_cas_opt.inp` | `run_opt -i h2o_cas_opt.inp -root 0`: root 0 of SA-2 CAS(4,4); trajectory in `h2o_cas_opt_opt.xyz` | -76.042457 | converged in 8 steps, max\|g\| 2.6e-6 |

## Pitfalls

1. **Converge the CAS-SCF tightly.** The gradient assumes a stationary CAS-SCF; its error is linear in the remaining orbital gradient.
   The examples use `grad=1e-10 step=1e-11 dE=1e-15` in `$CAS`. With the defaults (`1e-7`) expect errors of order 1e-6 Eh/bohr: fine
   for optimizations, not for frequencies by finite differences of gradients. A reference with an orbital, CI or `(H - E) c` residual
   above 1e-6 is refused.
2. **CI convergence.** Converge the CI tightly (`$dav de=1e-14 dr=1e-16 $end`; `dr` bounds the squared residual norm) when the CI
   space is large: with the defaults the residual can stay near 1e-5, and the gradient is then off by ~1e-3.
3. **Crossings.** At a degeneracy the root energy has no gradient; next to it the gradient is exact but the surface is strongly curved.
   Roots are numbered by energy: an optimization through a crossing changes state. An equal-weight pair of degenerate roots whose
   rotation the energy depends on is refused.
4. **Spin of the SA roots.** The determinant CI does not force the spin: in larger active spaces triplets can enter a singlet
   ensemble. Check `S^2` in the output.
5. **Discrete choices are not differentiable.** The dressed-to-bare root match of DSRG `relax=once` (overlap 0.9), dropped degenerate
   orbital pairs. The output warns when a quantity is close to such a threshold.
6. **DSRG `sa=1` implies `relax=once`** unless `relax=none` is given.
7. **`RI=0`**: only RHF; the exact-integral energy itself needs much memory (3.3 GB for ethylene cc-pVDZ).
8. **Diatomics** need Cartesian input and the fix of `Z_matrix::init` for molecules with fewer than 3 atoms (a separate change);
   without it the program stops before the energy.

Tests: `tests/gradients/regress_all.sh` runs every check (energy regression, derivative integrals, component checks, the RHF and
CAS-SCF gradients against finite differences, `run_opt`).
