# Analytic nuclear gradients

`run_grad` computes the energy of the input (exactly as `run_sQM` does) and its analytic gradient dE/dR.
`run_opt` optimizes the geometry with it (L-BFGS). No method falls back to a numerical gradient: unsupported input is refused.

```
run_grad -i input.inp [-root N] [-json g.jsonl] [-numgrad h]
run_opt  -i input.inp [-root N] [-maxit 100] [-gtol 3e-5] [-nofollow]
```

| option | meaning |
|---|---|
| `-root N` | state to differentiate: CAS-SCF root (without it: the state-averaged energy), XMCQDPT2 root (default 0). DSRG-PT2: the root is set by `$DSRG root=` |
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
| XMCQDPT2 | `XMC=1` on such a CAS-SCF (also `ifitd=1`) | energy of root N of the final effective Hamiltonian |
| DSRG-PT2 | `DSRG=1` on such a CAS-SCF, `$dsrg root=` or `sa=1`, `relax=none / once` | unrelaxed or relaxed energy defined by `$dsrg` |

Any non-negative SA weights; roots with zero weight can be differentiated too. Symmetry is not used (C1).
Refused: `$XMC D_ONLY` (the second-order Hamiltonian is read from a file, so the energy is not a function of the geometry),
`cisolver=dmrg`, ECP, spin-orbit, external fields, and every other method (CDAS-PT2, CIS, MP2, unrestricted variants).

## Examples

Every input here was checked against 4-point central differences of its energy (`tests/gradients/gradcheck.py`); errors are in Eh/bohr.

| directory / file | shows | E (Eh) | check vs FD |
|---|---|---|---|
| `01_rhf/c2h4_rhf.inp` | exact-integral RHF | -78.038188 | 2.6e-10 |
| `02_casscf/h2o_sa2.inp` | SA-2 CAS(4,4): SA energy; `-root 1`: excited root | -75.894452 / -75.748093 | 9.5e-10 / 2.5e-9 |
| `02_casscf/h2o_zero_weight.inp` | 4 roots, weights `1 1`: `-root 2` has zero weight | -75.425512 | 8.4e-8 |
| `04_xmc/h2o_xmc.inp`, `h2o_xmc_weights.inp` | XMCQDPT2 SA-2 with equal and with unequal weights (0.7 0.3); E of `-root 1` (root 0 of `h2o_xmc.inp`: -76.234520) | -75.923274 / -75.924932 | <= 1e-7 |
| `04_xmc/h2o_xmc_s2_ifitd.inp` (+`h2o_ifitd.dat`) | active orbital energies imported with `ifitd=1` | -76.242078 | 1.2e-9 |
| `04_xmc/c4h6_xmc_s7.inp` (+`c4h6_CAS.orb`) | butadiene cc-pVTZ, 7 states (4 with zero weight), 300 grid points | | 3.5e-8 |
| `04_xmc/naph_xmc.inp` | naphthalene cc-pVDZ (18 atoms, distorted from D2h), CAS(4,4) SA-3 | -384.712596 | 5.6e-8 |
| `05_dsrg/h2o_dsrg_ss.inp`, `_ss_relax.inp` | DSRG-PT2 state-specific, unrelaxed / `relax=once` | -76.221927 / -76.224045 | 1.0e-9 / 1.3e-9 |
| `05_dsrg/h2o_dsrg_s2_r1.inp`, `_s2_sa.inp`, `_cas66_r1.inp` | root of an SA-2 reference; SA ensemble; CAS(6,6) root 1 | -75.918186 / -76.075078 / -75.918739 | 3.6e-9 / 2.4e-9 / 1.3e-9 |
| `06_crossings/h3_near_d3h.inp` | XMCQDPT2 of H3 next to the D3h conical intersection (doublet, gap 0.005 Eh) | -1.540174 | 1.4e-9 |
| `06_crossings/c2h4_twist_pyr{51,55}.inp` | twisted ethylene on both sides of the S0/S1 seam (pyramidalization ~53 deg) | -78.152275 / -78.142773 | 1e-8 |
| `07_optimization/h2o_xmc_opt.inp` | `run_opt -i h2o_xmc_opt.inp -root 0`; trajectory in `h2o_xmc_opt_opt.xyz` | -76.236640 | converged in 7 steps, max\|g\| 4e-6 |
| `07_optimization/h2o_cas_opt.inp` | `run_opt -i h2o_cas_opt.inp -root 0`: root 0 of SA-2 CAS(4,4); trajectory in `h2o_cas_opt_opt.xyz` | -76.042457 | converged in 8 steps, max\|g\| 2.6e-6 |

## Cost: analytic vs numerical (XMCQDPT2)

`run_grad -root 0 -numgrad 0.01` times both in one run. The numerical gradient is the scheme of the old optimizer: 6 N energies,
central differences, step 0.01 bohr. "One energy" is the numerical time divided by the number of energies (in-process, without program start-up). The analytic gradient
costs 2-4 energies whatever the number of atoms, the numerical one 6 N.

| input | threads | one energy | energy + analytic gradient | numerical (energies) | ratio | max \|num - an\| |
|---|---|---|---|---|---|---|
| `h2o_xmc.inp` | 4 | 0.011 s | 0.03 s | 0.19 s (18) | 6 | 1.8e-5 |
| `c4h6_xmc_s7.inp` | 4 | 2.2 s | 8.7 s | 135 s (60) | 15 | 3.0e-5 |
| `naph_xmc.inp` | 8 | 5.2 s | 10.1 s | 565 s (108) | 56 | 2.8e-5 |

The differences are the O(h^2) error of the numerical gradient, not of the analytic one: 4-point differences agree with it to
4e-9 (`h2o_xmc.inp`), 3.5e-8 (`c4h6_xmc_s7.inp`) and 5.6e-8 (`naph_xmc.inp`), as in the table of examples above.
Memory: about two extra arrays of n_ao^2 x n_aux on top of the energy (C4H6 1.4 GB vs 0.5 GB, naphthalene 1.6 GB vs 0.7 GB; the timing lines
print the peak).

**Memory of the three-body resolvent tables.** XMCQDPT2 keeps two tables of n_act^6 x N_fit elements, and builds them with three more of
the same size (7.2 GB each for 12 active orbitals and 300 grid points). Energy and gradient build and use them in windows of the grid;
`NOPT_XMC_WINDOW_MB` (default 4096) bounds the five window-sized arrays, and the seeds of the gradient (`NOPT_XMC_ADJ_MB`, default the same).
The results do not depend on the windows (to 1e-12 Eh, 1e-13 Eh/bohr). Windows of about 25 grid points or more cost no time; narrower ones
repeat the three-body contraction and slow it down.

| run | whole tables | windows (MB) | peak memory | time |
|---|---|---|---|---|
| guanine CAS(10,10) SA-4, energy | 12.7 GB | 4096 / 1000 | 5.3 / 2.2 GB | same / same |
| indole SA-4 (12o,8e), energy | 36.1 GB | 4096 / 2000 | 5.8 / 3.7 GB | +2% / +4% |
| guanine, gradient of root 0 | 12.7 GB | 1000 | 2.6 GB | +10% |

`NOPT_XMC_FULL_TABLES=1` keeps the whole tables (as the self-tests `NOPT_XMC_VJP_TEST`, `NOPT_XMC_ADJ_TEST` do).

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
5. **Intruders.** XMCQDPT2 always shifts its denominators, ΔE/(ΔE² + s) with `$XMC EDSHIFT` (default 0.02), and the gradient
   differentiates that form: no singularity, but near an intruder (|ΔE| ~ √s) the surface bends sharply.
6. **XMC weights**: the Fock operator of QDPT2 must average with the actual SA-CASSCF weights (state tracking that changes them is refused).
7. **Discrete choices are not differentiable.** The dressed-to-bare root match of DSRG `relax=once` (overlap 0.9), dropped degenerate
   orbital pairs. The output warns when a quantity is close to such a threshold.
8. **DSRG `sa=1` implies `relax=once`** unless `relax=none` is given.
9. **`RI=0`**: only RHF; the exact-integral energy itself needs much memory (3.3 GB for ethylene cc-pVDZ).
10. **Diatomics** need Cartesian input and the fix of `Z_matrix::init` for molecules with fewer than 3 atoms (a separate change);
   without it the program stops before the energy.

Tests: `tests/gradients/regress_all.sh` runs every check (energy regression, derivative integrals, component checks, the RHF and
CAS-SCF gradients against finite differences, `run_opt`).
