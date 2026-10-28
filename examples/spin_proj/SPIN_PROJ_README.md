# Spin projection in the determinant CI

The determinant CI (`cisolver=aldet`) works in the space of determinants with Ms = (n_alp - n_bet)/2 and does not
force the total spin: that space holds every S from |Ms| up to min(n_alp + n_bet, 2 n_act - n_alp - n_bet)/2. The
Davidson guess vectors are spin-adapted, but in larger active spaces states of another spin still enter the
calculation. A singlet state-averaged CAS-SCF can then end up averaging over triplets.

The option is off by default; nothing in a run changes without it. To turn it on:

```
$dav spin_proj=1 $end
```

## What it does

Every Davidson expansion vector and the final roots are projected onto the spin S = (mult-1)/2 of `mult` in
`$act_space`, for any n_alp, n_bet. The projector is that of Lowdin,

```
P_S = prod_{S' != S} (S^2 - S'(S'+1)) / (S(S+1) - S'(S'+1)),
```

over the spins S' that the determinant space holds. S^2 = S+ S- + Sz^2 - Sz is applied to the CI vectors directly;
one application costs about N_det * n_act operations per vector, far less than the product with H. For n_alp = n_bet
the alpha-beta exchange parity C[b,a] = (-1)^S C[a,b] is applied first and removes every S' of the other parity at no
cost, so only S' = S +- 2, +- 4, ... are left for the product. H commutes with S^2, so the projected roots are exact
eigenvectors of H: the energies are those of the states of spin S, nothing else changes.

The spin is chosen by `mult`, the spin component by n_alp - n_bet. A triplet can be computed as Ms = 1
(n_alp = n_bet + 2) or as Ms = 0 (n_alp = n_bet, mult=3); both give the same energies, as far as the CI is
converged. With the default Davidson thresholds the guanine triplet roots of the two Ms differ by about 1e-5 Eh, with
`$dav spin_proj=1 dE=1e-14 dR=1e-16 $end` by less than 1e-8 Eh (dR bounds the squared residual norm). Without the
projection the Ms = S component is free of the lower spins only; with it, of the higher ones as well.

A `mult` that the space does not hold (S < |Ms|, S above the maximum, S - Ms not an integer, or mult=0) stops the run
with an error, and so does an `n_s` larger than the number of states of spin S in the active space (the number of CSFs).

The output lists the setting with the other Davidson settings, then the spins removed:

```
Spin projection onto S = 0.0 (Ms = 0.0): alpha-beta parity +1, removing S' = 2.0 4.0
```

and after the final roots the largest |(S^2 - S(S+1)) c| over them (about 1e-16).

## Example

`guanine_sa4.inp` is guanine with SA-4 CAS(10,10)/cc-pVDZ, RI, at the geometry of Y. V. Bruyakin, V. E. Bochenkov,
A. V. Bochenkova, J. Chem. Theory Comput. 2026, 22, 8767 (SI Table S3).

| | S^2 of the four SA roots | CAS-SCF energies (Eh) |
|---|---|---|
| without the option | 0.00 2.00 2.00 2.00 (S0 and three triplets) | -539.5212, -539.3702, -539.3288, -539.3168 |
| `$dav spin_proj=1` (`guanine_sa4.log`) | 0.00 0.00 0.00 0.00 | -539.5144, -539.3081, -539.3031, -539.2615 |

Without the option the run is identical to the code before this change, line for line in the RHF and CAS-SCF iterations.

## Tests

`tests/spin_proj/run_tests.py` (about 6 minutes; `--quick`, without guanine, about 2) checks the algebra of the
projector with `run_spinprojtest` (`make run_spinprojtest`): for every active space up to 7 orbitals, every
n_alp >= n_bet and every S, P_S is symmetric and idempotent, its trace is the number of CSFs, and the projectors of all S
sum to 1. Then it runs NOPT and checks that

- N2 at 2.0 A, CAS(6,6) CASCI, S = 0 to 3 with Ms = 0 to 3, and N2+ CAS(5,6), S = 1/2 to 5/2 with Ms = 1/2 to 5/2, give the
  lowest states of each S of the unprojected spectrum, and the same energies for every Ms;
- SA-CASSCF of CH2 (singlets, triplets, quintets), of the allyl radical (doublets, quartets) and of guanine (triplets)
  gives the same energies for every Ms;
- every projected root has |(S^2 - S(S+1)) c| < 1e-10;
- a `mult` the space does not hold, or more roots than it has states of that spin, stops with an error.
