# AVAS active-space steering

AVAS (Atomic Valence Active Space) picks *which* orbitals land in the CAS active window by
their overlap with a set of atomic reference shells, instead of by orbital energy or by a
hand-written `reorder=1 orbitals=` list. It is opt-in: without an `$AVAS` group nothing in a
run changes.

Minimal input:

```
$PAR RHF=1 CAS=1 D5=1 RI=1 NAME=cr2 $PAREND
...
$act_space n_alp=6 n_bet=6 n_val=12 mult=1 $end
$AVAS atoms=1 2; shells=4s 3d; $END
```

## What it does

After the reference orbitals are available (RHF, or `MO_orth` when `RHF=0`), AVAS builds the
projector of the requested atomic shells onto the occupied and the virtual orbital block
separately, diagonalizes each one, and rotates the two blocks so that

- the σ-largest occupied orbitals become the **last** occupied orbitals, and
- the σ-largest virtual orbitals become the **first** virtual orbitals,

which is exactly the window `[n_core, n_core+n_val)` that `$act_space` defines. The eigenvalues
σ ∈ [0,1] measure how much of each rotated orbital lies in the reference span; they are printed
for both blocks with the selection boundary marked, and are stored in the orbital-energy field
so the dumped orbital files carry them.

The counts are **not** chosen by AVAS: `$act_space` stays authoritative. AVAS fills
`k_occ = (n_alp+n_bet)/2` occupied and `n_val - k_occ` virtual slots. If the forced counts cut
across a σ tier rather than at the largest gap of the spectrum, a `NOTE:` line says so and the
run proceeds as asked.

Under `$SYMM` (Cs, Ci, C2, C2v, C2h, D2, D2h) both projectors are diagonalized per irrep, so every
rotated orbital keeps its irrep label; the σ ranking is merged across irreps and a table prints how
many selected orbitals each irrep contributes. Every input orbital must carry an irrep label, and
`atoms=`/`shells=` must be closed under the group: each symmetry image of a listed atom is listed
with the same labels.

The rotated orbitals are always written as `<NAME>_AVAS.orb`, `<NAME>_AVAS.orb_GAMESS` and
`<NAME>_AVAS.out`, so a steered run can be inspected and restarted from its window.

## Keywords

- **atoms=** *(required, no default)* — 1-based indices of the atoms carrying the target
  shells, `;`-terminated: `atoms=1 2;`.
- **shells=** *(required, no default)* — nl labels, `;`-terminated: `shells=4s 3d;`. A bare
  label applies to every atom of `atoms=`; the atom-qualified form `k:nl` applies to atom `k`
  alone, which must be listed in `atoms=`, so a heteroatomic target reads
  `atoms=1 2; shells=1:3d 1:4s 2:5p;`. The two forms mix in one list, and every listed atom
  needs at least one label that applies to it. Within an atom the k-th reference shell of
  angular momentum l is the principal number n = k+l+1, so for a 3d metal `4s` and `3d` are
  the valence labels. A label that the reference basis does not carry for an element is an error.
- **ref_basis=** *(cc-pvtz-minao)* — the minimal basis the reference shells are taken from
  (H–Kr in the shipped library; it is also the SAD-guess basis).
- **ncore=** *(0)* — the first `ncore` occupied orbitals are left out of the occupied projection
  and stay, unrotated, at the bottom of the core: `ncore=18` keeps the 1s–3p cores of both Cr
  atoms out of the selection. Accepted range 0 to the CAS core size.
- **canonicalize=** *(0)* — `1` rotates each block (the excluded `ncore` orbitals, the rest of the
  core, the active window, the virtuals) into eigenvectors of the closed-shell Fock matrix of the
  input determinant, which the AVAS rotation leaves unchanged; the orbital-energy field then holds
  these Fock eigenvalues instead of σ. Spans and the CAS energy do not change.

The virtual tier is built from the reference functions alone, so its rank is at most their
number: it holds the antibonding partners of the target shell, but it cannot supply a radially
distinct next shell (4d-like for a 3d reference) that the reference does not contain. A window
that needs the next shell lists it (`shells=4s 3d 4d;`); a virtual tier that runs out of
reference rank shows up as trailing near-zero σ in the printed spectrum.

## Restrictions

AVAS is rejected loudly, not silently ignored, when

- there is no `CAS=1` — nothing downstream consumes the steered window;
- `MP2=1` or `CIS=1` is set in the same run — both need canonical orbitals;
- `$act_space reorder=1` is set — two contradictory steering mechanisms;
- the point group has a two-dimensional irrep (C3, C3v, D3, D3h, C4v, C6v, D6h, LINEAR) — the
  per-irrep rotation needs one-dimensional irreps;
- under `$SYMM` an input orbital has no irrep label, `atoms=`/`shells=` are not closed under the
  group, or a rotated orbital fails the symmetry re-check;
- an ECP is in use — the reference shells are not all in the calculation basis.

Localization and DMRG orbital ordering (`localize=pm`, `loc_order=`) are unaffected: they take
the active window as given, so they compose with AVAS normally.
