#!/usr/bin/env bash
#
# The analytic gradient of an individual root energy of an equally weighted SA-CASSCF (orbital and CI response through the
# Z-vector, casscf_resp.cpp) against 4th-order central differences of that root's energy (every displaced energy a fresh run_sQM process started from
# the central orbitals, --follow). The root energy is not variational in the orbitals: the orbital convergence error of the displaced SA-CASSCF enters it to
# first order, so the fixtures converge the rotation to step=1e-11 (with the default 1e-7 the differences carry ~5e-7 of noise). Tolerance 2e-7.
# Zero-weight roots (h2o_cas44_s4w2: 4 roots, weights 1 1 0 0) are spectators of the orbital optimization: Hellmann-Feynman in their CI vector.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
rc=0
for spec in h2o_cas44_s2:0 h2o_cas44_s2:1 nh3_cas44:0 h2o_cas66:1 h2o_cas44_s4w2:1 h2o_cas44_s4w2:3 h2o_cas44_w73:0 h2o_cas44_w73:1; do
    f=${spec%%:*}; r=${spec##*:}
    echo ">> $f root $r"
    "$HERE/gradcheck.py" "$HERE/fixtures/cas/$f.inp" --method cas_scf --root "$r" --follow --inv-tol 1e-6 --h 4e-3 2e-3 --threads 1 8 --tol 2e-7 \
        | grep -E 'translation|threads|max\||PASS|FAIL' || rc=1
done
exit $rc
