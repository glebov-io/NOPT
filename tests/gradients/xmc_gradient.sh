#!/usr/bin/env bash
#
# The analytic gradient of XMCQDPT2 root energies against 4th-order central differences of the same energy (fresh run_sQM per displaced
# geometry, started from the central CAS-SCF orbitals). Water CAS(4,4), n_fit = 40 (the computed interpolated energy is differentiated,
# so a small grid is as good a test as the default one and much cheaper); the CAS-SCF rotation is converged to step=1e-11.
# Cases: 2 states equal weights (roots 0, 1); 3 states equal; 2 states 0.7/0.3; 3 states with a zero-weight model state (roots 0, 1, 2) --
# the zero-weight root carries its own multiplier, and the bare H01 part must stay out of the canonical-rotation term.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
F=$HERE/fixtures/xmc
rc=0
for spec in h2o_xmc_s2:0 h2o_xmc_s2:1 h2o_xmc_s3:0 h2o_xmc_w73:0 h2o_xmc_s3w2:0 h2o_xmc_s3w2:1 h2o_xmc_s3w2:2; do
    f=${spec%%:*}; r=${spec##*:}
    echo ">> $f root $r"
    "$HERE/gradcheck.py" "$F/$f.inp" --method xmcqdpt2 --root "$r" --follow --inv-tol 1e-6 --h 4e-3 2e-3 --tol 2e-7 \
        | grep -E 'translation|max\||PASS|FAIL' || rc=1
done
# the shipped example's settings: trans-butadiene (distorted), cc-pVTZ / cc-pVTZ-RI, CAS(4,4), 7 model states of which 4 have zero weight,
# default grid (n_fit 300): the whole chain on a real molecule (about 15 minutes)
echo ">> c4h6_xmc_s7 root 0"
"$HERE/gradcheck.py" "$HERE/fixtures/c4h6/c4h6_xmc_s7.inp" --method xmcqdpt2 --root 0 --follow --inv-tol 1e-6 --h 4e-3 2e-3 --tol 2e-7 --workers 4 \
    | grep -E 'translation|max\||PASS|FAIL' || rc=1
exit $rc
