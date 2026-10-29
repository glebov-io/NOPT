#!/usr/bin/env bash
#
# XMCQDPT2 gradients in the hard regions (near conical intersections and avoided crossings), against 4th-order central differences.
# Near a crossing the energy surface is strongly curved: the D4 truncation error ~ h^4 / gap^3 dominates at the usual steps, so these cases use
# h = 1e-3 and 5e-4 bohr (with h = 4e-3 the H3 reference itself is off by 1e-5 and converges as h^4 to the analytic value).
#   H3 near the D3h Jahn-Teller conical intersection (doublet, CAS(3,3) SA-2, gap 0.005 Eh): both roots
#   C2H4 twisted 90 deg, one CH2 pyramidalized 51 and 55 deg, on both sides of the S0/S1 intersection seam at ~53 deg (CAS(2,2) SA-2, gaps ~0.01): both roots
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
rc=0
run() {  # <fixture> <root> <label>
    local fx=$1 root=$2 label=$3 inp=$W/$3.inp
    cp "$fx" "$inp"
    echo ">> $label (root $root)"
    "$HERE/gradcheck.py" "$inp" --method xmcqdpt2 --root "$root" --follow --inv-tol 1e-6 --h 1e-3 5e-4 --tol 2e-7 \
        | grep -E 'translation|max\||PASS|FAIL' || rc=1
}
F=$HERE/fixtures/xmc_hard
for r in 0 1; do run "$F/h3_near_d3h.inp" $r h3_near_d3h_r$r; done
for r in 0 1; do run "$F/c2h4_twist_pyr51.inp" $r c2h4_pyr51_r$r; run "$F/c2h4_twist_pyr55.inp" $r c2h4_pyr55_r$r; done
exit $rc
