#!/usr/bin/env bash
#
# The analytic gradient of the unrelaxed DSRG-PT2 energy against 4th-order central differences of that energy (every displaced energy a fresh
# run_sQM, the CAS-SCF started from the central orbitals; the fixtures converge it tightly, step=1e-11).
# Cases: water CAS(4,4) state-specific (one CAS-SCF state); a 2-state SA-CASSCF reference with the state-specific DSRG on root 1; the same with the
# SA ensemble (sa=1); 4 roots with weights 1 1 0 0 and DSRG on the zero-weight root 2; CAS(6,6) with 3 states, root 1 (larger 3-body terms).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
rc=0
run() {  # <fixture> <variant> <label>
    local fx=$1 variant=$2 label=$3 inp=$W/$3.inp
    cp "$fx" "$inp"
    echo ">> $label"
    "$HERE/gradcheck.py" "$inp" --method dsrg_pt2 --variant "$variant" --index 0 --follow --inv-tol 1e-6 --h 4e-3 2e-3 --tol 2e-7 \
        | grep -E 'translation|max\||PASS|FAIL' || rc=1
}
run "$HERE/fixtures/dsrg/h2o_dsrg_ss.inp" root h2o_cas44_ss
run "$HERE/fixtures/dsrg/h2o_dsrg_s2_r1.inp" root h2o_cas44_sa2_root1
run "$HERE/fixtures/dsrg/h2o_dsrg_s2_sa.inp" sa h2o_cas44_sa2_ensemble
run "$HERE/fixtures/dsrg/h2o_dsrg_s4w2_r2.inp" root h2o_cas44_s4w2_root2_zero_weight
run "$HERE/fixtures/dsrg/h2o_dsrg_cas66_r1.inp" root h2o_cas66_sa3_root1
exit $rc
