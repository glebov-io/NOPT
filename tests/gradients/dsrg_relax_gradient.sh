#!/usr/bin/env bash
#
# relax=once: the analytic gradient of the relaxed DSRG-PT2 energy (the dressed active operator re-diagonalized; state-specific: the dressed
# root matched to the target root; sa=1: the weighted average of the dressed roots) against 4th-order central differences of that energy.
# Cases: water CAS(4,4) one state; SA-2 ensemble (sa=1 relax=once); CAS(6,6) SA-3 root 1.
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
run "$HERE/fixtures/dsrg/h2o_dsrg_ss_relax.inp" relaxed h2o_cas44_ss_relax
run "$HERE/fixtures/dsrg/h2o_dsrg_s2_sa_relax.inp" sa_relaxed h2o_cas44_sa2_ensemble_relax
run "$HERE/fixtures/dsrg/h2o_dsrg_cas66_r1_relax.inp" relaxed h2o_cas66_sa3_root1_relax
exit $rc
