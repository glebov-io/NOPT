#!/usr/bin/env bash
#
# Geometry optimization with the analytic gradient (run_opt). Each case must converge (|dE| < 1e-7 and max|g| < 3e-5, or a step below 1e-4
# with max|g| < 3e-5) with the energy going down overall. RHF/RI, the SA energy of a 3-state CAS(4,4) and root 0 of a 2-state CAS(4,4)
# (the CAS-based runs follow the CAS-SCF solution from geometry to geometry). Not root 1 of water: its S1 is dissociative (one OH goes to
# 3.5-4 A on a nearly flat H + OH surface), which tests the surface rather than the optimizer.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
rc=0
run(){  # <label> <fixture> [run_opt options]
    local d=$W/$1; mkdir -p "$d"
    cp "$2" "$d/case.inp"
    PROG=run_opt "$HERE/run.sh" "$d" case.inp "${@:3}" > "$d/out.log" 2>&1
    local r=$?
    local first=$(grep -m1 '^OPT' "$d/out.log" | awk '{print $5}') last=$(grep '^OPT' "$d/out.log" | tail -1 | awk '{print $5}') n=$(grep -c '^OPT' "$d/out.log")
    if [ $r -eq 0 ] && python3 -c "import sys; sys.exit(0 if float('$last') < float('$first') + 1e-9 else 1)"; then
        echo "$1: PASS (converged in $((n - 1)) steps, E $first -> $last)"
    else
        echo "$1: FAIL (rc $r)"; grep '^OPT\|Optimization\|ERROR' "$d/out.log" | tail -5; rc=1
    fi
}
run rhf_ri       "$HERE/fixtures/rhf/h2o_rhf_ri.inp"
run cas_sa3      "$HERE/fixtures/cas/h2o_cas44.inp"
run cas_s2_root0 "$HERE/fixtures/cas/h2o_cas44_s2.inp" -root 0
exit $rc
