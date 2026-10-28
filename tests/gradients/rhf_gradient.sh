#!/usr/bin/env bash
#
# The analytic RHF gradient (run_grad) against 4th-order
# central differences of the same RHF energy, each displacement a fresh process, on distorted water
# (cc-pVDZ and cc-pVTZ, exact and RI) and ammonia (RI); with translation/torque invariance and
# 1/8-thread independence. Tolerance 2e-7 Eh/bohr. Needs build.sh (run_sQM, run_grad).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
rc=0
for f in h2o_rhf_exact h2o_rhf_ri h2o_rhf_exact_tz h2o_rhf_ri_tz nh3_rhf_ri c4h6_rhf_ri_dz c2h4_rhf_exact_dz; do
    echo ">> $f"
    "$HERE/gradcheck.py" "$HERE/fixtures/rhf/$f.inp" --method rhf --h 2e-3 1e-3 5e-4 --threads 1 8 --tol 2e-7 \
        | grep -E 'translation|threads|max\||PASS|FAIL' || rc=1
done
exit $rc
