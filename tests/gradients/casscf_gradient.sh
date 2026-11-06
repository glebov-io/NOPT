#!/usr/bin/env bash
#
# The analytic gradient of the state-averaged CAS-SCF energy (run_grad, casscf_sa_gradient) against
# 4th-order central differences of sum_K w_K E_K / sum_K w_K (every displaced energy a fresh run_sQM process), on water
# CAS(4,4) with 3 equally weighted states, CAS(4,4) state-specific, CAS(6,6) with 2 states, and ammonia CAS(4,4) with 2
# states; translation/torque invariance (<= 1e-7, the CASSCF orbital convergence) and 1/8-thread independence included.
# Every displaced CAS-SCF starts from the orbitals of the central geometry (--follow), so all runs follow one solution:
# the state-specific case has several solutions and an RHF-started displaced run can land on another one (a 5 mEh jump seen
# at one displacement). Tolerance 1e-6 Eh/bohr.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
rc=0
for f in h2o_cas44 h2o_cas44_ss h2o_cas66 nh3_cas44; do
    echo ">> $f"
    "$HERE/gradcheck.py" "$HERE/fixtures/cas/$f.inp" --method cas_scf --weighted --follow --inv-tol 1e-7 --h 2e-3 1e-3 --threads 1 8 --tol 1e-7 \
        | grep -E 'translation|threads|max\||PASS|FAIL' || rc=1
done
exit $rc
