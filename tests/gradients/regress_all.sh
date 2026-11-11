#!/usr/bin/env bash
#
# Everything that must keep passing after a change to shared code, one after another: the energy regression, the derivative integrals,
# the component checks and the analytic gradients of every supported method against finite differences of their energies, and run_opt.
# Needs build.sh (the default TARGETS) with the derivative libint. Prints one PASS/FAIL line per step.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# step logs are kept (NOPT_REGRESS_LOGS, default a fresh directory under ${TMPDIR:-/tmp}); the path is printed at the end
W=${NOPT_REGRESS_LOGS:-$(mktemp -d "${TMPDIR:-/tmp}/nopt-regress.XXXXXX")}; mkdir -p "$W"
rc=0
step() { local name=$1; shift; printf '%-34s ' "$name"; if "$@" > "$W/$name.log" 2>&1; then echo PASS; else echo "FAIL (see below)"; tail -15 "$W/$name.log"; rc=1; fi; }
# the component programs ignore their input; any readable file will do
unit() { local d; d=$(mktemp -d); cp "$HERE/fixtures/cas/$2" "$d/"; PROG=$1 "$HERE/run.sh" "$d" "$2"; local r=$?; rm -rf "$d"; return $r; }
step regress               "$HERE/regress.py"
step derivtest             "$HERE/derivtest.sh"
step ritest                unit run_ritest h2o_cas44.inp
step chaintest             unit run_chaintest h2o_cas44.inp
step tapetest              unit run_tapetest h2o_cas44.inp
step resptest              unit run_resptest h2o_cas44.inp
step rhf_gradient          "$HERE/rhf_gradient.sh"
step casscf_gradient       "$HERE/casscf_gradient.sh"
step casscf_root_gradient  "$HERE/casscf_root_gradient.sh"
step dsrg_gradient         "$HERE/dsrg_gradient.sh"
step dsrg_relax_gradient   "$HERE/dsrg_relax_gradient.sh"
step opt_check             "$HERE/opt_check.sh"
echo "step logs: $W"
exit $rc
