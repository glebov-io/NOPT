#!/usr/bin/env bash
#
# Validation A: derivative integrals against finite differences (run_derivtest, see
# src/progs/derivtest.cpp), on distorted water in cc-pVTZ without RI, with RI, cc-pVDZ with the DZ and TZ auxiliary sets (the TZ metric has condition number 6e5), and cc-pV5Z with RI
# (shells up to l=5, auxiliary up to l=6). Needs build.sh with the derivative libint.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp "$HERE"/fixtures/deriv/*.inp "$W/"
rc=0
for f in h2o_exact h2o_ri h2o_dz_dzri h2o_dz_tzri h2o_v5z_ri c4h6_tz_ri c2h4_dz_exact; do
    echo ">> $f"
    PROG=run_derivtest "$HERE/run.sh" "$W" "$f.inp" 2>&1 | grep -E '^==|PASS|FAIL|ERROR' || true
    PROG=run_derivtest "$HERE/run.sh" "$W" "$f.inp" >/dev/null 2>&1 || rc=1
done
exit $rc
