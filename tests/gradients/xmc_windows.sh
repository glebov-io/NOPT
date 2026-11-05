#!/usr/bin/env bash
# The XMCQDPT2 energy with the three-body resolvent tables built and contracted in windows of the energy grid against the same energy with whole
# tables (NOPT_XMC_FULL_TABLES=1): examples/c4h6_pt (trans-butadiene cc-pVTZ, CAS(4,4), 300 grid points) with NOPT_XMC_WINDOW_MB=1, which gives
# windows of 6 points (50 windows). The energies must agree to 1e-10 Eh (they differ only in the order of the sums).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EX=$HERE/../../examples/c4h6_pt
W=$(mktemp -d "${TMPDIR:-/tmp}/nopt-xmcwin.XXXXXX"); trap 'rm -rf "$W"' EXIT
for v in full win; do
    mkdir -p $W/$v; cp $EX/* $W/$v/ 2>/dev/null
    if [ $v = full ]; then env NOPT_XMC_FULL_TABLES=1 NOPT_ENERGY_JSON=e.jsonl "$HERE/run.sh" $W/$v xmc.inp > $W/$v/o.log 2>&1
    else env NOPT_XMC_WINDOW_MB=1 NOPT_ENERGY_JSON=e.jsonl "$HERE/run.sh" $W/$v xmc.inp > $W/$v/o.log 2>&1; fi || { echo "FAIL: run $v"; exit 1; }
done
grep -h 'window(s)' $W/full/o.log $W/win/o.log | sed 's/^/  /'
python3 - $W <<'PY'
import json, sys
d = sys.argv[1]
e = {v: [json.loads(l) for l in open('%s/%s/e.jsonl' % (d, v)) if '"xmcqdpt2"' in l][-1]['e'] for v in ('full', 'win')}
dev = max(abs(a - b) for a, b in zip(e['full'], e['win']))
print('windowed vs whole tables: max |dE| = %.1e Eh  %s' % (dev, 'PASS' if dev < 1e-10 else 'FAIL'))
sys.exit(0 if dev < 1e-10 else 1)
PY
