#!/usr/bin/env python3
"""Energy regression harness (full-precision energy records).

For every case in cases.json: run the binary built by build.sh in a fresh copy of the example
directory, read the full-precision energies the run wrote through NOPT_ENERGY_JSON (see
include/energy_result.h), and check

  golden    the records equal reference/<case>.json                  (tol --tol, default 1e-9 Eh)
  log       the energies printed in the log agree with the records   (printed digits, 1e-9 Eh)
  example   the log agrees with the example's own benchmark log      (1e-6 Eh, other machine)
  variants  editing the input as the case says (e.g. no $PROP block) leaves the records unchanged
  threads   --threads N1 N2 ... gives the same records for every thread count
  image     --vs-image IMG (e.g. pupeza/nopt:gto27-dmrg-x86-64-v4) prints the same energies; the
            log is compared, as the image does not write the JSON records (printed digits, 1e-9 Eh)

--update rewrites the golden files from this run. Usage:
  tests/gradients/regress.py [--cases a b] [--threads 4] [--tol 1e-9] [--update] [--keep DIR]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
E_IDX = re.compile(r'^E\[\s*\d+\]\s*=\s*(-?\d+\.\d+)')
E_CIS = re.compile(r'ENERGY\s*=\s*(-?\d+\.\d+)\s*Eh')
E_KV = re.compile(r'=\s*(-?\d+\.\d{9,})\s*$')


def printed_energies(text):
    """The 'E[ i] = value' lines of a log, in order."""
    return [float(m.group(1)) for m in map(E_IDX.match, text.splitlines()) if m]


def all_energies(text):
    """Every final-energy line the log prints: E[ i], CIS 'ENERGY = .. Eh' and the DSRG block."""
    out, in_dsrg = [], False
    for line in text.splitlines():
        if 'DSRG-PT2 energy summary' in line:
            in_dsrg = True
            continue
        if in_dsrg and line.startswith('====='):
            in_dsrg = False
        m = E_IDX.match(line) or E_CIS.search(line) or (in_dsrg and E_KV.search(line))
        if m:
            out.append(float(m.group(1)))
    return out


def run_case(case, workdir, threads, variant=None):  # variant: drop_lines and/or env
    """Run one case in a fresh copy of its directory; return (records, log text, rc)."""
    src = os.path.join(ROOT, case['dir'])
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    for f in os.listdir(src):  # the example's own files, no stale outputs
        p = os.path.join(src, f)
        if os.path.isfile(p) and not f.endswith(('.log', '.out')):
            shutil.copy(p, workdir)
    inp = os.path.join(workdir, case['input'])
    text = open(inp).read()
    text = re.sub(r'NUM_THREADS=\d+', 'NUM_THREADS=%d' % threads, text)
    vspec = case['variants'][variant] if variant else {}
    if 'drop_lines' in vspec:
        pat = re.compile(vspec['drop_lines'])
        text = '\n'.join(l for l in text.splitlines() if not pat.match(l)) + '\n'
    open(inp, 'w').write(text)
    env = dict(os.environ, NOPT_ENERGY_JSON='energies.jsonl', **vspec.get('env', {}))
    p = subprocess.run([os.path.join(HERE, 'run.sh'), workdir, case['input']], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    records = []
    ej = os.path.join(workdir, 'energies.jsonl')
    if os.path.isfile(ej):
        records = [json.loads(l) for l in open(ej) if l.strip()]
    return records, p.stdout, p.returncode


def flat(records):
    return [x for r in records for x in r['e']]


def keys(records):
    return [(r['method'], r['variant'], r['root'], r['n']) for r in records]


def maxdiff(a, b):
    if keys(a) != keys(b):
        return None
    return max(abs(x - y) for x, y in zip(flat(a), flat(b))) if flat(a) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='*')
    ap.add_argument('--threads', nargs='*', type=int, default=[4])
    ap.add_argument('--tol', type=float, default=1e-9)
    ap.add_argument('--vs-image', help='published image to cross-check the printed energies against')
    ap.add_argument('--update', action='store_true')
    ap.add_argument('--keep', help='keep the run directories here')
    a = ap.parse_args()

    cases = json.load(open(os.path.join(HERE, 'cases.json')))['cases']
    if a.cases:
        cases = [c for c in cases if c['name'] in a.cases]
    tmp = a.keep or tempfile.mkdtemp(prefix='nopt-regress-')
    failures = 0

    def check(ok, what, detail=''):
        nonlocal failures
        print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', what, ('  ' + detail) if detail else ''))
        failures += 0 if ok else 1

    for c in cases:
        print('>> %s' % c['name'])
        tol = max(a.tol, c.get('tol', 0.0))
        base_t = a.threads[0]
        rec, log, rc = run_case(c, os.path.join(tmp, c['name']), base_t)
        check(rc == 0 and bool(rec), 'run (rc=%d, %d records)' % (rc, len(rec)))
        if rc != 0 or not rec:
            print(log[-1500:])
            continue
        # records vs what the log prints (the log has 10 digits)
        pr = printed_energies(log)
        stage = [r for r in rec if r['method'] in ('cas_scf', 'xmc_cas', 'xmcqdpt2')]
        if stage:
            d = max(abs(x - y) for x, y in zip(flat(stage), pr)) if len(flat(stage)) == len(pr) else None
            check(d is not None and d < 1e-9, 'records agree with the printed log',
                  'max|d|=%s' % ('n/a (count %d vs %d)' % (len(flat(stage)), len(pr)) if d is None else '%.1e' % d))
        # the example's own benchmark log
        ref_log = os.path.join(ROOT, c['dir'], c['log']) if c.get('log') else None
        if ref_log and os.path.isfile(ref_log):
            ref = all_energies(open(ref_log, errors='replace').read())
            pr = all_energies(log)
            d = max(abs(x - y) for x, y in zip(pr, ref)) if len(pr) == len(ref) and pr else None
            check(d is not None and d < 1e-6, 'log agrees with the example benchmark log',
                  'max|d|=%s' % ('n/a (count %d vs %d)' % (len(pr), len(ref)) if d is None else '%.1e' % d))
        # golden
        gpath = os.path.join(HERE, 'reference', c['name'] + '.json')
        if a.update:
            json.dump({'case': c['name'], 'records': rec}, open(gpath, 'w'), indent=1)
            print('  wrote %s' % os.path.relpath(gpath, ROOT))
        elif os.path.isfile(gpath):
            gold = json.load(open(gpath))['records']
            d = maxdiff(rec, gold)
            check(d is not None and d <= tol, 'records equal the golden file',
                  'max|d|=%s' % ('n/a (record layout differs)' if d is None else '%.1e' % d))
        else:
            check(False, 'golden file missing (run with --update)')
        # published image
        if a.vs_image:
            pr = all_energies(log)
            wd = os.path.join(tmp, c['name'] + '-image')
            run_case(c, wd, base_t)  # fresh copy + input; only its files are used
            p = subprocess.run(['docker', 'run', '--rm', '--shm-size=4g', '-v', wd + ':/data', a.vs_image,
                                '-i', c['input']], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors='replace')
            im = all_energies(p.stdout)
            d = max(abs(x - y) for x, y in zip(pr, im)) if len(pr) == len(im) and pr else None
            check(d is not None and d < 1e-9, 'printed energies equal %s' % a.vs_image,
                  'max|d|=%s' % ('n/a (count %d vs %d, rc=%d)' % (len(pr), len(im), p.returncode)
                                 if d is None else '%.1e' % d))
        # input variants
        for v in c.get('variants', {}):
            r2, _, rc2 = run_case(c, os.path.join(tmp, c['name'] + '-' + v), base_t, v)
            d = maxdiff(rec, r2) if rc2 == 0 else None
            check(d is not None and d <= tol, 'variant %s leaves the energies unchanged' % v,
                  'max|d|=%s' % ('n/a' if d is None else '%.1e' % d))
        # thread counts
        for t in a.threads[1:]:
            r2, _, rc2 = run_case(c, os.path.join(tmp, '%s-t%d' % (c['name'], t)), t)
            d = maxdiff(rec, r2) if rc2 == 0 else None
            check(d is not None and d <= tol, '%d threads equal %d threads' % (t, base_t),
                  'max|d|=%s' % ('n/a' if d is None else '%.1e' % d))

    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\n%s' % ('ALL PASS' if not failures else '%d FAILED' % failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
