#!/usr/bin/env python3
"""Analytic nuclear gradient against finite differences of the same energy.

  gradcheck.py <input.inp> --method rhf [--variant ""] [--index 0] [--h 2e-3 1e-3 5e-4] [--tol 2e-7]

The analytic gradient comes from run_grad. Every displaced energy is a fresh run_sQM process in its own
directory (no state shared between displacements), read from the 17-digit
NOPT_ENERGY_JSON record `method/variant` at `index`, and differenced with the 4-point formula

    D4(h) = (-E(+2h) + 8 E(+h) - 8 E(-h) + E(-2h)) / (12 h)         h in bohr, per Cartesian coordinate.

Rows are printed per coordinate and per h; the run passes if, for some h of the ladder, the largest
|analytic - D4| is below --tol (Eh/bohr). The geometry in the input is rewritten in its own units.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT_TO_BOHR = {'ang': 1.889725989, 'au': 1.0, 'nm': 18.89725989}


def read_geometry(text):
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.strip().startswith('_xyz'):
            n = int(lines[i + 1].split()[0])
            atoms = []
            for k in range(n):
                f = lines[i + 3 + k].split()
                atoms.append((f[0], [float(x) for x in f[1:4]]))
            return i + 3, atoms
    sys.exit('no _xyz block in the input')


def units(text):
    m = re.search(r'_UNITS\s*=\s*(\w+)', text)
    return m.group(1).lower() if m else 'ang'


def write_geometry(text, first, atoms):
    lines = text.splitlines()
    for k, (el, xyz) in enumerate(atoms):
        lines[first + k] = '%s %.15f %.15f %.15f' % (el, *xyz)
    return '\n'.join(lines) + '\n'


def run(prog, workdir, inp, env_extra, threads):
    p = subprocess.run([os.path.join(HERE, 'run.sh'), workdir, inp], env=dict(os.environ, PROG=prog, **env_extra),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    return p.returncode, p.stdout


def energy_of(workdir, method, variant, index, weighted=False):
    rec = [json.loads(l) for l in open(os.path.join(workdir, 'e.jsonl')) if l.strip()]
    rec = [r for r in rec if r['method'] == method and r['variant'] == variant]
    r = rec[-1]
    if weighted:   # state-averaged energy sum_K w_K E_K / sum_K w_K with the weights the run used
        w = r['w']
        return sum(wi * ei for wi, ei in zip(w, r['e'])) / sum(w)
    return r['e'][index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--method', required=True)
    ap.add_argument('--variant', default='')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--coords', default='', help='check only these Cartesian components, 3*atom+xyz, comma separated (default: all)')
    ap.add_argument('--h', nargs='*', type=float, default=[2e-3, 1e-3, 5e-4])
    ap.add_argument('--inv-tol', type=float, default=1e-8, help='allowed |sum of forces| and |torque| of the analytic gradient')
    ap.add_argument('--follow', action='store_true', help='start every displaced CAS-SCF from the orbitals of the central geometry (follow one solution)')
    ap.add_argument('--root', type=int, default=-1, help='differentiate the energy of this root (CAS-SCF response); implies --index')
    ap.add_argument('--weighted', action='store_true', help='differentiate the state-averaged energy (weights from the record)')
    ap.add_argument('--tol', type=float, default=2e-7)
    ap.add_argument('--threads', nargs='*', type=int, default=[], help='also rerun the analytic gradient with these thread counts')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.root >= 0: a.index = a.root

    src_dir = os.path.dirname(os.path.abspath(a.input))
    name = os.path.basename(a.input)
    text = open(a.input).read()
    first, atoms = read_geometry(text)
    to_bohr = UNIT_TO_BOHR[units(text)]
    tmp = tempfile.mkdtemp(prefix='nopt-gradcheck-')

    orb_file = [None]

    def prepare(tag, geom, follow=False):
        d = os.path.join(tmp, tag)
        os.makedirs(d)
        for f in os.listdir(src_dir):
            p = os.path.join(src_dir, f)
            if os.path.isfile(p) and not f.endswith(('.log', '.out', '.jsonl')):
                shutil.copy(p, d)
        t = write_geometry(text, first, geom)
        if follow and orb_file[0]:
            # start from the central CAS-SCF orbitals: RHF off, the orbital file read inside $MOL (as in examples/c4h6_pt)
            shutil.copy(orb_file[0], d)
            t = re.sub(r'RHF=1', 'RHF=0', t)
            if re.search(r'^\s*_MO\s+INP=', t, flags=re.M):   # the input names a starting orbital file: point it at the central orbitals instead
                t = re.sub(r'^(\s*_MO\s+INP=)\S+', r'\g<1>%s' % os.path.basename(orb_file[0]), t, count=1, flags=re.M)
            else:
                t = re.sub(r'(\$act_space[^\n]*\n)', r'\1_MO INP=%s\n' % os.path.basename(orb_file[0]), t, count=1)
        open(os.path.join(d, name), 'w').write(t)
        return d

    # analytic
    d0 = prepare('central', atoms)
    rc, out = run('run_grad', d0, name, dict({'NOPT_GRAD_JSON': 'g.jsonl', 'NOPT_ENERGY_JSON': 'e.jsonl'}, **({'NOPT_GRAD_ROOT': str(a.root)} if a.root >= 0 else {})), 4)
    if rc != 0 or not os.path.isfile(os.path.join(d0, 'g.jsonl')):
        print(out[-2000:]); sys.exit('run_grad failed (rc=%d)' % rc)
    rec = [json.loads(l) for l in open(os.path.join(d0, 'g.jsonl'))][-1]
    g = rec['g']
    if a.follow:
        cands = [f for f in os.listdir(d0) if f.endswith('_CAS.orb')]
        if not cands: sys.exit('--follow: no *_CAS.orb written by the central run')
        orb_file[0] = os.path.join(d0, cands[0])
    nat = len(atoms)
    print('analytic gradient (%s/%s), central energy %.12f' % (rec['method'], rec['variant'], energy_of(d0, a.method, a.variant, a.index, a.weighted)))

    # invariances of the analytic gradient: translation (sum of forces) and rotation (torque)
    nb = [[x * to_bohr for x in xyz] for _, xyz in atoms]
    tr = [sum(g[3 * i + q] for i in range(nat)) for q in range(3)]
    tq = [sum(nb[i][(q + 1) % 3] * g[3 * i + (q + 2) % 3] - nb[i][(q + 2) % 3] * g[3 * i + (q + 1) % 3] for i in range(nat)) for q in range(3)]
    inv_ok = max(abs(x) for x in tr + tq) < a.inv_tol
    print('translation sum %s  torque %s  %s' % (['%.1e' % x for x in tr], ['%.1e' % x for x in tq], 'PASS' if inv_ok else 'FAIL'))
    # thread count independence
    thr_ok = True
    for t in a.threads:
        dt = prepare('threads%d' % t, atoms)
        tt = open(os.path.join(dt, name)).read()
        open(os.path.join(dt, name), 'w').write(re.sub(r'NUM_THREADS=\d+', 'NUM_THREADS=%d' % t, tt))
        rc, out = run('run_grad', dt, name, dict({'NOPT_GRAD_JSON': 'g.jsonl', 'NOPT_ENERGY_JSON': 'e.jsonl'}, **({'NOPT_GRAD_ROOT': str(a.root)} if a.root >= 0 else {})), t)
        gt = [json.loads(l) for l in open(os.path.join(dt, 'g.jsonl'))][-1]['g']
        d = max(abs(x - y) for x, y in zip(g, gt))
        print('gradient with %d threads differs by %.1e from the default' % (t, d))
        thr_ok = thr_ok and d < max(1e-8, a.inv_tol)

    # displaced energies
    coords = [int(x) for x in a.coords.split(',')] if a.coords else list(range(3 * nat))
    jobs = []
    for c in coords:
        for h in a.h:
            for k in (-2, -1, 1, 2):
                jobs.append((c, h, k))

    def energy_job(job):
        c, h, k = job
        geom = [(el, list(xyz)) for el, xyz in atoms]
        geom[c // 3][1][c % 3] += k * h / to_bohr
        d = prepare('c%d_h%g_k%d' % (c, h, k), geom, a.follow)
        rc, out = run('run_sQM', d, name, {'NOPT_ENERGY_JSON': 'e.jsonl'}, 4)
        if rc != 0:
            raise RuntimeError('displaced run failed: ' + out[-500:])
        e = energy_of(d, a.method, a.variant, a.index, a.weighted)
        shutil.rmtree(d, ignore_errors=True)
        return job, e

    with ThreadPoolExecutor(a.workers) as ex:
        E = dict(ex.map(energy_job, jobs))

    table = []
    worst = {}
    for c in coords:
        row = {'atom': c // 3, 'xyz': 'xyz'[c % 3], 'analytic': g[c], 'fd': {}}
        for h in a.h:
            d4 = (-E[(c, h, 2)] + 8 * E[(c, h, 1)] - 8 * E[(c, h, -1)] + E[(c, h, -2)]) / (12 * h)
            row['fd'][h] = d4
            worst[h] = max(worst.get(h, 0.0), abs(d4 - g[c]))
        table.append(row)
    print('\n atom xyz        analytic   ' + ''.join('   D4(h=%g)   err' % h for h in a.h))
    for r in table:
        print('  %2d   %s  % .10f ' % (r['atom'], r['xyz'], r['analytic']) +
              ''.join('  % .10f %.1e' % (r['fd'][h], abs(r['fd'][h] - r['analytic'])) for h in a.h))
    rms = {h: (sum((r['fd'][h] - r['analytic']) ** 2 for r in table) / len(table)) ** 0.5 for h in a.h}
    print('\n max|analytic - D4|: ' + ', '.join('h=%g: %.2e (rms %.2e)' % (h, worst[h], rms[h]) for h in a.h))
    # evidence for the FD noise floor: the exact gradient sums to zero over the atoms (translation invariance); the D4 values do so only up to their own noise
    for h in a.h:
        fs = [sum(r['fd'][h] for r in table if r['xyz'] == ax) for ax in 'xyz']
        print(' D4 translation sum (h=%g): ' % h + ' '.join('%+.1e' % v for v in fs) + '   analytic: ' + ' '.join('%+.1e' % sum(r['analytic'] for r in table if r['xyz'] == ax) for ax in 'xyz'))
    best = min(worst.values())
    ok = best < a.tol and inv_ok and thr_ok
    print('%s (best max error %.2e, tolerance %.1e Eh/bohr)' % ('PASS' if ok else 'FAIL', best, a.tol))
    if a.out:
        json.dump({'input': a.input, 'method': a.method, 'variant': a.variant, 'analytic': g,
                   'table': table, 'max_err': {str(h): worst[h] for h in a.h}, 'tol': a.tol, 'pass': ok}, open(a.out, 'w'), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
