#!/usr/bin/env python3
"""Tests of the spin projection of the determinant CI ($dav spin_proj=1).

    python3 tests/spin_proj/run_tests.py [--quick] [--work DIR] [--cmd TEMPLATE] [--unit-cmd CMD]

The inputs are written to the work directory (default tests/spin_proj/work) and run there. --cmd is the command
for one input, with {dir} and {inp} replaced (default: "<repo>/run_sQM -i {inp}", run in {dir}); --unit-cmd runs the
algebraic test (default "<repo>/run_spinprojtest" if it was built with `make run_spinprojtest`, skipped otherwise).
NOPT_THREADS sets NUM_THREADS of the inputs (default 4). --quick leaves out guanine (4 runs of about a minute).

What is checked, for every projected run:
  - the roots are spin eigenfunctions: max |(S^2 - S(S+1)) c| < 1e-10 (printed by the solver) and S^2 = S(S+1);
  - the same S with a different Ms (n_alp - n_bet) gives the same energies;
  - CASCI: the energies are the lowest states of that S in an unprojected run with mult=0 (all spins);
  - where the unprojected run with the same mult finds only states of the target spin, it gives the same energies.
Systems: N2 at 2.0 A, CAS(6,6) CASCI (S = 0..3, Ms = 0..3); N2+ CAS(5,6) CASCI (S = 1/2..5/2, Ms = 1/2..5/2);
CH2 CAS(6,6) and the allyl radical CAS(3,3) by SA-CASSCF; guanine SA-CASSCF CAS(10,10), where the unprojected
singlet ensemble takes in triplets. Wrong input (a spin the space does not hold, mult=0, more roots than states of
that spin) must stop with an error. The energies of different Ms agree only as far as the CI is converged: the inputs set
$dav dE=1e-14 dR=1e-16 (with the defaults, guanine triplet roots of Ms = 0 and 1 differ by about 1e-5 Eh).
"""
import argparse, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
THREADS = os.environ.get('NOPT_THREADS', '4')

GEOM = {
    'n2': "2\nN2, stretched\nN 0.0 0.0 0.0\nN 0.0 0.0 2.0",
    'ch2': "3\nmethylene\nC  0.000000  0.000000  0.104000\nH  0.000000  0.990000 -0.312000\nH  0.000000 -0.990000 -0.312000",
    'allyl': """8
allyl, C2v
C   0.000000   0.000000   0.430000
C   0.000000   1.230000  -0.200000
C   0.000000  -1.230000  -0.200000
H   0.000000   0.000000   1.520000
H   0.000000   2.160000   0.350000
H   0.000000   1.290000  -1.280000
H   0.000000  -2.160000   0.350000
H   0.000000  -1.290000  -1.280000""",
    'guanine': """16
guanine, geometry of Bruyakin, Bochenkov, Bochenkova, JCTC 22, 8767 (2026), SI Table S3
C      0.000000000000    -2.827132420000    -5.424758309000
N      0.000000000000    -1.758758998000    -4.575570863000
H      0.000000000000    -0.822927268000    -4.830881393000
C      0.000000000000    -2.249041855000    -3.297846026000
C      0.000000000000    -3.643437847000    -3.417138842000
N      0.000000000000    -3.933900153000    -4.740985261000
C      0.000000000000    -4.423912346000    -2.237455129000
O      0.000000000000    -5.641774093000    -2.294076817000
N      0.000000000000    -3.786038908000    -1.046032887000
H      0.000000000000    -4.300096075000    -0.223447794000
C      0.000000000000    -2.423974018000    -0.994413550000
N      0.000000000000    -1.687653945000    -2.083169454000
N      0.000000000000    -1.803572326000     0.229529273000
H      0.000000000000    -2.763278662000    -6.502824696000
H      0.000000000000    -2.333561673000     1.041945018000
H      0.000000000000    -0.835058139000     0.282350383000""",
}


def write_input(work, name, mol, charge, na, nb, nval, mult, ns, proj, mo=None, cas='casci', rhf_only=False):
    """cas: 'casci' (maxit=0 on the orbitals of mo), 'casscf' (SA over ns roots, equal weights)."""
    if rhf_only:
        par, casline = 'RHF=1 CAS=0', ''
    else:
        par = 'RHF=0 CAS=1' if mo else 'RHF=1 CAS=1'
        if cas == 'casci':
            casline = f"$CAS n_s={ns} maxit=0 $END\n"
        else:
            w = ' '.join(['1'] * ns)
            casline = f"$CAS n_s={ns} w_state={w}; maxit=300 dE=1e-13 grad=1e-9 $END\n"
    mo_line = f"_MO INP={mo}\n" if mo else ''
    with open(os.path.join(work, name + '.inp'), 'w') as f:
        f.write(f"""$PAR NUM_THREADS={THREADS} N_MOL=1 {par} XMC=0 D5=1 RI=1 NAME={name} $PAREND
$MOL
 _CHARGE={charge};
 _BASIS=cc-pvdz;
 _RI_BASIS=cc-pvdz-ri;
 _UNITS= ang;
 _xyz
{GEOM[mol]}
$act_space n_alp={na} n_bet={nb} n_val={nval} mult={mult} p_n=4; $end
{mo_line}$END
$RHF maxit=200 dE=1e-13 grad=1e-9 $END
{casline}$dav spin_proj={proj} dE=1e-14 dR=1e-16 $end
""")
    return name


class Run:
    def __init__(self, name, rc, log, ns):
        self.name, self.rc, self.log = name, rc, log
        st = re.findall(r'^State\s+\d+\s+E\s+=\s+(\S+)\s+S\^2\s+=\s+(\S+?):?$', log, re.M)
        st = st[-ns:] if ns else []          # the last block holds the final roots
        self.E = [float(e) for e, _ in st]
        self.S2 = [float(s) for _, s in st]
        m = re.search(r'Spin projection: max \|\(S\^2 - S\(S\+1\)\) c\| over the roots = (\S+)', log)
        self.res = float(m.group(1)) if m else None


def run(args, work, name, ns):
    cmd = args.cmd.format(dir=work, inp=name + '.inp', root=ROOT)
    t = time.time()
    p = subprocess.run(cmd, shell=True, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = p.stdout
    lf = os.path.join(work, name + '.log')
    if os.path.exists(lf) and os.path.getsize(lf) > len(log):
        log = open(lf).read()
    else:
        open(lf, 'w').write(log)
    r = Run(name, p.returncode, log, ns)
    if args.verbose:
        print(f"    ran {name} in {time.time() - t:.0f} s, rc={p.returncode}")
    return r


class Checks:
    def __init__(self):
        self.n = 0
        self.fail = []

    def __call__(self, ok, what):
        self.n += 1
        print(('  ok    ' if ok else '  FAIL  ') + what)
        if not ok:
            self.fail.append(what)


def ss(mult):
    s = (mult - 1) / 2
    return s * (s + 1)


def check_projected(chk, r, mult):
    chk(r.rc == 0 and len(r.E) > 0, f"{r.name}: finished, {len(r.E)} roots")
    if r.rc or not r.E:
        return
    chk(r.res is not None and r.res < 1e-10, f"{r.name}: max |(S^2 - S(S+1)) c| = {r.res}")
    chk(all(abs(s - ss(mult)) < 0.006 for s in r.S2), f"{r.name}: S^2 = {r.S2} (S(S+1) = {ss(mult)})")


def same_energies(chk, a, b, tol, what):
    ok = a.E and b.E and len(a.E) == len(b.E) and max(abs(x - y) for x, y in zip(a.E, b.E)) < tol
    d = max(abs(x - y) for x, y in zip(a.E, b.E)) if a.E and b.E and len(a.E) == len(b.E) else None
    chk(bool(ok), f"{what}: {a.name} vs {b.name}, max |dE| = {d if d is None else f'{d:.1e}'} (tol {tol:.0e})")


def in_reference(chk, r, ref, mult, tol):
    lo = sorted(e for e, s in zip(ref.E, ref.S2) if abs(s - ss(mult)) < 0.006)[:len(r.E)]
    ok = len(lo) == len(r.E) and max(abs(x - y) for x, y in zip(sorted(r.E), lo)) < tol
    d = max(abs(x - y) for x, y in zip(sorted(r.E), lo)) if len(lo) == len(r.E) else None
    chk(ok, f"{r.name}: lowest S={(mult - 1) / 2} states of the unprojected spectrum {ref.name}, "
            f"max |dE| = {d if d is None else f'{d:.1e}'} (tol {tol:.0e})")


def casci_family(args, chk, work, mol, charge, nel, nval, mo, tag, spec):
    """spec: list of (na, nb, mult, ns). The reference has mult=0 and the smallest Ms of the list."""
    na0, nb0 = min(((na, nb) for na, nb, _, _ in spec), key=lambda x: x[0] - x[1])
    ref = run(args, work, write_input(work, f'{tag}_all', mol, charge, na0, nb0, nval, 0, 30, 0, mo), 30)
    chk(ref.rc == 0 and len(ref.E) == 30, f"{ref.name}: unprojected reference, 30 roots of all spins")
    runs = {}
    for na, nb, mult, ns in spec:
        r = run(args, work, write_input(work, f'{tag}_s{mult}_m{na - nb}', mol, charge, na, nb, nval, mult, ns, 1, mo), ns)
        check_projected(chk, r, mult)
        if r.E and ref.E:
            in_reference(chk, r, ref, mult, 1e-7)
        runs.setdefault(mult, []).append(r)
    for mult, rs in runs.items():
        for r in rs[1:]:
            same_energies(chk, rs[0], r, 1e-8, f"S={(mult - 1) / 2}, different Ms")
    # the same mult without projection: with pure spin-adapted guesses the small CASCI stays pure here
    na, nb, mult, ns = spec[0]
    r0 = runs[mult][0]
    u = run(args, work, write_input(work, f'{tag}_s{mult}_m{na - nb}_noproj', mol, charge, na, nb, nval, mult, ns, 0, mo), ns)
    if all(abs(s - ss(mult)) < 0.006 for s in u.S2):
        same_energies(chk, r0, u, 1e-8, "projection off, pure roots: same energies")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default=os.path.join(HERE, 'work'))
    ap.add_argument('--cmd', default='{root}/run_sQM -i {inp}')
    ap.add_argument('--unit-cmd', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    chk = Checks()
    t0 = time.time()

    print("== algebra of the projector (run_spinprojtest)")
    unit = args.unit_cmd or (os.path.join(ROOT, 'run_spinprojtest') if os.path.exists(os.path.join(ROOT, 'run_spinprojtest')) else None)
    if unit:
        p = subprocess.run(unit, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = [l for l in p.stdout.splitlines() if l.startswith('checks:')]
        chk(p.returncode == 0 and 'spinprojtest: PASS' in p.stdout, f"run_spinprojtest: {last[0] if last else p.stdout[-300:]}")
    else:
        print("  skipped (build it with: make run_spinprojtest)")

    print("== orbitals: RHF of N2, CH2 and the allyl anion")
    for name, mol, charge, na, nb, nv in (('n2', 'n2', 0, 3, 3, 6), ('ch2', 'ch2', 0, 3, 3, 6), ('allyl', 'allyl', -1, 2, 2, 3)):
        r = run(args, work, write_input(work, name, mol, charge, na, nb, nv, 1, 1, 0, rhf_only=True), 0)
        chk(r.rc == 0 and os.path.exists(os.path.join(work, name + '_RHF.orb')), f"{name}: RHF orbitals")

    print("== N2 at 2.0 A, CAS(6,6) CASCI: S = 0..3, Ms = 0..3")
    casci_family(args, chk, work, 'n2', 0, 6, 6, 'n2_RHF.orb', 'n2', [
        (3, 3, 1, 4), (3, 3, 3, 4), (3, 3, 5, 3), (3, 3, 7, 1),
        (4, 2, 3, 4), (4, 2, 5, 3), (4, 2, 7, 1),
        (5, 1, 5, 3), (5, 1, 7, 1),
        (6, 0, 7, 1)])

    print("== N2+ at 2.0 A, CAS(5,6) CASCI: S = 1/2..5/2, Ms = 1/2..5/2")
    casci_family(args, chk, work, 'n2', 1, 5, 6, 'n2_RHF.orb', 'n2p', [
        (3, 2, 2, 4), (3, 2, 4, 4), (3, 2, 6, 1),
        (4, 1, 4, 4), (4, 1, 6, 1),
        (5, 0, 6, 1)])

    print("== CH2, CAS(6,6) SA-CASSCF")
    cas = {}
    for na, nb, mult, ns in ((3, 3, 1, 3), (3, 3, 3, 2), (4, 2, 3, 2), (3, 3, 5, 1), (5, 1, 5, 1)):
        r = run(args, work, write_input(work, f'ch2_s{mult}_m{na - nb}', 'ch2', 0, na, nb, 6, mult, ns, 1, 'ch2_RHF.orb', 'casscf'), ns)
        check_projected(chk, r, mult)
        cas.setdefault(mult, []).append(r)
    for mult in (3, 5):
        same_energies(chk, cas[mult][0], cas[mult][1], 1e-8, f"SA-CASSCF S={(mult - 1) / 2}, different Ms")
    u = run(args, work, write_input(work, 'ch2_s1_m0_noproj', 'ch2', 0, 3, 3, 6, 1, 3, 0, 'ch2_RHF.orb', 'casscf'), 3)
    if all(abs(s) < 0.006 for s in u.S2):
        same_energies(chk, cas[1][0], u, 1e-8, "SA-CASSCF singlets, projection off (pure roots): same energies")

    print("== allyl radical, CAS(3,3) SA-CASSCF")
    a2 = run(args, work, write_input(work, 'allyl_s2_m1', 'allyl', 0, 2, 1, 3, 2, 2, 1, 'allyl_RHF.orb', 'casscf'), 2)
    check_projected(chk, a2, 2)
    a4 = [run(args, work, write_input(work, f'allyl_s4_m{na - nb}', 'allyl', 0, na, nb, 3, 4, 1, 1, 'allyl_RHF.orb', 'casscf'), 1)
          for na, nb in ((2, 1), (3, 0))]
    for r in a4:
        check_projected(chk, r, 4)
    same_energies(chk, a4[0], a4[1], 1e-8, "SA-CASSCF S=3/2, different Ms")
    u = run(args, work, write_input(work, 'allyl_s2_m1_noproj', 'allyl', 0, 2, 1, 3, 2, 2, 0, 'allyl_RHF.orb', 'casscf'), 2)
    same_energies(chk, a2, u, 1e-8, "SA-CASSCF doublets, projection off (pure roots): same energies")

    print("== input errors")
    # a spin the space does not hold; more roots than states of that spin (one septet in CAS(6,6), one quartet in CAS(3,3))
    for name, mol, charge, nv, na, nb, mult, ns, cas, mo in (
            ('err_s0_m2', 'n2', 0, 6, 4, 2, 1, 1, 'casci', 'n2_RHF.orb'),
            ('err_s4_m0', 'n2', 0, 6, 3, 3, 9, 1, 'casci', 'n2_RHF.orb'),
            ('err_all', 'n2', 0, 6, 3, 3, 0, 1, 'casci', 'n2_RHF.orb'),
            ('err_s7_2roots', 'n2', 0, 6, 3, 3, 7, 2, 'casci', 'n2_RHF.orb'),
            ('err_allyl_s4_2roots', 'allyl', 0, 3, 2, 1, 4, 2, 'casscf', 'allyl_RHF.orb')):
        r = run(args, work, write_input(work, name, mol, charge, na, nb, nv, mult, ns, 1, mo, cas), 0)
        chk(r.rc != 0 and 'ERROR: spin_proj' in r.log,
            f"{name}: n_alp={na} n_bet={nb} mult={mult} n_s={ns} stops with an error")

    if not args.quick:
        print("== guanine, CAS(10,10) SA-CASSCF (about a minute per run)")
        g = {}
        for name, na, nb, mult, ns, proj in (('gu_s1_m0', 5, 5, 1, 4, 1), ('gu_s1_m0_noproj', 5, 5, 1, 4, 0),
                                             ('gu_s3_m0', 5, 5, 3, 2, 1), ('gu_s3_m2', 6, 4, 3, 2, 1)):
            g[name] = run(args, work, write_input(work, name, 'guanine', 0, na, nb, 10, mult, ns, proj, None, 'casscf'), ns)
        check_projected(chk, g['gu_s1_m0'], 1)
        u = g['gu_s1_m0_noproj']
        chk(u.rc == 0 and any(s > 1 for s in u.S2), f"{u.name}: without projection the singlet ensemble has triplets, S^2 = {u.S2}")
        for k in ('gu_s3_m0', 'gu_s3_m2'):
            check_projected(chk, g[k], 3)
        same_energies(chk, g['gu_s3_m0'], g['gu_s3_m2'], 1e-7, "SA-CASSCF triplets, different Ms")

    print(f"\n{chk.n} checks, {len(chk.fail)} failed, {time.time() - t0:.0f} s")
    for f in chk.fail:
        print("FAIL:", f)
    print("spin_proj tests: " + ("FAIL" if chk.fail else "PASS"))
    return 1 if chk.fail else 0


if __name__ == '__main__':
    sys.exit(main())
