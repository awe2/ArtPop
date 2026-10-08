#!/usr/bin/env python
"""
Build (or verify) the high-order H I recombination-line table used by
`artpop.nebular`.

The MAPPINGS line list NebulaBayes distributes stops at Hdelta, Padelta and
Brgamma: Thomas et al. (2018, ApJ 856, 89, sec. 3.2) "exclude higher-order
recombination lines (e.g. H8, Pa7, etc.) that are difficult to predict
correctly without a very complete recombination-cascade solution as a function
of density". That complete solution exists for hydrogen: the case-B
emissivities of Storey & Hummer (1995), which PyNeb implements (n_upper <= 50)
and from which our Hbeta per recombination (`HBETA_ERG_PER_ION`) and the
nebular continuum's normalisation already come. Per captured photon these
lines are fixed atomic physics, like the continuum, so they need no MAPPINGS
grid and no knob: ``L(line) = L(Hbeta) * r(T_e)``.

Lines (each emission series, lower level n_l, from the first upper level the
MAPPINGS list does not carry, up to n_u = 40; vacuum wavelengths from the
Rydberg formula with R_H = 109677.583 cm^-1):

* Balmer   (n_l = 2): n_u >= 7  (H7 = Hepsilon 3971 A, H8 3890 A, ...)
* Paschen  (n_l = 3): n_u >= 8  (Pa8 = Paepsilon 9548 A, Pa9 9232 A, ...)
* Brackett (n_l = 4): n_u >= 8  (Br8 = Brdelta 19451 A, ...)
* Pfund    (n_l = 5): those of n_u <= 40 shortward of ``LAMBDA_MAX`` (30000 A)

Above n_u ~ 40 the lines crowd into the series limit, where the nebular
continuum table's free-bound jump takes over; n_u = 41-50 together are 0.7 %
of Hbeta (Balmer), 0.2 % (Paschen), 0.1 % (Brackett) at 10^4 K (measured
2026-10-08).

Stored per line: ``r_<T>`` = emissivity / emissivity(Hbeta) at temperature
nodes 5000-25000 K every 2500 K (the continuum table's nodes), n_e = 100 cm^-3
(``HBETA_ERG_PER_ION``'s density and the lowest in PyNeb's SH95 tables).
Density, measured at 10^4 K from 1e2 to 1e4 cm^-3: n_u <= 15 move <= 4 %;
the weak n_u >= 20 members rise by 14-48 % (collisional redistribution among
high levels); the summed Balmer n_u 7-40 rises 3.8 % (0.615 -> 0.638 Hbeta),
the summed Paschen n_u 8-40 3.1 % (0.158 -> 0.163). `artpop.nebular`
interpolates ``log r`` linearly in ``log T`` (<= 0.7 % at mid-nodes, 11250 K).
Each series falls monotonically with n_u except a 1 % Br29 < Br30 inversion
(3.3e-4 Hbeta) in the tabulated SH95 data, kept as is.
Junction with MAPPINGS: its Hgamma, Hdelta, Pagamma, Padelta, Brgamma agree with
these case-B ratios within 0.5-4 % at 10^4 K and ~1 % at 1.2 x 10^4 K (O/H 8.2,
log U -3, log P 6.2).

``--check`` recomputes the table from PyNeb and compares it with the file.

Usage::

    pip install pyneb                              # build-time only
    python tools/build_hi_recombination_lines.py
    python tools/build_hi_recombination_lines.py --check
"""
import argparse
import os
import sys
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from artpop.nebular import default_hi_line_table_path  # noqa: E402

T_NODES = np.arange(5000.0, 25000.1, 2500.0)
DENSITY = 100.0
N_UPPER_MAX = 40
LAMBDA_MAX = 30000.0
R_H = 109677.583                                  # cm^-1, hydrogen (reduced mass)
# (series name, lower level, first upper level not in the MAPPINGS list)
SERIES = (('H', 2, 7), ('Pa', 3, 8), ('Br', 4, 8), ('Pf', 5, 6))


def lambda_vac(n_u, n_l):
    return 1.0e8 / (R_H * (1.0 / n_l ** 2 - 1.0 / n_u ** 2))


def build():
    from astropy.table import Table
    import pyneb as pn
    warnings.filterwarnings('ignore')
    h = pn.RecAtom('H', 1)
    hb = np.array([h.getEmissivity(t, DENSITY, label='4_2') for t in T_NODES])
    rows = []
    for name, n_l, n_first in SERIES:
        for n_u in range(n_first, N_UPPER_MAX + 1):
            lam = lambda_vac(n_u, n_l)
            if lam > LAMBDA_MAX:
                continue
            em = np.array([h.getEmissivity(t, DENSITY, label=f'{n_u}_{n_l}') for t in T_NODES])
            rows.append((f'{name}{n_u}', name, n_u, n_l, lam, em / hb))
    t = Table()
    t['name'] = [r[0] for r in rows]
    t['series'] = [r[1] for r in rows]
    t['n_upper'] = np.array([r[2] for r in rows], dtype=np.int16)
    t['n_lower'] = np.array([r[3] for r in rows], dtype=np.int16)
    t['lambda_vac'] = np.array([r[4] for r in rows])
    t['lambda_vac'].format = '.4f'
    for j, T in enumerate(T_NODES):
        col = f'r_{int(T)}'
        t[col] = np.array([r[5][j] for r in rows])
        t[col].format = '.6e'
    t.meta = {
        'description': 'Case-B H I recombination lines relative to Hbeta, the '
                       'high-order members the MAPPINGS (NebulaBayes) list omits',
        'source': f'PyNeb {pn.__version__} RecAtom("H", 1): Storey & Hummer 1995, case B',
        'citations': ['Storey & Hummer 1995, MNRAS 272, 41',
                      'Luridiana, Morisset & Shaw 2015, A&A 573, A42 (PyNeb)'],
        'density_cm3': DENSITY,
        't_nodes_k': [float(x) for x in T_NODES],
        'n_upper_max': N_UPPER_MAX,
        'wavelengths': 'vacuum, Angstrom, Rydberg formula, R_H = 109677.583 cm^-1',
        'series_first_upper': {s: n for s, _, n in SERIES},
        'interpolation': 'log r linear in log T (artpop.nebular.HILineTable)',
    }
    return t


def _compare(a, b):
    if a.colnames != b.colnames or len(a) != len(b):
        return f'columns/rows differ: {a.colnames} x {len(a)} vs {b.colnames} x {len(b)}'
    for c in a.colnames:
        x, y = np.asarray(a[c]), np.asarray(b[c])
        if x.dtype.kind in 'fc':
            if not np.allclose(x, y, rtol=2e-6, atol=0):
                return f'column {c} differs'
        elif not np.array_equal(x, y):
            return f'column {c} differs'
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default=default_hi_line_table_path())
    p.add_argument('--check', action='store_true',
                   help='compare the file to a fresh computation; write nothing')
    args = p.parse_args(argv)
    table = build()
    if args.check:
        from astropy.table import Table
        from artpop.nebular import HILineTable
        if not os.path.isfile(args.out):
            print(f'MISSING {args.out}')
            return 1
        err = _compare(table, Table.read(args.out, format='ascii.ecsv'))
        HILineTable(args.out)
        print(f'{"OK" if err is None else "MISMATCH: " + err}  {args.out}')
        return 0 if err is None else 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f'{args.out}.{os.getpid()}.part'
    table.write(tmp, format='ascii.ecsv', overwrite=True)
    os.replace(tmp, args.out)
    print(f'wrote {args.out}: {len(table)} lines, T nodes {T_NODES[0]:g}-{T_NODES[-1]:g} K')
    return 0


if __name__ == '__main__':
    sys.exit(main())
