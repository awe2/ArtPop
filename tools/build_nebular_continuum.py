#!/usr/bin/env python
"""
Build (or verify) the nebular continuum table used by `artpop.nebular`.

The ionized gas that emits the lines also emits a continuum: free-bound
(recombination; H I with the Balmer, Paschen, ... jumps, and He I),
two-photon (H 2s -> 1s) and free-free. Per recombination it is fixed atomic
physics, so the table stores it **per Angstrom, relative to the Hbeta line
flux** at the same electron temperature, ``c(lambda; T_e)`` [1/A]. The model
then gets the continuum of each O/B row as ``L(Hbeta) * c``, with the same
``k Q_H`` scaling, dust and placement as the lines.

Computed with PyNeb's `Continuum` (Luridiana, Morisset & Shaw 2015): free-bound
from Ercolano & Storey 2006 (H I, He I), two-photon from D. Pequignot's fit to
Osterbrock's tabulation, free-free from Storey & Hummer 1991; normalised to
PyNeb's own Hbeta emissivity (Storey & Hummer 1995, case B). Density
``n_e = 100 cm^-3``, the case-B point ``HBETA_ERG_PER_ION`` is quoted at (the
two-photon term changes by 0.6 % between 1 and 100 cm^-3).

The continuum is linear in the ion abundances, so two components are stored
per temperature node:

* ``h_<T>``: everything from H+ (H I free-bound, two-photon, free-free off H+);
* ``he1_<T>``: per unit He+/H+ (He I free-bound and free-free off He+).

``c = h + (He+/H+) * he1``. He++ (He II) is not included: the MAPPINGS HII
grid's ionizing spectrum makes little He++ and our lines carry none of note.

Wavelengths are **vacuum** (PyNeb's continuum puts the Balmer jump between
3647.017 and 3647.018 A, the vacuum series limit), 1230-30000 A
(two-photon starts at Lyman alpha; 30000 A is past Roman F213 at z = 0),
2 A steps to 12000 A and 5 A beyond, with the H I series limits inserted
0.01 A either side so each jump is a step of the quadrature, not a ramp.

Temperature nodes 5000-25000 K every 2500 K; `artpop.nebular` interpolates
``log c`` linearly in ``log T``. Leave-one-out at the mid-points (2400-24000 A):
<= 1.6 % at 6250 K, <= 0.4 % above 10 kK.

``--check`` recomputes the table from PyNeb and compares it with the file.

Usage::

    pip install pyneb                              # build-time only
    python tools/build_nebular_continuum.py
    python tools/build_nebular_continuum.py --check
"""
import argparse
import os
import sys
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from artpop.nebular import default_continuum_table_path  # noqa: E402

T_NODES = np.arange(5000.0, 25000.1, 2500.0)
DENSITY = 100.0
HE_PROBE = 0.1          # He+/H+ used to separate the He component (linear)
# H I series limits (Balmer, Paschen, Brackett, Pfund) where PyNeb's free-bound
# data actually step, located to 0.001 A (they follow its level energies, about
# 0.04 A blue of 911.7636 n^2)
H_EDGES_AA = (3647.0175, 8205.8265, 14588.1655, 22794.0345)


def wavelength_grid():
    lam = np.concatenate([np.arange(1230.0, 12000.0, 2.0),
                          np.arange(12000.0, 30000.0 + 0.1, 5.0)])
    extra = np.concatenate([[e - 0.01, e + 0.01] for e in H_EDGES_AA])
    return np.unique(np.round(np.concatenate([lam, extra]), 3))


def build():
    import pyneb as pn
    from astropy.table import Table
    lam = wavelength_grid()
    cont = pn.Continuum()
    cols = {'lambda_vac': lam}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for t in T_NODES:
            kw = dict(tem=t, den=DENSITY, He2_H=0.0, wl=lam, HI_label='4_2',
                      cont_HeII=False)
            h = cont.get_continuum(He1_H=0.0, **kw)
            he = (cont.get_continuum(He1_H=HE_PROBE, **kw) - h) / HE_PROBE
            cols[f'h_{int(t)}'] = h
            cols[f'he1_{int(t)}'] = he
    tab = Table(cols)
    for name in tab.colnames[1:]:
        tab[name] = tab[name].astype(np.float32)
    tab.meta.update({
        'description': 'Nebular continuum per Angstrom relative to F(Hbeta): '
                       'c = h_<T> + (He+/H+) * he1_<T>, vacuum wavelengths (A)',
        'units': '1/Angstrom (L_lambda / L(Hbeta))',
        't_nodes_k': [float(t) for t in T_NODES],
        'density_cm3': DENSITY,
        'interpolation': 'log c linear in log T (artpop.nebular.NebularContinuumTable)',
        'source': f'PyNeb {pn.__version__} Continuum.get_continuum (HI_label 4_2, case B)',
        'references': ['Luridiana, Morisset & Shaw 2015, A&A 573, A42 (PyNeb)',
                       'Ercolano & Storey 2006, MNRAS 372, 1875 (free-bound H I, He I)',
                       'Storey & Hummer 1991, CoPhC 66, 129 (free-free)',
                       'Storey & Hummer 1995, MNRAS 272, 41 (Hbeta, case B)',
                       'two-photon: D. Pequignot fit to Osterbrock & Ferland 2006'],
        'licence': 'PyNeb is GPL-3.0; this table is computed output (coefficients), '
                   'cite the references above',
        'builder': 'tools/build_nebular_continuum.py',
    })
    return tab


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default=default_continuum_table_path())
    p.add_argument('--check', action='store_true',
                   help='recompute and compare with --out; write nothing')
    args = p.parse_args(argv)
    tab = build()
    if args.check:
        from astropy.table import Table
        old = Table.read(args.out, format='ascii.ecsv')
        if old.colnames != tab.colnames or len(old) != len(tab):
            raise SystemExit(f'{args.out}: columns or length differ')
        worst = max(float(np.max(np.abs(np.asarray(old[c], float) - np.asarray(tab[c], float))
                                 / np.maximum(np.abs(np.asarray(tab[c], float)), 1e-30)))
                    for c in tab.colnames)
        print(f'{args.out}: {len(tab)} rows x {len(tab.colnames)} columns, '
              f'max relative difference {worst:.2e}')
        if worst > 1e-6:
            raise SystemExit('check FAILED')
        print('check OK')
        return
    tab.write(args.out, format='ascii.ecsv', overwrite=True)
    print(f'wrote {args.out}: {len(tab)} rows, {os.path.getsize(args.out) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
