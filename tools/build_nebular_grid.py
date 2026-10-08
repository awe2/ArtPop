#!/usr/bin/env python
"""
Build (or verify) the MAPPINGS line table used by `artpop.nebular`.

The source is the MAPPINGS V 5.1 HII-region grid that NebulaBayes distributes
(``NebulaBayes/grids/NB_HII_grid.fits.gz`` + ``Linelist.csv``; Thomas et al.
2018, ApJ 856, 89). Its axes are 12 + log O/H (12 nodes, 7.06-9.30), log U
(9 nodes, -4 to -2) and log P/k (12 nodes, 4.2-8.6); every line flux is
relative to Hbeta = 1. The MAPPINGS site distributes the code, not grids, so a
bespoke grid (another ionizing spectrum, a denser U axis) would mean running
MAPPINGS itself -- a later upgrade; this table keeps the same format either
way.

This script copies the three axes and every line with rest wavelength in
``[--lambda-min, --lambda-max]`` (default 1250-30000 A: Lyman alpha and the
far-UV lines no rendered band can reach are dropped) into
``src/artpop/data/nebular/mappings51_hii_lines.ecsv``, with the air
wavelengths and the provenance in the header. Values are copied, not
resampled: the float32 grid round-trips exactly.

**Derived companions (2026-10-08).** The NebulaBayes line list carries only the
strong member of two fixed-ratio doublets: ``OIII5007`` (5006.8 A) without
[O III] 4959 and ``NII6583`` (6583.5 A) without [N II] 6548. Both members of
each pair come from the same upper level (1D2), so their ratio is the ratio of
the transition probabilities times the photon energies -- independent of
T_e, n_e and the model -- and MAPPINGS V computes both (Sutherland & Dopita
2017 use the full CHIANTI 8 line list). Thomas et al. (2018, sec. 3.2) keep
every line above 1 % of Hbeta in some model, which 4959 and 6548 always pass,
so their absence is a choice of the distributed list, not of the models; the
paper does not state it. The grid's columns are the single lines: summed
doublets are named with a suffix (``OII3726_29``, ``SII6716_31``), Linelist.csv
gives 5006.8 and 6583.5 A, and the pairs the grid does list twice sit exactly at
their atomic ratios ([O I] 6300/6364 = 3.13, [S III] 9531/9069 = 2.51,
[Ne III] 3869/3967 = 3.32). This script therefore adds ``OIII4959 =
OIII5007 / 2.984`` and ``NII6548 = NII6583 / 2.942`` (PyNeb 1.1.32 default
atomic data: O III Froese Fischer & Tachiev 2004 / Storey & Zeippen 2000,
N II Froese Fischer & Tachiev 2004; the same at every T_e and n_e), listed in
the header under ``derived_lines``.

``--check`` rebuilds the table in memory from the source and compares it to
the file, writing nothing.

Licences: MAPPINGS is CC-BY-SA-4.0 (Sutherland & Dopita 2017; cite them and
Thomas et al. 2018); NebulaBayes is MIT. The derived table carries both in its
header.

Usage::

    pip install NebulaBayes                       # build-time only
    python tools/build_nebular_grid.py
    python tools/build_nebular_grid.py --check
    python tools/build_nebular_grid.py --source /path/to/NebulaBayes/grids
"""
import argparse
import csv
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from artpop.nebular import MappingsLineTable, default_line_table_path  # noqa: E402

_AXES = {'12 + log O/H': 'oh', 'log U': 'log_u', 'log P/k': 'log_p'}

# the weak member of a fixed-ratio doublet the source list leaves out:
# name -> (parent column, strong/weak intensity ratio, air wavelength [A], source)
DERIVED = {
    'OIII4959': ('OIII5007', 2.984, 4958.911,
                 '[O III] 1D2: A-values FFT04/SZ00 via PyNeb 1.1.32 (T- and n-independent)'),
    'NII6548': ('NII6583', 2.942, 6548.050,
                '[N II] 1D2: A-values FFT04 via PyNeb 1.1.32 (T- and n-independent)'),
}


def _source_dir(arg):
    if arg:
        return arg
    try:
        import NebulaBayes
    except ImportError:
        raise SystemExit('NebulaBayes is not installed; pip install NebulaBayes '
                         'or pass --source <.../NebulaBayes/grids>')
    return os.path.join(os.path.dirname(NebulaBayes.__file__), 'grids')


def _nebulabayes_version(src):
    # an unpacked wheel or site-packages keeps its dist-info beside the package
    import glob
    for meta in glob.glob(os.path.join(src, os.pardir, os.pardir,
                                       'NebulaBayes-*.dist-info', 'METADATA')):
        with open(meta) as fh:
            for line in fh:
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    try:
        import NebulaBayes
        return getattr(NebulaBayes, '__version__', 'unknown')
    except ImportError:
        return f'unknown (from {src})'


def build(src, lam_min, lam_max):
    from astropy.table import Table
    grid = Table.read(os.path.join(src, 'NB_HII_grid.fits.gz'))
    with open(os.path.join(src, 'Linelist.csv')) as fh:
        lam_air = {row['Grid_name'].strip(): float(row['Lambda_AA'])
                   for row in csv.DictReader(fh)}
    for ax in _AXES:
        if ax not in grid.colnames:
            raise SystemExit(f'{src}: grid has no {ax!r} column')
    if 'Hbeta' not in grid.colnames or not np.allclose(grid['Hbeta'], 1.0):
        raise SystemExit(f'{src}: expected every line relative to Hbeta = 1')

    lines = [c for c in grid.colnames if c not in _AXES
             and c in lam_air and lam_min <= lam_air[c] <= lam_max]
    missing = [c for c in grid.colnames if c not in _AXES and c not in lam_air]
    if missing:
        print(f'note: no wavelength in Linelist.csv for {missing}; dropped')

    out = Table()
    for ax, name in _AXES.items():
        out[name] = np.asarray(grid[ax], dtype=np.float32)
    # MAPPINGS prints a line only above 1e-5 of Hbeta (the smallest finite
    # value anywhere in the grid is exactly 1.0e-5); below it the grid holds
    # NaN -- e.g. [OIII]5007 at the highest O/H and lowest U, where it truly
    # vanishes. NaN therefore means "< 1e-5 Hbeta" and is stored as 0.
    n_nan = {}
    for c in lines:
        v = np.asarray(grid[c], dtype=np.float32)
        bad = ~np.isfinite(v)
        if bad.any():
            n_nan[c] = int(bad.sum())
        out[c] = np.where(bad, np.float32(0.0), v)
    derived = {}
    for name, (parent, ratio, lam, why) in DERIVED.items():
        if name in grid.colnames:
            raise SystemExit(f'{src}: the grid already has {name}; drop it from DERIVED')
        if parent not in out.colnames:
            continue                                   # parent outside the wavelength cut
        if lam_min <= lam <= lam_max:
            out[name] = (np.asarray(out[parent], dtype=np.float64) / ratio).astype(np.float32)
            lam_air[name] = lam
            lines.append(name)
            derived[name] = {'parent': parent, 'parent_over_this': ratio, 'source': why}
    out.meta = {
        'description': 'MAPPINGS V HII-region line fluxes relative to Hbeta, '
                       'converted for artpop.nebular',
        'mappings_version': str(grid.meta.get('MPG_VRSN', 'unknown')),
        'source': 'NebulaBayes NB_HII_grid.fits.gz + Linelist.csv',
        'nebulabayes_version': _nebulabayes_version(src),
        'citations': ['Sutherland & Dopita 2017, ApJS 229, 34 (MAPPINGS V)',
                      'Thomas et al. 2018, ApJ 856, 89 (NebulaBayes grid)'],
        'licence': 'MAPPINGS: CC-BY-SA-4.0; NebulaBayes: MIT',
        'axes': {'oh': '12 + log O/H', 'log_u': 'log ionization parameter',
                 'log_p': 'log P/k [K cm^-3]'},
        'wavelengths': 'air, Angstrom (artpop converts to vacuum with air_to_vac)',
        'lambda_air_aa': {c: lam_air[c] for c in lines},
        'below_print_threshold': 'NaN in the source (a line under MAPPINGS\' '
                                 '1e-5 Hbeta print threshold) stored as 0',
        'n_nan_set_to_zero': n_nan,
        'derived_lines': derived,
    }
    return out


def _compare(a, b):
    if a.colnames != b.colnames or len(a) != len(b):
        return f'columns/rows differ: {len(a.colnames)}x{len(a)} vs {len(b.colnames)}x{len(b)}'
    for c in a.colnames:
        if not np.array_equal(np.asarray(a[c]), np.asarray(b[c])):
            return f'column {c} differs'
    if a.meta.get('lambda_air_aa') != b.meta.get('lambda_air_aa'):
        return 'line wavelengths differ'
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--source', help='directory holding NB_HII_grid.fits.gz '
                                    'and Linelist.csv (default: installed NebulaBayes)')
    p.add_argument('--out', default=default_line_table_path())
    p.add_argument('--lambda-min', type=float, default=1250.0)
    p.add_argument('--lambda-max', type=float, default=30000.0)
    p.add_argument('--check', action='store_true',
                   help='compare the file to a fresh conversion; write nothing')
    args = p.parse_args(argv)

    src = _source_dir(args.source)
    table = build(src, args.lambda_min, args.lambda_max)
    if args.check:
        from astropy.table import Table
        if not os.path.isfile(args.out):
            print(f'MISSING {args.out}')
            return 1
        err = _compare(table, Table.read(args.out, format='ascii.ecsv'))
        # and the reader must accept it
        MappingsLineTable(args.out)
        print(f'{"OK" if err is None else "MISMATCH: " + err}  {args.out}')
        return 0 if err is None else 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f'{args.out}.{os.getpid()}.part'
    table.write(tmp, format='ascii.ecsv', overwrite=True)
    os.replace(tmp, args.out)
    t = MappingsLineTable(args.out)
    print(f'wrote {args.out}: {len(t.names)} lines on a '
          f'{"x".join(str(a.size) for a in t.axes)} (O/H, log U, log P) grid')
    return 0


if __name__ == '__main__':
    sys.exit(main())
