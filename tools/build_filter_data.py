#!/usr/bin/env python
"""
Rebuild the three pickles in ``src/artpop/data`` from ``filter_curves/``.

These files -- ``mist_filter_names.pkl``, ``phot_system_lookup.pkl`` and
``filter_properties.pkl`` -- ship with ArtPop and are read at import time, but
until now there was no script that produced them. That is why adding a
photometric system meant hand-editing pickles, and it is a large part of why the
WFIRST filter naming drifted out of step with MIST's zero point table.

Run from the repository root::

    python tools/build_filter_data.py --check    # compare, write nothing
    python tools/build_filter_data.py            # regenerate in place

``--check`` is the one that matters: it regenerates in memory and diffs against
what is committed. It must pass before you trust a regeneration, because
``filter_properties`` feeds `~artpop.image.Imager`'s magnitude-to-counts
conversion and a silent change there is very hard to notice downstream.

Filter names come from the CSV file stems, so ``filter_curves/LSST/LSST_u.csv``
gives ``LSST_u`` and ``filter_curves/Roman/Roman_F158.csv`` gives
``Roman_F158``. That convention is the contract: the stems must match the
isochrone column names MIST delivers for that system.
"""
import argparse
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))

from artpop.filters import FilterSystem, phot_system_list  # noqa: E402

CURVE_DIR = os.path.join(REPO, 'filter_curves')
DATA_DIR = os.path.join(REPO, 'src', 'artpop', 'data')


def filter_order(system, found):
    """
    Order a system's filters the way MIST lists its isochrone columns.

    That order is not recoverable from the CSVs: it is grouped by instrument and
    only then by wavelength (UBVRIplus runs Bessell, 2MASS, Kepler, Hipparcos,
    Tycho, Gaia; HST_WFC3 runs all UVIS then all IR), so neither an alphabetical
    nor a wavelength sort reproduces it. So for a system that is already
    committed we keep its order exactly and only check the SET still matches --
    reordering these lists would silently permute `filter_properties`, which
    `~artpop.image.Imager` indexes by name but users read positionally.

    A genuinely new system is ordered by the MIST grid on disk if one is
    present, since that is the authority, and by effective wavelength otherwise.
    """
    try:
        committed = load('mist_filter_names.pkl').get(system)
    except FileNotFoundError:
        committed = None
    if committed and set(committed) == set(found):
        return list(committed)
    if committed:
        raise SystemExit(
            f'{system}: filter set changed ({sorted(set(committed) ^ set(found))}'
            ') -- resolve by hand before regenerating')

    grid = _mist_column_order(system)
    if grid:
        missing = [f for f in found if f not in grid]
        return [f for f in grid if f in found] + sorted(missing)
    return sorted(found)


def _mist_column_order(system):
    """Filter column order from a MIST grid on disk, if one can be found."""
    import glob
    from artpop import MIST_PATH
    hits = glob.glob(os.path.join(MIST_PATH, f'*{system}*', f'*.iso.{system}'))
    hits += glob.glob(os.path.join(MIST_PATH, f'*{system}*', '*.iso.cmd'))
    if not hits:
        return None
    with open(sorted(hits)[0]) as f:
        lines = [next(f) for _ in range(13)]
    return [t for t in lines[12].replace('#', ' ').split() if t.startswith(system)]


def collect():
    """Walk filter_curves/ and build the three data structures."""
    mist_filter_names, lookup = {}, {}
    bandpass, dlam, lam_eff = [], [], []

    for system in sorted(os.listdir(CURVE_DIR)):
        sys_dir = os.path.join(CURVE_DIR, system)
        if not os.path.isdir(sys_dir):
            continue
        if system not in phot_system_list:
            print(f'  skipping {system}: not in phot_system_list')
            continue
        found = [f[:-4] for f in os.listdir(sys_dir) if f.endswith('.csv')]
        names = filter_order(system, found)
        files = [os.path.join(sys_dir, f'{n}.csv') for n in names]
        mist_filter_names[system] = names
        for n in names:
            lookup[n] = system
        # the CSVs carry a `wave,trans` header row
        fs = FilterSystem(files, names, delimiter=',', skiprows=1)
        for n in names:
            bandpass.append(n)
            dlam.append(float(fs.dlam(n).value))
            lam_eff.append(float(fs.lam_eff(n).value))
        print(f'  {system}: {len(names)} filters')

    props = {'bandpass': bandpass, 'dlam': dlam, 'lam_eff': lam_eff}
    return mist_filter_names, lookup, props


def load(name):
    with open(os.path.join(DATA_DIR, name), 'rb') as f:
        return pickle.load(f)


def compare(name, built):
    """Diff a regenerated structure against the committed pickle."""
    try:
        have = load(name)
    except FileNotFoundError:
        print(f'  {name}: not present, would be created')
        return True
    ok = True
    if name == 'filter_properties.pkl':
        missing = set(have['bandpass']) - set(built['bandpass'])
        added = set(built['bandpass']) - set(have['bandpass'])
        if missing:
            print(f'  {name}: LOST {sorted(missing)}')
            ok = False
        if added:
            print(f'  {name}: adds {sorted(added)}')
        idx = {b: i for i, b in enumerate(built['bandpass'])}
        for i, b in enumerate(have['bandpass']):
            if b not in idx:
                continue
            j = idx[b]
            for key in ('dlam', 'lam_eff'):
                a, c = float(have[key][i]), float(built[key][j])
                if not np.isclose(a, c, rtol=1e-6, atol=0):
                    print(f'  {name}: {b} {key} {a!r} -> {c!r}')
                    ok = False
    else:
        for k in sorted(set(have) | set(built)):
            if k not in built:
                print(f'  {name}: LOST key {k!r}')
                ok = False
            elif k not in have:
                print(f'  {name}: adds key {k!r}')
            elif have[k] != built[k]:
                print(f'  {name}: {k!r} {have[k]!r} -> {built[k]!r}')
                ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true',
                    help='compare against the committed pickles, write nothing')
    args = ap.parse_args()

    print(f'reading {CURVE_DIR}')
    names, lookup, props = collect()
    built = {'mist_filter_names.pkl': names,
             'phot_system_lookup.pkl': lookup,
             'filter_properties.pkl': props}

    if args.check:
        print('\nchecking against committed pickles:')
        ok = all([compare(n, b) for n, b in built.items()])
        print('\nOK -- regeneration reproduces the committed data'
              if ok else '\nMISMATCH -- do not regenerate until understood')
        return 0 if ok else 1

    for n, b in built.items():
        with open(os.path.join(DATA_DIR, n), 'wb') as f:
            pickle.dump(b, f)
        print(f'wrote {n}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
