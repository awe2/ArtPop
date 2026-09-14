#!/usr/bin/env python
"""
Build (or verify) the cached K-correction grids.

Modelled on ``build_filter_data.py``, and for the same reason: the grid is a
derived artifact, so there has to be one command that regenerates it and one
that says whether the committed/cached copy still matches what the current code
and the current libraries produce.

``--check`` is the one that matters. It rebuilds every cached grid in memory and
compares it to the file, writing nothing. A mismatch means either the spectral
libraries on disk changed (run ``verify_c3k_staging.py``) or the integrator did
-- and either way a rendered population would silently change photometry.

The grids are small (about 13 MB for LSST) and fast (about 2 s per photometric
system), which is why nothing is committed: `~artpop.kcorrect.KCorrectionGrid`
builds on demand and caches, and this script exists to make that cache
inspectable rather than to feed it.

Usage::

    python tools/build_kcorrection_grid.py                 # LSST + Roman
    python tools/build_kcorrection_grid.py --check
    python tools/build_kcorrection_grid.py --systems LSST --z-max 0.5
    python tools/build_kcorrection_grid.py --a-v-host 0.3 --a-v-mw 0.1

Environment: ``ALVISS_SPECTRA_PATH`` for the libraries,
``ALVISS_KCORR_CACHE`` for the output directory.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from artpop.kcorrect import (KCorrectionGrid, C3KLibrary, TremblayWDLibrary,
                             kcorr_cache_dir, spectra_path, DEFAULT_Z_GRID)

DEFAULT_SYSTEMS = ['LSST', 'Roman']


def _grid_kw(args):
    z_grid = DEFAULT_Z_GRID
    if args.z_max is not None:
        z_grid = np.round(np.linspace(0.0, args.z_max, args.n_z), 6)
    return dict(z_grid=z_grid, a_v_host=args.a_v_host, a_v_mw=args.a_v_mw,
                extinction_law=args.law, r_v=args.r_v,
                resolution=args.resolution, a_over_fe=args.a_over_fe)


def build(systems, cache_dir, check=False, **kw):
    ok = True
    for system in systems:
        key = KCorrectionGrid.cache_key(system, **kw)
        path = os.path.join(cache_dir, key)
        t0 = time.time()
        fresh = KCorrectionGrid.build(system, **kw)
        dt = time.time() - t0

        if check:
            if not os.path.isfile(path):
                print(f'MISSING  {key}')
                ok = False
                continue
            cached = KCorrectionGrid.load(path)
            diffs = [name for name in KCorrectionGrid._ARRAYS
                     if not np.array_equal(np.asarray(getattr(cached, name)),
                                           np.asarray(getattr(fresh, name)))]
            if cached.bands != fresh.bands:
                diffs.append('bands')
            if cached.meta != fresh.meta:
                diffs.append('meta')
            if diffs:
                print(f'MISMATCH {key}\n         differs in: {diffs}')
                ok = False
            else:
                print(f'OK       {key}  ({dt:.1f} s to rebuild)')
            continue

        fresh.save(path)
        size = os.path.getsize(path) / 1e6
        print(f'wrote    {key}\n         {len(fresh.bands)} bands x '
              f'{len(fresh.z_grid)} z x {len(fresh.feh_grid)} [Fe/H] '
              f'x {fresh.c3k.shape[1]}x{fresh.c3k.shape[2]} nodes, '
              f'{size:.1f} MB, {dt:.1f} s')
        # a fallback that carries no flux is housekeeping; one that carries a
        # lot means a library is missing, so say which backends are live
        print(f'         backends: C3K + '
              + ('Tremblay + ' if fresh.has_wd else '')
              + 'blackbody')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--check', action='store_true',
                    help='verify the cache against a fresh build; write nothing')
    ap.add_argument('--systems', nargs='+', default=DEFAULT_SYSTEMS)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--z-max', type=float, default=None,
                    help='override the default redshift axis')
    ap.add_argument('--n-z', type=int, default=36)
    ap.add_argument('--a-v-host', type=float, default=0.0,
                    help='rest-frame (host galaxy) V-band extinction')
    ap.add_argument('--a-v-mw', type=float, default=0.0,
                    help='observer-frame (Milky Way foreground) extinction')
    ap.add_argument('--law', default='F99', choices=('F99', 'CCM89', 'grey'))
    ap.add_argument('--r-v', type=float, default=3.1)
    ap.add_argument('--resolution', default='c3k_hr',
                    choices=('c3k_hr', 'c3k_lr'))
    ap.add_argument('--a-over-fe', type=float, default=0.0)
    args = ap.parse_args(argv)

    cache_dir = args.cache_dir or kcorr_cache_dir()
    print(f'spectral libraries: {spectra_path()}')
    print(f'cache directory   : {cache_dir}')
    if not C3KLibrary(resolution=args.resolution).available:
        print('\nC3K is not staged. See data/spectral_libraries/PROVENANCE.md '
              'and run verify_c3k_staging.py.')
        return 1
    if not TremblayWDLibrary().available:
        print('NOTE: the Tremblay white-dwarf library is not staged; the '
              'log g >= 6.5 branch will use the blackbody fallback.')
    print()

    ok = build(args.systems, cache_dir, check=args.check, **_grid_kw(args))
    if args.check:
        print('\n' + ('OK -- the cache reproduces a fresh build' if ok else
                      'MISMATCH -- do not trust the cache until understood'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
