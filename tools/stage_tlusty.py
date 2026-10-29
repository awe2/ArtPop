#!/usr/bin/env python
"""
Pack unpacked TLUSTY 2025 O/B-star spectra (Hubeny, Bohlin, Gordon & Fitzpatrick
2025, AJ 169, 178; https://stsci.box.com/v/tlustyOB2025) into the blocks
`artpop.kcorrect.TlustyLibrary` reads.

    python tools/stage_tlusty.py --src ~/.artpop/tlusty2025 \\
        --out ~/.artpop/tlusty2025/blocks [--resolution 2000]

``--src`` holds the unpacked ``OSTAR_z*v*/`` and ``BSTAR_z*v*/`` directories.
Each model is ``z<Z>t<Teff>g<100 log g>v<vturb>.spec.gz``: two columns, vacuum
wavelength (Angstrom, 200 A - 32 micron) and the EDDINGTON flux H_lambda =
F_lambda / 4 pi (erg/s/cm^2/A) -- measured: INT H dlam = sigma Teff^4 / 4 pi to
0.2 % on z001t40000g400v10 (2026-10-16). The ``.cont.gz`` files are the
line-free continuum and are not used.

What is written, per metallicity node ([Z/H] = log10 Z/Z_sun):

* ``tlusty_zh{+0.00}.npz``: ``logg``, ``logt`` (the union mesh) and ``flux``
  ``(n_g, n_t, n_lam)`` float32, the SURFACE flux F_lambda = 4 pi H_lambda
  (erg/s/cm^2/A, so ``surface_scale = 1``), flux-conservingly rebinned onto one
  log-lambda grid; NaN where the grid has no model (the Eddington limit at low
  log g). Teff <= 30 kK from BSTAR, > 30 kK from OSTAR (at 30 kK OSTAR fills a
  log g BSTAR lacks); OSTAR's 27.5 kK column is dropped so the two meshes do not
  interleave.
* ``tlusty_wave_vac.npy`` and ``manifest.json`` (the source directories, the
  microturbulence per library, INT F / sigma T^4 per model, the node map).

BSTAR's z0032 and OSTAR's z0033 are both Z/Z_sun = 1/30 (one node, -1.48).
"""
import argparse
import glob
import gzip
import json
import os
import re

import numpy as np

SIGMA_SB = 5.670374419e-5
Z_LABELS = {'001': 0.01, '002': 0.02, '0032': 1 / 30, '0033': 1 / 30, '010': 0.1,
            '020': 0.2, '050': 0.5, '100': 1.0, '200': 2.0}
NAME = re.compile(r'z(\d+)t(\d+)g(\d+)v(\d+)\.spec\.gz$')


def target_grid(resolution, lo=200.0, hi=3.2e5):
    """Bin edges and centres, log-uniform at lambda / dlambda = resolution."""
    n = int(np.ceil(np.log(hi / lo) * resolution))
    edges = lo * np.exp(np.arange(n + 1) / resolution)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def rebin(lam, f, edges):
    """Mean flux per bin, conserving INT f dlam (trapezoid cumulative integral)."""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lam))])
    c = np.interp(edges, lam, cum)
    return np.diff(c) / np.diff(edges)


def read(path):
    """(lam, H_lam) with the rows whose flux is the literal ``NaN`` dropped (some
    files carry ~900 of them, all at 31.7-32 micron, 2026-10-16)."""
    with gzip.open(path, 'rb') as fh:
        d = np.array(fh.read().replace(b'NaN', b'nan').split(), dtype=float).reshape(-1, 2)
    ok = np.isfinite(d[:, 1])
    return d[ok, 0], d[ok, 1]


def pack(job):
    """One metallicity node -> (name, manifest entry); writes the block.

    Flux conservation is the quality test: a model with |INT F / sigma T^4 - 1|
    above ``max_err`` is left a hole (the star falls back to C3K / blackbody);
    the others are rescaled to sigma T^4 exactly (shape kept), so the 'teff'
    and 'spectrum' bolometric conventions coincide for TLUSTY."""
    zh, cell, loggs, teffs, edges, out, max_err = job
    logg, logt = np.array(loggs, float), np.log10(np.array(teffs, float))
    flux = np.full((logg.size, logt.size, edges.size - 1), np.nan, dtype=np.float32)
    cons, libs, rejected = [], {}, []
    for (teff, g), (lib, p) in sorted(cell.items()):
        lam, h = read(p)
        f = 4.0 * np.pi * rebin(lam, h, edges)
        ratio = float(np.sum(f * np.diff(edges)) / (SIGMA_SB * teff ** 4))
        cons.append(ratio)
        libs[f'{teff}_{g:.2f}'] = [lib, round(ratio, 5)]
        if abs(ratio - 1.0) > max_err:
            rejected.append(f'{teff}_{g:.2f}')
            continue
        flux[loggs.index(g), teffs.index(teff)] = f / ratio
    name = f'tlusty_zh{zh:+.2f}.npz'
    np.savez(os.path.join(out, name), logg=logg, logt=logt, flux=flux)
    cons = np.array(cons)
    return name, dict(zh=zh, n_models=len(cons) - len(rejected), n_rejected=len(rejected),
                      rejected=rejected,
                      n_missing=int(logg.size * logt.size - len(cons) + len(rejected)),
                      int_f_over_sigma_t4=[float(cons.min()), float(np.median(cons)),
                                           float(cons.max())], models=libs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    ap.add_argument('--src', default=os.path.expanduser('~/.artpop/tlusty2025'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--resolution', type=float, default=2000.0)
    ap.add_argument('--vturb-o', default='10', help='OSTAR microturbulence (km/s)')
    ap.add_argument('--vturb-b', default='2', help='BSTAR microturbulence (km/s)')
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--max-flux-error', type=float, default=0.05,
                    help='reject a model with |INT F / sigma T^4 - 1| above this')
    a = ap.parse_args()
    out = a.out or os.path.join(a.src, 'blocks')
    os.makedirs(out, exist_ok=True)
    edges, wave = target_grid(a.resolution)

    # (zh node) -> {(teff, logg): (lib, path)}
    models = {}
    dirs = {}
    for lib, vt in (('BSTAR', a.vturb_b), ('OSTAR', a.vturb_o)):
        for d in sorted(glob.glob(os.path.join(a.src, f'{lib}_z*v{vt}'))):
            for p in sorted(glob.glob(os.path.join(d, '*.spec.gz'))):
                m = NAME.search(os.path.basename(p))
                zl, teff, g100 = m.group(1), int(m.group(2)), int(m.group(3))
                zh = round(float(np.log10(Z_LABELS[zl])), 2)
                if lib == 'OSTAR' and teff < 30000:
                    continue
                if lib == 'BSTAR' and teff > 30000:
                    continue
                key = (teff, g100 / 100.0)
                cell = models.setdefault(zh, {})
                # at 30 kK BSTAR is preferred, OSTAR fills its gaps
                if key in cell and cell[key][0] == 'BSTAR':
                    continue
                cell[key] = (lib, p)
                dirs.setdefault(zh, set()).add(os.path.basename(d))

    teffs = sorted({k[0] for c in models.values() for k in c})
    loggs = sorted({k[1] for c in models.values() for k in c})
    logt = np.log10(np.array(teffs, float))
    logg = np.array(loggs, float)
    man = dict(max_flux_error=a.max_flux_error, source='TLUSTY 2025 O/B grids (Hubeny, Bohlin, Gordon & Fitzpatrick 2025, '
                      'AJ 169, 178), https://stsci.box.com/v/tlustyOB2025',
               flux='surface F_lambda = 4 pi H_lambda, erg/s/cm^2/A, vacuum wavelengths',
               resolution=a.resolution, vturb=dict(OSTAR=a.vturb_o, BSTAR=a.vturb_b),
               teff=teffs, logg=loggs, blocks={})
    np.save(os.path.join(out, 'tlusty_wave_vac.npy'), wave)
    from multiprocessing import Pool
    jobs = [(zh, models[zh], loggs, teffs, edges, out, a.max_flux_error) for zh in sorted(models)]
    with Pool(min(a.procs, len(jobs))) as pool:
        for (name, info), zh in zip(pool.map(pack, jobs), sorted(models)):
            info['dirs'] = sorted(dirs[zh])
            man['blocks'][name] = info
            c = info['int_f_over_sigma_t4']
            print(f"{name}: {info['n_models']} models ({info['n_rejected']} rejected), "
                  f"{info['n_missing']} holes, "
                  f"INT F / sigma T^4 {c[0]:.4f}-{c[2]:.4f} (median {c[1]:.4f}); from {info['dirs']}")
    with open(os.path.join(out, 'manifest.json'), 'w') as fh:
        json.dump(man, fh, indent=1)


if __name__ == '__main__':
    main()
