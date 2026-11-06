#!/usr/bin/env python
"""
Pack the hot-star libraries beyond TLUSTY into the blocks `artpop.hotstars`
reads (2026-10-29, the physics-first BPASS step 2):

* **Götberg et al. (2023)** CMFGEN stripped-star grid (ApJ 959, 125; Zenodo
  10.5281/zenodo.7976200, ``S41_spectra_spectral_model_grid.tar.gz``): weak
  winds (1e-9 Msun/yr), Teff(tau=20) 30-150 kK, log g 4.0-6.0, surface X_H
  0.01 / 0.1 / 0.3 / 0.5 / 0.7, one composition (X_Fe 2.5e-4: [Fe/H] -0.71).
  One family per X_H, axes (log g, log T) at tau = 20.
* **PoWR** (Hamann & Gräfener 2004; Todt+ 2015; Hainich+ 2019; Pauli+ 2026),
  www.astro.physik.uni-potsdam.de/PoWR, fetched model by model (SED + model
  header): the OB-Vd3 grids (axes log g, log T*) and the WN grids at X_H 0 /
  0.2 / 0.4 / 0.6 (axes log R_t, log T*), one node per metallicity (sub-SMC,
  SMC, LMC). WC grids are fetched but not packed (no evolution input says WC).

    python tools/stage_hotstars.py gotberg2023 --src DIR_WITH_SED_FILES --params S41_..._parameters.txt
    python tools/stage_hotstars.py powr --src /jobs/out/powr_dl
    (--out defaults to <ARTPOP_HOME>/hotstars/blocks; both write into the same root)

What is written, per family and metallicity node:

* ``<family>_zh{+0.00}.npz``: ``ax2`` (log g, or log R_t for the PoWR WN
  families), ``logt`` and ``flux`` ``(n_ax2, n_t, n_lam)`` float32: the SURFACE
  F_lambda (erg/s/cm^2/A, ``surface_scale = 1``), flux-conservingly rebinned onto
  the shared log-lambda grid and rescaled to INT F dlam = sigma T^4 (shape kept;
  the factor is recorded and models off by more than ``--max-flux-error`` are
  left holes), NaN where the grid has no model. Beyond a library's red end the
  flux continues as a Rayleigh-Jeans lambda^-4 tail (declared; Götberg stops at
  10 micron, past every band ALVISS renders).
* ``hotstars_wave_vac.npy`` and ``manifest.json``: per family its kind, X_H,
  nodes, axis name, source, citation and, for R_t families, the terminal
  velocity and clumping contrast of each node (the router needs them to turn a
  mass-loss rate into R_t).

T is the hydrostatic-core temperature in both (Götberg's tau = 20, PoWR's T*),
the one a stellar-evolution code without a wind reports; [Fe/H] of a node is
log10(X_Fe / 1.29e-3) (Asplund 2009, the MIST scale).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SIGMA_SB = 5.670374419e-5
L_SUN = 3.828e33
R_SUN = 6.957e10
PC = 3.0856775814913673e18
X_FE_SUN = 1.29e-3          # Asplund+ 2009 iron mass fraction (MIST's scale)
GOTBERG_X = {'H001': 0.01, 'H01': 0.1, 'H03': 0.3, 'H05': 0.5, 'H07': 0.7}
GOTBERG_X_FE = 2.5e-4       # parameter table, every model


def target_grid(resolution, lo=50.0, hi=3.2e5):
    """Bin edges and centres, log-uniform at lambda / dlambda = resolution."""
    n = int(np.ceil(np.log(hi / lo) * resolution))
    edges = lo * np.exp(np.arange(n + 1) / resolution)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def rebin(lam, f, edges):
    """Mean flux per target bin, conserving INT f dlam (trapezoid cumulative
    integral); zero blueward of the library, a lambda^-4 tail redward."""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lam))])
    c = np.interp(edges, lam, cum, left=0.0, right=cum[-1])
    out = np.diff(c) / np.diff(edges)
    red = edges[1:] > lam[-1]
    if red.any():
        ctr = np.sqrt(edges[:-1] * edges[1:])
        out[red] = f[-1] * (lam[-1] / ctr[red]) ** 4
    return out


def node_name(family, zh):
    return f'{family}_zh{zh:+.2f}.npz'


def write_block(out, family, zh, ax2, logt, cells, edges, max_err, conv):
    """cells: {(ax2, logt): (lam, f_surface)} -> block + manifest entry, on the
    mesh ``ax2`` x ``logt`` (the family's union over its nodes, so the nodes
    stack; NaN where this node has no model)."""
    ax2, logt = np.array(sorted(set(ax2)), float), np.array(sorted(set(logt)), float)
    flux = np.full((ax2.size, logt.size, edges.size - 1), np.nan, dtype=np.float32)
    ratios, rejected = {}, []
    for (a, t), (lam, f) in sorted(cells.items()):
        r = rebin(lam, f, edges)
        ratio = float(np.sum(r * np.diff(edges)) / (SIGMA_SB * 10.0 ** (4 * t)))
        ratios[f'{a:.3f}_{t:.4f}'] = round(ratio, 5)
        if not np.isfinite(ratio) or abs(ratio - 1.0) > max_err:
            rejected.append(f'{a:.3f}_{t:.4f}')
            continue
        flux[np.searchsorted(ax2, a), np.searchsorted(logt, t)] = r / ratio
    name = node_name(family, zh)
    np.savez(os.path.join(out, name), ax2=ax2, logt=logt, flux=flux)
    v = np.array(list(ratios.values()))
    return name, dict(zh=zh, n_models=len(ratios) - len(rejected), n_rejected=len(rejected),
                      rejected=rejected, n_holes=int(ax2.size * logt.size - len(ratios) + len(rejected)),
                      int_f_over_sigma_t4=[float(v.min()), float(np.median(v)), float(v.max())],
                      conversion=conv, ratios=ratios)


def load_manifest(out):
    p = os.path.join(out, 'manifest.json')
    if os.path.isfile(p):
        with open(p) as fh:
            return json.load(fh)
    return dict(families={}, blocks={})


def save_manifest(out, man, wave, resolution):
    man['resolution'] = resolution
    man['flux'] = ('surface F_lambda, erg/s/cm^2/A, vacuum wavelengths, rescaled to sigma T^4 '
                   '(T the hydrostatic-core temperature: tau = 20 / T*)')
    man['x_fe_sun'] = X_FE_SUN
    np.save(os.path.join(out, 'hotstars_wave_vac.npy'), wave)
    with open(os.path.join(out, 'manifest.json'), 'w') as fh:
        json.dump(man, fh, indent=1)


# --------------------------------------------------------------------- Götberg
def stage_gotberg2023(a, edges, wave, out):
    rows = {}
    with open(a.params) as fh:
        lines = [ln for ln in fh if not ln.startswith('#')]
    head = lines[0].split()
    for ln in lines[2:]:
        p = ln.split()
        if len(p) != len(head):
            continue
        rows[p[0]] = dict(zip(head, p))
    man = load_manifest(out)
    zh = round(float(np.log10(GOTBERG_X_FE / X_FE_SUN)), 2)
    pat = re.compile(r'SED_logg([0-9.]+)_T([0-9.]+)_(H\d+)\.txt$')
    fams = {}
    for path in sorted(glob.glob(os.path.join(a.src, 'SED_*.txt'))):
        m = pat.search(os.path.basename(path))
        g, teff, hx = float(m.group(1)), float(m.group(2)), m.group(3)
        key = f'logg{g:.1f}_Teff{teff:.1f}_{hx}'
        par = rows.get(key)
        if par is None:
            raise KeyError(f'{os.path.basename(path)}: no parameter row {key}')
        d = np.loadtxt(path, comments='#', usecols=(0, 1))
        lam, f = d[:, 0], d[:, 1]
        # the SED's distance from its own L_bol: INT F = L / (4 pi d^2)
        lbol = float(par['Lbol']) * L_SUN
        d2 = lbol / (4 * np.pi * np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(lam)))
        r20 = float(par['Reff(tau=20)']) * R_SUN
        fsurf = f * d2 / r20 ** 2
        fam = f'gotberg23_x{GOTBERG_X[hx]:.2f}'
        fams.setdefault(fam, {})[(g, round(float(np.log10(teff)), 4))] = (
            lam, fsurf, float(np.sqrt(d2) / PC))
    for fam, cells in sorted(fams.items()):
        dist = sorted({round(v[2], 3) for v in cells.values()})
        cc = {k: (v[0], v[1]) for k, v in cells.items()}
        name, info = write_block(out, fam, zh, [k[0] for k in cc], [k[1] for k in cc], cc,
                                 edges, a.max_flux_error,
                                 f'F_lambda at the distance implied by L_bol ({dist[0]}-{dist[-1]} pc) '
                                 f'x d^2 / R(tau=20)^2')
        man['blocks'][name] = info
        man['families'][fam] = dict(
            library='Götberg+2023', kind='logg', ax2='log g (tau = 20)', x_h=float(fam.split('_x')[1]),
            nodes=sorted(set(man['families'].get(fam, {}).get('nodes', [])) | {zh}),
            source='Götberg, Drout, Ji et al. 2023, ApJ 959, 125; zenodo 10.5281/zenodo.7976200 '
                   '(S41 CMFGEN grid, Mdot 1e-9 Msun/yr, vinf 2500 km/s)',
            wind='weak (Mdot 1e-9 Msun/yr)')
        print(f"{name}: {info['n_models']} models ({info['n_rejected']} rejected), "
              f"{info['n_holes']} holes, INT F/sigma T^4 {info['int_f_over_sigma_t4']}")
    save_manifest(out, man, wave, a.resolution)


# ------------------------------------------------------------------------ PoWR
def parse_modinfo(path):
    with open(path, errors='replace') as fh:
        t = fh.read()

    def num(pat):
        m = re.search(pat, t)
        return float(m.group(1)) if m else np.nan
    desc = re.search(r'MODEL START[^\n]*\n\*\s*(.*)', t)
    desc = desc.group(1) if desc else ''
    fe = re.search(r'Fe([0-9.]+[Ee]-?\d+)', desc)
    xh = re.search(r'\bH(\d+)', desc)
    return dict(teff=num(r'TEFF\s*=\s*([0-9.]+)'), log_l=num(r'LOG L\s*=\s*([-0-9.]+)'),
                rstar=num(r'RSTAR\s*=\s*([0-9.]+)'), log_mdot=num(r'M-DOT\s*=\s*([-0-9.]+)'),
                log_rt=num(r'RTRANS\s*=\s*[0-9.Ee+-]+\s*=\s*([-0-9.]+)\s*DEX'),
                vinf=num(r'VFINAL\s*=\s*([0-9.]+)'), dens=num(r'DENSCON\(1\)\s*=\s*([0-9.]+)'),
                logg=num(r'LOG G_GRAV\s*=\s*([-0-9.]+)'),
                x_fe=float(fe.group(1)) if fe else np.nan,
                x_h=float(xh.group(1)) / 100.0 if xh else np.nan, desc=desc.strip())


def read_powr_sed(path, rstar_rsun):
    """log lambda [A], log F_lambda at 10 pc -> (lam, surface F_lambda)."""
    d = np.loadtxt(path)
    lam = 10.0 ** d[:, 0]
    f = np.where(d[:, 1] > -99, 10.0 ** d[:, 1], 0.0)
    return lam, f * (10.0 * PC) ** 2 / (rstar_rsun * R_SUN) ** 2


def powr_family(grid):
    """grid code -> (family, kind, X_H, metallicity label); (None, ...) for WC."""
    g = grid.lower()
    z = 'subsmc' if g.startswith(('007', 'subsmc')) else g.split('-')[0]
    if '-ob-' in g:
        return 'powr_ob', 'logg', None, z
    if '-wne' in g:
        x = 0.0
    else:
        m = re.search(r'-wnl-h(\d+)', g)
        if not m:
            return None, None, None, z
        x = int(m.group(1)) / 100.0
    return f'powr_wn_x{x:.1f}', 'logrt', x, z


def stage_powr(a, edges, wave, out):
    man = load_manifest(out)
    # a PoWR restage replaces every PoWR family (nodes and meshes may change)
    for fam in [f for f in man['families'] if f.startswith('powr')]:
        del man['families'][fam]
    for b in [b for b in man['blocks'] if b.startswith('powr')]:
        del man['blocks'][b]
        if os.path.isfile(os.path.join(out, b)):
            os.remove(os.path.join(out, b))
    groups = {}
    for gdir in sorted(glob.glob(os.path.join(a.src, '*'))):
        if not os.path.isdir(gdir):
            continue
        grid = os.path.basename(gdir)
        fam, kind, x, zlab = powr_family(grid)
        if fam is None:
            print(f'{grid}: not packed (WC or unknown)')
            continue
        # the fetcher's verify.tsv (header vs grid title, SED L vs header L, SED
        # coverage; 2026-10-29: the server mixes files up under concurrent
        # requests, and a few models disagree with their own grid entry)
        vp = os.path.join(gdir, 'verify.tsv')
        if not os.path.isfile(vp):
            print(f'{grid}: no verify.tsv, not packed (run the verified fetch first)')
            continue
        good = {ln.split('\t')[0] for ln in open(vp) if ln.rstrip('\n').split('\t')[1] == 'ok'}
        for info_p in sorted(glob.glob(os.path.join(gdir, '*.modinfo.txt'))):
            sed_p = info_p.replace('.modinfo.txt', '.sed.txt')
            if os.path.basename(info_p)[:-len('.modinfo.txt')] not in good:
                continue
            p = parse_modinfo(info_p)
            if not np.isfinite(p['teff']) or not np.isfinite(p['rstar']):
                continue
            ax2 = p['logg'] if kind == 'logg' else p['log_rt']
            lam, f = read_powr_sed(sed_p, p['rstar'])
            g = groups.setdefault((fam, grid), dict(kind=kind, x=x, zlab=zlab, cells={}, par=[]))
            g['cells'][(round(ax2, 3), round(float(np.log10(p['teff'])), 4))] = (lam, f)
            g['par'].append(p)
    union = {}
    for (fam, grid), g in groups.items():
        u = union.setdefault(fam, (set(), set()))
        u[0].update(k[0] for k in g['cells'])
        u[1].update(k[1] for k in g['cells'])
    for (fam, grid), g in sorted(groups.items()):
        x_fe = np.nanmedian([p['x_fe'] for p in g['par']])
        zh = round(float(np.log10(x_fe / X_FE_SUN)), 2)
        cc = g['cells']
        name, info = write_block(out, fam, zh, union[fam][0], union[fam][1], cc, edges,
                                 a.max_flux_error, 'log F_lambda at 10 pc x (10 pc / R*)^2')
        info['grid'] = grid
        vinf = sorted({p['vinf'] for p in g['par']})
        dens = sorted({p['dens'] for p in g['par']})
        info['vinf_km_s'], info['clumping_d'] = vinf, dens
        man['blocks'][name] = info
        f = man['families'].setdefault(fam, dict(
            library='PoWR', kind=g['kind'], ax2='log g' if g['kind'] == 'logg' else 'log R_t [R_sun]',
            x_h=g['x'], nodes=[], node_wind={},
            source='PoWR grids, www.astro.physik.uni-potsdam.de/PoWR (Hamann & Gräfener 2004; '
                   'Todt+ 2015; Hainich+ 2019; Pauli+ 2026)'))
        f['nodes'] = sorted(set(f['nodes']) | {zh})
        f.setdefault('node_wind', {})[f'{zh:+.2f}'] = dict(grid=grid, vinf_km_s=float(np.median(vinf)),
                                                           clumping_d=float(np.median(dens)))
        print(f"{name} <- {grid}: {info['n_models']} models ({info['n_rejected']} rejected), "
              f"{info['n_holes']} holes, vinf {vinf}, D {dens}, INT F/sigma T^4 {info['int_f_over_sigma_t4']}")
    save_manifest(out, man, wave, a.resolution)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    ap.add_argument('what', choices=['gotberg2023', 'powr'])
    ap.add_argument('--src', required=True)
    ap.add_argument('--params', help='Götberg 2023 S41_spectral_model_grid_parameters.txt')
    ap.add_argument('--out', default=None)
    ap.add_argument('--resolution', type=float, default=2000.0)
    ap.add_argument('--max-flux-error', type=float, default=0.05,
                    help='reject a model with |INT F / sigma T^4 - 1| above this')
    a = ap.parse_args()
    home = os.environ.get('ARTPOP_HOME') or os.path.expanduser('~/.artpop')
    out = a.out or os.path.join(home, 'hotstars', 'blocks')
    os.makedirs(out, exist_ok=True)
    edges, wave = target_grid(a.resolution)
    if a.what == 'gotberg2023':
        stage_gotberg2023(a, edges, wave, out)
    else:
        stage_powr(a, edges, wave, out)


if __name__ == '__main__':
    main()
