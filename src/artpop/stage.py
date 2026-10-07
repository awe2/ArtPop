"""
Stage everything ArtPop's photometry needs on a new host, under one root.

    python -m artpop.stage --dp2-passbands dp2_standard_passbands.ecsv   # all steps
    python -m artpop.stage --check                                       # report only
    python -m artpop.stage marcs tables                                  # some steps

The root is ``$ARTPOP_HOME`` (default ``~/.artpop``)::

    <root>/mist/       MIST isochrones (ArtPop's own fetcher; ``$MIST_PATH`` overrides)
    <root>/spectra/    c3k/c3k_hr, tremblay_wd, marcs  (``$ARTPOP_SPECTRA_PATH``)
    <root>/passbands/  LSST = DP2 standard_passband      (``$ARTPOP_PHOTOMETRY_CURVES``)
    <root>/tables/     photometry tables, built here     (``$ARTPOP_TABLES``)

The older ``$ALVISS_SPECTRA_PATH`` / ``$ALVISS_KCORR_CACHE`` still override.
Staging writes exactly where `artpop.kcorrect` reads.

Steps, in order:

- ``mist``      -- MIST v2.5 isochrones for each system (mist.science).
- ``c3k``       -- C3K ``c3k_hr``, FSPS commit `C3K_COMMIT` (GitHub raw).
- ``tremblay``  -- Tremblay et al. (2011) DA white dwarfs (Warwick).
- ``marcs``     -- MARCS spherical giants (Gustafsson et al. 2008): streams the
  alpha-poor and standard flux archives from marcs.astro.uu.se (~8.3 GB),
  keeps the subset (`MARCS_PATTERN`, ~0.55 GB) and packs it into one block per
  [Fe/H] (`marcs_blocks`).
- ``passbands`` -- **DP2's standard passbands, from a file you supply**
  (``--dp2-passbands``). They are a Rubin data product retrieved on the RSP and
  are not distributed with ArtPop. Decision (user, 2026-10-07): staging FAILS
  without them, and LSST photometry refuses to run without them
  (`artpop.kcorrect.REQUIRED_PASSBANDS`); clearance to include them in the
  repository is expected at a future date.
- ``tables``    -- builds the photometry tables (dust-free and dust-axis) for
  each system, and the parsed-MIST binary cache for the grid ArtPop reads.

Every staged file is checked against ``data/staging_manifest.json``, which pins
the sha256 of each one (``--write-manifest`` regenerates it from a verified
host; the MARCS blocks are byte-reproducible, numpy writes ``.npz``
deterministically). ``--check`` downloads nothing and exits non-zero if
anything is missing or does not match.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request

import numpy as np

from . import ARTPOP_HOME, MIST_PATH, data_dir

MANIFEST = os.path.join(data_dir, 'staging_manifest.json')
STEPS = ('mist', 'c3k', 'tremblay', 'marcs', 'passbands', 'tables')

C3K_COMMIT = 'bd187a0d07da'
C3K_RAW = f'https://raw.githubusercontent.com/cconroy20/fsps/{C3K_COMMIT}/SPECTRA/C3K'
C3K_FEH = ['-2.50', '-2.25', '-2.00', '-1.75', '-1.50', '-1.25', '-1.00',
           '-0.75', '-0.50', '-0.25', '+0.00', '+0.25', '+0.50']
TREMBLAY_URL = ('https://warwick.ac.uk/fac/sci/physics/research/astro/people/'
                'tremblay/modelgrids')
TREMBLAY_FILES = ('grid_ir.tar', 'readme.txt')
MARCS_HOST = 'https://marcs.astro.uu.se'
MARCS_WAVE = 'flx_wavelengths.vac.gz'
# (archive, composition, keep this [Fe/H]?): alpha-poor below solar and the
# standard mixture at and above it, so [a/Fe] = 0 everywhere, as C3K and MIST
MARCS_ARCHIVES = (('data/marcs_ap_flx.tar', 'ap', lambda feh: feh < 0),
                  ('data/marcs_st_flx.tar', 'st', lambda feh: feh >= 0))
MARCS_PATTERN = re.compile(
    r's(\d+)_g([+-]\d\.\d)_m1\.0_t02_(ap|st)_z([+-]\d\.\d\d)_a([+-]\d\.\d\d)_.*\.flx\.gz$')
MARCS_TEFF_MAX = 5000
MARCS_FEH_RANGE = (-2.5, 0.5)


# --------------------------------------------------------------------------- #
# where things go (the same resolution `artpop.kcorrect` reads with)            #
# --------------------------------------------------------------------------- #
def layout(root=None):
    """The four directories, honouring the environment overrides."""
    root = root or ARTPOP_HOME
    env = os.environ.get
    return dict(
        root=root,
        mist=env('MIST_PATH') or (MIST_PATH if root == ARTPOP_HOME else os.path.join(root, 'mist')),
        spectra=env('ARTPOP_SPECTRA_PATH') or env('ALVISS_SPECTRA_PATH') or os.path.join(root, 'spectra'),
        passbands=env('ARTPOP_PHOTOMETRY_CURVES') or os.path.join(root, 'passbands'),
        tables=env('ARTPOP_TABLES') or env('ALVISS_KCORR_CACHE') or os.path.join(root, 'tables'),
    )


def _export(dirs):
    """Point this process's `artpop.kcorrect` at the staged layout."""
    os.environ.setdefault('ARTPOP_SPECTRA_PATH', dirs['spectra'])
    os.environ.setdefault('ARTPOP_PHOTOMETRY_CURVES', dirs['passbands'])
    os.environ.setdefault('ARTPOP_TABLES', dirs['tables'])


# --------------------------------------------------------------------------- #
# small helpers                                                                #
# --------------------------------------------------------------------------- #
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    if os.path.isfile(MANIFEST):
        with open(MANIFEST) as fh:
            return json.load(fh)
    return {}


def download(url, dest):
    """Fetch ``url`` to ``dest`` via ``dest.part``: absent or complete, never half."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=300) as r, open(part, 'wb') as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    os.replace(part, dest)
    print(f'    {os.path.basename(dest)}: {os.path.getsize(dest) / 1e6:.1f} MB '
          f'in {time.time() - t0:.0f} s')


class Report:
    """One line per step: OK, MISSING, MISMATCH, UNPINNED or FAIL."""

    def __init__(self):
        self.rows = []

    def add(self, step, status, detail):
        self.rows.append((step, status, detail))
        print(f'  {status:9s} {step:10s} {detail}')

    @property
    def failed(self):
        return any(s in ('MISSING', 'MISMATCH', 'FAIL') for _, s, _ in self.rows)


def _check_files(base, pins, names):
    """``(missing, mismatched, unpinned)`` for ``names`` under ``base``."""
    missing, bad, unpinned = [], [], []
    for n in names:
        p = os.path.join(base, n)
        if not os.path.isfile(p):
            missing.append(n)
        elif n not in pins:
            unpinned.append(n)
        elif sha256(p) != pins[n]:
            bad.append(n)
    return missing, bad, unpinned


def _report_files(rep, step, base, pins, names):
    missing, bad, unpinned = _check_files(base, pins, names)
    if missing:
        rep.add(step, 'MISSING', f'{len(missing)} of {len(names)} files under {base} '
                                 f'(e.g. {missing[0]})')
    elif bad:
        rep.add(step, 'MISMATCH', f'{len(bad)} files differ from the pinned sha256 '
                                  f'(e.g. {bad[0]}) under {base}')
    elif unpinned:
        rep.add(step, 'UNPINNED', f'{len(names)} files present, {len(unpinned)} without a '
                                  'pinned sha256 (run --write-manifest on a verified host)')
    else:
        rep.add(step, 'OK', f'{len(names)} files, sha256 as pinned, under {base}')
    return not (missing or bad)


# --------------------------------------------------------------------------- #
# steps                                                                        #
# --------------------------------------------------------------------------- #
def mist_files(system, version='2.5', v_over_vcrit=0.4, a_over_fe=0.0, mist_path=None):
    """The MIST files ArtPop reads for one system: every [Fe/H] grid point.
    Paths only -- unlike `mist_iso_path`, this never fetches."""
    from .stars.isochrones import (MISTIsochrone, phot_str_helper, _feh_token,
                                   _afe_token)
    from .util import mist_grid_dir, mist_version_layout
    mist_path = mist_path or os.environ.get('MIST_PATH', MIST_PATH)
    key, lay = mist_version_layout(version)
    p = phot_str_helper[system.lower()]
    grid = mist_grid_dir(p, v_over_vcrit, mist_path, key)
    return [os.path.join(grid, lay['iso_file'].format(
        v=f'{float(v_over_vcrit):.1f}', p=p, feh=_feh_token(float(f), key),
        afe=_afe_token(a_over_fe))) for f in MISTIsochrone._feh_grid]


def step_mist(dirs, args, rep):
    from .util import fetch_mist_grid_if_needed
    for system in args.systems:
        files = mist_files(system, args.mist_version, mist_path=dirs['mist'])
        missing = [f for f in files if not os.path.isfile(f)]
        if missing and not args.check:
            print(f'  fetching MIST v{args.mist_version} {system} from mist.science ...')
            fetch_mist_grid_if_needed(system, version=args.mist_version,
                                      mist_path=dirs['mist'])
            missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            rep.add('mist', 'MISSING', f'{system}: {len(missing)} of {len(files)} [Fe/H] files '
                                       f'(e.g. {os.path.basename(missing[0])})')
        else:
            rep.add('mist', 'OK', f'{system}: {len(files)} [Fe/H] files under '
                                  f'{os.path.dirname(files[0])}')


C3K_AFE_LR = ['-0.2', '+0.0', '+0.2', '+0.4', '+0.6']


def c3k_files(with_lr=False):
    """``c3k_hr`` (production); ``with_lr`` adds the R = 100 ``c3k_lr`` grid
    that the resolution and [a/Fe] cross-checks read (redshift_tests)."""
    files = ['c3k_hr/logt.dat', 'c3k_hr/logg.dat', 'c3k_hr/readme.md',
             'c3k_hr/c3k_hr.lambda', 'c3k_hr/c3k_hr.res', 'c3k_hr/c3k_hr_zlegend.dat']
    files += [f'c3k_hr/c3k_hr_feh{z}_afe+0.0.spec.bin' for z in C3K_FEH]
    if with_lr:
        files += ['c3k_lr/c3k_lr.lambda', 'c3k_lr/c3k_lr.res', 'c3k_lr/c3k_lr_zlegend.dat']
        files += [f'c3k_lr/c3k_lr_feh{z}_afe{a}.spec.bin' for z in C3K_FEH for a in C3K_AFE_LR]
    return files


def step_c3k(dirs, args, rep, man):
    base = os.path.join(dirs['spectra'], 'c3k')
    names = c3k_files(args.with_lr)
    if not args.check:
        for n in names:
            if not os.path.isfile(os.path.join(base, n)):
                download(f'{C3K_RAW}/{n}', os.path.join(base, n))
    _report_files(rep, 'c3k', base, man.get('c3k', {}).get('sha256', {}), names)


def step_tremblay(dirs, args, rep, man):
    base = os.path.join(dirs['spectra'], 'tremblay_wd')
    if not args.check:
        for n in TREMBLAY_FILES:
            if not os.path.isfile(os.path.join(base, n)):
                download(f'{TREMBLAY_URL}/{n}', os.path.join(base, n))
    _report_files(rep, 'tremblay', base, man.get('tremblay', {}).get('sha256', {}),
                  list(TREMBLAY_FILES))


def _marcs_keep(name):
    m = MARCS_PATTERN.search(name)
    if m is None:
        return False
    teff, _, comp, feh, afe = m.groups()
    feh = float(feh)
    keep_feh = dict((c, k) for _, c, k in MARCS_ARCHIVES)[comp]
    return (int(teff) <= MARCS_TEFF_MAX and float(afe) == 0.0 and keep_feh(feh)
            and MARCS_FEH_RANGE[0] <= feh <= MARCS_FEH_RANGE[1])


def stream_marcs(marcs_root, host=MARCS_HOST, archives=MARCS_ARCHIVES, opener=None):
    """
    Stream each MARCS flux archive and keep only the selected members, so the
    ~8.3 GB of archives never touch the disk. ``opener(url)`` returns a
    file-like object (default: urllib); tests pass a local one.
    """
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=600))
    flx = os.path.join(marcs_root, 'flx')
    os.makedirs(flx, exist_ok=True)
    kept = 0
    for archive, comp, _ in archives:
        t0 = time.time()
        with opener(f'{host}/{archive}') as r, tarfile.open(fileobj=r, mode='r|*') as tar:
            for member in tar:
                name = os.path.basename(member.name)
                if not member.isfile() or not _marcs_keep(name):
                    continue
                dest = os.path.join(flx, name)
                with tar.extractfile(member) as src, open(dest + '.part', 'wb') as out:
                    shutil.copyfileobj(src, out)
                os.replace(dest + '.part', dest)
                kept += 1
        print(f'    {archive}: kept {kept} models so far ({time.time() - t0:.0f} s)')
    return kept


def marcs_blocks(marcs_root):
    """
    Pack ``<marcs_root>/flx/*.flx.gz`` into ``marcs_sph_feh{+0.00}.npz`` blocks
    (``logg``, ``logt``, ``flux`` (n_g, n_t, n_lam) float32 with NaN where MARCS
    has no model, ``missing``), ``marcs_wave_vac.npy`` and ``manifest.json``.
    MARCS ``.flx`` = surface flux, erg/cm^2/s/A, on ``flx_wavelengths.vac.gz``.
    """
    lam = np.loadtxt(os.path.join(marcs_root, MARCS_WAVE))
    models = {}
    for name in sorted(os.listdir(os.path.join(marcs_root, 'flx'))):
        if not _marcs_keep(name):
            continue
        teff, logg, comp, feh, _ = MARCS_PATTERN.search(name).groups()
        models.setdefault(feh, []).append((int(teff), float(logg),
                                           os.path.join(marcs_root, 'flx', name)))
    blocks = {}
    for feh, rows in sorted(models.items(), key=lambda kv: float(kv[0])):
        teffs = sorted({r[0] for r in rows})
        loggs = sorted({r[1] for r in rows})
        flux = np.full((len(loggs), len(teffs), lam.size), np.nan, dtype=np.float32)
        for teff, logg, path in rows:
            f = np.loadtxt(path)
            if f.size != lam.size:
                raise ValueError(f'{path}: {f.size} points, expected {lam.size}')
            flux[loggs.index(logg), teffs.index(teff)] = f
        name = f'marcs_sph_feh{float(feh):+.2f}.npz'
        path = os.path.join(marcs_root, name)
        np.savez(path, logg=np.array(loggs), logt=np.log10(teffs), flux=flux,
                 missing=~np.isfinite(flux[..., 0]))
        blocks[name] = dict(feh=float(feh), n_models=len(rows),
                            n_missing=int((~np.isfinite(flux[..., 0])).sum()),
                            sha256=sha256(path))
    np.save(os.path.join(marcs_root, 'marcs_wave_vac.npy'), lam)
    with open(os.path.join(marcs_root, 'manifest.json'), 'w') as fh:
        json.dump(dict(
            what='MARCS spherical giant fluxes for artpop.kcorrect.MARCSLibrary',
            reference='Gustafsson et al. 2008, A&A 486, 951',
            source=[f'{MARCS_HOST}/{a} ([Fe/H] {"< 0" if c == "ap" else ">= 0"})'
                    for a, c, _ in MARCS_ARCHIVES] + [f'{MARCS_HOST}/{MARCS_WAVE}'],
            selection=f's*_m1.0_t02, Teff <= {MARCS_TEFF_MAX} K, [Fe/H] '
                      f'{MARCS_FEH_RANGE[0]}..{MARCS_FEH_RANGE[1]}, [a/Fe] = 0',
            flux_units='surface flux, erg/cm^2/s/A (vacuum wavelengths, A)',
            staged_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            blocks=blocks), fh, indent=1)
    return blocks


def marcs_block_names(man):
    return sorted(man.get('marcs', {}).get('sha256', {}))


def step_marcs(dirs, args, rep, man):
    base = os.path.join(dirs['spectra'], 'marcs')
    pins = man.get('marcs', {}).get('sha256', {})
    names = marcs_block_names(man) or ['manifest.json']
    present = all(os.path.isfile(os.path.join(base, n)) for n in names)
    if not args.check and (not present or args.force):
        if not os.path.isfile(os.path.join(base, MARCS_WAVE)):
            download(f'{MARCS_HOST}/{MARCS_WAVE}', os.path.join(base, MARCS_WAVE))
        print(f'  streaming MARCS flux archives from {MARCS_HOST} (~8.3 GB, keeps ~0.55 GB) ...')
        stream_marcs(base)
        print('  packing MARCS blocks ...')
        marcs_blocks(base)
    if not pins:
        names = sorted(n for n in os.listdir(base) if n.endswith('.npz') or n.endswith('.npy')) \
            if os.path.isdir(base) else ['marcs_wave_vac.npy']
    _report_files(rep, 'marcs', base, pins, names)


def dp2_passbands_to_curves(ecsv, dest_system_dir):
    """Write DP2's ``standard_passband`` table (nm, fraction) as ArtPop curve
    CSVs (Angstrom, fraction), keep the original next to them, and record it."""
    from astropy.table import Table
    t = Table.read(ecsv)
    os.makedirs(dest_system_dir, exist_ok=True)
    w = np.asarray(t['wavelength'], float) * (10.0 if str(t['wavelength'].unit) == 'nm' else 1.0)
    for b in 'ugrizy':
        thr = np.asarray(t[f'throughput_{b}'], float)
        with open(os.path.join(dest_system_dir, f'LSST_{b}.csv'), 'w') as fh:
            fh.write('wave,trans\n')
            for x, y in zip(w.tolist(), thr.tolist()):
                fh.write(f'{x:.1f},{y!r}\n')
    shutil.copy(ecsv, os.path.join(dest_system_dir, 'dp2_standard_passbands.ecsv'))
    with open(os.path.join(dest_system_dir, 'PROVENANCE.md'), 'w') as fh:
        fh.write(
            'DP2 standard passbands (LSSTCam; Butler dataset standard_passband, FGCM run '
            'LSSTCam/calib/fgcmcal/DM-50154, LSST stack v30_0_11), retrieved on the RSP by '
            'AlvissRubinCompleteness/dp2_passbands.py following DP2 tutorial 204.4: full-system '
            'throughput (top of atmosphere to detector) including FGCM\'s standard atmosphere, '
            'the reference DP2 photometry is calibrated to. Converted from nm to Angstrom, '
            'fraction unchanged; original table dp2_standard_passbands.ecsv '
            f'(sha256 {sha256(ecsv)}). Staged by python -m artpop.stage. Not distributed '
            'with ArtPop (decision 2026-10-07; clearance expected later).\n')


def step_passbands(dirs, args, rep, man):
    dest = os.path.join(dirs['passbands'], 'LSST')
    staged = os.path.join(dest, 'dp2_standard_passbands.ecsv')
    pin = man.get('passbands', {}).get('LSST', {}).get('sha256')
    src = args.dp2_passbands
    if src and not args.check:
        got = sha256(src)
        if pin and got != pin and not args.accept_new_passbands:
            rep.add('passbands', 'FAIL', f'{src} has sha256 {got[:12]}..., not the pinned '
                                         f'{pin[:12]}... (a different DP2 release?); pass '
                                         '--accept-new-passbands to stage it anyway')
            return
        dp2_passbands_to_curves(src, dest)
    if not os.path.isfile(staged):
        rep.add('passbands', 'FAIL',
                f'LSST: DP2 standard_passband not staged under {dest}. Pass --dp2-passbands '
                '<dp2_standard_passbands.ecsv> (from AlvissRubinCompleteness/dp2_passbands.py '
                'on the RSP). Decision 2026-10-07: no fallback to other curves.')
        return
    got = sha256(staged)
    if pin and got != pin:
        rep.add('passbands', 'MISMATCH', f'LSST: {staged} sha256 {got[:12]}... is not the '
                                         f'pinned {pin[:12]}...')
    else:
        rep.add('passbands', 'OK' if pin else 'UNPINNED',
                f'LSST: DP2 standard_passband, 6 curves under {dest}')


def warm_mist_cache(system, version, v_over_vcrit=0.4, a_over_fe=0.0):
    """Build the parsed-MIST binary cache for the files ArtPop reads (E-1)."""
    from .stars._read_mist_models import read_isocmd
    n = 0
    for fn in mist_files(system, version, v_over_vcrit, a_over_fe):
        n += hasattr(read_isocmd(fn), 'block')
    return n


def step_tables(dirs, args, rep):
    from .kcorrect import (KCorrectionGrid, DEFAULT_AV_HOST_GRID, DEFAULT_AV_MW_GRID)
    for system in args.systems:
        variants = [dict()]
        if not args.no_dust_axes:
            variants.append(dict(a_v_host_grid=DEFAULT_AV_HOST_GRID, a_v_mw_grid=DEFAULT_AV_MW_GRID))
        for kw in variants:
            try:
                key = KCorrectionGrid.cache_key(system, **kw)
            except FileNotFoundError as exc:
                rep.add('tables', 'FAIL', f'{system}: {exc}')
                break
            path = os.path.join(dirs['tables'], key)
            if not os.path.isfile(path) and not args.check:
                t0 = time.time()
                KCorrectionGrid.cached(system, cache_dir=dirs['tables'], **kw)
                print(f'    built {key} in {time.time() - t0:.0f} s')
            label = 'dust axes' if kw else 'dust-free'
            if os.path.isfile(path):
                rep.add('tables', 'OK', f'{system} {label}: {key}')
            else:
                rep.add('tables', 'MISSING', f'{system} {label}: {key} under {dirs["tables"]}')
        if not args.check:
            try:
                n = warm_mist_cache(system, args.mist_version)
                note = '' if n else (' -- none written: the MIST directory is not writable; '
                                     'set ARTPOP_MIST_CACHE to a writable directory '
                                     '(the cache only saves parse time)')
                print(f'    MIST binary cache: {n} {system} files{note}')
            except Exception as exc:                     # noqa: BLE001 - a cache, never fatal
                print(f'    MIST binary cache not built ({exc})')


# --------------------------------------------------------------------------- #
# the manifest                                                                 #
# --------------------------------------------------------------------------- #
def write_manifest(dirs):
    """Pin the sha256 of every staged file from this (verified) host."""
    def pins(base, names):
        return {n: sha256(os.path.join(base, n)) for n in names}
    s = dirs['spectra']
    marcs = os.path.join(s, 'marcs')
    man = dict(
        what='sha256 of every file python -m artpop.stage stages; regenerate with '
             '--write-manifest on a host whose files were verified',
        written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        c3k=dict(source=C3K_RAW, sha256=pins(os.path.join(s, 'c3k'), [
            n for n in c3k_files(with_lr=True) if os.path.isfile(os.path.join(s, 'c3k', n))])),
        tremblay=dict(source=TREMBLAY_URL, sha256=pins(os.path.join(s, 'tremblay_wd'),
                                                       list(TREMBLAY_FILES))),
        marcs=dict(source=[f'{MARCS_HOST}/{a}' for a, _, _ in MARCS_ARCHIVES],
                   sha256=pins(marcs, sorted(n for n in os.listdir(marcs)
                                             if n.endswith('.npz') or n.endswith('.npy')))),
        passbands=dict(LSST=dict(
            source='DP2 standard_passband via AlvissRubinCompleteness/dp2_passbands.py (RSP)',
            sha256=sha256(os.path.join(dirs['passbands'], 'LSST', 'dp2_standard_passbands.ecsv')))),
    )
    with open(MANIFEST, 'w') as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(f'wrote {MANIFEST}')


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m artpop.stage',
                                 description=__doc__.split('\n\n')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('steps', nargs='*', choices=STEPS + ('all',),
                    help='steps to run (default: all)')
    ap.add_argument('--root', help='staging root (default $ARTPOP_HOME or ~/.artpop)')
    ap.add_argument('--check', action='store_true', help='report only; download nothing')
    ap.add_argument('--dp2-passbands', help='DP2 dp2_standard_passbands.ecsv to stage')
    ap.add_argument('--accept-new-passbands', action='store_true',
                    help='stage passbands whose sha256 differs from the pinned one')
    ap.add_argument('--systems', nargs='+', default=['LSST'])
    ap.add_argument('--mist-version', default='2.5')
    ap.add_argument('--no-dust-axes', action='store_true',
                    help='build only the dust-free tables (the dust-axis one is ~0.6 GB)')
    ap.add_argument('--force', action='store_true', help='re-stage MARCS even if present')
    ap.add_argument('--with-lr', action='store_true',
                    help='also stage the R = 100 c3k_lr grid (cross-checks only)')
    ap.add_argument('--write-manifest', action='store_true',
                    help='pin the staged files sha256 into data/staging_manifest.json')
    args = ap.parse_args(argv)
    steps = STEPS if (not args.steps or 'all' in args.steps) else tuple(
        s for s in STEPS if s in args.steps)
    dirs = layout(args.root)
    _export(dirs)
    man = load_manifest()
    print('ArtPop staging' + (' (check only)' if args.check else ''))
    for k in ('root', 'mist', 'spectra', 'passbands', 'tables'):
        print(f'  {k:10s} {dirs[k]}')
    rep = Report()
    for step in steps:
        if step == 'mist':
            step_mist(dirs, args, rep)
        elif step == 'c3k':
            step_c3k(dirs, args, rep, man)
        elif step == 'tremblay':
            step_tremblay(dirs, args, rep, man)
        elif step == 'marcs':
            step_marcs(dirs, args, rep, man)
        elif step == 'passbands':
            step_passbands(dirs, args, rep, man)
        elif step == 'tables':
            step_tables(dirs, args, rep)
    if args.write_manifest:
        write_manifest(dirs)
    print('FAILED' if rep.failed else 'OK')
    return 1 if rep.failed else 0


if __name__ == '__main__':
    sys.exit(main())
