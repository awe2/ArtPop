"""
Hot, stripped and Wolf-Rayet stars beyond TLUSTY (2026-10-29; ALVISS's
physics-first BPASS work, step 2).

Two staged libraries (``tools/stage_hotstars.py``) join TLUSTY 2025:

* **Götberg et al. (2023)** CMFGEN stripped stars, weak winds: families
  ``gotberg23_x{X_H}`` on (log g, log T), one node at [Fe/H] -0.71.
* **PoWR**: ``powr_ob`` on (log g, log T*) at sub-SMC / SMC / LMC, and the
  WN families ``powr_wn_x{X_H}`` (X_H 0 / 0.2 / 0.4 / 0.6) on (log R_t, log T*).

`route` decides, per star, which library serves it, from the star's
**hydrogen class** (the BPASS split: X_s > 0.4, 1e-3 < X_s < 0.4, X_s < 1e-3;
MIST's phase 9 "WR" is X_s < 0.4), its temperature, and its **wind**: a
declared mass-loss law (Sander & Vink 2020, MNRAS 499, 873, Eqs. 28-32) gives
Mdot from L/M, X_s and [Fe/H]; with the PoWR node's terminal velocity and
clumping that becomes the transformed radius R_t (Schmutz et al. 1989;
the PoWR definition)::

    R_t = R* [ (v_inf / 2500 km/s) / (Mdot sqrt(D) / 1e-4 Msun/yr) ]^(2/3)

A star whose wind is thinner than PoWR's thinnest model (log R_t above the
family's top node) has a hydrostatic, weak-wind spectrum: Götberg serves it.
A denser wind: PoWR. Nothing is tuned to a target.

Every row's choice is recorded (`SOURCE_*` codes in ``kcorrect``), never
silent. Default off: ``ARTPOP_HOT_STARS=full`` turns it on (``tlusty`` keeps
the 2026-10-16 TLUSTY-only evaluation).
"""
import json
import os

import numpy as np

from . import ARTPOP_HOME

__all__ = ['hotstars_path', 'HotStarFamily', 'families', 'manifest_sha',
           'H_RICH', 'H_POOR', 'H_FREE', 'X_ASSUMED', 'h_class_from_x',
           'gamma_e', 'log_mdot_sv2020', 'log_rstar', 'log_rt', 'route',
           'HOT_T_MIN', 'MDOT_FLOOR']

H_RICH, H_POOR, H_FREE = 0, 1, 2
# The surface X_H a class stands for when the caller has no number (BPASS's
# HR diagram and MIST's phase 9 give only the class): declared midpoints.
X_ASSUMED = {H_RICH: 0.7, H_POOR: 0.2, H_FREE: 0.0}
# hot routing starts here (TLUSTY's and PoWR-OB's cool edge); cooler rows keep
# C3K / MARCS / Tremblay whatever their class
HOT_T_MIN = 15000.0
# an H-rich star reaches the compact (Götberg X 0.7) and WNh (PoWR WN-H60)
# templates only above these: below, a B star is never given a WR spectrum
H_RICH_COMPACT_T_MIN = 30000.0
H_RICH_WNH_T_MIN = 50000.0
# log Mdot below this is "no wind" for R_t (SV2020's breakdown term underflows)
MDOT_FLOOR = -14.0

L_SUN, R_SUN, SIGMA_SB = 3.828e33, 6.957e10, 5.670374419e-5


def hotstars_path():
    """``$ARTPOP_HOTSTARS_PATH``, else ``<ARTPOP_HOME>/hotstars/blocks``."""
    return os.environ.get('ARTPOP_HOTSTARS_PATH') or os.path.join(ARTPOP_HOME, 'hotstars', 'blocks')


_MANIFEST = {}


def _manifest(path=None):
    root = path or hotstars_path()
    p = os.path.join(root, 'manifest.json')
    mt = os.path.getmtime(p) if os.path.isfile(p) else None
    hit = _MANIFEST.get(root)
    if hit is None or hit[0] != mt:
        if mt is None:
            raise FileNotFoundError(f'hot-star libraries are not staged under {root} '
                                    '(tools/stage_hotstars.py)')
        with open(p) as fh:
            hit = _MANIFEST[root] = (mt, json.load(fh))
    return hit[1]


def families(path=None):
    """``{family: manifest entry}`` of every staged family."""
    return _manifest(path)['families']


def manifest_sha(path=None):
    import hashlib
    with open(os.path.join(path or hotstars_path(), 'manifest.json'), 'rb') as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def _spectral_library_base():
    from .kcorrect import SpectralLibrary
    return SpectralLibrary


class HotStarFamily(_spectral_library_base()):
    """
    One staged family: surface F_lambda (erg/s/cm^2/A; ``surface_scale = 1``)
    on (``ax2``, log T) per [Fe/H] node, NaN where the grid has no model.
    ``ax2`` is log g, or log R_t for the PoWR WN families (``kind``).
    """

    flux_unit = 'f_lam'
    surface_scale = 1.0

    def __init__(self, family, path=None):
        self.root = path or hotstars_path()
        self.family = family
        self.name = family
        self.info = families(self.root)[family]
        self.kind = self.info['kind']
        self._blocks = None

    def _load(self):
        if self._blocks is None:
            self._blocks = {}
            for zh in self.info['nodes']:
                with np.load(os.path.join(self.root, f'{self.family}_zh{zh:+.2f}.npz')) as z:
                    self._blocks[round(float(zh), 2)] = (z['ax2'], z['logt'], z['flux'])
            self._feh = np.array(sorted(self._blocks))
            self._wave = np.load(os.path.join(self.root, 'hotstars_wave_vac.npy'))
        return self._blocks

    @property
    def feh_grid(self):
        self._load()
        return self._feh

    @property
    def wave(self):
        self._load()
        return self._wave

    def grid(self, feh=None):
        blocks = self._load()
        key = round(float(feh), 2)
        if key not in blocks:
            raise ValueError(f'[Fe/H] = {feh} is not a staged {self.family} node {self.feh_grid.tolist()}')
        return blocks[key]

    def node_wind(self, feh):
        """``(v_inf km/s, clumping D)`` of the node nearest ``feh`` (R_t families)."""
        nw = self.info.get('node_wind') or {}
        if not nw:
            return 2500.0, 1.0
        keys = sorted(nw, key=lambda k: abs(float(k) - float(feh)))
        w = nw[keys[0]]
        return float(w['vinf_km_s']), float(w['clumping_d'])


# ------------------------------------------------------------------- winds
def h_class_from_x(x_h):
    """BPASS's three classes from a surface hydrogen mass fraction."""
    x = np.asarray(x_h, dtype=float)
    return np.where(x > 0.4, H_RICH, np.where(x > 1e-3, H_POOR, H_FREE)).astype(np.int8)


def gamma_e(log_l, mass, x_h):
    """Electron-scattering Eddington factor, SV2020 Eq. 4 with q_ion = (1 + X)/2
    (one free electron per amu of H, half one per amu of He; 0.5 at X = 0 as in
    their inner wind)."""
    q = 0.5 * (1.0 + np.asarray(x_h, dtype=float))
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 ** (-4.51) * q * 10.0 ** np.asarray(log_l, float) / np.asarray(mass, float)


def log_mdot_sv2020(log_l, mass, x_h, feh):
    """
    Sander & Vink (2020) Eq. 28 with Eqs. 29-32 (a = 2.932, Gamma_e,b, c_bd,
    Mdot_off linear in log Z/Z_sun; Z/Z_sun = 10^[Fe/H], the iron-group
    abundance that drives the wind). Below the breakdown it falls
    exponentially; floored at `MDOT_FLOOR`. log10(Msun/yr).
    """
    lz = np.asarray(feh, dtype=float)
    ge = np.clip(gamma_e(log_l, mass, x_h), 1e-12, 0.999)
    a = 2.932
    geb = -0.324 * lz + 0.244
    cbd = -0.44 * lz + 9.15
    off = 0.23 * lz - 2.61
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = np.log10(-np.log10(1.0 - ge))
        lm = a * x - np.log10(2.0) * (geb / ge) ** cbd + off
    lm = np.where(np.isfinite(lm), lm, MDOT_FLOOR)
    return np.maximum(lm, MDOT_FLOOR)


def log_rstar(log_l, log_teff):
    """log10 R / R_sun from L = 4 pi R^2 sigma T^4."""
    return (0.5 * (np.asarray(log_l, float) + np.log10(L_SUN)
                   - np.log10(4 * np.pi * SIGMA_SB) - 4 * np.asarray(log_teff, float))
            - np.log10(R_SUN))


def log_rt(log_l, log_teff, log_mdot, vinf_km_s, clumping_d):
    """The PoWR transformed radius, log10 R_t / R_sun."""
    return (log_rstar(log_l, log_teff)
            + (2.0 / 3.0) * (np.log10(vinf_km_s / 2500.0)
                             - (np.asarray(log_mdot, float) + 0.5 * np.log10(clumping_d) + 4.0)))


# ------------------------------------------------------------------- router
def _family_range(lib, feh):
    """(ax2, logt) node vectors of the node nearest feh."""
    node = lib.feh_grid[np.argmin(np.abs(lib.feh_grid - float(feh)))]
    ax2, logt, _ = lib.grid(node)
    return ax2, logt


def route(log_teff, log_g, feh, h_class=None, log_l=None, mass=None, x_h=None,
          available=None, path=None):
    """
    Candidate libraries per star, in order of preference.

    Parameters
    ----------
    log_teff, log_g, feh : arrays (feh may be a scalar)
    h_class : int array (`H_RICH` / `H_POOR` / `H_FREE`); default all H-rich
    log_l, mass : log L/L_sun and the present mass (Msun), for the wind; rows
        without them (NaN) cannot be routed to an R_t family
    x_h : surface X_H if known; default `X_ASSUMED` of the class
    available : the families that may be used (default: every staged one,
        plus ``'tlusty'``)

    Returns
    -------
    list of dict(family=str or tuple of (family, weight), rows=int array,
    ax2=float array, rank=int): evaluate each family on its rows, then take
    for every star the lowest rank with a finite value. ``ax2`` is log g, or
    log R_t (clipped into the family's range; ``weak`` marks rows whose wind is
    thinner than its thinnest model). Rows that no candidate serves keep the
    production backends (C3K / MARCS / Tremblay / blackbody).
    """
    lt = np.atleast_1d(np.asarray(log_teff, float))
    lg = np.broadcast_to(np.asarray(log_g, float), lt.shape)
    fe = np.broadcast_to(np.asarray(feh, float), lt.shape)
    n = lt.size
    hc = (np.zeros(n, np.int8) if h_class is None
          else np.broadcast_to(np.asarray(h_class, np.int8), lt.shape))
    xh = (np.vectorize(X_ASSUMED.get)(hc).astype(float) if x_h is None
          else np.broadcast_to(np.asarray(x_h, float), lt.shape))
    ll = np.full(n, np.nan) if log_l is None else np.broadcast_to(np.asarray(log_l, float), lt.shape)
    mm = np.full(n, np.nan) if mass is None else np.broadcast_to(np.asarray(mass, float), lt.shape)
    fams = families(path)
    avail = set(fams) | {'tlusty'} if available is None else set(available)
    hot = lt >= np.log10(HOT_T_MIN)
    out = []

    def add(family, rows, ax2, rank, **kw):
        if np.size(rows):
            out.append(dict(family=family, rows=np.asarray(rows), ax2=np.asarray(ax2, float),
                            rank=rank, **kw))

    def gotberg_mix(x):
        """Götberg families bracketing X_H, linear weights."""
        xs = sorted(f['x_h'] for k, f in fams.items()
                    if k.startswith('gotberg') and k in avail)
        if not xs:
            return None
        x = float(np.clip(x, xs[0], xs[-1]))
        j = int(np.clip(np.searchsorted(xs, x) - 1, 0, max(len(xs) - 2, 0)))
        if len(xs) == 1 or x <= xs[0]:
            return ((f'gotberg23_x{xs[0]:.2f}', 1.0),)
        lo, hi = xs[j], xs[j + 1]
        w = (x - lo) / (hi - lo)
        return tuple((f'gotberg23_x{v:.2f}', ww) for v, ww in ((lo, 1 - w), (hi, w)) if ww > 0)

    def wn_families(x):
        """Every available WN family, nearest X_H first (the fallback chain:
        the X-nearest grid may not reach the star's T*)."""
        xs = sorted((f['x_h'] for k, f in fams.items() if k.startswith('powr_wn') and k in avail),
                    key=lambda v: (abs(v - x), v))
        return [f'powr_wn_x{v:.1f}' for v in xs]

    # ---- H-rich: TLUSTY, then PoWR OB, then Götberg X 0.7 (compact), then WN-H60 (WNh)
    rich = np.flatnonzero(hot & (hc == H_RICH))
    if rich.size:
        if 'tlusty' in avail:
            add('tlusty', rich, lg[rich], 0)
        if 'powr_ob' in avail:
            add('powr_ob', rich, lg[rich], 1)
        mix = gotberg_mix(0.7)
        compact = rich[lt[rich] >= np.log10(H_RICH_COMPACT_T_MIN)]
        if mix and compact.size:
            add(mix, compact, lg[compact], 2)
        wnh = rich[lt[rich] >= np.log10(H_RICH_WNH_T_MIN)]
        for k, wn in enumerate(wn_families(0.7) if wnh.size else []):
            _add_wn(add, wn, wnh, lt, ll, mm, xh, fe, 3 + 0.1 * k, path)
    # ---- H-poor / H-free: weak wind -> Götberg; dense -> PoWR WN; else either
    for cls in (H_POOR, H_FREE):
        rows = np.flatnonzero(hot & (hc == cls))
        if not rows.size:
            continue
        for x in np.unique(xh[rows]):
            r = rows[xh[rows] == x]
            mix = gotberg_mix(x)
            weak = np.zeros(r.size, bool)
            for k, wn in enumerate(wn_families(x)):
                w = _add_wn(add, wn, r, lt, ll, mm, xh, fe, 1 + 0.1 * k, path)
                if k == 0:
                    weak = w      # thinner than the X-nearest grid's thinnest wind
            if mix:
                add(mix, r[weak], lg[r[weak]], 0)
                add(mix, r[~weak], lg[r[~weak]], 2)
    return out


def _add_wn(add, wn, rows, lt, ll, mm, xh, fe, rank, path):
    """Route rows with a known L and mass to a PoWR WN family at their own R_t;
    returns the 'weak' mask (wind thinner than the family's thinnest model)."""
    lib = HotStarFamily(wn, path)
    ok = np.isfinite(ll[rows]) & np.isfinite(mm[rows]) & (mm[rows] > 0)
    weak = np.zeros(rows.size, bool)
    if not ok.any():
        return weak
    r = rows[ok]
    lmd = log_mdot_sv2020(ll[r], mm[r], xh[r], fe[r])
    vals = np.empty(r.size)
    wk = np.empty(r.size, bool)
    for zf in np.unique(fe[r]):
        s = fe[r] == zf
        vinf, dd = lib.node_wind(zf)
        rt = log_rt(ll[r][s], lt[r][s], lmd[s], vinf, dd)
        ax2, _ = _family_range(lib, zf)
        wk[s] = rt > ax2[-1]
        vals[s] = np.clip(rt, ax2[0], ax2[-1])
    add(wn, r, vals, rank, log_mdot=lmd)
    weak[ok] = wk
    return weak


# ------------------------------------------------------------ row spectra
_SPEC = {}


def _unit_blocks(family, path=None):
    """Per node: (ax2, logt, flux / sigma T^4), i.e. unit bolometric flux."""
    key = (path or hotstars_path(), family)
    if key not in _SPEC:
        lib = HotStarFamily(family, path)
        nodes = {}
        for zh in lib.feh_grid:
            a2, lt, f = lib.grid(zh)
            nodes[float(zh)] = (a2, lt, np.asarray(f, float) / (SIGMA_SB * 10.0 ** (4 * lt))[None, :, None])
        _SPEC[key] = (np.asarray(lib.wave, float), lib.feh_grid, nodes)
    return _SPEC[key]


def family_spectra(family, feh, ax2, log_teff, path=None):
    """
    Spectra of one family (or a Götberg X_H mix) at (ax2, log T) per row,
    linear in [Fe/H] between nodes (clipped) and bilinear in (ax2, log T);
    normalised to INT f dlam = 1. NaN rows where a needed corner has no model.
    Returns ``(lam, f (n, n_lam))``.
    """
    mix = family if isinstance(family, tuple) else ((family, 1.0),)
    a2 = np.atleast_1d(np.asarray(ax2, float))
    lt = np.atleast_1d(np.asarray(log_teff, float))
    acc, wsum, lam = None, np.zeros(lt.size), None
    for name, w in mix:
        lam, zg, nodes = _unit_blocks(name, path)
        zf = float(np.clip(feh, zg[0], zg[-1]))
        if zg.size == 1:
            zw = ((float(zg[0]), 1.0),)
        else:
            iz = int(np.clip(np.searchsorted(zg, zf) - 1, 0, zg.size - 2))
            fz = (zf - zg[iz]) / (zg[iz + 1] - zg[iz])
            zw = tuple((float(zg[iz + d]), ww) for d, ww in ((0, 1 - fz), (1, fz)) if ww > 0)
        out = np.zeros((lt.size, lam.size))
        for z, wz in zw:
            ax, tx, f = nodes[z]
            ig = np.clip(np.searchsorted(ax, a2) - 1, 0, ax.size - 2)
            it = np.clip(np.searchsorted(tx, lt) - 1, 0, tx.size - 2)
            fg = np.clip((a2 - ax[ig]) / (ax[ig + 1] - ax[ig]), 0, 1)
            ft = np.clip((lt - tx[it]) / (tx[it + 1] - tx[it]), 0, 1)
            # outside the hull in either axis: no model (NaN), never extrapolate;
            # inside, a masked bilinear (weights renormalised over the corners
            # with a model: PoWR's grids are triangular), as in the K table
            inside = ((a2 >= ax[0] - 1e-9) & (a2 <= ax[-1] + 1e-9)
                      & (lt >= tx[0] - 1e-9) & (lt <= tx[-1] + 1e-9))
            node = np.zeros((lt.size, lam.size))
            wn = np.zeros(lt.size)
            for dg, wg in ((0, 1 - fg), (1, fg)):
                for dt, wt in ((0, 1 - ft), (1, ft)):
                    corner = f[ig + dg, it + dt]
                    fin = np.isfinite(corner[:, 0]) & (wg * wt > 0)
                    node[fin] += (wg * wt)[fin, None] * corner[fin]
                    wn[fin] += (wg * wt)[fin]
            good = inside & (wn > 0)
            node[good] /= wn[good, None]
            node[~good] = np.nan
            out += wz * node
        ok = np.isfinite(out).all(axis=1)
        if acc is None:
            acc = np.zeros_like(out)
        acc[ok] += w * out[ok]
        wsum[ok] += w
    res = np.full_like(acc, np.nan)
    g = wsum > 0
    res[g] = acc[g] / wsum[g, None]
    return lam, res
