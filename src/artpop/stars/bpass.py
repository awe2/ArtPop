"""
BPASS v2.2.1 HR-diagram star lists (2026-10-29; ALVISS's physics-first BPASS
work, step 3).

BPASS's ``hrs-<sin|bin>-imf135_100.<z>.dat`` files give, per metallicity and
per log age 6.0 ... 11.0 (0.1 dex), the number of stars per 10^6 Msun formed
in an instantaneous burst in 0.1 x 0.1 dex cells of (log T, log L), split by
surface hydrogen: X > 0.4, 1e-3 < X < 0.4, X < 1e-3 (manual v2.2.1 p. 18).
Verified 2026-10-29 against BPASS's own SSP spectra: sum N L over the cells
equals the spectra's bolometric luminosity to <= 1 % (single) and <= 3 %
(binary, 1-3 Myr) at every age; the axis values are cell centres.

What this module adds on top of the counts:

* **time averaging** over an age bin ``[lo, hi]`` at constant SFR, with
  BPASS's own time bins (log t +- 0.05; the first from t = 0);
* **surface gravity**: the files hold (log T, log g) only as a separate
  marginal per class and age (the log T^4/g grid is the same information at
  fixed T), so log g is assigned to each (log T, log L) cell by anti-monotone
  quantile matching within the T row, counted from the bright end (a cell's
  stars get the mean log g of the matching slice of the g distribution:
  brighter <-> lower g, exact when mass is constant along the row). The
  stars fainter than the L grid (log L < -2.9) sit at the high-g end and are
  never matched;
* the **present mass** each cell then implies, M = g L / (4 pi sigma T^4 G);
* **metallicity**: BPASS Z = ``z_sun`` x 10^[Fe/H] (default 0.0142, Asplund
  2009, the scale ALVISS's [Fe/H] lives on; declared), counts mixed linearly
  in log Z between the two bracketing BPASS nodes (a stochastic mixture of the
  two HR diagrams, not an interpolated isochrone).

`BPASSStarList` is the cell table of one (kind, [Fe/H], age bin); the SSP built
from it is `artpop.stars.populations.BPASSSSP`.
"""
import glob
import os

import numpy as np

from .. import ARTPOP_HOME

__all__ = ['bpass_path', 'BPASS_Z', 'load_hrs', 'bin_weights', 'BPASSStarList',
           'H_RICH', 'H_POOR', 'H_FREE', 'LOGT_AXIS', 'LOGL_AXIS', 'LOG_AGES']

H_RICH, H_POOR, H_FREE = 0, 1, 2
LOGT_AXIS = 0.1 * np.arange(1, 101)          # cell centres (manual: 0.1 ... 10.0)
LOGL_AXIS = -2.9 + 0.1 * np.arange(100)      # -2.9 ... 7.0 (also log g, log TG)
LOG_AGES = 6.0 + 0.1 * np.arange(51)
BPASS_Z = {'zem5': 1e-5, 'zem4': 1e-4, 'z001': 0.001, 'z002': 0.002, 'z003': 0.003,
           'z004': 0.004, 'z006': 0.006, 'z008': 0.008, 'z010': 0.010, 'z014': 0.014,
           'z020': 0.020, 'z030': 0.030, 'z040': 0.040}
_G, _SIGMA, _LSUN, _MSUN = 6.674e-8, 5.670374419e-5, 3.828e33, 1.98847e33


def bpass_path():
    """``$ARTPOP_BPASS_PATH``, else ``<ARTPOP_HOME>/bpass/v2.2.1/imf135_100``: the
    unpacked ``hrs-*.dat.gz`` (and the other BPASS products)."""
    return os.environ.get('ARTPOP_BPASS_PATH') or os.path.join(
        ARTPOP_HOME, 'bpass', 'v2.2.1', 'imf135_100')


_HRS = {}


def load_hrs(kind, zlabel, imf='imf135_100', path=None):
    """
    ``(3 quantities [L, g, TG], 3 classes, 51 ages, 100 log T, 100 columns)``
    float32 counts per 10^6 Msun formed; parsed once into a ``.npy`` beside the
    source (``hrs-...dat.gz.npy``).
    """
    if kind not in ('sin', 'bin'):
        raise ValueError(f"kind must be 'sin' or 'bin', got {kind!r}")
    root = path or bpass_path()
    key = (root, kind, zlabel, imf)
    if key in _HRS:
        return _HRS[key]
    cands = glob.glob(os.path.join(root, f'hrs-{kind}-{imf}.{zlabel}.dat*'))
    src = [c for c in cands if not c.endswith('.npy')]
    if not src:
        raise FileNotFoundError(f'no BPASS hrs-{kind}-{imf}.{zlabel}.dat under {root}')
    cache = src[0] + '.npy'
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(src[0]):
        arr = np.load(cache)
    else:
        arr = np.loadtxt(src[0], dtype=np.float64).astype(np.float32)
        if arr.shape != (45900, 100):
            raise ValueError(f'{src[0]}: expected 45900 x 100 values, got {arr.shape}')
        arr = arr.reshape(3, 3, 51, 100, 100)
        tmp = f'{cache}.{os.getpid()}.part'
        with open(tmp, 'wb') as fh:
            np.save(fh, arr)
        os.replace(tmp, cache)
    _HRS[key] = arr
    return arr


def bin_weights(lo_yr, hi_yr):
    """
    Constant-SFR weights of BPASS's 51 ages over ``[lo_yr, hi_yr]``: each age
    stands for log t +- 0.05 (the first from t = 0), weighted by its overlap
    with the bin over the bin's length. Sums to 1.
    """
    lo_yr, hi_yr = float(lo_yr), float(hi_yr)
    if not hi_yr > lo_yr >= 0:
        raise ValueError(f'need 0 <= lo < hi, got {lo_yr}, {hi_yr}')
    edges = 10.0 ** np.concatenate([[-np.inf], LOG_AGES[:-1] + 0.05, [LOG_AGES[-1] + 0.05]])
    edges[0] = 0.0
    a, b = np.maximum(edges[:-1], lo_yr), np.minimum(edges[1:], hi_yr)
    w = np.clip(b - a, 0.0, None) / (hi_yr - lo_yr)
    return w / w.sum()


def _z_mix(z):
    """BPASS nodes and log-Z weights bracketing ``z`` (clipped at the ends)."""
    labs = sorted(BPASS_Z, key=BPASS_Z.get)
    lz = np.log10([BPASS_Z[k] for k in labs])
    x = float(np.clip(np.log10(z), lz[0], lz[-1]))
    j = int(np.clip(np.searchsorted(lz, x) - 1, 0, lz.size - 2))
    f = (x - lz[j]) / (lz[j + 1] - lz[j])
    return [(labs[j], 1.0 - f), (labs[j + 1], f)], bool(np.log10(z) < lz[0] or np.log10(z) > lz[-1])


def _assign_logg(n_l, n_g):
    """
    Anti-monotone quantile matching in one T row: ``n_l`` (100 L cells) and
    ``n_g`` (100 g cells) counts; returns the mean log g of each L cell's
    slice, matched in counts from the bright / low-g end. Cells with no stars
    get NaN.
    """
    out = np.full(n_l.size, np.nan)
    tot_l, tot_g = n_l.sum(), n_g.sum()
    if tot_l <= 0 or tot_g <= 0:
        return out
    # cumulative counts from the brightest L cell / the lowest g cell
    order_l = np.arange(n_l.size)[::-1]
    c_l = np.concatenate([[0.0], np.cumsum(n_l[order_l])])
    c_g = np.concatenate([[0.0], np.cumsum(n_g)])
    # the g distribution's quantile function, uniform within each 0.1 dex cell
    g_edges = np.concatenate([LOGL_AXIS - 0.05, [LOGL_AXIS[-1] + 0.05]])
    # integral of log g dN from the low-g end, as a function of N (piecewise quadratic)
    def int_logg(nn):
        nn = np.clip(nn, 0.0, c_g[-1])
        k = np.clip(np.searchsorted(c_g, nn, side='right') - 1, 0, n_g.size - 1)
        full = np.concatenate([[0.0], np.cumsum(n_g * LOGL_AXIS)])[k]
        part = nn - c_g[k]
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(n_g[k] > 0, part / n_g[k], 0.0)
        g_at = g_edges[k] + 0.1 * x
        return full + part * 0.5 * (g_edges[k] + g_at)
    for j, i in enumerate(order_l):
        if n_l[i] <= 0:
            continue
        a, b = c_l[j], c_l[j + 1]
        b_eff = min(b, c_g[-1])
        if b_eff <= a:
            continue
        out[i] = (int_logg(b_eff) - int_logg(a)) / (b_eff - a)
    return out


class BPASSStarList:
    """
    The expected star list of one BPASS population, time-averaged over an age
    bin, as a table of occupied cells.

    Parameters
    ----------
    kind : 'bin' or 'sin'
    feh : float, [Fe/H] (BPASS Z = ``z_sun`` 10^feh)
    lo_yr, hi_yr : the age bin (yr) at constant SFR
    z_sun : float, see the module docstring

    Attributes (one entry per occupied cell)
    ----------
    h_class, log_teff, log_l, log_g, mass : arrays
    n_per_msun : expected number of stars per Msun FORMED in the bin
    """

    def __init__(self, kind, feh, lo_yr, hi_yr, z_sun=0.0142, path=None, imf='imf135_100'):
        self.kind, self.feh, self.lo_yr, self.hi_yr = kind, float(feh), float(lo_yr), float(hi_yr)
        self.z = float(z_sun) * 10.0 ** self.feh
        self.z_sun = float(z_sun)
        w_age = bin_weights(lo_yr, hi_yr)
        mix, self.z_clipped = _z_mix(self.z)
        nl = np.zeros((3, 100, 100))
        ng = np.zeros((3, 100, 100))
        for lab, wz in mix:
            if wz <= 0:
                continue
            h = load_hrs(kind, lab, imf, path)
            nl += wz * np.tensordot(w_age, h[0].astype(float).transpose(1, 0, 2, 3), axes=(0, 0))
            ng += wz * np.tensordot(w_age, h[1].astype(float).transpose(1, 0, 2, 3), axes=(0, 0))
        self.z_nodes = [(lab, float(wz)) for lab, wz in mix if wz > 0]
        self.age_weights = w_age
        # log g per (class, T, L) cell
        logg = np.full_like(nl, np.nan)
        for c in range(3):
            for t in range(100):
                if nl[c, t].any():
                    logg[c, t] = _assign_logg(nl[c, t], ng[c, t])
        cls, it, il = np.nonzero(nl > 0)
        self.h_class = cls.astype(np.int8)
        self.log_teff = LOGT_AXIS[it]
        self.log_l = LOGL_AXIS[il]
        self.log_g = logg[cls, it, il]
        self.n_per_msun = nl[cls, it, il] / 1e6
        # a cell whose stars were never matched to a g (the g grid ran out):
        # the class / T row's median g -- recorded
        bad = ~np.isfinite(self.log_g)
        self.logg_filled = bad
        if bad.any():
            for j in np.flatnonzero(bad):
                row = logg[cls[j], it[j]]
                fin = np.isfinite(row)
                self.log_g[j] = np.median(row[fin]) if fin.any() else 4.0
        self.mass = self.mass_of(self.log_g, self.log_l, self.log_teff)
        # stars per Msun formed below the L grid (mass only; no light)
        n_g_tot = ng.sum() / 1e6
        self.n_unlisted_per_msun = max(n_g_tot - self.n_per_msun.sum(), 0.0)

    @staticmethod
    def mass_of(log_g, log_l, log_teff):
        """Present mass (Msun) from g, L and T: M = g L / (4 pi sigma T^4 G)."""
        return 10.0 ** (np.asarray(log_g) + np.asarray(log_l) + np.log10(_LSUN)
                        - np.log10(4 * np.pi * _SIGMA * _G) - 4 * np.asarray(log_teff)
                        - np.log10(_MSUN))

    def __len__(self):
        return self.h_class.size

    @property
    def lbol_per_msun(self):
        """Bolometric L (Lsun) per Msun formed, from the cells."""
        return float(np.sum(self.n_per_msun * 10.0 ** self.log_l))

    @property
    def live_mass_per_msun(self):
        """Present stellar mass per Msun formed from the listed cells (the
        unlisted faint stars carry ``n_unlisted_per_msun`` more, at < 0.3 Msun)."""
        return float(np.sum(self.n_per_msun * self.mass))

    def summary(self):
        n = self.n_per_msun
        return dict(kind=self.kind, feh=self.feh, z=self.z, z_nodes=self.z_nodes,
                    z_clipped=self.z_clipped, lo_yr=self.lo_yr, hi_yr=self.hi_yr,
                    n_cells=len(self), n_per_msun=float(n.sum()),
                    n_by_class=[float(n[self.h_class == c].sum()) for c in range(3)],
                    lbol_per_msun=self.lbol_per_msun,
                    lbol_by_class=[float(np.sum((n * 10 ** self.log_l)[self.h_class == c]))
                                   for c in range(3)],
                    n_logg_filled=int(self.logg_filled.sum()))


# ----------------------------------------------------------------------------
# The population: a star list in ArtPop's isochrone-row interface
# ----------------------------------------------------------------------------
JITTER_MIX = 1009.0     # second within-cell coordinate: (frac * 1009) mod 1


class BPASSIsochrone:
    """
    One BPASS star list as the row table ArtPop's luminosity-function sampler
    draws from: **one row per occupied HR-diagram cell**.

    * Rows carry the cell-centre (log T, log L), the matched log g, the implied
      present mass and the hydrogen class; their magnitudes come from the K
      table with the 'full' hot-star set, routed per class (`artpop.hotstars`):
      C3K / MARCS / TLUSTY / PoWR-OB for H-rich stars, Götberg for weak-wind
      H-poor and H-free stars, PoWR-WN for dense winds -- so the integrated
      light follows BPASS's star counts by construction, through our templates.
    * The pseudo initial-mass coordinate gives row ``i`` the interval
      ``[i, i + 1)``: the LF sampler's uniform draw within a row's half-bin is
      the star's position **within its cell**. ``u1 = frac`` and
      ``u2 = (1009 frac) mod 1`` place it in (log T, log L) uniformly over the
      0.1 x 0.1 dex cell, and log g moves with them at fixed mass
      (``dlog g = 4 dlog T - dlog L``). Sampled stars' magnitudes are evaluated
      at their own (T, L, g); the smooth component uses cell centres.
    * Ionizing photons are each row's own (`artpop.nebular.ionizing_rate` on the
      routed spectra). Inside the birth-cloud time (mean bin age <= t_bc) the
      nebular model runs on the rows as on a MIST isochrone; older bins have no
      HII regions, and all their photons feed the diffuse ionized gas
      (``knob`` 0, ``dig_absorbed`` of the config): Halpha from the stripped
      stars that exist in the render.
    """

    def __init__(self, kind, feh, lo_yr, hi_yr, phot_system, redshift=0.0, a_v_host=0.0,
                 a_v_mw=0.0, r_v=3.1, extinction_law='F99', kcorr_grid=None, kcorr_kw=None,
                 nebular=None, bolometric='teff', cool_giants='marcs', z_sun=0.0142,
                 jitter=True, path=None, p_present=None):
        from ..filters import get_filter_names, phot_system_lookup
        from ..nebular import NebularConfig
        from ..nebular import is_active as nebular_active
        self.star_list = sl = BPASSStarList(kind, feh, lo_yr, hi_yr, z_sun=z_sun, path=path)
        self.kind, self.feh = kind, float(feh)
        self.lo_yr, self.hi_yr = float(lo_yr), float(hi_yr)
        # present-day mass (with remnants) per formed Msun in the caller's
        # convention (ALVISS: MIST v2.5 + Kroupa, per bin); None: total_mass is
        # the formed mass
        self.p_present = None if p_present is None else float(p_present)
        # the bin's mean age at constant SFR (what young_row_mask compares to t_bc)
        self.log_age = float(np.log10(0.5 * (self.lo_yr + self.hi_yr)))
        n = len(sl)
        self.n_rows = n
        self.mini = np.arange(n) + 0.5
        self.mact = np.asarray(sl.mass, float)
        self.h_class = sl.h_class
        self.log_Teff = np.asarray(sl.log_teff, float)
        self.log_L = np.asarray(sl.log_l, float)
        self.log_g = np.asarray(sl.log_g, float)
        self.eep = np.arange(n, dtype=float)
        self.jitter = bool(jitter)
        self.redshift, self.a_v_host, self.a_v_mw = float(redshift), float(a_v_host), float(a_v_mw)
        self.r_v, self.extinction_law = float(r_v), extinction_law
        self.bolometric, self.cool_giants = bolometric, cool_giants
        self._kcorr_kw = dict(kcorr_kw or {})
        systems = [phot_system] if isinstance(phot_system, str) else list(phot_system)
        self._filters = [f for s in systems for f in get_filter_names(s)]
        lookup = phot_system_lookup()
        self._by_system = {}
        for f in self._filters:
            self._by_system.setdefault(lookup[f], []).append(f)
        self._grids = {}
        for system, bands in self._by_system.items():
            self._grids[system] = self._grid(system, bands, kcorr_grid)
        mags, info = self._evaluate(self.log_Teff, self.log_g, self.log_L, self.h_class, self.mact)
        self.kcorr_info = info
        self.nebular = NebularConfig.coerce(nebular)
        self.nebular_blob_frac = None
        self.nebular_info = None
        self.nebular_delta_mag = {f: np.zeros(n) for f in self._filters}
        if nebular_active(self.nebular):
            mags = self._apply_nebular(mags)
        from astropy.table import Table
        self.mag_table = Table({f: mags[f] for f in self._filters})
        self._jit_cache = (None, None)
        self.photometry_info = dict(
            photometry='synthetic', stars=f'BPASS v2.2.1 {kind} HR-diagram star list',
            star_list=sl.summary(), templates="'full' hot-star set (artpop.hotstars.route)",
            frame=dict(redshift=self.redshift, a_v_host=self.a_v_host, a_v_mw=self.a_v_mw))

    # -- photometry ---------------------------------------------------------
    def _grid(self, system, bands, kcorr_grid):
        from ..kcorrect import KCorrectionGrid
        if kcorr_grid is not None:
            return kcorr_grid, (dict(a_v_host=self.a_v_host, a_v_mw=self.a_v_mw)
                                if kcorr_grid.has_dust_axes else {})
        kw = dict(extinction_law=self.extinction_law, r_v=self.r_v, bolometric=self.bolometric,
                  cool_giants=self.cool_giants)
        kw.update(self._kcorr_kw)
        return KCorrectionGrid.for_dust(system, bands=bands, a_v_host=self.a_v_host,
                                        a_v_mw=self.a_v_mw, hot_stars='full', **kw)

    def _hot(self, log_l, h_class, mass):
        return dict(h_class=np.asarray(h_class, np.int8), log_l=np.asarray(log_l, float),
                    mass=np.asarray(mass, float))

    def _evaluate(self, log_teff, log_g, log_l, h_class, mass):
        """Absolute magnitudes at this frame, routed per class."""
        out, info = {}, {}
        hot = self._hot(log_l, h_class, mass)
        for system, bands in self._by_system.items():
            grid, dust_kw = self._grids[system]
            m, inf = grid.magnitudes(log_teff, log_g, self.feh, self.redshift, log_l=log_l,
                                     bands=bands, hot=hot, **dust_kw)
            out.update(m)
            info[system] = inf
        return out, info

    def _curves(self):
        from ..filters import load_filter_system
        from ..kcorrect import photometry_curve_dir
        curves = {}
        for system, bands in self._by_system.items():
            fs = load_filter_system(system, bands=bands, curve_dir=photometry_curve_dir(
                system, self._kcorr_kw.get('curve_dir')))
            curves.update({b: fs.get_trans(b) for b in bands})
        return curves

    def _apply_nebular(self, mags):
        from ..nebular import (apply_to_rows, ionizing_rate, dig_band_abs_mags_per_photon,
                               extinction_curve)
        cfg = self.nebular
        n = self.n_rows
        hot = self._hot(self.log_L, self.h_class, self.mact)
        curves = self._curves()
        if 10.0 ** self.log_age <= cfg.t_bc_myr * 1e6 * (1 + 1e-9):
            new, blob, info = apply_to_rows(
                cfg, mags, curves, self.log_age, self.feh, self.log_Teff, self.log_g, self.log_L,
                redshift=self.redshift, a_v_host=self.a_v_host, a_v_mw=self.a_v_mw,
                extinction_law=self.extinction_law, r_v=self.r_v, hot=hot)
        else:
            # no birth clouds: every ionizing photon of every row goes to the DIG
            q_raw, src = ionizing_rate(self.log_Teff, self.log_g, self.log_L, self.feh, hot=hot)
            new = {b: np.array(v, float, copy=True) for b, v in mags.items()}
            blob = {b: np.zeros(n) for b in mags}
            info = dict(mask=np.zeros(n, bool), n_rows=0, q_h=np.zeros(n), q_h_raw=q_raw,
                        source=src, knob=0.0, a_v_bc=0.0, older_than_t_bc=True)
            if cfg.dig:
                info.update(dig_absorbed=float(cfg.dig_absorbed), dig_log_u=float(cfg.dig_log_u),
                            dig_m1=dig_band_abs_mags_per_photon(
                                cfg, curves, self.feh, redshift=self.redshift,
                                a_v_host=self.a_v_host, a_v_mw=self.a_v_mw,
                                ext=extinction_curve(self.extinction_law, self.r_v)))
        for f in self._filters:
            self.nebular_delta_mag[f] = np.asarray(new[f], float) - np.asarray(mags[f], float)
        self.nebular_blob_frac = blob
        self.nebular_info = info
        return new

    # -- the isochrone interface ArtPop's LF sampler uses ---------------------
    @property
    def filters(self):
        return list(self._filters)

    @property
    def m_min(self):
        return 0.0

    @property
    def m_max(self):
        return float(self.n_rows)

    def imf_weights(self, imf=None, m_min_norm=None, m_max_norm=None, norm_type='number'):
        """Expected number per row (BPASS's own IMF, imf135_100), normalised to
        sum 1 (``'number'``) or weighted by present mass (``'mass'``)."""
        w = np.asarray(self.star_list.n_per_msun, float)
        if norm_type == 'mass':
            w = w * self.mact
        return w / w.sum()

    def ssp_surviving_mass(self, imf=None, m_min=None, m_max=None, add_remnants=True, **kwargs):
        """Per formed Msun: the listed live mass, or (``add_remnants``) the
        caller's ``p_present`` (1 without it), so ArtPop's remnant factor is
        live / p_present and ``total_mass`` keeps the caller's convention."""
        if add_remnants:
            return 1.0 if self.p_present is None else self.p_present
        return self.star_list.live_mass_per_msun

    def imf_bin_edges(self, m_min_norm=None):
        i = np.arange(self.n_rows, dtype=float)
        return i, i + 1.0

    def _rows(self, x):
        x = np.asarray(x, float)
        r = np.clip(np.floor(x).astype(int), 0, self.n_rows - 1)
        return r, np.clip(x - r, 0.0, 1.0)

    def jittered(self, x):
        """Per-star (log T, log L, log g) inside the cell and the magnitudes there."""
        x = np.asarray(x, float)
        key = (x.size, hash(x.tobytes()))
        if self._jit_cache[0] == key:
            return self._jit_cache[1]
        r, u1 = self._rows(x)
        u2 = np.mod(u1 * JITTER_MIX, 1.0)
        dt = (u1 - 0.5) * 0.1 if self.jitter else 0.0 * u1
        dl = (u2 - 0.5) * 0.1 if self.jitter else 0.0 * u2
        lt, ll = self.log_Teff[r] + dt, self.log_L[r] + dl
        lg = self.log_g[r] + 4.0 * dt - dl
        mags, _ = self._evaluate(lt, lg, ll, self.h_class[r], self.mact[r]) if x.size else (
            {f: np.array([]) for f in self._filters}, None)
        out = dict(log_Teff=lt, log_L=ll, log_g=lg)
        for f in self._filters:
            out[f] = np.asarray(mags[f], float) + self.nebular_delta_mag[f][r]
        self._jit_cache = (key, out)
        return out

    def interpolate(self, y_name, x_interp, x_name='mini', slope=None):
        if x_name != 'mini':
            raise ValueError("BPASSIsochrone interpolates on the row coordinate 'mini' only")
        if y_name in self._filters or y_name in ('log_Teff', 'log_L', 'log_g'):
            return self.jittered(x_interp)[y_name]
        r, _ = self._rows(x_interp)
        return np.asarray(getattr(self, y_name), float)[r]


def _bpass_ssp_class():
    from .populations import SSP

    class BPASSSSP(SSP):
        """
        A BPASS star-list population (one age bin at constant SFR), sampled by
        ArtPop's luminosity-function path. ``total_mass`` follows ALVISS's
        convention when ``p_present`` is given (present-day mass with remnants,
        = formed mass x p_present); without it ``total_mass`` is the formed mass.
        """

        def __init__(self, kind, feh, lo_yr, hi_yr, phot_system, total_mass=None,
                     num_stars=None, p_present=None, distance=None, a_lam=0.0,
                     mag_limit=None, mag_limit_band=None, random_state=None, isochrone=None,
                     **iso_kw):
            from astropy import units as u
            iso = isochrone if isochrone is not None else BPASSIsochrone(
                kind, feh, lo_yr, hi_yr, phot_system, p_present=p_present, **iso_kw)
            self.kind, self.feh, self.log_age = kind, float(feh), iso.log_age
            self.lo_yr, self.hi_yr = float(lo_yr), float(hi_yr)
            self.phot_system = phot_system
            self.p_present = None if p_present is None else float(p_present)
            if mag_limit is None:
                mag_limit, mag_limit_band = np.inf, (mag_limit_band or iso.filters[0])
            if distance is None:
                distance = 10 * u.pc
            super().__init__(isochrone=iso, num_stars=num_stars, total_mass=total_mass,
                             distance=distance, a_lam=a_lam, mag_limit=mag_limit,
                             mag_limit_band=mag_limit_band, imf='kroupa', add_remnants=True,
                             random_state=random_state, sampling='lf')
            self._r.update({'BPASS': f'v2.2.1 {kind}', 'bin (Myr)': f'{lo_yr / 1e6:g}-{hi_yr / 1e6:g}',
                            '[Fe/H]': self.feh})

        def _attach_nebular(self):
            """Per-star nebular share and photons from the star's own row (the
            row coordinate is not a mass: no interpolation between rows)."""
            iso = self.isochrone
            if iso.nebular_blob_frac is None:
                return
            r, _ = iso._rows(self.initial_masses)
            self.nebular_blob_frac = {f: np.clip(np.asarray(v, float)[r], 0.0, 1.0)
                                      for f, v in iso.nebular_blob_frac.items()}
            info = iso.nebular_info or {}
            self.nebular_q_h = np.clip(np.asarray(info.get('q_h', np.zeros(iso.n_rows)), float)[r], 0, None)
            q_raw = np.asarray(info.get('q_h_raw', np.zeros(iso.n_rows)), float)[r]
            self._attach_dig(info, q_raw)

    return BPASSSSP


def __getattr__(name):
    # BPASSSSP is built lazily (populations imports isochrones imports kcorrect)
    if name == 'BPASSSSP':
        cls = _bpass_ssp_class()
        globals()['BPASSSSP'] = cls
        return cls
    raise AttributeError(name)
