"""
Nebular emission around young O/B stars: the "nebular emission knob".

ArtPop renders a young star as a bare photosphere at a point. A real O/B star
sits in its birth cloud: dust dims it, the gas around it turns its ionizing
photons (lambda < 912 A, which no rendered band ever sees) into Balmer and
forbidden lines that do land in u..y, and that light comes from a region tens
to hundreds of pc across rather than from a point. This module models all
three with **one dial**, ``NebularConfig.knob = k`` in [0, 1], read as the
fraction of the star's ionizing photons the nebula captures (1 - f_esc):

1. **Birth-cloud dust.** The young star's continuum sees an extra host-frame
   screen ``A_V,bc = k * a_v_max`` on top of the host's own ``A_V_host``,
   inside the same K/dust integral as everything else
   (`~artpop.kcorrect.band_offset`), applied as a *differential* on top of the
   row's existing offset -- so at k = 0 it is exactly zero.
2. **Lines** (the "spectral PSF"). ``Q_H`` is integrated from the star's own
   spectrum below the Lyman edge (C3K, the library MIST itself used; a
   blackbody above its 50 kK ceiling). The nebula emits
   ``L(Hbeta) = k * 4.757e-13 erg * Q_H`` (case B, 10^4 K) and every other line
   in the ratio a MAPPINGS V 5.1 HII-region model gives at the gas's
   (12 + log O/H, log U, log P/k). Each line is placed at ``(1+z) lambda`` and
   weighted by the filter curve there **exactly** -- lines are delta
   functions, so there is no z grid to interpolate across a filter edge.
3. **Spatial blob.** In each band the fraction
   ``f_blob = (k F_star + F_lines) / (F_star + F_lines)`` of the star's light
   is spread by a unit-sum Gaussian of physical FWHM ``fwhm_pc`` (converted to
   an angle with D_A, as r_eff is) before the PSF; the rest stays a point.
   Line light is always diffuse, continuum light in proportion to k. See
   `~artpop.image.IdealImager.observe`.

Only rows of an SSP no older than ``t_bc_myr`` (Charlot & Fall 2000's
birth-cloud lifetime) with ``log Teff >= log_teff_min`` are touched ("OB
rows"). Their line strength scales with their own ``Q_H``, so a late-B star
contributes almost nothing by itself; the threshold decides who gets dust and
blob.

**k = 0 is a no-op, bit for bit**: every entry point checks
`NebularConfig.active` before touching an array (test N1).

Approximations (each one also a systematics-ledger row in ALVISS):
one screen for stars and lines (Calzetti's ~2x extra nebular dust ignored);
gas O/H = MAPPINGS' solar + [Fe/H]; line *ratios* from a MAPPINGS model whose
ionizing spectrum is a cluster's, with the absolute scale from each star's
``Q_H``; LTE C3K atmospheres for ``Q_H``; ionization-bounded case B; no
nebular continuum (free-free, free-bound, two-photon) yet; one log U and P per
galaxy; smooth-component OB light gets the magnitudes but no blob.
"""
import logging
import os
from dataclasses import dataclass, asdict, fields

import numpy as np

from .log import logger
from collections import OrderedDict

from .kcorrect import (C_AA, C3KLibrary, C3K_HULL, _H, _trapz_weights,
                       air_to_vac, band_weights, extinction_curve,
                       planck_lam, c3k_missing_mask, c3k_fill_blackbody,
                       C3K_MISSING_FLUX)

__all__ = ['NebularConfig', 'MappingsLineTable', 'row_spectra', 'ionizing_rate',
           'ionizing_photons_per_erg', 'line_band_abs_mags',
           'apply_to_rows', 'nebular_kernel', 'default_line_table_path',
           'L_SUN', 'HBETA_ERG_PER_ION', 'LYMAN_EDGE_AA',
           'stromgren_diameter_pc', 'sphere_kernel', 'stromgren_classes']

# its own channel, like `ArtPop Logger.dust`, so pipeline log muting of the
# parent logger does not hide an O/H clip
nebular_logger = logging.getLogger(logger.name + '.nebular')
nebular_logger.setLevel(logging.WARNING)

L_SUN = 3.828e33                    # erg / s (IAU 2015 nominal)
PC_CM = 3.0856775814913673e18       # cm
LYMAN_EDGE_AA = 911.753             # Angstrom, vacuum
# Hbeta energy per hydrogen recombination, case B, T = 10^4 K, n_e = 100 cm^-3:
# h nu(Hbeta) * alpha_eff(Hbeta) / alpha_B = 4.086e-12 erg * 3.03e-14 / 2.59e-13
# (Hummer & Storey 1987; Osterbrock & Ferland 2006, Table 4.2)
HBETA_ERG_PER_ION = 4.757e-13
# MAPPINGS V's solar oxygen abundance (Nicholls et al. 2017), the scale the
# grid's 12 + log O/H axis is on
OH_SOLAR_MAPPINGS = 8.76
_MAG_AB0 = 48.6
_FOUR_PI_10PC2 = 4.0 * np.pi * (10.0 * PC_CM) ** 2


def default_line_table_path():
    """The converted MAPPINGS grid shipped in ``artpop/data/nebular/``."""
    return os.path.join(os.path.dirname(__file__), 'data', 'nebular',
                        'mappings51_hii_lines.ecsv')


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NebularConfig:
    """
    The nebular emission knob and the fixed physics behind it.

    Parameters
    ----------
    knob : float
        k in [0, 1]: fraction of ionizing photons captured by the nebula.
        Scales the line flux, the birth-cloud dust (``k * a_v_max``) and the
        continuum's share of the blob. **0 is an exact no-op.**
    a_v_max : float
        Birth-cloud V-band extinction at k = 1 (mag). Default 1.5.
    t_bc_myr : float
        Oldest SSP that still sits in its birth cloud (Myr). Default 10.
    log_teff_min : float
        Coolest row treated as an O/B star. Default 4.0 (10 kK).
    log_u : float
        Ionization parameter of the MAPPINGS model (grid: -4 to -2).
    log_p : float
        log P/k of the MAPPINGS model (grid: 4.2 to 8.6). Default 6.2, a
        typical local HII region.
    fwhm_pc : float
        FWHM of the blob, physical pc, when ``size_mode = 'fixed'``. Default 100.
    size_mode : str
        ``'fixed'`` (default): one Gaussian of FWHM ``fwhm_pc`` for every star.
        ``'stromgren'`` (2026-09-29, user): each star's nebula -- its line
        light only; the star's continuum stays a point -- is a uniformly
        emitting Stromgren sphere, projected, of diameter
        ``D_S = 2 (3 k Q_H / (4 pi n_e^2 alpha_B))^(1/3)`` from the photons it
        captures, at gas density ``n_e_cm3``; stars are grouped into
        ``n_size_classes`` log-spaced size classes (one convolution each).
    n_e_cm3 : float
        Gas density for ``'stromgren'`` (cm^-3). Default 1 (diffuse dwarf-
        irregular gas: ~60 pc for 1e48 photons/s, ~140 pc for one O star,
        ~290 pc for 1e50).
    n_size_classes : int
        Size classes for ``'stromgren'``. Default 6.
    oh_solar : float
        12 + log O/H at [Fe/H] = 0, on the grid's own scale.
    lambda_min_aa, lambda_max_aa : float
        Rest wavelengths of the lines kept (Lyman alpha, a resonance line that
        dust destroys, is excluded by the lower bound).
    line_table : str or None
        Override path of the converted MAPPINGS table.
    """
    knob: float = 0.0
    a_v_max: float = 1.5
    t_bc_myr: float = 10.0
    log_teff_min: float = 4.0
    log_u: float = -3.0
    log_p: float = 6.2
    fwhm_pc: float = 100.0
    size_mode: str = 'fixed'
    n_e_cm3: float = 1.0
    n_size_classes: int = 6
    oh_solar: float = OH_SOLAR_MAPPINGS
    lambda_min_aa: float = 2000.0
    lambda_max_aa: float = 25000.0
    line_table: str = None

    def __post_init__(self):
        k = float(self.knob)
        if not 0.0 <= k <= 1.0:
            raise ValueError(f'nebular knob must be in [0, 1], got {k}')
        if self.a_v_max < 0:
            raise ValueError(f'a_v_max must be >= 0, got {self.a_v_max}')
        if self.fwhm_pc <= 0:
            raise ValueError(f'fwhm_pc must be > 0, got {self.fwhm_pc}')
        if self.t_bc_myr <= 0:
            raise ValueError(f't_bc_myr must be > 0, got {self.t_bc_myr}')
        if self.size_mode not in ('fixed', 'stromgren'):
            raise ValueError(f"size_mode must be 'fixed' or 'stromgren', got {self.size_mode!r}")
        if self.n_e_cm3 <= 0:
            raise ValueError(f'n_e_cm3 must be > 0, got {self.n_e_cm3}')
        if int(self.n_size_classes) < 1:
            raise ValueError(f'n_size_classes must be >= 1, got {self.n_size_classes}')

    @property
    def active(self):
        """True when the knob is above zero; nothing is computed otherwise."""
        return float(self.knob) > 0.0

    @property
    def a_v_bc(self):
        """The birth-cloud screen at this knob setting."""
        return float(self.knob) * float(self.a_v_max)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def coerce(cls, obj):
        """None, a mapping (e.g. a YAML ``nebular:`` block) or a config."""
        if obj is None or isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            names = {f.name for f in fields(cls)}
            unknown = set(obj) - names
            if unknown:
                raise ValueError(f'unknown nebular parameters {sorted(unknown)}; '
                                 f'expected a subset of {sorted(names)}')
            return cls(**{k: v for k, v in obj.items() if v is not None})
        raise TypeError(f'cannot make a NebularConfig from {type(obj).__name__}')


def is_active(cfg):
    """`NebularConfig.active` that also accepts None."""
    return cfg is not None and cfg.active


# ---------------------------------------------------------------------------
# the MAPPINGS line table
# ---------------------------------------------------------------------------
class MappingsLineTable:
    """
    Line fluxes relative to Hbeta on MAPPINGS V 5.1's HII-region grid.

    The grid is the one NebulaBayes distributes (Thomas et al. 2018,
    ApJ 856, 89; MAPPINGS: Sutherland & Dopita 2017), converted by
    ``tools/build_nebular_grid.py``: axes 12 + log O/H (12 nodes, 7.06-9.30),
    log U (9, -4 to -2) and log P/k (12, 4.2-8.6). Interpolated linearly in all
    three. log U and log P outside the grid **raise**; O/H below or above it is
    **clipped to the edge with a warning** (a very metal-poor dwarf is common,
    and its Balmer lines, which dominate the broadband effect, barely depend
    on O/H), and never extrapolated.
    """

    _AXES = ('oh', 'log_u', 'log_p')

    def __init__(self, path=None):
        from astropy.table import Table
        self.path = path or default_line_table_path()
        if not os.path.isfile(self.path):
            raise FileNotFoundError(
                f'MAPPINGS line table not found at {self.path}; build it with '
                'tools/build_nebular_grid.py')
        t = Table.read(self.path, format='ascii.ecsv')
        self.meta = dict(t.meta)
        self.axes = [np.unique(np.asarray(t[a], dtype=float)) for a in self._AXES]
        shape = tuple(a.size for a in self.axes)
        if len(t) != int(np.prod(shape)):
            raise ValueError(f'{self.path}: {len(t)} rows, expected a full '
                             f'{shape} grid')
        lam_air = self.meta['lambda_air_aa']
        self.names = [n for n in t.colnames if n not in self._AXES]
        self.lambda_air = np.array([float(lam_air[n]) for n in self.names])
        self.lambda_vac = air_to_vac(self.lambda_air)
        # sort rows onto the regular grid, C order (oh, log_u, log_p)
        idx = [np.searchsorted(ax, np.asarray(t[a], dtype=float))
               for ax, a in zip(self.axes, self._AXES)]
        flat = np.ravel_multi_index(idx, shape)
        vals = np.stack([np.asarray(t[n], dtype=float) for n in self.names], -1)
        cube = np.full(shape + (len(self.names),), np.nan)
        cube.reshape(-1, len(self.names))[flat] = vals
        if not np.isfinite(cube).all():
            raise ValueError(f'{self.path}: grid has holes')
        self.cube = cube
        self._rgi = None
        self._warned = set()

    def ratios(self, oh, log_u, log_p):
        """``(names, lambda_vac, ratio_to_Hbeta)`` at one gas state."""
        from scipy.interpolate import RegularGridInterpolator
        oh_ax, u_ax, p_ax = self.axes
        for v, ax, name in ((log_u, u_ax, 'log_u'), (log_p, p_ax, 'log_p')):
            if not ax[0] - 1e-9 <= float(v) <= ax[-1] + 1e-9:
                raise ValueError(f'{name} = {v} is outside the MAPPINGS grid '
                                 f'[{ax[0]}, {ax[-1]}]')
        oh = float(oh)
        oh_c = float(np.clip(oh, oh_ax[0], oh_ax[-1]))
        if oh_c != oh:
            key = round(oh, 3)
            if key not in self._warned:
                self._warned.add(key)
                nebular_logger.warning(
                    f'12 + log O/H = {oh:.3f} is outside the MAPPINGS grid '
                    f'[{oh_ax[0]:.3f}, {oh_ax[-1]:.3f}]; using the edge value '
                    f'{oh_c:.3f} (clipped, not extrapolated)')
        if self._rgi is None:
            self._rgi = RegularGridInterpolator(self.axes, self.cube)
        r = self._rgi([[oh_c, float(log_u), float(log_p)]])[0]
        return list(self.names), self.lambda_vac.copy(), r

    _CACHE = {}

    @classmethod
    def cached(cls, path=None):
        path = path or default_line_table_path()
        if path not in cls._CACHE:
            cls._CACHE[path] = cls(path)
        return cls._CACHE[path]


# ---------------------------------------------------------------------------
# per-row spectra and the ionizing photon rate
# ---------------------------------------------------------------------------
# 0: C3K models only; 1: C3K, with at least one corner an empty cell (no model)
# replaced by the nearest model in log g at the same Teff -- a best guess; 2: blackbody
# 0: C3K models only; 1: C3K, with at least one corner a cell C3K has no model
# for (FSPS's 1e-33 floor), served by a blackbody at that cell's Teff -- exactly
# as MIST v2.5's BC table and the K table (table v2, ledger S-51) serve it; 2: a
# blackbody at the row's own Teff (outside C3K's rectangle: > 50 kK, log g > 5.5)
SOURCE_C3K, SOURCE_C3K_BB, SOURCE_BB = 0, 1, 2
_BB_WAVE = np.geomspace(10.0, 1.0e6, 20000)      # Angstrom
_SIGMA_SB = 5.670374419e-5                        # erg / s / cm^2 / K^4

_C3K_BLOCKS = OrderedDict()
_C3K_BLOCKS_MAX = 4


def _c3k_block(lib, feh):
    """
    C3K block at one [Fe/H] node **as MIST v2.5 used it**: cells with no model
    (`kcorrect.c3k_missing_mask`, FSPS's own test) replaced by a blackbody at the
    cell's Teff (`kcorrect.c3k_fill_blackbody`) -- the same spectra the K table
    is built from, so Q_H, the birth-cloud dust and Delta m all agree. Returns
    the axes, wavelength, the filled f_nu block, the f_nu -> f_lam factor, the
    bolometric flux of every cell and the mask of filled cells. The spectra
    stay float32 f_nu; only the corners a row uses are converted and
    normalised, in `row_spectra`. The last few blocks are kept in memory.
    """
    key = (lib.resolution, lib.a_over_fe, float(feh), lib.root)
    if key not in _C3K_BLOCKS:
        logg, logt, raw = lib.grid(feh)
        lam = np.asarray(lib.wave, dtype=float)
        flux, filled = c3k_fill_blackbody(raw, lam, logt)
        flux = np.array(flux, copy=True)             # never alias the library's cache
        to_flam = C_AA / lam ** 2
        bol = np.asarray(flux.reshape(-1, lam.size) @ (to_flam * _trapz_weights(lam)),
                         dtype=float).reshape(flux.shape[:2])
        _C3K_BLOCKS[key] = (logg, logt, lam, flux, to_flam, bol, filled)
        while len(_C3K_BLOCKS) > _C3K_BLOCKS_MAX:
            _C3K_BLOCKS.popitem(last=False)
    _C3K_BLOCKS.move_to_end(key)
    return _C3K_BLOCKS[key]


def row_spectra(log_teff, log_g, feh, resolution='c3k_hr', spectra=None):
    """
    One spectrum per row, **normalised to unit bolometric flux**.

    C3K inside its hull, bilinear in (log g, log Teff) between the four
    surrounding cells of the nearest [Fe/H] node, with C3K read **as MIST v2.5
    used it**: a cell with no model is a blackbody at that cell's Teff, and a
    row touching one is flagged `SOURCE_C3K_BB`. Outside C3K's rectangle
    (> 50 kK; log g > 5.5) a blackbody at the row's own Teff (`SOURCE_BB`), as
    in MIST and the K table. The source is recorded per row, never silent.

    Returns
    -------
    groups : list of ``(idx, lam, f_lam)``
        Rows ``idx`` share the wavelength grid ``lam``; ``f_lam`` is
        ``(len(idx), len(lam))`` with ``INT f_lam dlam = 1``.
    source : `~numpy.ndarray`
        0 = C3K, 1 = C3K with a blackbody-filled corner, 2 = blackbody.
    """
    lt = np.atleast_1d(np.asarray(log_teff, dtype=float))
    lg = np.atleast_1d(np.asarray(log_g, dtype=float))
    n = lt.size
    source = np.full(n, SOURCE_BB, dtype=np.int8)
    groups = []

    lib = C3KLibrary(resolution=resolution, path=spectra)
    t_lo, t_hi = np.log10(C3K_HULL['teff'][0]), np.log10(C3K_HULL['teff'][1])
    in_hull = ((lt >= t_lo) & (lt <= t_hi)
               & (lg >= C3K_HULL['logg'][0]) & (lg <= C3K_HULL['logg'][1]))
    if in_hull.any() and lib.available:
        feh_node = float(lib.feh_grid[np.argmin(np.abs(lib.feh_grid - float(feh)))])
        g_ax, t_ax, lam, flux, to_flam, bol, filled = _c3k_block(lib, feh_node)

        def norm(ig, it):
            return np.asarray(flux[ig, it], dtype=float) * to_flam / bol[ig, it]

        rows = np.flatnonzero(in_hull)
        out = np.zeros((rows.size, lam.size))
        touched = np.zeros(rows.size, dtype=bool)
        for j, r in enumerate(rows):
            ig = int(np.clip(np.searchsorted(g_ax, lg[r]) - 1, 0, g_ax.size - 2))
            it = int(np.clip(np.searchsorted(t_ax, lt[r]) - 1, 0, t_ax.size - 2))
            fg = (lg[r] - g_ax[ig]) / (g_ax[ig + 1] - g_ax[ig])
            ft = (lt[r] - t_ax[it]) / (t_ax[it + 1] - t_ax[it])
            fg, ft = float(np.clip(fg, 0, 1)), float(np.clip(ft, 0, 1))
            for dg, wg in ((0, 1 - fg), (1, fg)):
                for dt, wt in ((0, 1 - ft), (1, ft)):
                    w = wg * wt
                    if w > 0:
                        out[j] += w * norm(ig + dg, it + dt)
                        touched[j] |= bool(filled[ig + dg, it + dt])
        groups.append((rows, lam, out))
        source[rows] = np.where(touched, SOURCE_C3K_BB, SOURCE_C3K)

    bb = np.flatnonzero(source == SOURCE_BB)
    if bb.size:
        teff = 10 ** lt[bb]
        # analytic bolometric normalisation: INT B_lam dlam = sigma T^4 / pi,
        # with planck_lam per cm of wavelength -> 1e-8 per Angstrom
        f = np.stack([planck_lam(_BB_WAVE, t) * 1e-8 / (_SIGMA_SB * t ** 4 / np.pi)
                      for t in teff])
        groups.append((bb, _BB_WAVE, f))
    return groups, source


def ionizing_photons_per_erg(lam, f_lam):
    """
    ``Q_H / L_bol`` (photons per erg) of spectra normalised to unit bolometric
    flux: ``INT_{lam < 912 A} f_lam lam / (h c) dlam``.
    """
    lam = np.asarray(lam, dtype=float)
    m = lam <= LYMAN_EDGE_AA
    if m.sum() < 2:
        return np.zeros(np.atleast_2d(f_lam).shape[0])
    w = _trapz_weights(lam[m]) * lam[m] / (_H * C_AA)
    return np.atleast_2d(f_lam)[:, m] @ w


def ionizing_rate(log_teff, log_g, log_l, feh, resolution='c3k_hr',
                  spectra=None):
    """
    Hydrogen-ionizing photon rate ``Q_H`` (photons / s) of each row, before
    any capture by the nebula (no knob, no Teff threshold).

    Returns ``(q_h, source)``; ``source`` per row as in `row_spectra`.
    """
    lt = np.atleast_1d(np.asarray(log_teff, dtype=float))
    ll = np.atleast_1d(np.asarray(log_l, dtype=float))
    q_h = np.zeros(lt.size)
    groups, source = row_spectra(lt, log_g, feh, resolution=resolution,
                                 spectra=spectra)
    for idx, lam, f in groups:
        q_h[idx] = ionizing_photons_per_erg(lam, f) * 10 ** ll[idx] * L_SUN
    return q_h, source


# ---------------------------------------------------------------------------
# lines into bands
# ---------------------------------------------------------------------------
def line_band_abs_mags(line_lum, lambda_vac, trans_wave, trans, redshift=0.0,
                       a_v_host=0.0, a_v_mw=0.0, ext=None):
    """
    "Absolute" AB magnitude of a set of emission lines in one band.

    Defined so that ``m = M + 5 log10(D_L / 10 pc)`` is the observed AB
    magnitude, the same convention the stellar columns (MIST M plus the
    K/dust offset) follow. For a line of luminosity ``L`` at rest ``lambda``,
    in the photon-counting AB system,

        f_nu = [L / (4 pi D_L^2)] T((1+z) lambda) (1+z) lambda
               / (c INT T(l) / l dl)

    -- a line has no bandwidth-compression ``(1+z)`` beyond the one in its
    observed wavelength. Dust: host (+ birth cloud) at the rest wavelength,
    Milky Way at the observed one.

    Parameters
    ----------
    line_lum : `~numpy.ndarray`, shape ``(n_row, n_line)``
        Line luminosities, erg/s.
    lambda_vac : `~numpy.ndarray`, shape ``(n_line,)``
        Rest vacuum wavelengths, Angstrom.

    Returns
    -------
    mags : `~numpy.ndarray`, shape ``(n_row,)``; ``inf`` where no line
    transmits.
    """
    L = np.atleast_2d(np.asarray(line_lum, dtype=float))
    lam = np.asarray(lambda_vac, dtype=float)
    tw = np.asarray(trans_wave, dtype=float)
    tt = np.asarray(trans, dtype=float)
    lam_obs = lam * (1.0 + float(redshift))
    t_at = np.interp(lam_obs, tw, tt, left=0.0, right=0.0)
    att = np.ones_like(lam)
    if a_v_host or a_v_mw:
        ext = extinction_curve() if ext is None else ext
        m = t_at > 0
        if m.any():
            a = np.zeros_like(lam)
            if a_v_host:
                a[m] += float(a_v_host) * ext(lam[m])
            if a_v_mw:
                a[m] += float(a_v_mw) * ext(lam_obs[m])
            att = 10 ** (-0.4 * a)
    norm = C_AA * np.sum(_trapz_weights(tw) * tt / tw)
    f_nu = (L / _FOUR_PI_10PC2) @ (t_at * lam_obs * att) / norm
    with np.errstate(divide='ignore'):
        return -2.5 * np.log10(f_nu) - _MAG_AB0


# ---------------------------------------------------------------------------
# the isochrone-row operation
# ---------------------------------------------------------------------------
def young_row_mask(log_age, log_teff, cfg):
    """Rows that get the nebular treatment (all False for an old SSP)."""
    lt = np.asarray(log_teff, dtype=float)
    if not is_active(cfg) or 10 ** float(log_age) > cfg.t_bc_myr * 1e6 * (1 + 1e-9):
        return np.zeros(lt.shape, dtype=bool)
    return lt >= float(cfg.log_teff_min)


def apply_to_rows(cfg, mags, curves, log_age, feh, log_teff, log_g, log_l,
                  redshift=0.0, a_v_host=0.0, a_v_mw=0.0, extinction_law='F99',
                  r_v=3.1, resolution='c3k_hr', spectra=None):
    """
    The nebular correction for one isochrone's magnitude columns.

    Parameters
    ----------
    mags : dict of ``{band: array}``
        The row magnitudes **after** K and host/MW dust (absolute; AB).
    curves : dict of ``{band: (trans_wave, trans)}``

    Returns
    -------
    new_mags : dict ``{band: array}``, the corrected columns (copies)
    blob_frac : dict ``{band: array}``, per row, 0 outside the OB rows
    info : dict, per-row ``q_h`` (captured photons / s), ``source``, the mask,
        the gas state and line list used
    """
    rows = young_row_mask(log_age, log_teff, cfg)
    n = rows.size
    new = {b: np.array(mags[b], dtype=float, copy=True) for b in mags}
    blob = {b: np.zeros(n) for b in mags}
    info = dict(mask=rows, n_rows=int(rows.sum()), q_h=np.zeros(n),
                source=np.full(n, -1, dtype=np.int8), a_v_bc=cfg.a_v_bc,
                knob=float(cfg.knob))
    if not rows.any():
        return new, blob, info

    k = float(cfg.knob)
    a_young = float(a_v_host) + cfg.a_v_bc
    ext = extinction_curve(extinction_law, r_v)
    idx_all = np.flatnonzero(rows)
    groups, source = row_spectra(np.asarray(log_teff)[rows],
                                 np.asarray(log_g)[rows], feh,
                                 resolution=resolution, spectra=spectra)
    info['source'][idx_all] = source

    # gas: MAPPINGS ratios at (O/H, U, P); Hbeta from each row's Q_H
    table = MappingsLineTable.cached(cfg.line_table)
    oh = cfg.oh_solar + float(feh)
    names, lam_vac, ratio = table.ratios(oh, cfg.log_u, cfg.log_p)
    keep = (lam_vac >= cfg.lambda_min_aa) & (lam_vac <= cfg.lambda_max_aa)
    names = [nm for nm, kp in zip(names, keep) if kp]
    lam_vac, ratio = lam_vac[keep], ratio[keep]
    info.update(oh=oh, oh_used=float(np.clip(oh, table.axes[0][0], table.axes[0][-1])),
                log_u=cfg.log_u, log_p=cfg.log_p, lines=names)

    z = float(redshift)
    for sub, lam, f in groups:
        r = idx_all[sub]
        q = ionizing_photons_per_erg(lam, f)                     # photons / erg
        q_h = k * q * (10 ** np.asarray(log_l, dtype=float)[r]) * L_SUN
        info['q_h'][r] = q_h
        line_lum = (HBETA_ERG_PER_ION * q_h)[:, None] * ratio[None, :]
        qw = _trapz_weights(lam)
        for band, (tw, tt) in curves.items():
            # birth-cloud dust on the continuum, as a differential against the
            # host-only dust of the same spectrum: exactly 0 when a_v_bc = 0
            w_bc = band_weights(lam, tw, tt, z, a_young, a_v_mw, ext, 'f_lam', qw)
            w_h = band_weights(lam, tw, tt, z, a_v_host, a_v_mw, ext, 'f_lam', qw)
            num, den = f @ w_bc, f @ w_h
            with np.errstate(divide='ignore', invalid='ignore'):
                d_bc = -2.5 * np.log10(num / den)
            d_bc[~np.isfinite(d_bc)] = 0.0
            m_star = new[band][r] + d_bc
            m_line = line_band_abs_mags(line_lum, lam_vac, tw, tt, z,
                                        a_young, a_v_mw, ext)
            f_star = 10 ** (-0.4 * m_star)
            f_line = np.where(np.isfinite(m_line), 10 ** (-0.4 * m_line), 0.0)
            tot = f_star + f_line
            new[band][r] = -2.5 * np.log10(tot)
            # what the nebula spreads: in 'fixed' mode the lines and the k share of
            # the continuum (dusty blurring); in 'stromgren' mode only the lines --
            # the Stromgren sphere is the size of the glowing gas, and a star's
            # photosphere stays a point (dimmed by the birth-cloud dust) (user, 2026-09-29)
            cont_share = k if cfg.size_mode == 'fixed' else 0.0
            blob[band][r] = (cont_share * f_star + f_line) / tot
    return new, blob, info


# ---------------------------------------------------------------------------
# the blob
# ---------------------------------------------------------------------------
def nebular_kernel(fwhm_pc, distance_angular, pixel_scale):
    """
    Unit-sum, pixel-integrated Gaussian stamp of physical FWHM ``fwhm_pc``.

    ``distance_angular`` is D_A (a Quantity or Mpc), ``pixel_scale`` in
    arcsec/pixel (Quantity or float). Pixel-integrated (erf differences) so a
    sub-pixel blob stays flux-exact and tends to a delta function.
    """
    from math import erf, sqrt
    from astropy import units as u
    d_pc = (distance_angular.to(u.pc).value if hasattr(distance_angular, 'unit')
            else float(distance_angular) * 1e6)
    ps = (u.Quantity(pixel_scale).to(u.arcsec / u.pixel).value
          if hasattr(pixel_scale, 'unit') else float(pixel_scale))
    fwhm_arcsec = np.degrees(float(fwhm_pc) / d_pc) * 3600.0
    sigma = fwhm_arcsec / ps / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(1, int(np.ceil(4.0 * sigma)))
    edges = np.arange(-half, half + 2) - 0.5
    cdf = np.array([0.5 * (1.0 + erf(e / (sqrt(2.0) * sigma))) for e in edges])
    p = np.diff(cdf)
    kern = np.outer(p, p)
    return kern / kern.sum()


# case B recombination coefficient at 10^4 K (Osterbrock & Ferland 2006, Table 2.1)
ALPHA_B = 2.59e-13                  # cm^3 / s


def stromgren_diameter_pc(q_h, n_e_cm3=1.0):
    """
    Stromgren diameter (pc) of an ionization-bounded nebula around a source
    of ``q_h`` absorbed ionizing photons per second in gas of density
    ``n_e_cm3``: ``2 (3 q_h / (4 pi n^2 alpha_B))^(1/3)`` (pure hydrogen, case B,
    10^4 K, no filling factor). 0 for ``q_h <= 0``.
    """
    q = np.clip(np.asarray(q_h, dtype=float), 0.0, None)
    r_cm = (3.0 * q / (4.0 * np.pi * float(n_e_cm3) ** 2 * ALPHA_B)) ** (1.0 / 3.0)
    return 2.0 * r_cm / PC_CM


def _pixels_per_pc(distance_angular, pixel_scale):
    from astropy import units as u
    d_pc = (distance_angular.to(u.pc).value if hasattr(distance_angular, 'unit')
            else float(distance_angular) * 1e6)
    ps = (u.Quantity(pixel_scale).to(u.arcsec / u.pixel).value
          if hasattr(pixel_scale, 'unit') else float(pixel_scale))
    return np.degrees(1.0 / d_pc) * 3600.0 / ps


def sphere_kernel(radius_px, oversample=5):
    """
    Unit-sum stamp of a uniformly emitting sphere of radius ``radius_px``
    seen in projection: surface brightness ``~ sqrt(R^2 - r^2)``, integrated
    over each pixel by ``oversample`` x ``oversample`` sub-pixels. A radius
    below half a pixel returns the 1 x 1 delta function.
    """
    R = float(radius_px)
    if R < 0.5:
        return np.ones((1, 1))
    half = int(np.ceil(R))
    n = 2 * half + 1
    sub = (np.arange(n * oversample) + 0.5) / oversample - (half + 0.5)
    yy, xx = np.meshgrid(sub, sub, indexing='ij')
    prof = np.sqrt(np.clip(R ** 2 - (xx ** 2 + yy ** 2), 0.0, None))
    kern = prof.reshape(n, oversample, n, oversample).sum(axis=(1, 3))
    return kern / kern.sum()


def stromgren_classes(q_h, cfg, distance_angular, pixel_scale):
    """
    Group stars into ``cfg.n_size_classes`` log-spaced classes of Stromgren
    radius (in pixels). Returns ``(class_index, radii_px)``: ``class_index``
    is -1 for stars with no nebula (``q_h <= 0``); ``radii_px[c]`` is the
    radius used for class c (the geometric mean of its members).
    """
    q = np.asarray(q_h, dtype=float)
    r_px = 0.5 * stromgren_diameter_pc(q, cfg.n_e_cm3) * _pixels_per_pc(distance_angular, pixel_scale)
    idx = np.full(q.size, -1, dtype=int)
    has = r_px > 0
    if not has.any():
        return idx, np.zeros(0)
    lo, hi = np.log(r_px[has].min()), np.log(r_px[has].max())
    n_c = int(cfg.n_size_classes)
    edges = np.linspace(lo, hi, n_c + 1) if hi > lo else np.array([lo, lo + 1e-9])
    idx[has] = np.clip(np.searchsorted(edges, np.log(r_px[has]), side='right') - 1, 0, len(edges) - 2)
    radii = np.array([np.exp(np.mean(np.log(r_px[idx == c]))) if np.any(idx == c) else 0.0
                      for c in range(len(edges) - 1)])
    return idx, radii
