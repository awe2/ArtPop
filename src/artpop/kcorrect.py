# Standard library
import logging
import os
import tarfile
from abc import ABC, abstractmethod
from collections import OrderedDict

# Third-party
import numpy as np

# Project
from . import MIST_PATH
from .log import logger

# The out-of-range dust warning has its own channel with its own level, so
# muting ArtPop's routine messages (simulate_catalog raises 'ArtPop Logger' to
# ERROR) does not also mute this one: an A_V outside the table's dust axes costs
# an exact per-pair build and should always be seen. Records propagate to the
# ArtPop logger's handlers; silence it explicitly, by name, if you must.
dust_logger = logging.getLogger(logger.name + '.dust')
dust_logger.setLevel(logging.WARNING)
from .filters import filter_curve_dir, load_filter_system


__all__ = ['SpectralLibrary', 'C3KLibrary', 'TremblayWDLibrary',
           'BlackbodyLibrary', 'KCorrectionGrid', 'band_offset',
           'k_correction', 'band_weights', 'extinction_curve', 'air_to_vac',
           'planck_lam', 'c3k_missing_mask', 'c3k_fill_blackbody',
           'C3K_MISSING_FLUX', 'KCORR_TABLE_VERSION',
           'spectra_path', 'kcorr_cache_dir', 'C3K_HULL', 'WD_HULL',
           'DEFAULT_Z_GRID', 'DEFAULT_AV_HOST_GRID', 'DEFAULT_AV_MW_GRID',
           'SOURCE_NAMES', 'SOURCE_C3K',
           'SOURCE_WD', 'SOURCE_BB']


# ---------------------------------------------------------------------------
# where the libraries live
# ---------------------------------------------------------------------------
# One seam, mirroring how MIST_PATH is threaded. The natural home for the
# spectral libraries is beside the MIST grids, but ~/.artpop/mist is mounted
# read-only inside the ALVISS container (claude_podman.sh:175), so the staged
# copy lives in the repo tree and one environment variable moves it. See
# data/spectral_libraries/PROVENANCE.md.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def spectra_path():
    """Root of the staged spectral libraries (``$ALVISS_SPECTRA_PATH``)."""
    env = os.environ.get('ALVISS_SPECTRA_PATH')
    if env is not None:
        return env
    for cand in (os.path.join(_REPO_ROOT, 'data', 'spectral_libraries'),
                 os.path.join(MIST_PATH, 'spectra')):
        if os.path.isdir(cand):
            return cand
    return os.path.join(_REPO_ROOT, 'data', 'spectral_libraries')


def kcorr_cache_dir():
    """Where built K grids are cached (``$ALVISS_KCORR_CACHE``)."""
    return os.environ.get(
        'ALVISS_KCORR_CACHE', os.path.join(_REPO_ROOT, 'data', 'kcorr_grids'))


# ---------------------------------------------------------------------------
# physical constants and the redshift axis
# ---------------------------------------------------------------------------
C_AA = 2.99792458e18            # speed of light, Angstrom / s
_H = 6.62607015e-27             # erg s
_C_CGS = 2.99792458e10          # cm / s
_KB = 1.380649e-16              # erg / K

# Validated to z <= 0.2 (HANDOFF_REDSHIFT.md assumption 2). There is deliberately
# no hard cap in the code: validity is library- and coverage-driven and is
# reported through the validity mask, not enforced by a magic number. Above
# z ~ 0.2 the LSST u curve samples rest-frame NUV, where model atmospheres are
# least reliable -- the flag says so.
DEFAULT_Z_GRID = np.round(np.concatenate(
    [np.linspace(0.0, 0.10, 21), np.linspace(0.11, 0.25, 15)]), 6)

# The two dust axes of the K table (F3_dust.ipynb section 4.6). Sized by
# measurement, not guessed. Delta m is concave in A_V with a curvature that is
# FLAT across the range (worst |f''| 0.042 /mag^2 for the MW screen and 0.049
# for the host screen, both in Roman F062, the widest band), so uniform spacing
# is optimal and the linear-interpolation error is h^2 |f''| / 8. Measured over
# every spectrum of the three libraries, all 14 LSST + Roman bands and z = 0 to
# 0.25 (2026-09-25): MW 0.32 mmag at h = 0.25, host 0.38 mmag at 5 nodes. The
# two add (same sign), so <= ~0.7 mmag against a 1 mmag target.
# A_V_mw extended from 0-0.5 to 0-2.0, same h = 0.25 (2026-10-05, user): the
# completeness injections take A_V_mw per object from SFD, 60 % of crowded
# (|b| 10-20 deg) patches exceed 0.5, and those are capped at 2.0. Re-measured
# the same way over 0-2.0 (finesst_feasibility notebooks/dust_axis/
# size_mw_axis.py): the MW curvature stays flat and falls slowly (worst |f''|
# 0.041 /mag^2 near 0 on a 1/120 mag grid, falling to 0.037 at 2.0), so every
# MW cell stays <= 0.32 mmag;
# the joint bilinear error over a 1/16 mag lattice of (host, MW) pairs is
# 0.70 mmag (worst cell (0-0.25, 0-0.25); 0.59 in the top MW cell), against
# the 1 mmag target. Coarser MW spacing, measured jointly the same way:
# h = 1/3 (7 nodes) 0.95 mmag, a 5 % margin; h = 0.4 (6 nodes) 1.19, fails.
# The built 5 x 9 tables against exact builds at 19 off-node pairs, every
# node, all 36 z, every band: max 0.703 mmag (Roman F062), 0.330 in LSST
# (notebooks/dust_axis/verify_mw_axis_table.py).
# A dust pair outside these ranges is computed exactly, with a warning
# (`KCorrectionGrid.for_dust`).
DEFAULT_AV_HOST_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
DEFAULT_AV_MW_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

# Hulls of the two libraries MIST composited (MIST III, Bauer et al. 2025
# s III.3). The gap at 5.5 < log g < 6.5 is real, and is why there is a
# blackbody fallback rather than an extrapolation.
C3K_HULL = dict(teff=(2500.0, 50000.0), logg=(-1.0, 5.5))
WD_HULL = dict(teff=(1500.0, 140000.0), logg=(6.5, 9.5))

# C3K cells with NO model. FSPS ships C3K on the full (log g, Teff) rectangle
# and marks cells where ATLAS12 has no model -- past the Eddington limit, and
# log g 5.5 above ~15 kK -- with a constant f_nu = 1e-33 at every wavelength,
# below FSPS's own missing threshold `tiny30` = 1e-30 (fsps sps_vars.f90;
# getspec.f90: "the flux at 5000A should never be zero unless a spec is
# missing"). 399 of 1120 cells at [Fe/H] = 0. MIST v2.5's own BC table fills
# exactly those cells with BLACKBODY bolometric corrections (its fills match
# blackbody BCs to <= 0.034 mag; measured 2026-09-28), so the K table does the
# same: an empty cell is served by a blackbody at that cell's Teff, and every
# row that touches one is flagged in `kcorr_info` (`n_c3k_bbfill`).
C3K_MISSING_FLUX = 1e-30

# Bumped whenever a table's CONTENTS change for the same arguments, so a cache
# written by older code is never loaded silently. 2: C3K's empty cells served
# by a blackbody (2026-09-28); tables before that read the 1e-33 floor as a
# flat-f_nu star (ledger S-51).
KCORR_TABLE_VERSION = 2


def c3k_missing_mask(flux, wave):
    """
    ``(n_logg, n_logt)`` True where a C3K block holds no model: FSPS's test,
    f_nu at 5000 A <= `C3K_MISSING_FLUX`.
    """
    i5 = int(np.argmin(np.abs(np.asarray(wave, dtype=float) - 5000.0)))
    return np.asarray(flux[:, :, i5], dtype=float) <= C3K_MISSING_FLUX


def c3k_fill_blackbody(flux, wave, logt):
    """
    A copy of a C3K block with every empty cell replaced by a blackbody at
    that cell's Teff (f_nu, shape only -- K depends only on shape), and the
    mask of the cells replaced. The library's own block is never modified.
    """
    missing = c3k_missing_mask(flux, wave)
    if not missing.any():
        return flux, missing
    lam = np.asarray(wave, dtype=float)
    out = np.array(flux, copy=True)
    for j in np.flatnonzero(missing.any(axis=0)):
        # planck_lam is B_lambda; B_nu ~ B_lambda lambda^2 (shape only)
        bb = planck_lam(lam, 10 ** float(logt[j])) * lam ** 2
        bb = (bb / bb.max()).astype(out.dtype)
        out[missing[:, j], j] = bb
    return out, missing


C3K_FEH_GRID = np.array([-2.50, -2.25, -2.00, -1.75, -1.50, -1.25, -1.00,
                         -0.75, -0.50, -0.25, 0.00, 0.25, 0.50])
_N_LOGG, _N_LOGT = 14, 80
_N_LAM = {'c3k_hr': 10992, 'c3k_lr': 1936}

# Source of the K value at a node, recorded alongside every interpolation so a
# fallback is never silent (test B5).
SOURCE_C3K, SOURCE_WD, SOURCE_BB = 0, 1, 2
SOURCE_NAMES = {SOURCE_C3K: 'C3K', SOURCE_WD: 'Tremblay', SOURCE_BB: 'blackbody'}


# ---------------------------------------------------------------------------
# wavelength conventions
# ---------------------------------------------------------------------------
def air_to_vac(lam):
    """
    Air to vacuum wavelengths in Angstrom (Ciddor 1996, as in Morton 2000).

    Needed because the two libraries MIST composited do **not** share a
    convention. Tremblay's readme states air; C3K's does not state either way,
    and R = 3000 was judged too coarse to settle it from a single line centre.
    It is settled by measurement instead: on ``c3k_hr`` at 5750 K / log g 4.5,
    the centroids of Ca II H&K, H-delta/gamma/beta/alpha, Mg b, Na D and
    Ca II 8542 all sit at their **vacuum** wavelengths to <= 0.26e-4
    fractional, against the +2.77e-4 air offset -- and the lambda grid is a
    clean log grid with no kink at 2000 A, which is where an air convention
    would have to switch. **C3K is vacuum; Tremblay is air.**

    Below 2000 A air and vacuum coincide by convention, so the conversion is
    the identity there rather than an extrapolation of a fit that does not
    apply.
    """
    lam = np.asarray(lam, dtype=float)
    out = lam.copy()
    m = lam > 2000.0
    s2 = (1e4 / lam[m]) ** 2
    n = (1.0 + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s2)
         + 0.0001599740894897 / (38.92568793293 - s2))
    out[m] = lam[m] * n
    return out


def planck_lam(lam, teff):
    """Planck B_lambda in cgs per Angstrom-grid wavelength (shape only matters)."""
    lam = np.asarray(lam, dtype=float)
    l_cm = lam * 1e-8
    # clip the exponent at the float64 exp limit: past it the Wien tail is zero
    # to any precision that matters, and the alternative is an overflow warning
    # on every cool node of the fallback grid
    x = np.clip(_H * _C_CGS / (l_cm * _KB * float(teff)), None, 700.0)
    return (2 * _H * _C_CGS ** 2 / l_cm ** 5) / np.expm1(x)


# ---------------------------------------------------------------------------
# extinction
# ---------------------------------------------------------------------------
def _ccm89(x, r_v=3.1):
    """A(lambda)/A(V), Cardelli, Clayton & Mathis (1989). x = 1/micron."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    ir = x < 1.1
    if ir.any():
        a[ir] = 0.574 * x[ir] ** 1.61
        b[ir] = -0.527 * x[ir] ** 1.61

    opt = (x >= 1.1) & (x < 3.3)
    if opt.any():
        y = x[opt] - 1.82
        a[opt] = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3
                  + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6
                  + 0.32999 * y ** 7)
        b[opt] = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3
                  - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6
                  - 2.09002 * y ** 7)

    # The UV branch matters here in a way it does not for a z = 0 pivot
    # wavelength: at z = 0.2 the blueshifted LSST u curve reaches 2670 A
    # (3.75 /micron), past where the optical polynomial is defined.
    uv = x >= 3.3
    if uv.any():
        xu = np.clip(x[uv], 3.3, 8.0)
        fa = np.zeros_like(xu)
        fb = np.zeros_like(xu)
        hi = xu >= 5.9
        yy = xu[hi] - 5.9
        fa[hi] = -0.04473 * yy ** 2 - 0.009779 * yy ** 3
        fb[hi] = 0.2130 * yy ** 2 + 0.1207 * yy ** 3
        a[uv] = (1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa)
        b[uv] = (-3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb)

    return a + b / float(r_v)


def extinction_curve(law='F99', r_v=3.1):
    """
    Return ``f(lam_angstrom) -> A_lambda / A_V``.

    ``'F99'`` uses `dust_extinction`'s Fitzpatrick (1999) model when it is
    installed -- the usual choice for a Milky-Way-like screen and what ALVISS
    already uses -- and falls back to CCM89 otherwise, so this module never
    hard-depends on an optional package. ``'CCM89'`` forces the analytic law,
    which is what MIST's own BC A_V axis was built with. ``'grey'`` is a
    wavelength-independent screen; it exists so a test can assert the exact
    A_x = A_V limit in both frames at any redshift.
    """
    law = str(law).upper()
    if law == 'GREY':
        return lambda lam: np.ones_like(np.asarray(lam, dtype=float))
    if law == 'CCM89':
        return lambda lam: _ccm89(1e4 / np.asarray(lam, dtype=float), r_v)
    if law != 'F99':
        raise ValueError(f"unknown extinction law {law!r}; "
                         "expected 'F99', 'CCM89' or 'grey'")
    try:
        from dust_extinction.parameter_averages import F99
        model = F99(Rv=float(r_v))
        lo, hi = model.x_range
    except Exception as exc:                            # noqa: BLE001
        logger.warning(f'dust_extinction unavailable ({exc}); '
                       'falling back to CCM89 for the extinction curve')
        return lambda lam: _ccm89(1e4 / np.asarray(lam, dtype=float), r_v)

    def _f99(lam):
        x = 1e4 / np.asarray(lam, dtype=float)
        return np.asarray(model(np.clip(x, lo * 1.000001, hi * 0.999999)))
    return _f99


# ---------------------------------------------------------------------------
# the integral
# ---------------------------------------------------------------------------
def _trapz_weights(x):
    """Trapezoid quadrature weights, so an integral becomes a dot product."""
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def band_weights(lam, trans_wave, trans, redshift=0.0, a_v_host=0.0,
                 a_v_mw=0.0, ext=None, flux_unit='f_lam', quad_weights=None):
    """
    Quadrature weights ``w`` such that the band integral is ``spectrum . w``.

    The integral is the photon-counting one,
    ``INT L_lam(l) T((1+z) l) l dl``, evaluated on the **rest-frame** grid:
    redshifting the source is exactly blueshifting the filter, because a rest
    wavelength ``l`` lands at observer wavelength ``(1+z) l``. Throughput
    outside the tabulated curve is zero, never extrapolated.

    Two dust screens are folded in, in the frame each physically acts in:

    * ``a_v_host`` attenuates at **rest-frame** ``l`` -- dust inside the host
      galaxy, which is the frame MIST's own A_V axis works in;
    * ``a_v_mw`` attenuates at **observer-frame** ``(1+z) l`` -- Milky Way
      foreground, which is what ALVISS's ``dust.A_V`` actually is.

    At z = 0 the two coincide, which is why nobody has had to distinguish them.

    Returning weights rather than a number is what makes the grid build one
    BLAS call per metallicity instead of a million trapezoid loops.
    """
    lam = np.asarray(lam, dtype=float)
    tw = _trapz_weights(lam) if quad_weights is None else quad_weights
    lam_obs = lam * (1.0 + float(redshift))

    w = np.interp(lam_obs, trans_wave, trans, left=0.0, right=0.0) * lam * tw

    if a_v_host or a_v_mw:
        if ext is None:
            ext = extinction_curve()
        # only where the filter actually transmits: outside, w is already 0
        # and A(lam) may be outside the law's validity range
        m = w != 0.0
        if a_v_host:
            w[m] *= 10 ** (-0.4 * float(a_v_host) * ext(lam[m]))
        if a_v_mw:
            w[m] *= 10 ** (-0.4 * float(a_v_mw) * ext(lam_obs[m]))

    if flux_unit == 'f_nu':
        # L_lam = f_nu * c / lam^2
        w = w * C_AA / lam ** 2
    elif flux_unit != 'f_lam':
        raise ValueError(f"flux_unit must be 'f_lam' or 'f_nu', got {flux_unit!r}")
    return w


def band_offset(lam, spectrum, trans_wave, trans, redshift=0.0, a_v_host=0.0,
                a_v_mw=0.0, ext=None, flux_unit='f_lam'):
    r"""
    The full magnitude offset a redshifted, reddened star picks up in one band.

    .. math::
        \Delta m = -2.5\log_{10}\frac{(1+z)\int L_\lambda\,
          10^{-0.4A(\lambda)}10^{-0.4A((1+z)\lambda)}\,T((1+z)\lambda)\,
          \lambda\,d\lambda}{\int L_\lambda\,T(\lambda)\,\lambda\,d\lambda}

    Computed as **one** integral, so the K-correction, the two dust screens and
    their cross term cannot be composed wrongly. The decomposition is *defined*
    from this function rather than assembled from parts::

        K      = band_offset(z, 0, 0)
        A_host = band_offset(z, A_h, 0) - K
        A_MW   = band_offset(z, 0, A_mw) - K
        cross  = band_offset(z, A_h, A_mw) - K - A_host - A_MW

    Two properties make this tractable and are what the tests pin:

    * it depends only on the **shape** of ``spectrum`` -- the normalisation
      cancels -- which is why the library must be MIST's own, since shape is
      exactly where libraries disagree;
    * the leading ``(1+z)`` is the bandwidth/energy factor. It is the term a
      plausible-looking implementation drops, and dropping it still passes
      every "K is small and smooth" sanity check while being wrong by
      ``2.5 log10(1+z)``. Test A2 exists to catch precisely that.
    """
    kw = dict(lam=lam, trans_wave=trans_wave, trans=trans, ext=ext,
              flux_unit=flux_unit, quad_weights=_trapz_weights(lam))
    num = np.dot(spectrum, band_weights(redshift=redshift, a_v_host=a_v_host,
                                        a_v_mw=a_v_mw, **kw))
    den = np.dot(spectrum, band_weights(redshift=0.0, **kw))
    if not (num > 0.0 and den > 0.0):
        return np.nan
    return -2.5 * np.log10((1.0 + float(redshift)) * num / den)


def k_correction(lam, spectrum, trans_wave, trans, redshift, flux_unit='f_lam'):
    """Dust-free special case of `band_offset`: the K-correction itself."""
    return band_offset(lam, spectrum, trans_wave, trans, redshift=redshift,
                       flux_unit=flux_unit)


# ---------------------------------------------------------------------------
# spectral libraries
# ---------------------------------------------------------------------------
class SpectralLibrary(ABC):
    """
    Backend protocol: a rectangular (log g, log Teff) grid of SEDs on a shared
    vacuum wavelength axis, plus the hull over which it is valid.

    The production backends are the two libraries MIST v2.5 itself integrated.
    That is not fussiness: a K-correction depends only on SED shape, and shape
    is exactly where libraries disagree, so substituting one injects precisely
    the systematic the differential formulation exists to cancel.
    """

    name = 'abstract'
    flux_unit = 'f_nu'
    hull = dict(teff=(0.0, np.inf), logg=(-np.inf, np.inf))

    @property
    @abstractmethod
    def wave(self):
        """Vacuum wavelengths in Angstrom."""

    @abstractmethod
    def grid(self, feh=None):
        """``(log_g, log_teff, flux)`` with ``flux`` of shape (n_g, n_t, n_lam)."""

    def in_hull(self, log_teff, log_g):
        """Boolean mask: is this (log Teff, log g) inside the library?"""
        lt = np.asarray(log_teff, dtype=float)
        lg = np.asarray(log_g, dtype=float)
        return ((lt >= np.log10(self.hull['teff'][0]))
                & (lt <= np.log10(self.hull['teff'][1]))
                & (lg >= self.hull['logg'][0])
                & (lg <= self.hull['logg'][1]))


class C3KLibrary(SpectralLibrary):
    """
    C3K (ATLAS12 atmospheres, SYNTHE spectra) as FSPS distributes it.

    The production backend: on a real 10 Gyr isochrone its hull carries
    >= 99.97% of the IMF-weighted flux and 100% of the f^2 (SBF) weight.

    The files are raw little-endian float32 with no header, no record markers
    and no shape metadata -- the axis order is a property of how FSPS wrote
    them, not something the bytes announce -- so the expected shape is asserted
    rather than inferred, and a silently different release raises instead of
    being absorbed. Values are f_nu; reading them as f_lam is a smooth ~1 mag
    colour slope, i.e. exactly the kind of wrong that looks right.

    Wavelengths are **vacuum** (see `air_to_vac` for how that was settled).
    """

    name = 'C3K'
    flux_unit = 'f_nu'
    hull = C3K_HULL

    def __init__(self, resolution='c3k_hr', a_over_fe=0.0, path=None):
        self.resolution = resolution
        self.a_over_fe = float(a_over_fe)
        self.root = os.path.join(path or spectra_path(), 'c3k')
        self._dir = os.path.join(self.root, resolution)
        self.feh_grid = C3K_FEH_GRID
        self._wave = None
        self._axes = None
        self._cache = {}

    @property
    def available(self):
        return os.path.isdir(self._dir)

    def _load_axes(self):
        if self._axes is None:
            # logt.dat / logg.dat ship only under c3k_hr but describe both
            hr = os.path.join(self.root, 'c3k_hr')
            logt = np.loadtxt(os.path.join(hr, 'logt.dat'))[:, 0]
            logg = np.loadtxt(os.path.join(hr, 'logg.dat'))
            logg = logg[:, 0] if logg.ndim == 2 else logg
            lam = np.loadtxt(os.path.join(
                self._dir, f'{self.resolution}.lambda'))[:, 0]
            if (logg.size, logt.size) != (_N_LOGG, _N_LOGT):
                raise ValueError(
                    f'C3K axes are {logg.size}x{logt.size}, expected '
                    f'{_N_LOGG}x{_N_LOGT}; this is not the release that was '
                    'validated by verify_c3k_staging.py')
            self._axes = (logg, logt)
            self._wave = lam
        return self._axes

    @property
    def wave(self):
        self._load_axes()
        return self._wave

    def _feh_token(self, feh):
        i = int(np.argmin(np.abs(self.feh_grid - feh)))
        if abs(self.feh_grid[i] - feh) > 1e-6:
            raise ValueError(f'[Fe/H] = {feh} is not a C3K grid point '
                             f'{self.feh_grid.tolist()}')
        return f'{self.feh_grid[i]:+.2f}'

    def grid(self, feh=None):
        logg, logt = self._load_axes()
        token = self._feh_token(feh)
        if token not in self._cache:
            afe = f'{self.a_over_fe:+.1f}'
            path = os.path.join(
                self._dir, f'{self.resolution}_feh{token}_afe{afe}.spec.bin')
            raw = np.fromfile(path, dtype='<f4')
            want = _N_LOGG * _N_LOGT * self.wave.size
            if raw.size != want:
                raise ValueError(f'{os.path.basename(path)}: {raw.size} floats, '
                                 f'expected {want}')
            self._cache = {token: raw.reshape(_N_LOGG, _N_LOGT, self.wave.size)}
        return logg, logt, self._cache[token]


class TremblayWDLibrary(SpectralLibrary):
    """
    Tremblay et al. (2011) 1D pure-hydrogen NLTE white-dwarf spectra.

    The log g >= 6.5 branch of MIST's composite. About 19% of an old
    isochrone's rows and under 0.1% of its flux, but the seam between this and
    C3K is visible in MIST's own BC tables and is where the naive round trip of
    HANDOFF_REDSHIFT.md s1a is worst, so it is modelled rather than
    extrapolated across.

    Two traps the format lays, both handled here: ``gravity`` in the per-model
    header is **linear g in cm/s^2**, not log g; and the wavelengths are
    **air**, so they are converted to vacuum to share C3K's convention.
    """

    name = 'Tremblay'
    flux_unit = 'f_nu'
    hull = WD_HULL

    def __init__(self, path=None):
        self.root = os.path.join(path or spectra_path(), 'tremblay_wd')
        self.tar = os.path.join(self.root, 'grid_ir.tar')
        self._grid = None

    @property
    def available(self):
        return os.path.isfile(self.tar)

    def _load(self):
        if self._grid is not None:
            return self._grid
        logg_list, per_g, wave = [], [], None
        with tarfile.open(self.tar) as tf:
            # 'grid_IR' is the directory member and also ends in '_IR', so
            # filter on isfile() rather than on the name
            members = sorted(m.name for m in tf.getmembers()
                             if m.isfile() and m.name.endswith('_IR'))
            for name in members:
                tok = tf.extractfile(name).read().decode('ascii', 'replace').split()
                n_lam = int(tok[0])          # on disk; never hardcode it
                lam = np.array(tok[1:1 + n_lam], dtype=float)
                if wave is None:
                    lam_air, wave = lam, air_to_vac(lam)
                elif not np.array_equal(lam, lam_air):
                    raise ValueError(f'{name}: wavelength grid differs from '
                                     'the first file; the seven log g files '
                                     'must share one grid')
                rest = tok[1 + n_lam:]
                teffs, fluxes, i = [], [], 0
                while i < len(rest):
                    if rest[i] != 'Effective':
                        raise ValueError(f'{name}: unexpected token {rest[i]!r}')
                    teffs.append(float(rest[i + 3]))
                    g_linear = float(rest[i + 6])       # cm/s^2, NOT log g
                    i += 10
                    fluxes.append(np.array(rest[i:i + n_lam], dtype=float))
                    i += n_lam
                logg_list.append(np.log10(g_linear))
                per_g.append((np.array(teffs), np.array(fluxes)))
        order = np.argsort(logg_list)
        logg = np.array(logg_list)[order]
        teff = per_g[order[0]][0]
        for i in order[1:]:
            if not np.array_equal(per_g[i][0], teff):
                raise ValueError('Tremblay log g files disagree on the Teff '
                                 'axis; the grid is not rectangular')
        flux = np.stack([per_g[i][1] for i in order])   # (n_g, n_t, n_lam)
        self._grid = (logg, np.log10(teff), flux, wave)
        return self._grid

    @property
    def wave(self):
        return self._load()[3]

    def grid(self, feh=None):
        logg, logt, flux, _ = self._load()
        return logg, logt, flux


class BlackbodyLibrary(SpectralLibrary):
    """
    The fallback for everything neither production library covers: the
    log g 5.5-6.5 gap between C3K's ceiling and Tremblay's floor (five
    phase-6 post-AGB rows on a real isochrone) and the > 140 kK tail.

    It is exact for a true blackbody, which is what MIST itself assumes above
    200 kK, so it is self-consistent with the tables being corrected --
    ``K_x(z, T) = BC_x(T) - BC_x(T/(1+z))`` follows with no free parameters
    because a redshifted blackbody *is* a blackbody at ``T/(1+z)``. Test A4
    asserts that identity against this backend.

    It is a fallback, not a silent one: every node served this way is recorded
    in the validity mask (test B5).
    """

    name = 'blackbody'
    flux_unit = 'f_lam'
    hull = dict(teff=(0.0, np.inf), logg=(-np.inf, np.inf))

    def __init__(self, wave=None, log_teff=None):
        self._wave = (np.geomspace(200.0, 3.0e5, 6000)
                      if wave is None else np.asarray(wave, dtype=float))
        self._logt = (np.linspace(np.log10(1000.0), np.log10(1.0e6), 60)
                      if log_teff is None else np.asarray(log_teff, dtype=float))

    available = True

    @property
    def wave(self):
        return self._wave

    def grid(self, feh=None):
        flux = np.stack([planck_lam(self._wave, 10 ** lt) for lt in self._logt])
        return np.array([0.0]), self._logt, flux[None, :, :]


# ---------------------------------------------------------------------------
# the cached grid
# ---------------------------------------------------------------------------
def _grid_offsets(lib, feh, curves, bands, z_grid, a_v_host, a_v_mw, ext):
    """
    `band_offset` evaluated over a library's whole native mesh, as one BLAS call.

    The per-node integral is a dot product once the quadrature weights are
    precomputed, so the entire (n_g x n_t) mesh for every band and redshift is a
    single matrix multiply. Measured: 0.14 s for one C3K metallicity, six bands
    and 21 redshifts. That is what makes a precomputed, committed artifact
    unnecessary -- the grid builds on demand in seconds.

    Column 0 of the weight block is the **rest-frame, dust-free** reference
    integral, i.e. the denominator, computed by the same code path as the
    numerator so that at ``z = 0`` with no dust the two are bit-identical and
    the offset is exactly 0.0.
    """
    logg, logt, flux = lib.grid(feh)
    lam = np.asarray(lib.wave, dtype=float)
    if isinstance(lib, C3KLibrary):
        # empty cells -> blackbody at the cell's Teff, as MIST v2.5 did
        flux, _ = c3k_fill_blackbody(flux, lam, logt)
    qw = _trapz_weights(lam)
    n_b, n_z = len(bands), len(z_grid)

    W = np.empty((n_b * (n_z + 1), lam.size), dtype=float)
    for i, band in enumerate(bands):
        tw, tt = curves[band]
        base = i * (n_z + 1)
        W[base] = band_weights(lam, tw, tt, 0.0, 0.0, 0.0, ext,
                               lib.flux_unit, qw)
        for j, z in enumerate(z_grid):
            W[base + 1 + j] = band_weights(lam, tw, tt, z, a_v_host, a_v_mw,
                                           ext, lib.flux_unit, qw)

    flat = np.asarray(flux, dtype=float).reshape(-1, lam.size)
    integrals = (flat @ W.T).reshape(flux.shape[0], flux.shape[1], n_b, n_z + 1)
    den = integrals[..., :1]
    num = integrals[..., 1:] * (1.0 + np.asarray(z_grid, dtype=float))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = -2.5 * np.log10(num / den)
    out[~np.isfinite(out)] = np.nan
    # stored (n_g, n_t, n_z, n_band): RegularGridInterpolator wants the sample
    # axes first and the vector-valued axis last
    return logg, logt, np.ascontiguousarray(
        out.transpose(0, 1, 3, 2)).astype(np.float32)


class KCorrectionGrid:
    """
    ``(z, log Teff, log g, [Fe/H], band) -> Delta m``, on each library's native
    mesh.

    Built on **C3K's own 14 x 80 mesh**, not MIST's 902-node BC mesh. MIST's
    41 log Teff nodes are a resampling of C3K's 80, so mirroring MIST would mean
    interpolating C3K onto it and then interpolating again onto isochrone rows
    -- two interpolations where one will do. The differential cancellation
    argument does not depend on mesh identity; it depends on the offset being a
    small, smooth function added to columns MIST already fixed.

    Three backends, in priority order, with the one used recorded per point:
    C3K (>= 99.97% of an old population's flux and 100% of its SBF weight),
    Tremblay for the white-dwarf branch, and the blackbody fallback for the
    log g 5.5-6.5 gap and the > 140 kK tail. Nothing is ever silently
    extrapolated.
    """

    _ARRAYS = ('z_grid', 'feh_grid', 'c3k_logg', 'c3k_logt', 'c3k', 'c3k_bbfill',
               'wd_logg', 'wd_logt', 'wd', 'bb_logt', 'bb',
               'av_host_grid', 'av_mw_grid')

    # tables already loaded in this process, so an isochrone per SFH bin per
    # object does not re-read a few hundred MB from disk each time
    _MEMO = OrderedDict()
    _MEMO_MAX = 6
    # dust pairs already warned about by `for_dust`, so a fallback warns once
    _WARNED = set()

    def __init__(self, bands, z_grid, feh_grid, c3k_logg, c3k_logt, c3k,
                 wd_logg, wd_logt, wd, bb_logt, bb, meta=None,
                 av_host_grid=None, av_mw_grid=None, c3k_bbfill=None):
        self.bands = list(bands)
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.feh_grid = np.asarray(feh_grid, dtype=float)
        self.c3k_logg, self.c3k_logt, self.c3k = c3k_logg, c3k_logt, c3k
        # (n_feh, n_logg, n_logt): C3K cells with no model, served by a blackbody
        self.c3k_bbfill = (np.zeros(np.shape(c3k)[:3], dtype=bool) if c3k_bbfill is None
                           else np.asarray(c3k_bbfill, dtype=bool))
        self.wd_logg, self.wd_logt, self.wd = wd_logg, wd_logt, wd
        self.bb_logt, self.bb = bb_logt, bb
        # empty = the dust pair is baked in (meta['a_v_host'], meta['a_v_mw']);
        # otherwise the table carries (..., z, A_V_host, A_V_mw, band) axes
        self.av_host_grid = np.asarray([] if av_host_grid is None else av_host_grid, dtype=float)
        self.av_mw_grid = np.asarray([] if av_mw_grid is None else av_mw_grid, dtype=float)
        self.meta = dict(meta or {})
        self._interp = {}

    @property
    def has_dust_axes(self):
        return self.av_host_grid.size > 0

    @staticmethod
    def _dust_axes(a_v_host_grid, a_v_mw_grid):
        """Resolve the requested dust axes: None, or (host axis, MW axis)."""
        if a_v_host_grid is None and a_v_mw_grid is None:
            return None
        axes = []
        for ax, default, name in ((a_v_host_grid, DEFAULT_AV_HOST_GRID, 'a_v_host_grid'),
                                  (a_v_mw_grid, DEFAULT_AV_MW_GRID, 'a_v_mw_grid')):
            ax = np.asarray(default if ax is None else ax, dtype=float)
            if ax.size < 2 or np.any(np.diff(ax) <= 0) or ax[0] != 0.0:
                raise ValueError(f'{name} must be strictly increasing, start at '
                                 f'exactly 0.0 (so zero dust is a node) and have '
                                 f'at least two values; got {ax.tolist()}')
            axes.append(ax)
        return tuple(axes)

    # -- build ------------------------------------------------------------
    @classmethod
    def build(cls, phot_system, bands=None, z_grid=None, a_v_host=0.0,
              a_v_mw=0.0, extinction_law='F99', r_v=3.1, resolution='c3k_hr',
              a_over_fe=0.0, spectra=None, curve_dir=None, verbose=False,
              a_v_host_grid=None, a_v_mw_grid=None):
        """
        Build the grid from the staged libraries.

        Dust comes in one of two forms. Either ``a_v_host`` / ``a_v_mw`` are
        baked in as scalars (seconds to build, one dust pair per table), or
        ``a_v_host_grid`` / ``a_v_mw_grid`` make the two extinctions axes of
        the table, so any pair inside them is an interpolation (one build per
        catalog: ~45x the scalar build and size at the default 5 x 9 axes). Giving
        one axis selects the default for the other (`DEFAULT_AV_HOST_GRID`,
        `DEFAULT_AV_MW_GRID`). The dust-free ``(0, 0)`` table is the common case.
        """
        axes = cls._dust_axes(a_v_host_grid, a_v_mw_grid)
        if axes is not None and (a_v_host or a_v_mw):
            raise ValueError('give either fixed a_v_host / a_v_mw or dust axes, not both')
        pairs = ([(float(h), float(m)) for h in axes[0] for m in axes[1]]
                 if axes is not None else [(a_v_host, a_v_mw)])
        z_grid = DEFAULT_Z_GRID if z_grid is None else np.asarray(z_grid, float)
        if z_grid[0] != 0.0:
            raise ValueError('z_grid must start at exactly 0.0 so that '
                             'redshift = 0 is an exact no-op')
        fs = load_filter_system(phot_system, bands=bands, curve_dir=curve_dir)
        bands = list(fs.filter_names)
        curves = {b: fs.get_trans(b) for b in bands}
        ext = extinction_curve(extinction_law, r_v)

        c3k_lib = C3KLibrary(resolution=resolution, a_over_fe=a_over_fe,
                             path=spectra)
        if not c3k_lib.available:
            raise FileNotFoundError(
                f'C3K is not staged under {c3k_lib.root}; set '
                'ALVISS_SPECTRA_PATH (see data/spectral_libraries/PROVENANCE.md)')

        def over_pairs(lib, feh):
            # one pass per dust pair; with axes, stacked to (..., z, n_h, n_m, band)
            outs = [_grid_offsets(lib, feh, curves, bands, z_grid, h, m, ext) for h, m in pairs]
            lg_, lt_ = outs[0][0], outs[0][1]
            if axes is None:
                return lg_, lt_, outs[0][2]
            arr = np.stack([o[2] for o in outs], axis=-2)
            return lg_, lt_, arr.reshape(arr.shape[:-2] + (axes[0].size, axes[1].size, arr.shape[-1]))

        blocks, fills, feh_grid = [], [], c3k_lib.feh_grid
        for feh in feh_grid:
            logg, logt, arr = over_pairs(c3k_lib, feh)
            blocks.append(arr)
            fills.append(c3k_missing_mask(c3k_lib.grid(feh)[2], c3k_lib.wave))
            if verbose:
                logger.info(f'K grid: C3K [Fe/H] = {feh:+.2f} done')
        c3k = np.stack(blocks)                       # (n_feh, n_g, n_t, n_z, [n_h, n_m,] n_b)
        c3k_bbfill = np.stack(fills)                 # (n_feh, n_g, n_t)
        c3k_logg, c3k_logt = logg, logt

        wd_lib = TremblayWDLibrary(path=spectra)
        if wd_lib.available:
            wd_logg, wd_logt, wd = over_pairs(wd_lib, None)
        else:
            logger.warning(f'Tremblay WD library not staged under '
                           f'{wd_lib.root}; the log g >= 6.5 branch will fall '
                           'back to the blackbody backend')
            wd_logg = np.zeros(0)
            wd_logt = np.zeros(0)
            dust_shape = () if axes is None else (axes[0].size, axes[1].size)
            wd = np.zeros((0, 0, len(z_grid)) + dust_shape + (len(bands),), dtype=np.float32)

        bb_lib = BlackbodyLibrary()
        _, bb_logt, bb = over_pairs(bb_lib, None)
        bb = bb[0]                                   # (n_t, n_z, [n_h, n_m,] n_b)

        meta = dict(phot_system=phot_system, resolution=resolution,
                    a_over_fe=float(a_over_fe),
                    a_v_host=None if axes is not None else float(a_v_host),
                    a_v_mw=None if axes is not None else float(a_v_mw),
                    extinction_law=str(extinction_law),
                    r_v=float(r_v), has_wd=bool(wd_lib.available))
        return cls(bands, z_grid, feh_grid, c3k_logg, c3k_logt, c3k,
                   wd_logg, wd_logt, wd, bb_logt, bb, meta,
                   av_host_grid=None if axes is None else axes[0],
                   av_mw_grid=None if axes is None else axes[1],
                   c3k_bbfill=c3k_bbfill)

    # -- persistence ------------------------------------------------------
    @staticmethod
    def cache_key(phot_system, bands=None, z_grid=None, a_v_host=0.0,
                  a_v_mw=0.0, extinction_law='F99', r_v=3.1,
                  resolution='c3k_hr', a_over_fe=0.0, curve_dir=None,
                  a_v_host_grid=None, a_v_mw_grid=None):
        """
        A key that names the grid's *contents*, not the arguments it was asked
        for. ``bands`` is resolved to the actual filter list first, so
        ``bands=None`` and an explicit list of the same filters share one cache
        entry -- otherwise a grid built by `MISTIsochrone` (which always passes
        an explicit list) would never be the one
        ``tools/build_kcorrection_grid.py`` wrote or checks.
        """
        import hashlib
        z_grid = DEFAULT_Z_GRID if z_grid is None else np.asarray(z_grid, float)
        resolved = list(load_filter_system(phot_system, bands=bands,
                                           curve_dir=curve_dir).filter_names)
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(z_grid, dtype='<f8').tobytes())
        h.update(','.join(resolved).encode())
        axes = KCorrectionGrid._dust_axes(a_v_host_grid, a_v_mw_grid)
        if axes is None:
            dust = f'_avh{a_v_host:.3f}_avmw{a_v_mw:.3f}'
        else:
            for ax in axes:
                h.update(np.ascontiguousarray(ax, dtype='<f8').tobytes())
            dust = (f'_avhax{axes[0].size}x{axes[0][-1]:.2f}'
                    f'_avmwax{axes[1].size}x{axes[1][-1]:.2f}')
        return (f'kcorr_v{KCORR_TABLE_VERSION}_{phot_system}_{resolution}_afe{a_over_fe:+.1f}'
                f'{dust}_{str(extinction_law).lower()}_rv{r_v:.2f}'
                f'_z{len(z_grid)}-{h.hexdigest()[:8]}.npz')

    def save(self, path):
        """
        Write the grid to ``.npz``, atomically.

        Staged file plus ``os.replace``, the same invariant
        `~artpop.util.fetch_mist_grid_if_needed` keeps: the destination is
        either absent or complete, never a truncated file that a later
        ``isfile`` short-circuit accepts forever.
        """
        import json
        d = {k: np.asarray(getattr(self, k)) for k in self._ARRAYS}
        d['bands'] = np.array(self.bands)
        d['meta_json'] = np.array(json.dumps(self.meta, sort_keys=True))
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        tmp = f'{path}.{os.getpid()}.part'
        try:
            with open(tmp, 'wb') as fh:
                np.savez(fh, **d)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path

    @classmethod
    def load(cls, path):
        import json
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in cls._ARRAYS if k in z.files}
            bands = [str(b) for b in z['bands']]
            meta = json.loads(str(z['meta_json']))
        return cls(bands=bands, meta=meta, **arrays)

    @classmethod
    def cached(cls, phot_system, cache_dir=None, rebuild=False, persist=True,
               **kwargs):
        """
        Load the grid for these arguments, building and caching it if absent.

        A table is also kept in memory (the last `_MEMO_MAX` of them), so every
        isochrone of a render does not re-read it. ``persist=False`` builds
        without writing to disk: `for_dust` uses it for one-off out-of-range
        dust pairs, which would otherwise leave a ~30 MB file per object.
        """
        key = cls.cache_key(phot_system, **kwargs)
        path = os.path.join(cache_dir or kcorr_cache_dir(), key)
        mtime = os.path.getmtime(path) if os.path.isfile(path) else None
        hit = cls._MEMO.get(path)
        if hit is not None and not rebuild and hit[0] == mtime:
            cls._MEMO.move_to_end(path)
            return hit[1]
        grid = None
        if mtime is not None and not rebuild:
            try:
                grid = cls.load(path)
            except Exception as exc:                    # noqa: BLE001
                logger.warning(f'could not read cached K grid {path} ({exc}); '
                               'rebuilding')
        if grid is None:
            grid = cls.build(phot_system, **kwargs)
            if persist:
                try:
                    grid.save(path)
                    mtime = os.path.getmtime(path)
                except OSError as exc:
                    logger.warning(f'could not cache K grid to {path}: {exc}')
        cls._MEMO[path] = (mtime, grid)
        cls._MEMO.move_to_end(path)
        while len(cls._MEMO) > cls._MEMO_MAX:
            cls._MEMO.popitem(last=False)
        return grid

    @classmethod
    def for_dust(cls, phot_system, a_v_host=0.0, a_v_mw=0.0,
                 a_v_host_grid=None, a_v_mw_grid=None, **kwargs):
        """
        The table that serves one dust pair, and the dust arguments to look it
        up with: ``grid, dust_kw = for_dust(...)``, then
        ``grid.offsets(..., **dust_kw)``.

        * No dust: the dust-free table (small; exactly what it always was).
        * Inside the dust axes (default `DEFAULT_AV_HOST_GRID` x
          `DEFAULT_AV_MW_GRID`): the dust-axis table, interpolated in A_V to
          <= 0.7 mmag (F3_dust.ipynb section 4.6; MW axis re-measured to 2.0
          on 2026-10-05). One build per catalog.
        * Outside them: **computed exactly** -- a table with this pair baked in,
          built in memory (~10 s) and not written to disk -- with a warning,
          because the axes were sized for the range they cover and an
          extrapolation along them is not validated.
        """
        a_h, a_m = float(a_v_host), float(a_v_mw)
        if a_h == 0.0 and a_m == 0.0:
            return cls.cached(phot_system, **kwargs), {}
        ax_h, ax_m = cls._dust_axes(
            DEFAULT_AV_HOST_GRID if a_v_host_grid is None else a_v_host_grid,
            DEFAULT_AV_MW_GRID if a_v_mw_grid is None else a_v_mw_grid)
        tol = 1e-12
        if (ax_h[0] - tol <= a_h <= ax_h[-1] + tol) and (ax_m[0] - tol <= a_m <= ax_m[-1] + tol):
            grid = cls.cached(phot_system, a_v_host_grid=ax_h, a_v_mw_grid=ax_m, **kwargs)
            return grid, dict(a_v_host=a_h, a_v_mw=a_m)
        key = (phot_system, round(a_h, 6), round(a_m, 6))
        if key not in cls._WARNED:
            cls._WARNED.add(key)
            dust_logger.warning(
                f'{phot_system}: dust (A_V_host, A_V_mw) = ({a_h:g}, {a_m:g}) is outside the K '
                f'table\'s dust axes (A_V_host in [{ax_h[0]:g}, {ax_h[-1]:g}], A_V_mw in '
                f'[{ax_m[0]:g}, {ax_m[-1]:g}]); computing this pair exactly instead of '
                'extrapolating (one ~10 s build, not cached to disk)')
        return cls.cached(phot_system, a_v_host=a_h, a_v_mw=a_m, persist=False, **kwargs), {}

    # -- interpolation ----------------------------------------------------
    def _rgi(self, which):
        from scipy.interpolate import RegularGridInterpolator
        if which not in self._interp:
            if which == 'c3k':
                pts = (self.feh_grid, self.c3k_logg, self.c3k_logt, self.z_grid)
                val = self.c3k
            elif which == 'wd':
                pts = (self.wd_logg, self.wd_logt, self.z_grid)
                val = self.wd
            else:
                pts = (self.bb_logt, self.z_grid)
                val = self.bb
            if self.has_dust_axes:
                pts = pts + (self.av_host_grid, self.av_mw_grid)
            # the stored float32 values, not a float64 copy: scipy upcasts each
            # corner exactly, so results are bit-identical at half the memory
            self._interp[which] = RegularGridInterpolator(
                pts, np.asarray(val), bounds_error=False, fill_value=np.nan)
        return self._interp[which]

    @property
    def has_wd(self):
        return self.wd_logg.size > 0

    def interpolate(self, log_teff, log_g, feh, redshift, a_v_host=None,
                    a_v_mw=None):
        """
        Per-star magnitude offsets, plus which backend served each star.

        ``a_v_host`` / ``a_v_mw`` select the dust on a table with dust axes
        (default 0.0; outside the axes raises -- `for_dust` is what computes an
        out-of-range pair exactly). On a table with the dust baked in they may
        be omitted, and if given must equal the baked values.

        Returns
        -------
        offsets : `~numpy.ndarray`, shape ``(n_star, n_band)``
            Column order is `bands`.
        info : dict
            ``source`` (0 = C3K, 1 = Tremblay, 2 = blackbody), ``c3k_bbfill``
            (bool: a C3K row whose cell touches a C3K cell with no model, served
            by a blackbody as MIST v2.5 did) with its tally ``n_c3k_bbfill``, ``fallback``
            (bool, the blackbody rows), ``feh_clipped`` (bool) and the tallies.
            Nothing is served silently: every row outside the production hull
            is flagged here, which is what test B5 asserts on.
        """
        lt = np.atleast_1d(np.asarray(log_teff, dtype=float))
        lg = np.atleast_1d(np.asarray(log_g, dtype=float))
        z = float(redshift)
        if z < self.z_grid[0] - 1e-12 or z > self.z_grid[-1] + 1e-12:
            raise ValueError(
                f'redshift {z} is outside the grid [{self.z_grid[0]}, '
                f'{self.z_grid[-1]}]; rebuild with a wider z_grid rather than '
                'extrapolating a K-correction')
        z = float(np.clip(z, self.z_grid[0], self.z_grid[-1]))

        dust = []
        if self.has_dust_axes:
            for v, ax, name in ((a_v_host, self.av_host_grid, 'a_v_host'),
                                (a_v_mw, self.av_mw_grid, 'a_v_mw')):
                v = 0.0 if v is None else float(v)
                if v < ax[0] - 1e-12 or v > ax[-1] + 1e-12:
                    raise ValueError(
                        f'{name} = {v} is outside this table\'s dust axis '
                        f'[{ax[0]}, {ax[-1]}]; KCorrectionGrid.for_dust computes an '
                        'out-of-range pair exactly')
                dust.append(float(np.clip(v, ax[0], ax[-1])))
        else:
            for v, name in ((a_v_host, 'a_v_host'), (a_v_mw, 'a_v_mw')):
                baked = float(self.meta.get(name) or 0.0)
                if v is not None and abs(float(v) - baked) > 1e-9:
                    raise ValueError(f'this table has {name} = {baked} baked in and '
                                     f'cannot serve {name} = {v}')

        def at(n):
            # the redshift and dust columns shared by every row of one backend
            return [np.full(n, z)] + [np.full(n, d) for d in dust]

        fe = np.broadcast_to(np.atleast_1d(np.asarray(feh, dtype=float)),
                             lt.shape).astype(float)
        # [Fe/H] below -2.50 exists in MIST's BC tables and not in C3K as FSPS
        # distributes it. Clip and flag; never extrapolate a spectral library.
        fe_clip = np.clip(fe, self.feh_grid[0], self.feh_grid[-1])
        feh_clipped = fe_clip != fe

        n, n_b = lt.size, len(self.bands)
        out = np.full((n, n_b), np.nan)
        source = np.full(n, SOURCE_BB, dtype=np.int8)

        in_c3k = ((lt >= self.c3k_logt[0]) & (lt <= self.c3k_logt[-1])
                  & (lg >= self.c3k_logg[0]) & (lg <= self.c3k_logg[-1]))
        if in_c3k.any():
            out[in_c3k] = self._rgi('c3k')(
                np.column_stack([fe_clip[in_c3k], lg[in_c3k], lt[in_c3k]]
                                + at(in_c3k.sum())))
            source[in_c3k] = SOURCE_C3K

        # rows whose interpolation cell touches a C3K cell with no model (served
        # by a blackbody, as MIST v2.5 did there) -- recorded, never silent
        bbfill = np.zeros(n, dtype=bool)
        if in_c3k.any() and self.c3k_bbfill.any():
            def lo(ax, x):
                return np.clip(np.searchsorted(ax, x) - 1, 0, ax.size - 2)
            f0 = lo(self.feh_grid, fe_clip[in_c3k])
            g0 = lo(self.c3k_logg, lg[in_c3k])
            t0 = lo(self.c3k_logt, lt[in_c3k])
            hit = np.zeros(in_c3k.sum(), dtype=bool)
            for df in (0, 1):
                for dg in (0, 1):
                    for dt in (0, 1):
                        hit |= self.c3k_bbfill[f0 + df, g0 + dg, t0 + dt]
            bbfill[in_c3k] = hit

        in_wd = np.zeros(n, dtype=bool)
        if self.has_wd:
            in_wd = (~in_c3k
                     & (lt >= self.wd_logt[0]) & (lt <= self.wd_logt[-1])
                     & (lg >= self.wd_logg[0]) & (lg <= self.wd_logg[-1]))
            if in_wd.any():
                out[in_wd] = self._rgi('wd')(
                    np.column_stack([lg[in_wd], lt[in_wd]] + at(in_wd.sum())))
                source[in_wd] = SOURCE_WD

        # anything left, plus anything a backend returned NaN for
        rest = ~(in_c3k | in_wd) | ~np.isfinite(out).all(axis=1)
        if rest.any():
            out[rest] = self._rgi('bb')(np.column_stack(
                [np.clip(lt[rest], self.bb_logt[0], self.bb_logt[-1])]
                + at(rest.sum())))
            source[rest] = SOURCE_BB

        bbfill &= source == SOURCE_C3K
        info = dict(source=source, fallback=(source == SOURCE_BB),
                    c3k_bbfill=bbfill, n_c3k_bbfill=int(bbfill.sum()),
                    feh_clipped=feh_clipped,
                    n_c3k=int((source == SOURCE_C3K).sum()),
                    n_wd=int((source == SOURCE_WD).sum()),
                    n_fallback=int((source == SOURCE_BB).sum()),
                    n_feh_clipped=int(feh_clipped.sum()),
                    redshift=z, bands=list(self.bands),
                    a_v_host=dust[0] if dust else self.meta.get('a_v_host'),
                    a_v_mw=dust[1] if dust else self.meta.get('a_v_mw'))
        return out, info

    def offsets(self, log_teff, log_g, feh, redshift, bands=None,
                a_v_host=None, a_v_mw=None):
        """`interpolate` as a ``{band: array}`` dict, for the bands asked for."""
        out, info = self.interpolate(log_teff, log_g, feh, redshift,
                                     a_v_host=a_v_host, a_v_mw=a_v_mw)
        want = self.bands if bands is None else list(bands)
        idx = {b: self.bands.index(b) for b in want}
        return {b: out[:, idx[b]] for b in want}, info
