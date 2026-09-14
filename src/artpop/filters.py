# Standard library
import os
import pickle

# Third-party
import numpy as np
from astropy import units as u
from astropy.table import Table

# Project
from . import data_dir, package_dir


__all__ = ['phot_system_list',
           'FilterSystem',
           'filter_curve_dir',
           'load_filter_system',
           'get_filter_names',
           'get_filter_properties',
           'phot_system_lookup',
           'load_zero_point_converter']


# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid; support both so
# the filter-property calculations work across the versions ArtPop is used with.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


# list of photometric systems with pre-calculated filter properties
#
# 'WFIRST' is MIST v1.2's grid, built on a May 2018 preliminary filter set --
# its own header calls it "WFIRST hypothetical (Vega)". 'Roman' is MIST v2.5's,
# built on the flight filter curves and delivered in AB. They are different
# systems with different filter names, not two spellings of one, so both are
# listed and neither is aliased to the other.
phot_system_list = [
    'HST_WFC3', 'HST_ACSWF', 'SDSSugriz', 'CFHTugriz', 'DECam', 'HSC',
    'JWST', 'LSST', 'UBVRIplus', 'UKIDSS', 'WFIRST', 'GALEX', 'Roman'
]


class FilterSystem(object):
    """
    Class for calculating filter parameters from throughput curves.
    The parameter definitions are taken from
    `Fukugita et al. (1996) AJ 111, 1748.
    <https://ui.adsabs.harvard.edu/abs/1996AJ....111.1748F/abstract>`_

    Parameters
    ----------
    filter_curve_files : list
        List of the file names of the filter curves.
    filter_names : str
        Names of filters. Must be the same length as ``filter_curve_files``.
    **kwargs
        Optional argumnets for `~numpy.loadtxt`, which is used to load the
        filter curve files.
    """

    def __init__(self, filter_curve_files, filter_names, **kwargs):

        self.filter_names = filter_names
        for fn, name in zip(filter_curve_files, filter_names):
            data = np.loadtxt(fn, **kwargs)
            table = Table(data=data, names=['wave', 'trans'])
            setattr(self, name, table)

    def get_trans(self, bandpass):
        """
        Get the filter throughput curve for the given bandpass.

        Returns
        -------
        lam, trans : `~numpy.ndarray`
            Wavelengths in angstroms and the (dimensionless) throughput.

        Notes
        -----
        Public because a K-correction needs ``T(lambda)`` itself, not the
        Fukugita summaries: redshifting a source is exactly blueshifting the
        filter, so the whole curve is the operand. ``_get_trans`` remains as an
        alias for the internal callers that predate this.
        """
        lam = np.asarray(getattr(self, bandpass)['wave'], dtype=float)
        trans = np.asarray(getattr(self, bandpass)['trans'], dtype=float)
        return lam, trans

    _get_trans = get_trans

    def effective_throughput(self, bandpass):
        """
        Calculate the effective throughput for the given bandpass.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        theff : float
            Effective effective throughput.
        """
        lam, trans = self._get_trans(bandpass)
        theff = _trapezoid(trans, np.log(lam))
        return theff

    def lam_eff(self, bandpass):
        """
        Calculate effective wavelength for the given bandpass.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
            leff : float
                The effective wavelength.
        """
        lam, trans = self._get_trans(bandpass)
        log_leff = _trapezoid(np.log(lam) * trans, np.log(lam))
        log_leff /= _trapezoid(trans, np.log(lam))
        leff =  np.exp(log_leff) * u.angstrom
        return leff

    def lam_pivot(self, bandpass):
        """
        Calculate pivot wavelength for the given bandpass.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
            lpivot : float
                The pivot wavelength.
        """
        lam, trans = self._get_trans(bandpass)
        lpivot = _trapezoid(lam * trans, lam)
        lpivot /= _trapezoid(trans, np.log(lam))
        lpivot = np.sqrt(lpivot) * u.angstrom
        return lpivot

    def dlam(self, bandpass):
        """
        Calculate the bandpass width.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
            width : float
                The bandpass width.
        """
        lam, trans = self._get_trans(bandpass)
        norm = _trapezoid(trans, lam)
        width = (norm / trans.max()) * u.angstrom
        return width


class ZeroPointConverter(object):
    """
    Help class for converting between AB, ST, and Vega magnitudes & colors.

    Parameters
    ----------
    zpt_table : `~astropy.table.Table`
        MIST zero point table, which can be `downloaded here
        <https://mist.science/BC_tables/zeropoints.txt>`_.
    """

    def __init__(self, zpt_table):
        for f, system, v_to_st, v_to_ab in zpt_table:
            setattr(self, f, [system, v_to_st, v_to_ab])
        self.zpt_table = zpt_table

    def _resolve(self, bandpass):
        """
        Look up a filter's zero point row, tolerating the two naming conventions.

        MIST's ``zeropoints.txt`` prefixes some filters with their photometric
        system (``WFIRST_H158``) while the isochrone columns -- and therefore
        every filter name ArtPop passes around -- are bare (``H158``). Other
        systems agree in both places (``LSST_u``). Where they disagree the lookup
        used to raise `AttributeError`, which
        `~artpop.stars.isochrones.MISTIsochrone` caught and turned into a zero
        offset, silently leaving those magnitudes in their native system even
        when the caller asked for AB.

        The bare name is tried FIRST and the prefixed one only as a fallback:
        going the other way would turn ``LSST_u`` into ``LSST_LSST_u``. Resolving
        forwards, via `phot_system_lookup`, is unambiguous -- stripping prefixes
        instead would collide ``SDSS_u``, ``CFHT_u`` and ``LSST_u`` onto ``u``.
        """
        try:
            return getattr(self, bandpass)
        except AttributeError:
            pass
        try:
            system = phot_system_lookup(bandpass)
        except KeyError:
            raise KeyError(
                f"no zero point entry for filter '{bandpass}', and it is not in "
                "the photometric-system lookup table either")
        prefixed = f'{system}_{bandpass}'
        try:
            return getattr(self, prefixed)
        except AttributeError:
            raise KeyError(
                f"no zero point entry for filter '{bandpass}' (tried "
                f"'{bandpass}' and '{prefixed}'); is it missing from "
                "zeropoints.txt?")

    def to_vega(self, bandpass):
        """
        Convert to Vega magnitudes.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        system, v_to_st, v_to_ab = self._resolve(bandpass)
        if system == 'Vega':
            zpt_convert = 0.
        elif system == 'AB':
            zpt_convert = -v_to_ab
        return zpt_convert

    def to_ab(self, bandpass):
        """
        Convert to AB magnitudes.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        system, v_to_st, v_to_ab = self._resolve(bandpass)
        if system == 'AB':
            zpt_convert = 0.
        elif system == 'Vega':
            zpt_convert = v_to_ab
        return zpt_convert

    def to_st(self, bandpass):
        """
        Convert to ST magnitudes.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        system, v_to_st, v_to_ab = self._resolve(bandpass)
        if system == 'AB':
            zpt_convert = v_to_st - v_to_ab
        elif system == 'Vega':
            zpt_convert = v_to_st
        return zpt_convert

    def color_to_vega(self, blue, red):
        """
        Convert to Vega colors.

        Parameters
        ----------
        blue : str
            The name of the blue filter.
        red : str
            The name of the red filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        blue_convert = self.to_vega(blue)
        red_convert = self.to_vega(red)
        return blue_convert - red_convert

    def color_to_ab(self, blue, red):
        """
        Convert to AB colors.

        Parameters
        ----------
        blue : str
            The name of the blue filter.
        red : str
            The name of the red filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        blue_convert = self.to_ab(blue)
        red_convert = self.to_ab(red)
        return blue_convert - red_convert

    def color_to_st(self, blue, red):
        """
        Convert to ST colors.

        Parameters
        ----------
        blue : str
            The name of the blue filter.
        red : str
            The name of the red filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        blue_convert = self.to_st(blue)
        red_convert = self.to_st(red)
        return blue_convert - red_convert


def get_filter_names(phot_system=None):
    """
    Get MIST photometric systems and filter names.

    Parameters
    ----------
    phot_system : str or None
        The desired photometric system.

    Returns
    -------
    mist_filter_names : list or dict
        If ``phot_system`` given, then a list of filter names is returned. If
        ``phot_system`` is ``None``, then a dict of all photometric systems
        and filters is returned.
    """
    fn = os.path.join(data_dir, 'mist_filter_names.pkl')
    pickle_in = open(fn, 'rb')
    mist_filter_names = pickle.load(pickle_in)
    pickle_in.close()
    if phot_system is not None:
        if type(phot_system) == str:
            phot_system = [phot_system]
        names = []
        for p in phot_system:
            names.extend(mist_filter_names[p])
        mist_filter_names = names
    return mist_filter_names


def get_filter_properties():
    """Return astropy Table with the filter lam_eff and dlam values."""
    pkl_fn = os.path.join(data_dir, 'filter_properties.pkl')
    with open(pkl_fn, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        props = Table(data)
        props['dlam'] *= u.angstrom
        props['lam_eff'] *= u.angstrom
    return props


def phot_system_lookup(filter_name=None):
    """
    Lookup the photometric system name associated with a given filter name.

    Parameters
    ----------
    filter_name : str, optional
        Filter name to lookup.

    Returns
    -------
    lookup : dict or str
        If ``filter_name`` is ``None``, a dictionary with the filter names as
        keywords and the photometric systems as values. If ``filter_name`` is
        not ``None``, its photometric system is returned.
    """
    fn = os.path.join(data_dir, 'phot_system_lookup.pkl')
    pickle_in = open(fn, 'rb')
    lookup = pickle.load(pickle_in)
    pickle_in.close()
    if filter_name is not None:
        lookup = lookup[filter_name]
    return lookup


def load_zero_point_converter():
    """
    Create and return a `~artpop.filters.ZeroPointConverter` object.
    """
    from astropy.io import ascii
    fn = os.path.join(data_dir, 'zeropoints.txt')
    table= ascii.read(fn)
    return ZeroPointConverter(table)


# ---------------------------------------------------------------------------
# transmission curves on disk
# ---------------------------------------------------------------------------
# `filter_curves/` lives at the repo root, where `tools/build_filter_data.py`
# walks it. Nothing else in the package ever needed it, so it is not installed
# (setup.py ships only `data/*`) -- and the failure mode of that is a silent
# absence of curves at import time, not an error. The search order below makes
# the repo layout work today and lets an install work as soon as the curves are
# synced under `data/filter_curves/`; ARTPOP_FILTER_CURVES overrides both.
_CURVE_SEARCH = [
    lambda: os.environ.get('ARTPOP_FILTER_CURVES'),
    lambda: os.path.join(data_dir, 'filter_curves'),
    lambda: os.path.abspath(os.path.join(
        package_dir, os.pardir, os.pardir, 'filter_curves')),
]


def filter_curve_dir():
    """
    Directory holding the ``<system>/<band>.csv`` transmission curves.

    Raises
    ------
    FileNotFoundError
        Naming every location searched, because "no curves" must never read as
        "no curves needed".
    """
    tried = []
    for get in _CURVE_SEARCH:
        cand = get()
        if cand is None:
            continue
        tried.append(cand)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        'no filter_curves directory found; searched ' + ', '.join(tried)
        + '. Set ARTPOP_FILTER_CURVES to point at it.')


def load_filter_system(phot_system, bands=None, curve_dir=None):
    """
    Build a `FilterSystem` for one photometric system from its shipped curves.

    Parameters
    ----------
    phot_system : str
        A member of `phot_system_list`.
    bands : list of str, optional
        Restrict to these filters. Default: every curve in the directory,
        sorted, which is *not* the ``filter_properties`` order -- pass ``bands``
        when order matters.
    curve_dir : str, optional
        Override `filter_curve_dir`.

    Returns
    -------
    fs : `~artpop.filters.FilterSystem`
    """
    if phot_system not in phot_system_list:
        raise ValueError(f'{phot_system} is not a valid photometric system')
    root = os.path.join(curve_dir or filter_curve_dir(), phot_system)
    if not os.path.isdir(root):
        raise FileNotFoundError(f'no transmission curves for {phot_system} '
                                f'under {root}')
    have = {f[:-4] for f in os.listdir(root) if f.endswith('.csv')}
    # MIST's own column order, so a caller that reads results positionally sees
    # the same order `filter_properties` uses. Alphabetical would put LSST as
    # g,i,r,u,y,z -- correct but unreadable, and a trap for positional readers.
    try:
        canonical = [f for f in get_filter_names()[phot_system] if f in have]
    except Exception:                                   # noqa: BLE001
        canonical = []
    names = canonical + sorted(have - set(canonical))
    if bands is not None:
        missing = [b for b in bands if b not in names]
        if missing:
            raise FileNotFoundError(f'no transmission curve for {missing} '
                                    f'under {root}')
        names = list(bands)
    files = [os.path.join(root, n + '.csv') for n in names]
    return FilterSystem(files, names, delimiter=',', skiprows=1)
