# Standard library
import os, abc
import pickle
import numpy as np
from copy import deepcopy

# Third-party
from astropy.table import Table
from astropy import units as u

# Project
from .. import MIST_PATH
from ..util import check_random_state, check_units
from ..log import logger
from .imf import sample_imf, build_galaxy, IMFIntegrator
from .isochrones import MISTIsochrone


__all__ = ['SSP', 'MISTSSP', 'constant_sb_stars_per_pix']


class StellarPopulation(metaclass=abc.ABCMeta):
    """
    Stellar population base class.

    Parameters
    ----------
    distance : float or `~astropy.units.Quantity`, optional
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`. Default distance is 10 `~astropy.units.pc`
        (i.e., the mags are in absolute units).
    a_lam : float or dict, optional
        Magnitude(s) of extinction. If float, the same extinction will be applied
        to all bands. If dict, the keys must be the same as the observational filters.
    imf : str or dict
        Which IMF to use, if str then must be one of pre-defined: 'kroupa',
        'scalo' or 'salpeter'. Can also specify custom (broken) power law as dict,
        which must contain either 'a' as a Float (describing the slope of a
        single power law) or 'a' (a list with 3 elements describing the slopes
        of a broken power law) and 'b' (a list  with 2 elements describing the
        locations of the breaks).
    """

    def __init__(self, distance=10.0 * u.pc, a_lam=0.0, imf='kroupa',
                 imf_kw=None, redshift=0.0):
        self.imf = imf
        self.imf_kw = {} if imf_kw is None else imf_kw
        self.distance = check_units(distance, 'Mpc')
        # `distance` is the LUMINOSITY distance and `redshift` is free of it:
        # nothing here derives one from the other, in either direction, so
        # mu(z) stays something data can constrain rather than something this
        # simulator assumed. See `distance_angular` for the single place the
        # two are related, and why that is a metric identity and not a
        # cosmology.
        self.redshift = float(redshift)
        self.a_lam = a_lam

    def build_pop(self, num_stars=None, **kwargs):
        """Build stellar population."""
        return NotImplementedError()

    @property
    def dist_mod(self):
        """
        The distance modulus, ``5 log10(D_L / 10 pc)``.

        Purely a function of the luminosity distance. The band-shift term is
        **not** here -- it is per star, not per galaxy, so it lives on the
        isochrone's magnitude columns (`MISTIsochrone._apply_band_offsets`).
        Folding a galaxy-averaged K in here instead would look reasonable and
        would bias SBF distances by ~10%.
        """
        return 5 * np.log10(self.distance.to('pc').value) - 5

    def distance_angular(self, eta=1.0):
        """
        Angular-diameter distance, ``D_A = D_L / [(1 + z)^2 eta^2]``.

        This is **Etherington's distance-duality relation** -- a consequence of
        photon conservation in any metric theory, not a cosmological model. It
        does not smuggle in a cosmology and it does not constrain ``D_L`` given
        ``z``. It is the only place in this design where two of the three free
        parameters (``distance``, ``redshift``, ``r_eff``) are related, which
        is why it is a method with a visible ``eta`` rather than a hidden
        conversion.

        Parameters
        ----------
        eta : float, optional
            Distance-duality parameter, so a violation can itself be probed.
            Default: 1.0, i.e. duality holds.

        Returns
        -------
        d_a : `~astropy.units.Quantity`
            The angular-diameter distance. At ``z = 0`` this is exactly
            ``distance``, bit for bit.
        """
        if self.redshift == 0.0 and eta == 1.0:
            return self.distance
        return self.distance / ((1.0 + self.redshift) ** 2 * float(eta) ** 2)

    @property
    def num_pops(self):
        """
        The number of simple stellar populations that composes this pop.
        """
        return 1 if type(self.isochrone) != list else len(self.isochrone)

    @property
    def total_initial_live_mass(self):
        """Total initial stellar mass in solar units."""
        return self.initial_masses.sum() * u.M_sun

    @property
    def num_stars(self):
        """Number stars in population."""
        return len(self.star_masses)

    @property
    def abs_mag_table(self):
        """Absolute magnitudes in a `~astropy.table.Table` object."""
        return Table(self.abs_mags)

    @property
    def mag_table(self):
        """Apparent magnitudes in a `~astropy.table.Table` object."""
        _mags = {}
        for filt in self.filters:
            _mags[filt] = self.star_mags(filt)
        return Table(_mags)

    @property
    def a_lam(self):
        """Return extinction dictionary."""
        return self._a_lam

    @a_lam.setter
    def a_lam(self, a_lam):
        """
        Set the extinction magnitude dictionary. Warn user if the
        given a_lam filter is not in the filter set.

        Parameters
        ----------
        a_lam : float or dict, optional
            Magnitude(s) of extinction. If float, the same extinction will
            be applied to all bands. If dict, the keys must be the same as
            the observational filters.
        """
        if isinstance(a_lam, dict):
            self._a_lam = {filt: 0.0 for filt in self.filters}
            for filt in a_lam.keys():
                if filt in self.filters:
                    self._a_lam[filt] = a_lam[filt]
                else:
                    logger.warning(
                        f'{filt} not in filter set. '
                        f'Will ignore a_lam for this filter.'
                    )
        else:
            self._a_lam = {filt: a_lam for filt in self.filters}

    def to_pickle(self, file_name):
        """Pickle stellar population object."""
        pkl_file = open(file_name, 'wb')
        pickle.dump(self, pkl_file)
        pkl_file.close()

    @staticmethod
    def from_pickle(file_name):
        """Load pickle of stellar population object."""
        pkl_file = open(file_name, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data

    def _remnants_factor(self, **kwargs):
        """
        Correction factor to account for stellar remnants in the final mass.
        The mass in luminous stars is given by M_total * factor.
        """
        m_without_remnants = self.isochrone.ssp_surviving_mass(
            self.imf, add_remnants=False, **kwargs)
        m_with_remnants = self.isochrone.ssp_surviving_mass(
            self.imf, add_remnants=True, **kwargs)
        factor = m_without_remnants / m_with_remnants
        return factor

    def copy(self):
        return deepcopy(self)

    def set_distance(self, distance):
        """
        Change the distance to the stellar population.

        Parameters
        ----------
        distance : float or `~astropy.units.Quantity`
            Distance to source. If float is given, the units are assumed
            to be `~astropy.units.Mpc`.
        """
        self.distance = check_units(distance, 'Mpc')

    def set_redshift(self, redshift):
        """
        Change the redshift of the stellar population.

        Unlike `set_distance`, this is not a scalar swap: the band-shift
        correction is per star and is baked into the isochrone's magnitude
        columns, so the isochrone is recomputed and every star's absolute
        magnitude is re-derived from it.

        Raises
        ------
        Exception
            If the population has an integrated (smooth) component, i.e. if a
            ``mag_limit`` was given. Both the bright/faint row split and the
            analytic sums over the faint rows are functions of ``z``, so
            changing it after the fact would leave a population whose sampled
            stars had been corrected and whose smooth component had not --
            precisely the kind of silent inconsistency this correction exists
            to avoid. Rebuild with ``redshift=`` instead.
        """
        z = float(redshift)
        if z == self.redshift:
            return
        if getattr(self, 'has_integrated_component', False):
            raise Exception(
                'cannot set_redshift on a population with an integrated '
                'component: the mag_limit row split and the smooth-component '
                'moments both depend on z. Rebuild the population with '
                f'redshift={z} instead.')
        iso = self.isochrone
        if not hasattr(iso, 'set_redshift'):
            raise Exception(f'{type(iso).__name__} cannot recompute its '
                            'magnitudes at a new redshift')
        iso.set_redshift(z)
        self.redshift = z
        for filt in self.filters:
            self.abs_mags[filt] = iso.interpolate(filt, self.initial_masses)

    def star_mags(self, bandpass, select=None):
        """
        Get the stellar apparent magnitudes.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        select : `~numpy.ndarray`, optional
            Boolean mask for selecting stars (True for stars to include).

        Returns
        -------
        mags : `~numpy.ndarray`
            The stellar apparent magnitudes in the given bandpass.
        """
        mags = self.abs_mags[bandpass] + self.dist_mod + self.a_lam[bandpass]
        if select is not None:
            mags = mags[select]
        return mags

    def mag_integrated_component(self, bandpass):
        """
        Get the magnitude of the integrated component of the population if
        it exists.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).

        Returns
        -------
        mag : float
            The integrated magnitude if it exists. Otherwise None is returned.
        """
        if hasattr(self, 'integrated_abs_mags'):
            mag = self.integrated_abs_mags[bandpass] + self.dist_mod + self.a_lam[bandpass]
        else:
            mag = None
        return mag

    def sbf_mag(self, bandpass):
        """
        Calculate the apparent SBF magnitude of the stellar population.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).

        Returns
        -------
        mbar : float
            The apparent SBF magnitude of the stellar population in the given
            bandpass.
        """
        integrated = self.mag_integrated_component(bandpass)
        if integrated is not None:
            f_int = 10**(-0.4*integrated)
            log_dddd = 4 * np.log10(self.distance.to('cm').value)
            # _integrated_log_lumlum is stored unextincted; extinction must be
            # applied here so that mbar dims by +a_lam like every flux-like
            # quantity (matching f_int, which inherits a_lam through
            # mag_integrated_component). Applied at call time so an a_lam
            # assigned after the population is built is still honored.
            ff_int = 10**(self._integrated_log_lumlum[bandpass] - log_dddd
                          - 0.8 * self.a_lam[bandpass])
        else:
            f_int = 0.0
            ff_int = 0.0
        f_i = 10**(-0.4 * (self.star_mags(bandpass)))
        ff = np.sum(f_i**2) + ff_int
        f = np.sum(f_i) + f_int
        mbar = -2.5 * np.log10(ff / f)
        return mbar

    def integrated_color(self, blue, red, select=None):
        """
        Calculate the population's integrated color.

        Parameters
        ----------
        blue : str
            The blue bandpass. Must be a filter in the
            given photometric system(s).
        red : str
            The red bandpass. Must be a filter in the
            given photometric system(s).
        select : `~numpy.ndarray`, optional
            Boolean mask for selecting stars (True for stars to include).

        Returns
        -------
        color : float
            The integrated color.
        """
        blue_int = self.mag_integrated_component(blue)
        if select is None and blue_int is not None:
            f_blue_int = 10**(-0.4*blue_int)
            f_red_int = 10**(-0.4*self.mag_integrated_component(red))
        else:
            f_blue_int = 0.0
            f_red_int = 0.0
        blue_mag = self.star_mags(blue, select)
        F_blue = np.sum(10**(-0.4 * blue_mag)) + f_blue_int
        red_mag = self.star_mags(red, select)
        F_red = np.sum(10**(-0.4 * red_mag)) + f_red_int
        color = -2.5 * np.log10(F_blue / F_red)
        return color

    def mean_mag(self, bandpass, select=None):
        """
        Calculate the population's mean magnitude.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        select : `~numpy.ndarray`, optional
            Boolean mask for selecting stars (True for stars to include).

        Returns
        -------
        mag : float
            The mean magnitude in the given bandpass.
        """
        integrated = self.mag_integrated_component(bandpass)
        if select is None and integrated is not None:
            f_int = 10**(-0.4*integrated)
            n_int = self.num_stars_integrated
        else:
            f_int = 0.0
            n_int = 0.0
        mags = self.star_mags(bandpass, select)
        mean_flux = ((10**(-0.4*mags)).sum() + f_int) / (len(mags) + n_int)
        mag = -2.5 * np.log10(mean_flux)
        return mag

    def total_mag(self, bandpass, select=None):
        """
        Calculate the population's total magnitude.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        select : `~numpy.ndarray`, optional
            Boolean mask for selecting stars (True for stars to include).

        Returns
        -------
        mag : float
            The total magnitude in the given bandpass.
        """
        integrated = self.mag_integrated_component(bandpass)
        if select is None and integrated is not None:
            f_int = 10**(-0.4*integrated)
        else:
            f_int = 0.0
        mags = self.star_mags(bandpass, select)
        total_flux = (10**(-0.4*mags)).sum() + f_int
        mag = -2.5 * np.log10(total_flux)
        return mag


class SSP(StellarPopulation):
    """
    Generic Simple Stellar Population (SSP).

    .. note::
        You must give `total_mass` *or* `num_stars`.

    Parameters
    ----------
    isochrone : `~artpop.stars.Isochrone`
        Isochrone object.
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    total_mass : float or `~astropy.units.Quantity` or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. This
        mass accounts for stellar remnants when ``add_remnants = True``, which
        means the actual sampled mass will be less than the given value. If
        float is given, the units are assumed to be solar masses.
    distance : float or `~astropy.units.Quantity`, optional
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`. Default distance is 10 `~astropy.units.pc`.
    a_lam : float or dict, optional
        Magnitude(s) of extinction. If float, the same extinction will be applied
        to all bands. If dict, the keys must be the same as the observational filters.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    imf : str or dict
        Which IMF to use, if str then must be one of pre-defined: 'kroupa',
        'scalo' or 'salpeter'. Can also specify custom (broken) power law as dict,
        which must contain either 'a' as a Float (describing the slope of a
        single power law) or 'a' (a list with 3 elements describing the slopes
        of a broken power law) and 'b' (a list  with 2 elements describing the
        locations of the breaks).
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    mass_tolerance : float, optional
        Tolerance in the fractional difference between the input mass and the
        final mass of the population. The parameter is only used when
        `total_mass` is given.
    add_remnants : bool, optional
        If True (default), apply scaling factor to total mass to account for
        stellar remnants in the form of white dwarfs, neutron stars,
        and black holes.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, isochrone, num_stars=None, total_mass=None,
                 distance=10*u.pc, a_lam=0.0, mag_limit=None, mag_limit_band=None,
                 imf='kroupa', imf_kw=None, mass_tolerance=0.01,
                 add_remnants=True, random_state=None, sampling='lf'):
        if sampling not in ('lf', 'mass'):
            raise ValueError(f"sampling must be 'lf' or 'mass', got {sampling!r}")
        self.isochrone = isochrone
        self.filters = isochrone.filters
        # taken from the isochrone rather than accepted as a second argument,
        # so the population's z and the z its magnitudes were corrected at
        # cannot disagree
        super(SSP, self).__init__(distance=distance, a_lam=a_lam, imf=imf,
                                  imf_kw=imf_kw,
                                  redshift=getattr(isochrone, 'redshift', 0.0))
        self.mag_limit = mag_limit
        self.mag_limit_band = mag_limit_band
        self.sampling = sampling
        self.rng = check_random_state(random_state)
        self.build_pop(num_stars, total_mass, mass_tolerance, add_remnants)
        self._r = {'M_star': f'{self.total_mass.value:.2e} M_sun'}

    def build_pop(self, num_stars=None, total_mass=None, mass_tolerance=0.01,
                  add_remnants=True):
        """
        Build the stellar population. You must give `total_mass`
        *or* `num_stars` as an argument.

        Parameters
        ----------
        num_stars : int or `None`
            Number of stars in source. If `None`, then must give `total_mass`.
        total_mass : float or `~astropy.units.Quantity` or `None`
            Stellar mass of the source. If `None`, then must give `num_stars`.
            This mass accounts for stellar remnants when ``add_remnants = True``
            which means the actual sampled mass will be less than the given
            value. If float is given, the units are assumed to be solar masses.
        mass_tolerance : float, optional
            Tolerance in the fractional difference between the input mass and
            the final mass of the population. The parameter is only used when
            `total_mass` is given.
        add_remnants : bool, optional
            If True (default), apply scaling factor to total mass to account
            for stellar remnants in the form of white dwarfs, neutron stars,
            and black holes.
        """

        # Luminosity-function (isochrone-row) sampling is the default when a
        # mag_limit is given: the sampled set is selected by ROW BRIGHTNESS,
        # which the limit actually constrains, rather than by a mass threshold,
        # which it does not (on an old isochrone the entire white-dwarf cooling
        # track sits above the mass threshold but below the magnitude limit,
        # so the mass path allocates millions of stars to render a handful).
        # The mag_limit=None path below is unchanged from stock ArtPop.
        if self.mag_limit is not None and getattr(self, 'sampling', 'lf') == 'lf':
            return self._build_pop_lf(num_stars, total_mass, add_remnants)

        # get isochrone object and info
        m_min, m_max = self.isochrone.m_min, self.isochrone.m_max
        imf_kw = self.imf_kw.copy()
        iso = self.isochrone
        imfint = IMFIntegrator(self.imf, m_min=m_min, m_max=m_max)
        remnants_factor = self._remnants_factor() if add_remnants else 1.0

        # calculate the fraction of stars we will sample
        m_lim, f_num_sampled, f_mass_sampled = self.sample_fraction(
            self.mag_limit,
            self.mag_limit_band
        )

        # the limiting mass is min mass if 100% of stars will be sampled
        self.has_integrated_component = m_lim != m_min

        m_min = m_lim
        self.sampled_mass_lower_limit = m_lim
        self.frac_num_sampled = f_num_sampled
        self.frac_mass_sampled = f_mass_sampled

        if num_stars is not None:

            if f_num_sampled > 0:
                # sample imf
                num_stars_sample = int(num_stars * f_num_sampled)
                self.initial_masses = sample_imf(
                    num_stars_sample, m_min=m_min, m_max=m_max, imf=self.imf,
                    random_state=self.rng, imf_kw=imf_kw)
            else:
                # fully smooth: no star qualifies for individual sampling
                self.initial_masses = np.array([])

            # star masses are interpolated from "actual" mass
            self.star_masses = iso.interpolate('mact', self.initial_masses)

            # will be < num_stars if f_num_sampled < 1.0.
            self.num_stars_integrated = int(num_stars - len(self.star_masses))
            self.sampled_mass = self.star_masses.sum()

        elif total_mass is not None:

            total_mass = check_units(total_mass, 'Msun').to('Msun').value

            # calculate fraction of mass that remains after mass loss
            mass_loss = iso.ssp_surviving_mass(
                imf=self.imf, m_min=iso.m_min, m_max=iso.m_max,
                add_remnants=False)

            # we sample less mass than total_mass to account for
            # stellar remnants and the sample fraction
            sampled_mass = total_mass * remnants_factor * f_mass_sampled

            # we increase the sampled mass to account for mass loss
            sampled_mass /= mass_loss

            if f_num_sampled > 0:
                # sample initial masses
                mean_mass = imfint.m_integrate(m_min, m_max)
                mean_mass /= imfint.integrate(m_min, m_max)
                num_stars_iter = int(mass_tolerance * sampled_mass / mean_mass)
                self.initial_masses = build_galaxy(
                    sampled_mass, m_min=m_min, m_max=m_max, imf=self.imf,
                    random_state=self.rng, num_stars_iter=num_stars_iter,
                    **imf_kw)

                # star masses are interpolated from "actual" mass
                self.star_masses = iso.interpolate('mact', self.initial_masses)
                self.sampled_mass = self.star_masses.sum()

                # calculate approximate number of stars in integrated component
                factor = (1 - f_num_sampled) / f_num_sampled
                self.num_stars_integrated = int(self.num_stars * factor)
            else:
                # fully smooth: every star lives in the integrated component.
                # The stock expressions above divide by f_num_sampled and
                # cannot represent this case.
                self.initial_masses = np.array([])
                self.star_masses = np.array([])
                self.sampled_mass = 0.0
                mean_all = (imfint.m_integrate(iso.m_min, iso.m_max)
                            / imfint.integrate(iso.m_min, iso.m_max))
                self.num_stars_integrated = int(
                    total_mass * remnants_factor / mass_loss / mean_all)

        else:

            raise Exception('you must give total mass *or* number of stars')

        self.abs_mags = {}
        for filt in self.filters:
            self.abs_mags[filt] = iso.interpolate(filt, self.initial_masses)

        # update masses and mags if there is an integrated component
        if self.has_integrated_component:

            # find evolved stars that are fainter than mag_limit
            sampled_mags = self.abs_mags[self.mag_limit_band] + self.dist_mod
            evolved_faint = sampled_mags > self.mag_limit

            evolved_mags = {}
            for filt in self.filters:
                evolved_mags[filt] = self.abs_mags[filt][evolved_faint]
                self.abs_mags[filt] = self.abs_mags[filt][~evolved_faint]

            # calculate evolved mass and update integrated star count
            evolved_mass = self.star_masses[evolved_faint].sum()
            self.num_stars_integrated += evolved_faint.sum()

            # update initial and sampled masses
            self.initial_masses = self.initial_masses[~evolved_faint]
            self.star_masses = self.star_masses[~evolved_faint]
            self.sampled_mass = self.star_masses.sum()

            # calculate normalized IMF weights
            w = iso.imf_weights(self.imf, m_max_norm=m_max, norm_type='number')
            _, arg = iso.nearest_mini(m_lim)

            # update total mass
            num_stars = self.num_stars_integrated + self.num_stars
            _mass = num_stars * np.sum(iso.mact[:arg] * w[:arg])
            _mass += evolved_mass
            self.total_mass = (self.sampled_mass + _mass) / remnants_factor

            # updated integrated absolute magnitudes and luminosity variances
            self.integrated_abs_mags = {}
            self._integrated_log_lumlum = {}
            for filt in iso.filters:
                mag = iso.mag_table[filt]
                flux  = num_stars * np.sum(10**(-0.4 * mag[: arg]) * w[: arg])
                flux += np.sum(10**(-0.4 * evolved_mags[filt]))
                self.integrated_abs_mags[filt] = -2.5 * np.log10(flux)
                ff = num_stars * np.sum(10**(-0.8 * mag[: arg]) * w[: arg])
                ff += np.sum(10**(-0.8 * evolved_mags[filt]))
                log_dddd = 4 * np.log10((10 * u.pc).to('cm').value)
                self._integrated_log_lumlum[filt] = np.log10(ff) + log_dddd
        else:
            # calculate total_mass with stellar remnants
            self.total_mass = self.sampled_mass / remnants_factor

        self.live_star_mass = self.total_mass * remnants_factor
        self.ssp_labels = np.ones(len(self.star_masses), dtype=int)

        self.total_mass *= u.Msun
        self.sampled_mass *= u.Msun
        self.live_star_mass *= u.Msun

        for attr in ['eep', 'log_L', 'log_Teff']:
            if hasattr(iso, attr):
                if getattr(iso, attr) is not None:
                    vals_interp = iso.interpolate(attr, self.initial_masses)
                    setattr(self, attr, vals_interp)

        # row-sampling bookkeeping (populated by the LF path; trivial here)
        n_rows = len(np.asarray(iso.mini))
        self.sampled_row_mask = (np.ones(n_rows, dtype=bool)
                                 if self.mag_limit is None else None)
        self.row_counts = None
        self.n_total_expected = None

    def _lf_row_split(self):
        """
        Number-normalized IMF weight per isochrone row and the mask of rows
        brighter than ``mag_limit``.

        The comparison uses unextincted apparent magnitudes
        (``mag_table + dist_mod``, no ``a_lam``): ``mag_limit`` is an
        intrinsic apparent-magnitude limit, consistent with
        ``sample_fraction`` and with callers that assign extinction after the
        population is built.

        Returns
        -------
        w : `~numpy.ndarray`
            IMF weight per row (fraction of all stars formed in
            ``[m_min, m_max]`` that live in that row; sums to 1).
        bright : `~numpy.ndarray` of bool
            True for rows at or brighter than ``mag_limit``.
        """
        if self.mag_limit_band is None:
            raise Exception('Must give bandpass of limiting magnitude.')
        iso = self.isochrone
        w = iso.imf_weights(self.imf, m_max_norm=iso.m_max,
                            norm_type='number')
        mags = np.asarray(iso.mag_table[self.mag_limit_band]) + self.dist_mod
        bright = mags <= self.mag_limit
        return w, bright

    def _build_pop_lf(self, num_stars=None, total_mass=None,
                      add_remnants=True):
        """
        Build a mag-limited population by sampling the luminosity function.

        Stars brighter than ``mag_limit`` are drawn per isochrone row with
        probability proportional to the row's IMF weight (per-row Poisson
        counts, realized as a Poisson total plus a categorical row draw), and
        each star's initial mass is drawn uniformly within its row's half-bin
        so magnitudes are continuous. Every row fainter than the limit —
        including the entire white-dwarf cooling track — goes into the smooth
        component analytically. The sampled and smooth sets are therefore
        exact complements in row space: nothing is allocated and then culled,
        the number of stars sampled is the number rendered, and the total
        flux and luminosity-squared moments are independent of where the
        limit falls (in expectation).
        """
        iso = self.isochrone
        remnants_factor = self._remnants_factor() if add_remnants else 1.0

        w, bright = self._lf_row_split()
        faint = ~bright
        self.has_integrated_component = bool(faint.any())

        # live (surviving) mass per star formed, in the row-binned
        # representation used for every analytic sum below, so that mass
        # bookkeeping is internally consistent
        mact = np.asarray(iso.mact, dtype=float)
        live_mass_per_star = float(np.sum(w * mact))

        if num_stars is not None:
            n_total = float(num_stars)
        elif total_mass is not None:
            total_mass = check_units(total_mass, 'Msun').to('Msun').value
            n_total = total_mass * remnants_factor / live_mass_per_star
        else:
            raise Exception('you must give total mass *or* number of stars')

        w_bright = float(w[bright].sum())
        if self.has_integrated_component:
            self.frac_num_sampled = w_bright
            self.frac_mass_sampled = float(
                np.sum(w[bright] * mact[bright])) / live_mass_per_star
        else:
            # exact unity so `frac_num_sampled < 1.0` gates (source.py) do
            # not fire on float round-off
            self.frac_num_sampled = 1.0
            self.frac_mass_sampled = 1.0

        # informational, kept for API compatibility with the mass path
        self.sampled_mass_lower_limit = (
            float(np.asarray(iso.mini)[bright].min()) if bright.any()
            else iso.m_max)

        # draw the sampled stars: Poisson total, categorical row assignment,
        # then a uniform initial mass within each star's row half-bin (the
        # same bins the weights integrate, so the split stays an exact
        # complement)
        lam = n_total * w_bright
        n_sampled = int(self.rng.poisson(lam)) if lam > 0 else 0
        if n_sampled > 0:
            cdf = np.cumsum(w[bright])
            cdf /= cdf[-1]
            idx = np.searchsorted(cdf, self.rng.uniform(size=n_sampled))
            rows = np.flatnonzero(bright)[idx]
            m1, m2 = iso.imf_bin_edges()
            self.initial_masses = self.rng.uniform(m1[rows], m2[rows])
        else:
            rows = np.array([], dtype=int)
            self.initial_masses = np.array([])
        self.row_counts = np.bincount(rows, minlength=len(w))
        self.sampled_row_mask = bright
        self.n_total_expected = n_total

        self.star_masses = iso.interpolate('mact', self.initial_masses)
        self.sampled_mass = self.star_masses.sum()

        self.abs_mags = {}
        for filt in self.filters:
            self.abs_mags[filt] = iso.interpolate(filt, self.initial_masses)

        self.num_stars_integrated = max(
            int(round(n_total * (1.0 - w_bright))), 0)

        # smooth component: exact complement-row sums (expectation values)
        if self.has_integrated_component:
            faint_live_mass = n_total * float(np.sum(w[faint] * mact[faint]))
            self.total_mass = (
                (self.sampled_mass + faint_live_mass) / remnants_factor)

            log_dddd = 4 * np.log10((10 * u.pc).to('cm').value)
            self.integrated_abs_mags = {}
            self._integrated_log_lumlum = {}
            for filt in iso.filters:
                f_row = 10**(-0.4 * np.asarray(iso.mag_table[filt],
                                               dtype=float))
                flux = n_total * float(np.sum(w[faint] * f_row[faint]))
                self.integrated_abs_mags[filt] = -2.5 * np.log10(flux)
                ff = n_total * float(np.sum(w[faint] * f_row[faint]**2))
                self._integrated_log_lumlum[filt] = np.log10(ff) + log_dddd
        else:
            self.total_mass = self.sampled_mass / remnants_factor

        self.live_star_mass = self.total_mass * remnants_factor
        self.ssp_labels = np.ones(len(self.star_masses), dtype=int)

        self.total_mass *= u.Msun
        self.sampled_mass *= u.Msun
        self.live_star_mass *= u.Msun

        for attr in ['eep', 'log_L', 'log_Teff']:
            if hasattr(iso, attr):
                if getattr(iso, attr) is not None:
                    vals_interp = iso.interpolate(attr, self.initial_masses)
                    setattr(self, attr, vals_interp)

    def sample_fraction(self, mag_limit, mag_limit_band):
        """
        Calculate the fraction of stars by mass and number that will be
        sampled with the give limiting magnitude.

        Parameters
        ----------
        mag_limit : float, optional
            Only sample individual stars that are brighter than this magnitude.
        mag_limit_band : str, optional
            Bandpass of the limiting magnitude.

        Returns
        -------
        m_lim : float
            Initial stellar mass associated with `mag_limit`.
        f_num_sampled : float
            Fraction of stars that will be sampled by number.
        f_mass_sampled : float
            Fraction of stars that will be sampled by mass.
        """
        iso = self.isochrone
        imfint = IMFIntegrator(self.imf, iso.m_min, iso.m_max)
        m_lim = iso.m_min
        f_num_sampled = 1.0
        f_mass_sampled = 1.0
        if mag_limit is not None:
            if  mag_limit_band is None:
                raise Exception('Must give bandpass of limiting magnitude.')
            mags = iso.mag_table[mag_limit_band] + self.dist_mod
            if mag_limit < mags.max() and mag_limit > mags.min():
                m_lim = iso.mag_to_mass(
                    mag_limit - self.dist_mod, mag_limit_band).min()
                f_num_sampled  = imfint.integrate(m_lim, iso.m_max, True)
                f_mass_sampled = imfint.m_integrate(m_lim, iso.m_max, True)
            elif mag_limit <= mags.min():
                # a limit brighter than the isochrone tip means no star
                # qualifies for individual sampling: the population is fully
                # smooth. Falling through with the defaults (m_lim = m_min,
                # f = 1) would silently sample *every* star instead.
                m_lim = iso.m_max
                f_num_sampled = 0.0
                f_mass_sampled = 0.0
            else:
                logger.warning(f'mag_lim = {mag_limit} is outside mag range.')
        return m_lim, f_num_sampled, f_mass_sampled

    def __add__(self, ssp):
        assert StellarPopulation in ssp.__class__.__mro__, 'invalid type(s)'
        assert self.filters == ssp.filters, 'must have same filters'
        assert self.distance == ssp.distance, 'SSPs must have same distance'
        # flux and luminosity-luminosity addition across SSPs is only meaningful
        # if both were corrected to the same redshift
        assert self.redshift == ssp.redshift, 'SSPs must have same redshift'
        new = deepcopy(self)

        if (
            any([v != 0.0 for v in new.a_lam.values()]) or
            any([v != 0.0 for v in ssp.a_lam.values()])
        ):
            logger.warning(
                'Extinction is currently not supported for composite '
                'populations. All a_lam will be set to zero.'
            )
            # No need to set the ssp.a_lam values to zero.
            for filt in new.filters:
                new.a_lam[filt] = 0.0

        if type(new.isochrone) != list:
            new.isochrone = [new.isochrone]
        if type(ssp.isochrone) != list:
            new.isochrone.append(ssp.isochrone)
        else:
            new.isochrone.extend(ssp.isochrone)

        if not hasattr(new, 'ssp_total_masses'):
            new.ssp_total_masses = [new.total_mass]
        if not hasattr(ssp, 'ssp_total_masses'):
            ssp.ssp_total_masses = [ssp.total_mass]
        new.ssp_total_masses.extend(ssp.ssp_total_masses)

        if not hasattr(new, 'ssp_total_num_stars'):
            _n = new.num_stars + new.num_stars_integrated
            new.ssp_total_num_stars = [_n]
        if not hasattr(ssp, 'ssp_total_num_stars'):
            _n = ssp.num_stars + ssp.num_stars_integrated
            ssp.ssp_total_num_stars = [_n]
        new.ssp_total_num_stars.extend(ssp.ssp_total_num_stars)

        new.total_mass = new.total_mass + ssp.total_mass
        new.sampled_mass = new.sampled_mass + ssp.sampled_mass
        new.live_star_mass = new.live_star_mass + ssp.live_star_mass
        new.frac_mass_sampled = new.sampled_mass / new.live_star_mass

        new_num_stars_total = new.num_stars + new.num_stars_integrated
        ssp_num_stars_total = ssp.num_stars + ssp.num_stars_integrated
        total_num_stars = new_num_stars_total + ssp_num_stars_total
        new.num_stars_integrated += ssp.num_stars_integrated

        new.initial_masses = np.concatenate(
            [new.initial_masses, ssp.initial_masses])
        new.star_masses = np.concatenate([new.star_masses, ssp.star_masses])
        new.frac_num_sampled = new.num_stars / total_num_stars

        new.ssp_num_fracs = []
        new.ssp_mass_fracs = []
        for n, m in zip(new.ssp_total_num_stars, new.ssp_total_masses):
            new.ssp_num_fracs.append(n / total_num_stars)
            new.ssp_mass_fracs.append(m / new.total_mass)

        # Loop over optional attributes.
        # Both SSPs must have the attribute to add them.
        for attr in ['eep', 'log_L', 'log_Teff']:
            if hasattr(new, attr) and hasattr(ssp, attr):
                new_attr = getattr(new, attr)
                ssp_attr = getattr(ssp, attr)
                setattr(new, attr, np.concatenate([new_attr, ssp_attr]))

        new_label = np.ones(len(ssp.star_masses), dtype=int)
        new_label *= len(new.isochrone)
        new.ssp_labels = np.concatenate([new.ssp_labels, new_label])

        if hasattr(new, 'log_age'):
            if type(new.log_age) != list:
                new.log_age = [new.log_age]
            new.log_age.append(ssp.log_age)
        if hasattr(new, 'feh'):
            if type(new.feh) != list:
                new.feh = [new.feh]
            new.feh.append(ssp.feh)

        for filt in new.filters:
            _mags = [new.abs_mags[filt], ssp.abs_mags[filt]]
            new.abs_mags[filt] = np.concatenate(_mags)
            if new.has_integrated_component and ssp.has_integrated_component:
                new_flux = 10**(-0.4 * new.integrated_abs_mags[filt])
                ssp_flux = 10**(-0.4 * ssp.integrated_abs_mags[filt])
                flux = new_flux + ssp_flux
                new.integrated_abs_mags[filt] = -2.5 * np.log10(flux)
                new_lumlum = 10**new._integrated_log_lumlum[filt]
                ssp_lumlum = 10**ssp._integrated_log_lumlum[filt]
                log_lumlum = np.log10(new_lumlum + ssp_lumlum)
                new._integrated_log_lumlum[filt] = log_lumlum
            elif ssp.has_integrated_component:
                new.integrated_abs_mags[filt] = ssp.integrated_abs_mags[filt]
                log_lumlum =  ssp._integrated_log_lumlum[filt]
                new._integrated_log_lumlum[filt] = log_lumlum

        return CompositePopulation(new)

    def __repr__(self):
        r = [f'{k} = {v}' for k, v in self._r.items()]
        t = 'Simple Stellar Population\n-------------------------\n'
        return t + '\n'.join(r)


class MISTSSP(SSP):
    """
    MIST Simple Stellar Population.

    .. note::
        You must give `total_mass` *or* `num_stars`.

    Parameters
    ----------
    log_age : float
        Log (base 10) of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str or list-like
        Name of the photometric system(s).
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    total_mass : float or `~astropy.units.Quantity` or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. This
        mass accounts for stellar remnants when ``add_remnants = True``, which
        means the actual sampled mass will be less than the given value. If
        float is given, the units are assumed to be solar masses.
    distance : float or `~astropy.units.Quantity`, optional
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`. Default distance is 10 `~astropy.units.pc`.
    a_lam : float or dict, optional
        Magnitude(s) of extinction. If float, the same extinction will be applied
        to all bands. If dict, the keys must be the same as the observational filters.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    imf : str or dict
        Which IMF to use, if str then must be one of pre-defined: 'kroupa',
        'scalo' or 'salpeter'. Can also specify custom (broken) power law as dict,
        which must contain either 'a' as a Float (describing the slope of a
        single power law) or 'a' (a list with 3 elements describing the slopes
        of a broken power law) and 'b' (a list  with 2 elements describing the
        locations of the breaks).
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    mass_tolerance : float, optional
        Tolerance in the fractional difference between the input mass and the
        final mass of the population. The parameter is only used when
        `total_mass` is given.
    add_remnants : bool, optional
        If True (default), apply scaling factor to total mass to account for
        stellar remnants in the form of white dwarfs, neutron stars,
        and black holes.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    phases = ['PMS', 'MS', 'giants', 'RGB', 'CHeB', 'AGB',
              'EAGB', 'TPAGB', 'postAGB', 'WDCS']

    def __init__(self, log_age, feh, phot_system, num_stars=None,
                 total_mass=None, distance=10*u.pc, a_lam=0.0, mag_limit=None,
                 mag_limit_band=None, imf='kroupa', mist_path=MIST_PATH,
                 imf_kw=None, mass_tolerance=0.05, add_remnants=True,
                 random_state=None, sampling='lf', **kwargs):

        self.feh = feh
        self.log_age = log_age
        self.phot_system = phot_system
        self.mist_path = mist_path
        _iso = MISTIsochrone(log_age, feh, phot_system,  mist_path, **kwargs)

        super(MISTSSP, self).__init__(
            isochrone=_iso,
            num_stars=num_stars,
            total_mass=total_mass,
            distance=distance,
            mag_limit=mag_limit,
            mag_limit_band=mag_limit_band,
            imf=imf,
            imf_kw=imf_kw,
            mass_tolerance=mass_tolerance,
            add_remnants=add_remnants,
            a_lam=a_lam,
            random_state=random_state,
            sampling=sampling
        )

        self._r.update({'log(age/yr)': self.log_age,
                        '[Fe/H]': self.feh,
                        'photometric system': self.phot_system})

    def select_phase(self, phase):
        """
        Generate stellar evolutionary phase mask. The mask will be `True` for
        sources that are in the give phase according to the MIST EEPs.

        Parameters
        ----------
        phase : str
            Evolutionary phase to select. Options are 'all', 'MS', 'giants',
            'RGB', 'CHeB', 'AGB', 'EAGB', 'TPAGB', 'postAGB', or 'WDCS'.

        Returns
        -------
        mask : `~numpy.ndarray`
            Mask that is `True` for stars in input phase and `False` otherwise.

        Notes
        -----
        The MIST EEP phases were taken from Table II: Primary Equivalent
        Evolutionary Points (EEPs):
        http://waps.cfa.harvard.edu/MIST/README_tables.pdf
        """
        if phase == 'all':
            mask = np.ones_like(self.eep, dtype=bool)
        elif phase == 'PMS':
            mask = self.eep < 202
        elif phase == 'MS':
            mask = (self.eep >= 202) & (self.eep < 454)
        elif phase == 'giants':
            mask = (self.eep >= 454) & (self.eep < 1409)
        elif phase == 'RGB':
            mask = (self.eep >= 454) & (self.eep <= 605)
        elif phase == 'CHeB':
            mask = (self.eep > 605) & (self.eep < 707)
        elif phase == 'AGB':
            mask = (self.eep >= 707) & (self.eep < 1409)
        elif phase == 'EAGB':
            mask = (self.eep >= 707) & (self.eep < 808)
        elif phase == 'TPAGB':
            mask = (self.eep >= 808) & (self.eep < 1409)
        elif phase == 'postAGB':
            mask = (self.eep >= 1409) & (self.eep <= 1710)
        elif phase == 'WDCS':
            mask = self.eep > 1710
        else:
            raise Exception('Uh, what phase u want?')

        return mask

    def get_star_phases(self):
        """Returns the stellar phases (as defined by the MIST EEPs)."""
        phase_list = ['PMS', 'MS', 'RGB', 'CHeB', 'EAGB',
                      'TPAGB', 'postAGB', 'WDCS']
        star_phases = np.array([''] * self.num_stars, dtype='<U8')
        for phase in phase_list:
            star_phases[self.select_phase(phase)] = phase
        return star_phases


class CompositePopulation(SSP):
    """
    Composite stellar populations.
    """

    def __init__(self, pop):

        for name, attr in pop.__dict__.items():
            setattr(self, name, attr)

    def __repr__(self):
        num_fracs = self.ssp_num_fracs
        mass_fracs = self.ssp_mass_fracs
        r = {'N_pops': self.num_pops,
             'M_star': f'{self.total_mass.value:.2e} M_sun',
             'number fractions': [f'{p * 100:.2f}%' for p in num_fracs],
             'mass fractions': [f'{p * 100:.2f}%' for p in mass_fracs]}
        if hasattr(self, 'log_age'):
            r['log(age/yr)'] = self.log_age
        if hasattr(self, 'feh'):
            r['[Fe/H]'] = self.feh
        if hasattr(self, 'phot_system'):
            r['photometric system'] = self.phot_system
        r = [f'{k} = {v}' for k, v in r.items()]
        t = 'Composite Population\n--------------------\n'
        return t + '\n'.join(r)


def constant_sb_stars_per_pix(sb, mean_mag, distance=10*u.pc, pixel_scale=0.2):
    """
    Calculate the number of stars per pixel for a uniform
    distribution (i.e., constant surface brightness) of stars.

    Parameters
    ----------
    sb : float
        Surface brightness of stellar population.
    mean_mag : float
        Mean stellar magnitude of the stellar population.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `~astropy.units.arcsec` per `~astropy.units.pixels`.

    Returns
    -------
    num_stars_per_pix : float
        The number of stars per pixel.
    """
    distance = check_units(distance, 'Mpc').to('pc').value
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel).value
    dist_mod = 5 * np.log10(distance) - 5
    num_stars_per_arsec_sq = 10**(0.4 * (mean_mag + dist_mod -  sb))
    num_stars_per_pix = num_stars_per_arsec_sq * pixel_scale**2
    return num_stars_per_pix
