# Standard library
from copy import deepcopy

# Third-party
import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
from astropy.modeling.models import Sersic2D
from scipy.special import gammaincinv, gamma

# Project
from . import MIST_PATH
from .log import logger
from .stars import SSP, MISTSSP, MISTIsochrone, constant_sb_stars_per_pix
from .space import sersic_xy, plummer_xy, uniform_xy, Plummer2D, Constant2D
from .util import check_units, check_xy_dim


__all__ = [
    'Source',
    'SersicSP',
    'ClusteredSersicSP',
    'MISTSersicSSP',
    'PlummerSP',
    'MISTPlummerSSP',
    'UniformSSP',
    'MISTUniformSSP',
]


class Source(object):
    """
    Artificial stellar system to be mock imaged.

    Parameters
    ----------
    xy : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        Positions of the stars in (x, y) coordinates.
    mags : dict of `~numpy.ndarray` or `~astropy.table.Table`
        Stellar magnitudes.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    """
    def __init__(self, xy, mags, xy_dim, pixel_scale=None, labels=None):
        if type(mags) == dict:
            self.mags = Table(mags)
        else:
            self.mags = mags.copy()
        if len(xy) != len(self.mags):
            raise Exception('numbers of magnitudes and positions must match')
        if type(xy) == np.ma.MaskedArray:
            mask = np.any(~xy.mask, axis=1)
            self.mags = self.mags[mask]
            self.xy = xy.data[mask]
        else:
            self.xy = np.asarray(xy)
            mask = np.ones(len(self.xy), dtype=bool)
        self.labels = labels if labels is None else np.asarray(labels)[mask]
        # which of the population's stars survived the frame mask, so
        # per-star population arrays (e.g. the nebular blob fractions) can be
        # subset the same way the magnitudes just were
        self.star_mask = mask
        self.xy_dim = check_xy_dim(xy_dim)
        if pixel_scale is not None:
            self.pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

    @property
    def x(self):
        """x coordinates"""
        return self.xy[:, 0]

    @property
    def y(self):
        """y coordinates"""
        return self.xy[:, 1]

    @property
    def num_stars(self):
        """Number of stars in source"""
        return len(self.x)

    def copy(self):
        """Create deep copy of source object."""
        return deepcopy(self)

    def select_stars(self, label):
        """Return boolean mask that is set to True for stars of given label."""
        if self.labels is None:
            raise Exception('The stars of this source do not have labels.')
        return self.labels == label

    def __add__(self, src):
        if Source not in src.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')
        new = deepcopy(self)
        for s in [new, src]:
            if hasattr(s, 'sp') and s.sp.mag_limit is not None:
                logger.warning(
                    'Composite source objects currently only support '
                    'discrete components.\nIf you want to include smooth '
                    'components (i.e., mag_limit is not None),\n'
                    'add the individual source images together.'
                )
                break
        if not np.allclose(new.xy_dim, src.xy_dim):
            raise Exception('Sources must have the same xy_dim')
        if new.labels is None:
            new.labels = np.ones(new.num_stars, dtype=int)
        labels = np.ones(src.num_stars, dtype=int) * (new.labels.max() + 1)
        new_labels = np.concatenate([new.labels, labels])
        new.xy = np.concatenate([new.xy, src.xy])
        new.mags = vstack([new.mags, src.mags])
        comp_src = CompositeSource(new.xy, new.mags, new.xy_dim,
                                   new.pixel_scale, new_labels)
        return comp_src


class CompositeSource(Source):
    """A source consisting of 2 or more `~artpop.source.Source` objects."""
    pass


class SersicSP(Source):
    """
    Stellar population with a Sersic spatial distribution.

    Parameters
    ----------
    sp : `~artpop.stars.populations.StellarPopulation`
        A stellar population object.
    r_eff : float or `~astropy.units.Quantity`
        Effective radius of the source. If a float is given, the units are
        assumed to be `kpc`. Must be greater than zero.
    n : float
        Sersic index. Must be greater than zero.
    theta : float or `~astropy.units.Quantity`
        Rotation angle, counterclockwise from the positive x-axis. If a float
        is given, the units are assumed to be `degree`.
    ellip : float
        Ellipticity.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    num_r_eff : float, optional
        Number of r_eff to sample positions within. This parameter is needed
        because the current Sersic sampling function samples from within a
        discrete grid. Default is 10.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    eta : float, optional
        Distance-duality parameter. The physical size is converted to an angle
        with ``D_A = D_L / [(1+z)^2 eta^2]``, Etherington's relation, which
        follows from photon conservation in any metric theory and is not a
        cosmological model. Exposed so a violation of duality can itself be
        probed. Default: 1.0.
    distance_angular : float or `~astropy.units.Quantity`, optional
        Override the angular-diameter distance outright, for callers who want
        distance, redshift and size to be literally independent. If a float is
        given the units are assumed to be Mpc.
    """

    def __init__(self, sp, r_eff, n, theta, ellip, xy_dim, pixel_scale,
                 num_r_eff=10, dx=0, dy=0, labels=None, eta=1.0,
                 distance_angular=None):

        self.sp = sp
        self.mag_limit = sp.mag_limit
        self.mag_limit_band = sp.mag_limit_band
        self.smooth_model = None
        self.distance_angular = _angular_distance(sp, eta, distance_angular)

        if self.mag_limit is not None and sp.frac_num_sampled < 1.0:
            self._build_smooth_model(r_eff, n, theta, ellip, xy_dim, pixel_scale, dx, dy)

        # `distance` here is the ANGULAR-DIAMETER distance, not sp.distance:
        # sersic_xy's only use of it is arctan2(r_eff, distance), i.e. an angle.
        # Handing it D_A rather than editing that arctan2 is deliberate --
        # ALVISS monkey-patches over `sersic_xy` with its own memory-efficient
        # sampler (alviss_inject.py:138), so any fix written into the function
        # body would be replaced at import time and silently lost. At z = 0 the
        # two distances are the same object, so this is a no-op.
        self.xy_kw = dict(num_stars=sp.num_stars, r_eff=r_eff, n=n,
                          theta=theta, ellip=ellip,
                          distance=self.distance_angular,
                          xy_dim=xy_dim, num_r_eff=num_r_eff, dx=dx, dy=dy,
                          pixel_scale=pixel_scale, random_state=sp.rng)

        _xy = self._draw_xy()

        super(SersicSP, self).__init__(
            _xy, sp.mag_table, xy_dim, pixel_scale, labels)

    def _build_smooth_model(self, r_eff, n, theta, ellip, xy_dim, pixel_scale, dx, dy):
        """The Sersic2D smooth model (amplitude set later by the imager)."""
        _r_eff = check_units(r_eff, 'kpc').to('Mpc').value
        _theta = check_units(theta, 'deg').to('radian').value
        _distance = self.distance_angular.to('Mpc').value
        _pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

        if _r_eff <= 0:
            raise Exception('Effective radius must be greater than zero.')

        xy_dim = check_xy_dim(xy_dim)
        x_0, y_0 = xy_dim//2
        x_0 += dx
        y_0 += dy
        self.n = n
        self.ellip = ellip
        self.r_sky = np.arctan2(_r_eff, _distance) * u.radian.to('arcsec')
        self.r_sky *= u.arcsec
        r_pix = self.r_sky.to('pixel', u.pixel_scale(_pixel_scale)).value

        self.smooth_model = Sersic2D(
            x_0=x_0, y_0=y_0, n=n, r_eff=r_pix, theta=_theta, ellip=ellip)

    def _draw_xy(self):
        """Star positions (a masked array; off-frame stars masked)."""
        return sersic_xy(**self.xy_kw)

    def mag_to_image_amplitude(self, m_tot, zpt):
        """
        Convert total magnitude into amplitude parameter for the smooth model.

        Parameters
        ----------
        m_tot : float
            Total magnitude in the smooth component of the system.
        zpt : float
            Photometric zero point.

        Returns
        -------
        mu_e : float
            Surface brightness at the effective radius of the Sersic
            distribution in mags per square arcsec.
        amplitude : float
            Amplitude parameter for the smooth model in image flux units.
        param_name : str
            Name of amplitude parameter (needed to set its value when
            generating the smooth model).
        """
        param_name = 'amplitude'
        b_n = gammaincinv(2.0 * self.n, 0.5)
        f_n = gamma(2 * self.n) *self.n * np.exp(b_n) / b_n**(2 * self.n)
        r_circ = self.r_sky * np.sqrt(1 - self.ellip)
        area = np.pi * r_circ.to('arcsec').value**2
        mu_e = m_tot + 2.5 * np.log10(2 * area) + 2.5 * np.log10(f_n)
        amplitude = 10**(0.4 * (zpt - mu_e)) * self.pixel_scale.value**2
        return mu_e, amplitude, param_name


class ClusteredSersicSP(SersicSP):
    """
    A `SersicSP` whose young SSPs sit in star-forming complexes (2026-10-15).

    Every SSP without a `~artpop.space.clusters.ClusterSpec` is placed exactly
    as `SersicSP` places it (the Sersic draw for all stars happens first, from
    the same random state, so those stars land where `SersicSP` puts them).
    An SSP with a spec is a mixture: a share ``f_diffuse`` keeps its Sersic
    position, the rest is assigned to the spec's complexes with probability
    proportional to their weights and placed on that complex's projected
    Plummer. The SSP's smooth light follows the same mixture (`smooth_image`),
    so the split between sampled stars and smooth light does not move light
    between the complexes and the Sersic.

    **Where the complexes go.** The specs are placed youngest first (by
    ``age_myr``). A complex either starts a new site -- drawn in the galaxy's
    disk plane from a ring surface density, peak ``r_peak_reff`` r_eff, width
    ``width_reff`` r_eff, truncated at ``max_reff`` r_eff
    (`artpop.space.clusters.ring_disk_offsets`) -- or, with probability
    ``site_reuse``, sits near an earlier (younger) complex chosen in proportion
    to its mass, offset by a disk-plane Gaussian of
    ``sqrt(site_s0_pc^2 + (drift_kms t)^2)`` pc: older complexes have drifted
    further from where the youngest stars form now. Disk-plane offsets are
    projected with the galaxy's ellipticity and position angle.

    **Diffuse ionized gas.** An SSP whose ``ssp_components`` record carries
    ``integrated_dig_abs_mags`` (the photons the HII regions did not capture,
    absorbed in the diffuse gas) adds that light on the galaxy's Sersic profile
    to the gas image (`smooth_image`), so it is never part of the SBF field.

    With an active nebular knob in ``'stromgren'`` mode the gas light of a
    cluster -- its member stars' blob share and its share of the smooth
    component's -- fills **one** projected Stromgren sphere centred on the
    cluster, sized by the cluster's summed captured photons in the config's gas
    (`artpop.nebular.NebularConfig.stromgren_diameter_pc`). Stars outside
    clusters keep their own spheres.

    Parameters
    ----------
    clusters : list of `~artpop.space.clusters.ClusterSpec` (or dicts)
        One per clustered SSP (``label`` = its 1-based ``ssp_labels`` value).
        May be empty (a source that only needs the diffuse-gas Sersic).
    r_peak_reff, width_reff, max_reff : float
        The ring surface density of new sites, in units of r_eff.
    width_out_reff : float or None
        The ring's width outside its peak (default: ``width_reff``); smaller
        keeps the complexes from the outskirts.
    ssp_reff_scale : dict or None
        ``{label: s}``: the stars of SSP ``label`` that are not in complexes,
        and its smooth light, follow a Sersic of the same n, ellipticity and
        angle but effective radius ``s x r_eff`` (2026-10-15: dwarfs have age
        gradients, young and intermediate-age stars more concentrated than the
        old halo). Labels not listed (and the diffuse ionized gas) keep the
        galaxy's r_eff. The ring of complex centres is in units of the
        galaxy's r_eff.
    site_reuse : float
        Probability that a complex sits near an earlier one (0: independent).
    site_s0_pc, drift_kms : float
        The offset from the parent site: sigma = sqrt(s0^2 + (drift t)^2).
    centre_mode : {'ring', 'locus'}
        ``'ring'`` (default): new sites from the ring above. ``'locus'``
        (2026-10-20, P-52 variant): every new site from an isotropic Gaussian of
        ``locus_spread_reff`` r_eff around one point ``locus_offset_reff`` r_eff
        from the centre, at an angle drawn once per galaxy
        (`artpop.space.clusters.locus_disk_offsets`; ``max_reff`` still cuts).
    locus_offset_reff, locus_spread_reff : float
        The locus's distance from the centre and its spread, in r_eff.
    diffuse_mode : {'sersic', 'complexes'}
        Where the clustered SSPs' ``f_diffuse`` share (its stars and smooth
        light) and the diffuse ionized gas go. ``'sersic'`` (default): on the
        galaxy's Sersic (the SSP's ``ssp_reff_scale`` one for the stars).
        ``'complexes'`` (2026-10-20, P-52 variant; real Halpha has no Sersic-like
        central rise outside compact starbursts): on a smoothed copy of the
        complexes' layout -- each diffuse star joins a complex drawn by weight,
        offset by its Plummer plus a Gaussian of ``diffuse_smooth_reff`` r_eff;
        the smooth share likewise; the DIG on the complexes of every SSP
        weighted by their captured photons ``q_h``, smoothed the same way. A
        source without complexes (or without ``q_h`` for the DIG) keeps the
        Sersic.
    diffuse_smooth_reff : float
        The smoothing Gaussian's sigma, in r_eff.

    The other parameters are `SersicSP`'s.
    """

    def __init__(self, sp, r_eff, n, theta, ellip, xy_dim, pixel_scale,
                 clusters=(), r_peak_reff=0.4, width_reff=0.3, max_reff=2.0,
                 site_reuse=0.0, site_s0_pc=20.0, drift_kms=5.0, width_out_reff=None,
                 ssp_reff_scale=None, centre_mode='ring', locus_offset_reff=0.75,
                 locus_spread_reff=0.25, diffuse_mode='sersic', diffuse_smooth_reff=0.25,
                 num_r_eff=10, dx=0, dy=0, labels=None, eta=1.0,
                 distance_angular=None):
        from .space.clusters import ClusterSpec
        self.cluster_specs = [ClusterSpec.coerce(c) for c in (clusters or [])]
        seen = [c.label for c in self.cluster_specs]
        if len(set(seen)) != len(seen):
            raise ValueError(f'one ClusterSpec per SSP label, got labels {seen}')
        iso = getattr(sp, 'isochrone', None)
        n_ssp = len(iso) if isinstance(iso, list) else 1
        for c in self.cluster_specs:
            if not 1 <= c.label <= n_ssp:
                raise ValueError(f'cluster label {c.label} is not an SSP of this '
                                 f'population (labels 1..{n_ssp})')
        self.r_peak_reff = float(r_peak_reff)
        self.width_reff = float(width_reff)
        self.width_out_reff = None if width_out_reff is None else float(width_out_reff)
        self.max_reff = float(max_reff)
        if not (0.0 <= self.r_peak_reff < self.max_reff and self.width_reff > 0
                and (self.width_out_reff is None or self.width_out_reff > 0)):
            raise ValueError('need 0 <= r_peak_reff < max_reff and widths > 0')
        self.site_reuse = float(site_reuse)
        if not 0.0 <= self.site_reuse <= 1.0:
            raise ValueError(f'site_reuse must be in [0, 1], got {site_reuse}')
        self.site_s0_pc = float(site_s0_pc)
        self.drift_kms = float(drift_kms)
        if centre_mode not in ('ring', 'locus'):
            raise ValueError(f"centre_mode must be 'ring' or 'locus', got {centre_mode!r}")
        if diffuse_mode not in ('sersic', 'complexes'):
            raise ValueError(f"diffuse_mode must be 'sersic' or 'complexes', got {diffuse_mode!r}")
        self.centre_mode = centre_mode
        self.locus_offset_reff = float(locus_offset_reff)
        self.locus_spread_reff = float(locus_spread_reff)
        self.diffuse_mode = diffuse_mode
        self.diffuse_smooth_reff = float(diffuse_smooth_reff)
        if not (0.0 <= self.locus_offset_reff < self.max_reff and self.locus_spread_reff > 0
                and self.diffuse_smooth_reff > 0):
            raise ValueError('need 0 <= locus_offset_reff < max_reff, locus_spread_reff > 0 '
                             'and diffuse_smooth_reff > 0')
        self.locus_phi = None
        self._r_eff_kpc = check_units(r_eff, 'kpc').to('kpc').value
        self._theta_rad = check_units(theta, 'deg').to('radian').value
        self._ellip = float(ellip)
        self.ssp_reff_scale = {int(k): float(v) for k, v in (ssp_reff_scale or {}).items()
                               if float(v) != 1.0}
        if any(not v > 0 for v in self.ssp_reff_scale.values()):
            raise ValueError(f'ssp_reff_scale values must be > 0, got {ssp_reff_scale}')
        self._cache = {}
        super(ClusteredSersicSP, self).__init__(
            sp, r_eff, n, theta, ellip, xy_dim, pixel_scale,
            num_r_eff=num_r_eff, dx=dx, dy=dy, labels=labels, eta=eta,
            distance_angular=distance_angular)
        if self.smooth_model is None:
            # the diffuse gas needs the Sersic even when every star is sampled
            self._build_smooth_model(r_eff, n, theta, ellip, xy_dim, pixel_scale, dx, dy)
        keep = self.star_mask
        self.star_cluster = self._star_cluster_full[keep]
        self.star_component = self._star_component_full[keep]

    # geometry ------------------------------------------------------------
    @property
    def px_per_pc(self):
        """Pixels per physical pc at the source's angular-diameter distance."""
        d_pc = self.distance_angular.to('pc').value
        ps = check_units(self.xy_kw['pixel_scale'], u.arcsec / u.pixel).value
        return np.degrees(1.0 / d_pc) * 3600.0 / ps

    def _centre_px(self):
        xy_dim = check_xy_dim(self.xy_kw['xy_dim'])
        return (xy_dim[0] // 2 + self.xy_kw['dx'], xy_dim[1] // 2 + self.xy_kw['dy'])

    def _draw_xy(self):
        from .space.clusters import (ring_disk_offsets, locus_disk_offsets, disk_to_sky,
                                     plummer_offsets, KMS_PC_PER_MYR)
        xy = super(ClusteredSersicSP, self)._draw_xy()
        sp, rng = self.sp, self.xy_kw['random_state']
        pos = np.array(np.ma.getdata(xy), dtype=float, copy=True)
        n_star = len(pos)
        ssp_labels = np.asarray(getattr(sp, 'ssp_labels', np.ones(n_star, int)))
        if ssp_labels.size != n_star:
            raise ValueError(f'ssp_labels ({ssp_labels.size}) do not match the '
                             f'{n_star} stars')
        x0, y0 = self._centre_px()
        ppc = self.px_per_pc
        r_eff_px = self._r_eff_kpc * 1e3 * ppc
        q_star = np.asarray(getattr(sp, 'nebular_q_h', np.zeros(n_star)), dtype=float)
        if q_star.size != n_star:
            q_star = np.zeros(n_star)
        comps = getattr(sp, 'ssp_components', None) or [sp._ssp_component_record()]

        # per-SSP sizes (age gradients): redraw those SSPs' stars from a Sersic of
        # s x r_eff; complex members are moved onto their complexes below anyway
        for label, s in sorted(self.ssp_reff_scale.items()):
            idx = np.flatnonzero(ssp_labels == label)
            if idx.size:
                kw = dict(self.xy_kw, num_stars=int(idx.size), r_eff=self.xy_kw['r_eff'] * s)
                pos[idx] = np.array(np.ma.getdata(sersic_xy(**kw)), dtype=float)

        self._star_cluster_full = np.full(n_star, -1, dtype=int)
        self._star_component_full = np.full(n_star, -1, dtype=int)
        self.clusters = []
        specs = self.cluster_specs
        order = sorted(range(len(specs)), key=lambda i: (
            specs[i].age_myr is None, specs[i].age_myr or 0.0, i))
        site_disk, site_mass = np.zeros((0, 2)), np.zeros(0)
        site_ids = np.zeros(0, dtype=int)
        # the locus angle: one draw per galaxy, only in that mode (the ring's
        # random stream is untouched)
        if self.centre_mode == 'locus':
            self.locus_phi = float(rng.uniform(0.0, 2.0 * np.pi))
        sig_diff_px = self.diffuse_smooth_reff * r_eff_px
        for si in order:
            spec = specs[si]
            n_c = spec.n_clusters
            age = float(spec.age_myr or 0.0)
            mass = spec.masses if spec.masses is not None else spec.weights
            # sites: a fresh ring draw for every complex, replaced by an offset
            # from an earlier (younger) complex with probability site_reuse
            if self.centre_mode == 'locus':
                disk = locus_disk_offsets(n_c, r_eff_px, rng, self.locus_phi,
                                          self.locus_offset_reff, self.locus_spread_reff,
                                          self.max_reff)
            else:
                disk = ring_disk_offsets(n_c, r_eff_px, rng, self.r_peak_reff,
                                         self.width_reff, self.max_reff, self.width_out_reff)
            parent = np.full(n_c, -1, dtype=int)
            if site_mass.size and self.site_reuse > 0:
                linked = rng.uniform(size=n_c) < self.site_reuse
                if linked.any():
                    p = site_mass / site_mass.sum()
                    parent[linked] = rng.choice(site_mass.size, size=int(linked.sum()), p=p)
                    sig_px = np.hypot(self.site_s0_pc,
                                      self.drift_kms * KMS_PC_PER_MYR * age) * ppc
                    disk[linked] = (site_disk[parent[linked]]
                                    + rng.normal(0.0, sig_px, size=(int(linked.sum()), 2)))
            centres = disk_to_sky(disk, self._theta_rad, self._ellip)
            if spec.a_sigma_dex > 0:
                a_pc = spec.a_pc * 10 ** (spec.a_sigma_dex * rng.normal(size=n_c))
            else:
                a_pc = np.array(spec.a_pc, dtype=float)
            first_id = len(self.clusters)
            idx = np.flatnonzero(ssp_labels == spec.label)
            comp = (rng.choice(spec.n_clusters + 1, size=idx.size, p=spec.component_probs())
                    if idx.size else np.zeros(0, dtype=int))
            self._star_component_full[idx] = comp
            rec = comps[spec.label - 1] if spec.label - 1 < len(comps) else None
            q_smooth = float((rec or {}).get('integrated_q_h', 0.0) or 0.0)
            for c in range(spec.n_clusters):
                gid = len(self.clusters)
                members = idx[comp == c + 1]
                a_px = a_pc[c] * ppc
                pos[members] = (np.array([x0, y0]) + centres[c]
                                + plummer_offsets(members.size, a_px, rng))
                self._star_cluster_full[members] = gid
                w_smooth = (1.0 - spec.f_diffuse) * spec.weights[c]
                self.clusters.append(dict(
                    id=gid, label=spec.label, name=spec.name, index=c,
                    x=float(x0 + centres[c, 0]), y=float(y0 + centres[c, 1]),
                    r_pc=float(np.hypot(*centres[c]) / ppc),
                    r_disk_pc=float(np.hypot(*disk[c]) / ppc),
                    disk_pc=(float(disk[c, 0] / ppc), float(disk[c, 1] / ppc)),
                    a_pc=float(a_pc[c]), a_px=float(a_px),
                    weight=float(spec.weights[c]), smooth_weight=float(w_smooth),
                    mass=None if spec.masses is None else float(spec.masses[c]),
                    age_myr=spec.age_myr,
                    site=int(parent[c]) if parent[c] < 0 else int(site_ids[parent[c]]),
                    n_stars=int(members.size),
                    q_h=float(q_star[members].sum() + w_smooth * q_smooth)))
            # diffuse stars on the smoothed layout of this bin's complexes
            diffuse = idx[comp == 0]
            if self.diffuse_mode == 'complexes' and n_c > 0 and diffuse.size:
                w = np.asarray(spec.weights, dtype=float)
                host = rng.choice(n_c, size=diffuse.size, p=w / w.sum())
                for c in np.unique(host):
                    sel = diffuse[host == c]
                    pos[sel] = (np.array([x0, y0]) + centres[c]
                                + plummer_offsets(sel.size, a_pc[c] * ppc, rng)
                                + rng.normal(0.0, sig_diff_px, size=(sel.size, 2)))
            # this bin's complexes become sites for the older bins
            site_disk = np.vstack([site_disk, disk])
            site_mass = np.concatenate([site_mass, np.asarray(mass, dtype=float)])
            site_ids = np.concatenate([site_ids, np.arange(first_id, first_id + n_c)])

        xy_dim = check_xy_dim(self.xy_kw['xy_dim'])
        outside = ((pos[:, 0] < 0) | (pos[:, 0] > xy_dim[0] - 1) |
                   (pos[:, 1] < 0) | (pos[:, 1] > xy_dim[1] - 1))
        return np.ma.masked_array(pos, mask=np.column_stack((outside, outside)))

    # nebulae -------------------------------------------------------------
    def stromgren_clusters(self):
        """True when cluster gas goes into one sphere per cluster."""
        from .nebular import is_active
        cfg = getattr(self.sp, 'nebular', None)
        return is_active(cfg) and getattr(cfg, 'size_mode', 'fixed') == 'stromgren'

    def cluster_gas_radius_px(self, cluster):
        """Projected Stromgren radius (pixels) of one cluster's nebula."""
        q = float(cluster['q_h'])
        if q <= 0:
            return 0.0
        return 0.5 * float(self.sp.nebular.stromgren_diameter_pc(q)) * self.px_per_pc

    def paint_cluster_gas(self, image, signal):
        """Add each cluster's gas ``signal[id]`` to ``image`` as its projected
        Stromgren sphere centred on the cluster (flux-conserving in-frame).
        The kernels are band-independent and cached per cluster."""
        from .nebular import sphere_kernel
        from .space.clusters import paint_stamp
        for cl in self.clusters:
            s = float(signal[cl['id']])
            if s == 0.0:
                continue
            key = ('sphere', cl['id'])
            if key not in self._cache:
                self._cache[key] = sphere_kernel(self.cluster_gas_radius_px(cl))
            kern = self._cache[key]
            paint_stamp(image, kern, kern.shape[0] // 2, cl['x'], cl['y'], s)
        return image

    def _cluster_unit_image(self, label, shape):
        """Sum over the complexes of SSP ``label`` of smooth_weight x its
        Plummer stamp: band-independent, so built once per source and shape."""
        from .space.clusters import plummer_stamp, paint_stamp
        key = ('unit', int(label), tuple(shape))
        if key not in self._cache:
            img = np.zeros(shape)
            for cl in self.clusters:
                if cl['label'] != label or not cl['smooth_weight'] > 0:
                    continue
                ix, iy = np.floor(cl['x'] + 0.5), np.floor(cl['y'] + 0.5)
                stamp, half = plummer_stamp(cl['a_px'], cl['x'] - ix, cl['y'] - iy,
                                            max_half=max(shape))
                paint_stamp(img, stamp, half, cl['x'], cl['y'], cl['smooth_weight'])
            self._cache[key] = img
        return self._cache[key]

    def _layout_unit_image(self, shape, label=None):
        """The smoothed layout of the complexes at unit total weight
        (``diffuse_mode='complexes'``): for ``label``, its complexes weighted by
        their ``weight``; for None (the DIG), every complex weighted by its
        captured photons ``q_h``. Plummer stamps convolved with a Gaussian of
        ``diffuse_smooth_reff`` r_eff; light beyond the frame is lost, as the
        Sersic's is. None when there is nothing to weight."""
        from scipy.ndimage import gaussian_filter
        from .space.clusters import plummer_stamp, paint_stamp
        key = ('layout', label, tuple(shape))
        if key not in self._cache:
            cls = [c for c in self.clusters if (c['label'] == label if label is not None else True)]
            w = np.array([c['weight'] if label is not None else c['q_h'] for c in cls], dtype=float)
            if not cls or not np.sum(w) > 0:
                self._cache[key] = None
            else:
                img = np.zeros(shape)
                for cl, wc in zip(cls, w / w.sum()):
                    if not wc > 0:
                        continue
                    ix, iy = np.floor(cl['x'] + 0.5), np.floor(cl['y'] + 0.5)
                    stamp, half = plummer_stamp(cl['a_px'], cl['x'] - ix, cl['y'] - iy,
                                                max_half=max(shape))
                    paint_stamp(img, stamp, half, cl['x'], cl['y'], wc)
                sig = self.diffuse_smooth_reff * self._r_eff_kpc * 1e3 * self.px_per_pc
                self._cache[key] = gaussian_filter(img, sig, mode='constant')
        return self._cache[key]

    def _sersic_unit_image(self, shape, scale=1.0):
        """The Sersic model at unit total flux (zero point 0, magnitude 0), with
        its effective radius times ``scale`` (the amplitude goes as 1/scale^2)."""
        key = ('sersic', tuple(shape), float(scale))
        if key not in self._cache:
            _, amp, name = self.mag_to_image_amplitude(0.0, 0.0)
            model = self.smooth_model.copy()
            setattr(model, name, amp / float(scale) ** 2)
            model.r_eff = self.smooth_model.r_eff * float(scale)
            yy, xx = np.mgrid[:shape[0], :shape[1]]
            self._cache[key] = model(xx, yy)
        return self._cache[key]

    # smooth light --------------------------------------------------------
    def smooth_image(self, bandpass, mag_to_flux, shape):
        """
        The smooth component as two images, ``(stars, gas)``.

        Each SSP's integrated light (``sp.ssp_components``) is split like its
        stars: unclustered SSPs and the ``f_diffuse`` share onto the Sersic
        model; each cluster's share onto its projected Plummer -- except, with
        ``'stromgren'`` nebulae, the share ``integrated_blob_frac`` that is gas
        light, which goes into the cluster's sphere (``gas``). Diffuse ionized
        gas (``integrated_dig_abs_mags``) goes onto the Sersic in ``gas``.
        ``mag_to_flux`` converts an apparent magnitude to the image's flux units.
        """
        sp = self.sp
        stars = np.zeros(shape)
        gas = np.zeros(shape)
        comps = getattr(sp, 'ssp_components', None) or [sp._ssp_component_record()]
        specs = {c.label: c for c in self.cluster_specs}
        offset = sp.dist_mod + sp.a_lam[bandpass]
        to_gas = self.stromgren_clusters()
        sersic_flux = 0.0
        scaled = {}                       # per-SSP sizes: {scale: flux}
        dig_flux = 0.0
        gas_signal = np.zeros(len(self.clusters))
        for label, rec in enumerate(comps, start=1):
            rec = rec or {}
            dig = rec.get('integrated_dig_abs_mags')
            if dig:
                f_dig = float(mag_to_flux(dig[bandpass] + offset))
                if np.isfinite(f_dig) and f_dig > 0:
                    dig_flux += f_dig
            mags = rec.get('integrated_abs_mags')
            if not mags:
                continue
            flux = float(mag_to_flux(mags[bandpass] + offset))
            if not np.isfinite(flux) or flux <= 0:
                continue
            spec = specs.get(label)
            s_lab = self.ssp_reff_scale.get(label, 1.0)
            share = flux if spec is None else spec.f_diffuse * flux
            layout = (self._layout_unit_image(shape, label)
                      if spec is not None and self.diffuse_mode == 'complexes' else None)
            if layout is not None:
                stars = stars + share * layout
            elif s_lab == 1.0:
                sersic_flux += share
            else:
                scaled[s_lab] = scaled.get(s_lab, 0.0) + share
            if spec is None:
                continue
            blob = float(rec.get('integrated_blob_frac', {}).get(bandpass, 0.0)) if to_gas else 0.0
            for cl in self.clusters:
                if cl['label'] == label:
                    gas_signal[cl['id']] += cl['smooth_weight'] * flux * blob
            stars += flux * (1.0 - blob) * self._cluster_unit_image(label, shape)
        if (sersic_flux > 0 or dig_flux > 0) and self.smooth_model is not None:
            unit = self._sersic_unit_image(shape)
            if sersic_flux > 0:
                _, amp, name = self.mag_to_image_amplitude(-2.5 * np.log10(sersic_flux), 0.0)
                setattr(self.smooth_model, name, amp)     # ArtPop's own side effect
                stars = stars + sersic_flux * unit
            if dig_flux > 0:
                layout = (self._layout_unit_image(shape)
                          if self.diffuse_mode == 'complexes' else None)
                gas = gas + dig_flux * (unit if layout is None else layout)
        if self.smooth_model is not None:
            for s_lab, f_s in sorted(scaled.items()):
                if f_s > 0:
                    stars = stars + f_s * self._sersic_unit_image(shape, s_lab)
        if gas_signal.any():
            self.paint_cluster_gas(gas, gas_signal)
        return stars, gas


def _angular_distance(sp, eta=1.0, distance_angular=None):
    """
    The distance that converts a physical size to an angle.

    Flux runs on the luminosity distance and angles run on the
    angular-diameter distance; the two are equal only at z = 0, which is why
    ArtPop has been able to use one number for both. ``D_A = D_L/[(1+z)^2
    eta^2]`` is Etherington's distance-duality relation -- photon conservation
    in any metric theory, not a cosmological model -- and ``eta`` is exposed so
    that a duality violation can itself be probed. ``distance_angular``
    overrides the relation entirely, for anyone who wants the decoupling of
    distance, redshift and size to be literal.

    Surface-brightness dimming is **not** coded anywhere. It emerges from this
    function (solid angle) together with `dist_mod` (flux), and the fact that
    it comes out to exactly ``10 log10(1+z)`` is asserted rather than assumed.
    """
    if distance_angular is not None:
        return check_units(distance_angular, 'Mpc')
    if hasattr(sp, 'distance_angular'):
        return sp.distance_angular(eta=eta)
    return check_units(sp.distance, 'Mpc')


def _check_label_type(ssp, label_type, has_phases=True):
    if label_type is not None:
        if label_type == 'phases' and has_phases:
            labels = ssp.get_star_phases()
        elif hasattr(ssp, label_type):
            labels = getattr(ssp, label_type)
        else:
            raise Exception(f'{label_type} is not a valid label type.')
    else:
        labels = None
    return labels


class MISTSersicSSP(SersicSP):
    """
    MIST simple stellar population with a Sersic spatial distribution. This
    is a convenience class that combines `~artpop.space.sersic_xy` and
    `~artpop.stars.MISTSSP` to make a `~artpop.source.Source` object.

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
    r_eff : float or `~astropy.units.Quantity`
        Effective radius of the source. If a float is given, the units are
        assumed to be `kpc`. Must be greater than zero.
    n : float
        Sersic index. Must be greater than zero.
    theta : float or `~astropy.units.Quantity`
        Rotation angle, counterclockwise from the positive x-axis. If a float
        is given, the units are assumed to be `degree`.
    ellip : float
        Ellipticity defined as `1 - b/a`, where `b` is the semi-minor axis
        and `a` is the semi-major axis.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    total_mass : float or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. This
        mass accounts for stellar remnants, so the actual sampled mass will be
        less than the given value.
    a_lam : float or dict, optional
        Magnitude(s) of extinction. If float, the same extinction will be applied
        to all bands. If dict, the keys must be the same as the observational filters.
    add_remnants : bool, optional
        If True (default), apply scaling factor to total mass to account for
        stellar remnants in the form of white dwarfs, neutron stars,
        and black holes.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    num_r_eff : float, optional
        Number of r_eff to sample positions within. This parameter is needed
        because the current Sersic sampling function samples from within a
        discrete grid. Default is 10.
    mass_tolerance : float, optional
        Tolerance in the fractional difference between the input mass and the
        final mass of the population. The parameter is only used when
        `total_mass` is given.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    label_type : str, optional
        If not None, the type of labels to pass to the `~artpop.source.Source`
        object. Can be `phases` or an attribute of `~artpop.stars.MISTSSP`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, log_age, feh, phot_system, r_eff, n, theta, ellip,
                 distance, xy_dim, pixel_scale, num_stars=None,
                 total_mass=None, a_lam=0.0, add_remnants=True, mag_limit=None,
                 mag_limit_band=None, imf='kroupa', imf_kw=None,
                 mist_path=MIST_PATH, num_r_eff=10, mass_tolerance=0.01,
                 dx=0, dy=0, label_type=None, random_state=None,
                 eta=1.0, distance_angular=None, **kwargs):

        # **kwargs reaches MISTSSP and through it MISTIsochrone, so `redshift`,
        # `a_v_host`, `a_v_mw`, `version`, `a_over_fe` and `sampling` all work
        # here rather than only on a hand-built MISTSSP.
        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, a_lam=a_lam, total_mass=total_mass,
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state,
                           mag_limit=mag_limit, mag_limit_band=mag_limit_band,
                           add_remnants=add_remnants,
                           mass_tolerance=mass_tolerance, **kwargs)

        ssp = MISTSSP(**self.ssp_kw)
        labels = _check_label_type(ssp, label_type)

        super(MISTSersicSSP, self).__init__(
            sp=ssp, r_eff=r_eff, n=n, theta=theta, ellip=ellip, xy_dim=xy_dim,
            pixel_scale=pixel_scale, num_r_eff=num_r_eff, dx=dx, dy=dy,
            labels=labels, eta=eta, distance_angular=distance_angular)


class PlummerSP(Source):
    """
    Simple stellar population with a Plummer spatial distribution.

    Parameters
    ----------
    sp : `~artpop.stars.populations.StellarPopulation`
        A stellar population object.
    scale_radius : float or `~astropy.units.Quantity`
        Scale radius of the source. If a float is given, the units are
        assumed to be `kpc`. Must be greater than zero.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    eta : float, optional
        Distance-duality parameter. The physical size is converted to an angle
        with ``D_A = D_L / [(1+z)^2 eta^2]``, Etherington's relation, which
        follows from photon conservation in any metric theory and is not a
        cosmological model. Exposed so a violation of duality can itself be
        probed. Default: 1.0.
    distance_angular : float or `~astropy.units.Quantity`, optional
        Override the angular-diameter distance outright, for callers who want
        distance, redshift and size to be literally independent. If a float is
        given the units are assumed to be Mpc.
    """

    def __init__(self, sp, scale_radius, xy_dim, pixel_scale, dx=0, dy=0,
                 labels=None, eta=1.0, distance_angular=None):

        self.sp = sp
        self.mag_limit = sp.mag_limit
        self.mag_limit_band = sp.mag_limit_band
        self.smooth_model = None
        self.distance_angular = _angular_distance(sp, eta, distance_angular)

        if self.mag_limit is not None and sp.frac_num_sampled < 1.0:
            _rs = check_units(scale_radius, 'kpc').to('Mpc').value
            _distance = self.distance_angular.to('Mpc').value
            _pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

            self.r_sky = np.arctan2(_rs, _distance) * u.radian.to('arcsec')
            self.r_sky *= u.arcsec
            r_pix = self.r_sky.to('pixel', u.pixel_scale(_pixel_scale)).value

            xy_dim = check_xy_dim(xy_dim)
            x_0, y_0 = xy_dim//2
            x_0 += dx
            y_0 += dy

            self.smooth_model = Plummer2D(x_0=x_0, y_0=y_0, scale_radius=r_pix)

        # angular-diameter distance; see the note in SersicSP.__init__
        self.xy_kw = dict(num_stars=sp.num_stars, scale_radius=scale_radius,
                          distance=self.distance_angular, xy_dim=xy_dim,
                          pixel_scale=pixel_scale, dx=dx, dy=dy,
                          random_state=sp.rng)

        _xy = plummer_xy(**self.xy_kw)
        super(PlummerSP, self).__init__(
            _xy, sp.mag_table, xy_dim, pixel_scale, labels)

    def mag_to_image_amplitude(self, m_tot, zpt):
        """
        Convert total magnitude into amplitude parameter for the smooth model.

        Parameters
        ----------
        m_tot : float
            Total magnitude in the smooth component of the system.
        zpt : float
            Photometric zero point.

        Returns
        -------
        mu_0 : float
            Central surface brightness of the Plummer distribution in
            mags per square arcsec.
        amplitude : float
            Amplitude parameter for the smooth model in image flux units.
        param_name : str
            Name of amplitude parameter (needed to set its value when
            generating the smooth model).
        """
        param_name = 'amplitude'
        area = np.pi * self.r_sky.to('arcsec').value**2
        flux = 10**(0.4 * (zpt - m_tot))
        mu_0 = zpt - 2.5 * np.log10(flux / area)
        amplitude = (flux / area) * self.pixel_scale.value**2
        return mu_0, amplitude, param_name


class MISTPlummerSSP(PlummerSP):
    """
    MIST simple stellar population with a Plummer spatial distribution.

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
    scale_radius : float or `~astropy.units.Quantity`
        Scale radius of the source. If a float is given, the units are
        assumed to be `kpc`. Must be greater than zero.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    total_mass : float or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. This
        mass accounts for stellar remnants, so the actual sampled mass will be
        less than the given value.
    a_lam : float or dict, optional
        Magnitude(s) of extinction. If float, the same extinction will be applied
        to all bands. If dict, the keys must be the same as the observational filters.
    add_remnants : bool, optional
        If True (default), apply scaling factor to total mass to account for
        stellar remnants in the form of white dwarfs, neutron stars,
        and black holes.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    mass_tolerance : float, optional
        Tolerance in the fractional difference between the input mass and the
        final mass of the population. The parameter is only used when
        `total_mass` is given.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    label_type : str, optional
        If not None, the type of labels to pass to the `~artpop.source.Source`
        object. Can be `phases` or an attribute of `~artpop.stars.MISTSSP`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, log_age, feh, phot_system, scale_radius,
                 distance, xy_dim, pixel_scale, num_stars=None,
                 total_mass=None, a_lam=0.0, add_remnants=True, mag_limit=None,
                 mag_limit_band=None, imf='kroupa', imf_kw=None,
                 mist_path=MIST_PATH, mass_tolerance=0.01, dx=0, dy=0,
                 label_type=None, random_state=None,
                 eta=1.0, distance_angular=None, **kwargs):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, total_mass=total_mass, a_lam=a_lam,
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state,
                           mag_limit=mag_limit, mag_limit_band=mag_limit_band,
                           add_remnants=add_remnants,
                           mass_tolerance=mass_tolerance, **kwargs)

        ssp = MISTSSP(**self.ssp_kw)
        labels = _check_label_type(ssp, label_type)

        super(MISTPlummerSSP, self).__init__(
            sp=ssp, scale_radius=scale_radius, xy_dim=xy_dim,
            pixel_scale=pixel_scale, dx=dx, dy=dy, labels=labels,
            eta=eta, distance_angular=distance_angular)


class UniformSSP(Source):
    """
    Simple stellar population with a uniform spatial distribution.

    Parameters
    ----------
    isochrone  : `~artpop.stars.Isochrone`
        Isochrone object.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    sb : float
        Surface brightness of of uniform stellar distribution.
    sb_band : str
        Photometric filter of ``sb``. Must be part of ``phot_system``.
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
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    ssp_kw : dict, optional
        Additional keyword arguments for `~artpop.stars.SSP`.
    label_type : str, optional
        If not None, the type of labels to pass to the `~artpop.source.Source`
        object. Must be an attribute of `~artpop.stars.SSP`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, isochrone, distance, xy_dim, pixel_scale, sb, sb_band,
                 a_lam=0.0, mag_limit=None, mag_limit_band=None, imf='kroupa',
                 imf_kw=None, label_type=None, random_state=None):

        self.isochrone = isochrone
        self.mag_limit = mag_limit
        self.mag_limit_band = mag_limit_band
        self.smooth_model = None

        mean_mag = isochrone.ssp_mag(
            sb_band, norm_type='number', m_min_norm=isochrone.m_min,
            m_max_norm=isochrone.m_max)

        stars_per_pix = constant_sb_stars_per_pix(
            sb, mean_mag, distance, pixel_scale)

        xy_dim = check_xy_dim(xy_dim)
        self.ssp_kw = {
            'imf': imf,
            'a_lam': a_lam,
            'mag_limit': mag_limit,
            'mag_limit_band': mag_limit_band,
            'distance': distance,
            'total_mass': None,
            'num_stars': int(stars_per_pix * np.multiply(*xy_dim)),
            'random_state': random_state,
            'imf_kw': imf_kw
        }

        if type(isochrone) == MISTIsochrone:
            self.ssp_kw['log_age'] = isochrone.log_age
            self.ssp_kw['feh'] = isochrone.feh
            self.ssp_kw['phot_system'] = isochrone.phot_system
            self.ssp_kw['v_over_vcrit'] = isochrone.v_over_vcrit
            self.ssp_kw['mist_path'] = isochrone.mist_path
            # this branch rebuilds the isochrone from scalars, so anything not
            # copied across is silently dropped -- which for the redshift and
            # dust corrections would mean a uniform population rendered at
            # z = 0 while its caller believed otherwise
            self.ssp_kw['version'] = isochrone.version
            self.ssp_kw['a_over_fe'] = isochrone.a_over_fe
            for attr in ('redshift', 'a_v_host', 'a_v_mw', 'r_v',
                         'extinction_law', 'nebular', 'photometry', 'bolometric',
                         'cool_giants'):
                self.ssp_kw[attr] = getattr(isochrone, attr)
            self.sp = MISTSSP(**self.ssp_kw)
            labels = _check_label_type(self.sp, label_type)
        else:
            self.sp = SSP(isochrone, **self.ssp_kw)
            labels = _check_label_type(self.sp, label_type, False)

        if mag_limit is not None and self.sp.frac_num_sampled < 1.0:
            self.smooth_model = Constant2D()

        self.xy_kw = dict(num_stars=self.sp.num_stars,
                          xy_dim=xy_dim, random_state=self.sp.rng)

        _xy = uniform_xy(**self.xy_kw)
        super(UniformSSP, self).__init__(
            _xy, self.sp.mag_table, xy_dim, pixel_scale, labels)

    def mag_to_image_amplitude(self, m_tot, zpt):
        """
        Convert total magnitude into amplitude parameter for the smooth model.

        Parameters
        ----------
        m_tot : float
            Total magnitude in the smooth component of the system.
        zpt : float
            Photometric zero point.

        Returns
        -------
        mu : float
            Constant surface brightness in mags per square arcsec.
        amplitude : float
            Amplitude parameter for the smooth model in image flux units.
        param_name : str
            Name of amplitude parameter (needed to set its value when
            generating the smooth model).
        """
        param_name = 'amplitude'
        area = np.multiply(*self.xy_dim)
        flux = 10**(0.4 * (zpt - m_tot))
        amplitude = flux / area
        mu = zpt - 2.5 * np.log10(amplitude / self.pixel_scale.value**2)
        return mu, amplitude, param_name


class MISTUniformSSP(UniformSSP):
    """
    MIST simple stellar population with a uniform spatial distribution.

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
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    sb : float
        Surface brightness of of uniform stellar distribution.
    sb_band : str
        Photometric filter of ``sb``. Must be part of ``phot_system``.
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
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    ssp_kw : dict, optional
        Additional keyword arguments for `~artpop.stars.SSP`.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    label_type : str, optional
        If not None, the type of labels to pass to the `~artpop.source.Source`
        object. Can be `phases` or an attribute of `~artpop.stars.MISTSSP`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, log_age, feh, phot_system, distance, xy_dim,
                 pixel_scale, sb, sb_band, a_lam=0.0, mag_limit=None,
                 mag_limit_band=None, imf='kroupa', imf_kw=None,
                 mist_path=MIST_PATH, label_type=None, random_state=None,
                 **kwargs):

        self.iso_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           mist_path=mist_path)
        self.iso_kw.update(kwargs)

        iso = MISTIsochrone(**self.iso_kw)

        super(MISTUniformSSP, self).__init__(
            isochrone=iso, distance=distance, xy_dim=xy_dim, sb=sb,
            sb_band=sb_band, a_lam=a_lam, mag_limit=mag_limit,
            mag_limit_band=mag_limit_band, pixel_scale=pixel_scale, imf=imf,
            imf_kw=imf_kw, label_type=label_type,
            random_state=random_state)
