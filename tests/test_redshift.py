"""
Redshift, K-corrections and the two dust frames.

Three tiers, in the order the physics is argued:

  Tier A -- pure math. No MIST, no spectral library, no network: analytic
            identities that pin the ``(1+z)`` bookkeeping and the
            photon-counting convention. These are the load-bearing ones,
            because a wrong-but-reasonable implementation passes every "K is
            small and smooth" sanity check and fails here by exactly
            ``2.5 log10(1+z)``.
  Tier X -- the two extinction frames. Host-internal dust reddens the star's
            rest-frame spectrum; Milky Way foreground attenuates observer-frame
            wavelengths. At z = 0 they coincide, which is why nobody has had to
            distinguish them.
  Tier B -- the spectral libraries and the cached grid. SKIPs when the
            libraries are not staged.
  Tier C -- integration with the isochrone, population and source layers.
            Runs on the shipped test isochrone pickle (it already carries
            ``log_g`` and ``[Fe/H]``), so no MIST download is needed except for
            the two tests that explicitly check MIST's own columns.

See HANDOFF_REDSHIFT.md for the argument each of these encodes.
"""
# Standard library
import os
import pickle
import tempfile
from unittest import TestCase, skipUnless

# Third-party
import numpy as np
from astropy.table import Table
from astropy import units as u

# Project
from artpop import data_dir, MIST_PATH
from artpop.filters import load_filter_system
from artpop.stars import Isochrone, SSP
from artpop.source import SersicSP
from artpop.image import IdealImager, moffat_psf
from artpop.kcorrect import photometry_curve_dir
from artpop.kcorrect import (KCorrectionGrid, C3KLibrary, TremblayWDLibrary,
                             BlackbodyLibrary, band_offset, k_correction,
                             band_weights, extinction_curve, planck_lam,
                             air_to_vac, spectra_path, DEFAULT_Z_GRID,
                             SOURCE_C3K, SOURCE_WD, SOURCE_BB)

iso_fn = os.path.join(data_dir, 'feh_m1.00_vvcrit0.4_LSST_10gyr_test_iso.pkl')

FILTERS = ['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y']
Z_REF = 0.05                    # the 200 Mpc edge of the ALVISS distance prior
_HAVE_C3K = C3KLibrary().available
_HAVE_WD = TremblayWDLibrary().available


def _lam_grid(n=40000):
    """A rest-frame wavelength grid wide enough for every LSST band at z <= 0.25."""
    return np.geomspace(500.0, 60000.0, n)


def _power_law(lam, alpha):
    """f_nu ~ nu^alpha, expressed as L_lam."""
    return lam ** (-(alpha + 2.0))


def _redshift_sed(lam, L_lam, z):
    """
    The observed-frame SED of a source at ``z``, on the observed grid.

    An independent statement of the same physics as blueshifting the filter:
    ``f_lam(lam_obs) ~ L_lam(lam_obs / (1+z)) / (1+z)``. Used to check the
    implementation against a route that shares none of its code.
    """
    lam_obs = lam * (1.0 + z)
    return lam_obs, L_lam / (1.0 + z)


class TestRedshiftAnalytic(TestCase):
    """Tier A: identities with no data dependence at all."""

    @classmethod
    def setUpClass(cls):
        cls.fs = load_filter_system('LSST')
        cls.curves = {b: cls.fs.get_trans(b) for b in FILTERS}
        cls.lam = _lam_grid()

    def test_a1_zero_redshift_is_exactly_zero(self):
        """A1: K = 0 at z = 0, for every band and every SED -- exactly 0.0.

        Not 'small': exactly zero, because the numerator and denominator are
        computed by the same code path and must be bit-identical. This is what
        makes ``redshift=0.0`` a byte-for-byte no-op all the way to a rendered
        image.
        """
        seds = [_power_law(self.lam, a) for a in (-2.0, 0.0, 1.0)]
        seds.append(planck_lam(self.lam, 5000.0))
        for band in FILTERS:
            for sed in seds:
                k = k_correction(self.lam, sed, *self.curves[band], 0.0)
                self.assertEqual(abs(k), 0.0)

    def test_a2_power_law_identity(self):
        """A2: K = -2.5(1+alpha)log10(1+z) for f_nu ~ nu^alpha, in any filter.

        The sharpest available test of the ``(1+z)`` prefactor: an
        implementation that drops it is wrong by exactly ``2.5 log10(1+z)``
        here while still looking perfectly smooth everywhere else.
        """
        worst = 0.0
        for band in FILTERS:
            for alpha in (-2.0, -1.0, 0.0, 1.0):
                sed = _power_law(self.lam, alpha)
                for z in (0.02, Z_REF, 0.2):
                    k = k_correction(self.lam, sed, *self.curves[band], z)
                    exact = -2.5 * (1.0 + alpha) * np.log10(1.0 + z)
                    worst = max(worst, abs(k - exact))
        self.assertLess(worst, 1e-5, f'worst power-law residual {worst:.3e} mag')

    def test_a3_alpha_minus_one_is_a_no_op_sed(self):
        """A3: alpha = -1 gives K = 0 in every band at every z -- a free no-op SED.

        The identity is exact, so whatever is left is the integrator, not the
        physics. On the 40 000-point grid the other tests use it is 5e-6 in
        LSST y (the widest, most sparsely sampled band); refining to 200 000
        points brings it to 3e-7, which is why this test refines rather than
        loosens. A8 measures that convergence directly.
        """
        lam = _lam_grid(200000)
        sed = _power_law(lam, -1.0)
        for band in FILTERS:
            for z in (0.02, Z_REF, 0.2):
                k = k_correction(lam, sed, *self.curves[band], z)
                self.assertLess(abs(k), 1e-6, f'{band} z={z}: {k:.3e}')

    def test_a4_blackbody_identity(self):
        """A4: K(z, T) = BC(T) - BC(T/(1+z)), exactly.

        A redshifted blackbody *is* a blackbody at ``T/(1+z)``, and carrying
        that through the bolometric-correction definition leaves no residual
        constant: the ``(1+z)^4`` from the band integral cancels against the
        ``T^4`` in ``sigma T^4``. This is the test that ties the K path and the
        BC path together with no free parameters, so a convention mismatch
        between them -- energy- versus photon-counting, or a stray ``(1+z)`` --
        cannot hide.
        """
        sigma = 5.670374419e-5
        worst = 0.0
        for band in FILTERS:
            tw, tt = self.curves[band]
            qw = None

            def bc(temp):
                w = band_weights(self.lam, tw, tt, 0.0, quad_weights=qw)
                integral = np.dot(planck_lam(self.lam, temp), w)
                return 2.5 * np.log10(integral) - 2.5 * np.log10(sigma * temp ** 4)

            for temp in (3500.0, 5800.0, 10000.0):
                for z in (Z_REF, 0.2):
                    k = k_correction(self.lam, planck_lam(self.lam, temp),
                                     tw, tt, z)
                    worst = max(worst, abs(k - (bc(temp) - bc(temp / (1 + z)))))
        self.assertLess(worst, 1e-5, f'worst blackbody residual {worst:.3e} mag')

    def test_a5_filter_shift_equivalence(self):
        """A5: blueshifting the filter == redshifting the SED.

        The implementation samples ``T((1+z)lam)`` on the rest-frame grid. Here
        the source is redshifted onto an observer grid instead and integrated
        against the unshifted curve -- a route that shares no code with the
        one under test.
        """
        for band in FILTERS:
            tw, tt = self.curves[band]
            for temp in (3500.0, 6000.0):
                sed = planck_lam(self.lam, temp)
                den = np.dot(sed, band_weights(self.lam, tw, tt, 0.0))
                for z in (0.02, Z_REF, 0.2):
                    lam_obs, sed_obs = _redshift_sed(self.lam, sed, z)
                    num = np.dot(sed_obs, band_weights(lam_obs, tw, tt, 0.0))
                    k_explicit = -2.5 * np.log10(num / den)
                    k = k_correction(self.lam, sed, tw, tt, z)
                    self.assertAlmostEqual(k, k_explicit, places=5)

    def test_a6_scale_invariance(self):
        """A6: K is invariant under L_lam -> c L_lam. The normalisation cancels.

        This is why the spectral library never needs to agree with MIST on
        absolute flux -- and, read the other way, why it must agree on *shape*,
        which is exactly where libraries differ.
        """
        sed = planck_lam(self.lam, 4500.0)
        for band in FILTERS:
            base = k_correction(self.lam, sed, *self.curves[band], Z_REF)
            for c in (1e-12, 3.7, 1e15):
                self.assertAlmostEqual(
                    k_correction(self.lam, c * sed, *self.curves[band], Z_REF),
                    base, places=12)

    def test_a7_composition(self):
        """A7: K(z1) then K(z2) on the once-redshifted SED == K((1+z1)(1+z2)-1)."""
        sed = planck_lam(self.lam, 4500.0)
        for band in FILTERS:
            tw, tt = self.curves[band]
            for z1, z2 in ((0.02, 0.03), (0.05, 0.10)):
                k1 = k_correction(self.lam, sed, tw, tt, z1)
                lam1, sed1 = _redshift_sed(self.lam, sed, z1)
                k2 = k_correction(lam1, sed1, tw, tt, z2)
                z_tot = (1 + z1) * (1 + z2) - 1
                k_tot = k_correction(self.lam, sed, tw, tt, z_tot)
                self.assertAlmostEqual(k1 + k2, k_tot, places=5)

    def test_a8_integrator_convergence(self):
        """A8: halving the wavelength step moves K by less than 1e-4 mag."""
        for band in FILTERS:
            tw, tt = self.curves[band]
            prev = None
            for n in (10000, 20000, 40000):
                lam = _lam_grid(n)
                k = k_correction(lam, planck_lam(lam, 4000.0), tw, tt, Z_REF)
                if prev is not None:
                    self.assertLess(abs(k - prev), 1e-4)
                prev = k


class TestExtinctionFrames(TestCase):
    """
    Tier X: the two dust frames.

    MIST's own ``A_V`` axis reddens the star's rest-frame spectrum, which is
    correct for dust inside the host galaxy. ALVISS's dust is Milky Way
    foreground, which attenuates observer-frame wavelengths. At z = 0 the two
    coincide; at z > 0 they do not, and applying one where the other belongs is
    a systematic that grows with redshift.
    """

    @classmethod
    def setUpClass(cls):
        cls.fs = load_filter_system('LSST')
        cls.curves = {b: cls.fs.get_trans(b) for b in FILTERS}
        cls.lam = _lam_grid()
        cls.sed = planck_lam(cls.lam, 4500.0)

    def _offset(self, band, z, a_h=0.0, a_mw=0.0, law='F99'):
        return band_offset(self.lam, self.sed, *self.curves[band], redshift=z,
                           a_v_host=a_h, a_v_mw=a_mw,
                           ext=extinction_curve(law))

    def test_x1_grey_screen_is_exactly_a_v(self):
        """X1: a wavelength-independent screen gives A_x = A_V exactly, in
        either frame and at any redshift.

        With a grey law the attenuation factors out of the integral, so the
        offset must be the K-correction plus the two A_V values and nothing
        else. Any leakage here is a misplaced ``(1+z)`` inside the dust term.
        """
        for band in ('LSST_u', 'LSST_r', 'LSST_y'):
            for z in (0.0, Z_REF, 0.2):
                k = self._offset(band, z, law='grey')
                for a_h, a_mw in ((0.7, 0.0), (0.0, 1.3), (0.4, 0.9)):
                    got = self._offset(band, z, a_h, a_mw, law='grey')
                    self.assertAlmostEqual(got - k, a_h + a_mw, places=10)

    def test_x2_frames_coincide_at_zero_redshift(self):
        """X2: at z = 0 host-frame and observer-frame dust are the same thing.

        This is the degeneracy that has let the distinction go unnoticed, and
        asserting it makes the z > 0 difference below a real measurement rather
        than a possible bug in one of the two paths.
        """
        for band in FILTERS:
            for a_v in (0.3, 1.0):
                self.assertAlmostEqual(self._offset(band, 0.0, a_h=a_v),
                                       self._offset(band, 0.0, a_mw=a_v),
                                       places=12)

    def test_x3_frames_separate_at_nonzero_redshift(self):
        """X3: at z > 0 they differ, and the difference grows with z.

        Measured rather than assumed: the test asserts the *ordering* and that
        the effect is real at the 0.001 mag level in the blue, not a specific
        number.
        """
        band, a_v = 'LSST_u', 1.0
        prev = 0.0
        for z in (0.02, 0.05, 0.10, 0.20):
            k = self._offset(band, z)
            diff = abs((self._offset(band, z, a_h=a_v) - k)
                       - (self._offset(band, z, a_mw=a_v) - k))
            self.assertGreater(diff, prev)
            prev = diff
        self.assertGreater(prev, 1e-3)

    def test_x4_zero_extinction_is_a_no_op(self):
        """X4: A_V = 0 changes nothing, bit for bit."""
        for band in FILTERS:
            for z in (0.0, Z_REF):
                self.assertEqual(self._offset(band, z, 0.0, 0.0),
                                 self._offset(band, z))

    def test_x5_cross_term_is_reported_not_assumed_zero(self):
        """X5: the two screens do not simply add, and the code never assumes so.

        Extinction in a broad band saturates -- the reddest photons survive --
        so ``A_x(A1 + A2) != A_x(A1) + A_x(A2)`` even in a single frame. The
        implementation computes one integral with both screens in it, so the
        cross term is carried exactly. This test asserts it is non-zero (i.e.
        that summing separate terms would be wrong) and small (i.e. that the
        one-integral form has not gone haywire).
        """
        band, z, a_h, a_mw = 'LSST_u', Z_REF, 0.8, 0.8
        k = self._offset(band, z)
        a1 = self._offset(band, z, a_h=a_h) - k
        a2 = self._offset(band, z, a_mw=a_mw) - k
        joint = self._offset(band, z, a_h, a_mw) - k
        cross = joint - a1 - a2
        self.assertGreater(abs(cross), 1e-4)
        self.assertLess(abs(cross), 0.2)


@skipUnless(_HAVE_C3K, f'C3K not staged under {spectra_path()}')
class TestKCorrectionGrid(TestCase):
    """Tier B: the spectral libraries and the cached grid."""

    @classmethod
    def setUpClass(cls):
        cls.grid = KCorrectionGrid.build('LSST', z_grid=np.linspace(0, 0.2, 11))

    def test_b1_interpolation_at_a_node_returns_the_node(self):
        """B1: interpolating at a grid node reproduces the stored value."""
        i_f, i_g, i_t, i_z = 4, 6, 40, 5
        want = self.grid.c3k[i_f, i_g, i_t, i_z]
        got, info = self.grid.interpolate(
            self.grid.c3k_logt[i_t], self.grid.c3k_logg[i_g],
            self.grid.feh_grid[i_f], self.grid.z_grid[i_z])
        self.assertEqual(info['source'][0], SOURCE_C3K)
        np.testing.assert_allclose(got[0], want, atol=1e-6)

    def test_b2_monotonic_in_redshift(self):
        """B2: a cool giant's K in LSST u increases monotonically with z.

        Cool stars are steeply rising in the blue, so blueshifting the filter
        into the falling part of their spectrum can only cost flux.
        """
        z = np.linspace(0.0, 0.2, 11)
        k = [self.grid.interpolate(np.log10(4000.0), 0.5, -1.0, zz,
                                   quantity='offset')[0][0, 0] for zz in z]
        self.assertTrue(np.all(np.diff(k) > 0), f'K(u) not monotone: {k}')

    def test_b3_k_orders_by_temperature_but_not_monotonically_everywhere(self):
        """B3: K orders by temperature on the giant branch, and turns in u and z.

        Measured at log g 2.5, [Fe/H] = -1, z = 0.05 through DP2's standard
        passbands (2026-10-06; with the syseng v1.1 curves the u minimum sat at
        5000 K, K(u) = +0.473 / +0.309 / +0.214 / +0.203 at 3500-5000 K):

        ===== ====== ====== ====== ====== ====== ====== ====== ======
        T (K)   3500   4000   4500   5000   5500   6000   7000   8000
        K(u)  +0.402 +0.240 +0.169 +0.181 +0.234 +0.304 +0.488 +0.637
        K(z)  +0.055 +0.019 -0.006 -0.027 -0.041 -0.052 -0.056 -0.046
        ===== ====== ====== ====== ====== ====== ====== ====== ======

        Both turns are a spectral edge crossing the band as the filter
        blueshifts: the Balmer jump at 3646 A for *u* (DP2's u is redder, pivot
        3707 A against 3665 A, so it turns at a cooler star), the Paschen jump
        at 8204 A for *z*. g, r, i and y are monotone throughout.

        The ordering is asserted where it holds -- the cool giant branch, where
        the flux and all of the SBF weight live -- and the turns are asserted
        too, because structure like this is precisely what a per-galaxy scalar
        cannot represent.
        """
        kw = dict(quantity='offset')
        cool = [3500.0, 4000.0, 4500.0]
        k_cool = np.array([self.grid.interpolate(np.log10(t), 2.5, -1.0, Z_REF,
                                                 **kw)[0][0] for t in cool])
        for j, band in enumerate(self.grid.bands):
            self.assertTrue(np.all(np.diff(k_cool[:, j]) < 0),
                            f'K({band}) not ordered on the giant branch: '
                            f'{k_cool[:, j]}')

        full = [3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 7000.0, 8000.0]
        k_all = np.array([self.grid.interpolate(np.log10(t), 2.5, -1.0, Z_REF,
                                                **kw)[0][0] for t in full])
        turns = {'LSST_u', 'LSST_z'}
        for j, band in enumerate(self.grid.bands):
            if band in turns:
                self.assertGreater(k_all[:, j].max(), k_all[-1, j] - 1e-12)
                self.assertFalse(np.all(np.diff(k_all[:, j]) < 0),
                                 f'K({band}) was expected to turn back up')
            else:
                self.assertTrue(np.all(np.diff(k_all[:, j]) < 0),
                                f'K({band}) not monotone: {k_all[:, j]}')

    def test_b4_save_load_round_trip_is_bitwise(self):
        """B4: a cached grid reloads bit for bit."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self.grid.save(os.path.join(tmp, 'grid.npz'))
            other = KCorrectionGrid.load(path)
        for name in self.grid._ARRAYS:
            a, b = getattr(self.grid, name), getattr(other, name)
            if a is None:
                self.assertIsNone(b, name)
                continue
            # the MARCS table holds NaN where MARCS has no model, by design
            self.assertTrue(np.array_equal(a, b, equal_nan=np.asarray(a).dtype.kind == 'f'), name)
        self.assertEqual(self.grid.bands, other.bands)
        self.assertEqual(self.grid.meta, other.meta)

    def test_b5_out_of_hull_is_flagged_never_extrapolated(self):
        """B5: every backend switch is recorded; nothing is served silently.

        The log g 5.5-6.5 gap between C3K's ceiling and Tremblay's floor is
        real -- five post-AGB rows on a real isochrone land in it -- and so is
        the >140 kK tail. Both are served by the blackbody fallback, which is
        exact for a blackbody, and both are flagged.
        """
        log_teff = np.log10([4500.0, 20000.0, 80000.0, 1200.0])
        log_g = np.array([2.5, 8.0, 6.0, 4.0])
        _, info = self.grid.interpolate(log_teff, log_g, -1.0, Z_REF)
        expect = [SOURCE_C3K,
                  SOURCE_WD if self.grid.has_wd else SOURCE_BB,
                  SOURCE_BB, SOURCE_BB]
        self.assertEqual(list(info['source']), expect)
        np.testing.assert_array_equal(info['fallback'],
                                      np.array(expect) == SOURCE_BB)

    def test_b5b_feh_below_the_library_floor_is_clipped_and_flagged(self):
        """B5b: MIST's BC tables reach [Fe/H] = -3.0 and C3K stops at -2.5.

        Widening the prior below the library floor must flag, not extrapolate.
        """
        _, info = self.grid.interpolate(np.log10(4500.0), 2.5, -3.0, Z_REF)
        self.assertEqual(info['n_feh_clipped'], 1)
        _, info = self.grid.interpolate(np.log10(4500.0), 2.5, -2.0, Z_REF)
        self.assertEqual(info['n_feh_clipped'], 0)

    def test_b6_library_covers_the_blueshifted_filters(self):
        """B6: at z_max the blueshifted filter stays inside the library's blue edge.

        Trivially true for C3K, whose grid starts at 100 A. Kept as a guard: it
        is the one coverage limit that would be silent, because a filter
        sampling off the end of the grid just integrates zeros.
        """
        z_max = float(self.grid.z_grid[-1])
        blue_edge = C3KLibrary().wave[0]
        for band in FILTERS:
            lam, trans = load_filter_system(
                'LSST', curve_dir=photometry_curve_dir('LSST')).get_trans(band)
            lam_rest_min = lam[trans > 0].min() / (1.0 + z_max)
            self.assertGreater(lam_rest_min, blue_edge)

    def test_b7_redshift_zero_offset_is_identically_zero(self):
        """B7: the z = 0 column of every backend IS its rest table, so the
        offset at z = 0 without dust is 0.0 bit for bit, at nodes and between
        them."""
        for arr, rest in ((self.grid.c3k, self.grid.c3k_rest),
                          (self.grid.wd, self.grid.wd_rest),
                          (self.grid.bb, self.grid.bb_rest)):
            if arr.size:
                np.testing.assert_array_equal(arr[..., 0, :], rest)
        lt = np.log10([3700.0, 5123.0, 9876.0, 23456.0, 60000.0])
        lg = np.array([0.7, 2.3, 4.1, 7.2, 6.0])
        off, _ = self.grid.interpolate(lt, lg, -1.37, 0.0, quantity='offset')
        self.assertEqual(float(np.nanmax(np.abs(off))), 0.0)

    def test_b8_redshift_outside_the_grid_raises(self):
        """B8: a redshift off the end of the grid raises rather than extrapolating."""
        with self.assertRaises(ValueError):
            self.grid.interpolate(np.log10(4500.0), 2.5, -1.0,
                                  float(self.grid.z_grid[-1]) + 0.05)

    @skipUnless(_HAVE_WD, 'Tremblay WD library not staged')
    def test_b9_library_seam_is_real_and_its_redshift_part_small(self):
        """B9: the C3K / Tremblay seam is real in absolute photometry, and
        the redshift part of it is small.

        C3K at log g 5.5 and Tremblay at log g 6.5 are not the same star
        computed twice -- one is a metal atmosphere, the other a pure-hydrogen
        DA -- so their absolute photometry genuinely steps across the join, and
        since 2026-10-06 (absolute synthetic magnitudes) that step is in our
        magnitudes, as it is in MIST's composite (at log Teff = 3.17609 in
        ``bcl/feh+0.00_afe+0.0.LSST`` every row with log g <= 6.0 carries an
        identical BC and log g = 6.5 jumps). Measured at [Fe/H] = -1,
        z = 0.05: the colour step across the join is 0.39 mag at 20 kK and
        0.49 mag at 40 kK, while the step in the K-colours is 0.054 and 0.018.

        For scale, the rows this affects carry under 0.1% of an old
        population's flux and none of its SBF weight.
        """
        fs = load_filter_system('LSST', curve_dir=photometry_curve_dir('LSST'))
        c3k, wd = C3KLibrary(), TremblayWDLibrary()
        lg_c, lt_c, cube = c3k.grid(-1.0)
        lg_w, lt_w, flux_w = wd.grid()

        def ab_mag(lam, f_nu, band):
            w = band_weights(lam, *fs.get_trans(band), 0.0, flux_unit='f_nu')
            ref = np.full_like(np.asarray(lam, dtype=float), 3631e-23)
            return -2.5 * np.log10(np.dot(f_nu, w) / np.dot(ref, w))

        for teff in (20000.0, 40000.0):
            i_t = int(np.argmin(abs(10 ** lt_c - teff)))
            i_g = int(np.argmin(abs(lg_c - 5.5)))
            j_t = int(np.argmin(abs(10 ** lt_w - teff)))
            j_g = int(np.argmin(abs(lg_w - 6.5)))
            m_c = [ab_mag(c3k.wave, cube[i_g, i_t], b) for b in FILTERS]
            m_w = [ab_mag(wd.wave, flux_w[j_g, j_t], b) for b in FILTERS]
            k_c = self.grid.interpolate(np.log10(teff), 5.5, -1.0, Z_REF,
                                        quantity='offset')[0][0]
            k_w = self.grid.interpolate(np.log10(teff), 6.5, -1.0, Z_REF,
                                        quantity='offset')[0][0]

            # colours, so the arbitrary normalisation of each library cancels
            step_mag = max(abs((m_c[i] - m_c[i + 1]) - (m_w[i] - m_w[i + 1]))
                           for i in range(len(FILTERS) - 1))
            step_k = max(abs((k_c[i] - k_c[i + 1]) - (k_w[i] - k_w[i + 1]))
                         for i in range(len(FILTERS) - 1))
            self.assertLess(step_k, 0.15, f'K seam at {teff:.0f} K too large')
            self.assertGreater(step_mag / step_k, 3.0,
                               f'the K-colour step at {teff:.0f} K is only '
                               f'{step_mag / step_k:.1f}x below the colour step')

    @skipUnless(_HAVE_WD, 'Tremblay WD library not staged')
    def test_b10_blackbody_bridges_the_log_g_gap(self):
        """B10: the fallback in the log g 5.5-6.5 gap sits between its neighbours.

        Nothing covers 5.5 < log g < 6.5, and a handful of post-AGB rows land
        there. The blackbody fallback is used, is flagged, and -- asserted here
        -- lands within the C3K-to-Tremblay step rather than somewhere
        unrelated, which is what would happen if it were being read on the
        wrong wavelength or flux convention.
        """
        teff = 40000.0
        kw = dict(quantity='offset')
        k_c = self.grid.interpolate(np.log10(teff), 5.5, -1.0, Z_REF, **kw)[0][0]
        k_w = self.grid.interpolate(np.log10(teff), 6.5, -1.0, Z_REF, **kw)[0][0]
        k_b, info = self.grid.interpolate(np.log10(teff), 6.0, -1.0, Z_REF, **kw)
        self.assertEqual(info['source'][0], SOURCE_BB)
        span = np.abs(k_c - k_w).max()
        self.assertLess(np.abs(k_b[0] - 0.5 * (k_c + k_w)).max(),
                        max(span, 0.05) * 1.5)


def _test_isochrone(redshift=0.0, grid=None, a_v_host=0.0, a_v_mw=0.0):
    """
    The shipped 10 Gyr test isochrone, optionally corrected to ``redshift``.

    It carries ``log_g`` and ``[Fe/H]`` already, so the whole integration tier
    runs with no MIST download and no network -- exactly as
    ``test_lf_sampling.py`` does for the sampler.
    """
    with open(iso_fn, 'rb') as f:
        table = Table(pickle.load(f))
    log_teff = np.asarray(table['log_Teff'], dtype=float)
    log_g = np.asarray(table['log_g'], dtype=float)
    feh = np.asarray(table['[Fe/H]'], dtype=float)
    mags = Table({f: np.asarray(table[f], dtype=float) for f in FILTERS})
    if redshift != 0.0 or a_v_host != 0.0 or a_v_mw != 0.0:
        delta, _ = grid.offsets(log_teff, log_g, feh, redshift, bands=FILTERS)
        mags = Table({f: mags[f] + delta[f] for f in FILTERS})
    return Isochrone(mini=table['initial_mass'], mact=table['star_mass'],
                     mags=mags, log_L=table['log_L'], log_Teff=log_teff,
                     log_g=log_g, feh=feh, redshift=redshift)


class TestRedshiftIntegration(TestCase):
    """
    Tier C: the isochrone, population and source layers.

    Runs on the shipped test isochrone, so nothing here needs MIST or a
    network. `TestRedshiftMIST` below covers the two claims that can only be
    made against MIST's own columns.
    """

    @classmethod
    def setUpClass(cls):
        cls.grid = (KCorrectionGrid.build('LSST',
                                          z_grid=np.linspace(0, 0.2, 11))
                    if _HAVE_C3K else None)
        cls.iso0 = _test_isochrone()
        cls.band = 'LSST_i'
        dm = 5 * np.log10(10e6) - 5
        cls.tip = float(np.asarray(cls.iso0.mag_table[cls.band]).min() + dm)

    def _ssp(self, iso, mag_limit=None, seed=42, total_mass=2e5, **kw):
        # a mag_limit and a modest mass by default: with neither, every star
        # down to the bottom of the isochrone is realized, which is minutes of
        # sampling for tests that only need the photometry. The tests that care
        # about where the limit falls set it explicitly.
        if mag_limit is None:
            mag_limit = self.tip + 2.0
        return SSP(iso, total_mass=total_mass, distance=10 * u.Mpc,
                   mag_limit=mag_limit, mag_limit_band=self.band,
                   random_state=seed, **kw)

    def test_c1_zero_redshift_is_bit_identical(self):
        """C1: ``redshift=0.0`` reproduces the current renderer exactly.

        The regression gate for the whole change. Not "agrees to a tolerance":
        ``np.array_equal`` on the magnitudes, the integrated component, the
        second moment and a rendered image. Every new parameter defaults to
        today's behaviour, so a single changed value here means the additive
        contract has been broken.
        """
        iso_z0 = _test_isochrone(redshift=0.0, grid=self.grid)
        for filt in FILTERS:
            self.assertTrue(np.array_equal(
                np.asarray(self.iso0.mag_table[filt]),
                np.asarray(iso_z0.mag_table[filt])), filt)

        a = self._ssp(self.iso0, mag_limit=self.tip + 3.0)
        b = self._ssp(iso_z0, mag_limit=self.tip + 3.0)
        for filt in FILTERS:
            self.assertTrue(np.array_equal(a.abs_mags[filt], b.abs_mags[filt]))
            self.assertEqual(a.integrated_abs_mags[filt],
                             b.integrated_abs_mags[filt])
            self.assertEqual(a._integrated_log_lumlum[filt],
                             b._integrated_log_lumlum[filt])
        self.assertEqual(a.dist_mod, b.dist_mod)
        self.assertEqual(a.distance_angular(), a.distance)

        src_a = SersicSP(a, 0.5 * u.kpc, 1.0, 0 * u.deg, 0.0, 61, 0.2)
        src_b = SersicSP(b, 0.5 * u.kpc, 1.0, 0 * u.deg, 0.0, 61, 0.2)
        imager = IdealImager()
        self.assertTrue(np.array_equal(
            imager.observe(src_a, self.band, psf=None).image,
            imager.observe(src_b, self.band, psf=None).image))

    @skipUnless(_HAVE_C3K, 'C3K not staged')
    def test_c3_total_flux_moves_by_the_k_of_the_stars_it_contains(self):
        """C3: at fixed D_L, the total magnitude moves by the K of the stars.

        Sampled with ``mag_limit=None`` and the same seed, so the two
        populations contain *the same stars*: the mass draw depends on the IMF
        and the random state, not on the magnitudes. Their total flux therefore
        differs by exactly the flux-weighted correction over those stars, with
        no Poisson term at all, and the assertion can be exact.

        The looser comparison -- against the isochrone's IMF-weighted
        ``ssp_mag`` -- is reported alongside rather than asserted. It differs by
        a few thousandths of a magnitude, and that difference is the sampling
        noise of a finite draw, not an error in the correction. Asserting on it
        would have made this test a measurement of the random seed.
        """
        iso_z = _test_isochrone(redshift=Z_REF, grid=self.grid)
        a = SSP(self.iso0, num_stars=20000, distance=10 * u.Mpc,
                random_state=99)
        b = SSP(iso_z, num_stars=20000, distance=10 * u.Mpc, random_state=99)
        self.assertTrue(np.array_equal(a.initial_masses, b.initial_masses),
                        'the same seed must draw the same stars')

        for filt in ('LSST_u', 'LSST_r', 'LSST_y'):
            f0 = 10 ** (-0.4 * a.abs_mags[filt])
            f1 = 10 ** (-0.4 * b.abs_mags[filt])
            realized = -2.5 * np.log10(f1.sum() / f0.sum())
            self.assertAlmostEqual(b.total_mag(filt) - a.total_mag(filt),
                                   realized, places=9)
            imf_weighted = (iso_z.ssp_mag(filt, norm_type='number')
                            - self.iso0.ssp_mag(filt, norm_type='number'))
            self.assertLess(abs(realized - imf_weighted), 0.05,
                            f'{filt}: the realized K {realized:+.4f} is far '
                            f'from the IMF-weighted {imf_weighted:+.4f}, which '
                            'is more than a finite draw explains')

    @skipUnless(_HAVE_C3K, 'C3K not staged')
    def test_c4_sbf_k_is_not_the_integrated_k(self):
        """C4: the SBF K-correction differs from the integrated-light one.

        SBF is ``sum n f^2 / sum n f``, so it is weighted in ``f^2`` and is
        dominated by the tip of the RGB -- the coolest, most strongly corrected
        part of the population. Integrated light is weighted in ``f``. The two
        therefore move by different amounts, and 0.2 mag on ``mbar`` is ~10% in
        SBF distance. Anything that applies one number per galaxy gets ``mbar``
        wrong by construction.
        """
        iso_z = _test_isochrone(redshift=Z_REF, grid=self.grid)
        diffs = []
        for filt in FILTERS:
            k_int = (iso_z.ssp_mag(filt, norm_type='number')
                     - self.iso0.ssp_mag(filt, norm_type='number'))
            k_sbf = iso_z.ssp_sbf_mag(filt) - self.iso0.ssp_sbf_mag(filt)
            diffs.append(k_sbf - k_int)
        diffs = np.array(diffs)
        self.assertGreater(np.abs(diffs).max(), 0.02,
                           f'K_SBF and K_int are indistinguishable: {diffs}')
        self.assertTrue(np.all(diffs > 0),
                        'the f^2 weighting must push mbar further than the '
                        f'integrated light, got {diffs}')

    @skipUnless(_HAVE_C3K, 'C3K not staged')
    def test_c5_second_moment_carries_ten_to_the_minus_point_eight_k(self):
        """C5: ``_integrated_log_lumlum`` picks up ``10**(-0.8 K)``.

        It is a *second* moment, so the correction enters squared. Getting this
        wrong is a pure multiplicative error on the injected SBF amplitude that
        no flux-conservation or shape test would notice -- the repo already
        knows this failure mode, in as many words, from
        ``check_sbf_moments_vs_artpop``.

        Applying K at the isochrone rather than at the population is what makes
        it right by construction, since both moments are built from the same
        magnitude column. The test recomputes the sum independently and also
        computes what the ``10**(-0.4 K)`` version would give, to show the two
        are separated by far more than the tolerance.
        """
        iso_z = _test_isochrone(redshift=Z_REF, grid=self.grid)
        ssp = self._ssp(iso_z, mag_limit=self.tip + 3.0)
        w, bright = ssp._lf_row_split()
        faint = ~bright
        n_total = ssp.n_total_expected
        log_dddd = 4 * np.log10((10 * u.pc).to('cm').value)

        for filt in ('LSST_u', 'LSST_r'):
            m_z = np.asarray(iso_z.mag_table[filt], dtype=float)
            m_0 = np.asarray(self.iso0.mag_table[filt], dtype=float)
            k = m_z - m_0

            right = n_total * np.sum(w[faint] * 10 ** (-0.8 * m_z[faint]))
            wrong = n_total * np.sum(w[faint] * 10 ** (-0.4 * m_z[faint])
                                     * 10 ** (-0.4 * m_0[faint]))
            got = ssp._integrated_log_lumlum[filt] - log_dddd
            self.assertAlmostEqual(got, np.log10(right), places=9)
            # the two powers must be far apart, or the test proves nothing
            self.assertGreater(abs(np.log10(right) - np.log10(wrong)), 1e-3,
                               f'the -0.8K and -0.4K forms differ by too '
                               f'little in {filt} to discriminate; K spans '
                               f'{k.min():.3f}..{k.max():.3f}')

    @skipUnless(_HAVE_C3K, 'C3K not staged')
    def test_c6_distance_and_redshift_are_independent(self):
        """C6: K does not move with D_L, and ``dist_mod`` does not move with z.

        The point of the whole design: cosmology is a target, not an input.
        """
        iso_z = _test_isochrone(redshift=Z_REF, grid=self.grid)
        near = self._ssp(iso_z, seed=1)
        far = SSP(iso_z, total_mass=2e5, distance=200 * u.Mpc,
                  mag_limit=self.tip + 2.0, mag_limit_band=self.band,
                  random_state=1)
        for filt in FILTERS:
            self.assertTrue(np.array_equal(
                np.asarray(near.isochrone.mag_table[filt]),
                np.asarray(far.isochrone.mag_table[filt])))
        z0 = SSP(self.iso0, total_mass=2e5, distance=50 * u.Mpc,
                 mag_limit=self.tip + 2.0, mag_limit_band=self.band,
                 random_state=1)
        z5 = SSP(iso_z, total_mass=2e5, distance=50 * u.Mpc,
                 mag_limit=self.tip + 2.0, mag_limit_band=self.band,
                 random_state=1)
        self.assertEqual(z0.dist_mod, z5.dist_mod)

    def test_c7_angular_size_grows_as_one_plus_z_squared(self):
        """C7: at fixed D_L and fixed physical r_eff, the angular size **grows**
        as ``(1+z)^2``.

        The direction is the part worth pinning. ``D_A = D_L/(1+z)^2`` is
        smaller than ``D_L``, so the same physical size subtends a *larger*
        angle -- the object gets bigger on the sky, and its surface brightness
        drops by the ``10 log10(1+z)`` that nothing in the code computes. In a
        real universe D_L and z are tied together and the angular size turns
        over near z ~ 1.6; here they are free of each other by construction, so
        the growth is monotonic.

        Etherington's duality sets the power, and it is a metric identity
        rather than a cosmology -- which is why it is allowed to be the one
        relation between two of the three free parameters.
        """
        iso_z = _test_isochrone(redshift=0.0)
        iso_z.redshift = Z_REF          # geometry only; photometry untouched
        a = self._ssp(self.iso0, total_mass=1e5)
        b = self._ssp(iso_z, total_mass=1e5)
        src_a = SersicSP(a, 0.5 * u.kpc, 1.0, 0 * u.deg, 0.0, 61, 0.2)
        src_b = SersicSP(b, 0.5 * u.kpc, 1.0, 0 * u.deg, 0.0, 61, 0.2)
        self.assertAlmostEqual(
            (src_a.distance_angular / src_b.distance_angular).value,
            (1 + Z_REF) ** 2, places=12)

        def r_sky(src):
            return np.arctan2(0.5e-3, src.distance_angular.to('Mpc').value)

        self.assertAlmostEqual(r_sky(src_b) / r_sky(src_a), (1 + Z_REF) ** 2,
                               places=9)
        self.assertGreater(r_sky(src_b), r_sky(src_a))
        # the star sampler must be handed the same distance as the smooth model
        self.assertEqual(src_b.xy_kw['distance'], src_b.distance_angular)

    def test_c8_eta_and_distance_angular_override_the_default(self):
        """C8: both escape hatches from the duality default work.

        ``eta`` is exposed so a duality violation can itself be probed;
        ``distance_angular`` overrides the relation entirely, for anyone who
        wants distance, redshift and size to be literally decoupled.
        """
        iso_z = _test_isochrone(redshift=0.0)
        iso_z.redshift = Z_REF
        ssp = self._ssp(iso_z, total_mass=1e5)
        default = SersicSP(ssp, 0.5 * u.kpc, 1.0, 0 * u.deg, 0.0, 61, 0.2)
        eta = SersicSP(ssp, 0.5 * u.kpc, 1.0, 0 * u.deg, 0.0, 61, 0.2, eta=1.1)
        self.assertAlmostEqual(
            (default.distance_angular / eta.distance_angular).value,
            1.1 ** 2, places=12)
        forced = SersicSP(ssp, 0.5 * u.kpc, 1.0, 0 * u.deg, 0.0, 61, 0.2,
                          distance_angular=7 * u.Mpc)
        self.assertEqual(forced.distance_angular, 7 * u.Mpc)
        # and the sampler must see the same distance the smooth model does
        self.assertEqual(forced.xy_kw['distance'], 7 * u.Mpc)

    @skipUnless(_HAVE_C3K, 'C3K not staged')
    def test_c9_composite_rejects_a_redshift_mismatch(self):
        """C9: adding two SSPs corrected to different redshifts must raise.

        Flux and luminosity-luminosity addition across populations is only
        meaningful if both were corrected to the same z.
        """
        iso_z = _test_isochrone(redshift=Z_REF, grid=self.grid)
        a = self._ssp(self.iso0)
        b = self._ssp(iso_z)
        with self.assertRaises(AssertionError):
            a + b
        self.assertEqual((a + self._ssp(self.iso0, seed=7)).redshift, 0.0)
        self.assertEqual((b + self._ssp(iso_z, seed=7)).redshift, Z_REF)

    @skipUnless(_HAVE_C3K, 'C3K not staged')
    def test_c10_mag_limit_split_moves_and_stays_an_exact_complement(self):
        """C10: K moves rows across ``mag_limit``, and the split stays exact.

        The bright and faint row sets must be exact complements: SBF split
        invariance depends on it, and if one side were computed with the
        correction and the other without, the complement would break silently.
        Applying K to the isochrone means the limit is compared against
        corrected magnitudes on both sides, so this holds by construction --
        which is worth asserting precisely because it is invisible.
        """
        iso_z = _test_isochrone(redshift=Z_REF, grid=self.grid)
        limit = self.tip + 3.0
        a = self._ssp(self.iso0, mag_limit=limit)
        b = self._ssp(iso_z, mag_limit=limit)
        w_a, bright_a = a._lf_row_split()
        w_b, bright_b = b._lf_row_split()
        self.assertTrue(np.array_equal(bright_a, ~(~bright_a)))
        self.assertTrue(np.array_equal(bright_b, ~(~bright_b)))
        self.assertGreater((bright_a != bright_b).sum(), 0,
                           'the correction did not move a single row across '
                           'the limit; pick a limit nearer the tip')
        for split, ssp in ((bright_a, a), (bright_b, b)):
            self.assertEqual(int(split.sum()) + int((~split).sum()), split.size)
            self.assertTrue(np.array_equal(ssp.sampled_row_mask, split))

    @skipUnless(_HAVE_C3K, 'C3K not staged')
    def test_c11_set_redshift_refuses_a_stale_smooth_component(self):
        """C11: ``set_redshift`` works, and refuses where it would lie.

        Both the bright/faint row split and the analytic sums over the faint
        rows are functions of z. Changing z after the fact on a population that
        has an integrated component would leave the sampled stars corrected
        and the smooth component not -- so it raises and names the fix.
        """
        iso_z = _test_isochrone(redshift=Z_REF, grid=self.grid)
        limited = self._ssp(iso_z, mag_limit=self.tip + 3.0)
        self.assertTrue(limited.has_integrated_component)
        with self.assertRaises(Exception):
            limited.set_redshift(0.1)


@skipUnless(os.path.isdir(os.path.join(MIST_PATH, 'MIST_v2.5_LSST'))
            and _HAVE_C3K, 'MIST v2.5 LSST grid or C3K not staged')
class TestRedshiftMIST(TestCase):
    """The claims that can only be made against MIST's own columns."""

    _kw = dict(log_age=10.0, feh=-1.0, phot_system='LSST', version='2.5')

    def test_c1_mist_photometry_is_bit_identical_to_stock(self):
        """C1 (MIST): ``photometry='mist'`` serves MIST's shipped columns
        byte for byte, and they are kept as ``mag_table_mist`` on the
        synthetic default; MIST with a redshift or dust raises."""
        from artpop import MISTIsochrone
        mist = MISTIsochrone(photometry='mist', **self._kw)
        synth = MISTIsochrone(**self._kw)
        for filt in FILTERS:
            self.assertTrue(np.array_equal(np.asarray(mist.mag_table[filt]),
                                           np.asarray(synth.mag_table_mist[filt])), filt)
        self.assertIsNone(mist.mag_table_rest)
        for bad in (dict(redshift=Z_REF), dict(a_v_mw=0.1), dict(a_v_host=0.1)):
            with self.assertRaises(ValueError):
                MISTIsochrone(photometry='mist', **bad, **self._kw)
        with self.assertRaises(ValueError):
            mist.set_redshift(Z_REF)
        with self.assertRaises(ValueError):
            MISTIsochrone(ab_or_vega='vega', **self._kw)

    def test_c2_columns_are_the_library_integral(self):
        """C2: every column is the grid's absolute magnitude minus 2.5 log L, at
        z = 0 as at z > 0; ``delta_mag`` is exactly what redshift moved."""
        from artpop import MISTIsochrone
        rest = MISTIsochrone(**self._kw)
        shifted = MISTIsochrone(redshift=Z_REF, **self._kw)
        I = shifted.isochrone_full
        grid = KCorrectionGrid.for_dust('LSST', bands=FILTERS)[0]
        want, _ = grid.magnitudes(I['log_Teff'], I['log_g'], shifted.feh_star,
                                  Z_REF, log_l=I['log_L'], bands=FILTERS)
        for filt in FILTERS:
            np.testing.assert_array_equal(np.asarray(shifted.mag_table[filt]), want[filt])
            np.testing.assert_array_equal(np.asarray(shifted.mag_table_rest[filt]),
                                          np.asarray(rest.mag_table[filt]))
            np.testing.assert_allclose(np.asarray(shifted.mag_table[filt]),
                                       np.asarray(shifted.mag_table_rest[filt])
                                       + shifted.delta_mag[filt], rtol=0, atol=1e-12)
        info = shifted.kcorr_info['LSST']
        self.assertGreater(info['n_c3k'] + info['n_marcs'], 0.7 * len(rest.mini))
        self.assertEqual(info['n_feh_clipped'], 0)
        p = shifted.photometry_info
        self.assertEqual(p['photometry'], 'synthetic')
        self.assertEqual(set(p['tables']['LSST']['curves']['sha1']), set(FILTERS))
        self.assertEqual(p['tables']['LSST']['bolometric'], 'teff')

    def test_c2b_set_redshift_recomputes_rather_than_shifts(self):
        """C2b: ``set_redshift`` integrates afresh: z = 0.05 -> 0.10 equals a
        direct z = 0.10 build, and back to 0 equals the z = 0 build."""
        from artpop import MISTIsochrone
        iso = MISTIsochrone(redshift=Z_REF, **self._kw)
        direct = MISTIsochrone(redshift=0.10, **self._kw)
        iso.set_redshift(0.10)
        for filt in FILTERS:
            np.testing.assert_allclose(np.asarray(iso.mag_table[filt]),
                                       np.asarray(direct.mag_table[filt]),
                                       rtol=0, atol=1e-12)
        iso.set_redshift(0.0)
        stock = MISTIsochrone(**self._kw)
        for filt in FILTERS:
            np.testing.assert_allclose(np.asarray(iso.mag_table[filt]),
                                       np.asarray(stock.mag_table[filt]),
                                       rtol=0, atol=1e-12)

    def test_c2c_bolometric_teff_is_a_grey_shift(self):
        """C2c: ``bolometric='teff'`` (default) sits a grey
        -2.5 log10(INT F / sigma T^4) above ``'spectrum'``: every band of a star
        moves by the same amount, so colours do not change."""
        from artpop import MISTIsochrone
        a = MISTIsochrone(bolometric='spectrum', **self._kw)
        b = MISTIsochrone(**self._kw)
        src = a.kcorr_info['LSST']['source'] == SOURCE_C3K
        d = np.stack([np.asarray(b.mag_table[f]) - np.asarray(a.mag_table[f]) for f in FILTERS])
        # grey to the cubic interpolation of a ratio that is itself smooth
        self.assertLess(float(np.max(np.ptp(d[:, src], axis=0))), 2e-3)
        self.assertLess(float(np.median(d[:, src])), 0.0)


@skipUnless(_HAVE_C3K, f'C3K not staged under {spectra_path()}')
class TestC3KEmptyCells(TestCase):
    """
    Tier B (S-51, 2026-09-28): C3K's empty cells. FSPS ships C3K on a full
    (log g, Teff) rectangle and marks cells with no ATLAS12 model with a
    constant f_nu = 1e-33, below FSPS's own missing threshold (1e-30). MIST
    v2.5's BC table fills exactly those cells with blackbody BCs, so the K
    table serves them with a blackbody at the cell's Teff and flags every row
    that touches one. Before this, the floor was integrated as a flat-f_nu
    "star".
    """

    @classmethod
    def setUpClass(cls):
        cls.grid = KCorrectionGrid.build('LSST', z_grid=np.array([0.0, 0.05]))
        cls.lib = C3KLibrary()

    def test_e1_empty_cells_are_fsps_floor(self):
        """E1: 399 cells at [Fe/H] = 0 hold the 1e-33 floor and nothing else."""
        from artpop.kcorrect import c3k_missing_mask
        g, t, f = self.lib.grid(0.0)
        miss = c3k_missing_mask(f, self.lib.wave)
        self.assertEqual(int(miss.sum()), 399)
        self.assertTrue(np.all(np.asarray(f)[miss] == np.float32(1e-33)))
        i0 = int(np.argmin(abs(self.grid.feh_grid - 0.0)))
        np.testing.assert_array_equal(self.grid.c3k_bbfill[i0], miss)

    def test_e2_empty_node_holds_the_blackbody_offset(self):
        """E2: at an empty node the table holds a blackbody's Delta m at that
        cell's Teff; at a genuine node, the model's own Delta m."""
        from artpop.kcorrect import c3k_missing_mask
        g, t, f = self.lib.grid(0.0)
        lam = np.asarray(self.lib.wave, dtype=float)
        miss = c3k_missing_mask(f, lam)
        fs = load_filter_system('LSST', bands=['LSST_r'], curve_dir=photometry_curve_dir('LSST'))
        tw, tt = fs.get_trans('LSST_r')
        i0 = int(np.argmin(abs(self.grid.feh_grid - 0.0)))
        b = self.grid.bands.index('LSST_r')
        ig, it = map(int, np.argwhere(miss)[len(np.argwhere(miss)) // 2])
        bb = planck_lam(lam, 10 ** t[it]) * lam ** 2
        want = band_offset(lam, bb, tw, tt, 0.05, flux_unit='f_nu')
        got = float(self.grid.c3k[i0, ig, it, 1, b]) - float(self.grid.c3k_rest[i0, ig, it, b])
        self.assertAlmostEqual(got, want, places=5)
        jg, jt = map(int, np.argwhere(~miss)[0])
        want = band_offset(lam, np.asarray(f[jg, jt], float), tw, tt, 0.05, flux_unit='f_nu')
        got = float(self.grid.c3k[i0, jg, jt, 1, b]) - float(self.grid.c3k_rest[i0, jg, jt, b])
        self.assertAlmostEqual(got, want, places=5)

    def test_e3_rows_touching_empty_cells_are_flagged(self):
        """E3: a 45 kK, log g 3.3 row (next to the Eddington limit) is flagged;
        a 5000 K dwarf is not; the flag is only ever set on C3K rows."""
        _, info = self.grid.interpolate([np.log10(4.5e4), np.log10(5e3)],
                                        [3.3, 4.5], [0.0, 0.0], 0.05)
        self.assertEqual(info['c3k_bbfill'].tolist(), [True, False])
        self.assertEqual(info['n_c3k_bbfill'], 1)

    def test_e4_cache_key_is_versioned(self):
        """E4: a table written before the fix can never be loaded for it."""
        from artpop.kcorrect import KCORR_TABLE_VERSION
        self.assertGreaterEqual(KCORR_TABLE_VERSION, 4)
        self.assertTrue(KCorrectionGrid.cache_key('LSST').startswith(
            f'kcorr_v{KCORR_TABLE_VERSION}_'))


@skipUnless(_HAVE_C3K, f'C3K not staged under {spectra_path()}')
class TestDustAxes(TestCase):
    """
    Tier D: A_V_host and A_V_mw as axes of the K table (F3_dust.ipynb 4.6).

    The axes were sized to keep the interpolation error in A_V under 1 mmag
    over every library spectrum, band and redshift; these tests hold the
    implementation to that and to the out-of-range rule (computed exactly,
    with a warning, never extrapolated).
    """

    @classmethod
    def setUpClass(cls):
        from artpop.kcorrect import DEFAULT_AV_HOST_GRID, DEFAULT_AV_MW_GRID
        cls.tmp = tempfile.mkdtemp(prefix='kcorr_axes_')
        cls.AH, cls.AM = DEFAULT_AV_HOST_GRID, DEFAULT_AV_MW_GRID
        cls.kw = dict(bands=FILTERS, z_grid=np.array([0.0, 0.05, 0.10, 0.15, 0.20]))
        cls.G = KCorrectionGrid.cached('LSST', cache_dir=cls.tmp, a_v_host_grid=cls.AH,
                                       a_v_mw_grid=cls.AM, **cls.kw)
        # stars across the CMD, off every stellar-parameter node, at a z node
        cls.lt = np.log10(np.array([3800.0, 4200.0, 5000.0, 5800.0, 9000.0]))
        cls.lg = np.array([4.8, 1.5, 2.6, 4.4, 4.0])
        cls.fe, cls.z = -1.3, 0.15

    def _baked(self, h, m):
        return KCorrectionGrid.build('LSST', a_v_host=h, a_v_mw=m, **self.kw)

    def _both(self, h, m):
        a, _ = self.G.offsets(self.lt, self.lg, self.fe, self.z, a_v_host=h, a_v_mw=m)
        b, _ = self._baked(h, m).offsets(self.lt, self.lg, self.fe, self.z)
        return max(float(np.max(np.abs(a[k] - b[k]))) for k in FILTERS)

    def test_d1_axis_nodes_equal_a_baked_table(self):
        """D1: at a node of both dust axes the table IS the baked table."""
        self.assertTrue(self.G.has_dust_axes)
        self.assertEqual(self.G.c3k.shape[-3:-1], (self.AH.size, self.AM.size))
        self.assertEqual(float(self.AM[-1]), 2.0)        # crowded-patch cap (2026-10-05)
        for h, m in ((0.5, 0.25), (1.0, 0.5), (0.25, 0.0), (0.75, 1.25), (1.0, 2.0)):
            self.assertLess(self._both(h, m), 1e-9, (h, m))

    def test_d2_cell_centres_within_one_mmag(self):
        """D2: halfway between dust nodes -- worst case for linear -- < 1 mmag."""
        for h, m in ((0.125, 0.125), (0.375, 0.375), (0.875, 0.375), (0.625, 0.125),
                     (0.375, 0.875), (0.625, 1.375), (0.875, 1.875)):
            self.assertLess(self._both(h, m), 1e-3, (h, m))

    def test_d3_for_dust_routes_and_warns_once(self):
        """D3: none -> dust-free table; inside -> axes; outside -> exact + one warning."""
        kw = dict(cache_dir=self.tmp, **self.kw)
        g0, d0 = KCorrectionGrid.for_dust('LSST', 0.0, 0.0, **kw)
        self.assertFalse(g0.has_dust_axes); self.assertEqual(d0, {})
        g1, d1 = KCorrectionGrid.for_dust('LSST', 0.3, 0.2, **kw)
        self.assertTrue(g1.has_dust_axes); self.assertEqual(d1, dict(a_v_host=0.3, a_v_mw=0.2))
        KCorrectionGrid._WARNED.discard(('LSST', 1.2, 0.2))
        with self.assertLogs('ArtPop Logger', level='WARNING') as cm:
            g2, d2 = KCorrectionGrid.for_dust('LSST', 1.2, 0.2, **kw)
        self.assertTrue(any('outside the K table' in r for r in cm.output))
        self.assertFalse(g2.has_dust_axes); self.assertEqual(d2, {})
        self.assertEqual((g2.meta['a_v_host'], g2.meta['a_v_mw']), (1.2, 0.2))
        a, _ = g2.offsets(self.lt, self.lg, self.fe, self.z)
        b, _ = self._baked(1.2, 0.2).offsets(self.lt, self.lg, self.fe, self.z)
        for k in FILTERS:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertFalse(any('avh1.200' in f for f in os.listdir(self.tmp)),
                         'an out-of-range pair must not leave a cache file per object')
        import logging
        seen = []
        h = logging.Handler(); h.emit = lambda rec: seen.append(rec)
        lg = logging.getLogger('ArtPop Logger'); lg.addHandler(h)
        try:
            KCorrectionGrid.for_dust('LSST', 1.2, 0.2, **kw)
        finally:
            lg.removeHandler(h)
        self.assertEqual([r for r in seen if r.levelno >= logging.WARNING], [], 'warn once per pair')

    def test_d4_lookups_refuse_what_the_table_cannot_serve(self):
        """D4: out-of-axis dust on the axis table, or a different baked pair, raises."""
        with self.assertRaises(ValueError):
            self.G.offsets(self.lt, self.lg, self.fe, self.z, a_v_host=1.5, a_v_mw=0.1)
        with self.assertRaises(ValueError):
            self.G.offsets(self.lt, self.lg, self.fe, self.z, a_v_host=0.1, a_v_mw=-0.1)
        with self.assertRaises(ValueError):
            self.G.offsets(self.lt, self.lg, self.fe, self.z, a_v_host=0.1, a_v_mw=2.1)
        with self.assertRaises(ValueError):
            self._baked(0.2, 0.0).offsets(self.lt, self.lg, self.fe, self.z, a_v_host=0.3)

    def test_d5_cache_key_and_round_trip(self):
        """D5: the key names the axes; save/load keeps them; an old-format file loads baked."""
        k_ax = KCorrectionGrid.cache_key('LSST', a_v_host_grid=self.AH, a_v_mw_grid=self.AM, **self.kw)
        k_other = KCorrectionGrid.cache_key('LSST', a_v_host_grid=self.AH, a_v_mw_grid=[0.0, 0.5], **self.kw)
        k_baked = KCorrectionGrid.cache_key('LSST', **self.kw)
        self.assertEqual(len({k_ax, k_other, k_baked}), 3)
        path = os.path.join(self.tmp, 'roundtrip.npz')
        self.G.save(path)
        L = KCorrectionGrid.load(path)
        np.testing.assert_array_equal(L.av_host_grid, self.AH)
        np.testing.assert_array_equal(L.c3k, self.G.c3k)
        B = self._baked(0.0, 0.0)
        d = {k: np.asarray(getattr(B, k)) for k in KCorrectionGrid._ARRAYS if not k.startswith('av_')}
        import json
        d['bands'] = np.array(B.bands); d['meta_json'] = np.array(json.dumps(B.meta))
        old = os.path.join(self.tmp, 'old_format.npz')
        np.savez(old, **d)
        self.assertFalse(KCorrectionGrid.load(old).has_dust_axes)

    def test_d7_out_of_range_warning_survives_muted_artpop_logger(self):
        """D7: the pipeline mutes 'ArtPop Logger' (ERROR); the dust warning still shows."""
        import logging
        parent = logging.getLogger('ArtPop Logger')
        old = parent.level
        parent.setLevel(logging.ERROR)                  # what simulate_catalog does at import
        KCorrectionGrid._WARNED.discard(('LSST', 1.1, 0.6))
        try:
            with self.assertLogs('ArtPop Logger', level='WARNING') as cm:
                KCorrectionGrid.for_dust('LSST', 1.1, 0.6, cache_dir=self.tmp, **self.kw)
        finally:
            parent.setLevel(old)
        self.assertTrue(any('outside the K table' in r for r in cm.output))

    @skipUnless(os.path.isdir(os.path.join(MIST_PATH, 'MIST_v2.5_LSST')), 'MIST v2.5 LSST not staged')
    def test_d6_isochrone_uses_the_axes_to_one_mmag(self):
        """D6: MISTIsochrone(a_v_mw=...) at z > 0 goes through the axes, to < 1 mmag of exact."""
        from artpop import MISTIsochrone
        kw = dict(log_age=10.0, feh=-1.0, phot_system='LSST', version='2.5', redshift=0.05)
        via_axes = MISTIsochrone(a_v_host=0.4, a_v_mw=0.3,
                                 kcorr_kw=dict(cache_dir=self.tmp, z_grid=self.kw['z_grid']), **kw)
        exact = MISTIsochrone(a_v_host=0.4, a_v_mw=0.3, kcorr_grid=self._baked(0.4, 0.3), **kw)
        for f in FILTERS:
            d = np.abs(np.asarray(via_axes.mag_table[f]) - np.asarray(exact.mag_table[f]))
            self.assertLess(float(np.nanmax(d)), 1e-3, f)


@skipUnless(os.path.isdir(os.path.join(MIST_PATH, 'MIST_v2.5_LSST'))
            and _HAVE_C3K, 'MIST v2.5 LSST grid or C3K not staged')
class TestRepeatedIsochroneNoLeak(TestCase):
    """
    C12: building the same redshifted, dusty isochrone twice in one process
    gives the same magnitudes -- with the MIST binary cache bypassed, which is
    the path an unwritable MIST directory takes. Before the fix the uncached
    reader handed out its own array, the first isochrone's in-place K/dust
    correction leaked into it, and the second came out corrected twice.
    """

    def test_c12_second_isochrone_is_not_corrected_twice(self):
        from artpop import MISTIsochrone
        old = os.environ.get('ARTPOP_MIST_CACHE')
        os.environ['ARTPOP_MIST_CACHE'] = '0'
        try:
            kw = dict(log_age=10.0, feh=-1.0, phot_system='LSST', version='2.5',
                      redshift=0.05, a_v_mw=0.3)
            first, second = MISTIsochrone(**kw), MISTIsochrone(**kw)
            stock = MISTIsochrone(log_age=10.0, feh=-1.0, phot_system='LSST', version='2.5')
        finally:
            if old is None:
                os.environ.pop('ARTPOP_MIST_CACHE', None)
            else:
                os.environ['ARTPOP_MIST_CACHE'] = old
        for f in FILTERS:
            np.testing.assert_array_equal(np.asarray(first.mag_table[f]), np.asarray(second.mag_table[f]))
            np.testing.assert_array_equal(np.asarray(first.mag_table_rest[f]), np.asarray(stock.mag_table[f]))


@skipUnless(_HAVE_C3K, f'C3K not staged under {spectra_path()}')
class TestSyntheticPhotometry(TestCase):
    """
    Tier S (2026-10-06): the table holds absolute AB magnitudes of a 1 L_sun
    star, integrated from the library -- every rendered magnitude comes from
    here, z = 0 included. These pin the absolute zero point, the identity
    with `band_offset`, the blackbody, the interpolation and the provenance.
    """

    @classmethod
    def setUpClass(cls):
        # 'spectrum': S1's independent route normalises by the spectrum's integral
        cls.grid = KCorrectionGrid.build('LSST', z_grid=np.array([0.0, 0.05, 0.1]),
                                         bolometric='spectrum')
        cls.lib = C3KLibrary()
        cls.fs = load_filter_system('LSST', curve_dir=photometry_curve_dir('LSST'))

    def _direct(self, lam, f_nu, band, z=0.0):
        """Independent route: AB magnitude of L_sun * f / INT f at 10 pc,
        np.trapezoid, f_nu -> f_lam by hand, the source redshifted rather than
        the filter blueshifted, integrated in the OBSERVER frame on the
        redshifted library grid (the library resolves the lines; DP2's 5 A
        curve grid would not)."""
        from artpop.kcorrect import L_SUN, FOUR_PI_D10_SQ, C_AA
        tw, tt = self.fs.get_trans(band)
        f_lam = np.asarray(f_nu, float) * C_AA / lam ** 2
        L = f_lam / np.trapezoid(f_lam, lam) * L_SUN / FOUR_PI_D10_SQ
        # observed f_lam(l_obs) = L(l_obs / (1+z)) / (1+z), luminosity distance out
        l_obs, obs = lam * (1 + z), L / (1 + z)
        T = np.interp(l_obs, tw, tt, left=0.0, right=0.0)
        f0 = 3631e-23 * C_AA / l_obs ** 2
        return -2.5 * np.log10(np.trapezoid(obs * T * l_obs, l_obs) / np.trapezoid(f0 * T * l_obs, l_obs))

    def test_s1_node_equals_an_independent_integral(self):
        """S1: at C3K nodes the table equals an independently coded AB integral
        to < 0.3 mmag, at z = 0 and z = 0.1 (the two routes integrate on
        different grids -- library vs curve -- which costs ~0.1 mmag in u for
        cool stars; a zero-point or normalisation error would be >> 1 mmag)."""
        lg, lt, f = self.lib.grid(-1.0)
        lam = np.asarray(self.lib.wave, float)
        i0 = list(self.grid.feh_grid).index(-1.0)
        for ig, it in ((10, 30), (6, 20), (12, 55), (4, 15)):
            for iz, z in ((0, 0.0), (2, 0.1)):
                for b, band in enumerate(self.grid.bands):
                    want = self._direct(lam, f[ig, it], band, z)
                    self.assertLess(abs(float(self.grid.c3k[i0, ig, it, iz, b]) - want), 3e-4,
                                    (ig, it, z, band))

    def test_s2_redshift_part_is_band_offset(self):
        """S2: M(z) - M(0) at a node is `band_offset` of the node's spectrum."""
        lg, lt, f = self.lib.grid(0.0)
        lam = np.asarray(self.lib.wave, float)
        for ig, it in ((10, 30), (6, 20)):
            got, _ = self.grid.interpolate(lt[it], lg[ig], 0.0, 0.05, quantity='offset')
            for b, band in enumerate(self.grid.bands):
                want = band_offset(lam, f[ig, it], *self.fs.get_trans(band), redshift=0.05,
                                   flux_unit='f_nu')
                self.assertLess(abs(got[0, b] - want), 1e-5, band)

    def test_s3_blackbody_is_analytic(self):
        """S3: the blackbody backend is a 1 L_sun blackbody, L_lam = L pi B /
        (sigma T^4), on a grid wide enough for the Planck integral."""
        from artpop.kcorrect import L_SUN, FOUR_PI_D10_SQ, SIGMA_SB, C_AA
        for T in (40000.0, 2.0e5):
            lt = np.log10(T)
            got, info = self.grid.interpolate(lt, 6.0, -1.0, 0.0)
            self.assertEqual(info['source'][0], SOURCE_BB)
            for b, band in enumerate(self.grid.bands):
                tw, tt = self.fs.get_trans(band)
                L = np.pi * planck_lam(tw, T) * 1e-8 / (SIGMA_SB * T ** 4) * L_SUN / FOUR_PI_D10_SQ
                f0 = 3631e-23 * C_AA / tw ** 2
                want = -2.5 * np.log10(np.trapezoid(L * tt * tw, tw) / np.trapezoid(f0 * tt * tw, tw))
                # T is between blackbody nodes: cubic in log T on a 0.05 dex mesh
                self.assertLess(abs(got[0, b] - want), 2e-3, (T, band))

    def test_s4_cubic_beats_linear_on_held_out_nodes(self):
        """S4: drop every other log Teff node, interpolate the dropped ones:
        cubic is better than linear in every band, and its 95th-percentile error
        at TWICE the native spacing (4000-10000 K, log g 1-5, [Fe/H] = -1) is
        < 40 mmag in u and < 15 in g..y (measured 31, 13, 5, 6, 6, 6). For a
        fourth-order scheme the error at the native spacing is ~1/16 of that."""
        import copy
        g = self.grid
        h = copy.copy(g)
        h.c3k_logt = g.c3k_logt[::2]
        h.c3k, h.c3k_rest, h.c3k_bbfill = g.c3k[:, :, ::2], g.c3k_rest[:, :, ::2], g.c3k_bbfill[:, :, ::2]
        T = 10 ** g.c3k_logt
        it = np.flatnonzero((np.arange(T.size) % 2 == 1) & (T > 4000) & (T < 10000))
        ig = np.flatnonzero((g.c3k_logg >= 1) & (g.c3k_logg <= 5))
        i0 = list(g.feh_grid).index(-1.0)
        tt, gg = np.meshgrid(it, ig, indexing='ij')
        tt, gg = tt.ravel(), gg.ravel()
        keep = ~g.c3k_bbfill[i0, gg, tt]
        tt, gg = tt[keep], gg[keep]
        truth = g.c3k_rest[i0, gg, tt].astype(float)
        cub, info_c = h.interpolate(g.c3k_logt[tt], g.c3k_logg[gg], -1.0, 0.0, quantity='rest')
        lin, _ = h.interpolate(g.c3k_logt[tt], g.c3k_logg[gg], -1.0, 0.0, quantity='rest',
                               method='linear')
        ok = ~info_c['interp_linear']
        e_c = np.nanpercentile(np.abs(cub - truth)[ok], 95, axis=0)
        e_l = np.nanpercentile(np.abs(lin - truth)[ok], 95, axis=0)
        self.assertTrue(np.all(e_c < e_l), (e_c, e_l))
        self.assertTrue(np.all(e_c < np.array([0.040, 0.015, 0.015, 0.015, 0.015, 0.015])), e_c)

    def test_s5_cache_key_follows_the_curves_and_convention(self):
        """S5: changing one number in one curve file, or the bolometric
        convention, changes the cache key -- a table can never be served for
        passbands it was not integrated through."""
        import shutil
        from artpop.kcorrect import photometry_curve_dir
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(os.path.join(photometry_curve_dir('LSST'), 'LSST'), os.path.join(tmp, 'LSST'))
            k0 = KCorrectionGrid.cache_key('LSST', curve_dir=tmp)
            self.assertEqual(k0, KCorrectionGrid.cache_key('LSST'))
            path = os.path.join(tmp, 'LSST', 'LSST_u.csv')
            lines = open(path).read().splitlines()
            w, t = lines[200].split(',')
            lines[200] = f'{w},{float(t) * 1.001!r}'
            open(path, 'w').write('\n'.join(lines) + '\n')
            self.assertNotEqual(k0, KCorrectionGrid.cache_key('LSST', curve_dir=tmp))
        self.assertNotEqual(KCorrectionGrid.cache_key('LSST'),
                            KCorrectionGrid.cache_key('LSST', bolometric='spectrum'))

    def test_s6_meta_states_the_assumptions(self):
        """S6: the table names its library, curves (sha1 per file, provenance),
        bolometric convention and L_sun."""
        m = self.grid.meta
        self.assertIn('C3K', m['library']['c3k'])
        self.assertEqual(set(m['curves']['sha1']), set(FILTERS))
        self.assertIn('standard_passband', m['curves']['provenance'])
        from artpop.kcorrect import DEFAULT_BOLOMETRIC
        self.assertEqual(m['bolometric'], 'spectrum')      # as this class builds it
        self.assertEqual(DEFAULT_BOLOMETRIC, 'teff')
        self.assertEqual(m['l_sun_erg_s'], 3.828e33)

    def test_s7_lsst_photometry_uses_dp2_passbands(self):
        """S7: LSST synthetic photometry integrates through DP2's
        ``standard_passband`` (``passbands/LSST``); the inherited syseng v1.1
        curves that feed the imager stay selectable with ``curve_dir``, and
        name a different table."""
        from artpop.kcorrect import photometry_curve_dir
        from artpop.filters import filter_curve_dir
        root = photometry_curve_dir('LSST')
        self.assertEqual(os.path.basename(root.rstrip(os.sep)), 'passbands')
        self.assertIn('fgcmcal', open(os.path.join(root, 'LSST', 'PROVENANCE.md')).read())
        self.assertNotEqual(KCorrectionGrid.cache_key('LSST'),
                            KCorrectionGrid.cache_key('LSST', curve_dir=filter_curve_dir()))
        self.assertEqual(photometry_curve_dir('Roman'), filter_curve_dir())


_HAVE_MARCS = __import__('artpop.kcorrect', fromlist=['MARCSLibrary']).MARCSLibrary().available


@skipUnless(_HAVE_C3K and _HAVE_MARCS, 'C3K or MARCS not staged')
class TestMARCSCoolGiants(TestCase):
    """
    Tier M (2026-10-07): the MARCS spherical patch for M giants. Full MARCS at
    Teff <= 3900 K and log g <= 3.0, cos^2 handover to C3K by 4250 K / 3.5.
    """

    @classmethod
    def setUpClass(cls):
        from artpop.kcorrect import MARCSLibrary
        cls.grid = KCorrectionGrid.build('LSST', z_grid=np.array([0.0, 0.05]))
        cls.c3k_only = KCorrectionGrid.build('LSST', z_grid=np.array([0.0, 0.05]),
                                             cool_giants=None)
        cls.lib = MARCSLibrary()
        cls.fs = load_filter_system('LSST', curve_dir=photometry_curve_dir('LSST'))

    def test_m1_marcs_node_is_the_marcs_integral(self):
        """M1: at a MARCS node inside the full-MARCS zone the table returns the
        MARCS model's own AB magnitude (sigma T^4, surface flux), to < 0.3 mmag,
        computed independently on the filter curve with np.trapezoid."""
        from artpop.kcorrect import L_SUN, FOUR_PI_D10_SQ, SIGMA_SB, C_AA
        lg, lt, f = self.lib.grid(0.0)
        lam = np.asarray(self.lib.wave, float)
        for ig, it in ((3, 10), (2, 6), (5, 13)):          # (1.0, 3500), (0.5, 3100), (2.0, 3800)
            T = 10 ** lt[it]
            got, info = self.grid.interpolate(lt[it], lg[ig], 0.0, 0.0)
            self.assertEqual(info['source'][0], 3)
            L = f[ig, it] / (SIGMA_SB * T ** 4) * L_SUN / FOUR_PI_D10_SQ
            for b, band in enumerate(self.grid.bands):
                tw, tt = self.fs.get_trans(band)
                Tl = np.interp(lam, tw, tt, left=0.0, right=0.0)
                f0 = 3631e-23 * C_AA / lam ** 2
                want = -2.5 * np.log10(np.trapezoid(L * Tl * lam, lam) / np.trapezoid(f0 * Tl * lam, lam))
                self.assertLess(abs(got[0, b] - want), 3e-4, (ig, it, band))

    def test_m2_weight_and_handover(self):
        """M2: weight 1 at and below 3900 K, exactly 0 from 4250 K and above
        log g 3.5; values equal C3K-alone where the weight is 0 and change
        continuously across the handover."""
        T = np.array([3500, 3900, 4000, 4100, 4249, 4250, 4400, 6000.])
        v, info = self.grid.interpolate(np.log10(T), 1.5, -0.5, 0.0)
        v0, _ = self.c3k_only.interpolate(np.log10(T), 1.5, -0.5, 0.0)
        w = info['marcs_weight']
        self.assertEqual(list(w[:2]), [1.0, 1.0])
        self.assertTrue(np.all(w[5:] == 0.0))
        np.testing.assert_array_equal(v[5:], v0[5:])
        _, info_d = self.grid.interpolate(np.log10(3500.0), 4.6, -0.5, 0.0)
        self.assertEqual(info_d['marcs_weight'][0], 0.0)        # dwarfs stay C3K
        Tf = np.linspace(3850, 4300, 91)
        vf, _ = self.grid.interpolate(np.log10(Tf), 1.5, -0.5, 0.0)
        self.assertLess(float(np.max(np.abs(np.diff(vf, axis=0)))), 0.05)

    def test_m3_holes_fall_back_and_are_flagged(self):
        """M3: where MARCS has no model even after the [Fe/H] fill (log g -0.5
        at [Fe/H] -1), the row is served by C3K and flagged, never extrapolated."""
        v, info = self.grid.interpolate(np.log10(3300.0), -0.4, -1.0, 0.0)
        v0, _ = self.c3k_only.interpolate(np.log10(3300.0), -0.4, -1.0, 0.0)
        self.assertTrue(info['marcs_hole'][0])
        self.assertEqual(info['marcs_weight'][0], 0.0)
        np.testing.assert_array_equal(v, v0)

    def test_m4_marcs_conserves_flux_and_is_recorded(self):
        """M4: staged MARCS models integrate to sigma T^4 within 1 %; the table
        names the patch and the cache key separates it from C3K alone."""
        lg, lt, f = self.lib.grid(-1.0)
        lam = np.asarray(self.lib.wave, float)
        ok = np.isfinite(f[..., 0])
        ratio = np.trapezoid(f[ok], lam, axis=-1) / (5.670374419e-5 * (10 ** np.broadcast_to(lt, ok.shape)[ok]) ** 4)
        self.assertLess(float(np.max(np.abs(ratio - 1))), 0.01)
        self.assertIn('MARCS', self.grid.meta['library']['cool_giants'])
        self.assertIsNone(self.c3k_only.meta['library']['cool_giants'])
        self.assertNotEqual(KCorrectionGrid.cache_key('LSST'),
                            KCorrectionGrid.cache_key('LSST', cool_giants=None))
