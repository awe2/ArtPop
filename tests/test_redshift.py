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
        k = [self.grid.interpolate(np.log10(4000.0), 0.5, -1.0, zz)[0][0, 0]
             for zz in z]
        self.assertTrue(np.all(np.diff(k) > 0), f'K(u) not monotone: {k}')

    def test_b3_k_orders_by_temperature_but_not_monotonically_everywhere(self):
        """B3: K orders by temperature on the giant branch, and turns in u and z.

        HANDOFF_REDSHIFT.md's B3 claims the ordering "holds across the grid".
        Measured at log g 2.5, [Fe/H] = -1, z = 0.05, that is true on the cool
        branch and false at the hot end of two bands:

        ===== ====== ====== ====== ====== ====== ====== ====== ======
        T (K)   3500   4000   4500   5000   5500   6000   7000   8000
        K(u)  +0.473 +0.309 +0.214 +0.203 +0.230 +0.279 +0.429 +0.555
        K(z)  +0.054 +0.019 -0.006 -0.027 -0.041 -0.051 -0.055 -0.044
        ===== ====== ====== ====== ====== ====== ====== ====== ======

        Both turns are a spectral edge crossing the band as the filter
        blueshifts: the Balmer jump at 3646 A for *u*, the Paschen jump at
        8204 A for *z*. g, r, i and y are monotone throughout.

        The ordering is asserted where it holds -- the giant branch, which is
        where the flux and all of the SBF weight live -- and the turns are
        asserted too, because structure like this is precisely what a
        per-galaxy scalar cannot represent.
        """
        cool = [3500.0, 4000.0, 4500.0, 5000.0]
        k_cool = np.array([self.grid.interpolate(np.log10(t), 2.5, -1.0,
                                                 Z_REF)[0][0] for t in cool])
        for j, band in enumerate(self.grid.bands):
            self.assertTrue(np.all(np.diff(k_cool[:, j]) < 0),
                            f'K({band}) not ordered on the giant branch: '
                            f'{k_cool[:, j]}')

        full = [3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 7000.0, 8000.0]
        k_all = np.array([self.grid.interpolate(np.log10(t), 2.5, -1.0,
                                                Z_REF)[0][0] for t in full])
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
            self.assertTrue(np.array_equal(getattr(self.grid, name),
                                           getattr(other, name)), name)
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
            lam, trans = load_filter_system('LSST').get_trans(band)
            lam_rest_min = lam[trans > 0].min() / (1.0 + z_max)
            self.assertGreater(lam_rest_min, blue_edge)

    def test_b7_redshift_zero_slice_is_identically_zero(self):
        """B7: the z = 0 slice of every backend is exactly zero.

        The numerator and denominator of the z = 0 column are computed by the
        same code path with the same arguments, so they must be bit-identical.
        This is the grid-level statement of the bit-identity C1 asserts
        end to end.
        """
        for arr in (self.grid.c3k, self.grid.wd, self.grid.bb):
            if arr.size:
                self.assertEqual(np.nanmax(np.abs(arr[..., 0, :])), 0.0)

    def test_b8_redshift_outside_the_grid_raises(self):
        """B8: a redshift off the end of the grid raises rather than extrapolating."""
        with self.assertRaises(ValueError):
            self.grid.interpolate(np.log10(4500.0), 2.5, -1.0,
                                  float(self.grid.z_grid[-1]) + 0.05)

    @skipUnless(_HAVE_WD, 'Tremblay WD library not staged')
    def test_b9_differential_form_suppresses_the_library_seam(self):
        """B9: the seam is real, and the differential form is what tames it.

        C3K at log g 5.5 and Tremblay at log g 6.5 are not the same star
        computed twice -- one is a metal atmosphere, the other a pure-hydrogen
        DA -- so their absolute photometry genuinely steps across the join.
        That step is in MIST's composite already: at log Teff = 3.17609 in
        ``bcl/feh+0.00_afe+0.0.LSST`` every row with log g <= 6.0 carries an
        identical BC and log g = 6.5 jumps.

        So the thing worth asserting is not that K is continuous -- it is not,
        and it should not be -- but that the **difference** cancels most of the
        seam, which is the entire argument for applying redshift
        differentially rather than regenerating magnitudes. Measured at
        [Fe/H] = -1, z = 0.05: the colour step across the join is 0.39 mag at
        20 kK and 0.49 mag at 40 kK, while the step in the K-colours is 0.054
        and 0.018 -- a suppression of 7x and 27x.

        For scale, the rows this affects carry under 0.1% of an old
        population's flux and none of its SBF weight.
        """
        fs = load_filter_system('LSST')
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
            k_c = self.grid.interpolate(np.log10(teff), 5.5, -1.0, Z_REF)[0][0]
            k_w = self.grid.interpolate(np.log10(teff), 6.5, -1.0, Z_REF)[0][0]

            # colours, so the arbitrary normalisation of each library cancels
            step_mag = max(abs((m_c[i] - m_c[i + 1]) - (m_w[i] - m_w[i + 1]))
                           for i in range(len(FILTERS) - 1))
            step_k = max(abs((k_c[i] - k_c[i + 1]) - (k_w[i] - k_w[i + 1]))
                         for i in range(len(FILTERS) - 1))
            self.assertLess(step_k, 0.15, f'K seam at {teff:.0f} K too large')
            self.assertGreater(step_mag / step_k, 3.0,
                               f'differential form suppresses the {teff:.0f} K '
                               f'seam by only {step_mag / step_k:.1f}x')

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
        k_c = self.grid.interpolate(np.log10(teff), 5.5, -1.0, Z_REF)[0][0]
        k_w = self.grid.interpolate(np.log10(teff), 6.5, -1.0, Z_REF)[0][0]
        k_b, info = self.grid.interpolate(np.log10(teff), 6.0, -1.0, Z_REF)
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
    """The two claims that can only be made against MIST's own columns."""

    _kw = dict(log_age=10.0, feh=-1.0, phot_system='LSST', version='2.5')

    def test_c1_mist_zero_redshift_is_bit_identical(self):
        """C1 (MIST): ``MISTIsochrone(redshift=0.0)`` is byte-for-byte stock."""
        from artpop import MISTIsochrone
        stock = MISTIsochrone(**self._kw)
        zeroed = MISTIsochrone(redshift=0.0, a_v_host=0.0, a_v_mw=0.0,
                               **self._kw)
        for filt in FILTERS:
            self.assertTrue(np.array_equal(
                np.asarray(stock.mag_table[filt]),
                np.asarray(zeroed.mag_table[filt])), filt)
        self.assertIsNone(zeroed.mag_table_rest)

    def test_c2_columns_equal_stock_plus_the_interpolated_correction(self):
        """C2: at z > 0 the columns are exactly stock + the interpolated offset.

        Differential by construction: MIST's z = 0 calibration is preserved
        exactly and only a small, smooth correction is added on top. That is
        what makes the 0.05-0.15 mag error of regenerating magnitudes from the
        BC tables cancel rather than accumulate.
        """
        from artpop import MISTIsochrone
        stock = MISTIsochrone(**self._kw)
        shifted = MISTIsochrone(redshift=Z_REF, **self._kw)
        for filt in FILTERS:
            rest = np.asarray(shifted.mag_table_rest[filt])
            self.assertTrue(np.array_equal(
                rest, np.asarray(stock.mag_table[filt])), filt)
            self.assertTrue(np.allclose(
                np.asarray(shifted.mag_table[filt]),
                rest + shifted.delta_mag[filt], rtol=0, atol=0), filt)
        info = shifted.kcorr_info['LSST']
        self.assertGreater(info['n_c3k'], 0.7 * len(stock.mini))
        self.assertEqual(info['n_feh_clipped'], 0)

    def test_c2b_set_redshift_restores_before_reapplying(self):
        """C2b: ``set_redshift`` corrects the *rest-frame* columns, not the
        already-corrected ones.

        Applying a second correction on top of the first would be wrong by the
        old offset and would look almost right, which is the only reason this
        is worth a test.
        """
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
