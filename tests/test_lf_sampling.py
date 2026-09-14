# Standard library
import os
import pickle
from unittest import TestCase

# Third-party
import numpy as np
from astropy.table import Table
from astropy import units as u

# Project
from artpop import data_dir
from artpop.stars import Isochrone, SSP
from artpop.stars.imf import IMFIntegrator

iso_fn = os.path.join(data_dir, 'feh_m1.00_vvcrit0.4_LSST_10gyr_test_iso.pkl')

MAG_TO_FLUX = 0.4 * np.log(10)  # d(flux)/flux per mag


class TestLFSampling(TestCase):
    """
    Tests for luminosity-function (isochrone-row) sampling.

    The physical contracts asserted here:
    - the sampled counts realize the IMF-weighted luminosity function,
    - total flux and mass are conserved in expectation for ANY mag_limit,
    - the SBF magnitude equals the isochrone prediction ``ssp_sbf_mag`` for
      any mag_limit (the sampled and smooth sets are exact complements),
    - dust dims every flux-like quantity by +A,
    - memory: the number of stars allocated is the number rendered.
    """

    @classmethod
    def setUpClass(cls):
        with open(iso_fn, 'rb') as f:
            iso_table = Table(pickle.load(f))
        cls.filters = ['LSST_u', 'LSST_g', 'LSST_r',
                       'LSST_i', 'LSST_z', 'LSST_y']
        cls.iso = Isochrone(
            mini=iso_table['initial_mass'],
            mact=iso_table['star_mass'],
            mags=iso_table[cls.filters],
        )
        cls.band = 'LSST_i'
        cls.distance = 10 * u.Mpc
        cls.dm = 5 * np.log10(10e6) - 5
        mags = np.asarray(cls.iso.mag_table[cls.band]) + cls.dm
        cls.tip, cls.faint = float(mags.min()), float(mags.max())
        cls.w = cls.iso.imf_weights(
            'kroupa', m_max_norm=cls.iso.m_max, norm_type='number')
        cls.mean_mact = float(np.sum(cls.w * np.asarray(cls.iso.mact)))

    # ------------------------------------------------------------------ #
    def _ssp(self, mag_limit, total_mass=1e7, seed=42, **kw):
        return SSP(self.iso, total_mass=total_mass, distance=self.distance,
                   mag_limit=mag_limit, mag_limit_band=self.band,
                   random_state=seed, **kw)

    def _n_total(self, ssp):
        return ssp.n_total_expected

    def _bright_mask(self, mag_limit):
        mags = np.asarray(self.iso.mag_table[self.band]) + self.dm
        return mags <= mag_limit

    # ------------------------------------------------------------------ #
    def test_mag_limit_none_regression(self):
        """mag_limit=None must reproduce the stock sampler exactly."""
        ssp = SSP(self.iso, num_stars=1e5, random_state=1)
        self.assertAlmostEqual(42503.7698205, ssp.total_mass.value, 3)
        self.assertAlmostEqual(
            30038.3808821, ssp.total_initial_live_mass.value, 3)
        self.assertTrue(ssp.sampled_row_mask.all())
        self.assertEqual(len(ssp.sampled_row_mask), len(self.iso.mini))

    def test_count_correctness(self):
        """Realized per-row counts follow Lambda_k = N_total * w_k."""
        ml = self.tip + 2.0
        ssp = self._ssp(ml, total_mass=1e9, seed=11)
        n_total = self._n_total(ssp)
        bright = ssp.sampled_row_mask
        lam = n_total * self.w
        # total sampled count vs its Poisson expectation
        lam_tot = float(lam[bright].sum())
        self.assertLess(abs(ssp.num_stars - lam_tot), 4 * np.sqrt(lam_tot))
        # per-row counts (rows with enough expected stars for the normal
        # approximation)
        big = bright & (lam >= 25)
        self.assertGreater(big.sum(), 10)  # the test must not be vacuous
        counts = ssp.row_counts[big]
        expect = lam[big]
        self.assertTrue(np.all(np.abs(counts - expect)
                               <= 5 * np.sqrt(expect)))
        # no star was drawn from a faint row
        self.assertEqual(int(ssp.row_counts[~bright].sum()), 0)

    def test_flux_conservation(self):
        """E[total flux] is the fully-sampled expectation for any limit."""
        for ml in (self.tip + 1.0, 0.5 * (self.tip + self.faint)):
            ssp = self._ssp(ml, total_mass=1e8, seed=5)
            n_total = self._n_total(ssp)
            for band in self.filters:
                expect = (self.iso.ssp_mag(band, norm_type='number')
                          + self.dm - 2.5 * np.log10(n_total))
                got = ssp.total_mag(band)
                # Poisson error on the total flux from the sampled rows
                f_row = 10**(-0.4 * np.asarray(self.iso.mag_table[band]))
                bright = ssp.sampled_row_mask
                var = n_total * float(np.sum(self.w[bright]
                                             * f_row[bright]**2))
                total = n_total * float(np.sum(self.w * f_row))
                sigma_mag = np.sqrt(var) / total / MAG_TO_FLUX
                self.assertLess(abs(got - expect),
                                max(5 * sigma_mag, 1e-4),
                                msg=f'{band} at ml-tip={ml-self.tip:.1f}')
                self.assertLess(abs(got - expect), 0.02)

    def test_mass_conservation(self):
        """Realized total mass matches the requested mass (both modes)."""
        for ml in (self.tip - 1.0, self.tip + 2.0,
                   0.5 * (self.tip + self.faint)):
            ssp = self._ssp(ml, total_mass=1e7, seed=3)
            self.assertLess(
                abs(ssp.total_mass.value - 1e7) / 1e7, 0.01,
                msg=f'total_mass mode at ml-tip={ml-self.tip:.1f}')
        # num_stars mode: realized mass ~ N * mean_mact / remnants factor
        ssp = SSP(self.iso, num_stars=2e5, distance=self.distance,
                  mag_limit=self.tip + 2.0, mag_limit_band=self.band,
                  random_state=3)
        expect = 2e5 * self.mean_mact / ssp._remnants_factor()
        self.assertLess(abs(ssp.total_mass.value - expect) / expect, 0.01)

    def test_sbf_identity(self):
        """sbf_mag equals the isochrone prediction at any mag_limit."""
        truth = self.iso.ssp_sbf_mag(self.band) + self.dm
        # fully smooth: the identity is exact (no realization noise)
        ssp = self._ssp(self.tip - 1.0, total_mass=1e7)
        self.assertAlmostEqual(ssp.sbf_mag(self.band), truth, delta=1e-8)
        # partial regimes: Poisson-limited agreement
        for ml in (self.tip + 0.5, self.tip + 2.0,
                   0.5 * (self.tip + self.faint)):
            drifts = [self._ssp(ml, total_mass=1e8, seed=s).sbf_mag(self.band)
                      - truth for s in range(4)]
            self.assertLess(abs(np.mean(drifts)), 0.05,
                            msg=f'ml-tip={ml-self.tip:.2f}: {drifts}')

    def test_dust_plus_A(self):
        """Extinction dims total_mag AND sbf_mag by exactly +A."""
        A = 0.5
        kw = dict(mag_limit=self.tip + 2.0, total_mass=1e6, seed=7)
        s0 = self._ssp(a_lam=0.0, **kw)
        s1 = self._ssp(a_lam={self.band: A}, **kw)
        self.assertAlmostEqual(
            s1.total_mag(self.band) - s0.total_mag(self.band), A, delta=1e-9)
        self.assertAlmostEqual(
            s1.sbf_mag(self.band) - s0.sbf_mag(self.band), A, delta=1e-9)
        # extinction assigned after the build (the ALVISS apply_dust route)
        s2 = self._ssp(a_lam=0.0, **kw)
        s2.a_lam = {f: (A if f == self.band else 0.0) for f in self.filters}
        self.assertAlmostEqual(
            s2.total_mag(self.band) - s0.total_mag(self.band), A, delta=1e-9)
        self.assertAlmostEqual(
            s2.sbf_mag(self.band) - s0.sbf_mag(self.band), A, delta=1e-9)

    def test_allocation_equals_rendered(self):
        """mag_limit now bounds the allocation, not just the render.

        On this 10 Gyr isochrone the old mass-threshold path allocates the
        entire white-dwarf cooling track (~12% of the IMF by number) no
        matter where the limit falls; the LF path allocates exactly the
        stars it renders.
        """
        ml = self.tip + 2.0
        ssp = self._ssp(ml, total_mass=1e7, seed=13)
        # every allocated star is a rendered star
        self.assertEqual(len(ssp.initial_masses), ssp.num_stars)
        self.assertEqual(len(ssp.star_masses), ssp.num_stars)
        self.assertEqual(len(ssp.abs_mags[self.band]), ssp.num_stars)
        # and the number is the LF expectation, thousands of times smaller
        # than the mass-threshold allocation
        lam = self._n_total(ssp) * self.w[ssp.sampled_row_mask].sum()
        self.assertLess(ssp.num_stars, lam + 5 * np.sqrt(lam) + 10)
        # the mass-threshold allocation barely responds to the limit (the WD
        # track dominates it), so near the tip its ratio to the rendered
        # count blows up; the LF path's allocation IS the rendered count
        ml_near_tip = self.tip + 0.5
        near = self._ssp(ml_near_tip, total_mass=1e7, seed=13)
        self.assertEqual(len(near.initial_masses), near.num_stars)
        imfint = IMFIntegrator('kroupa', self.iso.m_min, self.iso.m_max)
        m_lim = self.iso.mag_to_mass(ml_near_tip - self.dm, self.band).min()
        f_num_mass_path = imfint.integrate(m_lim, self.iso.m_max, True)
        n_alloc_mass_path = self._n_total(near) * f_num_mass_path
        self.assertGreater(n_alloc_mass_path / max(near.num_stars, 1), 50)

    def test_edge_cases(self):
        """Limits off either end of the isochrone, and composite adds."""
        truth = self.iso.ssp_sbf_mag(self.band) + self.dm
        # brighter than every row: fully smooth, no crash, exact bookkeeping
        for mode in (dict(total_mass=1e7), dict(num_stars=1e5)):
            ssp = SSP(self.iso, distance=self.distance,
                      mag_limit=self.tip - 1.0, mag_limit_band=self.band,
                      random_state=1, **mode)
            self.assertEqual(ssp.num_stars, 0)
            self.assertEqual(ssp.frac_num_sampled, 0.0)
            self.assertEqual(ssp.frac_mass_sampled, 0.0)
            self.assertTrue(np.isfinite(ssp.total_mass.value))
            self.assertGreater(ssp.total_mass.value, 0)
            self.assertAlmostEqual(ssp.sbf_mag(self.band), truth, delta=1e-8)
        # fainter than every row: fully sampled, no smooth component
        ssp = self._ssp(self.faint + 1.0, total_mass=1e5, seed=2)
        self.assertFalse(ssp.has_integrated_component)
        self.assertEqual(ssp.frac_num_sampled, 1.0)
        self.assertEqual(ssp.frac_mass_sampled, 1.0)
        self.assertIsNone(ssp.mag_integrated_component(self.band))
        # composite add preserves the SBF identity
        c = (self._ssp(self.tip + 2.0, total_mass=1e8, seed=1)
             + self._ssp(self.tip + 2.0, total_mass=1e8, seed=2))
        self.assertLess(abs(c.sbf_mag(self.band) - truth), 0.05)
        # fully smooth + partial add works
        c2 = (self._ssp(self.tip - 1.0, total_mass=1e7, seed=1)
              + self._ssp(self.tip + 2.0, total_mass=1e7, seed=2))
        self.assertTrue(np.isfinite(c2.sbf_mag(self.band)))

    def test_mass_mode_ab(self):
        """LF and mass-threshold sampling render the same physical stars.

        A star is rendered by the mass path iff its mass is above the
        threshold AND its magnitude is brighter than the limit — which is
        the LF path's bright-row set. The two are therefore different
        estimators of the same physical count, and their rendered star
        counts must agree statistically.
        """
        ml = self.tip + 2.0
        n_lf = [self._ssp(ml, total_mass=1e7, seed=s).num_stars
                for s in range(8)]
        n_ms = [self._ssp(ml, total_mass=1e7, seed=s,
                          sampling='mass').num_stars for s in range(8)]
        lam = np.mean(n_lf)
        # means agree within combined Poisson error of the two estimators
        # plus a 5% allowance for the row-binned representation
        err = 5 * np.sqrt(2 * lam / 8) + 0.05 * lam
        self.assertLess(abs(np.mean(n_lf) - np.mean(n_ms)), err,
                        msg=f'lf={np.mean(n_lf):.1f} mass={np.mean(n_ms):.1f}')

    def test_sample_fraction_tip_fix(self):
        """A limit brighter than the tip yields f = 0, not sample-all."""
        ssp = self._ssp(self.tip + 2.0, total_mass=1e6, seed=1)
        m_lim, f_num, f_mass = ssp.sample_fraction(self.tip - 5.0, self.band)
        self.assertEqual(f_num, 0.0)
        self.assertEqual(f_mass, 0.0)
        self.assertEqual(m_lim, self.iso.m_max)
