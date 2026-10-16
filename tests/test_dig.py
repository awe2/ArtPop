"""
Diffuse ionized gas powered by the photons the HII regions do not capture
(2026-10-15; wiki/pipeline/young-clusters.md, "Diffuse ionized gas").

Tiers:

  Tier R -- `apply_to_rows` on two synthetic hot rows: DIG off is bit-identical;
            the rows do not change with the DIG on; the DIG's light per photon
            is the HII regions' at the same gas state and dust; HII + DIG light
            is independent of k when every leaked photon is absorbed; no
            birth-cloud dust on the DIG; k = 1 has no DIG.
  Tier S -- MIST SSPs and the clustered source: the SSP's DIG photons are
            (1 - k) a x (sampled + smooth raw photons); composites carry it per
            SSP; the source paints it on the Sersic in the gas image only.
"""
import os
from unittest import TestCase, skipUnless

import numpy as np
from astropy import units as u

from artpop import MIST_PATH
from artpop import nebular as neb
from artpop.nebular import (NebularConfig, apply_to_rows, extinction_curve,
                            dig_band_abs_mags_per_photon, default_continuum_table_path)
from artpop.filters import load_filter_system
from artpop.kcorrect import C3KLibrary

FILTERS = ['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y']
_HAVE_TABLE = os.path.isfile(neb.default_line_table_path())
_HAVE_CONT = os.path.isfile(default_continuum_table_path())
_HAVE_C3K = C3KLibrary().available
_HAVE_MIST25 = os.path.isdir(os.path.join(MIST_PATH, 'MIST_v2.5_LSST'))


def _curves():
    fs = load_filter_system('LSST', bands=FILTERS)
    return {b: fs.get_trans(b) for b in FILTERS}


def _rows(**kw):
    """Two O rows at 3 Myr, [Fe/H] -1, z 0.02, host A_V 0.2, MW 0.1."""
    base = dict(knob=0.4, a_v_max=0.0, size_mode='stromgren', log_u=-3.0, log_p=6.2)
    cfg = NebularConfig(**dict(base, **kw))
    lt = np.log10([45e3, 35e3])
    mags = {b: np.zeros(2) for b in FILTERS}
    return apply_to_rows(cfg, mags, _curves(), 6.5, -1.0, lt, np.array([4.0, 4.0]),
                         np.array([5.3, 4.5]), redshift=0.02, a_v_host=0.2, a_v_mw=0.1)


@skipUnless(_HAVE_C3K and _HAVE_TABLE and _HAVE_CONT, 'needs C3K, MAPPINGS and the continuum table')
class TestDIGRows(TestCase):

    def test_d1_off_is_bit_identical_and_rows_unchanged(self):
        """D1: dig_absorbed 0 is the pre-feature output (no dig_m1); with the
        DIG on the rows' magnitudes and blob fractions do not move (the DIG is
        an SSP-level smooth light, not star light); q_h = k q_h_raw."""
        off, b_off, i_off = _rows()
        on, b_on, i_on = _rows(dig_absorbed=0.95)
        self.assertNotIn('dig_m1', i_off)
        self.assertIn('dig_m1', i_on)
        for b in FILTERS:
            np.testing.assert_array_equal(off[b], on[b])
            np.testing.assert_array_equal(b_off[b], b_on[b])
        np.testing.assert_allclose(i_on['q_h'], 0.4 * i_on['q_h_raw'], rtol=1e-14)
        self.assertTrue(np.all(i_on['q_h_raw'] > 0))

    def test_d2_dig_light_per_photon_is_the_hii_light_at_the_same_state(self):
        """D2: at the HII regions' (U, P) and with no birth cloud, the DIG's light
        per photon equals a row's gas light per captured photon, band by band."""
        new, _, info = _rows(dig_absorbed=0.95, dig_log_u=-3.0, dig_log_p=6.2)
        for b in FILTERS:
            gas_row = 10 ** (-0.4 * new[b]) - 1.0          # star = 1 (mag 0), d_bc = 0
            per_photon = gas_row / info['q_h']
            np.testing.assert_allclose(per_photon, 10 ** (-0.4 * info['dig_m1'][b]),
                                       rtol=1e-9, err_msg=b)

    def test_d5_hii_plus_dig_independent_of_k(self):
        """D5: with every leaked photon absorbed (a = 1) at the HII gas state, the
        galaxy's recombination light, HII + DIG, does not depend on k."""
        tot = {}
        for k in (0.2, 0.5, 0.9):
            new, _, info = _rows(knob=k, dig_absorbed=1.0, dig_log_u=-3.0, dig_log_p=6.2)
            q_dig = (1 - k) * 1.0 * info['q_h_raw'].sum()
            tot[k] = {b: float(np.sum(10 ** (-0.4 * new[b]) - 1.0)
                               + q_dig * 10 ** (-0.4 * info['dig_m1'][b])) for b in FILTERS}
        for b in FILTERS:
            self.assertAlmostEqual(tot[0.2][b] / tot[0.9][b], 1.0, places=9)
            self.assertAlmostEqual(tot[0.5][b] / tot[0.9][b], 1.0, places=9)

    def test_d3_k_one_has_no_dig(self):
        """D3: k = 1 captures every photon: no DIG, whatever dig_absorbed."""
        _, _, info = _rows(knob=1.0, dig_absorbed=0.95)
        self.assertNotIn('dig_m1', info)
        self.assertFalse(NebularConfig(knob=1.0, dig_absorbed=0.95).dig)
        self.assertTrue(NebularConfig(knob=0.0, dig_absorbed=0.95).active)

    def test_d6_no_birth_cloud_dust_on_the_dig(self):
        """D6: the birth-cloud screen dims the HII rows, not the DIG."""
        _, _, i0 = _rows(dig_absorbed=0.95)
        new1, _, i1 = _rows(dig_absorbed=0.95, birth_cloud_a_v=1.0)
        for b in FILTERS:
            self.assertEqual(i0['dig_m1'][b], i1['dig_m1'][b])
        m = dig_band_abs_mags_per_photon(NebularConfig(knob=0.4, dig_absorbed=0.95), _curves(),
                                         -1.0, 0.02, 0.2, 0.1, extinction_curve('F99', 3.1))
        for b in FILTERS:
            self.assertAlmostEqual(m[b], i0['dig_m1'][b], places=12)


def _ssps(dig, knob=0.4):
    from artpop.stars import MISTSSP
    cfg = NebularConfig(knob=knob, size_mode='stromgren', gas_density='consistent',
                        source_q_h=1e50, dig_absorbed=dig)
    kw = dict(feh=-1.0, phot_system='LSST', version='2.5', distance=20 * u.Mpc,
              mag_limit=28.0, mag_limit_band='LSST_r', nebular=cfg)
    old = MISTSSP(log_age=9.5, total_mass=1e7, random_state=1, **kw)
    young = MISTSSP(log_age=6.7, total_mass=5e4, random_state=2, **kw)
    return old, young, old + young


@skipUnless(_HAVE_MIST25 and _HAVE_C3K and _HAVE_TABLE and _HAVE_CONT,
            'needs MIST v2.5 LSST, C3K, MAPPINGS and the continuum table')
class TestDIGPopulation(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old0, cls.young0, cls.pop0 = _ssps(0.0)
        cls.old, cls.young, cls.pop = _ssps(0.95)

    def test_d7_photon_bookkeeping(self):
        """D7: dig_q_h = (1 - k) a (sampled raw + smooth raw); the stars are the
        same draws with the DIG on or off; composites keep it per SSP; old SSPs
        have none."""
        y = self.young
        info = y.isochrone.nebular_info
        from scipy.interpolate import interp1d
        raw = interp1d(np.asarray(y.isochrone.mini), info['q_h_raw'])(y.initial_masses)
        want = 0.6 * 0.95 * (raw.sum() + y.integrated_q_h_raw)
        self.assertAlmostEqual(y.dig_q_h / want, 1.0, places=12)
        self.assertGreater(y.integrated_q_h_raw, 0.0)
        np.testing.assert_array_equal(self.young0.initial_masses, y.initial_masses)
        np.testing.assert_array_equal(self.pop0.star_mags('LSST_r'), self.pop.star_mags('LSST_r'))
        self.assertIsNone(self.old.integrated_dig_abs_mags)
        self.assertIsNone(self.young0.integrated_dig_abs_mags)
        recs = self.pop.ssp_components
        self.assertIsNone(recs[0]['integrated_dig_abs_mags'])
        self.assertEqual(recs[1]['integrated_dig_abs_mags'], y.integrated_dig_abs_mags)

    def test_d4_on_the_sersic_in_the_gas_image(self):
        """D4: the clustered source (no complexes) paints the DIG as the Sersic in
        the gas image; the stellar image is unchanged; IdealImager adds both."""
        from artpop.source import ClusteredSersicSP
        from artpop.image import IdealImager
        kw = dict(r_eff=0.5, n=0.8, theta=30, ellip=0.3, xy_dim=401, pixel_scale=0.2)
        self.pop0.rng = np.random.RandomState(5)
        s0 = ClusteredSersicSP(self.pop0, clusters=[], **kw)
        self.pop.rng = np.random.RandomState(5)
        s1 = ClusteredSersicSP(self.pop, clusters=[], **kw)
        zpt = 27.0
        f = lambda m: 10 ** (0.4 * (zpt - m))
        for band in ('LSST_g', 'LSST_r'):
            st0, gas0 = s0.smooth_image(band, f, (401, 401))
            st1, gas1 = s1.smooth_image(band, f, (401, 401))
            np.testing.assert_allclose(st1, st0, rtol=1e-12, atol=0)
            self.assertEqual(gas0.sum(), 0.0)
            dig = f(self.young.integrated_dig_abs_mags[band] + self.pop.dist_mod
                    + self.pop.a_lam[band])
            unit = s1._sersic_unit_image((401, 401))
            np.testing.assert_allclose(gas1, dig * unit, rtol=1e-12)
            self.assertGreater(unit.sum(), 0.95)                 # 401 px = 6.5 r_eff
            a = IdealImager().observe(s1, band, zpt=zpt).image.sum()
            b = IdealImager().observe(s0, band, zpt=zpt).image.sum()
            self.assertAlmostEqual((a - b) / gas1.sum(), 1.0, places=9)
