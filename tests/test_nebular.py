"""
The nebular emission knob (`artpop.nebular`).

Tiers, in the order the argument is made:

  Tier A -- pure math and plumbing. No MIST, no library, no line table:
            the config, the line -> band photon-counting formula, line dust in
            its two frames, the blob kernel, and the imager split (flux
            conservation; an inactive knob is the old code path).
  Tier B -- the MAPPINGS line table (SKIPs when it has not been built with
            tools/build_nebular_grid.py).
  Tier S -- the C3K spectra: Q_H / L_bol against an external O-star
            calibration, the blackbody seam, and the birth-cloud dust
            differential (SKIPs when C3K is not staged).
  Tier C -- MISTIsochrone / SSP / image integration: k = 0 is bit-identical,
            an old SSP is untouched at k = 1, lines are linear in k, and the
            constant-SFR ionizing budget against Murphy et al. 2011 / KE12.

Test ids N1..N9 match the plan in wiki/pipeline/nebular-emission.md.
"""
# Standard library
import os
from types import SimpleNamespace
from unittest import TestCase, skipUnless

# Third-party
import numpy as np
from astropy import units as u

# Project
from artpop import MIST_PATH
from artpop.filters import load_filter_system
from artpop.source import Source
from artpop.image import IdealImager
from artpop.kcorrect import (C3KLibrary, band_offset, extinction_curve,
                             spectra_path, _trapz_weights, C_AA)
from artpop import nebular as neb
from artpop.nebular import (NebularConfig, MappingsLineTable, nebular_kernel,
                            line_band_abs_mags, apply_to_rows, young_row_mask,
                            row_spectra, ionizing_photons_per_erg,
                            ionizing_rate, L_SUN, HBETA_ERG_PER_ION)

FILTERS = ['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y']
_HAVE_TABLE = os.path.isfile(neb.default_line_table_path())
_HAVE_C3K = C3KLibrary().available
_HAVE_MIST25 = os.path.isdir(os.path.join(MIST_PATH, 'MIST_v2.5_LSST'))
_PC10_CM2 = 4.0 * np.pi * (10.0 * neb.PC_CM) ** 2


def _lsst_curves():
    fs = load_filter_system('LSST', bands=FILTERS)
    return {b: fs.get_trans(b) for b in FILTERS}


def _top_hat(lo=6000.0, hi=7000.0, n=20001):
    tw = np.linspace(lo - 50, hi + 50, n)
    return tw, ((tw >= lo) & (tw <= hi)).astype(float)


# ---------------------------------------------------------------------------
# Tier A
# ---------------------------------------------------------------------------
class TestNebularConfig(TestCase):

    def test_n0_config(self):
        self.assertFalse(NebularConfig().active)
        self.assertFalse(neb.is_active(None))
        c = NebularConfig.coerce({'knob': 0.4, 'a_v_max': 2.0})
        self.assertTrue(c.active)
        self.assertAlmostEqual(c.a_v_bc, 0.8)
        self.assertIs(NebularConfig.coerce(c), c)
        for bad in ({'knob': -0.1}, {'knob': 1.2}, {'fwhm_pc': 0.0},
                    {'a_v_max': -1.0}, {'not_a_param': 1.0}):
            with self.assertRaises(ValueError, msg=bad):
                NebularConfig.coerce(bad)


class TestLineIntoBand(TestCase):

    def test_n4_single_line_matches_the_photon_counting_formula(self):
        """N4: one line in a top-hat band, against the formula by hand."""
        tw, tt = _top_hat()
        L, lam0 = 1e38, 6564.6
        for z in (0.0, 0.03):
            got = line_band_abs_mags([[L]], [lam0], tw, tt, redshift=z)[0]
            lam_obs = lam0 * (1 + z)
            den = C_AA * np.sum(_trapz_weights(tw) * tt / tw)
            want = -2.5 * np.log10(L / _PC10_CM2 * lam_obs / den) - 48.6
            self.assertAlmostEqual(got, want, places=10)

    def test_n4b_line_leaves_the_band_past_its_edge(self):
        """N4b: past the red edge a line contributes nothing (inf mag)."""
        tw, tt = _top_hat()
        m = line_band_abs_mags([[1e38]], [6564.6], tw, tt, redshift=0.08)
        self.assertTrue(np.isinf(m[0]))

    def test_n4c_halpha_moves_from_r_to_i_with_redshift(self):
        """N4c: on the real LSST curves Halpha is an r line at z = 0 and an
        i line at z = 0.12 -- the reason lines are placed exactly, not on a
        z grid."""
        c = _lsst_curves()
        m = {z: {b: line_band_abs_mags([[1e38]], [6564.6], *c[b], redshift=z)[0]
                 for b in ('LSST_r', 'LSST_i')} for z in (0.0, 0.12)}
        self.assertLess(m[0.0]['LSST_r'], m[0.0]['LSST_i'] - 2.0)
        self.assertLess(m[0.12]['LSST_i'], m[0.12]['LSST_r'] - 2.0)

    def test_n8_linear_in_luminosity(self):
        """N8: line flux is linear in L (so in Q_H, so in k)."""
        tw, tt = _top_hat()
        m1 = line_band_abs_mags([[1e38, 3e37]], [6564.6, 6585.3], tw, tt)[0]
        m2 = line_band_abs_mags([[2e38, 6e37]], [6564.6, 6585.3], tw, tt)[0]
        self.assertAlmostEqual(m1 - m2, 2.5 * np.log10(2.0), places=12)

    def test_n5b_line_dust_in_its_two_frames(self):
        """N5b: host dust at the rest wavelength, MW at the observed one."""
        tw, tt = _top_hat(5000, 8000)
        ext = extinction_curve('F99', 3.1)
        lam0, z = 6564.6, 0.05
        base = line_band_abs_mags([[1e38]], [lam0], tw, tt, z)[0]
        host = line_band_abs_mags([[1e38]], [lam0], tw, tt, z, a_v_host=1.0, ext=ext)[0]
        mw = line_band_abs_mags([[1e38]], [lam0], tw, tt, z, a_v_mw=0.5, ext=ext)[0]
        self.assertAlmostEqual(host - base, 1.0 * float(ext(np.array([lam0]))[0]), places=10)
        self.assertAlmostEqual(mw - base, 0.5 * float(ext(np.array([lam0 * (1 + z)]))[0]), places=10)


class TestKernelAndImager(TestCase):

    def test_kernel_is_unit_sum_and_sized(self):
        k = nebular_kernel(100.0, 10 * u.Mpc, 0.02)
        self.assertAlmostEqual(k.sum(), 1.0, places=14)
        self.assertTrue(np.allclose(k, k[::-1, ::-1]))
        # FWHM = 100 pc at 10 Mpc = 2.06" = 103 px at 0.02"/px
        prof = k[k.shape[0] // 2]
        above = np.flatnonzero(prof >= prof.max() / 2)
        fwhm_px = np.degrees(100.0 / 10e6) * 3600 / 0.02
        self.assertLess(abs((above[-1] - above[0] + 1) - fwhm_px), 2.0)
        tiny = nebular_kernel(1.0, 200 * u.Mpc, 0.2)
        self.assertGreater(tiny[tiny.shape[0] // 2, tiny.shape[1] // 2], 0.999)

    def _source(self, nebular, frac, n=60, dim=201, seed=3):
        rng = np.random.RandomState(seed)
        xy = rng.uniform(70, 130, size=(n, 2))
        mags = {'LSST_r': rng.uniform(24, 27, n)}
        src = Source(xy, mags, dim, pixel_scale=0.2)
        src.sp = SimpleNamespace(nebular=nebular,
                                 nebular_blob_frac={'LSST_r': frac})
        src.distance_angular = 5 * u.Mpc
        return src

    def test_n1a_inactive_knob_is_the_old_code_path(self):
        """N1a: no config, k = 0, or all-zero fractions give the image of the
        stock `inject_stars` call, bit for bit."""
        frac = np.full(60, 0.7)
        ref_src = self._source(None, frac)
        ref = IdealImager().observe(ref_src, 'LSST_r', psf=None).image
        for cfg, f in ((NebularConfig(knob=0.0), frac),
                       (NebularConfig(knob=0.5), np.zeros(60))):
            img = IdealImager().observe(self._source(cfg, f), 'LSST_r', psf=None).image
            self.assertTrue(np.array_equal(img, ref))

    def test_n6_blob_conserves_flux(self):
        """N6: splitting and convolving moves light but creates none."""
        rng = np.random.RandomState(7)
        frac = rng.uniform(0, 1, 60)
        ref = IdealImager().observe(self._source(None, frac), 'LSST_r').image
        img = IdealImager().observe(
            self._source(NebularConfig(knob=0.5, fwhm_pc=30.0), frac), 'LSST_r').image
        self.assertAlmostEqual(img.sum() / ref.sum(), 1.0, places=12)
        self.assertLess(img.max(), ref.max())       # the light did spread

    def test_n7a_old_ssp_selects_no_rows(self):
        """N7a: an SSP older than t_bc has no nebular rows even at k = 1."""
        cfg = NebularConfig(knob=1.0)
        lt = np.array([3.6, 4.2, 4.6])
        self.assertFalse(young_row_mask(8.0, lt, cfg).any())
        self.assertEqual(young_row_mask(6.5, lt, cfg).tolist(), [False, True, True])
        mags = {'LSST_r': np.array([1.0, -2.0, -4.0])}
        new, blob, info = apply_to_rows(cfg, mags, {}, 8.0, -1.0, lt,
                                        np.full(3, 4.0), np.zeros(3))
        self.assertTrue(np.array_equal(new['LSST_r'], mags['LSST_r']))
        self.assertEqual(info['n_rows'], 0)


# ---------------------------------------------------------------------------
# Tier B -- the MAPPINGS table
# ---------------------------------------------------------------------------
@skipUnless(_HAVE_TABLE, 'MAPPINGS table not built (tools/build_nebular_grid.py)')
class TestMappingsTable(TestCase):

    def test_n3_ratios_at_nodes_and_balmer_decrement(self):
        """N3: nodes reproduce the table; Halpha/Hbeta is case-B-like."""
        t = MappingsLineTable()
        self.assertEqual([a.size for a in t.axes], [12, 9, 12])
        i, j, k = 5, 4, 5
        names, lam, r = t.ratios(t.axes[0][i], t.axes[1][j], t.axes[2][k])
        np.testing.assert_allclose(r, t.cube[i, j, k], rtol=0, atol=1e-12)
        ha = r[names.index('Halpha')]
        self.assertEqual(r[names.index('Hbeta')], 1.0)
        self.assertTrue(2.7 < ha < 3.3, ha)
        # vacuum wavelengths
        self.assertAlmostEqual(lam[names.index('Halpha')], 6564.6, delta=0.2)

    def test_n3b_out_of_grid(self):
        t = MappingsLineTable()
        with self.assertRaises(ValueError):
            t.ratios(8.5, -1.0, 6.2)
        with self.assertLogs(neb.nebular_logger, 'WARNING'):
            names, _, lo = t.ratios(6.0, -3.0, 6.2)
        _, _, edge = t.ratios(t.axes[0][0], -3.0, 6.2)
        np.testing.assert_array_equal(lo, edge)


# ---------------------------------------------------------------------------
# Tier S -- spectra, Q_H and the birth-cloud dust
# ---------------------------------------------------------------------------
@skipUnless(_HAVE_C3K, f'C3K not staged under {spectra_path()}')
class TestIonizingPhotons(TestCase):

    def test_n2_o_star_q_over_l_against_martins05(self):
        """N2: log(Q_H / L_bol) of O dwarfs vs Martins, Schaerer & Hillier
        (2005, A&A 436, 1049) theoretical scale, Table 1 (O3 V and O5 V).
        LTE C3K against NLTE line-blanketed CMFGEN: 0.15 dex tolerance.
        (Values transcribed for the test; check against the table before
        quoting the comparison in the paper.)"""
        cases = [(44852.0, 3.92, 5.84, 49.64), (40862.0, 3.92, 5.51, 49.26)]
        lt = np.log10([c[0] for c in cases])
        lg = np.array([c[1] for c in cases])
        ll = np.array([c[2] for c in cases])
        q, src = ionizing_rate(lt, lg, ll, 0.0)
        self.assertTrue(np.isin(src, (neb.SOURCE_C3K, neb.SOURCE_C3K_LOGG)).all())
        for (t, g, l, logq), qq in zip(cases, q):
            self.assertLess(abs(np.log10(qq) - logq), 0.15,
                            f'{t:.0f} K: log Q = {np.log10(qq):.3f} vs {logq}')

    def test_n2b_monotonic_and_blackbody_seam(self):
        """N2b: Q_H/L rises with Teff along the C3K hull and does not jump by
        more than 0.3 dex onto the blackbody at the 50 kK ceiling."""
        lt = np.log10([15e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3, 49.9e3, 50.1e3])
        groups, src = row_spectra(lt, np.full(lt.size, 4.0), 0.0)
        q = np.zeros(lt.size)
        for idx, lam, f in groups:
            q[idx] = ionizing_photons_per_erg(lam, f)
        self.assertTrue(np.all(np.diff(q[:-1]) > 0))
        self.assertEqual(src[-1], neb.SOURCE_BB)
        self.assertLess(abs(np.log10(q[-1] / q[-2])), 0.3)

    def test_n2c_fsps_fill_cells_are_detected_and_never_used(self):
        """N2c: FSPS's C3K file fills the unphysical corner of the (log g, Teff)
        rectangle with ONE placeholder spectrum (not a blackbody, not any
        genuine model). At [Fe/H] = 0 it sits in 399 cells, exactly the cells
        whose spectrum is also byte-identical at [Fe/H] = -1 (no real model is
        metallicity-blind). A row there is served by genuine models only."""
        lib = C3KLibrary()
        g, t, f0 = lib.grid(0.0)
        fill0 = neb.c3k_fill_mask(f0)
        f0 = np.array(f0)                       # the library caches one block
        _, _, f1 = lib.grid(-1.0)
        same_feh = np.array([[np.array_equal(f0[i, j], f1[i, j]) and np.any(f0[i, j] > 0)
                              for j in range(t.size)] for i in range(g.size)])
        self.assertEqual(int(fill0.sum()), 399)
        self.assertEqual(len({f0[i, j].tobytes() for i, j in np.argwhere(fill0)}), 1)
        self.assertTrue(np.array_equal(fill0, same_feh))
        cool_low = (int(np.argmin(abs(g + 1.0))), int(np.argmin(abs(t - 4.0))))
        ms_hot = (int(np.argmin(abs(g - 4.0))), int(np.argmin(abs(t - np.log10(3e4)))))
        self.assertTrue(fill0[cool_low])       # log g -1 at 10 kK: past Eddington
        self.assertFalse(fill0[ms_hot])        # a 30 kK dwarf: a real model
        # a supergiant-like row whose cell touches fills: never the fill spectrum
        _, src = row_spectra([np.log10(4.5e4)], [3.3], 0.0)
        self.assertEqual(int(src[0]), neb.SOURCE_C3K_LOGG)

    @skipUnless(_HAVE_TABLE, 'MAPPINGS table not built')
    def test_n5_birth_cloud_dust_is_the_band_offset_differential(self):
        """N5: with negligible line light the row moves by exactly
        band_offset(A_host + k A_max) - band_offset(A_host) of its spectrum."""
        cfg = NebularConfig(knob=0.6, a_v_max=1.5)
        curves = _lsst_curves()
        lt = np.log10([25e3, 35e3])
        lg = np.array([4.0, 4.0])
        mags = {b: np.zeros(2) for b in FILTERS}
        z, a_h, a_mw = 0.03, 0.2, 0.1
        # log L = -40: the lines are ~1e-40 of the continuum, so only the dust moves
        new, blob, info = apply_to_rows(cfg, mags, curves, 6.5, 0.0, lt, lg,
                                        np.full(2, -40.0), redshift=z,
                                        a_v_host=a_h, a_v_mw=a_mw)
        ext = extinction_curve('F99', 3.1)
        groups, _ = row_spectra(lt, lg, 0.0)
        idx, lam, f = groups[0]
        for b in FILTERS:
            tw, tt = curves[b]
            for j in range(2):
                want = (band_offset(lam, f[j], tw, tt, z, a_h + cfg.a_v_bc, a_mw, ext)
                        - band_offset(lam, f[j], tw, tt, z, a_h, a_mw, ext))
                self.assertAlmostEqual(new[b][j], want, places=9, msg=b)
            self.assertTrue(np.allclose(blob[b], cfg.knob, atol=1e-12))


# ---------------------------------------------------------------------------
# Tier C -- MIST integration
# ---------------------------------------------------------------------------
@skipUnless(_HAVE_MIST25 and _HAVE_C3K and _HAVE_TABLE,
            'needs MIST v2.5 LSST, C3K and the MAPPINGS table')
class TestNebularMIST(TestCase):

    _kw = dict(feh=-1.0, phot_system='LSST', version='2.5')

    def test_n1_knob_zero_is_bit_identical(self):
        """N1: nebular=None, {} and knob = 0 all give stock magnitudes,
        population and image, byte for byte -- the switch-back guarantee."""
        from artpop import MISTIsochrone
        from artpop.stars import MISTSSP
        from artpop.source import SersicSP
        stock = MISTIsochrone(log_age=6.5, **self._kw)
        for nb in ({}, {'knob': 0.0}, NebularConfig()):
            iso = MISTIsochrone(log_age=6.5, nebular=nb, **self._kw)
            for f in FILTERS:
                self.assertTrue(np.array_equal(np.asarray(stock.mag_table[f]),
                                               np.asarray(iso.mag_table[f])), f)
            self.assertIsNone(iso.nebular_blob_frac)

        def image(nb):
            sp = MISTSSP(log_age=6.5, total_mass=1e5, distance=5 * u.Mpc,
                         random_state=11, nebular=nb, **self._kw)
            src = SersicSP(sp, r_eff=0.3, n=0.8, theta=0, ellip=0.2,
                           xy_dim=101, pixel_scale=0.2)
            return IdealImager().observe(src, 'LSST_r').image
        self.assertTrue(np.array_equal(image(None), image({'knob': 0.0})))

    def test_n7_old_isochrone_untouched_at_full_knob(self):
        """N7: a 100 Myr SSP is outside the birth cloud: k = 1 changes nothing."""
        from artpop import MISTIsochrone
        stock = MISTIsochrone(log_age=8.0, **self._kw)
        iso = MISTIsochrone(log_age=8.0, nebular={'knob': 1.0}, **self._kw)
        for f in FILTERS:
            self.assertTrue(np.array_equal(np.asarray(stock.mag_table[f]),
                                           np.asarray(iso.mag_table[f])), f)
        self.assertEqual(iso.nebular_info['n_rows'], 0)

    def test_n8b_young_rows_brighten_in_r_and_restore(self):
        """N8b: at k > 0 young O/B rows brighten in r (Halpha) net of the
        birth-cloud dust at small A_V; set_redshift re-derives from the
        rest-frame columns; the blob fraction is in (0, 1]."""
        from artpop import MISTIsochrone
        cfg = {'knob': 1.0, 'a_v_max': 0.0}
        stock = MISTIsochrone(log_age=6.5, **self._kw)
        iso = MISTIsochrone(log_age=6.5, nebular=cfg, **self._kw)
        rows = iso.nebular_info['mask']
        self.assertGreater(rows.sum(), 10)
        hot = rows & (np.asarray(iso.log_Teff) > 4.5)
        dr = np.asarray(iso.mag_table['LSST_r']) - np.asarray(stock.mag_table['LSST_r'])
        self.assertTrue(np.all(dr[hot] < 0))
        self.assertTrue(np.all(dr[~rows] == 0))
        b = iso.nebular_blob_frac['LSST_r']
        self.assertTrue(np.all((b[rows] > 0) & (b[rows] <= 1)))
        direct = MISTIsochrone(log_age=6.5, redshift=0.05, nebular=cfg, **self._kw)
        iso.set_redshift(0.05)
        for f in FILTERS:
            np.testing.assert_allclose(np.asarray(iso.mag_table[f]),
                                       np.asarray(direct.mag_table[f]),
                                       rtol=0, atol=1e-10)

    def _q_over_sfr(self, feh):
        """Q_H / SFR (photons/s per Msun/yr) of a constant SFR held 20 Myr."""
        from artpop import MISTIsochrone
        log_ages = np.round(np.arange(5.0, 7.3001, 0.05), 2)
        q_per_msun = []
        for la in log_ages:
            iso = MISTIsochrone(log_age=la, feh=feh, phot_system='LSST',
                                version='2.5')
            # rows cooler than ~8 kK emit no measurable Q_H; skipping them
            # keeps this test to seconds
            m = (np.asarray(iso.mini) <= 100.0) & (np.asarray(iso.log_Teff) >= 3.9)
            w = iso.imf_weights('kroupa', m_min_norm=0.1, m_max_norm=100.0,
                                norm_type='mass')
            q, _ = ionizing_rate(np.asarray(iso.log_Teff)[m],
                                 np.asarray(iso.log_g)[m],
                                 np.asarray(iso.log_L)[m], iso.feh)
            q_per_msun.append(float(np.sum(w[m] * q)))
        t_yr = 10 ** log_ages
        # Q(t) = SFR * INT_0^t q(t') dt'; the first node stands in for 0-0.1 Myr
        return np.sum(_trapz_weights(t_yr) * np.array(q_per_msun)) \
            + q_per_msun[0] * t_yr[0]

    def test_n9_constant_sfr_ionizing_budget(self):
        """N9 (external): at SOLAR metallicity, a constant SFR held for 20 Myr
        emits Q_H / SFR within 0.2 dex of 1.37e53 photons/s per Msun/yr
        (Murphy et al. 2011 eq. 2, the Halpha calibration Kennicutt & Evans
        2012 adopt; Starburst99, Kroupa 0.1-100 Msun, solar). Measured
        2026-09-28: +0.08 dex. At [Fe/H] = -1 the budget must come out HIGHER
        (hotter, longer-lived metal-poor rotating massive stars; +0.34 dex
        measured) -- the calibration does not apply there, the direction does."""
        solar = self._q_over_sfr(0.0)
        offset = np.log10(solar / 1.37e53)
        poor = self._q_over_sfr(-1.0)
        print(f'N9: log[Q/SFR / Murphy+11] = {offset:+.3f} dex (solar), '
              f'{np.log10(poor / 1.37e53):+.3f} dex ([Fe/H] = -1)')
        self.assertLess(abs(offset), 0.2)
        self.assertGreater(poor, solar)
