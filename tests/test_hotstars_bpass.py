"""
The hot-star set beyond TLUSTY (`artpop.hotstars`) and BPASS star lists
(`artpop.stars.bpass`), 2026-10-29.

Tiers:

  Tier H -- physics and routing without any staged library: the Sander & Vink
            (2020) mass-loss law, the transformed radius against PoWR's own
            definition, the hydrogen classes.
  Tier R -- the staged libraries (SKIPs when `tools/stage_hotstars.py` has not
            run): a B star is never routed to a WR template, weak winds go to
            Götberg and dense ones to PoWR, nothing is extrapolated past a mesh,
            every staged model conserves flux, the default (switch off) is the
            production path.
  Tier B -- BPASS v2.2.1 (SKIPs when the hrs files are not staged): the star
            list's sum N L equals BPASS's own SSP L_bol, the age weights, the
            implied masses, and BPASSSSP's sampled + smooth light against its
            expectation.

Test ids HS1.., BP1.. are quoted in ALVISS's wiki (pipeline/bpass-star-lists).
"""
import os
from unittest import TestCase, skipUnless

import numpy as np

from artpop import hotstars as H
from artpop.stars import bpass as B


def _have_hot():
    try:
        return bool(H.families())
    except FileNotFoundError:
        return False


def _have_bpass():
    import glob
    return bool(glob.glob(os.path.join(B.bpass_path(), 'hrs-bin-imf135_100.z002.dat*')))


_HAVE_HOT = _have_hot()
_HAVE_BPASS = _have_bpass()


class TierH(TestCase):

    def test_HS1_mdot_monotone(self):
        """Mdot rises with L/M (Gamma_e) and with Z, and collapses below the
        breakdown (Sander & Vink 2020 Eq. 28)."""
        # above the breakdown (Gamma_e 0.3-1); below it both sit at MDOT_FLOOR
        lm = H.log_mdot_sv2020(np.array([5.6, 5.9, 6.2]), np.array([20.0, 20.0, 20.0]), 0.0, -0.5)
        self.assertTrue(np.all(np.diff(lm) > 0), lm)
        lz = H.log_mdot_sv2020(6.0, 20.0, 0.0, np.array([-1.5, -1.0, -0.5, 0.0]))
        self.assertTrue(np.all(np.diff(lz) > 0))
        # a 3 Msun stripped star (log L 4) at 0.1 Zsun: no WR-type wind
        self.assertLess(float(H.log_mdot_sv2020(4.0, 3.0, 0.0, -1.0)), -9.0)
        # a 20 Msun He star at log L 5.7, solar: the WR regime of SV2020 Fig. 11 (~1e-5)
        self.assertTrue(-5.5 < float(H.log_mdot_sv2020(5.7, 20.0, 0.0, 0.0)) < -4.0)

    def test_HS2_rt_definition(self):
        """R_t reproduces a PoWR header: SMC WNE 12-13 (T* 89125 K, log L 5.3,
        log Mdot -5.284, vinf 1600, D 4) has log R_t 0.800."""
        rt = H.log_rt(5.3, np.log10(89125.0), -5.284, 1600.0, 4.0)
        self.assertAlmostEqual(float(rt), 0.800, delta=0.01)

    def test_HS3_classes(self):
        np.testing.assert_array_equal(H.h_class_from_x([0.7, 0.41, 0.4, 0.01, 1e-3, 0.0]),
                                      [0, 0, 1, 1, 2, 2])


@skipUnless(_HAVE_HOT, 'hot-star libraries not staged (tools/stage_hotstars.py)')
class TierR(TestCase):

    def test_HS4_b_star_not_wr(self):
        """An H-rich 20 kK main-sequence B star is offered TLUSTY / PoWR-OB only."""
        plan = H.route(np.log10(20000.0), 4.0, -1.0, h_class=H.H_RICH, log_l=3.5, mass=8.0)
        fams = {c['family'] if isinstance(c['family'], str) else 'gotberg' for c in plan}
        self.assertTrue(fams <= {'tlusty', 'powr_ob'}, fams)

    def test_HS5_weak_vs_dense(self):
        """A 3 Msun stripped star (log L 4, 70 kK) at 0.1 Zsun is weak-wind:
        Götberg at rank 0. A 30 Msun He star at log L 6.0 (100 kK) at 0.5 Zsun
        has a WR wind: PoWR WN ranks first."""
        fams = H.families()
        if not any(k.startswith('powr_wn') for k in fams):
            self.skipTest('no PoWR WN family staged')
        plan = H.route(np.log10(70000.0), 5.3, -1.0, h_class=H.H_FREE, log_l=4.0, mass=3.0)
        first = min(plan, key=lambda c: c['rank'])
        self.assertIsInstance(first['family'], tuple)            # a Götberg X_H mix
        plan = H.route(np.log10(100000.0), 5.0, -0.3, h_class=H.H_FREE, log_l=6.0, mass=30.0)
        first = min(plan, key=lambda c: c['rank'])
        self.assertTrue(str(first['family']).startswith('powr_wn'), first['family'])

    def test_HS6_flux_conservation(self):
        """Every packed model's INT F / sigma T^4 was within 5 % before rescaling."""
        import json
        man = json.load(open(os.path.join(H.hotstars_path(), 'manifest.json')))
        for name, info in man['blocks'].items():
            bad = [k for k, v in info['ratios'].items() if abs(v - 1) > 0.05 and k not in info['rejected']]
            self.assertFalse(bad, f'{name}: {bad[:5]}')

    def test_HS7_no_extrapolation(self):
        """A star hotter than every family is not served by any (its spectrum is
        NaN); inside a mesh the spectrum has unit bolometric flux."""
        for fam in H.families():
            lib = H.HotStarFamily(fam)
            ax2, lt, _ = lib.grid(lib.feh_grid[0])
            lam, f = H.family_spectra(fam, lib.feh_grid[0], [ax2[0]], [lt[-1] + 0.3])
            self.assertTrue(np.isnan(f).all(), fam)
            ok = np.isfinite(_[:, :, 0])
            i, j = np.argwhere(ok)[0]
            lam, f = H.family_spectra(fam, lib.feh_grid[0], [ax2[i]], [lt[j]])
            tot = float(np.sum(0.5 * (f[0, 1:] + f[0, :-1]) * np.diff(lam)))
            self.assertAlmostEqual(tot, 1.0, delta=0.02, msg=fam)

    def test_HS8_default_off(self):
        """Without ARTPOP_HOT_STARS and without a `hot` dict, interpolate never
        routes (the production path)."""
        from artpop.kcorrect import KCorrectionGrid, SOURCE_POWR_WN, SOURCE_GOTBERG
        old = os.environ.pop('ARTPOP_HOT_STARS', None)
        try:
            grid = KCorrectionGrid.cached('LSST')
            out, info = grid.interpolate([np.log10(80000.0)], [5.5], -1.0, 0.0)
            self.assertFalse(np.isin(info['source'], [SOURCE_POWR_WN, SOURCE_GOTBERG]).any())
        finally:
            if old is not None:
                os.environ['ARTPOP_HOT_STARS'] = old


@skipUnless(_HAVE_BPASS, 'BPASS v2.2.1 hrs files not staged (ARTPOP_BPASS_PATH)')
class TierB(TestCase):

    def test_BP1_age_weights(self):
        for lo, hi in ((0, 1e7), (1e7, 3e7), (3e7, 1e8), (1e8, 5e8)):
            w = B.bin_weights(lo, hi)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
            self.assertTrue((w >= 0).all())

    def test_BP2_luminosity_closure(self):
        """sum N L of the cells equals BPASS's own SSP L_bol (spectra file),
        time-averaged over the bin: to 1 % (single) / 3 % (binary)."""
        for kind, tol in (('sin', 0.01), ('bin', 0.03)):
            for lo, hi in ((0, 1e7), (1e7, 3e7), (1e8, 5e8)):
                sl = B.BPASSStarList(kind, -1.0, lo, hi)
                lb = 0.0
                for lab, wz in sl.z_nodes:
                    a = np.loadtxt(os.path.join(B.bpass_path(), f'spectra-{kind}-imf135_100.{lab}.dat.gz'))
                    lbol = np.sum(0.5 * (a[1:, 1:] + a[:-1, 1:]) * np.diff(a[:, 0])[:, None], axis=0) / 1e6
                    lb += wz * float(np.dot(B.bin_weights(lo, hi), lbol))
                self.assertAlmostEqual(sl.lbol_per_msun / lb, 1.0, delta=tol, msg=(kind, lo, hi))

    def test_BP3_masses(self):
        """Implied present masses are physical: the live mass per formed Msun is
        0.5-1, and the brightest H-rich main-sequence cells at 10-30 Myr are
        5-20 Msun."""
        sl = B.BPASSStarList('sin', -1.0, 1e7, 3e7)
        self.assertTrue(0.5 < sl.live_mass_per_msun < 1.0)
        ms = (sl.h_class == 0) & (sl.log_g > 3.8) & (sl.log_teff > 4.3)
        top = np.argsort(sl.log_l[ms])[-5:]
        self.assertTrue(np.all((sl.mass[ms][top] > 5) & (sl.mass[ms][top] < 20)), sl.mass[ms][top])

    @skipUnless(_HAVE_HOT, 'hot-star libraries not staged')
    def test_BP4_ssp_light(self):
        """BPASSSSP: sampled + smooth light within 0.03 mag of the row expectation
        at 1e8 Msun (at 1e6 the red supergiants' Poisson scatter is +-0.1 mag; the
        in-cell jitter's own bias is <= 0.006, 2026-10-29), the remnant
        convention, and a stripped-star bin's ionizing photons on its stars."""
        from astropy import units as u
        ssp = B.BPASSSSP('bin', -1.0, 1e7, 3e7, 'LSST', total_mass=1e8, p_present=0.866,
                         distance=5 * u.Mpc, mag_limit=27.0, mag_limit_band='LSST_r',
                         random_state=3, nebular=dict(knob=0.5, dig_absorbed=0.95))
        iso = ssp.isochrone
        w = iso.imf_weights()
        for b in ('LSST_g', 'LSST_r'):
            exp = ssp.n_total_expected * np.sum(w * 10 ** (-0.4 * np.asarray(iso.mag_table[b])))
            got = np.sum(10 ** (-0.4 * ssp.abs_mags[b])) + 10 ** (-0.4 * ssp.integrated_abs_mags[b])
            self.assertLess(abs(2.5 * np.log10(got / exp)), 0.03, b)
        self.assertAlmostEqual(ssp.total_mass.value / 1e8, 1.0, delta=0.02)
        self.assertGreater(ssp.dig_q_h, 1e51)        # 10-30 Myr binaries: ~8e50 / 1e6 Msun formed

    @skipUnless(_HAVE_HOT, 'hot-star libraries not staged')
    def test_BP5_star_lists_reach_hot_libraries(self):
        """The BPASS rows are served by the new atmospheres, not the blackbody:
        in a 10-30 Myr binary bin at [Fe/H] -1 Götberg stripped stars carry
        > 30 % of Q_H (measured 0.56, 2026-10-30), PoWR WN rows exist in both
        the photometry and the ionizing budget, the hydrogen-free rows hotter
        than 50 kK are never on C3K / TLUSTY, and the blackbody fallback keeps
        < 5 % of Q_H (measured 0.03; 0-10 Myr: 0.02)."""
        from artpop import kcorrect as K
        from artpop import nebular as N
        for lo, hi, got_min in ((1e7, 3e7, 0.3), (0.0, 1e7, 0.05)):
            iso = B.BPASSIsochrone('bin', -1.0, lo, hi, 'LSST', jitter=False, p_present=0.9)
            hot = dict(h_class=iso.h_class, log_l=iso.log_L, mass=iso.mact)
            # the ionizing budget (nebular's own source codes)
            q, src_q = N.ionizing_rate(iso.log_Teff, iso.log_g, iso.log_L, -1.0, hot=hot)
            nq = iso.star_list.n_per_msun * q
            share = {k: float(nq[src_q == k].sum() / nq.sum()) for k in np.unique(src_q)}
            self.assertGreater(share.get(N.SOURCE_GOTBERG, 0.0), got_min, (lo, share))
            self.assertGreater(share.get(N.SOURCE_POWR_WN, 0.0), 0.0, (lo, share))
            self.assertLess(share.get(N.SOURCE_BB, 0.0), 0.05, (lo, share))
            # the photometry (K-table source codes)
            src = iso.kcorr_info['LSST']['source']
            self.assertTrue(np.isin([K.SOURCE_GOTBERG, K.SOURCE_POWR_WN], src).all(), np.unique(src))
            h_free_hot = (iso.h_class == H.H_FREE) & (iso.log_Teff > np.log10(5e4))
            self.assertTrue(h_free_hot.any())
            self.assertFalse(np.isin(src[h_free_hot], [K.SOURCE_C3K, K.SOURCE_TLUSTY]).any())
            self.assertFalse(np.isin(src_q[h_free_hot], [N.SOURCE_C3K, N.SOURCE_TLUSTY]).any())
