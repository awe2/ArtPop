"""
Young stars in star clusters and the one-density gas state (2026-10-15).

Tiers:

  Tier G -- the gas state (`artpop.nebular`): one hydrogen density sets the
            Stromgren size, U and P/k; the consistent config overwrites the
            three; the independent config is unchanged. Pure math.
  Tier P -- cluster placement (`artpop.space.clusters`): CMF weights, the
            centre radius distribution, projected-Plummer offsets and stamps.
            Pure math.
  Tier Q -- `ClusteredSersicSP` on MIST populations: unclustered stars land
            where `SersicSP` puts them, members sit on their clusters, the
            smooth light follows the mixture exactly, the gas of a cluster is
            one sphere on its centre, and flux is conserved; the expected
            ionizing photons match the sampled ones.

Test ids G1.., P1.., Q1.. are quoted in wiki/pipeline/young-clusters.md.
"""
import os
from unittest import TestCase, skipUnless

import numpy as np
from astropy import units as u

from artpop import MIST_PATH
from artpop import nebular as neb
from artpop.nebular import (NebularConfig, gas_state, gas_log_p, stromgren_radius_pc,
                            ionization_parameter, stromgren_diameter_pc)
from artpop.space.clusters import (cluster_weights, cluster_centres, plummer_offsets,
                                   plummer_stamp, paint_stamp, ClusterSpec,
                                   sample_cmf_to_budget, cmf_mass_fraction,
                                   complex_scale_pc, ring_surface_density,
                                   ring_disk_offsets, KMS_PC_PER_MYR)

_HAVE_MIST25 = os.path.isdir(os.path.join(MIST_PATH, 'MIST_v2.5_LSST'))
_HAVE_TABLE = os.path.isfile(neb.default_line_table_path())
_HAVE_C3K = neb.C3KLibrary().available


# ---------------------------------------------------------------------------
# Tier G
# ---------------------------------------------------------------------------
class TestGasState(TestCase):

    def test_g1_radius_reduces_to_the_old_formula(self):
        """G1: eps = 1, no helium: the consistent radius is the old n_e-only
        diameter / 2 at n_e = n_H."""
        q = np.array([1e47, 1e49, 1e51])
        np.testing.assert_allclose(2 * stromgren_radius_pc(q, 7.0, 1.0, 0.0),
                                   stromgren_diameter_pc(q, 7.0), rtol=1e-12)

    def test_g2_scalings(self):
        """G2: R ~ (Q / (n^2 eps))^(1/3), U ~ (Q n eps^2)^(1/3), log P/k by hand."""
        r0 = stromgren_radius_pc(1e50, 10.0, 0.22, 0.08)
        self.assertAlmostEqual(stromgren_radius_pc(8e50, 10.0, 0.22, 0.08) / r0, 2.0, places=10)
        self.assertAlmostEqual(stromgren_radius_pc(1e50, 20.0, 0.22, 0.08) / r0, 2 ** (-2 / 3), places=10)
        self.assertAlmostEqual(stromgren_radius_pc(1e50, 10.0, 0.22 / 8, 0.08) / r0, 2.0, places=10)
        u0 = ionization_parameter(1e50, 10.0, 0.22, 0.08)
        self.assertAlmostEqual(ionization_parameter(8e50, 10.0, 0.22, 0.08) / u0, 2.0, places=10)
        self.assertAlmostEqual(ionization_parameter(1e50, 80.0, 0.22, 0.08) / u0, 2.0, places=10)
        self.assertAlmostEqual(ionization_parameter(1e50, 10.0, 0.22 * 2 ** 1.5, 0.08) / u0, 2.0,
                               places=10)
        self.assertAlmostEqual(gas_log_p(10.0, 1e4, 0.08), np.log10(2 * 1.08 * 10 * 1e4), places=12)
        # U from its definition at the radius
        r_cm = r0 * neb.PC_CM
        self.assertAlmostEqual(u0, 1e50 / (4 * np.pi * r_cm ** 2 * 10.0 * neb.C_CM_S), places=15)

    def test_g3_consistent_config_overwrites_the_three(self):
        """G3: gas_density='consistent' derives log_p, log_u, n_e from one n_H;
        the given values are ignored; 'independent' keeps them."""
        c = NebularConfig(knob=1.0, size_mode='stromgren', gas_density='consistent',
                          n_h_cm3=10.0, filling_factor=0.22, source_q_h=1e50,
                          log_u=-3.7, log_p=7.0, n_e_cm3=1.0)
        st = gas_state(1e50, 10.0, 0.22, c.t_e_k, c.he1_h)
        self.assertEqual(c.log_p, st['log_p'])
        self.assertEqual(c.log_u, st['log_u'])
        self.assertAlmostEqual(c.n_e_cm3, 10.8, places=12)
        self.assertAlmostEqual(c.stromgren_diameter_pc(1e50), 2 * st['r_pc'], places=10)
        i = NebularConfig(knob=1.0, log_u=-3.7, log_p=7.0, n_e_cm3=1.0)
        self.assertEqual((i.log_u, i.log_p, i.n_e_cm3), (-3.7, 7.0, 1.0))
        self.assertEqual(i.stromgren_diameter_pc(1e50), stromgren_diameter_pc(1e50, 1.0))
        # the copy helper
        self.assertEqual(i.with_source_q_h(1e50).log_u, st['log_u'])    # defaults n_H 10, eps 0.22

    def test_g4_refusals_and_clip(self):
        """G4: consistent without source_q_h, a bad eps or a P/k off the grid
        raise; U off the grid is clipped and the raw value kept."""
        with self.assertRaises(ValueError):
            NebularConfig(knob=1.0, gas_density='consistent')
        with self.assertRaises(ValueError):
            NebularConfig(knob=1.0, gas_density='consistent', source_q_h=1e50, filling_factor=0.0)
        with self.assertRaises(ValueError):
            NebularConfig(knob=1.0, gas_density='consistent', source_q_h=1e50, n_h_cm3=1e6)
        with self.assertRaises(ValueError):
            NebularConfig(knob=1.0, gas_density='other')
        c = NebularConfig(knob=1.0, gas_density='consistent', source_q_h=1e56, filling_factor=1.0,
                          n_h_cm3=1000.0)
        self.assertEqual(c.log_u, -2.0)
        self.assertGreater(c.log_u_unclipped, -2.0)


    def test_g5_birth_cloud_dust_from_the_gas(self):
        """G5: A_V = (A_V/N_H)_sun Z^p f n_H R_S: grows as (n_H Q eps)^(1/3) at
        fixed Q, linearly in the shell factor, as Z^p; by hand at one point."""
        from artpop.nebular import birth_cloud_a_v_from_gas, AV_PER_NH_SOLAR, PC_CM
        a0 = birth_cloud_a_v_from_gas(1e50, 40.0, 0.22, 0.08)
        self.assertAlmostEqual(birth_cloud_a_v_from_gas(1e50, 320.0, 0.22, 0.08) / a0, 2.0, places=10)
        self.assertAlmostEqual(birth_cloud_a_v_from_gas(8e50, 40.0, 0.22, 0.08) / a0, 2.0, places=10)
        self.assertAlmostEqual(birth_cloud_a_v_from_gas(1e50, 40.0, 0.22, 0.08, shell_factor=3.0) / a0,
                               3.0, places=12)
        self.assertAlmostEqual(birth_cloud_a_v_from_gas(1e50, 40.0, 0.22, 0.08, z_rel=0.1) / a0,
                               0.1, places=12)
        self.assertAlmostEqual(birth_cloud_a_v_from_gas(1e50, 40.0, 0.22, 0.08, z_rel=0.1,
                                                        z_power=0.0) / a0, 1.0, places=12)
        want = AV_PER_NH_SOLAR * 40.0 * stromgren_radius_pc(1e50, 40.0, 0.22, 0.08) * PC_CM
        self.assertAlmostEqual(a0 / want, 1.0, places=12)


# ---------------------------------------------------------------------------
# Tier P
# ---------------------------------------------------------------------------
class TestPlacement(TestCase):

    def test_p1_cmf_weights(self):
        """P1: weights sum to 1, positive, reproducible; slope 2 makes the
        largest cluster dominant on average."""
        w = cluster_weights(5, np.random.RandomState(1))
        self.assertAlmostEqual(w.sum(), 1.0, places=12)
        self.assertTrue(np.all(w > 0))
        np.testing.assert_array_equal(w, cluster_weights(5, np.random.RandomState(1)))
        top = [cluster_weights(4, np.random.RandomState(s)).max() for s in range(400)]
        self.assertGreater(np.mean(top), 0.45)          # 0.49 measured; 0.25 for equal clusters
        flat = [cluster_weights(4, np.random.RandomState(s), slope=0.0, mass_range=1.0 + 1e-9).max()
                for s in range(50)]
        np.testing.assert_allclose(flat, 0.25, rtol=1e-6)

    def test_p1b_cmf_budget(self):
        """P1': complex masses sum to the budget exactly; N grows ~ budget; a
        budget under m_min is one complex; the cap raises m_min and reports the
        CMF mass below it."""
        rng = np.random.RandomState(4)
        m, info = sample_cmf_to_budget(5e4, rng, m_min=1e3)
        self.assertAlmostEqual(m.sum(), 5e4, delta=1e-6)
        self.assertTrue(np.all(m > 0))
        n_small = np.mean([sample_cmf_to_budget(1e4, np.random.RandomState(s))[0].size
                           for s in range(300)])
        n_big = np.mean([sample_cmf_to_budget(1e5, np.random.RandomState(s))[0].size
                         for s in range(300)])
        self.assertGreater(n_big / n_small, 4.0)           # ~ budget, damped by M_max = budget
        one, info1 = sample_cmf_to_budget(500.0, rng, m_min=1e3)
        self.assertEqual(one.tolist(), [500.0])
        capped, info2 = sample_cmf_to_budget(1e8, np.random.RandomState(5), m_min=1e3, n_cap=50)
        self.assertGreater(info2['m_min_used'], 1e3)
        self.assertAlmostEqual(info2['n_expected'], 50.0, delta=0.5)
        self.assertLess(capped.size, 120)
        self.assertAlmostEqual(capped.sum(), 1e8 * (1 - info2['f_below']), delta=1.0)
        self.assertAlmostEqual(info2['f_below'],
                               cmf_mass_fraction(1e3, info2['m_min_used'], 2.0, 1e3, 1e8), places=12)

    def test_p2_centres_on_a_ring(self):
        """P2: the centres' surface density follows the ring Sigma(R): non-zero
        at the centre, highest near R_p, nothing past the cut; the angle is
        uniform; the ellipse squashes."""
        rng = np.random.RandomState(2)
        xy = ring_disk_offsets(200000, 100.0, rng, r_peak_reff=0.4, width_reff=0.3, max_reff=2.0)
        r = np.hypot(*xy.T)
        self.assertLessEqual(r.max(), 200.0)
        edges = np.linspace(0, 200, 41)
        n, _ = np.histogram(r, bins=edges)
        area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sig = n / area
        mid = 0.5 * (edges[1:] + edges[:-1])
        model = ring_surface_density(mid, 40.0, 30.0)
        model *= sig[5:12].sum() / model[5:12].sum()
        ok = n > 400
        np.testing.assert_allclose(sig[ok], model[ok], rtol=0.12)
        self.assertGreater(sig[0], 0.2 * sig.max())                  # non-zero at the centre
        self.assertTrue(30.0 <= mid[int(np.argmax(sig))] <= 50.0)    # peak at R_p = 40
        self.assertAlmostEqual(np.mean(np.arctan2(xy[:, 1], xy[:, 0]) > 0), 0.5, delta=0.01)
        c2 = cluster_centres(20000, 100.0, 0.0, 0.5, np.random.RandomState(2))
        self.assertAlmostEqual(np.std(c2[:, 1]) / np.std(c2[:, 0]), 0.5, delta=0.02)

    def test_p2b_asymmetric_ring(self):
        """P2': an outer width below the inner one moves the light inside the
        peak's far side: Sigma falls faster outside R_p than inside, the
        centre stays non-zero, nothing lands past the cut."""
        rng = np.random.RandomState(8)
        xy = ring_disk_offsets(200000, 100.0, rng, r_peak_reff=0.8, width_reff=0.4,
                               max_reff=1.4, width_out_reff=0.2)
        r = np.hypot(*xy.T)
        self.assertLessEqual(r.max(), 140.0)
        edges = np.linspace(0, 140, 29)
        n, _ = np.histogram(r, bins=edges)
        sig = n / (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))
        mid = 0.5 * (edges[1:] + edges[:-1])
        model = ring_surface_density(mid, 80.0, 40.0, 20.0)
        model *= sig[12:20].sum() / model[12:20].sum()
        ok = n > 400
        np.testing.assert_allclose(sig[ok], model[ok], rtol=0.12)
        self.assertTrue(70.0 <= mid[int(np.argmax(sig))] <= 90.0)
        self.assertGreater(sig[0], 0.05 * sig.max())
        # 20 px beyond the peak (one outer width) vs 20 px inside it
        self.assertAlmostEqual(float(ring_surface_density(100.0, 80.0, 40.0, 20.0)),
                               np.exp(-0.5), places=12)
        self.assertAlmostEqual(float(ring_surface_density(60.0, 80.0, 40.0, 20.0)),
                               np.exp(-0.125), places=12)

    def test_g6_lyc_efficiency(self):
        """G6: lyc_efficiency scales every row's raw and captured photons (so
        the lines, the DIG and the sizes alike); 1 is the pre-feature model."""
        if not (_HAVE_C3K and _HAVE_TABLE):
            self.skipTest('needs C3K and MAPPINGS')
        from artpop.nebular import apply_to_rows
        from artpop.filters import load_filter_system
        fs = load_filter_system('LSST', bands=['LSST_r'])
        curves = {'LSST_r': fs.get_trans('LSST_r')}
        lt = np.log10([45e3, 35e3])

        def run(**kw):
            # a_v_max 0: no coupled birth-cloud screen, so the row's added light is the gas
            cfg = NebularConfig(**dict(dict(knob=0.4, a_v_max=0.0, size_mode='stromgren',
                                            dig_absorbed=0.95), **kw))
            return apply_to_rows(cfg, {'LSST_r': np.zeros(2)}, curves, 6.5, -1.0, lt,
                                 np.array([4.0, 4.0]), np.array([5.3, 4.5]))
        a, _, ia = run()
        b, _, ib = run(lyc_efficiency=1.0)
        c, _, ic = run(lyc_efficiency=0.5)
        np.testing.assert_array_equal(a['LSST_r'], b['LSST_r'])
        np.testing.assert_allclose(ic['q_h_raw'], 0.5 * ia['q_h_raw'], rtol=1e-14)
        np.testing.assert_allclose(ic['q_h'], 0.5 * ia['q_h'], rtol=1e-14)
        gas_a = 10 ** (-0.4 * a['LSST_r']) - 1.0
        gas_c = 10 ** (-0.4 * c['LSST_r']) - 1.0
        np.testing.assert_allclose(gas_c, 0.5 * gas_a, rtol=1e-9)
        with self.assertRaises(ValueError):
            NebularConfig(knob=0.4, lyc_efficiency=0.0)

    def test_p7_scale_mass_age(self):
        """P7: a(M, t) = sqrt([a0 (M/M_ref)^beta]^2 + (sigma t)^2)."""
        self.assertAlmostEqual(float(complex_scale_pc(1e4, 0.0)), 10.0, places=12)
        self.assertAlmostEqual(float(complex_scale_pc(1e5, 0.0)), 10.0 * 10 ** 0.3, places=10)
        a = float(complex_scale_pc(1e4, 20.0, sigma_exp_kms=3.0))
        self.assertAlmostEqual(a, np.hypot(10.0, 3.0 * KMS_PC_PER_MYR * 20.0), places=10)

    def test_p3_plummer_half_light_is_a(self):
        """P3: projected Plummer offsets have median radius a (half-light) and
        the stamp is unit-sum with the same half-light radius."""
        off = plummer_offsets(200000, 4.0, np.random.RandomState(3))
        self.assertAlmostEqual(np.median(np.hypot(*off.T)), 4.0, delta=0.05)
        st, half = plummer_stamp(4.0, 0.2, -0.3)
        self.assertAlmostEqual(st.sum(), 1.0, places=12)
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
        rr = np.hypot(xx - 0.2, yy + 0.3).ravel()
        o = np.argsort(rr)
        self.assertAlmostEqual(rr[o][np.searchsorted(np.cumsum(st.ravel()[o]), 0.5)], 4.0, delta=0.4)
        d, h0 = plummer_stamp(0.05)
        self.assertEqual((d.shape, h0), ((1, 1), 0))

    def test_p4_paint_clips_at_the_edge(self):
        """P4: a stamp painted inside the frame keeps all its flux; at the edge
        only the in-frame part lands."""
        st, half = plummer_stamp(1.0)                   # 30 a = 31 px fits the frame
        im = np.zeros((101, 101))
        paint_stamp(im, st, half, 50, 50, 7.0)
        self.assertAlmostEqual(im.sum(), 7.0, places=10)
        im2 = np.zeros((101, 101))
        paint_stamp(im2, st, half, 0, 50, 1.0)
        self.assertAlmostEqual(im2.sum(), st[:, half:].sum(), places=12)

    def test_p5_spec_validation(self):
        """P5: ClusterSpec refuses bad weights, sizes and diffuse shares."""
        with self.assertRaises(ValueError):
            ClusterSpec(2, [0.5, 0.4], 20.0)
        with self.assertRaises(ValueError):
            ClusterSpec(2, [1.0], 0.0)
        with self.assertRaises(ValueError):
            ClusterSpec(2, [1.0], 20.0, f_diffuse=1.5)
        s = ClusterSpec(2, [0.25, 0.75], 20.0, f_diffuse=0.4)
        np.testing.assert_allclose(s.component_probs(), [0.4, 0.15, 0.45])


# ---------------------------------------------------------------------------
# Tier Q
# ---------------------------------------------------------------------------
def _pops(knob=1.0, distance=20.0):
    from artpop.stars import MISTSSP
    cfg = NebularConfig(knob=knob, size_mode='stromgren', gas_density='consistent',
                        source_q_h=1e50) if knob > 0 else None
    kw = dict(feh=-1.0, phot_system='LSST', version='2.5', distance=distance * u.Mpc,
              mag_limit=28.0, mag_limit_band='LSST_r')
    if cfg is not None:
        kw['nebular'] = cfg
    old = MISTSSP(log_age=9.5, total_mass=1e7, random_state=1, **kw)
    young = MISTSSP(log_age=6.7, total_mass=5e4, random_state=2, **kw)
    mid = MISTSSP(log_age=7.8, total_mass=2e5, random_state=3, **kw)
    return old, young, mid, old + young + mid


_SRC_KW = dict(r_eff=0.5, n=0.8, theta=30, ellip=0.3, xy_dim=901, pixel_scale=0.2 / 3)


@skipUnless(_HAVE_MIST25 and _HAVE_C3K and _HAVE_TABLE, 'needs MIST v2.5 LSST, C3K and MAPPINGS')
class TestClusteredSource(TestCase):

    @classmethod
    def setUpClass(cls):
        from artpop.source import ClusteredSersicSP, SersicSP
        cls.old, cls.young, cls.mid, cls.pop = _pops()
        cls.specs = [ClusterSpec(2, [0.7, 0.3], 20.0, a_sigma_dex=0.2),
                     ClusterSpec(3, [0.5, 0.25, 0.25], 100.0, f_diffuse=0.5)]
        cls.pop.rng = np.random.RandomState(11)
        cls.src0 = SersicSP(cls.pop, **_SRC_KW)
        cls.pop.rng = np.random.RandomState(11)
        cls.src = ClusteredSersicSP(cls.pop, clusters=cls.specs, r_peak_reff=0.3,
                                    width_reff=0.2, **_SRC_KW)

    def test_q1_unclustered_stars_unchanged(self):
        """Q1: the old SSP's stars sit exactly where SersicSP puts them."""
        n = self.old.num_stars
        k0, k1 = self.src0.star_mask[:n].sum(), self.src.star_mask[:n].sum()
        self.assertEqual(k0, k1)
        np.testing.assert_array_equal(self.src0.xy[:k0], self.src.xy[:k1])
        self.assertTrue(np.all(self.src.star_cluster[:k1] == -1))

    def test_q2_members_on_their_clusters(self):
        """Q2: every young star is in a cluster (f_diffuse 0); the 100 Myr SSP is
        about half diffuse; members' median distance from their centre is ~a."""
        lab = self.pop.ssp_labels[self.src.star_mask]
        cl, comp = self.src.star_cluster, self.src.star_component
        self.assertTrue(np.all(cl[lab == 2] >= 0))
        frac_diff = np.mean(comp[lab == 3] == 0)
        self.assertAlmostEqual(frac_diff, 0.5, delta=4 * np.sqrt(0.25 / max(1, (lab == 3).sum())) + 0.02)
        for c in self.src.clusters:
            m = cl == c['id']
            if m.sum() < 30:
                continue
            r = np.hypot(self.src.x[m] - c['x'], self.src.y[m] - c['y'])
            self.assertLess(abs(np.median(r) / c['a_px'] - 1.0), 0.35)
        self.assertEqual(len(self.src.clusters), 5)

    def test_q3_smooth_mixture_exact(self):
        """Q3: the clusters' share of the smooth light (stars + gas) is exactly
        sum_c smooth_weight x the SSP's integrated flux, and the gas share is
        integrated_blob_frac of it."""
        zpt = 27.0
        f = lambda m: 10 ** (0.4 * (zpt - m))
        for band in ('LSST_g', 'LSST_r'):
            stars, gas = self.src.smooth_image(band, f, (901, 901))
            yy, xx = np.mgrid[:901, :901]
            sersic = self.src.smooth_model(xx, yy).sum()
            exp_cl, exp_gas = 0.0, 0.0
            for c in self.src.clusters:
                rec = self.pop.ssp_components[c['label'] - 1]
                fl = f(rec['integrated_abs_mags'][band] + self.pop.dist_mod)
                exp_cl += c['smooth_weight'] * fl
                exp_gas += c['smooth_weight'] * fl * rec['integrated_blob_frac'].get(band, 0.0)
            # 1e-4: the 100 pc clusters' 30 a stamps (465 px) reach the frame edge
            self.assertAlmostEqual((stars.sum() - sersic + gas.sum()) / exp_cl, 1.0, places=4)
            self.assertAlmostEqual(gas.sum() / exp_gas, 1.0, places=9)

    def test_q4_one_sphere_per_cluster(self):
        """Q4: the young stars' gas lands in one sphere per cluster centred on it,
        with the cluster's size; the total image flux equals the per-star-sphere
        render of the same stars (only positions moved)."""
        from artpop.image import IdealImager
        im = IdealImager()
        band = 'LSST_r'
        flux = 10 ** (0.4 * (27.0 - self.src.mags[band]))
        frac = np.asarray(self.pop.nebular_blob_frac[band])[self.src.star_mask]
        img = im._stromgren_blobs(self.src, flux * frac, self.src.distance_angular)
        exp = self.src.paint_cluster_gas(
            np.zeros_like(img),
            np.bincount(self.src.star_cluster[self.src.star_cluster >= 0],
                        weights=(flux * frac)[self.src.star_cluster >= 0],
                        minlength=len(self.src.clusters)))
        rest = np.where(self.src.star_cluster >= 0, 0.0, flux * frac)
        if rest.any():
            exp = exp + im._stromgren_star_blobs(self.src, rest, self.src.distance_angular)
        np.testing.assert_allclose(img, exp, rtol=0, atol=1e-12 * img.max())
        self.assertAlmostEqual(img.sum() / (flux * frac).sum(), 1.0, places=9)
        c = max(self.src.clusters, key=lambda c: c['q_h'])
        self.assertGreater(self.src.cluster_gas_radius_px(c), 1.0)
        self.assertAlmostEqual(self.src.cluster_gas_radius_px(c) / self.src.px_per_pc,
                               0.5 * self.pop.nebular.stromgren_diameter_pc(c['q_h']), places=8)

    def test_q5_flux_conserved(self):
        """Q5: total image flux, clustered vs Sersic placement of the same
        population: equal to 0.5 % (the residual is the Sersic model's own
        pixel-sampling and frame truncation, which the clusters' share no
        longer goes through)."""
        from artpop.image import IdealImager
        for band in ('LSST_g', 'LSST_i'):
            a = IdealImager().observe(self.src0, band, zpt=27.0).image.sum()
            b = IdealImager().observe(self.src, band, zpt=27.0).image.sum()
            self.assertAlmostEqual(b / a, 1.0, delta=0.005)

    def test_q6_expected_photons(self):
        """Q6: q_h_per_msun x mass is the expectation of the sampled + smooth
        captured photons (5 seeds, 10 %)."""
        from artpop.stars import MISTSSP
        cfg = NebularConfig(knob=1.0, size_mode='stromgren')
        tot = []
        for s in range(5):
            y = MISTSSP(log_age=6.7, feh=-1.0, phot_system='LSST', version='2.5', total_mass=2e5,
                        distance=20 * u.Mpc, mag_limit=28.0, mag_limit_band='LSST_r',
                        nebular=cfg, random_state=s)
            tot.append(y.nebular_q_h.sum() + y.integrated_q_h)
        exp = neb.q_h_per_msun(y.isochrone) * 2e5
        self.assertAlmostEqual(np.mean(tot) / exp, 1.0, delta=0.1)

    def test_p6_age_linked_sites(self):
        """P6: with site_reuse = 1 every older complex sits near a younger one;
        the disk-plane offsets have <d^2> = 2 sigma^2, sigma^2 = s0^2 + (v t)^2;
        site_reuse = 0 links none."""
        from artpop.source import ClusteredSersicSP
        n = 400
        specs = [ClusterSpec(3, np.full(n, 1.0 / n), 50.0, age_myr=65.0),
                 ClusterSpec(2, [0.5, 0.5], 20.0, age_myr=5.0, masses=[1e4, 1e4])]
        self.pop.rng = np.random.RandomState(12)
        src = ClusteredSersicSP(self.pop, clusters=specs, site_reuse=1.0, site_s0_pc=20.0,
                                drift_kms=5.0, **_SRC_KW)
        by_id = {c['id']: c for c in src.clusters}
        young = [c for c in src.clusters if c['label'] == 2]
        old = [c for c in src.clusters if c['label'] == 3]
        self.assertEqual(young[0]['id'], 0)                     # youngest placed first
        self.assertTrue(all(c['site'] == -1 for c in young))
        self.assertTrue(all(c['site'] in (0, 1) for c in old))
        d2 = [np.subtract(c['disk_pc'], by_id[c['site']]['disk_pc']) for c in old]
        sig2 = 20.0 ** 2 + (5.0 * KMS_PC_PER_MYR * 65.0) ** 2
        self.assertAlmostEqual(np.mean(np.sum(np.square(d2), axis=1)) / (2 * sig2), 1.0, delta=0.15)
        self.pop.rng = np.random.RandomState(12)
        src0 = ClusteredSersicSP(self.pop, clusters=specs, site_reuse=0.0, **_SRC_KW)
        self.assertTrue(all(c['site'] == -1 for c in src0.clusters))

    def test_q8_per_ssp_sizes(self):
        """Q8: ssp_reff_scale puts an SSP's non-complex stars on a Sersic of
        s x r_eff (median radius ~ s x the unscaled one) and its smooth light on
        the matching unit-flux Sersic; an empty map is the plain source."""
        from artpop.source import ClusteredSersicSP
        kw = dict(_SRC_KW, xy_dim=1201)
        self.pop.rng = np.random.RandomState(21)
        s0 = ClusteredSersicSP(self.pop, clusters=[], **kw)
        self.pop.rng = np.random.RandomState(21)
        s1 = ClusteredSersicSP(self.pop, clusters=[], ssp_reff_scale={1: 0.5}, **kw)
        c = (np.asarray(kw['xy_dim']) - 1) / 2
        lab0 = self.pop.ssp_labels[s0.star_mask]
        lab1 = self.pop.ssp_labels[s1.star_mask]
        r0 = np.median(np.hypot(*(s0.xy[lab0 == 1] - c).T))
        r1 = np.median(np.hypot(*(s1.xy[lab1 == 1] - c).T))
        self.assertAlmostEqual(r1 / r0, 0.5, delta=0.05)
        zpt = 27.0
        f = lambda m: 10 ** (0.4 * (zpt - m))
        for band in ('LSST_g', 'LSST_r'):
            a, _ = s0.smooth_image(band, f, (1201, 1201))
            b, _ = s1.smooth_image(band, f, (1201, 1201))
            self.assertAlmostEqual(b.sum() / a.sum(), 1.0, delta=0.01)
            yy, xx = np.mgrid[:1201, :1201]
            rr = np.hypot(xx - c, yy - c)
            inner = rr < 0.3 * np.median(rr)
            self.assertGreater(b[inner].sum(), a[inner].sum())     # the old light is more compact
        u1 = s1._sersic_unit_image((1201, 1201), 0.5)
        self.assertAlmostEqual(u1.sum(), s1._sersic_unit_image((1201, 1201)).sum(), delta=0.02)

    def test_q7_bad_label(self):
        """Q7: a spec for a label the population does not have is refused."""
        from artpop.source import ClusteredSersicSP
        with self.assertRaises(ValueError):
            ClusteredSersicSP(self.pop, clusters=[ClusterSpec(7, [1.0], 20.0)], **_SRC_KW)
