# Standard library
import logging
from unittest import TestCase

# Third-party
import numpy as np

# Project
from artpop.stars import MISTIsochrone, MISTSSP
from artpop.filters import load_zero_point_converter


class TestMIST(TestCase):
    """Unit tests for the MIST isochrones."""

    def setUp(self):
        """Build single isochrone and ssp objects for all test cases."""
        # MIST's own columns: these tests check MIST's AB / Vega zero points,
        # which the synthetic default (AB by integration) does not use
        self.ab = MISTIsochrone(10, -1.5, 'LSST', ab_or_vega='ab', photometry='mist')
        self.vega = MISTIsochrone(10, -1.5, 'LSST', ab_or_vega='vega', photometry='mist')
        self.rng = np.random.RandomState(1234)
        self.ssp = MISTSSP(10.1, -1, 'LSST',  num_stars=1e4,
                           random_state=self.rng)

    def test_mist_iso(self):
        """Test that the MIST ischrones loaded correctly."""
        self.assertEqual(1464, len(self.ab.eep))
        self.assertEqual('ab', self.ab.ab_or_vega)
        self.assertEqual('vega', self.vega.ab_or_vega)
        diff = self.ab.mag_table['LSST_i'] - self.vega.mag_table['LSST_i']
        self.assertTrue(all([abs(d - 0.363627) < 1e-6 for d in diff]))

    def test_mist_ssp(self):
        """Test MIST-specific SSP methods/attributes."""
        self.assertEqual(9950, self.ssp.select_phase('MS').sum())
        self.assertEqual(['PMS', 'MS', 'giants', 'RGB', 'CHeB',
                          'AGB', 'EAGB', 'TPAGB', 'postAGB', 'WDCS'],
                          self.ssp.phases)
        self.assertGreater(self.ssp.select_phase('RGB').sum(),
                           self.ssp.select_phase('AGB').sum())
        giants = self.ssp.select_phase('AGB').sum()
        giants += self.ssp.select_phase('RGB').sum()
        giants += self.ssp.select_phase('CHeB').sum()
        self.assertEqual(giants, self.ssp.select_phase('giants').sum())


class TestZeroPointNames(TestCase):
    """
    Filter names must resolve to a zero point row under BOTH conventions.

    MIST's ``zeropoints.txt`` prefixes some filters with their photometric
    system (``WFIRST_H158``, ``JWST_F070W``) while the isochrone columns are
    bare (``H158``, ``F070W``). When the lookup missed, the offset silently
    became 0.0, so ``ab_or_vega='ab'`` returned MIST's Vega-native WFIRST
    magnitudes unchanged -- a 1.29 mag error in H158 with no error raised.
    """

    # from zeropoints.txt: mag(AB) = mag(Vega) + mag(Vega/AB)
    WFIRST_TO_AB = {'R062': 0.137095, 'Z087': 0.487379, 'Y106': 0.653780,
                    'J129': 0.958363, 'W146': 1.024467, 'H158': 1.287404,
                    'F184': 1.551332}

    def setUp(self):
        self.zpt = load_zero_point_converter()

    def test_bare_and_prefixed_names_agree(self):
        """A bare MIST column name resolves to its prefixed zero point row."""
        for filt, offset in self.WFIRST_TO_AB.items():
            self.assertAlmostEqual(offset, self.zpt.to_ab(filt), places=6)
            self.assertEqual(self.zpt.to_ab(f'WFIRST_{filt}'),
                             self.zpt.to_ab(filt))
        # JWST is bare in the isochrones but prefixed in the current table too
        self.assertAlmostEqual(0.277827, self.zpt.to_ab('F070W'), places=6)
        # LSST agrees in both places and is natively AB -- must stay a no-op
        for filt in ('LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y'):
            self.assertEqual(0.0, self.zpt.to_ab(filt))

    def test_unknown_filter_raises(self):
        """An unresolvable filter is loud, not silently zero."""
        with self.assertRaises(KeyError):
            self.zpt.to_ab('NOT_A_REAL_FILTER')

    def test_wfirst_isochrone_is_converted_to_ab(self):
        """The WFIRST grid is Vega-native; ab_or_vega='ab' must shift it."""
        ab = MISTIsochrone(10, -1.5, 'WFIRST', ab_or_vega='ab', photometry='mist')
        vega = MISTIsochrone(10, -1.5, 'WFIRST', ab_or_vega='vega', photometry='mist')
        for filt, offset in self.WFIRST_TO_AB.items():
            diff = np.unique(np.round(
                np.asarray(ab.mag_table[filt]) -
                np.asarray(vega.mag_table[filt]), 6))
            self.assertEqual(1, len(diff))
            self.assertAlmostEqual(offset, float(diff[0]), places=6)
            self.assertAlmostEqual(offset, ab.zpt_offsets[filt], places=6)
            self.assertEqual(0.0, vega.zpt_offsets[filt])

    def test_no_missing_conversion_warning(self):
        """Building a WFIRST isochrone must not warn about a missing offset."""
        with self.assertLogs('artpop', level='WARNING') as ctx:
            logging.getLogger('artpop').warning('sentinel')
            MISTIsochrone(10, -1.5, 'WFIRST', ab_or_vega='ab', photometry='mist')
        missing = [m for m in ctx.output if 'No AB / Vega conversion' in m]
        self.assertEqual([], missing)



class TestMISTVersions(TestCase):
    """
    v1.2 and v2.5 are different grids, not two spellings of one.

    v1.2's WFIRST is a May 2018 preliminary filter set delivered in Vega; v2.5's
    Roman is the flight filter set delivered in AB, adds F213 (the K-band filter
    v1.2 has no bolometric corrections for), and grids [a/Fe] rather than
    assuming solar-scaled. So both must keep working, and neither may silently
    stand in for the other.
    """

    def test_url_and_path_construction(self):
        """Layouts differ in every particular; assert both, without network."""
        from artpop.util import mist_grid_dir, MIST_GRID_LAYOUT, MIST_HOST
        v12 = MIST_GRID_LAYOUT['1.2']
        self.assertEqual(
            'https://mist.science/data/tarballs_v1.2/'
            'MIST_v1.2_vvcrit0.4_WFIRST.txz',
            v12['url'].format(host=MIST_HOST, v='0.4', p='WFIRST'))
        v25 = MIST_GRID_LAYOUT['2.5']
        self.assertEqual(
            'https://mist.science/data/tarballs_v2.5/isos/Roman.txz',
            v25['url'].format(host=MIST_HOST, v='0.4', p='Roman'))
        # v2.5 unpacks flat, so ArtPop has to create the directory itself
        self.assertTrue(v25['flat_tarball'])
        self.assertFalse(v12['flat_tarball'])
        self.assertTrue(mist_grid_dir('Roman', version='2.5')
                        .endswith('MIST_v2.5_Roman'))
        self.assertTrue(mist_grid_dir('WFIRST', 0.4, version='1.2')
                        .endswith('MIST_v1.2_vvcrit0.4_WFIRST'))
        with self.assertRaises(ValueError):
            mist_grid_dir('Roman', version='9.9')

    def test_feh_and_afe_tokens(self):
        """The two releases encode [Fe/H] in file names differently."""
        from artpop.stars.isochrones import _feh_token, _afe_token
        self.assertEqual('m1.50', _feh_token(-1.5, '1.2'))
        self.assertEqual('p0.00', _feh_token(0.0, '1.2'))
        self.assertEqual('m150', _feh_token(-1.5, '2.5'))
        self.assertEqual('p000', _feh_token(0.0, '2.5'))
        self.assertEqual('m025', _feh_token(-0.25, '2.5'))
        self.assertEqual('m2', _afe_token(-0.2))
        self.assertEqual('p0', _afe_token(0.0))
        self.assertEqual('p6', _afe_token(0.6))

    def test_a_over_fe_is_rejected_off_grid(self):
        """[a/Fe] snaps to MIST's grid; it is not interpolated."""
        with self.assertRaises(Exception):
            MISTIsochrone(10, -1.5, 'Roman', version='2.5', a_over_fe=0.3)
        # v1.2 is solar-scaled only
        with self.assertRaises(Exception):
            MISTIsochrone(10, -1.5, 'WFIRST', version='1.2', a_over_fe=0.4)

    def test_v12_default_unchanged(self):
        """The default is still v1.2, solar-scaled, Vega-converted WFIRST."""
        iso = MISTIsochrone(10, -1.5, 'WFIRST')
        self.assertEqual('1.2', iso.version)
        self.assertEqual(0.0, iso.a_over_fe)
        self.assertAlmostEqual(1.287404, iso.zpt_offsets['H158'], places=6)


class TestMISTBinaryCache(TestCase):
    """The parsed-grid binary cache is a bit-identical no-op (ALVISS E-1 / V-12)."""

    def setUp(self):
        import os
        import tempfile
        from artpop.stars import _read_mist_models as m
        from artpop.stars.isochrones import mist_iso_path
        self._env = os.environ.get('ARTPOP_MIST_CACHE')
        self.tmp = tempfile.mkdtemp(prefix='artpop_mist_cache_')
        os.environ['ARTPOP_MIST_CACHE'] = self.tmp
        m._read_isocmd_keyed.cache_clear()
        self.m = m
        self.fn = mist_iso_path(-1.5, 'LSST')

    def tearDown(self):
        import os
        import shutil
        if self._env is None:
            os.environ.pop('ARTPOP_MIST_CACHE', None)
        else:
            os.environ['ARTPOP_MIST_CACHE'] = self._env
        self.m._read_isocmd_keyed.cache_clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cache_bit_identical(self):
        import os
        m = self.m
        ref = m.IsoCmdReader(self.fn)
        first = m.read_isocmd(self.fn)               # parses and writes
        npy, meta = m.isocmd_cache_paths(self.fn, self.tmp)
        self.assertTrue(os.path.isfile(npy) and os.path.isfile(meta))
        m._read_isocmd_keyed.cache_clear()
        cached = m.read_isocmd(self.fn)              # served from disk
        self.assertIsInstance(cached, m.CachedIsoCmd)
        self.assertEqual(cached.num_ages, ref.num_ages)
        self.assertEqual(cached.ages, ref.ages)
        self.assertEqual(cached.hdr_list, ref.hdr_list)
        self.assertEqual((cached.version, cached.photo_sys, cached.abun,
                          cached.Av_extinction, cached.rot),
                         (ref.version, ref.photo_sys, ref.abun,
                          ref.Av_extinction, ref.rot))
        for i in range(ref.num_ages):
            b = cached.block(i)
            self.assertEqual(b.dtype, ref.isocmds[i].dtype)
            self.assertTrue(np.array_equal(b, ref.isocmds[i]))
        for age in (5.0, 7.13, 9.5, 10.3):
            self.assertEqual(cached.age_index(age), ref.age_index(age))

    def test_cache_invalidated_and_disabled(self):
        import os
        m = self.m
        m.read_isocmd(self.fn)
        npy, meta = m.isocmd_cache_paths(self.fn, self.tmp)
        # a different source stamp -> the sidecar is stale and is rebuilt
        import json
        d = json.load(open(meta))
        d['mtime_ns'] -= 1
        json.dump(d, open(meta, 'w'))
        m._read_isocmd_keyed.cache_clear()
        self.assertIsNone(m._load_cached(self.fn, self.tmp))
        self.assertIsInstance(m.read_isocmd(self.fn), m.CachedIsoCmd)
        # ARTPOP_MIST_CACHE=0 bypasses the cache completely
        os.environ['ARTPOP_MIST_CACHE'] = '0'
        m._read_isocmd_keyed.cache_clear()
        self.assertIsNone(m.isocmd_cache_root('~/.artpop/mist'))
        self.assertIsInstance(m.read_isocmd(self.fn), m.IsoCmdReader)
        # both routes give the same isochrone through the public function
        from artpop.stars.isochrones import fetch_mist_iso_cmd
        text = fetch_mist_iso_cmd(9.0, -1.5, 'LSST')
        os.environ['ARTPOP_MIST_CACHE'] = self.tmp
        m._read_isocmd_keyed.cache_clear()
        cached = fetch_mist_iso_cmd(9.0, -1.5, 'LSST')
        self.assertEqual(text.dtype, cached.dtype)
        self.assertTrue(np.array_equal(text, cached))


class TestFehInterpolationOnEEP(TestCase):
    """
    Off-grid [Fe/H] blends two MIST isochrones point by point in EEP, not by
    row index (`blend_isochrones_on_eep`). Two metallicities at one age start
    at different EEPs and skip different ones, in v1.2 and v2.5 alike.
    """

    DTYPE = [('EEP', float), ('initial_mass', float), ('log_Teff', float),
             ('phase', float)]

    @classmethod
    def _iso(cls, eep, slope, offset, phase):
        eep = np.asarray(eep, dtype=float)
        return np.rec.fromarrays(
            [eep, 0.1 + 0.01 * eep, offset + slope * eep, phase(eep)],
            dtype=cls.DTYPE)

    def test_E1_matches_on_eep_not_row(self):
        """Columns linear in EEP with one slope: the blend is exact at every
        EEP of the union, shared, filled inside a gap or beyond an end; the
        row-index blend is not."""
        from artpop.stars.isochrones import blend_isochrones_on_eep
        ph = lambda e: np.where(e < 30, 0.0, 2.0)
        e0 = np.r_[10:30, 33:50]                      # starts later, has a gap
        e1 = np.r_[5:60]
        a = self._iso(e0, 0.002, 3.6, ph)
        b = self._iso(e1, 0.002, 3.5, ph)
        w = 0.3
        r = blend_isochrones_on_eep(a, b, w)
        self.assertTrue(np.array_equal(r['EEP'], np.union1d(e0, e1)))
        want = (1 - w) * (3.6 + 0.002 * r['EEP']) + w * (3.5 + 0.002 * r['EEP'])
        np.testing.assert_allclose(r['log_Teff'], want, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r['initial_mass'], 0.1 + 0.01 * r['EEP'],
                                   rtol=0, atol=1e-12)
        # what the row-index blend gives at the first shared EEP (10): it
        # pairs EEP 10 of iso_0 with EEP 5 of iso_1
        row = (1 - w) * a['log_Teff'][0] + w * b['log_Teff'][0]
        i = int(np.flatnonzero(r['EEP'] == 10)[0])
        self.assertGreater(abs(row - r['log_Teff'][i]), 1e-3)

    def test_E2_labels_are_not_averaged(self):
        """EEP is exact and phase is the nearer isochrone's label."""
        from artpop.stars.isochrones import blend_isochrones_on_eep
        a = self._iso(np.r_[0:20], 0.0, 3.6, lambda e: np.full(e.size, 3.0))
        b = self._iso(np.r_[0:20], 0.0, 3.6, lambda e: np.full(e.size, 9.0))
        for w, want in ((0.3, 3.0), (0.5, 3.0), (0.7, 9.0)):
            r = blend_isochrones_on_eep(a, b, w)
            self.assertTrue(np.all(r['phase'] == want))
            self.assertTrue(np.array_equal(r['EEP'], a['EEP']))
        # 3.0 blended with itself is not always exactly 3.0 in floating point
        r = blend_isochrones_on_eep(a, a, 0.29)
        self.assertTrue(np.all(r['phase'] == 3.0))

    def test_E3_aligned_isochrones_unchanged(self):
        """Identical EEP sets: bit-identical to the old row blend."""
        from artpop.stars.isochrones import blend_isochrones_on_eep
        rng = np.random.RandomState(7)
        eep = np.r_[200:260]
        a = self._iso(eep, 0.0, 3.6, lambda e: np.zeros(e.size))
        b = self._iso(eep, 0.0, 3.6, lambda e: np.zeros(e.size))
        a['log_Teff'] = 3.6 + rng.normal(0, 0.1, eep.size)
        b['log_Teff'] = 3.7 + rng.normal(0, 0.1, eep.size)
        b['initial_mass'] = a['initial_mass'] * 1.01
        w = 0.37
        r = blend_isochrones_on_eep(a, b, w)
        for n in ('initial_mass', 'log_Teff'):
            self.assertTrue(np.array_equal(r[n], a[n] * (1 - w) + b[n] * w))

    def test_E4_real_grids(self):
        """On MIST v2.5: shared EEPs lie between the two parents, initial mass
        never decreases, and the truncated grids keep their coverage."""
        from artpop.stars.isochrones import (blend_isochrones_on_eep,
                                             fetch_mist_iso_cmd)
        kw = dict(version='2.5', a_over_fe=0.0)
        a = fetch_mist_iso_cmd(10.0, -1.25, 'LSST', **kw)
        b = fetch_mist_iso_cmd(10.0, -1.0, 'LSST', **kw)
        self.assertFalse(np.array_equal(a['EEP'][:5], b['EEP'][:5]))
        r = blend_isochrones_on_eep(a, b, 0.4)
        _, ia, ib = np.intersect1d(a['EEP'], b['EEP'], return_indices=True)
        ir = np.searchsorted(r['EEP'], a['EEP'][ia])
        for n in ('initial_mass', 'log_Teff', 'log_L', 'log_g'):
            lo = np.minimum(a[n][ia], b[n][ib])
            hi = np.maximum(a[n][ia], b[n][ib])
            self.assertTrue(np.all((r[n][ir] >= lo - 1e-12) & (r[n][ir] <= hi + 1e-12)), n)
        self.assertTrue(np.all(np.diff(r['initial_mass']) >= 0))
        self.assertTrue(set(np.unique(r['phase'])) <= set(np.unique(a['phase'])) | set(np.unique(b['phase'])))

        # [Fe/H] = +0.5 starts near 0.5 M_sun; the row blend paired 0.1 M_sun
        # with it. The EEP blend keeps +0.25's low-mass end.
        iso = MISTIsochrone(10.0, 0.4, 'LSST', photometry='mist', **kw)
        self.assertGreater(fetch_mist_iso_cmd(10.0, 0.5, 'LSST', **kw)['initial_mass'][0], 0.45)
        self.assertLess(iso.mini.min(), 0.12)
        self.assertTrue(np.all(np.diff(iso.mini) >= 0))

        # [Fe/H] = -3.0 at log age 10.2 stops at EEP 808: the blend still has
        # the TP-AGB, post-AGB and white dwarfs of -2.5
        lo30 = fetch_mist_iso_cmd(10.2, -3.0, 'LSST', **kw)
        self.assertLessEqual(lo30['EEP'][-1], 808)
        iso = MISTIsochrone(10.2, -2.8, 'LSST', photometry='mist', **kw)
        self.assertGreater(iso.eep.max(), 1710)
        self.assertTrue(np.all(iso.eep == np.round(iso.eep)))


class TestMetalRichLowMassCopy(TestCase):
    """
    MIST v2.5's most metal-rich grids ship without 0.1-0.5 M_sun (the opacity
    tables do not cover them). As alpha-MC (Park et al. 2024, arXiv:2410.21375
    s2) does, those rows are copied from the nearest metallicity.
    """

    KW = dict(version='2.5', a_over_fe=0.0)

    def test_F1_table_is_alpha_mc(self):
        """The donor table is alpha-MC's list; v1.2 and complete grids have none."""
        from artpop.stars.isochrones import MIST_LOW_MASS_DONOR, low_mass_donor
        self.assertEqual({(0.50, 0.0): 0.25, (0.50, 0.2): 0.25, (0.25, 0.4): 0.00,
                          (0.50, 0.4): 0.00, (0.00, 0.6): -0.25, (0.25, 0.6): -0.25},
                         MIST_LOW_MASS_DONOR)
        self.assertEqual(0.25, low_mass_donor('2.5', 0.5, 0.0))
        self.assertIsNone(low_mass_donor('1.2', 0.5, 0.0))
        self.assertIsNone(low_mass_donor('2.5', 0.25, 0.0))

    def test_F2_on_grid_copy(self):
        """+0.5 gets +0.25's rows below its first star, verbatim except [Fe/H]."""
        from artpop.stars.isochrones import fetch_mist_iso_cmd
        raw = fetch_mist_iso_cmd(10.0, 0.5, 'LSST', **self.KW)
        donor = fetch_mist_iso_cmd(10.0, 0.25, 'LSST', **self.KW)
        self.assertGreater(raw['initial_mass'][0], 0.45)
        iso = MISTIsochrone(10.0, 0.5, 'LSST', photometry='mist', **self.KW)
        full = iso.isochrone_full
        rec = iso.low_mass_copied
        self.assertEqual(1, len(rec))
        n = rec[0]['n_rows']
        self.assertGreater(n, 30)
        self.assertEqual(0.25, rec[0]['donor_feh'])
        self.assertIn('arXiv:2410.21375', rec[0]['reference'])
        self.assertEqual(rec, iso.photometry_info['isochrones']['low_mass_copied'])
        self.assertAlmostEqual(0.1, iso.mini.min(), places=3)
        self.assertTrue(np.all(np.diff(full['EEP']) > 0))
        self.assertTrue(np.all(np.diff(full['initial_mass']) >= 0))
        # the native rows are untouched
        for c in raw.dtype.names:
            self.assertTrue(np.array_equal(full[c][n:], raw[c]), c)
        # the copied rows are the donor's, [Fe/H] moved by +0.25
        for c in raw.dtype.names:
            want = donor[c][:n] + (0.25 if c.startswith('[Fe/H]') else 0.0)
            np.testing.assert_allclose(full[c][:n], want, rtol=0, atol=1e-12, err_msg=c)

    def test_F3_complete_grids_untouched(self):
        """A grid MIST ships complete is served exactly as read."""
        from artpop.stars.isochrones import fetch_mist_iso_cmd
        for feh, kw in ((0.25, self.KW), (0.5, dict(version='1.2'))):
            iso = MISTIsochrone(10.0, feh, 'LSST', photometry='mist', **kw)
            raw = fetch_mist_iso_cmd(10.0, feh, 'LSST', **kw)
            self.assertEqual([], iso.low_mass_copied)
            for c in ('EEP', 'initial_mass', 'log_Teff', 'log_L'):
                self.assertTrue(np.array_equal(iso.isochrone_full[c], raw[c]), c)

    def test_F4_blend_takes_the_copy(self):
        """Between +0.25 and +0.5 the low-mass end is +0.25's, as copied."""
        from artpop.stars.isochrones import fetch_mist_iso_cmd
        iso = MISTIsochrone(10.0, 0.4, 'LSST', photometry='mist', **self.KW)
        donor = fetch_mist_iso_cmd(10.0, 0.25, 'LSST', **self.KW)
        n = iso.low_mass_copied[0]['n_rows']
        full = iso.isochrone_full
        for c in ('initial_mass', 'log_Teff', 'log_L', 'log_g'):
            np.testing.assert_allclose(full[c][:n], donor[c][:n], rtol=0, atol=1e-12, err_msg=c)
        np.testing.assert_allclose(full['[Fe/H]_init'][:n], donor['[Fe/H]_init'][:n] + 0.6 * 0.25,
                                   rtol=0, atol=1e-12)
