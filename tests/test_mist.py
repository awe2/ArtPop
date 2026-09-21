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
        self.ab = MISTIsochrone(10, -1.5, 'LSST', ab_or_vega='ab')
        self.vega = MISTIsochrone(10, -1.5, 'LSST', ab_or_vega='vega')
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
        ab = MISTIsochrone(10, -1.5, 'WFIRST', ab_or_vega='ab')
        vega = MISTIsochrone(10, -1.5, 'WFIRST', ab_or_vega='vega')
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
            MISTIsochrone(10, -1.5, 'WFIRST', ab_or_vega='ab')
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
