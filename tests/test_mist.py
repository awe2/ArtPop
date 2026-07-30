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

