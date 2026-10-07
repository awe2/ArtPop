"""
`artpop.stage`: one-root staging of everything the photometry needs.

Offline: no network and no staged data. MARCS streaming runs on a synthetic tar
served from memory; DP2 passbands on a synthetic ECSV. The real downloads are
exercised by running ``python -m artpop.stage`` on a host (wiki: pipeline/photometry).
"""
import gzip
import io
import os
import tarfile
import tempfile
from unittest import TestCase, mock

import numpy as np

from artpop import stage
from artpop.filters import filter_curve_dir, load_filter_system
from artpop.kcorrect import photometry_curve_dir


class TestPassbandRule(TestCase):
    """Decision 2026-10-07: LSST photometry fails without DP2's passbands."""

    def test_p1_lsst_without_passbands_raises(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'ARTPOP_PHOTOMETRY_CURVES': tmp}):
            with self.assertRaises(FileNotFoundError) as cm:
                photometry_curve_dir('LSST')
            self.assertIn('2026-10-07', str(cm.exception))
            # an explicit choice is honoured; systems without a requirement fall back
            self.assertEqual(photometry_curve_dir('LSST', filter_curve_dir()), filter_curve_dir())
            self.assertEqual(photometry_curve_dir('Roman'), filter_curve_dir())

    def test_p2_dp2_table_becomes_artpop_curves(self):
        from astropy.table import Table
        import astropy.units as u
        with tempfile.TemporaryDirectory() as tmp:
            w = np.arange(300.0, 1100.5, 0.5)
            t = Table({'wavelength': w * u.nm})
            for k, b in enumerate('ugrizy'):
                t[f'throughput_{b}'] = 0.5 * np.exp(-0.5 * ((w - 360 - 120 * k) / 40) ** 2)
            ecsv = os.path.join(tmp, 'dp2_standard_passbands.ecsv')
            t.write(ecsv)
            dest = os.path.join(tmp, 'passbands', 'LSST')
            stage.dp2_passbands_to_curves(ecsv, dest)
            with mock.patch.dict(os.environ, {'ARTPOP_PHOTOMETRY_CURVES': os.path.dirname(dest)}):
                root = photometry_curve_dir('LSST')
                tw, tt = load_filter_system('LSST', curve_dir=root).get_trans('LSST_r')
            np.testing.assert_allclose(tw, w * 10.0)                     # nm -> Angstrom
            np.testing.assert_array_equal(tt, np.asarray(t['throughput_r']))
            self.assertIn(stage.sha256(ecsv), open(os.path.join(dest, 'PROVENANCE.md')).read())


def _fake_marcs_tar(wave_n=5):
    """A tar of gzipped MARCS-style .flx files: two to keep, three to drop."""
    def flx(v):
        return gzip.compress(('\n'.join(f'{v * (i + 1):.5E}' for i in range(wave_n)) + '\n').encode())
    members = {
        's3500_g+1.0_m1.0_t02_ap_z-1.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx.gz': flx(1.0),
        's3600_g+1.0_m1.0_t02_ap_z-1.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx.gz': flx(2.0),
        's6000_g+1.0_m1.0_t02_ap_z-1.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx.gz': flx(3.0),
        'p3500_g+4.5_m0.0_t01_ap_z-1.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx.gz': flx(4.0),
        's3500_g+1.0_m1.0_t02_ap_z+0.25_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx.gz': flx(5.0),
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for name, data in members.items():
            info = tarfile.TarInfo('./' + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestMARCSStaging(TestCase):

    def test_m1_stream_keeps_the_subset_and_packs_reproducibly(self):
        data = _fake_marcs_tar()
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, 'marcs')
            os.makedirs(root)
            with gzip.open(os.path.join(root, stage.MARCS_WAVE), 'wt') as fh:
                fh.write('\n'.join(str(3000.0 + 1000 * i) for i in range(5)) + '\n')
            # one archive, the alpha-poor one: keeps s3500 and s3600 at [Fe/H] -1;
            # drops 6000 K (> 5000), the plane-parallel dwarf and the
            # supersolar ap model (ap is used below solar only)
            kept = stage.stream_marcs(root, host='mem', archives=stage.MARCS_ARCHIVES[:1],
                                      opener=lambda url: io.BytesIO(data))
            self.assertEqual(kept, 2)
            a = stage.marcs_blocks(root)
            self.assertEqual(list(a), ['marcs_sph_feh-1.00.npz'])
            with np.load(os.path.join(root, 'marcs_sph_feh-1.00.npz')) as z:
                self.assertEqual(z['flux'].shape, (1, 2, 5))
                np.testing.assert_allclose(10 ** z['logt'], [3500, 3600])
            b = stage.marcs_blocks(root)                  # numpy writes .npz deterministically
            self.assertEqual(a['marcs_sph_feh-1.00.npz']['sha256'],
                             b['marcs_sph_feh-1.00.npz']['sha256'])


class TestCheck(TestCase):

    def test_c1_empty_root_reports_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('ARTPOP_', 'ALVISS_SPECTRA', 'ALVISS_KCORR'))}
            env['MIST_PATH'] = os.path.join(tmp, 'mist')
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch('sys.stdout', new_callable=io.StringIO) as out:
                code = stage.main(['--check', '--root', tmp, 'c3k', 'passbands'])
            self.assertEqual(code, 1)
            self.assertIn('MISSING   c3k', out.getvalue())
            self.assertIn('FAIL      passbands', out.getvalue())
            self.assertFalse(os.listdir(tmp), '--check must not create or download anything')

    def test_c2_manifest_pins_every_component(self):
        man = stage.load_manifest()
        if not man:
            self.skipTest('no staging manifest yet')
        self.assertTrue(set(stage.c3k_files()) <= set(man['c3k']['sha256']))
        self.assertEqual(set(man['tremblay']['sha256']), set(stage.TREMBLAY_FILES))
        self.assertTrue(any(n.startswith('marcs_sph_feh') for n in man['marcs']['sha256']))
        self.assertEqual(len(man['passbands']['LSST']['sha256']), 64)
