#!/usr/bin/env python
"""
Pack staged MARCS ``.flx.gz`` files into the blocks `artpop.kcorrect.MARCSLibrary`
reads. Superseded by ``python -m artpop.stage marcs`` (which also downloads them);
kept as a convenience for re-packing an existing ``flx/`` directory.

    python tools/stage_marcs.py --root ~/.artpop/spectra/marcs
"""
import argparse
import os

from artpop.kcorrect import spectra_path
from artpop.stage import marcs_blocks

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    ap.add_argument('--root', default=os.path.join(spectra_path(), 'marcs'))
    for name, info in marcs_blocks(ap.parse_args().root).items():
        print(f"{name}: {info['n_models']} models, {info['n_missing']} holes")
