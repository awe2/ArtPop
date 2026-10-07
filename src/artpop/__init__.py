__version__ = "0.1.1"

try:
    __ARTPOP_SETUP__
except NameError:
    __ARTPOP_SETUP__ = False

if not __ARTPOP_SETUP__:
    import os
    package_dir = os.path.dirname(__file__)
    data_dir = os.path.join(package_dir, 'data')
    # One root for everything ArtPop stages on a host (`python -m artpop.stage`):
    # <ARTPOP_HOME>/mist, spectra, passbands, tables. Each also has its own
    # environment override (MIST_PATH, ARTPOP_SPECTRA_PATH, ...).
    ARTPOP_HOME = os.getenv('ARTPOP_HOME') or os.path.join(os.path.expanduser('~'), '.artpop')
    MIST_PATH = os.getenv('MIST_PATH')
    if MIST_PATH is None:
        MIST_PATH = ARTPOP_HOME
        if not os.path.exists(MIST_PATH):
            os.makedirs(MIST_PATH, exist_ok=True)
        MIST_PATH = os.path.join(MIST_PATH, 'mist')
        if not os.path.exists(MIST_PATH):
            print('\033[33mWARNING:\033[0m MIST_PATH environment variable does'
                  ' not exist. When you first use a MIST grid, it will be '
                 f'saved in\n{MIST_PATH}. To change this '
                  'location, create a MIST_PATH environment variable.')
            os.mkdir(MIST_PATH)
    from .filters import *
    from .kcorrect import *
    from .nebular import NebularConfig, MappingsLineTable
    from .stars import *
    from .image import *
    from .space import *
    from .source import *
    from .visualization import *
