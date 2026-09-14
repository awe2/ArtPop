# Standard library
import os
import shutil
import tarfile
import tempfile

# Third-party
import requests
import numpy as np
from astropy import units as u
from astropy.utils.misc import isiterable

# Project
from . import MIST_PATH
from .log import logger
from .filters import phot_system_list


__all__ = ['check_random_state',
           'check_units',
           'check_odd',
           'check_xy_dim',
           'embed_slices',
           'fetch_mist_grid_if_needed',
           'mist_grid_dir',
           'mist_version_layout',
           'DEFAULT_MIST_VERSION',
           'MIST_GRID_LAYOUT']


def check_random_state(seed):
    """
    Turn seed into a `~numpy.random.RandomState` instance.

    Parameters
    ----------
    seed : `None`, int, list of ints, or `~numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.

    Returns
    -------
    random_state : `~numpy.random.RandomState`
        RandomState object.

    Notes
    -----
    This routine is adapted from scikit-learn. See
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """
    import numbers

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState) or isinstance(seed, np.random.Generator):
        return seed
    if type(seed)==list:
        if type(seed[0])==int:
            return np.random.RandomState(seed)

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def check_units(value, default_unit):
    """
    Check if an object has units. If not, apply the default unit.

    Parameters
    ----------
    value : float, list-like, or `~astropy.units.Quantity`
        Parameter that has units.
    default_unit : str or `astropy` unit
        The default unit to apply to `value` if it does not have units.

    Returns
    -------
    quantity : `~astropy.units.Quantity`
        `value` with ``astropy`` units.
    """
    t = type(default_unit)
    if type(value) == u.Quantity:
        quantity = value
    elif (t == u.IrreducibleUnit) or (t == u.Unit) or (t == u.CompositeUnit):
        quantity = value * default_unit
    elif t == str:
        quantity = value * getattr(u, default_unit)
    else:
        raise Exception('default_unit must be an astropy unit or string')
    return quantity


def check_odd(val, name='value'):
    """
    Raise Exception if  `val` is not odd.
    """
    if isiterable(val):
        if np.any(np.asarray(val) % 2 == 0):
            raise Exception(f'{name} must be odd')
    else:
        if val % 2 == 0:
            raise Exception(f'{name} must be odd')


def check_xy_dim(xy_dim, force_odd=True):
    """
    Check the format of `xy_dim`.

    Parameters
    ----------
    xy_dim : int or list-like
        The dimensions of mock image in xy units. If `int` is given, it is
        assumed to be both the x and y dimensions and (xy_dim, xy_dim) will be
        returned. Otherwise xy_dim will be returned.
    force_odd : bool, optional
        If True (default), force both the x and y dimensions to be odd.

    Returns
    -------
    xy_dim : `~numpy.ndarray`
        Dimensions of mock image in xy units.
    """
    if not isiterable(xy_dim):
        xy_dim = [xy_dim, xy_dim]
    xy_dim = np.asarray(xy_dim).astype(int)
    if force_odd:
        check_odd(xy_dim, 'xy dimensions')
    return xy_dim


def embed_slices(center, model_shape, image_shape):
    """
    Get slices to embed smaller model array into larger image array.

    Parameters
    ----------
    center : `~numpy.ndarray`
        Center of array in the image coordinates.
    model_shape : tuple
        Shape of the array to embed (dimensions must be odd).
    image_shape : tuple
        Shape of the main image array.

    Returns
    -------
    img_slice, mod_slice : tuples of slices
        Slicing indices. To embed array in image,
        use the following: image[img_slice] = model[mod_slice]
    """
    model_shape = np.asarray(model_shape)
    image_shape = np.asarray(image_shape)

    check_odd(model_shape, 'embed_slices array shape')

    imin = center - model_shape//2
    imax = center + model_shape//2

    amin = (imin < np.array([0,0])) * (-imin)
    amax = model_shape * (imax <= image_shape - 1) +\
           (model_shape - (imax - image_shape + 1)) * (imax > image_shape - 1)

    imin = np.maximum(imin, np.array([0, 0]))
    imax = np.minimum(imax, np.array(image_shape)-1)
    imax += 1

    img_slice = np.s_[imin[0]:imax[0], imin[1]:imax[1]]
    mod_slice = np.s_[amin[0]:amax[0], amin[1]:amax[1]]

    return img_slice, mod_slice


MIST_HOST = 'https://mist.science'

DEFAULT_MIST_VERSION = '1.2'

# How each MIST release lays its synthetic photometry out. The two differ in
# almost every particular, so they are described rather than special-cased:
#
#   v1.2  one tarball per (system, v/vcrit), containing a directory of
#         `.iso.cmd` files at [a/Fe] = 0 only.
#   v2.5  one tarball per system holding BOTH rotation rates, with [a/Fe] a real
#         axis. It unpacks FLAT -- no enclosing directory -- so we make one, or
#         the grid would land loose in MIST_PATH and never be found again.
MIST_GRID_LAYOUT = {
    '1.2': dict(
        url='{host}/data/tarballs_v1.2/MIST_v1.2_vvcrit{v}_{p}.txz',
        grid_dir='MIST_v1.2_vvcrit{v}_{p}',
        iso_file='MIST_v1.2_feh_{feh}_afe_p0.0_vvcrit{v}_{p}.iso.cmd',
        flat_tarball=False,
        has_a_over_fe=False,
    ),
    '2.5': dict(
        url='{host}/data/tarballs_v2.5/isos/{p}.txz',
        grid_dir='MIST_v2.5_{p}',
        iso_file='feh_{feh}_afe_{afe}_vvcrit{v}_full.iso.{p}',
        flat_tarball=True,
        has_a_over_fe=True,
    ),
}


def mist_version_layout(version):
    """Layout description for a MIST release, keyed as a string ('1.2', '2.5')."""
    key = str(version)
    if key not in MIST_GRID_LAYOUT:
        raise ValueError(f'MIST version must be one of '
                         f'{sorted(MIST_GRID_LAYOUT)}, got {version!r}.')
    return key, MIST_GRID_LAYOUT[key]


def mist_grid_dir(phot_system, v_over_vcrit=0.4, mist_path=MIST_PATH,
                  version=DEFAULT_MIST_VERSION):
    """Directory holding one MIST grid, whether or not it has been fetched."""
    key, layout = mist_version_layout(version)
    return os.path.join(mist_path, layout['grid_dir'].format(
        v=f'{float(v_over_vcrit):.1f}', p=phot_system))


def fetch_mist_grid_if_needed(phot_system, v_over_vcrit=0.4,
                              mist_path=MIST_PATH, overwrite=False,
                              version=DEFAULT_MIST_VERSION):
    """
    If needed, fetch a MIST grid from https://mist.science.

    Parameters
    ----------
    phot_system : str
        Photometric system grid to fetch. Must be a supported ArtPop filter
        system, where are listed in `~artpop.filters.phot_system_list`.
    v_over_vcrit : float, optional
        Rotation rate divided by the critical surface linear velocity. Current
        options are 0.4 (default) and 0.0.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the default location of ~/.artpop/mist (or the `MIST_PATH`
        environment variable if you have it set).
    overwrite : bool, optional
        If True, force an overwrite of grid if it exists.
    version : str, optional
        MIST release to fetch. ``'1.2'`` (default, preserving the historical
        behaviour) or ``'2.5'``.
     """
    if phot_system not in phot_system_list:
        raise Exception(f'Photometric system must be in {phot_system_list}.')
    key, layout = mist_version_layout(version)
    v = f'{float(v_over_vcrit):.1f}'
    url = layout['url'].format(host=MIST_HOST, v=v, p=phot_system)
    grid_path = mist_grid_dir(phot_system, v_over_vcrit, mist_path, key)
    if not (overwrite or not os.path.isdir(grid_path)):
        return grid_path

    tarball = os.path.join(mist_path, os.path.basename(url))
    logger.info(f'Fetching MIST v{key} photometry grid for {phot_system}.')
    r = requests.get(url, stream=True)
    # Without this a 404 is written out as a "tarball" and fails later with an
    # unreadable-archive error that says nothing about the real problem.
    r.raise_for_status()
    with open(tarball, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    logger.info(f'Extracting grid from {os.path.basename(url)}.')

    # Extract into a staging directory and move it into place only once the
    # tar has closed, so `grid_path` is either ABSENT or COMPLETE and never the
    # half-populated thing the `isdir` check above would happily accept
    # forever. A v2.5 grid is ~150 files / several GB and takes minutes to
    # unpack; an in-place extractall that gets killed part-way leaves a
    # directory that looks fetched, and every later run then dies on a
    # FileNotFoundError for whichever [Fe/H] never landed.
    staging = tempfile.mkdtemp(prefix=os.path.basename(grid_path) + '.staging.',
                               dir=mist_path)
    try:
        # A flat tarball is unpacked into the directory we name; one that
        # carries its own top-level directory is unpacked beside it, as before.
        with tarfile.open(tarball) as tar:
            # numeric_owner=False + no chown: extracting as root would otherwise
            # try to restore the archive's uid/gid and abort.
            try:
                tar.extractall(staging, filter='data')
            except TypeError:                  # filter= is Python 3.12+
                tar.extractall(staging)
        # flat tarballs unpacked loose into `staging`; the others brought their
        # own top-level directory, which is the one to move
        unpacked = (staging if layout['flat_tarball']
                    else os.path.join(staging, os.path.basename(grid_path)))
        if not os.path.isdir(unpacked):
            raise Exception(
                f'{os.path.basename(url)} did not contain the expected '
                f'directory {os.path.basename(grid_path)}.')
        stale = None
        if os.path.exists(grid_path):          # overwrite=True, or a bad grid
            stale = grid_path + '.stale'
            shutil.rmtree(stale, ignore_errors=True)
            os.replace(grid_path, stale)
        os.replace(unpacked, grid_path)        # atomic within one filesystem
        if stale is not None:
            shutil.rmtree(stale, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    # only now, with a complete grid in place, is the tarball redundant
    os.remove(tarball)
    return grid_path
