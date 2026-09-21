#########################################################################################
# Helper class for reading MIST iso.cmd files
# Slightly modified from 
# https://github.com/jieunchoi/MIST_codes/blob/master/scripts/read_mist_models.py
#########################################################################################
import os
import json
import logging
import tempfile
import functools
import numpy as np

__all__ = ['IsoReader', 'IsoCmdReader', 'CachedIsoCmd', 'read_isocmd',
           'isocmd_block', 'isocmd_cache_root', 'isocmd_cache_paths',
           'write_isocmd_cache']

logger = logging.getLogger(__name__)


class IsoReader(object):

    """

    Reads in MIST isochrone files.

    """

    def __init__(self, filename, verbose=False):

        """

        Args:
            filename: the name of .iso file.

        Usage:
            >> iso = read_mist_models.ISO('MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4.iso')
            >> age_ind = iso.age_index(8.0)
            >> logTeff = iso.isos[age_ind]['log_Teff']
            >> logL = iso.isos[age_ind]['log_L']

        Attributes:
            version     Dictionary containing the MIST and MESA version numbers.
            abun        Dictionary containing Yinit, Zinit, [Fe/H], and [a/Fe] values.
            rot         Rotation in units of surface v/v_crit.
            ages        List of ages.
            num_ages    Number of isochrones.
            hdr_list    List of column headers.
            isos        Data.

        """

        self.filename = filename
        if verbose:
            print('Reading in: ' + self.filename)

        self.version, self.abun, self.rot, self.ages, self.num_ages, self.hdr_list, self.isos = self.read_iso_file()

    def read_iso_file(self):

        """
        Reads in the isochrone file.

        Args:
            filename: the name of .iso file.

        """

        #open file and read it in
        with open(self.filename) as f:
            content = [line.split() for line in f]
        version = {'MIST': content[0][-1], 'MESA': content[1][-1]}
        abun = {content[3][i]:float(content[4][i]) for i in range(1,5)}
        rot = float(content[4][-1])
        num_ages = int(content[6][-1])

        #read one block for each isochrone
        iso_set = []
        ages = []
        counter = 0
        data = content[8:]
        for i_age in range(num_ages):
            #grab info for each isochrone
            num_eeps = int(data[counter][-2])
            num_cols = int(data[counter][-1])
            hdr_list = data[counter+2][1:]
            formats = tuple([np.int32]+[np.float64 for i in range(num_cols-1)])
            iso = np.zeros((num_eeps),{'names':tuple(hdr_list),'formats':tuple(formats)})
            #read through EEPs for each isochrone
            for eep in range(num_eeps):
                iso_chunk = data[3+counter+eep]
                iso[eep]=tuple(iso_chunk)
            iso_set.append(iso)
            ages.append(iso[0][1])
            counter+= 3+num_eeps+2
        return version, abun, rot, ages, num_ages, hdr_list, iso_set

    def age_index(self, age):

        """
        Returns the index for the user-specified age.

        Args:
            age: the age of the isochrone.

        """

        diff_arr = abs(np.array(self.ages) - age)
        age_index = np.where(diff_arr == min(diff_arr))[0][0]

        if ((age > max(self.ages)) | (age < min(self.ages))):
            print('The requested age is outside the range. Try between ' + str(min(self.ages)) + ' and ' + str(max(self.ages)))

        return age_index


            
class IsoCmdReader(object):
    """
    Reads in MIST CMD files.
    """
    
    def __init__(self, filename, verbose=False):
        """
        
        Parameters
        -----------
        filename: the name of .iso.cmd file.
         
        Usage:
            >> isocmd = IsoCmdReader('MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4.iso.cmd')
            >> age_ind = isocmd.age_index(7.0)
            >> B = isocmd.isocmds[age_ind]['Bessell_B']
            >> V = isocmd.isocmds[age_ind]['Bessell_V']
        
        Attributes:
            version         Dictionary containing the MIST and MESA version numbers.
            photo_sys       Photometric system. 
            abun            Dictionary containing Yinit, Zinit, [Fe/H], and [a/Fe] values.
            Av_extinction   Av for CCM89 extinction.
            rot             Rotation in units of surface v/v_crit.
            ages            List of ages.
            num_ages        Number of ages.
            hdr_list        List of column headers.
            isocmds         Data.
        """
        
        self.filename = filename
        if verbose:
            print('Reading in: ' + self.filename)
            
        self.version, self.photo_sys, self.abun, self.Av_extinction, self.rot, self.ages, self.num_ages, self.hdr_list, self.isocmds = self.read_isocmd_file()
    
    def read_isocmd_file(self):

        """

        Reads in the cmd file.
        
        Args:
            filename: the name of .iso.cmd file.
        
        """
        
        #open file and read it in
        with open(self.filename) as f:
            content = [line.split() for line in f]
        version = {'MIST': content[0][-1], 'MESA': content[1][-1]}
        photo_sys = ' '.join(content[2][4:])
        abun = {content[4][i]:float(content[5][i]) for i in range(1,5)}
        rot = float(content[5][-1])
        num_ages = int(content[7][-1])
        Av_extinction = float(content[8][-1])
        
        #read one block for each isochrone
        isocmd_set = []
        ages = []
        counter = 0
        data = content[10:]
        for i_age in range(num_ages):
            #grab info for each isochrone
            num_eeps = int(data[counter][-2])
            num_cols = int(data[counter][-1])
            hdr_list = data[counter+2][1:]
            formats = tuple([np.int32]+[np.float64 for i in range(num_cols-1)])
            isocmd = np.zeros((num_eeps),{'names':tuple(hdr_list),'formats':tuple(formats)})
            #read through EEPs for each isochrone
            for eep in range(num_eeps):
                isocmd_chunk = data[3+counter+eep]
                isocmd[eep]=tuple(isocmd_chunk)
            isocmd_set.append(isocmd)
            ages.append(isocmd[0][1])
            counter+= 3+num_eeps+2
        return version, photo_sys, abun, Av_extinction, rot, ages, num_ages, hdr_list, isocmd_set

    def age_index(self, age):
        
        """

        Returns the index for the user-specified age.
        
        Args:
            age: the age of the isochrone.
        
        """
        
        diff_arr = abs(np.array(self.ages) - age)
        age_index = np.where(diff_arr == min(diff_arr))[0][0]
        
        if ((age > max(self.ages)) | (age < min(self.ages))):
            print('The requested age is outside the range. Try between ' + str(min(self.ages)) + ' and ' + str(max(self.ages)))
            
        return age_index



#########################################################################################
# Binary cache for the parsed .iso.cmd grids (ALVISS E-1)
#
# `IsoCmdReader` parses the whole text file (107 ages, ~43 MB) to serve one age
# block; that is ~0.5 s per call and dominates a multi-bin population build.
# The functions below save the parsed structured arrays ONCE as one .npy (all
# ages concatenated) plus a .json with the header metadata and block offsets,
# and afterwards memory-map the .npy and slice the requested age out. The
# values stored are exactly the float64/int32 the text parse produced, so a
# cached read is bit-identical to the text read (tests/test_mist.py checks it).
#
# Location: $ARTPOP_MIST_CACHE, or <mist_path>/binary_cache/<grid dir>/. Set
# ARTPOP_MIST_CACHE=0 (or "off"/"false") to bypass the cache entirely. The
# cache is never load-bearing: any failure to write or read it falls back to
# the text parser with a warning.
#########################################################################################

CACHE_FORMAT = 1
_CACHE_DISABLED = {'0', 'off', 'false', 'no', 'none', ''}


def isocmd_cache_root(mist_path=None):
    """
    Directory holding the binary cache, or None if the cache is disabled.

    `mist_path` is the directory the grids live under (the parent of the
    per-system grid directories); it is only used when ARTPOP_MIST_CACHE is
    unset.
    """
    env = os.environ.get('ARTPOP_MIST_CACHE')
    if env is not None:
        if env.strip().lower() in _CACHE_DISABLED:
            return None
        return os.path.expanduser(env)
    if mist_path is None:
        return None
    return os.path.join(os.path.expanduser(mist_path), 'binary_cache')


def isocmd_cache_paths(filename, root):
    """(npy, json) sidecar paths for one grid file under cache root `root`."""
    filename = os.path.abspath(filename)
    grid_dir = os.path.basename(os.path.dirname(filename))
    base = os.path.basename(filename)
    d = os.path.join(root, grid_dir)
    return os.path.join(d, base + '.npy'), os.path.join(d, base + '.json')


def _source_stamp(filename):
    st = os.stat(filename)
    return {'size': int(st.st_size), 'mtime_ns': int(st.st_mtime_ns)}


def write_isocmd_cache(reader, filename, root):
    """
    Write the binary sidecar for `reader` (an `IsoCmdReader` of `filename`).

    All age blocks must share one dtype (they do for every MIST release); the
    blocks are concatenated and the start offsets recorded. The .npy is written
    to a temporary file in the target directory and moved into place, so a
    concurrent reader never sees a partial file. Returns the .npy path.
    """
    npy, meta_path = isocmd_cache_paths(filename, root)
    blocks = reader.isocmds
    dtype = blocks[0].dtype
    if any(b.dtype != dtype for b in blocks):
        raise ValueError(f'{filename}: age blocks do not share one dtype')
    data = np.concatenate(blocks)
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    meta = {
        'format': CACHE_FORMAT,
        'source': os.path.abspath(filename),
        **_source_stamp(filename),
        'version': reader.version,
        'photo_sys': reader.photo_sys,
        'abun': reader.abun,
        'Av_extinction': reader.Av_extinction,
        'rot': reader.rot,
        'ages': [float(a) for a in reader.ages],
        'num_ages': int(reader.num_ages),
        'hdr_list': list(reader.hdr_list),
        'offsets': [int(o) for o in offsets],
    }
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(npy), suffix='.npy.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.save(f, data)
        os.replace(tmp, npy)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(npy), suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(meta, f)
        os.replace(tmp, meta_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return npy


class CachedIsoCmd(object):
    """
    A memory-mapped, cached MIST CMD grid with the `IsoCmdReader` attributes.

    `block(i)` returns age block `i` as a plain structured ndarray (a copy of
    the memmap slice, the same type `IsoCmdReader.isocmds[i]` is). `isocmds`
    materialises every block for API compatibility; prefer `block`.
    """

    def __init__(self, npy, meta):
        self.filename = meta['source']
        self.cache_path = npy
        self.version = meta['version']
        self.photo_sys = meta['photo_sys']
        self.abun = meta['abun']
        self.Av_extinction = meta['Av_extinction']
        self.rot = meta['rot']
        self.ages = [np.float64(a) for a in meta['ages']]
        self.num_ages = meta['num_ages']
        self.hdr_list = meta['hdr_list']
        self._offsets = np.asarray(meta['offsets'], dtype=np.int64)
        self._mm = np.load(npy, mmap_mode='r')
        if self._mm.shape[0] != self._offsets[-1]:
            raise ValueError(f'{npy}: row count does not match its metadata')

    def block(self, i):
        """Age block `i` as an in-memory structured array."""
        lo, hi = self._offsets[i], self._offsets[i + 1]
        return np.array(self._mm[lo:hi])

    @property
    def isocmds(self):
        return [self.block(i) for i in range(self.num_ages)]

    def age_index(self, age):
        """Same nearest-age rule (first match on ties) as `IsoCmdReader`."""
        diff_arr = abs(np.array(self.ages) - age)
        age_index = np.where(diff_arr == min(diff_arr))[0][0]
        if ((age > max(self.ages)) | (age < min(self.ages))):
            print('The requested age is outside the range. Try between ' + str(min(self.ages)) + ' and ' + str(max(self.ages)))
        return age_index


def _load_cached(filename, root):
    """`CachedIsoCmd` for `filename` if a valid sidecar exists, else None."""
    npy, meta_path = isocmd_cache_paths(filename, root)
    if not (os.path.isfile(npy) and os.path.isfile(meta_path)):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        stamp = _source_stamp(filename)
        if (meta.get('format') != CACHE_FORMAT or meta.get('size') != stamp['size']
                or meta.get('mtime_ns') != stamp['mtime_ns']):
            logger.info(f'MIST cache for {os.path.basename(filename)} is stale; rebuilding')
            return None
        return CachedIsoCmd(npy, meta)
    except Exception as e:  # noqa: BLE001 -- the cache must never break a read
        logger.warning(f'ignoring unreadable MIST cache {npy}: {e!r}')
        return None


@functools.lru_cache(maxsize=64)
def _read_isocmd_keyed(filename, root, size, mtime_ns):
    # size / mtime_ns are part of the key only so an edited grid file is
    # re-read within a long process; they are not used here.
    if root is not None:
        cached = _load_cached(filename, root)
        if cached is not None:
            return cached
    reader = IsoCmdReader(filename, verbose=False)
    if root is not None:
        try:
            write_isocmd_cache(reader, filename, root)
            cached = _load_cached(filename, root)
            if cached is not None:
                return cached
        except Exception as e:  # noqa: BLE001
            logger.warning(f'could not write MIST cache under {root}: {e!r}')
    return reader


def read_isocmd(filename, mist_path=None):
    """
    Read a MIST .iso.cmd grid through the binary cache.

    Returns a `CachedIsoCmd` (cache hit, or just built) or an `IsoCmdReader`
    (cache disabled or unwritable). Both expose `ages`, `age_index`, `isocmds`
    and the header metadata; `CachedIsoCmd` also has `block(i)`.
    """
    filename = os.path.abspath(filename)
    if mist_path is None:
        mist_path = os.path.dirname(os.path.dirname(filename))
    root = isocmd_cache_root(mist_path)
    stamp = _source_stamp(filename)
    return _read_isocmd_keyed(filename, root, stamp['size'], stamp['mtime_ns'])


def isocmd_block(reader, i):
    """Age block `i` from either reader type as a structured ndarray."""
    if hasattr(reader, 'block'):
        return reader.block(i)
    return reader.isocmds[i]
