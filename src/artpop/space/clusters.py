"""
Young stars in star-forming complexes: a mixture of projected Plummer models
(2026-10-15; revised the same day for physical fidelity).

ArtPop places every star of a composite population on the galaxy's one Sersic
profile, so young stars scatter like old ones, while in real star-forming dwarfs
the young light sits in compact knots. Here an SSP's stars (and its smooth
light) are a **mixture**: a weight ``f_diffuse`` stays on the galaxy's Sersic
profile and the rest is shared by N complexes, each a projected Plummer sphere.

The pieces, all pure functions of their arguments and a random state:

* **How many complexes, and how massive** -- `sample_cmf_to_budget`: masses from
  a power-law mass function dN/dM ~ M^-slope (slope 2: Lada & Lada 2003;
  Portegies Zwart et al. 2010) drawn until the SSP's clustered mass is used up
  (stop-nearest, Haas & Anders 2010), so N follows from the formed mass, hence
  from the galaxy's mass and SFR together.
* **How big** -- `complex_scale_pc`: a Plummer scale that grows with mass and
  with age (expansion): a^2 = [a0 (M/M_ref)^beta]^2 + (sigma_exp t)^2.
* **Where** -- `ring_disk_offsets`: centres in the galaxy's disk plane with a
  surface density that is a ring, Sigma(R) ~ exp(-(R - R_p)^2 / 2 w^2): non-zero
  at the centre, highest at R_p (inside r_eff), truncated at R_max. `disk_to_sky`
  projects disk-plane offsets onto the sky (squash by 1 - ellip, rotate by
  theta, the convention of ALVISS's Sersic sampler).

The complex *masses* are not drawn by the source: they set each complex's
ionizing photons, which set the gas state (`artpop.nebular.gas_state`) before
the isochrones are built, so the caller draws them with the population and
hands them in (`ClusterSpec`). Everything spatial -- centres, which star goes
where -- is drawn by `artpop.source.ClusteredSersicSP` from its random state.
"""
import numpy as np

__all__ = ['cluster_weights', 'sample_cmf_to_budget', 'cmf_mass_fraction',
           'complex_scale_pc', 'ring_surface_density', 'ring_disk_offsets',
           'disk_to_sky', 'cluster_centres', 'plummer_offsets', 'plummer_stamp',
           'paint_stamp', 'ClusterSpec', 'KMS_PC_PER_MYR']

# 1 km/s in pc/Myr
KMS_PC_PER_MYR = 1.0227121650537077


def cluster_weights(n, rng, slope=2.0, mass_range=100.0):
    """
    Mass weights of ``n`` clusters drawn from a power-law cluster mass function
    ``dN/dM ~ M^-slope`` over ``mass_range`` (max/min), normalized to sum to 1.
    For a fixed N (tests, explicit specs); `sample_cmf_to_budget` is the
    mass-budget form.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f'need at least one cluster, got {n}')
    m = _cmf_draw(n, rng, float(slope), 1.0, float(mass_range))
    return m / m.sum()


def _cmf_draw(n, rng, slope, lo, hi):
    """``n`` masses from dN/dM ~ M^-slope on [lo, hi] (inverse transform)."""
    u = rng.uniform(size=int(n))
    if hi <= lo:
        return np.full(int(n), float(lo))
    if abs(slope - 1.0) < 1e-12:
        return lo * (hi / lo) ** u
    g = 1.0 - slope
    return (lo ** g + u * (hi ** g - lo ** g)) ** (1.0 / g)


def _moment(k, slope, lo, hi):
    """int_lo^hi M^(k - slope) dM."""
    p = k - slope + 1.0
    if abs(p) < 1e-12:
        return np.log(hi / lo)
    return (hi ** p - lo ** p) / p


def cmf_mass_fraction(m_lo, m_hi, slope, m_min, m_max):
    """Fraction of the CMF's mass (on [m_min, m_max]) in [m_lo, m_hi]."""
    tot = _moment(1, slope, m_min, m_max)
    lo, hi = max(m_lo, m_min), min(m_hi, m_max)
    return 0.0 if hi <= lo else float(_moment(1, slope, lo, hi) / tot)


def sample_cmf_to_budget(budget, rng, slope=2.0, m_min=1.0e3, m_max=None, n_cap=300):
    """
    Complex masses drawn from dN/dM ~ M^-slope on [m_min, min(m_max, budget)]
    until their sum reaches ``budget`` (M_sun); the last draw is kept if that
    lands nearer the budget than leaving it out ("stop-nearest", Haas & Anders
    2010). The masses are then rescaled to sum to ``budget`` exactly, so the
    light shares are mass-conserving.

    ``n_cap`` bounds N for compute: when the expected N (budget / mean mass)
    exceeds it, ``m_min`` is raised until the expectation equals the cap, and
    the CMF's mass between the configured and the raised ``m_min`` is reported
    as ``f_below`` -- the caller moves it to the diffuse share (in a real galaxy
    those small complexes are sub-PSF anyway). A budget below ``m_min`` is one
    complex holding all of it.

    Returns ``(masses, info)`` with ``info = {m_min_used, f_below, n_expected}``.
    """
    budget = float(budget)
    if not budget > 0:
        raise ValueError(f'budget must be > 0, got {budget}')
    slope = float(slope)
    lo = float(m_min)
    hi = budget if m_max is None else min(float(m_max), budget)
    if hi <= lo:
        return np.array([budget]), dict(m_min_used=lo, f_below=0.0, n_expected=1.0)

    def n_exp(lo_):
        mean = _moment(1, slope, lo_, hi) / _moment(0, slope, lo_, hi)
        return budget / mean

    lo_used = lo
    if n_cap is not None and n_exp(lo) > n_cap:
        a, b = np.log(lo), np.log(hi)
        for _ in range(80):
            mid = 0.5 * (a + b)
            if n_exp(np.exp(mid)) > n_cap:
                a = mid
            else:
                b = mid
        lo_used = float(np.exp(b))
    f_below = cmf_mass_fraction(lo, lo_used, slope, lo, hi)

    target = budget * (1.0 - f_below)
    masses, total = [], 0.0
    batch = max(8, int(1.5 * n_exp(lo_used)) + 1)
    while True:
        for m in _cmf_draw(batch, rng, slope, lo_used, hi):
            if total + m >= target:
                if abs(total + m - target) < abs(total - target) or not masses:
                    masses.append(m)
                    total += m
                m_arr = np.asarray(masses, dtype=float)
                return (m_arr * (target / m_arr.sum()),
                        dict(m_min_used=lo_used, f_below=f_below, n_expected=n_exp(lo_used)))
            masses.append(m)
            total += m


def complex_scale_pc(mass, age_myr, a0_pc=10.0, beta=0.3, m_ref=1.0e4,
                     sigma_exp_kms=3.0):
    """
    Plummer scale (pc) of a complex of ``mass`` M_sun at ``age_myr``:
    ``sqrt([a0 (M/m_ref)^beta]^2 + (sigma_exp t)^2)`` -- a size-mass relation at
    birth plus expansion of the unbound members at ``sigma_exp_kms``.
    """
    m = np.asarray(mass, dtype=float)
    a_birth = float(a0_pc) * (m / float(m_ref)) ** float(beta)
    a_exp = float(sigma_exp_kms) * KMS_PC_PER_MYR * float(age_myr)
    return np.sqrt(a_birth ** 2 + a_exp ** 2)


def ring_surface_density(r, r_peak, width, width_out=None):
    """Unnormalized Sigma(R) of the complex centres: a ring, exp(-(R - R_p)^2 / 2 w^2)
    with w = ``width`` inside R_p and ``width_out`` (default: the same) outside,
    so the outer edge can fall faster than the inner side (2026-10-15, user)."""
    r = np.asarray(r, dtype=float)
    w_out = float(width if width_out is None else width_out)
    w = np.where(r < float(r_peak), float(width), w_out)
    return np.exp(-0.5 * ((r - float(r_peak)) / w) ** 2)


def ring_disk_offsets(n, r_eff_px, rng, r_peak_reff=0.4, width_reff=0.3,
                      max_reff=2.0, width_out_reff=None):
    """
    ``n`` centres as disk-plane offsets (pixels): angle uniform, radius with pdf
    ~ R Sigma(R) on [0, max_reff r_eff] (`ring_surface_density`, inner width
    ``width_reff``, outer ``width_out_reff``), by rejection.
    """
    n = int(n)
    r_p = float(r_peak_reff) * float(r_eff_px)
    w = float(width_reff) * float(r_eff_px)
    w_out = w if width_out_reff is None else float(width_out_reff) * float(r_eff_px)
    r_max = float(max_reff) * float(r_eff_px)
    if n == 0:
        return np.zeros((0, 2))
    if not (r_max > 0 and w > 0 and w_out > 0):
        return np.zeros((n, 2))
    grid = np.linspace(0.0, r_max, 2001)
    pmax = 1.05 * float(np.max(grid * ring_surface_density(grid, r_p, w, w_out)))
    r = np.empty(0)
    while r.size < n:
        cand = rng.uniform(0.0, r_max, size=4 * (n - r.size) + 16)
        keep = (rng.uniform(0.0, pmax, size=cand.size)
                < cand * ring_surface_density(cand, r_p, w, w_out))
        r = np.concatenate([r, cand[keep]])
    r = r[:n]
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def disk_to_sky(xy_disk, theta_rad, ellip):
    """Disk-plane offsets -> sky offsets (y squashed by 1 - ellip, rotated by theta)."""
    xy = np.atleast_2d(np.asarray(xy_disk, dtype=float))
    x, y = xy[:, 0], xy[:, 1] * (1.0 - float(ellip))
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    return np.column_stack([ct * x - st * y, st * x + ct * y])


def cluster_centres(n, r_eff_px, theta_rad, ellip, rng, r_peak_reff=0.4,
                    width_reff=0.3, max_reff=2.0, width_out_reff=None):
    """``n`` complex centres as sky offsets (pixels) from the galaxy centre:
    `ring_disk_offsets` projected with `disk_to_sky`."""
    return disk_to_sky(ring_disk_offsets(n, r_eff_px, rng, r_peak_reff, width_reff,
                                         max_reff, width_out_reff), theta_rad, ellip)


# projected Plummer: Sigma(R) = (1 + R^2/a^2)^-2 / (pi a^2), enclosed R^2/(R^2 + a^2)
_PLUMMER_EXTENT = 30.0
_PLUMMER_U_MAX = 1.0 - 1.0 / (1.0 + _PLUMMER_EXTENT ** 2)   # truncate at R = 30 a


def plummer_offsets(num, a_px, rng):
    """``num`` (dx, dy) offsets (pixels) from a projected Plummer of scale ``a_px``,
    truncated at 30 a (0.1 % of the light)."""
    u = rng.uniform(0.0, _PLUMMER_U_MAX, size=int(num))
    r = float(a_px) * np.sqrt(u / (1.0 - u))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=int(num))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def plummer_stamp(a_px, frac_x=0.0, frac_y=0.0, extent=_PLUMMER_EXTENT, oversample=None,
                  max_half=None):
    """
    Stamp of a projected Plummer of scale ``a_px`` centred at a sub-pixel offset
    (``frac_x``, ``frac_y``) in [-0.5, 0.5) from the stamp's central pixel,
    pixel-integrated on ``oversample`` sub-pixels (default: 5 below a = 3 px,
    3 below 10 px, else 1) and truncated at ``extent`` a. Unit-sum, unless
    ``max_half`` cuts the stamp (a frame smaller than the light): then it is
    normalized analytically, so the light beyond the cut is lost, as light off
    the frame is. Returns ``(stamp, half)``; a scale under a tenth of a pixel
    is a delta.
    """
    a = float(a_px)
    if a < 0.1:
        return np.ones((1, 1)), 0
    if oversample is None:
        oversample = 5 if a < 3 else (3 if a < 10 else 1)
    half = int(np.ceil(extent * a + 1))
    cut = max_half is not None and half > int(max_half)
    if cut:
        half = int(max_half)
    n = 2 * half + 1
    sub = (np.arange(n * oversample) + 0.5) / oversample - (half + 0.5)
    yy, xx = np.meshgrid(sub - frac_y, sub - frac_x, indexing='ij')
    prof = (1.0 + (xx ** 2 + yy ** 2) / a ** 2) ** -2
    prof[(xx ** 2 + yy ** 2) > (extent * a) ** 2] = 0.0
    stamp = prof.reshape(n, oversample, n, oversample).sum(axis=(1, 3))
    if cut:
        # surface density (1 + R^2/a^2)^-2 / (pi a^2) per unit area, sub-pixel
        # area 1/oversample^2, renormalized to the 30 a truncation
        norm = np.pi * a ** 2 * oversample ** 2 * _PLUMMER_U_MAX
        return stamp / norm, half
    return stamp / stamp.sum(), half


def paint_stamp(image, stamp, half, x, y, weight):
    """Add ``weight * stamp`` to ``image`` centred on pixel (round(x), round(y)),
    clipped at the edges (light off the frame is lost, as stars off it are)."""
    ny, nx = image.shape
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    x0, x1 = ix - half, ix + half + 1
    y0, y1 = iy - half, iy + half + 1
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = stamp.shape[1] - max(0, x1 - nx)
    sy1 = stamp.shape[0] - max(0, y1 - ny)
    if sx1 <= sx0 or sy1 <= sy0:
        return image
    image[max(y0, 0):min(y1, ny), max(x0, 0):min(x1, nx)] += weight * stamp[sy0:sy1, sx0:sx1]
    return image


class ClusterSpec:
    """
    How one SSP is spread: ``weights`` (one per complex, summing to 1) share
    ``1 - f_diffuse`` of its stars and smooth light, ``f_diffuse`` stays on the
    galaxy's Sersic profile. ``a_pc`` is the Plummer scale, one number or one
    per complex, times a lognormal scatter of ``a_sigma_dex`` drawn by the
    source. ``masses`` (M_sun, one per complex) and ``age_myr`` are optional
    records; the source uses them to link complexes of different ages to the
    same star-forming sites. ``label`` names the SSP in the composite
    (``ssp_labels``, 1-based).
    """

    def __init__(self, label, weights, a_pc, f_diffuse=0.0, a_sigma_dex=0.0,
                 name=None, masses=None, age_myr=None):
        self.label = int(label)
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.ndim != 1 or self.weights.size < 1 or np.any(self.weights < 0):
            raise ValueError(f'cluster weights must be a non-empty 1-d array >= 0, got {weights}')
        s = self.weights.sum()
        if not abs(s - 1.0) < 1e-6:
            raise ValueError(f'cluster weights must sum to 1, got {s}')
        a = np.asarray(a_pc, dtype=float)
        if a.ndim == 0:
            a = np.full(self.weights.size, float(a))
        if a.shape != self.weights.shape:
            raise ValueError(f'a_pc must be a number or one per cluster ({self.weights.size}), '
                             f'got shape {a.shape}')
        if not np.all(a > 0):
            raise ValueError(f'a_pc must be > 0, got {a_pc}')
        self.a_pc = a
        self.f_diffuse = float(f_diffuse)
        if not 0.0 <= self.f_diffuse <= 1.0:
            raise ValueError(f'f_diffuse must be in [0, 1], got {f_diffuse}')
        self.a_sigma_dex = float(a_sigma_dex)
        self.name = name
        self.masses = None if masses is None else np.asarray(masses, dtype=float)
        if self.masses is not None and self.masses.shape != self.weights.shape:
            raise ValueError('masses must be one per cluster')
        self.age_myr = None if age_myr is None else float(age_myr)

    @property
    def n_clusters(self):
        return int(self.weights.size)

    def component_probs(self):
        """[p_diffuse, p_cluster_1, ...] for assigning a star."""
        return np.concatenate([[self.f_diffuse], (1.0 - self.f_diffuse) * self.weights])

    def as_dict(self):
        return dict(label=self.label, name=self.name, weights=self.weights.tolist(),
                    a_pc=self.a_pc.tolist(), a_sigma_dex=self.a_sigma_dex,
                    f_diffuse=self.f_diffuse, age_myr=self.age_myr,
                    masses=None if self.masses is None else self.masses.tolist())

    @classmethod
    def coerce(cls, obj):
        if obj is None or isinstance(obj, cls):
            return obj
        return cls(**dict(obj))
