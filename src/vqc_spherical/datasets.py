"""Spherical dataset generators, coordinate transforms, and target metadata."""

import numpy as np

def make_sphere_moons(n_samples=300, noise_std=0.08, seed=42,
                      latitude_center=np.pi / 3, separation=np.pi / 3):
    """
    Two crescent-shaped classes on the surface of a unit sphere.
    Both theta (latitude) and phi (longitude) determine the class.
    Radius is constant -- the geometry is purely spherical.

    Class 0: centred at latitude 60 deg, eastern hemisphere (phi in [0, pi]).
    Class 1: centred at latitude 120 deg, western hemisphere (phi in [pi, 2*pi]).

    Parameters
    ----------
    n_samples : int
    noise_std : float
        Controls Gaussian spread of the latitude band (scaled by pi).
    seed : int
    latitude_center : float
        Polar angle of class 0's band centre.  The default pi/3 reproduces the
        original generator exactly.
    separation : float
        Angular gap; class 1 sits at latitude_center + separation.

    DEPRECATED FOR REPRESENTATION WORK -- RETAINED DELIBERATELY
    ----------------------------------------------------------
    Class 0 draws phi in (0, pi) and class 1 in (pi, 2pi), so LONGITUDE ALONE
    SEPARATES THE CLASSES.  Measured on 6000 points:

        phi alone      1.000
        theta alone    0.983
        affine ceiling 1.000

    Every model reaches 100%, which is the saturated-benchmark failure mode:
    nothing about inductive bias can be learned from a task everyone solves by
    reading one number.

    It is not fixed in place, on purpose.  It is the dataset behind the
    existing sphere_moons figure, so changing what the name means would make
    that figure unreproducible; and as a documented negative example it is
    worth more than it would be deleted.  It is listed in LEAKY_GENERATORS and
    require_clean() refuses it.

    Use make_latitude_bands (axis-aligned control) or make_tilted_bands (needs
    both coordinates) instead.

    Returns
    -------
    theta  : ndarray (n_samples,)   -- polar angle in [0, pi].
    phi    : ndarray (n_samples,)   -- azimuthal angle in [0, 2*pi).
    labels : ndarray (n_samples,)   -- 0 or 1.
    """
    rng = np.random.default_rng(seed)
    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0

    theta_0 = rng.normal(latitude_center, noise_std * np.pi, n_class_0)
    phi_0 = rng.uniform(0, np.pi, n_class_0)

    theta_1 = rng.normal(
        latitude_center + separation, noise_std * np.pi, n_class_1
    )
    phi_1 = rng.uniform(np.pi, 2 * np.pi, n_class_1)

    theta = np.concatenate([theta_0, theta_1])
    phi = np.concatenate([phi_0, phi_1])
    labels = np.concatenate([np.zeros(n_class_0), np.ones(n_class_1)])

    theta = np.clip(theta, 0, np.pi)
    phi = phi % (2 * np.pi)

    return theta, phi, labels


def water_earth_dataset():
    """
    Fetch the water/land classification dataset from Google Earth Engine.

    `earthengine-api` is imported lazily and Initialize() is called here rather
    than at module scope, so importing this module never contacts a network
    service and never requires Earth Engine credentials.
    """
    import ee

    # Remember that you need to authenticate with Google Earth Engine before running this function

    ee.Initialize()
    dataset = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
    return dataset.size().getInfo()


def sphere_to_cartesian(theta, phi):
    """Convert spherical (theta, phi) to Cartesian (x, y, z) on the unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


# ======================================================================
# Falsification targets and corrected generators
# ======================================================================

def _uniform_sphere(n, rng):
    """
    Uniform on S^2: z uniform in [-1, 1].

    Sampling theta uniformly instead would concentrate points at the poles and
    quietly change every ceiling computed from the result.
    """
    z = rng.uniform(-1, 1, n)
    return np.arccos(z), rng.uniform(0, 2 * np.pi, n), z


def make_latitude_bands(n_samples=400, noise_std=0.08, seed=42,
                        latitude_center=np.pi / 3, separation=np.pi / 3):
    """
    The corrected degree-1 task: two latitude bands with FULL longitude
    coverage for both classes, so nothing about the label is readable from phi.

    This is the axis-aligned control -- theta alone suffices by design, which
    is the intent, not a fault.  For a task that genuinely needs both
    coordinates use make_tilted_bands.
    """
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2
    theta_0 = rng.normal(latitude_center, noise_std * np.pi, n_half)
    theta_1 = rng.normal(latitude_center + separation, noise_std * np.pi,
                         n_samples - n_half)
    theta = np.clip(np.concatenate([theta_0, theta_1]), 0, np.pi)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    labels = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)])
    return theta, phi, labels


def make_quadrupole(n_samples=400, noise_std=0.05, seed=42):
    """
    Label = sign(3 cos^2(theta) - 1): positive near both poles, negative around
    the equator.  The boundary is the nodal set of the pure degree-2 harmonic.

    A single-upload model provably cannot express it: f = alpha + beta.r has no
    z^2 term for any (alpha, beta), so its decision region is ONE spherical cap
    and this target needs two.

    The class balance is 42/58, not 50/50 -- the label is positive iff
    |z| > 1/sqrt(3) ~ 0.577.  Report the majority baseline beside the accuracy,
    and prefer balanced accuracy.
    """
    rng = np.random.default_rng(seed)
    theta, phi, z = _uniform_sphere(n_samples, rng)
    signal = (3 * z ** 2 - 1) + rng.normal(0, noise_std, n_samples)
    return theta, phi, (signal > 0).astype(float)


def make_sectoral(n_samples=400, noise_std=0.05, seed=42):
    """
    Label = sign(cos 2 phi): four alternating longitude sectors, the m = +-2
    companion to the quadrupole.  Exactly balanced, which makes it a cleaner
    falsification instrument.

    Do NOT assume its affine ceiling is 0.5: a cap of angular radius 45 degrees
    centred on the equator sits entirely inside one sector and already scores
    about 0.65.  Compute the ceiling; never guess it.
    """
    rng = np.random.default_rng(seed)
    theta, phi, _ = _uniform_sphere(n_samples, rng)
    signal = np.cos(2 * phi) + rng.normal(0, noise_std, n_samples)
    return theta, phi, (signal > 0).astype(float)


def make_banded_target(n_samples=400, degree=4, seed=42):
    """Alternating latitude bands -- a degree-`degree` target, for the ladder."""
    rng = np.random.default_rng(seed)
    theta, phi, _ = _uniform_sphere(n_samples, rng)
    return theta, phi, (np.cos(degree * theta) > 0).astype(float)


def make_hyperbolic(n_samples=400, noise_std=0.05, seed=42):
    """
    Label = sign(x * z): positive where x and z share a sign.  This is the real
    product structure -- BOTH coordinate marginals sit at chance, so no single
    threshold on theta or phi carries any information at all.

    x*z = sin(theta) cos(theta) cos(phi) is proportional to the degree-2
    harmonic Y_21, so the price of that property is exactly what the theory
    predicts: a single-upload model cannot express it at any depth.  It is a
    falsification target, not a task the circuit is expected to solve.
    """
    rng = np.random.default_rng(seed)
    theta, phi, _ = _uniform_sphere(n_samples, rng)
    x = np.sin(theta) * np.cos(phi)
    z = np.cos(theta)
    return theta, phi, ((x * z + rng.normal(0, noise_std, n_samples)) > 0).astype(float)


# ======================================================================
# Rigid rotations -- the faithfulness instrument
# ======================================================================

def cartesian_to_sphere(xyz):
    """Inverse of sphere_to_cartesian; theta in [0, pi], phi in [0, 2pi)."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.arccos(np.clip(z, -1, 1)), np.mod(np.arctan2(y, x), 2 * np.pi)


def rotation_y(angle):
    """Rotation about the y axis -- tilts the cut away from the latitudes."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(angle):
    """Rotation about the z axis -- rotates the frame in longitude."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation(seed=0):
    """Haar-random element of SO(3), via QR with a determinant fix."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotate_dataset(theta, phi, R):
    """
    Conjugate a dataset by R in SO(3) and re-express it in (theta, phi).

    Every geodesic distance is preserved exactly, so the task -- its difficulty,
    its class balance, its affine ceiling -- is unchanged and only the
    coordinate description differs.  Any variation in accuracy across
    placements is therefore purely representational, which is what makes this
    a measurement of faithfulness rather than of difficulty.
    """
    return cartesian_to_sphere(sphere_to_cartesian(theta, phi) @ np.asarray(R).T)


def make_tilted_bands(n_samples=400, noise_std=0.07, seed=42,
                      separation=np.pi / 6, tilt=np.pi / 4, twist=np.pi / 3):
    """
    Two latitude bands, rigidly rotated so the separating circle is not a line
    of latitude.  The boundary becomes

        sin(tilt) sin(theta) cos(phi - twist) + cos(tilt) cos(theta) = c

    so neither coordinate alone determines the label, while the task itself is
    unchanged -- rotation preserves every geodesic distance.

    tilt = 0 reproduces make_latitude_bands (theta alone suffices).
    tilt = pi/2 puts the cut through the poles and PHI alone suffices: the leak
    moves rather than disappearing.  Intermediate tilts near pi/4 are where
    both coordinates are genuinely required.  Verify with
    Analysis.coordinate_leakage rather than assuming.

    Note what this means for the model: the task is still exactly degree 1, so
    a single-upload circuit can solve it.  A task needing both coordinates in
    EVERY frame is necessarily degree >= 2 and provably out of reach -- that is
    make_hyperbolic, and the trade is the restriction result stated as a
    property of datasets.
    """
    theta, phi, labels = make_latitude_bands(
        n_samples=n_samples, noise_std=noise_std, seed=seed,
        latitude_center=np.pi / 2 - separation / 2, separation=separation)
    theta, phi = rotate_dataset(theta, phi, rotation_z(twist) @ rotation_y(tilt))
    return theta, phi, labels


# ======================================================================
# Registry
# ======================================================================
#
# Every generator is reachable through TARGETS with the SAME signature,
# make(n_samples, seed, **kw), so experiment code can loop over datasets
# without special-casing any of them.

def _banded(n_samples=400, seed=42, degree=4, **kw):
    return make_banded_target(n_samples=n_samples, degree=degree, seed=seed)


TARGETS = {
    "sphere_moons": make_sphere_moons,
    "latitude_bands": make_latitude_bands,
    "tilted_bands": make_tilted_bands,
    "quadrupole": make_quadrupole,
    "sectoral": make_sectoral,
    "banded_4": _banded,
    "hyperbolic": make_hyperbolic,
}

# role:
#   "baseline"      degree <= 1, solvable, axis-aligned -- the control
#   "primary"       degree <= 1, solvable, needs both coordinates
#   "falsification" degree > 1, provably out of reach at one upload
#   "deprecated"    retained for reproducibility only
TARGET_META = {
    "sphere_moons":   dict(degree=1, role="deprecated",
                           note="leaks the label into phi; retained for the existing figure"),
    "latitude_bands": dict(degree=1, role="baseline",
                           note="theta alone suffices, by design -- the control"),
    "tilted_bands":   dict(degree=1, role="primary",
                           note="needs both coordinates; use tilt near pi/4"),
    "quadrupole":     dict(degree=2, role="falsification",
                           note="two polar caps; affine ceiling 0.786"),
    "sectoral":       dict(degree=2, role="falsification",
                           note="balanced; affine ceiling 0.669"),
    "banded_4":       dict(degree=4, role="falsification",
                           note="alternating latitude bands"),
    "hyperbolic":     dict(degree=2, role="falsification",
                           note="sign(x*z); both coordinate marginals at chance"),
}

# name -> (which coordinate leaks, best accuracy from it alone)
LEAKY_GENERATORS = {
    "sphere_moons": ("phi", 1.000),
}

# A generator leaks when one coordinate alone gets within this margin of the
# best any degree-1 model could do: the second coordinate then buys nothing.
LEAKAGE_MARGIN = 0.02


def make_target(name, n_samples=400, seed=42, **kw):
    """Uniform entry point.  Raises on an unknown name rather than KeyError."""
    if name not in TARGETS:
        raise ValueError(f"unknown target '{name}'; choose from {sorted(TARGETS)}")
    return TARGETS[name](n_samples=n_samples, seed=seed, **kw)


def require_clean(name):
    """
    Guard for any experiment that compares representations.  Call it before
    generating data; it raises on a dataset that cannot support the claim.
    """
    if name in LEAKY_GENERATORS:
        what, acc = LEAKY_GENERATORS[name]
        raise ValueError(
            f"'{name}' leaks the label into {what} ({acc:.3f} from that "
            f"coordinate alone).  A task solvable by reading one number cannot "
            f"measure representation quality.  Use 'tilted_bands' (needs both "
            f"coordinates) or 'latitude_bands' (axis-aligned control).")
    return name
