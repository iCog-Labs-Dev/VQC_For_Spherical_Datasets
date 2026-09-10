"""
Data preparation and cleaning for spherical datasets. we generate sphere_moons datasets on the surface of the unit sphere, and 
adpoted the water-earth classfication datatset, as another real datasets to experiments with.
"""

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
    n_half = n_samples // 2

    theta_0 = rng.normal(latitude_center, noise_std * np.pi, n_half)
    phi_0 = rng.uniform(0, np.pi, n_half)

    theta_1 = rng.normal(latitude_center + separation, noise_std * np.pi, n_half)
    phi_1 = rng.uniform(np.pi, 2 * np.pi, n_half)

    theta = np.concatenate([theta_0, theta_1])
    phi = np.concatenate([phi_0, phi_1])
    labels = np.array([0.0] * n_half + [1.0] * n_half)

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


if __name__ == "__main__":
    print(water_earth_dataset())


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
