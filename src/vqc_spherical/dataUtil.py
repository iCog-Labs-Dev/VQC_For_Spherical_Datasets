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
