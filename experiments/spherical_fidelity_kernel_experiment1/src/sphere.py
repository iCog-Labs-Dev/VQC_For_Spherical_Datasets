"""Uniform sampling on the unit sphere S^2."""

import numpy as np


def sample_sphere(n_points, rng):
    """Draw points uniformly on S^2.

    Normalising i.i.d. 3D Gaussian vectors gives the uniform distribution on
    the sphere because the Gaussian is rotation invariant.

    Args:
        n_points: Number of points to draw.
        rng: A ``numpy.random.Generator``.

    Returns:
        Array of shape ``(n_points, 3)`` with unit-norm rows.
    """
    v = rng.standard_normal((n_points, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)
