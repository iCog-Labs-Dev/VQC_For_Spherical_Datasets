"""Spherical-harmonic targets on S^2: zonal and general real harmonics."""

import numpy as np
from scipy.special import eval_legendre, lpmv, gammaln


def zonal_target(degree, X):
    """Evaluate normalized zonal harmonic sqrt(2l+1) P_l(x_3) (m=0)."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    X = np.atleast_2d(X)
    return np.sqrt(2 * degree + 1) * eval_legendre(degree, np.clip(X[:, 2], -1, 1))


def real_spherical_harmonic(degree, order, X, component="cos"):
    """Evaluate a real spherical harmonic using polar angle theta and azimuth phi.

    ``component`` is ``'cos'`` or ``'sin'`` for |m|>0; for m=0 it is ignored.
    The normalization is orthonormal under surface-area measure dOmega, so
    sqrt(4*pi) times this function has unit RMS under the uniform sphere measure.
    """
    if degree < 0 or abs(order) > degree:
        raise ValueError("require degree >= 0 and |order| <= degree")
    if component not in ("cos", "sin"):
        raise ValueError("component must be 'cos' or 'sin'")
    X = np.atleast_2d(X)
    z = np.clip(X[:, 2], -1.0, 1.0)
    phi = np.arctan2(X[:, 1], X[:, 0])
    m = abs(order)
    log_norm = 0.5 * (np.log((2 * degree + 1) / (4 * np.pi))
                      + gammaln(degree - m + 1) - gammaln(degree + m + 1))
    norm = np.exp(log_norm)
    associated = lpmv(m, degree, z)
    if m == 0:
        y = norm * associated
    elif component == "cos":
        y = np.sqrt(2.0) * norm * associated * np.cos(m * phi)
    else:
        y = np.sqrt(2.0) * norm * associated * np.sin(m * phi)
    return np.sqrt(4 * np.pi) * y


def random_degree_target(degree, X, seed=0):
    """A deterministic random unit-RMS-in-expectation combination of all m modes.

    Coefficients are fixed by ``seed`` and degree, so training and test points
    evaluate the same target function. The target is not normalized on each sample.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    rng = np.random.default_rng([int(seed), int(degree), 8723])
    coeff = rng.normal(size=2 * degree + 1)
    coeff /= np.linalg.norm(coeff)
    result = coeff[0] * real_spherical_harmonic(degree, 0, X)
    j = 1
    for m in range(1, degree + 1):
        result += coeff[j] * real_spherical_harmonic(degree, m, X, "cos")
        result += coeff[j + 1] * real_spherical_harmonic(degree, m, X, "sin")
        j += 2
    return result
