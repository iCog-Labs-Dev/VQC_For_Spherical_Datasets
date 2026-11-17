"""Real spherical harmonics and target functions.

Convention: orthonormal on S^2 (integral of Y^2 over the sphere is 1).
  m = 0: N_l0 P_l^0(cos t)
  m > 0: sqrt(2) N_lm P_l^m(cos t) cos(m phi)
  m < 0: sqrt(2) N_l|m| P_l^|m|(cos t) sin(|m| phi)
with N_lm = sqrt((2l+1)/(4 pi) (l-|m|)!/(l+|m|)!).
"""
from dataclasses import dataclass
from math import factorial, pi, sqrt

import numpy as np
from scipy.special import lpmv

from .sphere import validate_unit_vectors


def real_sph_harm(l, m, x):
    """Evaluate Y_lm at unit vectors x of shape (k, 3)."""
    if l < 0 or abs(m) > l:
        raise ValueError("require l >= 0 and |m| <= l")
    x = validate_unit_vectors(x)
    am = abs(m)
    norm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - am) / factorial(l + am))
    legendre = lpmv(am, l, x[:, 2])
    phi = np.arctan2(x[:, 1], x[:, 0])
    if m == 0:
        return norm * legendre
    trig = np.cos(am * phi) if m > 0 else np.sin(am * phi)
    return sqrt(2.0) * norm * legendre * trig


def harmonic_basis(x, max_degree):
    """Matrix of all Y_lm up to max_degree, plus the degree of each column."""
    cols, degrees = [], []
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            cols.append(real_sph_harm(l, m, x))
            degrees.append(l)
    return np.column_stack(cols), np.array(degrees)


@dataclass(frozen=True)
class Target:
    """Linear combination of real harmonics: sum_i coef_i * Y_{l_i, m_i}."""
    name: str
    terms: tuple  # ((l, m, coef), ...)

    def __call__(self, x):
        return sum(c * real_sph_harm(l, m, x) for l, m, c in self.terms)


def build_target(spec):
    """Create a Target from {'name': str, 'terms': [(l, m, coef), ...]}."""
    terms = tuple((int(l), int(m), float(c)) for l, m, c in spec["terms"])
    return Target(spec["name"], terms)
