"""Exact spherical fidelity kernel k_n(x, y) = ((1 + x.y) / 2)^n."""
import numpy as np

from .sphere import validate_unit_vectors

BOUND_TOL = 1e-12


def exact_kernel(x, y, n):
    """Kernel matrix of shape (len(x), len(y)) for copy count n."""
    if int(n) != n or n < 1:
        raise ValueError("copy count n must be a positive integer")
    x, y = validate_unit_vectors(x), validate_unit_vectors(y)
    k = ((1.0 + x @ y.T) / 2.0) ** int(n)
    if k.min() < -BOUND_TOL or k.max() > 1.0 + BOUND_TOL:
        raise ValueError("kernel values outside [0, 1]")
    return k


def exact_kernel_pairs(x, y, n):
    """Elementwise kernel between matching rows of x and y (shape (m,))."""
    x, y = validate_unit_vectors(x), validate_unit_vectors(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    p = ((1.0 + np.sum(x * y, axis=1)) / 2.0) ** int(n)
    return np.clip(p, 0.0, 1.0)  # rounding drift only
