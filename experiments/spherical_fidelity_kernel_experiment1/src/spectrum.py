"""Theoretical harmonic spectrum of the n-copy fidelity kernel."""

import numpy as np
from scipy.special import gammaln


def eigenvalues(n_copies, max_degree=None):
    """Population eigenvalues lambda_l^(n) = (n!)^2 / ((n-l)! (n+l+1)!).

    Computed through log-gamma for stability at large n. Degrees above n have
    exactly zero eigenvalue (no representational support).

    Args:
        n_copies: Number of copies n.
        max_degree: Largest degree to return; defaults to ``n_copies``.

    Returns:
        Array of shape ``(max_degree + 1,)`` indexed by degree l.
    """
    max_degree = n_copies if max_degree is None else max_degree
    ell = np.arange(max_degree + 1)
    lam = np.zeros(max_degree + 1)
    lo = ell[ell <= n_copies]
    lam[lo] = np.exp(2 * gammaln(n_copies + 1)
                     - gammaln(n_copies - lo + 1)
                     - gammaln(n_copies + lo + 2))
    return lam


def eigenvalue(n_copies, degree):
    """Eigenvalue lambda_l^(n) for a single degree."""
    return float(eigenvalues(n_copies, max(degree, n_copies))[degree])


def shrinkage(n_copies, degree, tau):
    """Population shrinkage factor g_l = lambda_l / (lambda_l + tau)."""
    lam = eigenvalue(n_copies, degree)
    return lam / (lam + tau)


def effective_dimension(n_copies, tau):
    """d_eff(n, tau) = sum_{l=0}^n (2l+1) lambda_l / (lambda_l + tau)."""
    lam = eigenvalues(n_copies)
    return float(np.sum((2 * np.arange(n_copies + 1) + 1) * lam / (lam + tau)))
