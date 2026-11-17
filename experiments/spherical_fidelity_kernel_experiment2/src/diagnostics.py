"""Estimator and matrix diagnostics."""
import numpy as np


def symmetry_error(k):
    return float(np.max(np.abs(k - k.T)))


def spectrum_summary(k):
    """Extreme eigenvalues and 2-norm condition number of a symmetric matrix."""
    w = np.linalg.eigvalsh((k + k.T) / 2.0)
    smallest = np.min(np.abs(w))
    cond = float(np.max(np.abs(w)) / smallest) if smallest > 0 else float("inf")
    return {"min_eig": float(w.min()), "max_eig": float(w.max()), "cond": cond}


def estimator_error(estimate, exact, shots):
    """Empirical error statistics of sampled entries against exact values.

    theory_var is the mean binomial variance p(1-p)/S, to compare with the empirical var.
    """
    err = np.asarray(estimate) - np.asarray(exact)
    shots = np.asarray(shots, dtype=float)
    return {"mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "var": float(np.var(err)),
            "theory_var": float(np.mean(exact * (1.0 - exact) / shots))}
