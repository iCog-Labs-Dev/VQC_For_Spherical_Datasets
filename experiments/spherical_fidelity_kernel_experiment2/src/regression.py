"""Kernel ridge regression: solve (K + lambda I) alpha = y, with validation-only lambda choice."""
import warnings

import numpy as np
from scipy import linalg


def fit_path(gram, targets, lambdas):
    """Solve for every lambda (and every target column at once).

    Returns alphas of shape (L, N, T) and a boolean array (L,) marking solves that were
    ill-conditioned or singular; those fall back to least squares and stay flagged.
    """
    gram = np.asarray(gram, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if not (np.all(np.isfinite(gram)) and np.all(np.isfinite(targets))):
        raise ValueError("non-finite values in Gram matrix or targets")
    eye = np.eye(len(gram))
    alphas = np.empty((len(lambdas),) + targets.shape)
    flagged = np.zeros(len(lambdas), dtype=bool)
    for i, lam in enumerate(lambdas):
        a = gram + lam * eye
        with warnings.catch_warnings():
            warnings.simplefilter("error", linalg.LinAlgWarning)
            try:
                alphas[i] = linalg.solve(a, targets, assume_a="sym")
            except (linalg.LinAlgWarning, linalg.LinAlgError):
                alphas[i] = linalg.lstsq(a, targets)[0]
                flagged[i] = True
    return alphas, flagged


def predict(cross, alpha):
    """Predictions k(x*, X) alpha."""
    return cross @ alpha


def select_lambda(alphas, cross_val, targets_val):
    """Per-target lambda index minimising validation MSE.

    alphas is (L, N, T); returns (indices (T,), validation MSE (L, T)). Non-finite
    validation errors are never selected.
    """
    pred = np.einsum("vn,lnt->lvt", cross_val, alphas)
    val_mse = np.mean((pred - targets_val[None]) ** 2, axis=1)
    safe = np.where(np.isfinite(val_mse), val_mse, np.inf)
    return np.argmin(safe, axis=0), val_mse
