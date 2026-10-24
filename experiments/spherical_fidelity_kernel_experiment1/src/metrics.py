"""Agreement, error, amplitude and correlation metrics."""

import numpy as np


def kernel_agreement(K_a, K_b):
    """Max absolute error and RMSE between two kernel matrices.
    Whether the circuit and exact formula agree numerically.
    """
    d = K_a - K_b
    return {"max_abs_err": float(np.max(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2)))}


def matrix_checks(K):
    """Symmetry error, unit-diagonal error and smallest eigenvalue of a Gram matrix."""
    return {
        "symmetry_err": float(np.max(np.abs(K - K.T))),
        "diag_err": float(np.max(np.abs(np.diag(K) - 1.0))),
        "min_eigenvalue": float(np.linalg.eigvalsh((K + K.T) / 2).min()),
    }


def prediction_agreement(pred_a, pred_b):
    """Max absolute difference and RMSE between two prediction vectors."""
    d = pred_a - pred_b
    return {"max_abs_diff": float(np.max(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2)))}


def mse(pred, target):
    """Mean squared error against the known target."""
    return float(np.mean((pred - target) ** 2))


def target_amplitude(pred, target):
    """Projection of ``pred`` onto ``target``: <pred, target> / <target, target>.

    How much of the original target remains in the prediction.

    Equals the fraction of the target harmonic that survives in the prediction
    (1 means full recovery; compare with the shrinkage factor g_l).
    """
    return float(pred @ target / (target @ target))


def correlation(pred, target):
    """Pearson correlation; NaN when either vector is constant (e.g. l = 0).
    How closely predictions track target variation.
    """
    if np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])
