"""Prediction and harmonic-recovery metrics."""
import numpy as np

VAR_EPS = 1e-12


def _check(*arrays):
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite values in metric inputs")


def mse(y_true, y_pred):
    _check(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def normalized_mse(y_true, y_pred):
    """MSE divided by the population variance of y_true; NaN if that variance is below 1e-12."""
    var = float(np.var(y_true))
    return mse(y_true, y_pred) / var if var >= VAR_EPS else float("nan")


def correlation(y_true, y_pred):
    """Pearson correlation; NaN if either input is (nearly) constant."""
    _check(y_true, y_pred)
    if np.var(y_true) < VAR_EPS or np.var(y_pred) < VAR_EPS:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def harmonic_recovery(f_true, f_pred, basis, degrees):
    """Coefficient error of the prediction on the test points.

    Coefficients use Monte Carlo quadrature over the uniform test points,
    c = (4 pi / M) Y^T f. The error vector is c(f_pred) - c(f_true); the relative
    error divides by ||c(f_true)||. Per-degree errors are the norms within each degree.
    """
    _check(f_true, f_pred, basis)
    scale = 4.0 * np.pi / len(f_true)
    delta = scale * basis.T @ (f_pred - f_true)
    ref = scale * basis.T @ f_true
    out = {"coef_err": float(np.linalg.norm(delta)),
           "coef_err_rel": float(np.linalg.norm(delta) / np.linalg.norm(ref))}
    for l in range(int(degrees.max()) + 1):
        out[f"coef_err_l{l}"] = float(np.linalg.norm(delta[degrees == l]))
    return out
