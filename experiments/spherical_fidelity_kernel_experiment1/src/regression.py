"""Kernel ridge regression shared by the classical and quantum branches."""

import numpy as np
from scipy.linalg import solve


def fit_krr(K_train, y, tau):
    """Solve (K + N * tau * I) alpha = y.

    Args:
        K_train: Training Gram matrix of shape ``(N, N)``.
        y: Training targets of shape ``(N,)``.
        tau: Regularisation strength (> 0); the ridge term is scaled by N.

    Returns:
        Dual coefficients ``alpha`` of shape ``(N,)``.
    """
    N = len(y)
    return solve(K_train + N * tau * np.eye(N), y, assume_a="pos")


def predict_krr(K_test_train, alpha):
    """Predict with k(X_test, X_train) @ alpha."""
    return K_test_train @ alpha
