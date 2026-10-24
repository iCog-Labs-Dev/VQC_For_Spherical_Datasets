"""Classical closed-form fidelity kernel on S^2."""

import numpy as np


def exact_kernel(X, Y, n_copies):
    """Compute k_n(x, y) = Tr[rho_n(x) rho_n(y)] = ((1 + x.y) / 2)^n.

    Args:
        X: Array of shape ``(N, 3)``.
        Y: Array of shape ``(M, 3)``.
        n_copies: Number of copies n of the single-qubit encoding.

    Returns:
        Gram matrix of shape ``(N, M)``.
    """
    # Clip guards against x.y drifting outside [-1, 1] by rounding error.
    cos = np.clip(X @ Y.T, -1.0, 1.0)
    return ((1.0 + cos) / 2.0) ** n_copies
