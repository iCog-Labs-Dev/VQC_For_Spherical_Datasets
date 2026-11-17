"""Exact and shot-sampled training/cross kernel matrices."""
import numpy as np

from .sphere import validate_unit_vectors


def sampled_training_gram(x, n, shots, backend, rng, ledger):
    """Symmetric sampled Gram matrix: each unordered pair is estimated once and mirrored.

    shots is a scalar or an array with one entry per upper-triangle pair (row-major).
    The diagonal is set to 1 analytically and is not counted.
    """
    x = validate_unit_vectors(x)
    m = len(x)
    iu, ju = np.triu_indices(m, k=1)
    shots = np.broadcast_to(np.asarray(shots), iu.shape)
    est = backend.estimate(x[iu], x[ju], n, shots, rng)
    gram = np.eye(m)
    gram[iu, ju] = est
    gram[ju, iu] = est
    ledger.record("train", est.size, shots.sum())
    return gram


def sampled_cross(xq, xtr, n, shots, backend, rng, ledger, category):
    """Sampled query-to-training matrix of shape (len(xq), len(xtr)); shots per entry, row-major."""
    xq, xtr = validate_unit_vectors(xq), validate_unit_vectors(xtr)
    q, m = len(xq), len(xtr)
    shots = np.broadcast_to(np.asarray(shots), (q * m,))
    est = backend.estimate(np.repeat(xq, m, axis=0), np.tile(xtr, (q, 1)), n, shots, rng)
    ledger.record(category, est.size, shots.sum())
    return est.reshape(q, m)
