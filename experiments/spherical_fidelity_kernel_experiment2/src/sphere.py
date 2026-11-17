"""Uniform sampling and validation of unit vectors on S^2."""
import numpy as np


def make_rng(*seed_parts):
    """Deterministic generator from one or more integers."""
    return np.random.default_rng(np.random.SeedSequence([int(s) for s in seed_parts]))


def validate_unit_vectors(x, tol=1e-9):
    """Return x as a float array of shape (m, 3) after checking finiteness and unit norm."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"expected shape (m, 3), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite entries in unit vectors")
    if np.max(np.abs(np.linalg.norm(x, axis=1) - 1.0)) > tol:
        raise ValueError("vectors are not unit norm within tolerance")
    return x


def sample_sphere(m, rng):
    """Draw m points uniformly by surface area: z ~ U[-1, 1], phi ~ U[0, 2*pi)."""
    z = rng.uniform(-1.0, 1.0, size=m)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=m)
    r = np.sqrt(1.0 - z**2)
    return validate_unit_vectors(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
