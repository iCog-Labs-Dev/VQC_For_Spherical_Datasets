"""Finite-shot estimator of the fidelity kernel.

Model: each shot is a Bernoulli(p) bit with p = k_n(x, y); the estimate is the mean of S
bits, which is unbiased with variance p(1-p)/S. The sum of S i.i.d. Bernoulli draws is
Binomial(S, p), so the binomial is sampled directly. This is a shot-noise simulation, not
a circuit result; circuit-based estimates come from quantum_kernel.PennyLaneBackend.
"""
import numpy as np

from .kernel import exact_kernel_pairs

PROB_TOL = 1e-9


def estimate_from_probability(p, shots, rng):
    """Mean of `shots` Bernoulli(p) draws; p and shots broadcast elementwise."""
    p = np.asarray(p, dtype=float)
    shots = np.asarray(shots)
    if np.any(shots < 1) or np.any(shots != np.floor(shots)):
        raise ValueError("shots must be positive integers")
    if np.any(p < -PROB_TOL) or np.any(p > 1 + PROB_TOL):
        raise ValueError("probability outside [0, 1]")
    p = np.clip(p, 0.0, 1.0)  # rounding drift only
    return rng.binomial(shots.astype(np.int64), p) / shots


class BernoulliBackend:
    """Reference backend: Bernoulli shot noise on the exact kernel probability."""
    name = "bernoulli_reference"

    def estimate(self, a, b, n, shots, rng):
        """Estimates for matching rows of a and b; shots is a scalar or per-pair array."""
        return estimate_from_probability(exact_kernel_pairs(a, b, n), shots, rng)


def get_backend(name):
    if name == "bernoulli_reference":
        return BernoulliBackend()
    if name == "pennylane_simulator":
        from .quantum_kernel import PennyLaneBackend
        return PennyLaneBackend()
    raise ValueError(f"unknown measurement backend: {name}")
