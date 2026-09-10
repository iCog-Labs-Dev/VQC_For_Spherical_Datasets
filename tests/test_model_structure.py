"""
Restriction, proved algebraically rather than by failing to train.

rho = (I + r.sigma)/2 is affine in the Bloch vector r, and every downstream
operation is a linear map on rho, so at one upload

    <O> = alpha + beta . r    exactly, four real parameters,

regardless of depth, width, ancilla count or observable.  These tests assert
that at machine precision -- no dataset, no labels, no optimiser, so no result
here can be blamed on an optimisation failure.
"""
import numpy as np
import pytest

import Analysis as A
from AnsatzLayer import AnsatzLayer
from EmbeddingLayer import EmbeddingLayer
from VQCModel import VQCModel

THETA, PHI = A.sample_sphere(2000, seed=17)


def _probe(n_qubits, n_layers, method="spherical", n_uploads=1, seed=0):
    m = VQCModel(n_qubits, EmbeddingLayer(method),
                 AnsatzLayer("strong", n_layers=n_layers), n_uploads=n_uploads)
    w = np.random.default_rng(seed).uniform(0, 2 * np.pi, m.weight_shape())
    return A.probe_model(m, w, THETA, PHI)


@pytest.mark.parametrize("n_layers", [2, 4, 6, 10])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_one_upload_model_is_affine_in_bloch_vector(n_qubits, n_layers):
    """Depth is inert and the ancilla does not help: ell <= 1 at every setting."""
    v = _probe(n_qubits, n_layers, seed=n_qubits * 100 + n_layers)
    assert A.affine_residual(v, THETA, PHI) < 1e-10


@pytest.mark.parametrize("n_layers", [2, 6])
def test_one_upload_has_no_energy_above_ell_one(n_layers):
    spec = A.harmonic_spectrum(_probe(1, n_layers, seed=n_layers), THETA, PHI, ell_max=4)
    assert spec[2:].sum() < 1e-12


@pytest.mark.parametrize("n", [2, 3])
def test_spin_ladder_is_exactly_degree_limited(n):
    """
    The broadcast encoding gives rho^{tensor n}, a degree-n polynomial in r, so
    the model function lies EXACTLY in ell <= n -- and genuinely reaches ell = n.
    The spin ladder is a harmonic-degree knob.
    """
    v = _probe(n, 3, method="broadcast", seed=n)
    assert A.harmonic_residual(v, THETA, PHI, ell_max=n) < 1e-10
    spec = A.harmonic_spectrum(v, THETA, PHI, ell_max=n + 2)
    assert spec[n] > 1e-4, "the top rung must actually be occupied"


def test_reuploading_is_not_degree_limited():
    """
    Data re-uploading bounds the Fourier spectrum in the ENCODING ANGLES, which
    is not the same as bounding harmonic degree.  At U = 2 the accessible
    functions include sin(theta) cos(2 phi), which is not a polynomial in
    (x, y, z) and therefore has an infinite Legendre expansion.

    So "a target of degree ell becomes learnable at exactly U >= ell" does not
    follow from the frequency argument, and the two ladders are not two knobs
    on one quantity:

      spin ladder     raises harmonic degree, stays rotation-covariant, remains
                      a fixed geodesic kernel, Schoenberg-admissible
      re-uploading    raises coordinate frequency, is NOT rotation-covariant
                      and NOT band-limited on the sphere

    This test pins the distinction so it cannot regress unnoticed.
    """
    v = _probe(1, 2, n_uploads=2, seed=5)
    assert A.harmonic_residual(v, THETA, PHI, ell_max=2) > 1e-3
    assert A.harmonic_residual(v, THETA, PHI, ell_max=1) > 1e-2


def test_affine_ceiling_recovers_a_known_cap():
    """
    A dataset whose label IS a spherical cap must have ceiling ~1, and the
    recovered direction must be the cap's axis (up to sign -- the search tries
    both orientations of the inequality, so the antipodal axis is equivalent).
    Guards the search itself.
    """
    theta, phi = A.sample_sphere(4000, seed=23)
    import Kernels as K
    y = (K.to_cartesian(theta, phi) @ np.array([0.0, 0.0, 1.0]) > 0.3).astype(float)
    res = A.affine_ceiling(theta, phi, y, n_directions=8192)
    assert res["accuracy"] > 0.99
    assert abs(abs(float(res["direction"][2])) - 1.0) < 0.02


def test_affine_ceiling_is_a_grid_limited_lower_bound():
    """
    The ceiling is computed by scanning a finite set of directions, so it is a
    LOWER bound on the true ceiling: the best available direction can sit half
    a grid spacing off the optimum.  At 1024 directions the spacing is roughly
    6 degrees, which costs about a point of accuracy on a sharply defined cap.

    Reported numbers should therefore use a fine grid, and a measured accuracy
    marginally above a coarse-grid ceiling is a resolution artefact, not a
    falsification.  This test documents the convergence rather than hiding it.
    """
    theta, phi = A.sample_sphere(4000, seed=23)
    import Kernels as K
    y = (K.to_cartesian(theta, phi) @ np.array([0.0, 0.0, 1.0]) > 0.3).astype(float)
    coarse = A.affine_ceiling(theta, phi, y, n_directions=256)["accuracy"]
    fine = A.affine_ceiling(theta, phi, y, n_directions=8192)["accuracy"]
    assert coarse <= fine + 1e-12, "a finer grid can only find a better cap"
    assert fine - coarse < 0.05, "but the coarse grid is not wildly wrong either"
