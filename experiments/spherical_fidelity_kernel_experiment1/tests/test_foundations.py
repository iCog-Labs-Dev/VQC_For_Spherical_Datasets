"""Mathematical and circuit tests for the experiment's building blocks."""

import numpy as np
import pytest
from scipy.special import eval_legendre

from experiments.experiment_1 import Config, run_grid, run_pilot, summarize
from src import metrics, spectrum
from src.kernel import exact_kernel
try:
    import pennylane  # noqa: F401
    from src.quantum_kernel import quantum_kernel_matrix, quantum_kernel_pair
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
from src.regression import fit_krr, predict_krr
from src.sphere import sample_sphere
from src.targets import zonal_target, random_degree_target, real_spherical_harmonic


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_sphere_sampling_sanity(rng):
    X = sample_sphere(100_000, rng)
    assert X.shape == (100_000, 3)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    assert np.all(np.abs(X.mean(axis=0)) < 0.01)
    assert np.allclose((X ** 2).mean(axis=0), 1 / 3, atol=0.01)


def test_zonal_targets(rng):
    X = sample_sphere(200_000, rng)
    z = X[:, 2]
    assert np.allclose(zonal_target(0, X), 1.0)
    assert np.allclose(zonal_target(1, X), np.sqrt(3) * z)
    assert np.allclose(zonal_target(2, X), np.sqrt(5) * (3 * z ** 2 - 1) / 2)
    for ell in (0, 1, 2, 4, 8):
        assert np.sqrt(np.mean(zonal_target(ell, X) ** 2)) == pytest.approx(1.0, abs=0.03)


def test_general_real_harmonic_normalization_and_determinism(rng):
    X = sample_sphere(100_000, rng)
    for ell in (0, 1, 2, 4):
        for m in range(ell + 1):
            for component in (("cos", "sin") if m else ("cos",)):
                y = real_spherical_harmonic(ell, m, X, component)
                assert np.mean(y ** 2) == pytest.approx(1.0, abs=0.04)
        a = random_degree_target(ell, X, seed=42)
        b = random_degree_target(ell, X, seed=42)
        assert np.array_equal(a, b)
        assert np.mean(a ** 2) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_classical_kernel_properties(rng, n):
    X = sample_sphere(60, rng)
    K = exact_kernel(X, X, n)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_n1_kernel_formula_and_rank(rng):
    X = sample_sphere(50, rng)
    K = exact_kernel(X, X, 1)
    assert np.allclose(K, (1 + X @ X.T) / 2)
    assert np.linalg.matrix_rank(K, tol=1e-9) <= 4


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane is not installed")
def test_circuit_matches_exact_kernel_on_pairs(rng, n):
    A, B = sample_sphere(20, rng), sample_sphere(20, rng)
    for x, y in zip(A, B):
        expected = exact_kernel(x[None], y[None], n)[0, 0]
        assert quantum_kernel_pair(x, y, n) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane is not installed")
def test_circuit_gram_matrix(rng, n):
    X, Y = sample_sphere(15, rng), sample_sphere(9, rng)
    assert np.allclose(quantum_kernel_matrix(X, Y, n), exact_kernel(X, Y, n), atol=1e-12)

    K = quantum_kernel_matrix(X, X, n)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.allclose(np.diag(K), 1.0, atol=1e-12)
    assert np.allclose(quantum_kernel_matrix(X, X, n, symmetric=True), K, atol=1e-12)


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane is not installed")
def test_circuit_handles_poles():
    X = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    assert np.allclose(quantum_kernel_matrix(X, X, 3), exact_kernel(X, X, 3), atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_spectrum_identities(n):
    lam = spectrum.eigenvalues(n)
    ell = np.arange(n + 1)
    assert np.sum((2 * ell + 1) * lam) == pytest.approx(1.0)
    t = np.linspace(-1, 1, 21)
    series = sum((2 * l + 1) * lam[l] * eval_legendre(l, t) for l in ell)
    assert np.allclose(series, ((1 + t) / 2) ** n)


def test_spectrum_support_and_values():
    assert np.allclose(spectrum.eigenvalues(1), [1 / 2, 1 / 6])
    assert np.all(spectrum.eigenvalues(2, max_degree=6)[3:] == 0.0)
    assert spectrum.shrinkage(2, 5, 1e-3) == 0.0
    assert spectrum.effective_dimension(4, 1e-12) == pytest.approx(25, rel=1e-6)
    assert spectrum.effective_dimension(4, 1e3) < 1e-2


def test_krr_solves_regularised_system(rng):
    X = sample_sphere(30, rng)
    K, y, tau = exact_kernel(X, X, 3), zonal_target(1, X), 1e-2
    alpha = fit_krr(K, y, tau)
    assert np.allclose((K + 30 * tau * np.eye(30)) @ alpha, y)
    assert predict_krr(K, alpha).shape == (30,)


def test_representational_support_controls_recovery(rng):
    """H1: degree above n is unrecoverable, degree at most n is recovered."""
    X, X_test = sample_sphere(128, rng), sample_sphere(500, rng)
    K, Kt = exact_kernel(X, X, 1), exact_kernel(X_test, X, 1)
    for ell, recoverable in ((1, True), (2, False)):
        pred = predict_krr(Kt, fit_krr(K, zonal_target(ell, X), 1e-3))
        err = metrics.mse(pred, zonal_target(ell, X_test))
        assert (err < 0.05) if recoverable else (err > 0.9)


def test_metrics():
    t = np.array([1.0, -1.0, 2.0, 0.5])
    assert metrics.mse(t, t) == 0.0
    assert metrics.target_amplitude(0.5 * t, t) == pytest.approx(0.5)
    assert metrics.correlation(0.5 * t, t) == pytest.approx(1.0)
    assert np.isnan(metrics.correlation(np.ones(4), t))
    K = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert metrics.matrix_checks(K)["min_eigenvalue"] == pytest.approx(0.5)
    assert metrics.kernel_agreement(K, K)["max_abs_err"] == 0.0


def test_pilot_and_grid_smoke():
    cfg = Config(copies=(1, 2), degrees=(0, 1, 2), train_sizes=(16,),
                 taus=(1e-3, 1e-2), target_modes=("zonal", "random_harmonic"),
                 seeds=(0, 1), target_seeds=(0, 1), n_test=50, n_validation=40,
                 pilot_pairs=5, pilot_gram_size=8)
    if HAS_PENNYLANE:
        assert all(r["passed"] for r in run_pilot(cfg))

    df = run_grid(cfg)
    # 2 seeds x 2 copies x 3 degrees x (1 zonal + 2 random realizations) x 2 taus
    assert len(df) == 2 * 2 * 3 * 3 * 2
    assert {"mse_krr", "mse_krr_selected", "mse_harmonic_matched_span",
            "mse_harmonic_higher_capacity", "selected_tau_krr"}.issubset(df.columns)
    assert np.isfinite(df["mse_krr"]).all()
    assert np.isfinite(df["mse_harmonic_matched_span"]).all()
    assert np.isfinite(df["mse_harmonic_higher_capacity"]).all()
    assert len(summarize(df)) == 2 * 3 * 3 * 2
