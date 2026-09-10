"""The kernel, its Legendre expansion, and its spectrum, against closed forms."""
import numpy as np
import pytest

import Kernels as K


def _sample(n=400, seed=0):
    """Uniform on the sphere: z uniform in [-1, 1], phi uniform in [0, 2pi)."""
    rng = np.random.default_rng(seed)
    return np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)


def test_kernel_endpoints_and_monotonicity():
    g = np.linspace(0, np.pi, 501)
    k = 0.5 * (1 + np.cos(g))
    assert k[0] == pytest.approx(1.0)
    assert k[-1] == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(k) < 0), "faithfulness requires strict monotonicity in gamma"


def test_legendre_expansion_terminates_at_ell_one():
    """k = (1/2) P_0 + (1/2) P_1 -- restriction, read off the coefficients."""
    a = K.legendre_coefficients(lambda t: 0.5 * (1 + t), ell_max=8)
    assert a[0] == pytest.approx(0.5, abs=1e-12)
    assert a[1] == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose(a[2:], 0.0, atol=1e-12)
    assert np.all(a >= -1e-12), "Schoenberg: all coefficients must be non-negative"


def test_funk_hecke_eigenvalues():
    lam = K.funk_hecke_eigenvalues(lambda t: 0.5 * (1 + t), ell_max=4)
    assert lam[0] == pytest.approx(1 / 2, abs=1e-12)
    assert lam[1] == pytest.approx(1 / 6, abs=1e-12)
    np.testing.assert_allclose(lam[2:], 0.0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_spin_ladder_eigenvalues_and_normalisation(n):
    """Closed form against quadrature, plus sum (2l+1) lambda_l = k_n(0) = 1."""
    quad = K.funk_hecke_eigenvalues(lambda t: ((1 + t) / 2) ** n, ell_max=n + 2)
    analytic = K.spin_eigenvalues_analytic(n, ell_max=n + 2)
    np.testing.assert_allclose(quad, analytic, atol=1e-12)
    total = sum((2 * ell + 1) * analytic[ell] for ell in range(n + 1))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_gram_rank_is_four_with_predicted_eigenvalues():
    """
    The empirical Gram matrix has exactly 4 eigenvalues above numerical noise,
    in ratio 3 : 1 : 1 : 1 -- the ell = 1 eigenvalue is threefold degenerate in
    m, which a bare "3 : 1" elides.  Values are n * (1/2, 1/6, 1/6, 1/6) and
    they sum to the trace, n.
    """
    theta, phi = _sample(600, seed=1)
    n = len(theta)
    ev = K.gram_spectrum(theta, phi, K.bloch_kernel)

    assert K.effective_rank(ev) == 4
    np.testing.assert_allclose(ev[0] / n, 0.5, rtol=0.08)
    np.testing.assert_allclose(ev[1:4] / n, 1 / 6, rtol=0.15)
    assert np.sum(ev) == pytest.approx(n, rel=1e-10), "trace = sum k(0) = n"


@pytest.mark.parametrize("n", [1, 2, 3])
def test_spin_gram_rank_grows_as_n_plus_one_squared(n):
    theta, phi = _sample(600, seed=2)
    ev = K.gram_spectrum(theta, phi, K.bloch_kernel_spin, n=n)
    assert K.effective_rank(ev, rtol=1e-9) == (n + 1) ** 2


def test_sincos_is_not_a_function_of_geodesic_distance():
    """
    The falsifying property of the torus kernel: two point pairs at the SAME
    geodesic separation receive different kernel values.  Faithfulness fails
    here, and this is what tears at the poles.
    """
    a = K.sincos_kernel(0.1, 0.0, 0.1, np.pi)
    b = K.sincos_kernel(np.pi / 2, 0.0, np.pi / 2, 0.2)
    ga = K.geodesic_angle(0.1, 0.0, 0.1, np.pi)
    gb = K.geodesic_angle(np.pi / 2, 0.0, np.pi / 2, 0.2)
    assert ga == pytest.approx(gb, abs=1e-3), "pairs chosen to be equidistant"
    assert abs(a - b) > 1e-2, "yet the torus kernel assigns them different values"


def test_amplitude_kernel_identifies_antipodes():
    """cos^2 gamma is distance-only but not injective: it embeds RP^2, not S^2."""
    g = 0.7
    assert K.amplitude_kernel(g, 0.0, 0.0, 0.0) == pytest.approx(
        K.amplitude_kernel(np.pi - g, 0.0, 0.0, 0.0), abs=1e-12)


def test_geodesic_angle_clips_before_arccos():
    """A point against itself must give exactly 0, never NaN."""
    t = np.array([0.3, 1.7, 3.0])
    p = np.array([0.0, 2.2, 5.9])
    g = K.geodesic_angle(t, p, t, p)
    assert not np.any(np.isnan(g))
    np.testing.assert_allclose(g, 0.0, atol=1e-7)
