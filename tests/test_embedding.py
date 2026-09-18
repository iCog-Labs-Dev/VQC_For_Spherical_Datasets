"""
Convention lock and the topological pole check.

The trap this file exists to catch: RZ(phi) = diag(e^{-i phi/2}, e^{+i phi/2})
contributes an overall phase e^{-i phi/2}, so the circuit statevector does NOT
equal [cos(t/2), e^{i p} sin(t/2)] amplitude by amplitude.  It equals it up to
a global phase.  A test comparing amplitudes directly FAILS on a correct
implementation.  Compare |<analytic|circuit>| = 1 instead.

The operator convention of the analysis is U = RZ(phi) RY(theta).  PennyLane
applies gates in circuit order, the reverse of matrix order, so the correct
circuit really is RY first then RZ -- which is what EmbeddingLayer does.  Do
not "fix" that ordering; this file is what stops someone trying.
"""
import numpy as np
import pennylane as qml
import pytest

from vqc_spherical import Kernels as K
from vqc_spherical.EmbeddingLayer import EmbeddingLayer

POINTS = [(0.7, 1.2), (2.3, 5.1), (np.pi / 2, 0.0), (1.0, 2 * np.pi - 0.01)]


def circuit_state(theta, phi, n_qubits=1, method="spherical"):
    dev = qml.device("default.qubit", wires=n_qubits)
    emb = EmbeddingLayer(method)

    @qml.qnode(dev)
    def qc():
        emb.apply([theta, phi], range(n_qubits))
        return qml.state()

    return np.asarray(qc())


@pytest.mark.parametrize("theta,phi", POINTS)
def test_statevector_matches_analytic_up_to_global_phase(theta, phi):
    analytic = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    overlap = np.abs(np.vdot(analytic, circuit_state(theta, phi)))
    assert overlap == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("theta,phi", POINTS)
def test_bloch_vector_is_the_data_point(theta, phi):
    """The encoded Bloch vector must equal (sin t cos p, sin t sin p, cos t)."""
    psi = circuit_state(theta, phi)
    rho = np.outer(psi, psi.conj())
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    r = np.real([np.trace(rho @ P) for P in (X, Y, Z)])
    np.testing.assert_allclose(r, K.to_cartesian(theta, phi), atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, 1.0, 3.0, 6.0])
def test_pole_identifies_all_longitudes(phi):
    """
    At theta = 0 the phase multiplies zero, so |q(0, phi)> = |0> for every phi.
    The encoding enforces the identification of all longitudes at the pole --
    the defining topological feature of S^2 -- with no special-case code.  This
    is exactly what the sin-cos representation cannot do.
    """
    psi = circuit_state(0.0, phi)
    assert np.abs(np.vdot(np.array([1.0, 0.0]), psi)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("theta,phi", POINTS)
def test_fidelity_matches_bloch_kernel(theta, phi):
    """|<q(p)|q(p')>|^2 must equal (1 + cos gamma)/2 -- the claim in one line."""
    t2, p2 = 1.9, 0.4
    fidelity = np.abs(np.vdot(circuit_state(theta, phi), circuit_state(t2, p2))) ** 2
    assert fidelity == pytest.approx(K.bloch_kernel(theta, phi, t2, p2), abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("theta,phi", POINTS[:2])
def test_broadcast_fidelity_matches_spin_kernel(n, theta, phi):
    """
    The n-qubit broadcast state has fidelity cos^{2n}(gamma/2) -- still a
    function of geodesic distance alone, so the spin ladder stays faithful at
    every rung while its harmonic degree grows.
    """
    t2, p2 = 1.9, 0.4
    f = np.abs(np.vdot(circuit_state(theta, phi, n, "broadcast"),
                       circuit_state(t2, p2, n, "broadcast"))) ** 2
    assert f == pytest.approx(K.bloch_kernel_spin(theta, phi, t2, p2, n=n), abs=1e-12)


def test_broadcast_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported embedding method"):
        EmbeddingLayer("angle")
