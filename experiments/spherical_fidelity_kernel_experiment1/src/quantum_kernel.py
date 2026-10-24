"""PennyLane overlap-circuit implementation of the spherical fidelity kernel.

The state for x is |psi(x)> = cos(t/2)|0> + e^{i p} sin(t/2)|1> with
t = arccos(x_3), p = atan2(x_2, x_1), prepared on each of n qubits. Applying
the inverse preparation for y afterwards makes the all-zero probability equal
|<psi(y)|psi(x)>|^(2n) = ((1 + x.y) / 2)^n.

The default is the ideal analytic simulator; optional finite-shot QNodes estimate
the same overlap probability from measurement samples.
"""

import numpy as np
import pennylane as qml

_QNODES = {}


def sphere_angles(X):
    """Return polar and azimuthal angles ``(theta, phi)`` of points on S^2."""
    X = np.atleast_2d(X)
    theta = np.arccos(np.clip(X[:, 2], -1.0, 1.0))
    phi = np.arctan2(X[:, 1], X[:, 0])
    return theta, phi


def _prepare(theta, phi, wires):
    """Prepare |psi> on every wire (RZ.RY|0> equals |psi> up to global phase)."""
    for w in wires:
        qml.RY(theta, wires=w)
        qml.RZ(phi, wires=w)


def _overlap_qnode(n_copies, shots=None, seed=None):
    """Build/cache an overlap QNode; ``shots=None`` means analytic probabilities."""
    key = (n_copies, shots, seed)
    if key not in _QNODES:
        dev = qml.device("default.qubit", wires=n_copies, shots=shots, seed=seed)
        wires = list(range(n_copies))

        @qml.qnode(dev, diff_method=None)
        def circuit(theta_x, phi_x, theta_y, phi_y):
            _prepare(theta_x, phi_x, wires)
            qml.adjoint(_prepare)(theta_y, phi_y, wires)
            return qml.probs(wires=wires)

        _QNODES[key] = circuit
    return _QNODES[key]


def overlap_probabilities(angles_x, angles_y, n_copies, shots=None, seed=None):
    """All-zero probabilities for a batch of (x, y) pairs.

    Uses parameter broadcasting, so a whole batch is one simulator call.

    Args:
        angles_x: ``(theta, phi)`` arrays of shape ``(B,)`` for the x states.
        angles_y: ``(theta, phi)`` arrays of shape ``(B,)`` for the y states.
        n_copies: Number of qubits n.

    Returns:
        Array of shape ``(B,)`` with the probability of measuring |0...0>.
    """
    circuit = _overlap_qnode(n_copies, shots=shots, seed=seed)
    probs = circuit(angles_x[0], angles_x[1], angles_y[0], angles_y[1])
    return np.asarray(probs)[..., 0]


def quantum_kernel_pair(x, y, n_copies):
    """Kernel value for a single pair of points via the circuit."""
    return float(overlap_probabilities(sphere_angles(x), sphere_angles(y), n_copies)[0])


def quantum_kernel_matrix(X, Y, n_copies, symmetric=False, batch_size=2048,
                          shots=None, seed=None):
    """Build the Gram matrix between ``X`` and ``Y`` from the circuit.

    Args:
        X: Array of shape ``(N, 3)``.
        Y: Array of shape ``(M, 3)``.
        n_copies: Number of qubits n.
        symmetric: If True, ``Y`` must equal ``X``; only the upper triangle is
            simulated and mirrored, roughly halving the cost.
        shots: ``None`` for exact probabilities, otherwise finite measurements.
        seed: Simulator seed for reproducible finite-shot measurements.
        batch_size: Number of pairs per simulator call (bounds memory).

    Returns:
        Gram matrix of shape ``(N, M)``.
    """
    tx, px = sphere_angles(X)
    ty, py = sphere_angles(Y)
    if symmetric:
        rows, cols = np.triu_indices(len(X))
    else:
        rows, cols = (a.ravel() for a in np.meshgrid(
            np.arange(len(X)), np.arange(len(Y)), indexing="ij"))

    values = np.empty(len(rows))
    for start in range(0, len(rows), batch_size):
        sl = slice(start, start + batch_size)
        values[sl] = overlap_probabilities(
            (tx[rows[sl]], px[rows[sl]]), (ty[cols[sl]], py[cols[sl]]),
            n_copies, shots=shots, seed=seed)

    K = np.empty((len(X), len(Y)))
    K[rows, cols] = values
    if symmetric:
        K[cols, rows] = values
    return K
