"""Optional PennyLane backend: shot-sampled overlap circuit on a statevector simulator.

Encoding: a unit vector with polar angle theta and azimuth phi is prepared on one qubit as
RZ(phi) RY(theta) |0>, whose Bloch vector is x. The n-copy state uses n qubits, each
prepared the same way, so |<psi(x)^n | psi(y)^n>|^2 = ((1 + x.y)/2)^n.
Overlap circuit: encode x on every qubit, apply the adjoint encoding of y, and measure all
qubits in the computational basis. A shot counts as 1 when the outcome is all zeros, whose
probability is k_n(x, y); the estimate is the all-zeros frequency over the shots.
Measurement noise comes only from finite shots (noiseless simulator).
"""
import numpy as np


def _angles(v):
    return np.arccos(np.clip(v[:, 2], -1.0, 1.0)), np.arctan2(v[:, 1], v[:, 0])


class PennyLaneBackend:
    name = "pennylane_simulator"
    chunk = 2000  # pairs per broadcast circuit execution

    def __init__(self):
        import pennylane as qml
        self._qml = qml

    def _encode(self, theta, phi, wires):
        qml = self._qml
        for w in wires:
            qml.RY(theta, wires=w)
            qml.RZ(phi, wires=w)

    def _circuit(self, n):
        qml, wires = self._qml, list(range(n))

        def circuit(theta_x, phi_x, theta_y, phi_y):
            self._encode(theta_x, phi_x, wires)
            qml.adjoint(self._encode)(theta_y, phi_y, wires)
            return qml.probs(wires=wires)

        return circuit

    def estimate(self, a, b, n, shots, rng):
        """Estimates for matching rows of a and b; shots is a scalar or per-pair array."""
        qml = self._qml
        shots = np.broadcast_to(np.asarray(shots), (len(a),))
        dev = qml.device("default.qubit", wires=n, seed=int(rng.integers(2**32)))
        circuit = self._circuit(n)
        tx, px = _angles(a)
        ty, py = _angles(b)
        out = np.empty(len(a))
        for s in np.unique(shots):
            qnode = qml.set_shots(qml.QNode(circuit, dev), shots=int(s))
            idx = np.flatnonzero(shots == s)
            for start in range(0, len(idx), self.chunk):
                sel = idx[start:start + self.chunk]
                probs = np.atleast_2d(qnode(tx[sel], px[sel], ty[sel], py[sel]))
                out[sel] = probs[:, 0]
        return out
