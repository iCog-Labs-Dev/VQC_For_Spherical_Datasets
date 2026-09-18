import pennylane as qml
from pennylane import numpy as np

from .AnsatzLayer import AnsatzLayer
from .EmbeddingLayer import EmbeddingLayer



class VQCModel:
    """
    Central variational quantum circuit model for pure spherical dataset.

    Assembles a geometry-preserving embedding, a variational ansatz, and a
    measurement into a single QNode.  
    Provides ``forward()`` for the Trainer and quantum dropout for regularisation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    embedding : EmbeddingLayer
        Configured embedding strategy (spherical).
    ansatz : AnsatzLayer
        Configured ansatz strategy.
    measurement : str
        'expval'     -- expectation of PauliZ on qubit 0 (scalar output).
        'probs'      -- probability vector over all computational basis states.
        'expval_all' -- expectation of PauliZ on every qubit.
    device_name : str
        PennyLane device backend.
    dropout_rate : float
        Probability of zeroing each rotation weight during training (0 = off).
    """

    SUPPORTED_MEASUREMENTS = ("expval", "probs", "expval_all")

    def __init__(
        self,
        n_qubits,
        embedding=None,
        ansatz=None,
        measurement="expval",
        device_name="default.qubit",
        dropout_rate=0.0,
        n_uploads=1,
    ):
        self.n_qubits = int(n_qubits)
        self.n_uploads = int(n_uploads)
        self.embedding = embedding or EmbeddingLayer(method="spherical")
        self.ansatz = ansatz or AnsatzLayer()
        self.dropout_rate = float(dropout_rate)
        self.training = True

        if self.n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        if self.n_uploads < 1:
            raise ValueError("n_uploads must be a positive integer")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if measurement not in self.SUPPORTED_MEASUREMENTS:
            raise ValueError(
                f"Unsupported measurement: '{measurement}'. "
                f"Choose from {self.SUPPORTED_MEASUREMENTS}"
            )
        self.measurement = measurement

        self.device = qml.device(device_name, wires=self.n_qubits)
        self._qnode = qml.QNode(self._circuit, self.device)

    def _apply_dropout(self, weights):
        """Zero out rotation angles with probability ``dropout_rate``."""
        if self.training and self.dropout_rate > 0:
            mask = np.random.binomial(
                1, 1 - self.dropout_rate, size=weights.shape
            )
            return weights * mask
        return weights

    def _circuit(self, features, weights):
        """Encode features, apply the ansatz, and return the measurement.

        With n_uploads = U > 1 the encoding is interleaved with trainable
        layers -- E(x) A(w_0) E(x) A(w_1) ... -- which is data re-uploading.
        The weights then have shape (U, L, n_wires, 3) and are kept as ONE
        array and sliced; a Python list of arrays breaks autograd tracing.

        Re-uploading raises the accessible Fourier frequency in the encoding
        ANGLES.  That is not the same as raising harmonic degree on the sphere,
        and the difference is measured in ExperimentLadders.py."""
        wires = range(self.n_qubits)
        if self.n_uploads == 1:
            self.embedding.apply(features, wires)
            self.ansatz.apply(weights, wires)
        else:
            for u in range(self.n_uploads):
                self.embedding.apply(features, wires)
                self.ansatz.apply(weights[u], wires)
        return self._measure(wires)

    def weight_shape(self):
        """Weight shape, with the upload axis prepended when n_uploads > 1."""
        base = tuple(self.ansatz.get_weight_shape(self.n_qubits))
        return base if self.n_uploads == 1 else (self.n_uploads,) + base

    def n_params(self):
        """Total number of trainable parameters."""
        return int(np.prod(self.weight_shape()))

    def _measure(self, wires):
        if self.measurement == "expval":
            return qml.expval(qml.PauliZ(wires[0]))
        if self.measurement == "probs":
            return qml.probs(wires=wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    def forward(self, x, weights):
        """Run one input sample through the circuit and return the result."""
        w = self._apply_dropout(weights)
        return self._qnode(x, w)

    def train(self):
        """Enable training mode (dropout active)."""
        self.training = True

    def eval(self):
        """Enable evaluation mode (dropout disabled)."""
        self.training = False
