import pennylane as qml
from pennylane import numpy as np

try:  # works both as a package import and as a flat script directory
    from .AnsatzLayer import AnsatzLayer
    from .EmbeddingLayer import EmbeddingLayer
except ImportError:  # pragma: no cover - flat sys.path
    from AnsatzLayer import AnsatzLayer
    from EmbeddingLayer import EmbeddingLayer



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
    ):
        
        self.n_qubits = n_qubits
        self.embedding = embedding or EmbeddingLayer(method="spherical")
        self.ansatz = ansatz or AnsatzLayer()
        self.dropout_rate = dropout_rate
        self.training = True

        if measurement not in VQCModel.SUPPORTED_MEASUREMENTS:
            raise ValueError(
                f"Unsupported measurement: '{measurement}'. "
                f"Choose from {VQCModel.SUPPORTED_MEASUREMENTS}"
            )
        self.measurement = measurement

        self.device = qml.device(device_name, wires=n_qubits)
        self._qnode = qml.QNode(self._circuit, self.device)


    def _apply_dropout(self, weights):

        """Zero-out rotation angles with probability ``dropout_rate``.
        In addition I am thinking if this is the best regularization method for 
        quantum circuits, or if we should consider other methods 
        like adding noise to the weights or using a different kind of regularization. 
        This is something we can explore in future iterations of the model.
        """
        if self.training and self.dropout_rate > 0:
            mask = np.random.binomial(
                1, 1 - self.dropout_rate, size=weights.shape
            )
            return weights * mask
        return weights

    def _circuit(self, features, weights):
        
        """Here we define the quantum circuit that will be executed on the quantum device.
        The circuit consists of an embedding layer that encodes the input features into quantum states,
        followed by a variational ansatz that applies parameterized quantum gates to the qubits.
        Finally, we measure the qubits according to the specified measurement type."""
        wires = range(self.n_qubits)
        self.embedding.apply(features, wires)
        self.ansatz.apply(weights, wires)
        return self._measure(wires)

    def _measure(self, wires):
        if self.measurement == "expval":
            return qml.expval(qml.PauliZ(wires[0]))
        elif self.measurement == "probs":
            return qml.probs(wires=wires)
        elif self.measurement == "expval_all":
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
