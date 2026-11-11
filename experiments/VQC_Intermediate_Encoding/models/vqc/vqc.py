"""VQC architecture: repeated encoding + trainable ansatz + measurement."""
import pennylane as qml
from pennylane import numpy as np
from .config import NUM_QUBITS, NUM_LAYERS, SEED
from .encoding import FeatureEncoder
from .ansatz import variational_layer

dev = qml.device("default.qubit", wires=NUM_QUBITS)
encoder = FeatureEncoder()

@qml.qnode(dev)
def vqc_circuit(weights, x):
    wires = range(NUM_QUBITS)
    for layer in range(NUM_LAYERS):
        encoder.intermediate_encode(x, wires=wires)
        qml.Barrier()
        variational_layer(weights[layer], wires=wires)
        qml.Barrier()
    return qml.expval(qml.PauliZ(0))

def init_weights():
    np.random.seed(SEED)
    return np.random.uniform(
        0, 2 * np.pi,
        size=(NUM_LAYERS, NUM_QUBITS, 3),
        requires_grad=True
    )

def draw_circuit(weights, sample_x):
    return qml.draw(vqc_circuit)(weights, sample_x)
