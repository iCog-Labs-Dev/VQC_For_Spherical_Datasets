"""Trainable variational ansatz, independent of classical data encoding."""
import pennylane as qml

def variational_layer(weights_layer, wires):
    """Apply trainable RX/RY/RZ rotations and a ring of CNOT entanglers."""
    for i, wire in enumerate(wires):
        qml.RX(weights_layer[i, 0], wires=wire)
        qml.RY(weights_layer[i, 1], wires=wire)
        qml.RZ(weights_layer[i, 2], wires=wire)

    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
