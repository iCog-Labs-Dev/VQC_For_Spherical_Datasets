"""Quantum feature-encoding operations, kept separate from the ansatz."""
import pennylane as qml

class FeatureEncoder:
    def angle_encode(self, x, wires):
        for feature, wire in zip(x, wires):
            qml.RY(feature, wires=wire)

    def rx_rz_encode(self, x, wires):
        for i, wire in enumerate(wires):
            feature = x[i % len(x)]
            qml.RX(feature, wires=wire)
            qml.RZ(feature, wires=wire)

    def amplitude_encode(self, x, wires):
        qml.AmplitudeEmbedding(x, wires=wires, normalize=True)

    def basis_encode(self, x, wires):
        qml.BasisEmbedding(x, wires=wires)

    def encode(self, x, wires, method="rx_rz"):
        methods = {
            "angle": self.angle_encode,
            "amplitude": self.amplitude_encode,
            "basis": self.basis_encode,
            "rx_rz": self.rx_rz_encode,
        }
        if method not in methods:
            raise ValueError(f"Unsupported encoding method: {method}")
        methods[method](x, wires)

    def intermediate_encode(self, x, wires):
        """The experiment's repeated RX-RZ feature re-uploading step."""
        self.rx_rz_encode(x, wires)
