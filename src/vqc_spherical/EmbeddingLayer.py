import pennylane as qml
import math
import numpy as np



class EmbeddingLayer:

    """
    manifold-preserving encoding of classical data into quantum states.

    Supported method
    -----------------
    spherical  : Maps paired (theta, phi) coordinates onto the Bloch sphere
                 using RY(theta) then RZ(phi) per qubit.  Preserves the
                 wrap-around manifold so it aviod seam and torsion.
    """

    SUPPORTED_METHOD = ["spherical"]

    def __init__(self, method="spherical", rotation="Y"):
        self.rotation = rotation

        if self.method not in EmbeddingLayer.SUPPORTED_METHOD:
            raise ValueError(
                f"Unsupported embedding method: '{method}'. "
                f"Choose from {EmbeddingLayer.SUPPORTED_METHOD}"
            )

    def apply(self, features, wires):
        """Applies the chosen embedding to the quantum tape."""

        self._apply_spherical(features, wires)
   

    @staticmethod
    def _apply_spherical(features, wires):
        """
        Expects *features* to contain pairs: [theta0, phi0, theta1, phi1, ...].
        Each consecutive (theta, phi) pair is mapped to one qubit via RY then RZ,
        placing the data point on the corresponding location of the Bloch sphere.
        """
        n_pairs = len(features) // 2
        for i in range(n_pairs):
            theta = features[2 * i]
            phi = features[2 * i + 1]
            qml.RY(theta, wires=wires[i])
            qml.RZ(phi, wires=wires[i])



    def get_required_qubits(self, num_features):
        """Calculate how many qubits are needed for the given feature count."""

        if self.method == "spherical":
            return num_features // 2
        

    @staticmethod
    def _maybe_pad(features, target_length):
        """Pad a feature vector with zeros to reach *target_length*."""
        if hasattr(features, "__len__") and len(features) >= target_length:
            return features
        pad_width = target_length - len(features)
        return np.pad(features, (0, pad_width), mode="constant")
