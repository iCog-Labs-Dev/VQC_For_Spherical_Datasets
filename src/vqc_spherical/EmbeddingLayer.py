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
    broadcast  : Applies the SAME (theta, phi) to every wire, giving the
                 spin-n/2 coherent state |q>^{tensor n}.  Its kernel is
                 cos^{2n}(gamma/2): faithful for every n, with the Legendre
                 expansion terminating at ell = n.  This is the
                 geometry-preserving way to raise expressivity -- contrast
                 data re-uploading, which raises coordinate frequency instead.

    Convention (locked -- see tests/test_embedding.py)
    -------------------------------------------------
    The operator convention of the analysis is U = RZ(phi) RY(theta).
    PennyLane applies gates in *circuit* order, the reverse of matrix order,
    so the correct circuit is RY(theta) then RZ(phi).  Do not reorder these.
    """

    SUPPORTED_METHOD = ["spherical", "broadcast"]

    def __init__(self, method="spherical", rotation="Y"):
        self.method = str(method).lower()
        self.rotation = rotation

        if self.method not in EmbeddingLayer.SUPPORTED_METHOD:
            raise ValueError(
                f"Unsupported embedding method: '{method}'. "
                f"Choose from {EmbeddingLayer.SUPPORTED_METHOD}"
            )

    def apply(self, features, wires):
        """Applies the chosen embedding to the quantum tape."""

        if self.method == "broadcast":
            self._apply_spherical_broadcast(features, wires)
        else:
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



    @staticmethod
    def _apply_spherical_broadcast(features, wires):
        """
        Spin-n/2 coherent state encoding: the SAME (theta, phi) on every wire.

        The resulting state is |q(theta, phi)>^{tensor n}, whose fidelity
        kernel is cos^{2n}(gamma/2) = ((1 + cos gamma)/2)^n -- still a function
        of the geodesic angle alone, with Legendre coefficients terminating at
        ell = n and lambda_ell = (n!)^2 / [(n-ell)! (n+ell+1)!].
        """
        theta, phi = features[0], features[1]
        for w in wires:
            qml.RY(theta, wires=w)
            qml.RZ(phi, wires=w)

    def get_required_qubits(self, num_features):
        """Calculate how many qubits are needed for the given feature count."""

        if self.method == "broadcast":
            return 1          # any n >= 1 is valid; n sets ell_max
        if self.method == "spherical":
            return num_features // 2
        

    @staticmethod
    def _maybe_pad(features, target_length):
        """Pad a feature vector with zeros to reach *target_length*."""
        if hasattr(features, "__len__") and len(features) >= target_length:
            return features
        pad_width = target_length - len(features)
        return np.pad(features, (0, pad_width), mode="constant")
