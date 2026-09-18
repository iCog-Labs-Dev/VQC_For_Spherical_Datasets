import pennylane as qml


class AnsatzLayer:
    """
    Trainable variational layers for the quantum circuit.

    Default is 'strong' (StronglyEntanglingLayers) which provides full SU(2)
    coverage per qubit per layer -- critical for exploiting the entire Bloch
    sphere surface when working with non-Euclidean data.

    Supported methods: 'basic', 'strong', 'random'
    """

    SUPPORTED_METHODS = ("basic", "strong", "random")

    def __init__(self, method="strong", n_layers=3, rotation=None):
        self.method = str(method).lower()
        self.n_layers = int(n_layers)

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported ansatz method: '{method}'. "
                f"Choose from {self.SUPPORTED_METHODS}"
            )
        if self.n_layers < 1:
            raise ValueError("n_layers must be a positive integer")

        self.rotation = rotation
        if self.rotation is None and self.method == "basic":
            self.rotation = qml.RX

    def apply(self, weights, wires):
        """Applies the chosen parameterized layers to the quantum tape."""

        if self.method == "basic":
            kwargs = {"weights": weights, "wires": wires}
            if self.rotation is not None:
                kwargs["rotation"] = self.rotation
            qml.BasicEntanglerLayers(**kwargs)

        if self.method == "strong":
            qml.StronglyEntanglingLayers(weights=weights, wires=wires)

        if self.method == "random":
            kwargs = {"weights": weights, "wires": wires}
            if self.rotation is not None:
                kwargs["rotations"] = [self.rotation] * len(wires)
            qml.RandomLayers(**kwargs)

    def get_weight_shape(self, n_qubits):
        """
        Dynamically calculates the exact tensor shape required for the weights
        so you never get a matrix mismatch error during initialization.
        """
        
        if self.method == "basic":
            return qml.BasicEntanglerLayers.shape(
                n_layers=self.n_layers, n_wires=n_qubits
            )
        if self.method == "strong":
            return qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_layers, n_wires=n_qubits
            )
        return qml.RandomLayers.shape(
            n_layers=self.n_layers, n_rotations=n_qubits
        )
