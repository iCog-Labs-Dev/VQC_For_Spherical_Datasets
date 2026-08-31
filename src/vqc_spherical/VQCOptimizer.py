import pennylane as qml
from pennylane import numpy as np


class Trainer:
    """
    Classical optimisation loop for a VQCModel.

    Parameters
    ----------
    model : VQCModel
        Must expose ``forward(x, weights)``, ``ansatz``, ``n_qubits``,
        ``train()`` and ``eval()``.
    optimizer_type : str
        One of 'adam', 'gd', 'nesterov', 'spsa'.
    stepsize : float
        Learning rate.
    batch_size : int or None
        If set, each epoch trains on a random mini-batch of this size.
    """

    _OPTIMIZERS = {
    "adam": qml.AdamOptimizer,
    "gd": qml.GradientDescentOptimizer,
    "nesterov": qml.NesterovMomentumOptimizer,
    "spsa": qml.SPSAOptimizer,
}

    def __init__(self, model, optimizer_type="adam", stepsize=0.1, batch_size=None):
        self.model = model
        self.batch_size = batch_size

        key = optimizer_type.lower()
        if key not in Trainer._OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: '{optimizer_type}'. "
                f"Choose from {list(Trainer._OPTIMIZERS)}"
            )
        self.opt = Trainer._OPTIMIZERS[key](stepsize=stepsize)


    def cost_function(self, weights, X, Y):
        """Binary cross-entropy on rescaled VQC output.

        The PauliZ expectation value in [-1, +1] is mapped to [0, 1]
        via (raw + 1) / 2, then standard BCE is applied.
        """
        raw = np.array([self.model.forward(x, weights) for x in X])
        probs = (raw + 1.0) / 2.0
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return -np.mean(
            Y * np.log(probs) + (1.0 - Y) * np.log(1.0 - probs)
        )

    def fit(
        self,
        X,
        Y,
        epochs=20,
        X_val=None,
        Y_val=None,
        patience=None,
        verbose_every=5,
    ):
        """
        Initialise weights and run the training loop.

        Parameters
        ----------
        X, Y : array-like
            Training features and labels.
        epochs : int
            Number of training epochs.
        X_val, Y_val : array-like or None
            Optional validation set for tracking generalisation.
        patience : int or None
            Early-stop after *patience* epochs without validation improvement.
        verbose_every : int
            Print progress every *verbose_every* epochs (0 = silent).

        Returns
        -------
        dict  {weights, train_history, val_history}
        """
        weight_shape = self.model.ansatz.get_weight_shape(self.model.n_qubits)
        weights = np.random.random(weight_shape, requires_grad=True)

        train_history = []
        val_history = []
        best_val_cost = float("inf")
        stale_epochs = 0

        self.model.train()

        for epoch in range(epochs):
            if self.batch_size is not None and self.batch_size < len(X):
                idx = np.random.choice(len(X), self.batch_size, replace=False)
                X_batch, Y_batch = X[idx], Y[idx]
            else:
                X_batch, Y_batch = X, Y

            weights, _, _ = self.opt.step(
                self.cost_function, weights, X_batch, Y_batch
            )

            train_cost = self.cost_function(weights, X, Y)
            train_history.append(float(train_cost))

            val_cost = None
            if X_val is not None and Y_val is not None:
                self.model.eval()
                val_cost = float(self.cost_function(weights, X_val, Y_val))
                val_history.append(val_cost)
                self.model.train()

                if patience is not None:
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= patience:
                            if verbose_every:
                                print(
                                    f"Early stopping at epoch {epoch + 1} "
                                    f"(no improvement for {patience} epochs)"
                                )
                            break

            if verbose_every and (epoch + 1) % verbose_every == 0:
                msg = f"Epoch {epoch + 1:4d} | Train cost: {train_cost:.5f}"
                if val_cost is not None:
                    msg += f" | Val cost: {val_cost:.5f}"
                print(msg)

        self.model.eval()
        return {
            "weights": weights,
            "train_history": train_history,
            "val_history": val_history,
        }
