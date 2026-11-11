from shared.plotting import plot_training_loss as _plot_loss, plot_decision_regions
from .vqc import vqc_circuit


def plot_training_loss(loss_history, save_path=None):
    _plot_loss(
        loss_history,
        "VQC Training Loss (Intermediate Encoding)",
        "MSE Loss",
        save_path,
    )


def plot_decision_boundary(weights, X, y, save_path=None):
    plot_decision_regions(
        X,
        y,
        lambda grid: [vqc_circuit(weights, point) for point in grid],
        "Quantum Decision Boundary",
        "QNode Output",
        save_path=save_path,
    )
