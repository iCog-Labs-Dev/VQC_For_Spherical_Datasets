from shared.plotting import plot_training_loss as _plot_loss, plot_decision_regions
from .logistic_model import logistic_model_batch


def plot_training_loss(loss_history, save_path=None):
    _plot_loss(
        loss_history,
        "Logistic Regression Training Loss",
        "Binary Cross-Entropy Loss",
        save_path,
    )


def plot_decision_boundary(weights, X, y, save_path=None):
    plot_decision_regions(
        X,
        y,
        lambda grid: logistic_model_batch(weights, grid),
        "Logistic Regression Decision Boundary",
        "P(Pass)",
        probability=True,
        save_path=save_path,
    )
