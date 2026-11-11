from shared.plotting import plot_training_loss as _plot_loss, plot_decision_regions
from .linear_model import linear_model_batch


def plot_training_loss(loss_history, save_path=None):
    _plot_loss(
        loss_history, "Linear Regression Training Loss", "MSE Loss", save_path
    )


def plot_decision_boundary(weights, X, y, save_path=None):
    plot_decision_regions(
        X,
        y,
        lambda grid: linear_model_batch(weights, grid),
        "Linear Regression Decision Boundary",
        "Model Output",
        save_path=save_path,
    )
