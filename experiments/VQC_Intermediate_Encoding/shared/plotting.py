"""Consistent academic plotting helpers (blue palette, publication-friendly)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#1f4e79"
LIGHT_BLUE = "#9bbad3"
MID_BLUE = "#477da8"


def plot_training_loss(loss_history, title, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        marker="o",
        markersize=3.5,
        color=BLUE,
        linewidth=1.8,
    )
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()
    plt.close(fig)


def plot_decision_regions(
    X, y, predict_fn, title, colorbar_label, probability=False, save_path=None
):
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 30),
        np.linspace(y_min, y_max, 30),
    )
    other_means = X[:, 2:].mean(axis=0) if X.shape[1] > 2 else np.array([])
    grid = np.c_[xx.ravel(), yy.ravel()]
    if other_means.size:
        grid = np.hstack([grid, np.tile(other_means, (grid.shape[0], 1))])
    Z = np.asarray(predict_fn(grid)).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.linspace(0, 1, 21) if probability else 20
    contour = ax.contourf(
        xx, yy, Z, levels=levels, cmap="Blues", alpha=0.85
    )
    ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="binary",
        edgecolors="black", linewidths=0.45, s=24,
    )
    ax.set(title=title, xlabel="Feature 1", ylabel="Feature 2")
    fig.colorbar(contour, ax=ax, label=colorbar_label)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()
    plt.close(fig)
