"""
Sphere Moons: VQC (spherical embedding) vs classical FFNN.

Two crescent-shaped classes on the surface of a unit sphere, where
BOTH theta (latitude) and phi (longitude) determine the class.
Radius is constant -- there is zero radial information.

The VQC receives (theta, phi) directly into RY(theta) RZ(phi), placing
each point on the Bloch sphere.  The FFNN receives the Cartesian
projection (x, y, z) and must reconstruct the spherical geometry
using ReLU activations.

We compare:
  1. VQC with spherical embedding  (1 data qubit + 1 auxiliary)
  2. Classical FFNN on Cartesian (x, y, z)  -- 3 inputs

The VQC hyperparameters are selected via grid search before the
final comparison run.

Usage
-----
    python experiment_sphere_moons.py
"""

import sys
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from pennylane import numpy as pnp

from Embedding import EmbeddingLayer
from Ansatz import AnsatzLayer
from VQCModel import VQCModel
from Optimizer import Trainer
from ClassicalBaseline import ClassicalBaseline
from data_utils import make_sphere_moons, sphere_to_cartesian

def load_data(n_samples=400, noise_std=0.08, seed=42):
    theta, phi, labels = make_sphere_moons(
        n_samples=n_samples, noise_std=noise_std, seed=seed
    )

    X_spherical = np.column_stack([theta, phi])
    X_cartesian = sphere_to_cartesian(theta, phi)

    (Xs_train, Xs_test,
     Xc_train, Xc_test,
     y_train, y_test) = train_test_split(
        X_spherical, X_cartesian, labels,
        test_size=0.25, random_state=seed, stratify=labels,
    )

    return (Xs_train, Xs_test, Xc_train, Xc_test,
            y_train, y_test, X_spherical, X_cartesian, labels)


def _train_vqc_single(Xs_train, y_train, Xs_test, y_test,
                       n_layers, epochs, stepsize, verbose_every=0):
    """Train one VQC configuration and return (val_acc, results, model, weights, n_params)."""
    n_qubits = 2

    embedding = EmbeddingLayer(method="spherical")
    ansatz = AnsatzLayer(method="strong", n_layers=n_layers)
    model = VQCModel(
        n_qubits=n_qubits,
        embedding=embedding,
        ansatz=ansatz,
        measurement="expval",
        dropout_rate=0.0,
    )

    X_tr = pnp.array(Xs_train, requires_grad=False)
    Y_tr = pnp.array(y_train, requires_grad=False)
    X_te = pnp.array(Xs_test, requires_grad=False)
    Y_te = pnp.array(y_test, requires_grad=False)

    trainer = Trainer(model, optimizer_type="adam", stepsize=stepsize)

    weight_shape = ansatz.get_weight_shape(n_qubits)
    n_params = int(np.prod(weight_shape))

    results = trainer.fit(
        X_tr, Y_tr,
        epochs=epochs,
        X_val=X_te, Y_val=Y_te,
        verbose_every=verbose_every,
    )

    weights = results["weights"]
    model.eval()

    preds_raw = np.array([float(model.forward(x, weights)) for x in X_te])
    preds_class = (preds_raw > 0.0).astype(int)
    acc = np.mean(preds_class == y_test.astype(int))

    return acc, results, model, weights, n_params


SEARCH_GRID = {
    "n_layers":  [2, 4, 6],
    "stepsize":  [0.05, 0.10, 0.20],
    "epochs":    [80, 150],
}


def tune_vqc(Xs_train, y_train, Xs_test, y_test):
    """
    Grid search over VQC hyperparameters.

    Returns the best configuration and its results.
    """
    keys = list(SEARCH_GRID.keys())
    combos = list(itertools.product(*[SEARCH_GRID[k] for k in keys]))
    n_combos = len(combos)

    print(f"\n{'='*60}")
    print(f"  VQC Hyperparameter Tuning  |  {n_combos} configurations")
    print(f"{'='*60}")
    print(f"  Grid: n_layers={SEARCH_GRID['n_layers']}, "
          f"stepsize={SEARCH_GRID['stepsize']}, "
          f"epochs={SEARCH_GRID['epochs']}")
    print(f"{'-'*60}")

    best_acc = -1.0
    best_cfg = {}
    best_result = None
    all_results = []

    for i, vals in enumerate(combos, 1):
        cfg = dict(zip(keys, vals))
        print(f"  [{i:2d}/{n_combos}]  layers={cfg['n_layers']}  "
              f"lr={cfg['stepsize']:.2f}  epochs={cfg['epochs']}", end="")

        acc, results, model, weights, n_params = _train_vqc_single(
            Xs_train, y_train, Xs_test, y_test,
            n_layers=cfg["n_layers"],
            epochs=cfg["epochs"],
            stepsize=cfg["stepsize"],
            verbose_every=0,
        )

        cfg["accuracy"] = acc
        cfg["n_params"] = n_params
        cfg["final_cost"] = results["cost_history"][-1] if "cost_history" in results else results["train_history"][-1]
        all_results.append(cfg)

        print(f"  →  acc={acc:.2%}  params={n_params}")

        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_result = (results, model, weights, n_params)

    print(f"{'-'*60}")
    print(f"  Best: layers={best_cfg['n_layers']}  "
          f"lr={best_cfg['stepsize']:.2f}  "
          f"epochs={best_cfg['epochs']}  "
          f"→  acc={best_acc:.2%}  ({best_cfg['n_params']} params)")
    print(f"{'='*60}")

    return best_cfg, best_result, all_results



def train_vqc_final(Xs_train, y_train, Xs_test, y_test, cfg):
    """Re-train the VQC with the best hyperparameters, printing progress."""
    print(f"\n{'='*55}")
    print(f"  VQC Final Run  |  layers={cfg['n_layers']}  "
          f"lr={cfg['stepsize']:.2f}  epochs={cfg['epochs']}")
    print(f"{'='*55}")

    acc, results, model, weights, n_params = _train_vqc_single(
        Xs_train, y_train, Xs_test, y_test,
        n_layers=cfg["n_layers"],
        epochs=cfg["epochs"],
        stepsize=cfg["stepsize"],
        verbose_every=10,
    )
    print(f"VQC test accuracy: {acc:.2%}  ({n_params} params)")
    return results, model, weights, acc, n_params



def train_classical(Xc_train, y_train, Xc_test, y_test, epochs=400):
    baseline = ClassicalBaseline(n_input=3, hidden_size=5, lr=0.01)
    n_params = baseline.param_count()
    print(f"\n{'='*55}")
    print(f"  FFNN  |  Cartesian (x,y,z)  |  {n_params} params")
    print(f"{'='*55}")
    cls_results = baseline.fit(
        Xc_train.astype(np.float32), y_train.astype(np.float32),
        epochs=epochs,
        X_val=Xc_test.astype(np.float32), Y_val=y_test.astype(np.float32),
        verbose_every=50,
    )
    preds = baseline.predict_classes(Xc_test.astype(np.float32))
    acc = np.mean(preds == y_test.astype(int))
    print(f"FFNN test accuracy: {acc:.2%}")
    return cls_results, baseline, acc, n_params

def plot_results(X_cart, labels, X_sph,
                 vqc_res, cls_res, vqc_acc, cls_acc,
                 vqc_params, cls_params, best_cfg):
    fig = plt.figure(figsize=(18, 5))

    # (a) 3-D scatter on the unit sphere
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    colours = ["steelblue" if l == 0 else "salmon" for l in labels]
    ax.scatter(X_cart[:, 0], X_cart[:, 1], X_cart[:, 2],
               c=colours, s=12, alpha=0.7, edgecolors="k", linewidths=0.2)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = 0.98 * np.outer(np.cos(u), np.sin(v))
    ys = 0.98 * np.outer(np.sin(u), np.sin(v))
    zs = 0.98 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="grey", alpha=0.08, linewidth=0.3)
    ax.set_title("Sphere Moons dataset")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # (b) Training curves
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(vqc_res["train_history"], label="VQC train", linewidth=1.5)
    if vqc_res["val_history"]:
        ax2.plot(vqc_res["val_history"], label="VQC val",
                 linestyle="--", linewidth=1.5)
    ax2.plot(cls_res["train_history"], label="FFNN train", alpha=0.7)
    if cls_res["val_history"]:
        ax2.plot(cls_res["val_history"], label="FFNN val",
                 linestyle="--", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cost / Loss (BCE)")
    ax2.set_title("Training curves (best VQC config)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # (c) Accuracy bar chart
    ax3 = fig.add_subplot(1, 3, 3)
    vqc_label = (f"VQC\nspherical\n({vqc_params}p, L={best_cfg['n_layers']}, "
                 f"lr={best_cfg['stepsize']})")
    cls_label = f"FFNN\nCartesian\n({cls_params} params)"
    names = [vqc_label, cls_label]
    accs = [vqc_acc, cls_acc]
    colors = ["steelblue", "salmon"]
    bars = ax3.bar(names, accs, color=colors)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Test Accuracy")
    ax3.set_title("Accuracy comparison")
    for bar, val in zip(bars, accs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "sphere_moons_results.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Figure saved to {out_path}")


def plot_tuning_heatmap(all_results):
    """Heatmap of accuracy vs n_layers and stepsize (averaged over epochs)."""
    layers_vals = sorted(set(r["n_layers"] for r in all_results))
    lr_vals = sorted(set(r["stepsize"] for r in all_results))

    grid = np.zeros((len(lr_vals), len(layers_vals)))
    for r in all_results:
        li = layers_vals.index(r["n_layers"])
        ri = lr_vals.index(r["stepsize"])
        grid[ri, li] = max(grid[ri, li], r["accuracy"])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="YlGnBu", vmin=0.5, vmax=1.0,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(layers_vals)))
    ax.set_xticklabels(layers_vals)
    ax.set_yticks(range(len(lr_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in lr_vals])
    ax.set_xlabel("n_layers")
    ax.set_ylabel("stepsize")
    ax.set_title("VQC Hyperparameter Tuning (best accuracy per cell)")

    for i in range(len(lr_vals)):
        for j in range(len(layers_vals)):
            ax.text(j, i, f"{grid[i, j]:.1%}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if grid[i, j] > 0.75 else "black")

    fig.colorbar(im, ax=ax, label="Test Accuracy")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "vqc_tuning_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Heatmap saved to {out_path}")


# ======================================================================
# 7.  Main
# ======================================================================

def main():
    (Xs_train, Xs_test, Xc_train, Xc_test,
     y_train, y_test, X_sph, X_cart, labels) = load_data()

    best_cfg, _, all_results = tune_vqc(
        Xs_train, y_train, Xs_test, y_test
    )

    plot_tuning_heatmap(all_results)

    vqc_res, _, _, vqc_acc, vqc_p = train_vqc_final(
        Xs_train, y_train, Xs_test, y_test, best_cfg
    )
    cls_res, _, cls_acc, cls_p = train_classical(
        Xc_train, y_train, Xc_test, y_test
    )

    print(f"\n{'='*55}")
    print(f"  VQC  accuracy: {vqc_acc:.2%}  ({vqc_p} params)")
    print(f"  FFNN accuracy: {cls_acc:.2%}  ({cls_p} params)")
    print(f"{'='*55}")

    plot_results(X_cart, labels, X_sph,
                 vqc_res, cls_res, vqc_acc, cls_acc,
                 vqc_p, cls_p, best_cfg)


if __name__ == "__main__":
    main()
