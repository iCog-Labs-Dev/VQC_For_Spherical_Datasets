"""Seeds, splits, metrics, the results schema, and the shared training call."""

import csv
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from . import Config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

FIELDS = ["experiment", "target", "representation", "model", "encoding",
          "n_uploads", "n_qubits", "n_layers", "n_params", "n_train",
          "rotation_id", "polar_angle", "seed", "stepsize",
          "train_acc", "test_acc", "balanced_acc", "majority", "ceiling",
          "final_loss", "wall_time"]


def seeds(i):
    """Three independent streams, so runs never correlate by accident."""
    return dict(data=C.DATA_SEED_BASE + i,
                init=C.INIT_SEED_BASE + i,
                split=C.SPLIT_SEED_BASE + i)


def split(theta, phi, y, seed, test_fraction=C.TEST_FRACTION, n_train=None):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_fraction,
                              random_state=seed, stratify=y)
    if n_train is not None and n_train < len(tr):
        tr = np.random.default_rng(seed).choice(tr, size=n_train, replace=False)
    return ((theta[tr], phi[tr], y[tr]), (theta[te], phi[te], y[te]))


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred).astype(int) == np.asarray(y_true).astype(int)))


def balanced_accuracy(y_true, y_pred):
    """
    Mean of the per-class recalls.  Required, not optional: on a 42/58 target
    plain accuracy rewards a model for guessing the majority class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in (0, 1) if np.any(y_true == c)]
    return float(np.mean(recalls))


def majority_rate(y):
    p = float(np.mean(np.asarray(y)))
    return max(p, 1 - p)


def blank_row(**kw):
    row = {f: "" for f in FIELDS}
    row.update(kw)
    return row


def write_rows(filename, rows, append=False):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, filename)
    mode = "a" if (append and os.path.exists(path)) else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    print(f"  -> results/{filename}  ({len(rows)} rows)")
    return path


def train_vqc(train, test, n_qubits=2, n_layers=C.DEPTH_DEFAULT, n_uploads=1,
              encoding="spherical", seed_init=0, epochs=C.EPOCHS,
              stepsize=0.05, verbose=0):
    """
    Fit one VQC configuration and report train and test accuracy.

    Train accuracy is returned deliberately.  A model that cannot fit its own
    training set is expressivity-limited; one that fits it and generalises
    badly is not.  Reporting only test accuracy makes those look alike, which
    is exactly the confusion the falsification phase must avoid.
    """
    from pennylane import numpy as pnp

    from .AnsatzLayer import AnsatzLayer
    from .EmbeddingLayer import EmbeddingLayer
    from .VQCModel import VQCModel
    from .VQCOptimizer import Trainer

    (th_tr, ph_tr, y_tr), (th_te, ph_te, y_te) = train, test
    model = VQCModel(n_qubits=n_qubits,
                     embedding=EmbeddingLayer(encoding),
                     ansatz=AnsatzLayer("strong", n_layers=n_layers),
                     measurement="expval", n_uploads=n_uploads)

    X_tr = pnp.array(np.column_stack([th_tr, ph_tr]), requires_grad=False)
    Y_tr = pnp.array(np.asarray(y_tr, dtype=float), requires_grad=False)
    X_te = pnp.array(np.column_stack([th_te, ph_te]), requires_grad=False)
    Y_te = pnp.array(np.asarray(y_te, dtype=float), requires_grad=False)

    pnp.random.seed(seed_init)
    trainer = Trainer(model, optimizer_type=C.OPTIMIZER, stepsize=stepsize)

    t0 = time.time()
    res = trainer.fit(X_tr, Y_tr, epochs=epochs, X_val=X_te, Y_val=Y_te,
                      verbose_every=verbose)
    wall = time.time() - t0

    w = res["weights"]
    model.eval()

    def predict(X):
        return (np.asarray(trainer._evaluate(w, X), dtype=float) > 0.0).astype(int)

    p_tr, p_te = predict(X_tr), predict(X_te)
    return dict(model=model, weights=w, n_params=model.n_params(),
                train_acc=accuracy(y_tr, p_tr),
                test_acc=accuracy(y_te, p_te),
                balanced_acc=balanced_accuracy(y_te, p_te),
                final_loss=res["train_history"][-1] if res["train_history"] else float("nan"),
                wall_time=wall)


def run_classical(model_obj, train, test):
    (th_tr, ph_tr, y_tr), (th_te, ph_te, y_te) = train, test
    t0 = time.time()
    model_obj.fit(th_tr, ph_tr, y_tr)
    p_tr = model_obj.predict(th_tr, ph_tr)
    p_te = model_obj.predict(th_te, ph_te)
    return dict(n_params=model_obj.param_count(),
                train_acc=accuracy(y_tr, p_tr),
                test_acc=accuracy(y_te, p_te),
                balanced_acc=balanced_accuracy(y_te, p_te),
                final_loss="", wall_time=time.time() - t0)
