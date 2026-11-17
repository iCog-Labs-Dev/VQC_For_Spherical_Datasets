"""Shot accounting and allocation.

Diagonal policy: training-Gram diagonal entries are k(x, x) = 1 analytically and are never
sampled, so a training matrix has N(N-1)/2 sampled unordered pairs. Prediction (validation
and test) adds one sampled evaluation per (query point, training point).
"""
from dataclasses import dataclass

import numpy as np


class ShotLedger:
    """Counts sampled kernel evaluations and shots per category."""

    def __init__(self):
        self.evaluations = {}
        self.shots = {}

    def record(self, category, evaluations, shots):
        self.evaluations[category] = self.evaluations.get(category, 0) + int(evaluations)
        self.shots[category] = self.shots.get(category, 0) + int(shots)

    def total(self, categories=None):
        """(evaluations, shots) summed over the given categories (all if None)."""
        cats = self.shots.keys() if categories is None else categories
        return (sum(self.evaluations.get(c, 0) for c in cats),
                sum(self.shots.get(c, 0) for c in cats))


@dataclass(frozen=True)
class ShotPlan:
    label: str
    allocation: str
    budget: int           # 0 in per-evaluation mode
    shots_nominal: float  # shots per evaluation (mean in fixed-total mode)
    train_shots: np.ndarray
    pred_shots: np.ndarray  # validation evaluations first, then test, row-major


def split_total(total, count):
    """Positive integer shots for `count` evaluations summing exactly to `total`."""
    if count < 1 or total < count:
        raise ValueError(f"cannot give {count} evaluations at least one shot from {total}")
    base, rem = divmod(int(total), int(count))
    out = np.full(count, base, dtype=np.int64)
    out[:rem] += 1
    return out


def allocate_fixed_total(total, n_train_evals, n_pred_evals, train_fraction=None):
    """Split a total budget between training-matrix and prediction evaluations.

    train_fraction=None gives uniform allocation (equal shots per evaluation up to
    integer rounding); otherwise that fraction of the budget goes to the training matrix.
    """
    if train_fraction is None:
        train_fraction = n_train_evals / (n_train_evals + n_pred_evals)
    b_train = int(round(total * train_fraction))
    train = split_total(b_train, n_train_evals)
    pred = split_total(total - b_train, n_pred_evals)
    assert train.sum() + pred.sum() == total
    return train, pred


def build_plans(cfg, train_size):
    """All shot plans for one training size under the configured budget mode."""
    n_train = train_size * (train_size - 1) // 2
    n_pred = (cfg.n_val + cfg.n_test) * train_size
    if cfg.budget_mode == "per_evaluation":
        return [ShotPlan(f"S={s}", "uniform", 0, float(s),
                         np.full(n_train, s, dtype=np.int64), np.full(n_pred, s, dtype=np.int64))
                for s in cfg.shots_per_evaluation]
    plans = []
    for total in cfg.total_budgets:
        for alloc in cfg.allocations:
            frac = None if alloc == "uniform" else float(alloc)
            label = "uniform" if frac is None else f"train_{frac:g}"
            train, pred = allocate_fixed_total(total, n_train, n_pred, frac)
            plans.append(ShotPlan(f"B={total}|{label}", label, int(total),
                                  total / (n_train + n_pred), train, pred))
    return plans
