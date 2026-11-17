"""Experiment configurations. Select one with `run_experiment.py --config NAME`."""
from dataclasses import asdict, dataclass, field, replace

import numpy as np

CONDITIONS = ("exact_exact", "sampled_exact", "exact_sampled", "sampled_sampled")
BACKENDS = ("bernoulli_reference", "pennylane_simulator")
BUDGET_MODES = ("per_evaluation", "fixed_total")

DEFAULT_TARGETS = (
    {"name": "Y_1_0", "terms": ((1, 0, 1.0),)},
    {"name": "Y_2_0", "terms": ((2, 0, 1.0),)},
    {"name": "Y_3_2", "terms": ((3, 2, 1.0),)},
    {"name": "Y_4_1", "terms": ((4, 1, 1.0),)},
    {"name": "mixed", "terms": ((1, 0, 0.5), (2, 1, 0.3), (3, -2, 0.2))},
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    copy_counts: tuple = (1, 2, 4, 8)
    shots_per_evaluation: tuple = (100, 500, 1000, 5000)
    train_sizes: tuple = (32, 64, 128, 256)
    n_val: int = 100
    n_test: int = 300
    targets: tuple = DEFAULT_TARGETS
    conditions: tuple = CONDITIONS
    data_seeds: tuple = tuple(range(5))
    shot_seeds: tuple = tuple(range(10))
    ridge_lambdas: tuple = tuple(float(v) for v in np.logspace(-8, 0, 9))
    ridge_convention: str = "(K + lambda*I) alpha = y"
    harmonic_max_degree: int = 6
    measurement_backend: str = "bernoulli_reference"
    budget_mode: str = "per_evaluation"
    total_budgets: tuple = ()
    allocations: tuple = ("uniform",)  # "uniform" or a training-matrix budget fraction

    def __post_init__(self):
        if self.measurement_backend not in BACKENDS:
            raise ValueError(f"measurement_backend must be one of {BACKENDS}")
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"budget_mode must be one of {BUDGET_MODES}")
        if not set(self.conditions) <= set(CONDITIONS):
            raise ValueError(f"conditions must be a subset of {CONDITIONS}")
        if self.budget_mode == "fixed_total":
            if not self.total_budgets:
                raise ValueError("fixed_total mode requires total_budgets")
            if not set(self.conditions) <= {"exact_exact", "sampled_sampled"}:
                raise ValueError("fixed_total mode supports exact_exact and sampled_sampled only")

    def to_dict(self):
        return asdict(self)


CONFIGS = {
    "default": ExperimentConfig(name="default"),
    "fixed_budget": ExperimentConfig(
        name="fixed_budget",
        budget_mode="fixed_total",
        train_sizes=(64,),
        conditions=("exact_exact", "sampled_sampled"),
        total_budgets=(2_000_000, 8_000_000, 32_000_000),
        allocations=("uniform", 0.25, 0.5, 0.75),
    ),
    "smoke": ExperimentConfig(
        name="smoke",
        copy_counts=(1, 2),
        shots_per_evaluation=(50, 200),
        train_sizes=(16,),
        n_val=20,
        n_test=40,
        targets=(DEFAULT_TARGETS[0], DEFAULT_TARGETS[-1]),
        data_seeds=(0, 1),
        shot_seeds=(0, 1),
        ridge_lambdas=(1e-6, 1e-3, 1e-1),
        harmonic_max_degree=4,
    ),
}


def get_config(name):
    if name not in CONFIGS:
        raise KeyError(f"unknown config '{name}'; choose from {sorted(CONFIGS)}")
    return CONFIGS[name]


def with_overrides(cfg, **kwargs):
    return replace(cfg, **kwargs)
