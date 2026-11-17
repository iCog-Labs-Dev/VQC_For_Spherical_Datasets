"""One experimental context (copy count, training size, data seed) across conditions and plans."""
import numpy as np

from .budget import ShotLedger, build_plans
from .diagnostics import estimator_error, spectrum_summary, symmetry_error
from .kernel import exact_kernel
from .kernel_matrices import sampled_cross, sampled_training_gram
from .metrics import correlation, harmonic_recovery, mse, normalized_mse
from .regression import fit_path, select_lambda
from .shot_estimator import get_backend
from .sphere import make_rng, sample_sphere
from .targets import build_target, harmonic_basis

# condition -> (sampled training Gram, sampled prediction vectors)
CONDITION_FLAGS = {
    "exact_exact": (False, False),
    "sampled_exact": (True, False),
    "exact_sampled": (False, True),
    "sampled_sampled": (True, True),
}
COST_CATEGORIES = {
    "exact_exact": (),
    "sampled_exact": ("train",),
    "exact_sampled": ("val", "test"),
    "sampled_sampled": ("train", "val", "test"),
}


def _evaluate(condition, alphas, ill, cross_val, cross_test, ctx, plan_info, cost, data):
    """Select lambda on validation data, then score the test set; one row per target."""
    idx, val_mse = select_lambda(alphas, cross_val, data["y_val"])
    rows = []
    for t, target in enumerate(data["targets"]):
        alpha = alphas[idx[t], :, t]
        pred = cross_test @ alpha
        y = data["y_test"][:, t]
        row = {
            **ctx, **plan_info, "condition": condition, "target": target.name,
            "selected_lambda": data["lambdas"][idx[t]],
            "val_mse": float(val_mse[idx[t], t]),
            "test_mse": mse(y, pred),
            "nmse": normalized_mse(y, pred),
            "corr": correlation(y, pred),
            "ill_conditioned": bool(ill[idx[t]]),
            "evaluations": cost[0],
            "total_shots": cost[1],
        }
        row.update(harmonic_recovery(y, pred, data["basis"], data["degrees"]))
        rows.append(row)
    return rows


def run_context(cfg, n, train_size, data_seed):
    """Run all conditions and shot plans for one (n, N, data seed). Returns (rows, diagnostics)."""
    backend = get_backend(cfg.measurement_backend)
    rng = make_rng(data_seed, train_size)
    x_tr = sample_sphere(train_size, rng)
    x_va = sample_sphere(cfg.n_val, rng)
    x_te = sample_sphere(cfg.n_test, rng)
    targets = [build_target(spec) for spec in cfg.targets]
    basis, degrees = harmonic_basis(x_te, cfg.harmonic_max_degree)
    data = {
        "targets": targets,
        "lambdas": np.array(cfg.ridge_lambdas),
        "y_val": np.column_stack([t(x_va) for t in targets]),
        "y_test": np.column_stack([t(x_te) for t in targets]),
        "basis": basis,
        "degrees": degrees,
    }
    y_tr = np.column_stack([t(x_tr) for t in targets])

    gram = exact_kernel(x_tr, x_tr, n)
    cross_val = exact_kernel(x_va, x_tr, n)
    cross_test = exact_kernel(x_te, x_tr, n)
    alphas_exact, ill_exact = fit_path(gram, y_tr, data["lambdas"])

    ctx = {"n": n, "N": train_size, "data_seed": data_seed}
    exact_info = {"plan": "exact", "allocation": "none", "budget": 0,
                  "shots_nominal": 0.0, "shot_seed": -1}
    rows = []
    if "exact_exact" in cfg.conditions:
        rows += _evaluate("exact_exact", alphas_exact, ill_exact, cross_val, cross_test,
                          ctx, exact_info, (0, 0), data)

    sampled_conditions = [c for c in cfg.conditions if c != "exact_exact"]
    diag_rows = []
    if not sampled_conditions:
        return rows, diag_rows

    need_train = any(CONDITION_FLAGS[c][0] for c in sampled_conditions)
    need_pred = any(CONDITION_FLAGS[c][1] for c in sampled_conditions)
    exact_spec = spectrum_summary(gram)
    iu = np.triu_indices(train_size, k=1)
    n_val_evals = cfg.n_val * train_size

    for plan_idx, plan in enumerate(build_plans(cfg, train_size)):
        plan_info = {"plan": plan.label, "allocation": plan.allocation,
                     "budget": plan.budget, "shots_nominal": plan.shots_nominal}
        for shot_seed in cfg.shot_seeds:
            srng = make_rng(shot_seed, data_seed, n, train_size, plan_idx)
            ledger = ShotLedger()
            diag = {**ctx, **plan_info, "shot_seed": shot_seed}
            gram_s = alphas_s = ill_s = cross_val_s = cross_test_s = None
            if need_train:
                gram_s = sampled_training_gram(x_tr, n, plan.train_shots, backend, srng, ledger)
                alphas_s, ill_s = fit_path(gram_s, y_tr, data["lambdas"])
                diag.update({f"train_{k}": v for k, v in estimator_error(
                    gram_s[iu], gram[iu], plan.train_shots).items()})
                diag["train_symmetry_error"] = symmetry_error(gram_s)
                diag.update({f"train_{k}": v for k, v in spectrum_summary(gram_s).items()})
                diag.update({f"exact_train_{k}": v for k, v in exact_spec.items()})
            if need_pred:
                cross_val_s = sampled_cross(x_va, x_tr, n, plan.pred_shots[:n_val_evals],
                                            backend, srng, ledger, "val")
                cross_test_s = sampled_cross(x_te, x_tr, n, plan.pred_shots[n_val_evals:],
                                             backend, srng, ledger, "test")
                est = np.concatenate([cross_val_s.ravel(), cross_test_s.ravel()])
                ref = np.concatenate([cross_val.ravel(), cross_test.ravel()])
                diag.update({f"cross_{k}": v for k, v in estimator_error(
                    est, ref, plan.pred_shots).items()})
            diag["evaluations"], diag["total_shots"] = ledger.total()
            diag_rows.append(diag)

            info = {**plan_info, "shot_seed": shot_seed}
            for cond in sampled_conditions:
                s_train, s_pred = CONDITION_FLAGS[cond]
                rows += _evaluate(
                    cond,
                    alphas_s if s_train else alphas_exact,
                    ill_s if s_train else ill_exact,
                    cross_val_s if s_pred else cross_val,
                    cross_test_s if s_pred else cross_test,
                    ctx, info, ledger.total(COST_CATEGORIES[cond]), data)
    return rows, diag_rows
