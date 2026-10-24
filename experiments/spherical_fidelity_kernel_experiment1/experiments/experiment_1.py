"""Experiment 1: exact-kernel expressivity and recoverability on S^2.

The full grid uses the closed-form exact kernel for reproducibility and speed.
PennyLane's analytic overlap circuit is checked in a small pilot.

From the project root:
    python -m experiments.experiment_1 --pilot
    python -m experiments.experiment_1 --quick
    python -m experiments.experiment_1 --no-quantum
    python -m experiments.experiment_1
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import pennylane as qml
except ImportError:  # permits the classical exact-kernel grid without PennyLane
    qml = None
import scipy
from scipy.linalg import eigh

from src import metrics, spectrum
from src.kernel import exact_kernel
from src.sphere import sample_sphere
from src.targets import random_degree_target, zonal_target, real_spherical_harmonic

KERNEL_TOL = 1e-10
PREDICTION_TOL = 1e-6


@dataclass
class Config:
    """Predeclared main Experiment 1 grid."""
    copies: tuple = (1, 2, 4, 8)
    degrees: tuple = tuple(range(0, 11))
    train_sizes: tuple = (32, 64, 128, 256)
    taus: tuple = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    target_modes: tuple = ("zonal", "random_harmonic")
    seeds: tuple = tuple(range(20))
    target_seeds: tuple = (0, 1, 2)
    n_test: int = 1000
    n_validation: int = 400
    test_seed: int = 12345
    validation_seed: int = 23456
    pilot_pairs: int = 50
    pilot_gram_size: int = 24
    use_quantum: bool = True
    run_baselines: bool = True


def environment_info():
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "pennylane_version": getattr(qml, "__version__", None),
        "full_grid_kernel": "closed-form exact kernel k_n(x,y)=((1+x dot y)/2)^n",
        "quantum_validation": "analytic overlap-circuit validation pilot; shots=None",
        "finite_shot_end_to_end_krr": "reserved for Experiment 2",
    }

    """
    _krr_eigendecomp and _krr_predictions are helpers use
    eigendecomposition to solve the regularized regression
    problem efficiently across several values of tau.
    Tiny negative eigenvalues caused by floating-point error
    are clipped to zero for computational optimization and
    numerical safeguard.
    """
def _krr_eigendecomp(K):
    vals, vecs = eigh((K + K.T) * 0.5, check_finite=False)
    # Exact fidelity Gram matrices are PSD(Positive semidefinite). Clip tiny negative numerical values.
    return np.maximum(vals, 0.0), vecs


def _krr_predictions(K_train, K_eval, y, taus):
    """Predict for each tau using one symmetric eigendecomposition."""
    vals, vecs = _krr_eigendecomp(K_train)
    projected = vecs.T @ y
    n = len(y)
    return {float(tau): K_eval @ (vecs @ (projected / (vals + n * tau)))
            for tau in taus}


def run_pilot(cfg: Config):
    """Compare PennyLane analytic overlap circuit to the exact formula."""
    if qml is None:
        raise RuntimeError("PennyLane is required for the circuit pilot. Install requirements.txt or use --no-quantum.")
    from src.quantum_kernel import quantum_kernel_matrix, quantum_kernel_pair
    rng = np.random.default_rng(cfg.test_seed)
    report = []
    tau = float(cfg.taus[0])
    for n in cfg.copies:
        a, b = sample_sphere(cfg.pilot_pairs, rng), sample_sphere(cfg.pilot_pairs, rng)
        pair_errors = [abs(quantum_kernel_pair(x, y, n) - exact_kernel(x[None, :], y[None, :], n)[0, 0])
                       for x, y in zip(a, b)]
        X = sample_sphere(cfg.pilot_gram_size, rng)
        X_eval = sample_sphere(cfg.pilot_gram_size, rng)
        K_exact = exact_kernel(X, X, n)
        K_circuit = quantum_kernel_matrix(X, X, n, symmetric=True)
        K_eval_exact = exact_kernel(X_eval, X, n)
        K_eval_circuit = quantum_kernel_matrix(X_eval, X, n)
        y = zonal_target(min(1, n), X)
        pred_exact = _krr_predictions(K_exact, K_eval_exact, y, [tau])[tau]
        pred_circuit = _krr_predictions(K_circuit, K_eval_circuit, y, [tau])[tau]
        ka = metrics.kernel_agreement(K_exact, K_circuit)
        pa = metrics.prediction_agreement(pred_exact, pred_circuit)
        passed = (max(pair_errors) < KERNEL_TOL and ka["max_abs_err"] < KERNEL_TOL
                  and pa["max_abs_diff"] < PREDICTION_TOL)
        report.append({
            "n": int(n), "random_pair_max_abs_err": float(max(pair_errors)),
            "gram_agreement": ka, "prediction_agreement": pa,
            "exact_matrix_checks": metrics.matrix_checks(K_exact),
            "circuit_matrix_checks": metrics.matrix_checks(K_circuit),
            "passed": bool(passed),
        })
    return report

    """
        Harmonic_design and _harmonic_predictions construct
        spherical-harmonic feature matrices and fit the two
        classical ridge-regression baselines. One uses the
        same nominal harmonic span as the fidelity kernel;
        the other includes degrees up to the maximum target
        degree. The baseline feature matrices and eigendecompositions
        are reused where possible to avoid redundant computation.

    """

def harmonic_design(X, max_degree):
    """Real spherical-harmonic feature matrix through degree max_degree.

    Columns are normalized to unit RMS under the uniform sphere measure. The
    number of features is (max_degree + 1)^2.
    """
    cols = []
    for ell in range(max_degree + 1):
        cols.append(real_spherical_harmonic(ell, 0, X))
        for m in range(1, ell + 1):
            cols.append(real_spherical_harmonic(ell, m, X, "cos"))
            cols.append(real_spherical_harmonic(ell, m, X, "sin"))
    return np.column_stack(cols)


def _harmonic_predictions(Phi_train, Phi_eval, y, taus):
    """Ridge predictions with objective mean squared error + tau*||beta||^2."""
    gram = Phi_train.T @ Phi_train
    vals, vecs = _krr_eigendecomp(gram)
    rhs = Phi_train.T @ y
    projected = vecs.T @ rhs
    n = len(y)
    return {float(tau): Phi_eval @ (vecs @ (projected / (vals + n * tau)))
            for tau in taus}


def _target(mode, degree, X, target_seed):
    if mode == "zonal":
        return zonal_target(degree, X)
    if mode == "random_harmonic":
        return random_degree_target(degree, X, seed=target_seed)
    raise ValueError(f"Unknown target mode: {mode}")


def _realizations(mode, cfg):
    # Zonal target is fixed for each degree. Random harmonic combinations vary
    # independently by target_seed, in addition to the independent train seeds.
    return cfg.target_seeds if mode == "random_harmonic" else (cfg.target_seeds[0],)


def _choose_tau(predictions, y_validation):
    scores = {tau: metrics.mse(pred, y_validation) for tau, pred in predictions.items()}
    best_tau = min(scores, key=scores.get)
    return float(best_tau), float(scores[best_tau])


def run_grid(cfg: Config):
    """Run exact-kernel KRR and two spherical-harmonic ridge baselines.

    Baselines:
      1. matched-span: real harmonics through degree n (kernel's support span)
      2. higher-capacity: real harmonics through max target degree

    The full tau curve is a predeclared sensitivity analysis. Separately, tau is
    selected on a fixed validation set and that selected model is evaluated on
    the held-out test set. Test data never select tau.
    """
    max_degree = max(cfg.degrees)
    X_test = sample_sphere(cfg.n_test, np.random.default_rng(cfg.test_seed))
    X_val = sample_sphere(cfg.n_validation, np.random.default_rng(cfg.validation_seed))
    Y_test = { (mode, ell, ts): _target(mode, ell, X_test, ts)
               for mode in cfg.target_modes for ell in cfg.degrees for ts in _realizations(mode, cfg) }
    Y_val = { (mode, ell, ts): _target(mode, ell, X_val, ts)
              for mode in cfg.target_modes for ell in cfg.degrees for ts in _realizations(mode, cfg) }
    Phi_test_high = harmonic_design(X_test, max_degree) if cfg.run_baselines else None
    Phi_val_high = harmonic_design(X_val, max_degree) if cfg.run_baselines else None
    rows = []

    for seed in cfg.seeds:
        for N in cfg.train_sizes:
            # Train points are matched across n, target families, and model types.
            X = sample_sphere(N, np.random.default_rng([int(seed), int(N), 1907]))
            Phi_train_high = harmonic_design(X, max_degree) if cfg.run_baselines else None
            for n in cfg.copies:
                start = time.perf_counter()
                K_train = exact_kernel(X, X, n)
                K_val = exact_kernel(X_val, X, n)
                K_test = exact_kernel(X_test, X, n)
                kernel_seconds = time.perf_counter() - start
                vals, vecs = _krr_eigendecomp(K_train)
                Phi_train_matched = harmonic_design(X, n) if cfg.run_baselines else None
                Phi_val_matched = harmonic_design(X_val, n) if cfg.run_baselines else None
                Phi_test_matched = harmonic_design(X_test, n) if cfg.run_baselines else None
                if cfg.run_baselines:
                    # Reuse baseline Gram eigendecompositions for each target.
                    gm_vals, gm_vecs = _krr_eigendecomp(Phi_train_matched.T @ Phi_train_matched)
                    gh_vals, gh_vecs = _krr_eigendecomp(Phi_train_high.T @ Phi_train_high)

                for mode in cfg.target_modes:
                    for ell in cfg.degrees:
                        for target_seed in _realizations(mode, cfg):
                            y = _target(mode, ell, X, target_seed)
                            f_val = Y_val[(mode, ell, target_seed)]
                            f_test = Y_test[(mode, ell, target_seed)]
                            target_power = max(float(np.mean(f_test ** 2)), 1e-15)
                            krr_projected = vecs.T @ y
                            krr_val_preds, krr_test_preds = {}, {}
                            for tau in cfg.taus:
                                krr_alpha = vecs @ (krr_projected / (vals + N * tau))
                                krr_val_preds[float(tau)] = K_val @ krr_alpha
                                krr_test_preds[float(tau)] = K_test @ krr_alpha

                            krr_best_tau, krr_val_mse = _choose_tau(krr_val_preds, f_val)
                            krr_selected_pred = krr_test_preds[krr_best_tau]
                            base_val_preds, base_test_preds = {}, {}
                            if cfg.run_baselines:
                                rhs_m = Phi_train_matched.T @ y
                                rhs_h = Phi_train_high.T @ y
                                pm = gm_vecs.T @ rhs_m
                                ph = gh_vecs.T @ rhs_h
                                for tau in cfg.taus:
                                    tau = float(tau)
                                    beta_m = gm_vecs @ (pm / (gm_vals + N * tau))
                                    beta_h = gh_vecs @ (ph / (gh_vals + N * tau))
                                    base_val_preds[("matched", tau)] = Phi_val_matched @ beta_m
                                    base_test_preds[("matched", tau)] = Phi_test_matched @ beta_m
                                    base_val_preds[("higher", tau)] = Phi_val_high @ beta_h
                                    base_test_preds[("higher", tau)] = Phi_test_high @ beta_h
                                best_m, val_m = _choose_tau(
                                    {t: base_val_preds[("matched", float(t))] for t in cfg.taus}, f_val)
                                best_h, val_h = _choose_tau(
                                    {t: base_val_preds[("higher", float(t))] for t in cfg.taus}, f_val)
                                pred_m_selected = base_test_preds[("matched", best_m)]
                                pred_h_selected = base_test_preds[("higher", best_h)]

                            for tau in cfg.taus:
                                tau = float(tau)
                                pred = krr_test_preds[tau]
                                row = {
                                    "n": int(n), "ell": int(ell), "target_mode": mode,
                                    "target_seed": int(target_seed), "N": int(N), "tau": tau,
                                    "seed": int(seed),
                                    "lambda_ell": spectrum.eigenvalue(n, ell),
                                    "g_ell": spectrum.shrinkage(n, ell, tau),
                                    "d_eff": spectrum.effective_dimension(n, tau),
                                    "mse_krr": metrics.mse(pred, f_test),
                                    "nmse_krr": metrics.mse(pred, f_test) / target_power,
                                    "amp_krr": metrics.target_amplitude(pred, f_test),
                                    "corr_krr": metrics.correlation(pred, f_test),
                                    "validation_mse_krr": metrics.mse(krr_val_preds[tau], f_val),
                                    "selected_tau_krr": krr_best_tau,
                                    "selected_tau_validation_mse_krr": krr_val_mse,
                                    "mse_krr_selected": metrics.mse(krr_selected_pred, f_test),
                                    "nmse_krr_selected": metrics.mse(krr_selected_pred, f_test) / target_power,
                                    "amp_krr_selected": metrics.target_amplitude(krr_selected_pred, f_test),
                                    "time_kernel_exact_seconds": kernel_seconds,
                                }
                                if cfg.run_baselines:
                                    pred_m = base_test_preds[("matched", tau)]
                                    pred_h = base_test_preds[("higher", tau)]
                                    row.update({
                                        "mse_harmonic_matched_span": metrics.mse(pred_m, f_test),
                                        "nmse_harmonic_matched_span": metrics.mse(pred_m, f_test) / target_power,
                                        "amp_harmonic_matched_span": metrics.target_amplitude(pred_m, f_test),
                                        "validation_mse_harmonic_matched_span": metrics.mse(base_val_preds[("matched", tau)], f_val),
                                        "selected_tau_harmonic_matched_span": best_m,
                                        "selected_tau_validation_mse_harmonic_matched_span": val_m,
                                        "mse_harmonic_matched_span_selected": metrics.mse(pred_m_selected, f_test),
                                        "nmse_harmonic_matched_span_selected": metrics.mse(pred_m_selected, f_test) / target_power,
                                        "amp_harmonic_matched_span_selected": metrics.target_amplitude(pred_m_selected, f_test),
                                        "mse_harmonic_higher_capacity": metrics.mse(pred_h, f_test),
                                        "nmse_harmonic_higher_capacity": metrics.mse(pred_h, f_test) / target_power,
                                        "amp_harmonic_higher_capacity": metrics.target_amplitude(pred_h, f_test),
                                        "validation_mse_harmonic_higher_capacity": metrics.mse(base_val_preds[("higher", tau)], f_val),
                                        "selected_tau_harmonic_higher_capacity": best_h,
                                        "selected_tau_validation_mse_harmonic_higher_capacity": val_h,
                                        "mse_harmonic_higher_capacity_selected": metrics.mse(pred_h_selected, f_test),
                                        "nmse_harmonic_higher_capacity_selected": metrics.mse(pred_h_selected, f_test) / target_power,
                                        "amp_harmonic_higher_capacity_selected": metrics.target_amplitude(pred_h_selected, f_test),
                                    })
                                rows.append(row)
        print(f"training seed {seed} complete ({seed + 1}/{len(cfg.seeds)})", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    """Aggregate train-seed variation while retaining target realization IDs."""
    keys = ["n", "ell", "target_mode", "target_seed", "N", "tau"]
    metric_cols = [c for c in df.columns if c.startswith(("mse_", "nmse_", "amp_", "corr_", "validation_mse_"))]
    summary = df.groupby(keys)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    references = df.groupby(keys)[[c for c in ["lambda_ell", "g_ell", "d_eff", "selected_tau_krr",
        "selected_tau_harmonic_matched_span", "selected_tau_harmonic_higher_capacity"] if c in df.columns]].first()
    return references.join(summary).reset_index()


def make_plots(summary, out_dir: Path):
    """Create plots for support, regularization, sample size, and baselines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. Recovery as a function of target degree, split by n and N.
    for tau, sub in summary.groupby("tau"):
        agg = sub.groupby(["n", "ell", "N", "target_mode"], as_index=False).agg(
            mse=("mse_krr_mean", "mean"), mse_sd=("mse_krr_std", "mean"),
            amp=("amp_krr_mean", "mean"))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for (n, N, mode), g in agg.groupby(["n", "N", "target_mode"]):
            g = g.sort_values("ell")
            ax.errorbar(g.ell, np.maximum(g.mse, 1e-12), yerr=g.mse_sd.fillna(0),
                        marker="o", ls="-" if N == max(agg.N) else "--", capsize=2,
                        label=f"n={n}, N={N}, {mode}")
        ax.set(xlabel="target degree ℓ", ylabel="test MSE", yscale="log",
               title=f"Exact-kernel KRR recovery by degree (τ={tau:g})")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(out_dir / f"mse_vs_degree_tau{tau:g}.png", dpi=150); plt.close(fig)

    # 2. Compare all three model families at validation-selected regularization.
    cols = ["mse_krr_selected_mean", "mse_harmonic_matched_span_selected_mean",
            "mse_harmonic_higher_capacity_selected_mean"]
    if all(c in summary.columns for c in cols):
        vals = [summary[c].mean() for c in cols]
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["Fidelity-kernel KRR", "Harmonic ridge: degree ≤ n", "Harmonic ridge: degree ≤ max target"]
        ax.bar(labels, np.maximum(vals, 1e-12))
        ax.set(ylabel="mean test MSE (validation-selected τ)", yscale="log",
               title="Model-family comparison; descriptive aggregate")
        ax.tick_params(axis="x", labelrotation=12)
        fig.tight_layout(); fig.savefig(out_dir / "baseline_comparison_selected_tau.png", dpi=150); plt.close(fig)

    # 3. Population shrinkage reference against empirical amplitude.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(summary.g_ell, summary.amp_krr_mean, s=14, alpha=0.45)
    ax.plot([0, 1], [0, 1], "k:", label="amplitude = population shrinkage")
    ax.set(xlabel="population shrinkage gℓ", ylabel="empirical recovery amplitude",
           title="Empirical harmonic recovery vs theoretical shrinkage")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "amplitude_vs_shrinkage.png", dpi=150); plt.close(fig)

    # 4. Training size curves for validation-selected test performance.
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = summary.groupby(["N", "n"], as_index=False).agg(mse=("mse_krr_selected_mean", "mean"))
    for n, g in sample.groupby("n"):
        g = g.sort_values("N")
        ax.plot(g.N, np.maximum(g.mse, 1e-12), marker="o", label=f"n={n}")
    ax.set(xlabel="training size N", ylabel="mean test MSE", yscale="log",
           title="Recovery vs training-set size (validation-selected τ)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "recovery_vs_training_size.png", dpi=150); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Exact-kernel expressivity and recoverability study on S^2")
    p.add_argument("--pilot", action="store_true", help="run only the analytic PennyLane validation pilot")
    p.add_argument("--quick", action="store_true", help="small end-to-end smoke run, not the final study")
    p.add_argument("--no-quantum", action="store_true", help="skip PennyLane analytic validation pilot")
    p.add_argument("--no-baselines", action="store_true", help="skip the two harmonic regression baselines")
    p.add_argument("--copies", type=int, nargs="+", default=list(Config.copies))
    p.add_argument("--degrees", type=int, nargs="+", default=list(Config.degrees))
    p.add_argument("--train-sizes", type=int, nargs="+", default=list(Config.train_sizes))
    p.add_argument("--taus", type=float, nargs="+", default=list(Config.taus))
    p.add_argument("--target-modes", choices=["zonal", "random_harmonic"], nargs="+", default=list(Config.target_modes))
    p.add_argument("--seeds", type=int, nargs="+", default=list(Config.seeds))
    p.add_argument("--target-seeds", type=int, nargs="+", default=list(Config.target_seeds))
    p.add_argument("--n-test", type=int, default=Config.n_test)
    p.add_argument("--n-validation", type=int, default=Config.n_validation)
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(copies=tuple(args.copies), degrees=tuple(args.degrees), train_sizes=tuple(args.train_sizes),
                 taus=tuple(args.taus), target_modes=tuple(args.target_modes), seeds=tuple(args.seeds),
                 target_seeds=tuple(args.target_seeds), n_test=args.n_test, n_validation=args.n_validation,
                 use_quantum=not args.no_quantum, run_baselines=not args.no_baselines)
    if args.quick:
        cfg = Config(copies=(1, 2), degrees=(0, 1, 2), train_sizes=(16, 32), taus=(1e-3, 1e-2),
                     target_modes=("zonal", "random_harmonic"), seeds=(0, 1), target_seeds=(0, 1),
                     n_test=120, n_validation=80, test_seed=cfg.test_seed, validation_seed=cfg.validation_seed,
                     pilot_pairs=8, pilot_gram_size=8, use_quantum=cfg.use_quantum,
                     run_baselines=cfg.run_baselines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({
        "experiment": "spherical_fidelity_kernel_experiment1",
        "research_question": "How do kernel spectral support, regularization, training-set size, and target harmonic composition affect recovery of functions on S^2?",
        "config": asdict(cfg), "environment": environment_info(),
    }, indent=2))
    if cfg.use_quantum:
        if qml is None:
            raise SystemExit("PennyLane is not installed. Install requirements.txt or rerun with --no-quantum.")
        pilot = run_pilot(cfg)
        (args.out / "pilot.json").write_text(json.dumps(pilot, indent=2))
        for r in pilot:
            print(f"analytic circuit pilot n={r['n']}: max Gram error {r['gram_agreement']['max_abs_err']:.2e} "
                  f"{'PASS' if r['passed'] else 'FAIL'}")
        if not all(r["passed"] for r in pilot):
            raise SystemExit("Analytic PennyLane pilot failed: circuit disagrees with exact formula.")
    if args.pilot:
        print(f"Pilot outputs saved to {args.out.resolve()}")
        return
    df = run_grid(cfg)
    df.to_csv(args.out / "results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.out / "summary.csv", index=False)
    make_plots(summary, args.out)
    print(f"Experiment 1 outputs saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
