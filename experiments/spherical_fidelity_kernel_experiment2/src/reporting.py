"""Aggregation, CSV/JSON export and figures."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

GROUP_KEYS = ["plan", "allocation", "budget", "shots_nominal", "n", "N", "target", "condition"]
SUMMARY_METRICS = ["test_mse", "nmse", "corr", "coef_err", "coef_err_rel",
                   "test_mse_delta", "nmse_delta", "coef_err_delta",
                   "selected_lambda", "total_shots"]
SAMPLED = ["sampled_exact", "exact_sampled", "sampled_sampled"]
COLORS = {"exact_exact": "black", "sampled_exact": "tab:blue",
          "exact_sampled": "tab:orange", "sampled_sampled": "tab:red"}


def add_paired_deltas(df):
    """Add metric differences against exact_exact on the same (n, N, target, data seed)."""
    keys = ["n", "N", "target", "data_seed"]
    base = df.loc[df.condition == "exact_exact", keys + ["test_mse", "nmse", "coef_err"]]
    base = base.rename(columns={m: f"{m}_exact" for m in ["test_mse", "nmse", "coef_err"]})
    out = df.merge(base, on=keys, how="left")
    for m in ["test_mse", "nmse", "coef_err"]:
        out[f"{m}_delta"] = out[m] - out[f"{m}_exact"]
    return out


def summarize(df):
    """Per-group summary with data-split and shot-seed uncertainty kept separate.

    mean: average of per-data-seed means. sd_data / ci95: spread across data seeds
    (t-interval on the seed means). sd_shot: mean within-data-seed std across shot seeds.
    """
    g = df.groupby(GROUP_KEYS + ["data_seed"])
    seed_mean = g[SUMMARY_METRICS].mean().reset_index()
    seed_std = g[SUMMARY_METRICS].std(ddof=1).reset_index()
    seed_cnt = g.size().rename("n_shot").reset_index()
    rows = []
    for key, grp in seed_mean.groupby(GROUP_KEYS):
        idx = grp.index
        n_data = len(grp)
        tcrit = stats.t.ppf(0.975, n_data - 1) if n_data > 1 else np.nan
        row = dict(zip(GROUP_KEYS, key))
        row["n_data"] = n_data
        row["n_shot"] = int(seed_cnt.loc[idx, "n_shot"].iloc[0])
        for m in SUMMARY_METRICS:
            sd = grp[m].std(ddof=1) if n_data > 1 else np.nan
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_sd_data"] = sd
            row[f"{m}_ci95"] = tcrit * sd / np.sqrt(n_data) if n_data > 1 else np.nan
            row[f"{m}_sd_shot"] = seed_std.loc[idx, m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _footer(fig, meta):
    fig.text(0.005, 0.003, meta, fontsize=6, color="gray")


def plot_error_vs_shots(summary, outdir, meta):
    s = summary[summary.condition.isin(["exact_exact"] + SAMPLED)]
    for size in sorted(s.N.unique()):
        d = s[s.N == size]
        ns, targets = sorted(d.n.unique()), list(d.target.unique())
        shots = d.loc[d.condition.isin(SAMPLED), "shots_nominal"]
        fig, axes = plt.subplots(len(ns), len(targets), figsize=(3 * len(targets), 2.4 * len(ns)),
                                 squeeze=False, sharex=True)
        for i, n in enumerate(ns):
            for j, tgt in enumerate(targets):
                ax = axes[i][j]
                cell = d[(d.n == n) & (d.target == tgt)]
                lows, highs = [], []
                for cond in SAMPLED:
                    c = cell[cell.condition == cond].sort_values("shots_nominal")
                    if len(c):
                        ci = c.nmse_ci95.fillna(0)
                        ax.errorbar(c.shots_nominal, c.nmse_mean, yerr=ci,
                                    color=COLORS[cond], marker="o", ms=3, lw=1, label=cond)
                        lows += list(np.maximum(c.nmse_mean - ci, c.nmse_mean / 3))
                        highs += list(c.nmse_mean + ci)
                exact = cell[cell.condition == "exact_exact"]
                if len(exact) and lows:
                    e = exact.nmse_mean.iloc[0]
                    lo, hi = min(lows) / 1.5, max(highs) * 1.5
                    ax.set_ylim(lo, hi)
                    if lo <= e <= hi:
                        ax.axhline(e, color="black", ls="--", lw=1)
                    else:
                        ax.text(0.03, 0.04, f"exact: {e:.1e}", transform=ax.transAxes, fontsize=6)
                ax.set_xlim(shots.min() / 1.5, shots.max() * 1.5)
                ax.set_xscale("log")
                ax.set_yscale("log")
                if i == 0:
                    ax.set_title(tgt, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"n={n}\nnormalized MSE", fontsize=7)
                if i == len(ns) - 1:
                    ax.set_xlabel("shots per evaluation", fontsize=7)
                ax.tick_params(labelsize=6)
        axes[0][0].legend(fontsize=6)
        fig.suptitle(f"Test error vs shots (N={size}); dashed line = exact kernel", fontsize=9)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        _footer(fig, meta)
        fig.savefig(Path(outdir) / f"error_vs_shots_N{size}.png", dpi=150)
        plt.close(fig)


def plot_degree_profile(summary, outdir, meta, target_degrees):
    s = summary[summary.target.isin(target_degrees)]
    sampled = s[s.condition.isin(SAMPLED)]
    if sampled.empty:
        return
    size = s.N.max()
    top = sampled[(sampled.N == size) & (sampled.allocation.isin(["uniform"]))]
    if top.empty:
        top = sampled[sampled.N == size]
    plan = top.loc[top.shots_nominal.idxmax(), "plan"]
    d = pd.concat([s[(s.N == size) & (s.condition == "exact_exact")],
                   s[(s.N == size) & (s.plan == plan)]])
    d = d.assign(degree=d.target.map(target_degrees))
    ns = sorted(d.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3 * len(ns), 2.8), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], ns):
        for cond in ["exact_exact"] + SAMPLED:
            c = d[(d.n == n) & (d.condition == cond)].sort_values("degree")
            if len(c):
                ax.errorbar(c.degree, c.nmse_mean, yerr=c.nmse_ci95.fillna(0), marker="o", ms=3,
                            lw=1, color=COLORS[cond], label=cond)
        ax.set_yscale("log")
        ax.set_title(f"n={n}", fontsize=8)
        ax.set_xlabel("target degree", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel("normalized MSE", fontsize=7)
    axes[0][0].legend(fontsize=6)
    fig.suptitle(f"Error by harmonic degree (N={size}, plan {plan})", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _footer(fig, meta)
    fig.savefig(Path(outdir) / "error_by_degree.png", dpi=150)
    plt.close(fig)


def plot_exact_vs_sampled(summary, outdir, meta):
    keys = ["n", "N", "target"]
    base = summary[summary.condition == "exact_exact"][keys + ["nmse_mean"]]
    base = base.rename(columns={"nmse_mean": "nmse_exact"})
    d = summary[summary.condition.isin(SAMPLED)].merge(base, on=keys)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for cond in SAMPLED:
        c = d[d.condition == cond]
        ax.scatter(c.nmse_exact, c.nmse_mean, s=8, alpha=0.6, color=COLORS[cond], label=cond)
    lims = [d[["nmse_exact", "nmse_mean"]].min().min(), d[["nmse_exact", "nmse_mean"]].max().max()]
    ax.plot(lims, lims, color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("exact-kernel normalized MSE", fontsize=8)
    ax.set_ylabel("sampled-kernel normalized MSE", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("Exact vs sampled (each point = one configuration)", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _footer(fig, meta)
    fig.savefig(Path(outdir) / "exact_vs_sampled.png", dpi=150)
    plt.close(fig)


def plot_budget_performance(summary, outdir, meta):
    d = summary[summary.condition.isin(SAMPLED)]
    if d.empty:
        return
    sizes = sorted(d.N.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 3.2), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        sub = d[d.N == size]
        grouped = sub.groupby(["condition", "allocation", "plan", "n"], as_index=False).agg(
            shots=("total_shots_mean", "mean"), nmse=("nmse_mean", "mean"))
        for (cond, alloc, n), c in grouped.groupby(["condition", "allocation", "n"]):
            c = c.sort_values("shots")
            style = "-" if alloc == "uniform" else "--"
            ax.plot(c.shots, c.nmse, style, marker="o", ms=3, lw=1, color=COLORS[cond],
                    alpha=0.35 + 0.65 * (n == sub.n.max()),
                    label=f"{cond}, {alloc}, n={n}" if n == sub.n.max() else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"N={size}", fontsize=8)
        ax.set_xlabel("total shots", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel("normalized MSE (mean over targets)", fontsize=7)
    axes[0][0].legend(fontsize=5)
    fig.suptitle("Error vs total shots", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _footer(fig, meta)
    fig.savefig(Path(outdir) / "budget_performance.png", dpi=150)
    plt.close(fig)


def export(rows, diag_rows, manifest, cfg, outdir):
    """Write manifest, per-run metrics, summary, diagnostics and figures to outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = add_paired_deltas(pd.DataFrame(rows))
    summary = summarize(df)
    df.to_csv(outdir / "metrics_per_run.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(outdir / "kernel_diagnostics.csv", index=False)

    manifest = {**manifest, "rows_written": len(df), "diagnostic_rows_written": len(diag_rows)}
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    meta = (f"config={cfg.name} backend={cfg.measurement_backend} "
            f"commit={manifest.get('commit')} {manifest.get('timestamp_utc')}")
    degrees = {}
    for spec in cfg.targets:
        if len(spec["terms"]) == 1:
            degrees[spec["name"]] = spec["terms"][0][0]
    if cfg.budget_mode == "per_evaluation":
        plot_error_vs_shots(summary, outdir, meta)
    plot_degree_profile(summary, outdir, meta, degrees)
    plot_exact_vs_sampled(summary, outdir, meta)
    plot_budget_performance(summary, outdir, meta)
    return df, summary
