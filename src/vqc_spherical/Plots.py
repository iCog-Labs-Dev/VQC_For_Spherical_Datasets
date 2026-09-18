"""
All report figures, redrawn from results/*.csv alone.

No experiment ever draws a figure: runners write CSV, this module draws.  A
plot can then be restyled a dozen times without re-running a circuit.

Palette
-------
Four categorical slots -- blue, orange, aqua, violet -- validated as a set for
all-pairs colour-vision separation on the light surface (worst CVD deltaE 9.2,
worst normal-vision deltaE 16.3).  Aqua sits below 3:1 contrast on this
surface, so every series also carries a distinct marker and a direct label:
identity is never colour alone.

These are static figures for a printed report, so they deliberately commit to
the light surface only.  Sequential encodings use a single blue ramp, light to
dark -- never a rainbow.

Usage:  python -m vqc_spherical.Plots [--only datasets|spectrum|harmonic|staircase|contrast|faithfulness|curves]
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import ExperimentUtils as U

SERIES = {"bloch": "#2a78d6", "cartesian": "#eb6834",
          "sincos": "#1baf7a", "raw": "#4a3aa7"}
MARKERS = {"bloch": "o", "cartesian": "s", "sincos": "^", "raw": "D"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORDINAL = SEQ[1:]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "font.family": "sans-serif",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.9)
    ax.set_axisbelow(True)


def _read(name):
    path = os.path.join(U.RESULTS, name)
    if not os.path.exists(path):
        print(f"  (skipped: results/{name} not found -- run its experiment first)")
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig, name):
    os.makedirs(U.FIGURES, exist_ok=True)
    path = os.path.join(U.FIGURES, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}")


def _label(ax, x, y, text):
    """Direct labels in ink, not the series colour -- the mark carries identity."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                color=INK2, fontsize=8, va="center")


def fig_datasets():
    """
    Every dataset drawn on the sphere it lives on, the deprecated leaking one
    first so the flaw is visible beside its replacements.
    """
    from . import dataUtil as D

    order = ["sphere_moons", "latitude_bands", "tilted_bands",
             "quadrupole", "sectoral", "hyperbolic"]
    titles = {"sphere_moons": "sphere_moons - LEAKS into longitude",
              "latitude_bands": "latitude_bands - control, theta only",
              "tilted_bands": "tilted_bands - needs both coordinates",
              "quadrupole": "quadrupole - degree 2, ceiling 0.79",
              "sectoral": "sectoral - degree 2, ceiling 0.67",
              "hyperbolic": "hyperbolic - degree 2, both marginals at chance"}

    fig = plt.figure(figsize=(10.5, 5.8))
    for i, name in enumerate(order):
        theta, phi, y = D.make_target(name, n_samples=700, seed=11)
        xyz = D.sphere_to_cartesian(theta, phi)
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        u = np.linspace(0, 2 * np.pi, 44)
        v = np.linspace(0, np.pi, 22)
        ax.plot_wireframe(0.985 * np.outer(np.cos(u), np.sin(v)),
                          0.985 * np.outer(np.sin(u), np.sin(v)),
                          0.985 * np.outer(np.ones_like(u), np.cos(v)),
                          color=GRID, alpha=0.5, linewidth=0.3)
        for cls, colour, marker in ((0, SERIES["bloch"], "o"),
                                    (1, SERIES["cartesian"], "^")):
            m = y == cls
            ax.scatter(xyz[m, 0], xyz[m, 1], xyz[m, 2], s=7, marker=marker,
                       color=colour, alpha=0.85, linewidths=0, depthshade=False)
        ax.set_title(titles[name], fontsize=8.5, color=INK, pad=2)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=35)
        for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            lim(-0.62, 0.62)
        ax.set_axis_off()
    fig.text(0.5, 0.015, "circles: class 0     triangles: class 1",
             ha="center", color=INK2, fontsize=8.5)
    fig.subplots_adjust(wspace=0.02, hspace=0.06, top=0.96, bottom=0.06)
    _save(fig, "fig_datasets.png")


def fig_spectrum():
    """Rank-4 spectrum on a log axis.  The cliff after four is the point."""
    from . import Analysis as A, Kernels as K
    theta, phi = A.sample_sphere(600, seed=1)
    n = len(theta)
    ev = np.maximum(K.gram_spectrum(theta, phi, K.bloch_kernel) / n, 1e-18)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogy(range(1, 13), ev[:12], marker="o", markersize=7,
                color=SERIES["bloch"], linestyle="none")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.axhline(1 / 6, color=MUTED, lw=1, ls=":")
    ax.annotate("lambda_0 = 1/2", (1.15, 0.5), xytext=(0, 9),
                textcoords="offset points", color=INK2, fontsize=8)
    ax.annotate("lambda_1 = 1/6, threefold degenerate in m", (2.0, 1 / 6),
                xytext=(0, -16), textcoords="offset points", color=INK2, fontsize=8)
    ax.axvspan(4.5, 12.5, color=GRID, alpha=0.5, zorder=0)
    ax.annotate("below here is numerical noise:\nthe kernel has rank 4",
                (8.5, float(np.median(ev[4:12]))), xytext=(0, 26),
                textcoords="offset points", color=INK2, fontsize=8, ha="center")
    _style(ax, "Gram spectrum of the fidelity kernel: exactly four nonzero eigenvalues",
           "eigenvalue index", "eigenvalue / n")
    ax.set_xlim(0.3, 12.9)
    ax.set_ylim(1e-17, 6.0)
    _save(fig, "fig_kernel_spectrum.png")


def fig_harmonic():
    """
    Residual energy above each rung's own degree.  The broadcast ladder sits at
    floating-point zero; re-uploading does not.  Dots rather than bars: a bar's
    length is only honest against a true zero, and a log axis has none.
    """
    rows = _read("ladder.csv")
    if not rows:
        return
    spin = sorted([r for r in rows if r["model"].startswith("spin_")],
                  key=lambda r: int(r["n_qubits"]))
    reup = sorted([r for r in rows if r["model"].startswith("reupload_")],
                  key=lambda r: int(r["n_uploads"]))
    if not spin or not reup:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(spin))
    y_spin = np.array([float(r["test_acc"]) for r in spin])
    y_reup = np.array([float(r["test_acc"]) for r in reup])[:len(spin)]
    ax.set_yscale("log")
    dx = 0.09
    for xi, a, b in zip(x, y_spin, y_reup):
        ax.plot([xi - dx, xi + dx], [a, b], color=GRID, lw=1.5, zorder=0)
    ax.plot(x - dx, y_spin, "o", markersize=9, color=SERIES["bloch"],
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="spin ladder (broadcast)")
    ax.plot(x + dx, y_reup, "s", markersize=9, color=SERIES["cartesian"],
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="re-uploading ladder")
    ax.set_xticks(x, [f"rung {i + 1}" for i in x])
    ax.axhline(1e-10, color=MUTED, lw=1, ls=":")
    ax.annotate("machine precision", (x[0] - 0.35, 1e-10), xytext=(0, 6),
                textcoords="offset points", color=INK2, fontsize=8)
    ax.annotate("exactly degree-limited", (x[-1] - dx, y_spin[-1]), xytext=(0, 14),
                textcoords="offset points", color=INK2, fontsize=8, ha="right")
    ax.annotate("not degree-limited", (x[-1] + dx, y_reup[-1]), xytext=(0, -18),
                textcoords="offset points", color=INK2, fontsize=8, ha="right")
    _style(ax, "Energy left outside ell <= rung: the two ladders are not the same knob",
           None, "relative residual at ell <= rung")
    ax.set_xlim(-0.6, len(spin) - 0.4)
    ax.legend(frameon=False, loc="center left", labelcolor=INK2)
    _save(fig, "fig_harmonic_residual.png")


def fig_contrast():
    """
    k versus gamma for both ladders.  The spin ladder collapses onto one
    monotone curve; the re-uploading model does not.  The scatter IS the
    finding, so it is drawn rather than averaged.
    """
    path = os.path.join(U.RESULTS, "kernel_contrast.csv")
    if not os.path.exists(path):
        print("  (skipped: results/kernel_contrast.csv not found)")
        return
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, ladder, title in [(axes[0], "spin", "Spin ladder: a fixed geodesic kernel"),
                              (axes[1], "reupload", "Re-uploading: not a function of gamma")]:
        sub = [r for r in rows if r["ladder"] == ladder]
        rungs = sorted({int(r["rung"]) for r in sub})
        shown = rungs if ladder == "spin" else rungs[1:2]
        for k, ru in enumerate(shown):
            g = np.array([float(r["gamma"]) for r in sub if int(r["rung"]) == ru])
            v = np.array([float(r["k"]) for r in sub if int(r["rung"]) == ru])
            o = np.argsort(g)
            if ladder == "spin":
                ax.plot(g[o], v[o], color=SEQ_ORDINAL[min(k * 2, len(SEQ_ORDINAL) - 1)], lw=2)
                level = [0.78, 0.62, 0.46, 0.30][k % 4]
                j = int(np.argmin(np.abs(v[o] - level)))
                _label(ax, g[o][j], level, f"n={ru}")
            else:
                ax.plot(g[o], v[o], ".", color=SERIES["cartesian"], markersize=3, alpha=0.45)
                ax.annotate(f"U = {ru}: every gamma carries a spread of k,\n"
                            f"so k is not a function of gamma alone",
                            (0.06, 0.06), xycoords="axes fraction",
                            color=INK2, fontsize=8)
        _style(ax, title, "geodesic angle gamma", "kernel value k")
        ax.set_xlim(0, np.pi)
        ax.set_xticks([0, np.pi / 2, np.pi], ["0", "pi/2", "pi"])
    _save(fig, "fig_kernel_contrast.png")


def _series_plot(sub, xkey, xlabel, title, filename, logx=False):
    reps = [r for r in SERIES if any(x["representation"] == r for x in sub)]
    if not reps:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for rep in reps:
        pts = defaultdict(list)
        for r in sub:
            if r["representation"] == rep:
                pts[float(r[xkey])].append(float(r["balanced_acc"]))
        xs = sorted(pts)
        m = np.array([np.mean(pts[x]) for x in xs])
        s = np.array([np.std(pts[x]) for x in xs])
        ax.fill_between(xs, m - s, m + s, color=SERIES[rep], alpha=0.14, linewidth=0)
        ax.plot(xs, m, color=SERIES[rep], marker=MARKERS[rep], markersize=7,
                label=rep, markeredgecolor=SURFACE, markeredgewidth=1.5)
        _label(ax, xs[-1], m[-1], rep)
    if logx:
        ax.set_xscale("log")
    _style(ax, title, xlabel, "balanced accuracy")
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right", ncols=2)
    _save(fig, filename)


def fig_faithfulness():
    rows = _read("faithfulness.csv")
    if not rows:
        return
    sub = [r for r in rows if r["experiment"].startswith("faithfulness")
           and r["polar_angle"] not in ("", "nan")]
    if sub:
        _series_plot(sub, "polar_angle", "rotation angle (rad)",
                     "Faithfulness: identical task, rotated frame",
                     "fig_faithfulness.png")


def fig_curves():
    rows = _read("faithfulness.csv")
    if not rows:
        return
    sub = [r for r in rows if r["experiment"] == "sample-efficiency"]
    if sub:
        _series_plot(sub, "n_train", "training samples",
                     "Sample efficiency at a polar placement",
                     "fig_learning_curves.png", logx=True)


def fig_staircase():
    """Rung x target, balanced accuracy as colour.  One blue ramp, never a rainbow."""
    rows = _read("ladder.csv")
    if not rows:
        return
    for arm, label in [("ladder-reupload-fixed_L", "re-uploading (L fixed)"),
                       ("ladder-reupload-fixed_budget", "re-uploading (U*L fixed)"),
                       ("ladder-spin", "spin ladder")]:
        sub = [r for r in rows if r["experiment"] == arm and r["target"]]
        if not sub:
            continue
        key = "n_qubits" if arm == "ladder-spin" else "n_uploads"
        rungs = sorted({int(r[key]) for r in sub})
        targets = sorted({r["target"] for r in sub})
        acc = defaultdict(list)
        for r in sub:
            acc[(int(r[key]), r["target"])].append(float(r["balanced_acc"]))
        grid = np.full((len(rungs), len(targets)), np.nan)
        for i, ru in enumerate(rungs):
            for j, t in enumerate(targets):
                if acc[(ru, t)]:
                    grid[i, j] = np.mean(acc[(ru, t)])

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
        fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(targets), 1.0 + 0.7 * len(rungs)))
        im = ax.imshow(grid, cmap=cmap, vmin=0.5, vmax=1.0, origin="lower", aspect="auto")
        ax.set_xticks(range(len(targets)), targets, rotation=15, ha="right")
        ax.set_yticks(range(len(rungs)), [f"{key.split('_')[1]}={r}" for r in rungs])
        for i in range(len(rungs)):
            for j in range(len(targets)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="#ffffff" if grid[i, j] > 0.78 else INK)
        _style(ax, f"Learnability staircase -- {label}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="balanced accuracy")
        _save(fig, f"fig_staircase_{arm.replace('-', '_')}.png")


ALL = {"datasets": fig_datasets, "spectrum": fig_spectrum, "harmonic": fig_harmonic,
       "staircase": fig_staircase, "contrast": fig_contrast,
       "faithfulness": fig_faithfulness, "curves": fig_curves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(ALL))
    a = ap.parse_args()
    for name, fn in ALL.items():
        if a.only and name != a.only:
            continue
        print(f"[{name}]")
        fn()


if __name__ == "__main__":
    main()
