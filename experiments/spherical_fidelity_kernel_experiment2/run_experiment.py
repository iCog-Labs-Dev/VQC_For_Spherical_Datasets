"""Command-line runner: python run_experiment.py --config default"""
import argparse
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from experiment_config import CONFIGS, get_config, with_overrides
from src.budget import build_plans
from src.experiment import run_context
from src.reporting import export

ROOT = Path(__file__).resolve().parent


def _context(args):
    return run_context(*args)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions(cfg):
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}
    if cfg.measurement_backend == "pennylane_simulator":
        import pennylane
        info["pennylane"] = pennylane.__version__
    return info


def planned_rows(cfg, contexts):
    """Expected number of metric rows, used to confirm every planned run was recorded."""
    sampled = [c for c in cfg.conditions if c != "exact_exact"]
    total = 0
    for _, size, _ in contexts:
        plans = len(build_plans(cfg, size))
        per_target = int("exact_exact" in cfg.conditions) + plans * len(cfg.shot_seeds) * len(sampled)
        total += per_target * len(cfg.targets)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Finite-shot spherical fidelity kernel experiment")
    parser.add_argument("--config", default="default", choices=sorted(CONFIGS))
    parser.add_argument("--backend", choices=["bernoulli_reference", "pennylane_simulator"],
                        help="override the config's measurement backend")
    parser.add_argument("--output-dir", help="default: results/<config>")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    args = parser.parse_args(argv)

    cfg = get_config(args.config)
    if args.backend:
        cfg = with_overrides(cfg, measurement_backend=args.backend)
    outdir = Path(args.output_dir) if args.output_dir else ROOT / "results" / cfg.name
    contexts = list(product(cfg.copy_counts, cfg.train_sizes, cfg.data_seeds))
    expected = planned_rows(cfg, contexts)

    start = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_context, [(cfg, *c) for c in contexts]))
    else:
        results = []
        for i, c in enumerate(contexts, 1):
            results.append(run_context(cfg, *c))
            print(f"[{i}/{len(contexts)}] n={c[0]} N={c[1]} data_seed={c[2]}", flush=True)
    rows = [r for res in results for r in res[0]]
    diag_rows = [d for res in results for d in res[1]]
    if len(rows) != expected:
        raise RuntimeError(f"recorded {len(rows)} rows but planned {expected}")

    manifest = {
        "config": cfg.to_dict(),
        "measurement_backend": cfg.measurement_backend,
        "budget_mode": cfg.budget_mode,
        "versions": versions(cfg),
        "commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "planned_contexts": len(contexts),
        "planned_rows": expected,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    export(rows, diag_rows, manifest, cfg, outdir)
    print(f"wrote {len(rows)} rows to {outdir}")


if __name__ == "__main__":
    main()
