"""
Phase driver.

    python -m vqc_spherical.RunAll --phase 0
    python -m vqc_spherical.RunAll --phase 1
    python -m vqc_spherical.RunAll --phase 2
    python -m vqc_spherical.RunAll --phase 3
    python -m vqc_spherical.RunAll --figures
    python -m vqc_spherical.RunAll --all --quick

Order matters.  Phase 0 begins with ValidateDatasets.py, which refuses to let a
dataset into the pipeline unless it does what its declared role requires; that
check is what catches a label leaking into one coordinate before it reaches a
figure.  A non-zero exit there stops the run.

Phase 0 is then the theory gate: if the affine residual or the Gram spectrum
disagrees with the closed forms, the derivation is wrong and no amount of
training will rescue it.  Phase 1 is the second gate: if any configuration
beats its target's computed ceiling, stop and re-derive before spending another
week.
"""
import argparse
import os
import subprocess
import sys

PHASES = {
    "0": ["ValidateDatasets", "ExperimentAnalyticCore"],
    "1": ["ExperimentKernelSpectrum", "ExperimentFalsification"],
    "2": ["ExperimentLadders"],
    "3": ["ExperimentFaithfulness"],
}

# Stages that involve no training run at full scale regardless of --quick:
# they take seconds and carry most of the analytic content.
NO_QUICK = ("ValidateDatasets", "ExperimentAnalyticCore",
            "ExperimentKernelSpectrum")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def run(module, quick, extra=()):
    cmd = [sys.executable, "-m", f"vqc_spherical.{module}"]
    if quick and module not in NO_QUICK:
        cmd.append("--quick")
    cmd += list(extra)
    print(f"\n{'=' * 70}\n  {' '.join(cmd)}\n{'=' * 70}")
    return subprocess.call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=sorted(PHASES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()

    scripts = []
    if a.all:
        for k in sorted(PHASES):
            scripts += PHASES[k]
    elif a.phase:
        scripts = PHASES[a.phase]

    extra = ["--seeds", str(a.seeds)] if a.seeds else []
    for s in scripts:
        code = run(s, a.quick, extra if s not in NO_QUICK else ())
        if code != 0:
            print(f"\n{s} exited {code} -- stopping.")
            return code

    if a.figures or a.all:
        subprocess.call(
            [sys.executable, "-m", "vqc_spherical.Plots"], cwd=ROOT
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
