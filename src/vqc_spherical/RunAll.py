"""
Phase driver.

    python RunAll.py --phase 0     dataset gate, ceilings, circuit structure
    python RunAll.py --phase 1     kernel spectra and the falsification runs
    python RunAll.py --phase 2     both expansion ladders
    python RunAll.py --phase 3     faithfulness and sample efficiency
    python RunAll.py --figures     redraw every figure from results/*.csv
    python RunAll.py --all --quick smoke-test the whole pipeline

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
    "0": ["ValidateDatasets.py", "ExperimentAnalyticCore.py"],
    "1": ["ExperimentKernelSpectrum.py", "ExperimentFalsification.py"],
    "2": ["ExperimentLadders.py"],
    "3": ["ExperimentFaithfulness.py"],
}

# Stages that involve no training run at full scale regardless of --quick:
# they take seconds and carry most of the analytic content.
NO_QUICK = ("ValidateDatasets.py", "ExperimentAnalyticCore.py",
            "ExperimentKernelSpectrum.py")

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, quick, extra=()):
    cmd = [sys.executable, os.path.join(HERE, script)]
    if quick and script not in NO_QUICK:
        cmd.append("--quick")
    cmd += list(extra)
    print(f"\n{'=' * 70}\n  {' '.join(os.path.basename(c) for c in cmd)}\n{'=' * 70}")
    return subprocess.call(cmd, cwd=HERE)


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
        subprocess.call([sys.executable, os.path.join(HERE, "Plots.py")], cwd=HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
