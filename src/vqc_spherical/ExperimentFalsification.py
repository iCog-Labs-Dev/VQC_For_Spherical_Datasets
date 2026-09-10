"""
Falsification first.  Run before anything else: it is cheap, and it is the only
experiment that can prove the analysis wrong.

The degree-2 wall.  The prediction is NOT "the circuit fails at 50-65%".  It is
that the circuit saturates at the target's computed affine ceiling -- 0.786 on
the quadrupole, 0.669 on the sectoral target -- and does not exceed it at any
depth, width, learning rate or seed.  The 50-65% band was wrong twice over: it
sits below the real ceiling, and it straddles the majority-class baseline of
0.575, so it could not have distinguished an expressivity limit from a model
that always guesses the common class.

Depth inertness.  On a degree-1 target, accuracy is flat in n_layers.

Every table carries five rows: VQC, BestAffine (the computed ceiling),
LogisticOnCartesian, HarmonicRegression(L=1), and Majority.

Read the sincos row with care.  LogisticOnSinCos scores 1.00 on the quadrupole,
because sin(theta) is one of its features and |z| > 1/sqrt(3) is exactly
sin(theta) < sqrt(2/3).  The torus representation is NOT degree-limited: it is
strictly more expressive than the circuit here while being less faithful.  That
is the honest story -- the encoding buys a prior, not power.

Usage:  python ExperimentFalsification.py [--quick]
"""
import argparse
import warnings

import numpy as np

try:
    from . import ClassicalBaselines as B, Config as C, dataUtil as D, ExperimentUtils as U
except ImportError:  # pragma: no cover - flat sys.path
    import ClassicalBaselines as B, Config as C, dataUtil as D, ExperimentUtils as U

warnings.filterwarnings("ignore")

TARGETS = {"quadrupole": 2, "sectoral": 2, "latitude_bands": 1}


def run(targets, depths, seeds, stepsizes, epochs, n_samples, n_directions):
    rows = []
    for tname in targets:
        degree = TARGETS[tname]
        print(f"\n=== {tname}  (degree {degree}) " + "=" * 40)
        th, ph, y = D.make_target(tname, n_samples=C.N_CEILING, seed=99)
        ceiling = B.BestAffine(n_directions).fit(th, ph, y).ceiling_
        print(f"affine ceiling (population, n={C.N_CEILING}): {ceiling:.4f}\n")
        print(f"{'model':26s} {'params':>6s} {'train':>7s} {'test':>7s} "
              f"{'balanced':>9s} {'sd':>7s}")
        print("-" * 68)

        for L in depths:
            for lr in stepsizes:
                runs = []
                for s in seeds:
                    sd = U.seeds(s)
                    th, ph, y = D.make_target(tname, n_samples=n_samples, seed=sd["data"])
                    tr, te = U.split(th, ph, y, sd["split"])
                    r = U.train_vqc(tr, te, n_qubits=2, n_layers=L, n_uploads=1,
                                    seed_init=sd["init"], epochs=epochs, stepsize=lr)
                    runs.append(r)
                    rows.append(U.blank_row(
                        experiment="falsification" if degree > 1 else "depth-inertness",
                        target=tname, representation="bloch", model="VQC",
                        encoding="spherical", n_uploads=1, n_qubits=2, n_layers=L,
                        n_params=r["n_params"], n_train=len(tr[2]), seed=sd["init"],
                        stepsize=lr, train_acc=r["train_acc"], test_acc=r["test_acc"],
                        balanced_acc=r["balanced_acc"],
                        majority=U.majority_rate(tr[2]), ceiling=ceiling,
                        final_loss=r["final_loss"], wall_time=r["wall_time"]))
                te_m = float(np.mean([a["test_acc"] for a in runs]))
                sd_ = float(np.std([a["test_acc"] for a in runs]))
                flag = "  <-- ABOVE CEILING" if te_m > ceiling + 3 * sd_ + 0.02 else ""
                print(f"{'VQC L=%d lr=%.2f' % (L, lr):26s} {runs[0]['n_params']:6d} "
                      f"{np.mean([a['train_acc'] for a in runs]):7.3f} {te_m:7.3f} "
                      f"{np.mean([a['balanced_acc'] for a in runs]):9.3f} {sd_:7.3f}{flag}")

        sd = U.seeds(list(seeds)[0])
        th, ph, y = D.make_target(tname, n_samples=n_samples, seed=sd["data"])
        tr, te = U.split(th, ph, y, sd["split"])
        for M in [B.BestAffine(n_directions), B.LogisticOnCartesian(),
                  B.LogisticOnSinCos(), B.HarmonicRegression(1),
                  B.HarmonicRegression(2), B.MajorityBaseline()]:
            r = U.run_classical(M, tr, te)
            print(f"{M.name:26s} {r['n_params']:6d} {r['train_acc']:7.3f} "
                  f"{r['test_acc']:7.3f} {r['balanced_acc']:9.3f}")
            rows.append(U.blank_row(experiment="falsification-control", target=tname,
                                    model=M.name, n_params=r["n_params"],
                                    n_train=len(tr[2]), seed=sd["split"],
                                    train_acc=r["train_acc"], test_acc=r["test_acc"],
                                    balanced_acc=r["balanced_acc"],
                                    majority=U.majority_rate(tr[2]), ceiling=ceiling))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    if a.quick:
        rows = run(["quadrupole"], (2,), range(2), (0.2,), 30, 160, 512)
    else:
        rows = run(list(TARGETS), C.DEPTHS, range(a.seeds or C.N_SEEDS),
                   C.STEPSIZES, C.EPOCHS, C.N_SAMPLES, C.N_DIRECTIONS)
    U.write_rows("falsification.csv", rows)


if __name__ == "__main__":
    main()
