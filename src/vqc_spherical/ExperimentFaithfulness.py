"""
Faithfulness, measured by rotation conjugation.

Generate the dataset ONCE, then rotate it by R in SO(3) and re-express it in
(theta, phi).  Every geodesic distance is preserved exactly, so the task is
literally identical at every placement and any variation in accuracy is purely
representational.  Faithfulness is then the VARIANCE across placements, not the
mean.

Two INDEPENDENT axes, which must not be collapsed into one sweep because a
representation can pass one and fail the other:

  --axis tilt    rotates the cut away from the lines of latitude, so the
                 boundary is sin(tau) sin(theta) cos(phi - psi)
                 + cos(tau) cos(theta) = c.  Tests whether a representation can
                 COMBINE coordinates.
  --axis polar   carries the configuration toward a pole, where phi is
                 degenerate -- all longitudes name one point.  Tests whether a
                 representation respects the TOPOLOGY.
  --axis haar    Haar-random rotations, to show an effect is not an artefact of
                 one special axis.

The classifier is held fixed structurally: FFNNOnRepresentation takes the
representation as a constructor argument, and every representation gets the same
architecture, optimiser, epoch budget and seed.

The train/test gap is reported beside accuracy: a representation that can fit
structure the data does not contain overfits before it misclassifies.

Usage:  python ExperimentFaithfulness.py [--quick] [--axis tilt|polar|haar]
"""
import argparse
import warnings

import numpy as np

try:
    from . import ClassicalBaselines as B, Config as C, dataUtil as D, ExperimentUtils as U
except ImportError:  # pragma: no cover - flat sys.path
    import ClassicalBaselines as B, Config as C, dataUtil as D, ExperimentUtils as U

warnings.filterwarnings("ignore")


def placements(n_random, axis="polar"):
    if axis == "tilt":
        return [(f"tilt_{i}", float(a), D.rotation_z(0.7) @ D.rotation_y(a))
                for i, a in enumerate(C.POLAR_ANGLES)]
    if axis == "haar":
        return [(f"haar_{j}", float("nan"), D.random_rotation(C.ROTATION_SEED_BASE + j))
                for j in range(n_random)]
    return [(f"polar_{i}", float(a), D.rotation_y(a))
            for i, a in enumerate(C.POLAR_ANGLES)]


def _make(generator, seed):
    return D.make_target(generator, n_samples=_make.n, seed=seed,
                         noise_std=C.BAND_NOISE,
                         latitude_center=np.pi / 2 - C.BAND_SEPARATION / 2,
                         separation=C.BAND_SEPARATION)


def sweep(seeds, n_samples, epochs, n_random, reps=C.REPRESENTATIONS,
          axis="polar", generator="latitude_bands", n_train=None):
    D.require_clean(generator)
    _make.n = n_samples
    rows = []
    print(f"Faithfulness -- axis = {axis}, generator = {generator} "
          f"(identical task, rotated frame)\n")
    print(f"{'placement':14s} {'angle':>7s}  " + "  ".join(f"{r:>10s}" for r in reps))
    print("-" * (24 + 12 * len(reps)))

    per_rep = {r: [] for r in reps}
    gap_rep = {r: [] for r in reps}
    for pname, angle, R in placements(n_random, axis=axis):
        line = []
        for rep in reps:
            accs, gaps = [], []
            for s in seeds:
                sd = U.seeds(s)
                th, ph, y = _make(generator, sd["data"])
                th, ph = D.rotate_dataset(th, ph, R)
                tr, te = U.split(th, ph, y, sd["split"], n_train=n_train)
                if rep == "bloch":
                    r = U.train_vqc(tr, te, n_qubits=2, n_layers=C.DEPTH_DEFAULT,
                                    seed_init=sd["init"], epochs=epochs, stepsize=0.2)
                else:
                    r = U.run_classical(B.FFNNOnRepresentation(rep, seed=sd["init"]),
                                        tr, te)
                accs.append(r["balanced_acc"])
                gaps.append(r["train_acc"] - r["test_acc"])
                rows.append(U.blank_row(
                    experiment=f"faithfulness-{axis}", target=generator,
                    representation=rep,
                    model="VQC" if rep == "bloch" else f"FFNN({rep})",
                    n_params=r["n_params"], n_train=len(tr[2]), rotation_id=pname,
                    polar_angle=angle, seed=sd["init"], train_acc=r["train_acc"],
                    test_acc=r["test_acc"], balanced_acc=r["balanced_acc"],
                    majority=U.majority_rate(tr[2])))
            per_rep[rep].append(float(np.mean(accs)))
            gap_rep[rep].append(float(np.mean(gaps)))
            line.append(f"{np.mean(accs):10.3f}")
        ang = "    nan" if np.isnan(angle) else f"{angle:7.3f}"
        print(f"{pname:14s} {ang}  " + "  ".join(line))

    print("\nEffect size is the VARIANCE across placements, not the mean.")
    print(f"{'representation':16s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'gap':>8s}")
    print("-" * 54)
    for rep in reps:
        v, g = np.array(per_rep[rep]), np.array(gap_rep[rep])
        print(f"{rep:16s} {v.mean():8.3f} {v.std():8.4f} {v.min():8.3f} {g.mean():8.3f}")

    worst = max(np.mean(per_rep[r]) for r in reps)
    if worst > C.SATURATION_LIMIT:
        print(f"\n  !! SATURATED: mean accuracy {worst:.3f} > {C.SATURATION_LIMIT}. "
              f"Every model solves this task, so it measures nothing about\n"
              f"     inductive bias. Reduce BAND_SEPARATION, raise BAND_NOISE, or "
              f"cut n_train before reporting anything from this sweep.")
    return rows


def sample_efficiency(seeds, train_sizes, epochs, n_samples,
                      reps=C.REPRESENTATIONS, generator="latitude_bands"):
    """Learning curves at a polar placement, with the train/test gap."""
    _make.n = n_samples
    rows = []
    R = D.rotation_y(np.pi / 2 - 0.1)
    print("\nSample efficiency -- balanced accuracy vs n_train at a polar placement")
    print(f"{'n_train':>8s}  " + "  ".join(f"{r:>10s}" for r in reps))
    print("-" * (10 + 12 * len(reps)))

    for n_train in train_sizes:
        line = []
        for rep in reps:
            accs = []
            for s in seeds:
                sd = U.seeds(s)
                th, ph, y = _make(generator, sd["data"])
                th, ph = D.rotate_dataset(th, ph, R)
                tr, te = U.split(th, ph, y, sd["split"], n_train=n_train)
                if rep == "bloch":
                    r = U.train_vqc(tr, te, n_qubits=2, n_layers=C.DEPTH_DEFAULT,
                                    seed_init=sd["init"], epochs=epochs, stepsize=0.2)
                else:
                    r = U.run_classical(B.FFNNOnRepresentation(rep, seed=sd["init"]),
                                        tr, te)
                accs.append(r["balanced_acc"])
                rows.append(U.blank_row(
                    experiment="sample-efficiency", target=generator,
                    representation=rep,
                    model="VQC" if rep == "bloch" else f"FFNN({rep})",
                    n_params=r["n_params"], n_train=n_train, rotation_id="polar",
                    seed=sd["init"], train_acc=r["train_acc"], test_acc=r["test_acc"],
                    balanced_acc=r["balanced_acc"], majority=U.majority_rate(tr[2])))
            line.append(f"{np.mean(accs):10.3f}")
        print(f"{n_train:8d}  " + "  ".join(line))
    print("\n  Never report final accuracy alone -- the leftward shift is the claim.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--axis", choices=("tilt", "polar", "haar"), default=None)
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    if a.quick:
        rows = sweep(range(1), 160, 25, 1, reps=("cartesian", "sincos", "raw"),
                     axis=a.axis or "tilt")
        rows += sample_efficiency(range(1), (20, 80), 25, 160,
                                  reps=("cartesian", "sincos", "raw"))
    else:
        n_seeds = a.seeds or C.N_SEEDS
        axes = (a.axis,) if a.axis else ("tilt", "polar", "haar")
        rows = []
        for axis in axes:
            rows += sweep(range(n_seeds), C.N_SAMPLES, C.EPOCHS,
                          C.N_RANDOM_ROTATIONS, axis=axis)
        rows += sample_efficiency(range(a.seeds or C.N_SEEDS_EFFICIENCY),
                                  C.TRAIN_SIZES, C.EPOCHS, C.N_SAMPLES)
    U.write_rows("faithfulness.csv", rows)


if __name__ == "__main__":
    main()
