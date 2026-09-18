"""
The dataset gate.  Run before any experiment, and after touching any generator.

The failure it prevents
-----------------------
make_sphere_moons draws longitude from (0, pi) for one class and (pi, 2pi) for
the other, so longitude ALONE classifies it perfectly.  Every model scores
100%, the benchmark is saturated, and nothing can be concluded from it about
geometry or inductive bias.  That flaw reached a committed figure because
nobody checked; one call to this script reports it in seconds.

Verdict -- the test depends on what the dataset is FOR
------------------------------------------------------
A single criterion does not work, and the first version of this script got it
wrong in an instructive way.  It flagged the quadrupole as leaking because a
theta threshold alone reaches 0.786, exactly the affine ceiling.  That is not a
defect: the best degree-1 approximation to a theta-only target IS a theta cut,
so the two coincide by construction.  The comparison only means something
relative to what the dataset is supposed to do.

  role "primary"        must be SOLVABLE by a degree-1 model and must NEED both
                        coordinates (joint gain >= MIN_JOINT_GAIN).
  role "baseline"       must be solvable; axis-aligned on purpose, so theta
                        alone reaching the ceiling is the intent.  The control,
                        never the headline task.
  role "falsification"  must be OUT OF REACH of a degree-1 model: the ceiling
                        must sit well below 1, since the point is that the true
                        label is expressible and the circuit still cannot get
                        there.  Single-coordinate numbers inform, not gate.
  role "deprecated"     reported, never gates.

A generator failing the test for its own role fails the run with a non-zero
exit code, so a newly written dataset cannot quietly enter the pipeline broken.

Usage:  python -m vqc_spherical.ValidateDatasets [--n 20000] [--directions 4096]
"""
import argparse

from . import Analysis as A, datasets as D, ExperimentUtils as U

# A "solvable" task must be within reach of a degree-1 model; a falsification
# target must not be.  The gap is deliberately wide, so a dataset sitting
# between them is reported rather than silently accepted.
SOLVABLE_MIN_CEILING = 0.85
FALSIFICATION_MAX_CEILING = 0.90
MIN_JOINT_GAIN = 0.10

TARGET_ORDER = ["sphere_moons", "latitude_bands", "tilted_bands",
                "quadrupole", "sectoral", "banded_4", "hyperbolic"]


def validate(n_samples=20000, n_directions=2048, seed=99):
    rows, failures = [], []

    print(f"Dataset validation -- n = {n_samples}, {n_directions} directions\n")
    print(f"{'generator':16s} {'role':>13s} {'deg':>4s} {'bal':>6s} {'maj':>6s} "
          f"{'theta':>7s} {'phi':>7s} {'ceil':>7s} {'gain':>7s}  verdict")
    print("-" * 96)

    for name in TARGET_ORDER:
        meta = D.TARGET_META[name]
        theta, phi, y = D.make_target(name, n_samples=n_samples, seed=seed)
        lk = A.coordinate_leakage(theta, phi, y, n_directions=n_directions)
        best_single = max(lk["theta_only"], lk["phi_only"])
        role = meta["role"]

        if role == "deprecated":
            what, _ = D.LEAKY_GENERATORS.get(name, ("one coordinate", 0))
            verdict = f"KNOWN LEAK ({what})"
        elif role == "falsification":
            ok = lk["ceiling"] <= FALSIFICATION_MAX_CEILING
            verdict = "OK (unreachable)" if ok else "FAIL -- reachable"
        elif role == "baseline":
            ok = lk["ceiling"] >= SOLVABLE_MIN_CEILING
            verdict = "OK (axis-aligned)" if ok else "FAIL -- unsolvable"
        else:
            if lk["ceiling"] < SOLVABLE_MIN_CEILING:
                ok, verdict = False, "FAIL -- unsolvable"
            elif best_single >= lk["ceiling"] - D.LEAKAGE_MARGIN:
                ok, verdict = False, "FAIL -- one coord suffices"
            elif lk["joint_gain"] < MIN_JOINT_GAIN:
                ok, verdict = False, "FAIL -- weak coupling"
            else:
                ok, verdict = True, "OK (needs both)"
        if role != "deprecated" and not ok:
            failures.append(name)

        print(f"{name:16s} {role:>13s} {meta['degree']:>4d} "
              f"{y.mean():6.3f} {U.majority_rate(y):6.3f} "
              f"{lk['theta_only']:7.3f} {lk['phi_only']:7.3f} "
              f"{lk['ceiling']:7.3f} {lk['joint_gain']:7.3f}  {verdict}")

        rows.append(U.blank_row(
            experiment="dataset-validation", target=name, model=role,
            n_qubits=meta["degree"], n_train=n_samples,
            train_acc=lk["theta_only"], test_acc=lk["phi_only"],
            balanced_acc=lk["joint_gain"], majority=U.majority_rate(y),
            ceiling=lk["ceiling"], representation=verdict))

    print()
    for name in TARGET_ORDER:
        print(f"  {name:16s} {D.TARGET_META[name]['note']}")

    if failures:
        print(f"\nFAILED: {', '.join(failures)} do not satisfy the test for their "
              f"declared role.  Fix the generator, or change its role in "
              f"datasets.TARGET_META if the role was wrong.")
    else:
        print("\nEvery generator satisfies the test for its declared role.")
    return rows, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--directions", type=int, default=2048)
    a = ap.parse_args()
    rows, failures = validate(a.n, a.directions)
    U.write_rows("dataset_validation.csv", rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
