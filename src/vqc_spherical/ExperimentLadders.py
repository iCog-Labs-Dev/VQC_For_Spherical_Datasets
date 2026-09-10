"""
The two expansion ladders.

Part A (no training, seconds).  Spectral occupancy per rung: probe each circuit
and project onto spherical harmonics.  This is the strongest evidence in the
project because no optimiser is involved, and it establishes the contrast:

  spin ladder    exactly degree-limited.  Energy above ell = n is zero to
                 machine precision, and ell = n is genuinely occupied.  The
                 induced feature map stays the fixed geodesic kernel
                 cos^{2n}(gamma/2), Schoenberg-admissible at every rung.

  re-uploading   NOT degree-limited.  It bounds the Fourier spectrum in the
                 ENCODING ANGLES, a different thing: at U = 2 the accessible
                 functions include sin(theta) cos(2 phi), which is not a
                 polynomial in (x, y, z) and so has an infinite Legendre
                 expansion.  Energy leaks above ell = U at every rung.

So the two ladders are not two knobs on one quantity.  One raises harmonic
degree and stays rotation-covariant; the other raises coordinate frequency and
leaves the Legendre basis altogether.

Part B (training).  Learnability against degree 1, 2 and 4 targets, in TWO
arms: fixed L (parameters grow with the rung) and U * L held constant
(parameters do not).  A staircase appearing only in the first arm is a
parameter-count effect, not an expressivity threshold, so it must appear in
both.

Usage:  python ExperimentLadders.py [--quick] [--no-training]
"""
import argparse
import csv
import os
import warnings

import numpy as np

try:
    from . import Analysis as A, Config as C, dataUtil as D, ExperimentUtils as U, Kernels as K
    from .AnsatzLayer import AnsatzLayer
    from .EmbeddingLayer import EmbeddingLayer
    from .VQCModel import VQCModel
except ImportError:  # pragma: no cover - flat sys.path
    import Analysis as A, Config as C, dataUtil as D, ExperimentUtils as U, Kernels as K
    from AnsatzLayer import AnsatzLayer
    from EmbeddingLayer import EmbeddingLayer
    from VQCModel import VQCModel

warnings.filterwarnings("ignore")

LADDER_TARGETS = {"latitude_bands": 1, "quadrupole": 2, "banded_4": 4}


def _probe(theta, phi, n_qubits, n_uploads, encoding, n_layers, draw):
    m = VQCModel(n_qubits, EmbeddingLayer(encoding),
                 AnsatzLayer("strong", n_layers=n_layers), n_uploads=n_uploads)
    w = np.random.default_rng(1000 * draw + 10 * n_qubits + n_uploads).uniform(
        0, 2 * np.pi, m.weight_shape())
    return A.probe_model(m, w, theta, phi)


def spectral_occupancy(n_probe=C.N_PROBE, ell_max=6, n_draws=3):
    theta, phi = A.sample_sphere(n_probe, seed=21)
    rows = []
    print("Part A -- harmonic energy per degree (mean over random weight draws)\n")
    print(f"{'ladder rung':22s} {'resid at ell<=rung':>19s}   "
          + "  ".join(f"ell={i}" for i in range(5)))
    print("-" * 82)

    for n in C.SPIN_QUBITS:
        specs = [A.harmonic_spectrum(_probe(theta, phi, n, 1, "broadcast", 3, d),
                                     theta, phi, ell_max) for d in range(n_draws)]
        resids = [A.harmonic_residual(_probe(theta, phi, n, 1, "broadcast", 3, d),
                                      theta, phi, ell_max=n) for d in range(n_draws)]
        spec = np.mean(specs, axis=0)
        print(f"{'spin  n=%d' % n:22s} {np.mean(resids):19.2e}   "
              + "  ".join('%.4f' % x for x in spec[:5]))
        rows.append(U.blank_row(experiment="ladder-spectral", model=f"spin_n{n}",
                                encoding="broadcast", n_qubits=n, n_uploads=1,
                                test_acc=float(np.mean(resids)),
                                balanced_acc=float(spec[min(n, ell_max)])))

    for u in C.UPLOADS:
        specs = [A.harmonic_spectrum(_probe(theta, phi, 1, u, "spherical", 2, d),
                                     theta, phi, ell_max) for d in range(n_draws)]
        resids = [A.harmonic_residual(_probe(theta, phi, 1, u, "spherical", 2, d),
                                      theta, phi, ell_max=u) for d in range(n_draws)]
        spec = np.mean(specs, axis=0)
        print(f"{'reupload  U=%d' % u:22s} {np.mean(resids):19.2e}   "
              + "  ".join('%.4f' % x for x in spec[:5]))
        rows.append(U.blank_row(experiment="ladder-spectral", model=f"reupload_U{u}",
                                encoding="spherical", n_qubits=1, n_uploads=u,
                                test_acc=float(np.mean(resids)),
                                balanced_acc=float(spec[min(u, ell_max)])))

    print("\n  spin: residual at ell <= n must be ~1e-15  (exactly degree-limited)")
    print("  reupload: residual at ell <= U is O(0.1)   (NOT degree-limited)")
    return rows


def kernel_contrast(n_pairs=4000, seed=5):
    """
    k versus gamma for both ladders.

    The spin ladder collapses onto a single monotone curve.  The re-uploading
    model does not: its scatter at fixed gamma IS the coordinate dependence,
    and it is the point of the figure.  Do not average it away.
    """
    rng = np.random.default_rng(seed)
    t1, p1 = A.sample_sphere(n_pairs, seed=seed)
    t2, p2 = A.sample_sphere(n_pairs, seed=seed + 1)
    gamma = K.geodesic_angle(t1, p1, t2, p2)
    rows = []

    for n in C.SPIN_QUBITS:
        k = K.bloch_kernel_spin(t1, p1, t2, p2, n=n)
        rows += [{"ladder": "spin", "rung": n, "gamma": float(g), "k": float(v)}
                 for g, v in zip(gamma, k)]

    for u in C.UPLOADS:
        m = VQCModel(1, EmbeddingLayer("spherical"),
                     AnsatzLayer("strong", n_layers=2), n_uploads=u)
        w = rng.uniform(0, 2 * np.pi, m.weight_shape())
        f1 = A.probe_model(m, w, t1, p1)
        f2 = A.probe_model(m, w, t2, p2)
        k = (f1 * f2 + 1.0) / 2.0
        rows += [{"ladder": "reupload", "rung": u, "gamma": float(g), "k": float(v)}
                 for g, v in zip(gamma, k)]

    os.makedirs(U.RESULTS, exist_ok=True)
    path = os.path.join(U.RESULTS, "kernel_contrast.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["ladder", "rung", "gamma", "k"])
        w_.writeheader()
        w_.writerows(rows)
    print(f"  -> results/kernel_contrast.csv  ({len(rows)} rows)")
    return rows


def learnability(seeds, epochs, n_samples, arms=("fixed_L", "fixed_budget")):
    rows = []
    for arm in arms:
        print(f"\nPart B -- {arm}: balanced accuracy by rung x target")
        print(f"{'rung':12s} " + "  ".join(f"{t:>16s}" for t in LADDER_TARGETS))
        print("-" * 64)

        for rung in C.UPLOADS:
            L = C.DEPTH_DEFAULT if arm == "fixed_L" else max(1, C.FIXED_BUDGET // rung)
            line = []
            for tname in LADDER_TARGETS:
                accs = []
                for s in seeds:
                    sd = U.seeds(s)
                    th, ph, y = D.make_target(tname, n_samples=n_samples, seed=sd["data"])
                    tr, te = U.split(th, ph, y, sd["split"])
                    r = U.train_vqc(tr, te, n_qubits=1, n_layers=L, n_uploads=rung,
                                    encoding="spherical", seed_init=sd["init"],
                                    epochs=epochs, stepsize=0.2)
                    accs.append(r["balanced_acc"])
                    rows.append(U.blank_row(
                        experiment=f"ladder-reupload-{arm}", target=tname, model="VQC",
                        representation="bloch", encoding="spherical", n_uploads=rung,
                        n_qubits=1, n_layers=L, n_params=r["n_params"],
                        n_train=len(tr[2]), seed=sd["init"], train_acc=r["train_acc"],
                        test_acc=r["test_acc"], balanced_acc=r["balanced_acc"],
                        majority=U.majority_rate(tr[2]), wall_time=r["wall_time"]))
                line.append(f"{np.mean(accs):16.3f}")
            print(f"U={rung} L={L:<6d} " + "  ".join(line))

        if arm != "fixed_L":
            continue
        for rung in C.SPIN_QUBITS:
            line = []
            for tname in LADDER_TARGETS:
                accs = []
                for s in seeds:
                    sd = U.seeds(s)
                    th, ph, y = D.make_target(tname, n_samples=n_samples, seed=sd["data"])
                    tr, te = U.split(th, ph, y, sd["split"])
                    r = U.train_vqc(tr, te, n_qubits=rung, n_layers=C.DEPTH_DEFAULT,
                                    n_uploads=1, encoding="broadcast",
                                    seed_init=sd["init"], epochs=epochs, stepsize=0.2)
                    accs.append(r["balanced_acc"])
                    rows.append(U.blank_row(
                        experiment="ladder-spin", target=tname, model="VQC",
                        representation="bloch", encoding="broadcast", n_uploads=1,
                        n_qubits=rung, n_layers=C.DEPTH_DEFAULT, n_params=r["n_params"],
                        n_train=len(tr[2]), seed=sd["init"], train_acc=r["train_acc"],
                        test_acc=r["test_acc"], balanced_acc=r["balanced_acc"],
                        majority=U.majority_rate(tr[2]), wall_time=r["wall_time"]))
                line.append(f"{np.mean(accs):16.3f}")
            print(f"spin n={rung:<5d} " + "  ".join(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-training", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()

    rows = spectral_occupancy(n_probe=800 if a.quick else C.N_PROBE,
                              n_draws=1 if a.quick else 3)
    kernel_contrast(n_pairs=300 if a.quick else 4000)
    if not a.no_training:
        rows += learnability(seeds=range(1) if a.quick else range(a.seeds or C.N_SEEDS),
                             epochs=25 if a.quick else C.EPOCHS,
                             n_samples=160 if a.quick else C.N_SAMPLES,
                             arms=("fixed_L",) if a.quick else ("fixed_L", "fixed_budget"))
    U.write_rows("ladder.csv", rows)


if __name__ == "__main__":
    main()
