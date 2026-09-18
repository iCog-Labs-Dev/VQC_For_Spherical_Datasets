"""
Stage 0 of the report: everything that needs NO training.

Produces
  results/affine_ceilings.csv   the provable ceiling of any degree-1 model on
                                every target -- the numbers that go into
                                PREDICTIONS.md
  results/model_structure.csv   the harmonic spectrum of the actual circuit,
                                confirming ell <= 1 at one upload

This is the gate on everything downstream: if the affine residual or the
spectrum disagrees with the closed forms, the derivation is wrong and no amount
of training will rescue it.

Usage:  python -m vqc_spherical.ExperimentAnalyticCore
"""
import numpy as np

from . import Analysis as A, Config as C, dataUtil as D, ExperimentUtils as U
from .AnsatzLayer import AnsatzLayer
from .EmbeddingLayer import EmbeddingLayer
from .VQCModel import VQCModel

SPECS = [("latitude_bands", 1), ("sphere_moons", 1), ("tilted_bands", 1),
         ("quadrupole", 2), ("sectoral", 2), ("banded_4", 4), ("hyperbolic", 2)]


def ceilings(n=C.N_CEILING, n_directions=C.N_DIRECTIONS):
    rows = []
    print(f"{'target':16s} {'degree':>6s} {'balance':>8s} {'majority':>9s} "
          f"{'ceiling':>8s} {'bal.ceiling':>12s}")
    print("-" * 66)
    for name, degree in SPECS:
        kw = dict(noise_std=0.0) if name in ("quadrupole", "sectoral", "hyperbolic") else {}
        theta, phi, y = D.make_target(name, n_samples=n, seed=1, **kw)
        plain = A.affine_ceiling(theta, phi, y, n_directions)
        bal = A.affine_ceiling(theta, phi, y, n_directions, balanced=True)
        print(f"{name:16s} {degree:6d} {y.mean():8.3f} "
              f"{plain['majority_baseline']:9.3f} {plain['accuracy']:8.3f} "
              f"{bal['accuracy']:12.3f}")
        rows.append(U.blank_row(experiment="ceiling", target=name, n_qubits=degree,
                                n_train=n, majority=plain["majority_baseline"],
                                ceiling=plain["accuracy"],
                                test_acc=plain["accuracy"],
                                balanced_acc=bal["accuracy"]))
    return rows


def model_structure(n_probe=C.N_PROBE):
    """
    The affine-residual test.  Probe the circuit itself -- no labels, no
    optimiser -- and project its output onto spherical harmonics.
    """
    theta, phi = A.sample_sphere(n_probe, seed=7)
    rows = []
    print(f"\n{'config':30s} {'resid at ell<=1':>16s}   energy at ell = 0,1,2,3")
    print("-" * 84)

    def probe(n_qubits, n_layers, encoding, n_uploads, seed):
        m = VQCModel(n_qubits, EmbeddingLayer(encoding),
                     AnsatzLayer("strong", n_layers=n_layers), n_uploads=n_uploads)
        w = np.random.default_rng(seed).uniform(0, 2 * np.pi, m.weight_shape())
        return A.probe_model(m, w, theta, phi)

    for n_qubits in (1, 2):
        for n_layers in C.DEPTHS:
            v = probe(n_qubits, n_layers, "spherical", 1, n_qubits * 100 + n_layers)
            resid = A.affine_residual(v, theta, phi)
            spec = A.harmonic_spectrum(v, theta, phi, ell_max=4)
            print(f"{'spherical q=%d L=%-2d' % (n_qubits, n_layers):30s} "
                  f"{resid:16.2e}   {np.round(spec[:4], 6)}")
            rows.append(U.blank_row(experiment="model-structure", encoding="spherical",
                                    n_qubits=n_qubits, n_layers=n_layers, n_uploads=1,
                                    test_acc=resid, balanced_acc=float(spec[2:].sum())))

    for n in (2, 3):
        v = probe(n, 3, "broadcast", 1, n)
        resid = A.harmonic_residual(v, theta, phi, ell_max=n)
        spec = A.harmonic_spectrum(v, theta, phi, ell_max=n + 2)
        print(f"{'broadcast q=%d L=3' % n:30s} {resid:16.2e}   "
              f"{np.round(spec[:4], 6)}   (residual at ell<=n)")
        rows.append(U.blank_row(experiment="model-structure", encoding="broadcast",
                                n_qubits=n, n_layers=3, n_uploads=1,
                                test_acc=resid, balanced_acc=float(spec[n])))
    return rows


def main():
    U.write_rows("affine_ceilings.csv", ceilings())
    U.write_rows("model_structure.csv", model_structure())


if __name__ == "__main__":
    main()
