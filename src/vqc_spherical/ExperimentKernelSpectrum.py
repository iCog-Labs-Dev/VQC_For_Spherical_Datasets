"""
The empirical kernel spectrum.  No training, seconds to run.

Direct numerical confirmation: the Gram matrix of the encoding's fidelity
kernel has exactly four non-zero eigenvalues, n * (1/2, 1/6, 1/6, 1/6).  The
ell = 1 eigenvalue is threefold degenerate in m, so the ratio is 3 : 1 : 1 : 1,
not "3 : 1".  They sum to the trace, which is n because k(0) = 1.

Also runs the spin ladder, where the rank must be (n + 1)^2, and the two
unfaithful controls for contrast.

Usage:  python -m vqc_spherical.ExperimentKernelSpectrum
"""
import numpy as np

from . import Analysis as A, Config as C, ExperimentUtils as U, Kernels as K


def main(n_points=600, seed=1):
    theta, phi = A.sample_sphere(n_points, seed=seed)
    rows = []

    print(f"Gram spectra, n = {n_points}\n")
    print(f"{'kernel':22s} {'rank':>5s} {'trace/n':>9s}   top eigenvalues / n")
    print("-" * 78)
    for name, fn in [("bloch", K.bloch_kernel), ("cartesian", K.cartesian_kernel),
                     ("sincos (torus)", K.sincos_kernel),
                     ("amplitude (RP^2)", K.amplitude_kernel)]:
        ev = K.gram_spectrum(theta, phi, fn)
        rank = K.effective_rank(ev, C.RANK_RTOL)
        print(f"{name:22s} {rank:5d} {np.sum(ev) / n_points:9.4f}   "
              f"{np.round(ev[:5] / n_points, 4)}")
        rows.append(U.blank_row(experiment="kernel-spectrum", model=name,
                                n_qubits=1, n_params=rank,
                                test_acc=float(ev[0] / n_points)))

    print(f"\n{'spin ladder':22s} {'rank':>5s} {'predicted':>10s}   "
          f"top eigenvalues / n  vs  analytic lambda_ell")
    print("-" * 78)
    for n in C.SPIN_QUBITS:
        ev = K.gram_spectrum(theta, phi, K.bloch_kernel_spin, n=n)
        rank = K.effective_rank(ev, 1e-9)
        lam = K.spin_eigenvalues_analytic(n)
        print(f"n = {n:<18d} {rank:5d} {(n + 1) ** 2:10d}   "
              f"{np.round(ev[:4] / n_points, 4)}  vs  {np.round(lam[:4], 4)}")
        rows.append(U.blank_row(experiment="kernel-spectrum-spin", model=f"spin_n{n}",
                                n_qubits=n, n_params=rank,
                                test_acc=float(ev[0] / n_points)))

    ev = K.gram_spectrum(theta, phi, K.bloch_kernel)
    print("\nChecks")
    print(f"  rank(bloch) == 4                : {K.effective_rank(ev) == 4}")
    print(f"  lambda_0 / n ~ 1/2              : {ev[0] / n_points:.4f}")
    print(f"  lambda_1..3 / n ~ 1/6 (3-fold)  : {np.round(ev[1:4] / n_points, 4)}")
    print(f"  sum eigenvalues == trace == n   : {abs(np.sum(ev) - n_points) < 1e-8}")

    U.write_rows("kernel_spectra.csv", rows)
    return rows


if __name__ == "__main__":
    main()
