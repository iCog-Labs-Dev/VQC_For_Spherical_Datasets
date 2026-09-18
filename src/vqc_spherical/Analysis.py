"""
The instruments that turn restriction from an experimental claim into an
algebraic one.

Why this module exists
----------------------
A model that fails to learn a target has two possible explanations: it cannot
express the function, or the optimiser did not find it.  No accuracy number
separates them.  Two constructions do, and neither trains anything.

1. affine_ceiling -- the encoded density matrix rho = (I + r.sigma)/2 is AFFINE
   in the Bloch vector r, and everything downstream (ansatz, ancillas,
   entanglement, measurement) is a LINEAR map on rho.  Therefore

       <O> = alpha + beta . r        exactly, four real parameters,

   regardless of depth, width, ancilla count or observable.  The decision
   region of any such model is a single spherical cap.  So the best accuracy
   ANY one-upload model can reach on a dataset is computable directly, by
   scanning cap orientations -- a provable ceiling, not a trained one.

2. harmonic_spectrum / harmonic_residual -- project the model's own output onto
   real spherical harmonics.  At one upload all energy must sit in ell <= 1;
   with n broadcast qubits, in ell <= n.  This measures ell_max directly rather
   than inferring it from accuracy.

3. coordinate_leakage -- how much of a dataset's label is readable from a
   single coordinate.  A dataset that fails this cannot test anything about
   geometry, however good the model is.

Run all three before any training experiment.  If (2) disagrees with the closed
form, the derivation is wrong and no experiment will rescue it.
"""

import numpy as np
from scipy.special import factorial, lpmv

from . import Kernels as K


# ----------------------------------------------------------------------
# The affine ceiling
# ----------------------------------------------------------------------

def fibonacci_directions(n=4096):
    """
    n near-uniformly spread unit vectors on S^2 (Fibonacci sphere).

    Uniform coverage matters: a lat/long grid over-samples the poles and would
    bias the ceiling upward for pole-centred targets.
    """
    i = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    return np.stack([r * np.cos(golden * i), r * np.sin(golden * i), z], axis=-1)


def affine_ceiling(theta, phi, labels, n_directions=4096, balanced=False, chunk=256):
    """
    Best accuracy achievable by ANY model of the form f = alpha + beta . r.

    For a fixed direction beta the classifier is a threshold on the scalar
    projection, so the optimal threshold is found exactly by sorting -- no
    optimisation, no seeds, no learning rate.  Both orientations of the
    inequality are tried, so the search covers every spherical cap.
    Vectorised over directions in chunks.

    The direction grid is finite, so the returned accuracy is a LOWER bound on
    the true ceiling: the best available direction can sit half a grid spacing
    off the optimum, which costs roughly a point of accuracy at 1024 directions
    on a sharply defined cap.  Use a fine grid for reported numbers, and read a
    measured accuracy marginally above a coarse-grid ceiling as a resolution
    artefact rather than a falsification.

    Returns a dict with accuracy, direction, threshold and majority_baseline.
    The majority baseline is returned alongside deliberately: on an unbalanced
    target, "the model scored 58%" and "the model predicts one class for
    everything" are indistinguishable unless both numbers are in the table.
    """
    r = K.to_cartesian(theta, phi)
    y = np.asarray(labels).astype(int).ravel()
    n = len(y)

    if balanced:
        w = np.where(y == 1, 0.5 / max(int(np.sum(y == 1)), 1),
                     0.5 / max(int(np.sum(y == 0)), 1))
    else:
        w = np.full(n, 1.0 / n)

    dirs = fibonacci_directions(n_directions)
    total_pos = float(np.sum(w[y == 1]))
    best_acc, best_dir, best_thr = -1.0, None, None

    for start in range(0, len(dirs), chunk):
        D = dirs[start:start + chunk]
        S = r @ D.T
        order = np.argsort(S, axis=0, kind="stable")
        ys, ws = y[order], w[order]

        below0 = np.cumsum(np.where(ys == 0, ws, 0.0), axis=0)
        above1 = total_pos - np.cumsum(np.where(ys == 1, ws, 0.0), axis=0)
        acc = np.empty((n + 1, D.shape[0]))
        acc[0] = total_pos
        acc[1:] = below0 + above1
        acc = np.maximum(acc, 1.0 - acc)

        j = np.argmax(acc, axis=0)
        vals = acc[j, np.arange(D.shape[0])]
        c_best = int(np.argmax(vals))
        if vals[c_best] > best_acc:
            best_acc = float(vals[c_best])
            best_dir = D[c_best].copy()
            jb = int(j[c_best])
            s_sorted = np.take_along_axis(S[:, [c_best]], order[:, [c_best]], axis=0).ravel()
            if jb == 0:
                best_thr = float(s_sorted[0] - 1e-9)
            elif jb >= n:
                best_thr = float(s_sorted[-1] + 1e-9)
            else:
                best_thr = float((s_sorted[jb - 1] + s_sorted[jb]) / 2)

    p1 = float(np.mean(y == 1))
    return {"accuracy": best_acc, "direction": best_dir, "threshold": best_thr,
            "majority_baseline": 0.5 if balanced else max(p1, 1 - p1)}


# ----------------------------------------------------------------------
# Real spherical harmonics and the ell-spectrum of a model
# ----------------------------------------------------------------------

def real_sph_harm(ell, m, theta, phi):
    """
    Real orthonormal spherical harmonic Y_{ell m}(theta, phi).

    Built from lpmv rather than scipy.special.sph_harm so it is immune to the
    sph_harm / sph_harm_y deprecation churn across scipy versions.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    am = abs(m)
    norm = np.sqrt((2 * ell + 1) / (4 * np.pi) * factorial(ell - am) / factorial(ell + am))
    P = lpmv(am, ell, np.cos(theta))
    if m == 0:
        return norm * P
    return np.sqrt(2.0) * norm * P * (np.cos(am * phi) if m > 0 else np.sin(am * phi))


def harmonic_design(theta, phi, ell_max):
    """Design matrix whose columns are Y_{ell m} for ell <= ell_max."""
    return np.stack([real_sph_harm(ell, m, theta, phi)
                     for ell in range(ell_max + 1)
                     for m in range(-ell, ell + 1)], axis=-1)


def harmonic_spectrum(values, theta, phi, ell_max=6):
    """
    Energy of a function on the sphere per degree ell, normalised to sum 1.
    The one-upload model must put ~100% of its energy in ell <= 1.
    """
    A = harmonic_design(theta, phi, ell_max)
    coef, *_ = np.linalg.lstsq(A, np.asarray(values, dtype=float), rcond=None)
    out, k = [], 0
    for ell in range(ell_max + 1):
        width = 2 * ell + 1
        out.append(float(np.sum(coef[k:k + width] ** 2)))
        k += width
    out = np.array(out)
    total = out.sum()
    return out / total if total > 0 else out


def harmonic_residual(values, theta, phi, ell_max):
    """
    Relative residual after fitting only harmonics up to ell_max.  ~1e-12 means
    the function lives entirely in ell <= ell_max: the direct numerical
    statement of restriction.
    """
    A = harmonic_design(theta, phi, ell_max)
    v = np.asarray(values, dtype=float)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    scale = np.linalg.norm(v - v.mean()) or 1.0
    return float(np.linalg.norm(v - A @ coef) / scale)


def affine_residual(values, theta, phi):
    """
    Relative residual of the fit f ~ alpha + beta . r.  Identical to
    harmonic_residual at ell_max = 1 -- the ell <= 1 harmonics span exactly
    {1, x, y, z} -- but named for what it tests.
    """
    return harmonic_residual(values, theta, phi, ell_max=1)


# ----------------------------------------------------------------------
# Probing a circuit
# ----------------------------------------------------------------------

def sample_sphere(n=2000, seed=0):
    """Uniform points on S^2: z uniform in [-1, 1], NOT theta uniform."""
    rng = np.random.default_rng(seed)
    return np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)


def probe_model(model, weights, theta, phi):
    """
    Evaluate a VQCModel over sphere points -> 1-D array.

    Uses PennyLane parameter broadcasting: passing [theta_array, phi_array] as
    the features runs the whole batch in one device call, two orders of
    magnitude faster than a per-sample loop, returning identical values.  Falls
    back to the loop if a device or ansatz does not support broadcasting.
    """
    model.eval()
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    try:
        out = np.asarray(model.forward([theta, phi], weights), dtype=float)
        if out.shape == theta.shape:
            return out
    except Exception:
        pass
    return np.array([float(model.forward(np.array([t, p]), weights))
                     for t, p in zip(theta, phi)])


# ----------------------------------------------------------------------
# Dataset diagnostics
# ----------------------------------------------------------------------

def coordinate_leakage(theta, phi, labels, n_grid=180, n_directions=2048):
    """
    How much of the label is readable from ONE coordinate alone?

    theta_only : best single threshold on theta (a latitude cut)
    phi_only   : best ARC in phi -- cos(phi - phi_0) > c, scanned over phi_0
                 and c, because phi is circular and a plain threshold would
                 understate what is readable from it
    ceiling    : affine_ceiling, the best any degree-1 model could do
    joint_gain : ceiling minus the better single coordinate

    A dataset where one coordinate reaches the ceiling is solvable by reading
    one number, so it cannot test anything about how a representation handles
    the sphere.  This is the check that catches a label leaking into longitude.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(labels).astype(int)

    def best_threshold(scores):
        order = np.argsort(scores)
        ys = y[order]
        below0 = np.cumsum(ys == 0) / len(y)
        above1 = (np.sum(ys == 1) - np.cumsum(ys == 1)) / len(y)
        acc = np.concatenate([[np.mean(ys == 1)], below0 + above1])
        return float(np.max(np.maximum(acc, 1.0 - acc)))

    theta_only = best_threshold(theta)
    phi_only = max(best_threshold(np.cos(phi - p0))
                   for p0 in np.linspace(0, 2 * np.pi, n_grid, endpoint=False))
    ceiling = affine_ceiling(theta, phi, y, n_directions)["accuracy"]
    return {"theta_only": theta_only, "phi_only": phi_only, "ceiling": ceiling,
            "joint_gain": ceiling - max(theta_only, phi_only)}
