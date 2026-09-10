"""
The analytic core, made executable.

Every kernel here is a function of two points on S^2 given in spherical
coordinates (theta, phi).  All of them broadcast: pass theta[:, None] against
theta[None, :] to build a Gram matrix in one call.

The central object is

    k(gamma) = (1 + cos gamma) / 2 = cos^2(gamma/2)
             = (1/2) P_0(cos gamma) + (1/2) P_1(cos gamma)

the fidelity kernel of the encoding RZ(phi) RY(theta) |0>.

Faithfulness is *which basis*: the kernel depends on the geodesic angle gamma
and nothing else, so it is expanded in Legendre polynomials at all.
Restriction is *how many terms*: the expansion terminates at ell = 1.
"""

import numpy as np
from numpy.polynomial import legendre as L


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

def to_cartesian(theta, phi):
    """(theta, phi) -> unit vectors, stacked on the last axis."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=-1)


def geodesic_angle(t1, p1, t2, p2):
    """
    Great-circle angle gamma in [0, pi] between two points on S^2.

    The clip before arccos is not cosmetic: dot products of unit vectors exceed
    1 by ~1e-16 in floating point and arccos then returns NaN silently.
    """
    dot = np.sum(to_cartesian(t1, p1) * to_cartesian(t2, p2), axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


# ----------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------

def bloch_kernel(t1, p1, t2, p2):
    """
    (1 + cos gamma) / 2 -- the fidelity kernel of the single-qubit encoding.
    Faithful (strictly decreasing in gamma) and restricted (ell <= 1).
    Funk-Hecke eigenvalues: lambda_0 = 1/2, lambda_1 = 1/6 (threefold
    degenerate in m), lambda_ell = 0 for ell >= 2.
    """
    return 0.5 * (1.0 + np.cos(geodesic_angle(t1, p1, t2, p2)))


def bloch_kernel_spin(t1, p1, t2, p2, n=1):
    """
    cos^{2n}(gamma/2) -- the n-qubit broadcast (spin-n/2 coherent state) kernel.

    Faithful for every n; the Legendre expansion terminates at ell = n with

        lambda_ell^(n) = (n!)^2 / [(n - ell)! (n + ell + 1)!],  0 <= ell <= n.

    For small gamma, k_n ~ exp(-n gamma^2 / 4): a geodesic kernel whose
    bandwidth sharpens as 1/sqrt(n).
    """
    return np.cos(geodesic_angle(t1, p1, t2, p2) / 2.0) ** (2 * n)


def cartesian_kernel(t1, p1, t2, p2):
    """cos gamma -- the linear kernel on (x, y, z).  Faithful, ell <= 1."""
    return np.cos(geodesic_angle(t1, p1, t2, p2))


def sincos_kernel(t1, p1, t2, p2):
    """
    cos(d.theta) + cos(d.phi) -- the TORUS kernel.

    Not faithful and not positive-definite on the sphere: it is a function of
    the coordinate differences, not of the geodesic distance, so Schoenberg's
    condition does not apply.  This is the representation that embeds the wrong
    manifold, and it is the honest comparison target.  Note k(0) = 2, not 1 --
    it is unnormalised by construction.
    """
    return (np.cos(np.asarray(t1) - np.asarray(t2))
            + np.cos(np.asarray(p1) - np.asarray(p2)))


def amplitude_kernel(t1, p1, t2, p2):
    """
    cos^2 gamma -- distance-only but NOT injective: k(gamma) = k(pi - gamma),
    so antipodal points are identified and the embedded manifold is RP^2, not
    S^2.  Depends on gamma alone yet fails faithfulness, because it is not
    strictly monotone on [0, pi].
    """
    return np.cos(geodesic_angle(t1, p1, t2, p2)) ** 2


KERNELS = {
    "bloch": bloch_kernel,
    "cartesian": cartesian_kernel,
    "sincos": sincos_kernel,
    "amplitude": amplitude_kernel,
}


# ----------------------------------------------------------------------
# Gram matrices and spectra
# ----------------------------------------------------------------------

def gram_matrix(theta, phi, kernel_fn=bloch_kernel, **kw):
    """Full n x n Gram matrix, built by broadcasting rather than looping."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return kernel_fn(theta[:, None], phi[:, None], theta[None, :], phi[None, :], **kw)


def gram_spectrum(theta, phi, kernel_fn=bloch_kernel, **kw):
    """Eigenvalues of the Gram matrix, sorted descending."""
    return np.sort(np.linalg.eigvalsh(gram_matrix(theta, phi, kernel_fn, **kw)))[::-1]


def effective_rank(eigenvalues, rtol=1e-10):
    """
    Count eigenvalues above rtol * lambda_max.

    A rank-4 Gram matrix of size 400 carries ~1e-14 of numerical noise, so
    "exactly four nonzero eigenvalues" is only meaningful relative to the
    largest one.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    return int(np.sum(np.abs(ev) > rtol * np.max(np.abs(ev))))


# ----------------------------------------------------------------------
# Analytic references
# ----------------------------------------------------------------------

def funk_hecke_eigenvalues(kernel_1d, ell_max=6, n_quad=400):
    """
    lambda_ell = (1/2) * integral_{-1}^{1} f(t) P_ell(t) dt

    for a kernel written as a function f of t = cos gamma.  Gauss-Legendre
    quadrature is exact for the polynomial kernels used here.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    f = kernel_1d(nodes)
    out = []
    for ell in range(ell_max + 1):
        c = np.zeros(ell + 1)
        c[ell] = 1.0
        out.append(0.5 * np.sum(weights * f * L.legval(nodes, c)))
    return np.array(out)


def spin_eigenvalues_analytic(n, ell_max=None):
    """
    Closed form: lambda_ell = (n!)^2 / [(n-ell)! (n+ell+1)!], zero above
    ell = n.  Satisfies sum_ell (2 ell + 1) lambda_ell = k_n(0) = 1.
    """
    from math import factorial as fact
    ell_max = n if ell_max is None else ell_max
    return np.array([
        (fact(n) ** 2) / (fact(n - ell) * fact(n + ell + 1)) if ell <= n else 0.0
        for ell in range(ell_max + 1)])


def legendre_coefficients(kernel_1d, ell_max=6, n_quad=400):
    """
    Coefficients a_ell of k(gamma) = sum_ell a_ell P_ell(cos gamma), related to
    the Funk-Hecke eigenvalues by a_ell = (2 ell + 1) lambda_ell.  Schoenberg's
    condition is a_ell >= 0 for all ell.
    """
    lam = funk_hecke_eigenvalues(kernel_1d, ell_max=ell_max, n_quad=n_quad)
    return np.array([(2 * ell + 1) * lam[ell] for ell in range(ell_max + 1)])
