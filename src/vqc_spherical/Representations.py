"""
The four featurisations, behind one dispatch.

The experimental logic of the faithfulness phase is "hold the classifier fixed,
vary the representation".  That has to be enforced structurally rather than by
discipline, or the comparison quietly becomes four different experiments.

    raw        (theta, phi)                   the coordinate chart itself.
                                              Longitude jumps from 2pi to 0, a
                                              discontinuity the data does not
                                              have.
    sincos     (sin t, cos t, sin p, cos p)   the TORUS embedding.  Each angle
                                              wrapped independently, which is
                                              right for a torus and wrong for a
                                              sphere: it never identifies the
                                              longitudes at the pole.
    cartesian  (x, y, z)                      faithful.  The honest classical
                                              comparison, expected to tie with
                                              bloch.
    bloch      (theta, phi) -> RZ RY |0>      the quantum encoding.  Not a
                                              classical feature vector: the
                                              model consuming it is the VQC.

`raw` and `bloch` carry the SAME numbers.  What differs is what consumes them
-- which is precisely the claim under test: the advantage, if any, is in the
encoding, not in the coordinates.
"""

import numpy as np

from . import datasets as D

CLASSICAL = ("raw", "sincos", "cartesian")
QUANTUM = ("bloch",)
ALL = CLASSICAL + QUANTUM


def featurize(theta, phi, kind):
    """Feature matrix (n, d) for a representation."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if kind == "raw":
        return np.column_stack([theta, phi])
    if kind == "sincos":
        return np.column_stack([np.sin(theta), np.cos(theta),
                                np.sin(phi), np.cos(phi)])
    if kind == "cartesian":
        return D.sphere_to_cartesian(theta, phi)
    if kind == "bloch":
        return np.column_stack([theta, phi])
    raise ValueError(f"unknown representation '{kind}'; choose from {ALL}")


def is_quantum(kind):
    return kind in QUANTUM


def n_features(kind):
    return {"raw": 2, "sincos": 4, "cartesian": 3, "bloch": 2}[kind]
