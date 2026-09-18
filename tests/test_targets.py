"""Datasets must be what they claim, and must not leak the label."""
import numpy as np
import pytest

from vqc_spherical import Analysis as A
from vqc_spherical import Kernels as K
from vqc_spherical import datasets as D


def test_sphere_moons_leaks_the_label_into_longitude():
    """
    Documents the flaw rather than hiding it: class 0 draws phi in (0, pi) and
    class 1 in (pi, 2pi), so a threshold on phi alone is a perfect classifier.
    Any faithfulness result from this generator is meaningless.
    """
    _, phi, y = D.make_sphere_moons(4000, seed=3)
    assert np.mean((phi > np.pi).astype(float) == y) > 0.99


def test_latitude_bands_does_not_leak_into_longitude():
    _, phi, y = D.make_latitude_bands(4000, seed=3)
    acc = max(np.mean((phi > c).astype(float) == y) for c in np.linspace(0, 2 * np.pi, 64))
    assert acc < 0.60, "longitude must carry no information about the label"


def test_quadrupole_balance_and_ceiling():
    """42/58, and an affine ceiling near 0.786 -- not a 50-65% failure band."""
    theta, phi, y = D.make_quadrupole(8000, noise_std=0.0, seed=1)
    assert 0.40 < y.mean() < 0.45
    res = A.affine_ceiling(theta, phi, y, n_directions=1024)
    assert 0.76 < res["accuracy"] < 0.81
    assert abs(res["direction"][2]) > 0.98, "the best cap is polar, as symmetry demands"


def test_sectoral_is_balanced_and_below_the_quadrupole_ceiling():
    theta, phi, y = D.make_sectoral(8000, noise_std=0.0, seed=1)
    assert 0.48 < y.mean() < 0.52
    res = A.affine_ceiling(theta, phi, y, n_directions=1024)
    assert 0.62 < res["accuracy"] < 0.71


def test_hyperbolic_has_no_readable_marginal_but_is_out_of_reach():
    """
    Both coordinate marginals at chance is achievable -- at the cost of being a
    degree-2 function the single-upload model provably cannot express.  That
    trade is the restriction result stated as a property of datasets.
    """
    theta, phi, y = D.make_hyperbolic(6000, noise_std=0.0, seed=1)
    r = A.coordinate_leakage(theta, phi, y, n_directions=1024)
    assert r["theta_only"] < 0.56 and r["phi_only"] < 0.56
    assert r["ceiling"] < 0.75, "an affine model cannot solve it"


def test_tilted_bands_need_both_coordinates():
    theta, phi, y = D.make_tilted_bands(6000, seed=1, tilt=np.pi / 4)
    r = A.coordinate_leakage(theta, phi, y, n_directions=1024)
    assert r["theta_only"] < 0.70 and r["phi_only"] < 0.70
    assert r["ceiling"] > 0.85 and r["joint_gain"] > 0.15


def test_tilt_extremes_move_the_leak_rather_than_removing_it():
    """tilt = 0 leaks into theta; tilt = pi/2 leaks into phi.  Both are traps."""
    t0, p0, y0 = D.make_tilted_bands(4000, seed=1, tilt=0.0)
    t9, p9, y9 = D.make_tilted_bands(4000, seed=1, tilt=np.pi / 2)
    assert A.coordinate_leakage(t0, p0, y0, n_directions=512)["theta_only"] > 0.85
    assert A.coordinate_leakage(t9, p9, y9, n_directions=512)["phi_only"] > 0.85


def test_rotation_preserves_every_geodesic_distance():
    theta, phi, _ = D.make_latitude_bands(200, seed=2)
    t2, p2 = D.rotate_dataset(theta, phi, D.random_rotation(4))
    g1 = K.geodesic_angle(theta[:, None], phi[:, None], theta[None, :], phi[None, :])
    g2 = K.geodesic_angle(t2[:, None], p2[:, None], t2[None, :], p2[None, :])
    # arccos is ill-conditioned near coincident points -- a 1e-16 error in the
    # dot product becomes ~1e-8 in the angle -- so compare the cosines, which
    # are what the kernels actually consume, and the angles only loosely.
    np.testing.assert_allclose(np.cos(g1), np.cos(g2), atol=1e-14)
    assert np.abs(g1 - g2).max() < 1e-7


def test_rotation_does_not_change_the_difficulty():
    """The affine ceiling is invariant: a rotated task is the SAME task."""
    base = D.make_tilted_bands(6000, seed=1, tilt=0.0)
    tilt = D.make_tilted_bands(6000, seed=1, tilt=np.pi / 4)
    c0 = A.affine_ceiling(base[0], base[1], base[2], 1024)["accuracy"]
    c1 = A.affine_ceiling(tilt[0], tilt[1], tilt[2], 1024)["accuracy"]
    assert abs(c0 - c1) < 0.02


def test_every_target_has_the_same_signature():
    """
    All generators callable as make_target(name, n_samples, seed).  The earlier
    banded_4 entry took degree positionally and could not be, which is the kind
    of inconsistency that silently feeds a sweep the wrong dataset.
    """
    for name in D.TARGETS:
        theta, phi, y = D.make_target(name, n_samples=200, seed=7)
        assert len(theta) == len(phi) == len(y) == 200
        assert set(np.unique(y)) <= {0.0, 1.0}
        assert np.all((theta >= 0) & (theta <= np.pi))
        assert np.all((phi >= 0) & (phi < 2 * np.pi + 1e-9))


def test_every_target_honours_an_odd_sample_count():
    for name in D.TARGETS:
        theta, phi, y = D.make_target(name, n_samples=201, seed=7)
        assert len(theta) == len(phi) == len(y) == 201


def test_every_target_is_documented():
    assert set(D.TARGETS) == set(D.TARGET_META)
    for meta in D.TARGET_META.values():
        assert meta["role"] in {"baseline", "primary", "falsification", "deprecated"}
        assert meta["degree"] >= 1 and meta["note"]


def test_require_clean_blocks_the_leaking_generator():
    with pytest.raises(ValueError, match="leaks the label"):
        D.require_clean("sphere_moons")
    assert D.require_clean("tilted_bands") == "tilted_bands"
