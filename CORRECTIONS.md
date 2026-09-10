# Corrections

What the original research roadmap assumed, and what measurement showed. Every
number here is produced by the code on this branch; the commands that produce
each are given at the end.

This file exists because three of these would ordinarily surface only after
weeks of training runs, or during peer review. All were found within minutes by
instruments that train nothing, and recording them is more useful than quietly
shipping the corrected version.

---

## 1. The degree-2 failure prediction was wrong, in a way that inverts its meaning

**Assumed.** The circuit fails on the `quadrupole` target at 50-65% accuracy.

**Measured.** The affine ceiling is **0.786**. It is also derivable by hand: the
target labels a point positive when |z| > 1/sqrt(3) ~ 0.577, which is two caps
around the poles, and the best single circle covers one cap plus the entire
equatorial band, giving 0.211 + 0.577 = 0.789.

Two things follow. A circuit scoring 79% -- the theory working exactly as
derived -- would have been recorded as the theory **failing**. And the predicted
band straddles the majority-class baseline of 0.575, so it could not have
distinguished a genuine expressivity limit from a model that always guesses the
common class.

**Now.** `PREDICTIONS.md` pre-registers the computed ceiling per target and the
sharper claim: accuracy saturates there and never exceeds it, at any depth,
width, learning rate or seed. `ExperimentFalsification.py` prints the ceiling
first and flags any configuration that beats it.

## 2. The existing dataset could not measure what it was used for

**Assumed.** `make_sphere_moons` is a valid geometry task.

**Measured.** Class 0 draws phi in (0, pi) and class 1 in (pi, 2pi), so
**longitude alone classifies it at 1.000** and the affine ceiling is 1.000.
Every model scores 100%. This is the saturated-benchmark failure mode: nothing
about inductive bias can be learned from a task everyone solves by reading one
number.

**Now.** The generator is retained unchanged -- it is behind the existing
figure, and as a documented negative example it is worth more than it would be
deleted -- but it is declared in `dataUtil.LEAKY_GENERATORS`, `require_clean()`
refuses it, and `ValidateDatasets.py` reports it. `make_latitude_bands` (the
axis-aligned control) and `make_tilted_bands` (needs both coordinates) replace
it.

## 3. The two expansion ladders are not two settings of one dial

**Assumed.** A target of harmonic degree ell becomes learnable at exactly rung
ell on either ladder.

**Measured.** Only the broadcast ladder behaves that way, and it does so
exactly. Residual energy above each rung's own degree:

| rung | broadcast ladder | re-uploading ladder |
|---|---|---|
| 1 | 8e-16 | 8e-16 |
| 2 | 1e-15 | 0.274 |
| 3 | 1e-15 | 0.251 |
| 4 | 2e-15 | 0.190 |

Re-uploading bounds the Fourier spectrum in the **encoding angles**, which is
not the same as bounding harmonic degree: at two uploads the accessible
functions include sin(theta) cos(2 phi), which is not a polynomial in (x, y, z)
and therefore has an infinite Legendre expansion. At one upload the two notions
coincide, which is why the degree-1 result is exact; they part company at two.

So one ladder raises harmonic degree and stays rotation-covariant, remaining a
fixed geodesic kernel that is Schoenberg-admissible at every rung; the other
raises coordinate frequency and leaves the Legendre basis altogether. This
sharpens the contrast the roadmap already gestured at, and it is pinned by
`tests/test_model_structure.py::test_reuploading_is_not_degree_limited`.

## 4. Faithfulness has not yet beaten the standard classical fix

**Assumed.** Faithful representations beat sine-cosine near the poles.

**Measured, preliminary.** Across three task families the ordering is
consistently **Cartesian ~ sine-cosine > raw**, with sine-cosine showing no
deficit anywhere:

| task | cartesian | sincos | raw |
|---|---|---|---|
| latitude bands, equatorial, n_train = 20 | 0.803 | 0.762 | 0.798 |
| latitude bands, polar, n_train = 20 | 0.781 | 0.793 | 0.822 |
| offset cap through the pole, local data | 0.887 | 0.908 | 0.848 |
| offset cap through the pole, global data | 0.805 | 0.863 | 0.748 |
| variation across tilts (lower is better) | 0.002 | 0.005 | **0.017** |

The mechanism is identifiable rather than mysterious. When the label depends
only on latitude, all three representations have latitude directly available --
sine-cosine carries cos(theta) -- so the longitude degeneracy at the pole never
bites, because longitude carries no signal there for any of them to misuse. The
one representation that consistently loses is the raw chart, whose longitude has
a hard discontinuity at the 0 / 2pi wrap.

**Consequence.** If this holds at full seeds, the claim narrows honestly: the
encoding supplies a prior that the **raw coordinate chart** lacks, while the
sine-cosine featurisation -- the standard practitioner's fix -- already supplies
most of it. That is materially weaker than the planned claim, because the raw
chart is not a serious competitor.

These numbers come from 6-12 seeds at reduced settings. They are a strong
signal, not a settled conclusion; the full sweeps are what would settle it.

## 5. Our own analysis was wrong about the latitude sweep

**Assumed** (in our implementation plan, not the original roadmap): sweeping
`latitude_center` confounds placement with difficulty, so it must be replaced by
rotation conjugation.

**Measured.** The affine ceiling is **0.884 at every placement** from 0.15 to
1.31, with balance 0.500 throughout. The label depends only on theta and the
noise is applied to theta, so moving the band changes neither the Bayes rate nor
the ceiling.

Rotation conjugation remains the better instrument -- it preserves the entire
joint distribution rather than only the ceiling, and extends to labels that are
not theta-only -- but the stated reason for preferring it was not correct, and
is corrected here rather than quietly dropped.

## 6. The dataset gate's first criterion was wrong too

The first version of `ValidateDatasets.py` flagged `quadrupole`, `sectoral` and
`banded_4` as leaking, because a single coordinate reached the affine ceiling.
That is not a defect: for a theta-only target the best degree-1 approximation IS
a theta cut, so the two numbers coincide by construction and say nothing about
the dataset.

The criterion is now role-dependent -- a task meant to be solvable is tested
differently from one meant to be out of reach -- and all seven generators pass
their own test. The gate's first act being to reject three healthy datasets on a
bad rule is a reasonable argument for having written it.

---

## Reproducing every number here

```
cd src/vqc_spherical
python ValidateDatasets.py          # tables in sections 2 and 6
python ExperimentAnalyticCore.py    # ceilings in section 1, residuals in 3
python ExperimentLadders.py --no-training   # the table in section 3
python ExperimentFaithfulness.py    # the table in section 4
python -m pytest ../../tests -q     # every closed form asserted
```
