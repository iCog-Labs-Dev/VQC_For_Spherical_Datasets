# Variational quantum circuits on pure spherical manifolds

Whether the way spherical data is loaded into a quantum circuit imposes a
useful bias on what the circuit can learn.

Two properties follow from the encoding alone, and both are computable in
closed form:

**Faithfulness** -- the induced kernel `k(gamma) = (1 + cos gamma)/2` depends on
the geodesic angle and nothing else, so rotating the coordinate frame changes no
similarity value. The representation describes points, not coordinates.

**Restriction** -- that kernel's Legendre expansion terminates at degree 1.
Equivalently: the encoded state is affine in the Bloch vector r and every later
operation is linear on it, so the circuit's output is `alpha + beta.r` exactly
-- four real parameters, whatever the depth. Its decision boundary is one
circle on the sphere and nothing else.

No claim of quantum advantage is made. The model is classically simulable, and
a four-parameter classical model reproduces it on every task within its range.
That is reported as a finding and appears in every comparison table.

## Running it

```bash
pip install -r requirements.txt
pip install -e . --no-deps

python -m vqc_spherical.RunAll --phase 0  # dataset gate and analytic core
python -m vqc_spherical.RunAll --phase 1  # spectra and falsification
python -m vqc_spherical.RunAll --phase 2  # both expansion ladders
python -m vqc_spherical.RunAll --phase 3  # faithfulness and sample efficiency
python -m vqc_spherical.RunAll --figures  # redraw figures from results/*.csv

python -m pytest tests -q      # from the repository root
```

Phases 0 and 1's analytic stages involve **no training** and finish in minutes;
they carry most of the analytic content and are the gate on everything else.
Results are written to `results/` as CSV and figures to `figures/`; experiments
never draw, and `Plots.py` never computes.

## Layout

| path | what it is |
|---|---|
| `EmbeddingLayer.py` | the encoding, `spherical` and `broadcast` (spin-n/2) |
| `AnsatzLayer.py`, `VQCModel.py`, `VQCOptimizer.py` | circuit, training loop, data re-uploading |
| `Kernels.py` | every kernel in closed form, Gram spectra, analytic references |
| `Analysis.py` | affine ceiling, harmonic projection, coordinate leakage |
| `datasets.py` | datasets, rigid rotations, and the target registry |
| `ClassicalBaselines.py` | five controls, including the computed ceiling as a model |
| `ValidateDatasets.py` | the dataset gate -- run it before trusting a dataset |
| `Experiment*.py`, `RunAll.py` | the four stages and the phase driver |
| `Plots.py` | all figures, drawn from results files alone |
| `tests/` | the closed forms, asserted |

## Two documents worth reading first

`PREDICTIONS.md` -- what is predicted, with computed constants, registered
before the sweeps run.

`CORRECTIONS.md` -- what the original research roadmap assumed against what
measurement showed. Six items, two of them errors in our own analysis.

## A note on the deprecated dataset

`make_sphere_moons` draws longitude from `(0, pi)` for one class and
`(pi, 2pi)` for the other, so longitude alone classifies it perfectly and every
model scores 100%. It is kept unchanged, because it is behind the existing
figure and is a useful recorded instance of a saturated benchmark, but
`require_clean()` refuses it and `ValidateDatasets.py` reports it. Use
`tilted_bands` (needs both coordinates) or `latitude_bands` (the axis-aligned
control) instead.
