# Experiment 2: Finite-Shot Spherical Fidelity Kernels

This experiment studies how finite-shot estimates of a spherical fidelity kernel affect kernel ridge regression (KRR), generalization, and spherical-harmonic recovery. Experiment 1 provides the exact-kernel reference.

## Kernel

For unit vectors $x, y \in S^2$ and copy count $n \in \{1, 2, 4, 8\}$:

$$
k_n(x, y) = \left(\frac{1 + x \cdot y}{2}\right)^n
$$

Each shot returns a binary outcome with success probability $k_n(x, y)$. The sample mean is an unbiased estimate with variance $k_n(1 - k_n)/S$, where $S$ is the number of shots.

## Experimental Design

The experiment compares four conditions using the same data and targets:

* **`exact_exact`:** Exact training and prediction kernels.
* **`sampled_exact`:** Sampled training kernel, exact prediction kernel.
* **`exact_sampled`:** Exact training kernel, sampled prediction kernel.
* **`sampled_sampled`:** Sampled training and prediction kernels.

The default configuration varies copy count, shot count, training-set size, and spherical-harmonic target. It selects the KRR regularization parameter using validation data only. The `fixed_budget` configuration compares different allocations of a total measurement-shot budget.

### Measurement Backends

* **`bernoulli_reference` (default):** Samples shot noise directly from the exact kernel probability. This is a shot-noise simulation, not a circuit result.
* **`pennylane_simulator`:** Uses a shot-sampled overlap circuit on PennyLane's noiseless `default.qubit` simulator.

## Setup and Run


Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

Run an experiment from this directory:

```bash
python run_experiment.py --config default
```

Other configurations:

```bash
python run_experiment.py --config smoke
python run_experiment.py --config fixed_budget
```

To use the PennyLane backend:

```bash
python run_experiment.py --config smoke --backend pennylane_simulator
```

Optional arguments: `--workers N` enables parallel processes, and `--output-dir PATH` sets the output directory.

Run tests:

```bash
python -m pytest
```

## Results

The included `results/default/` directory contains the recorded experiment outputs: per-run metrics, summary metrics, kernel diagnostics, plots, and a run manifest. New runs are written to `results/<config>/` by default.
