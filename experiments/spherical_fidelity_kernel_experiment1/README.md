# Spherical Fidelity Kernel — Experiment 1

## Overview

This experiment investigates how the spectral support of a spherical fidelity kernel affects the recovery of spherical-harmonic functions on the unit sphere, \(S^2\).

The kernel is

$$
k_n(x,y)=\left(\frac{1+x\cdot y}{2}\right)^n,
$$

where \(x,y\in S^2\), and \(n\) is both the number of copies of the encoded single-qubit state and the total number of qubits used by the overlap circuit.

## Research Question

> How do kernel spectral support, regularization, training-set size, and target harmonic composition affect the recovery of functions on \(S^2\)?

This experiment establishes a controlled baseline for studying recoverability with the ideal exact kernel, before introducing finite-shot measurement noise in future experiments. End-to-end finite-shot kernel ridge regression (KRR) is not part of the current protocol.

## The Quantum Component

The repository includes a PennyLane overlap circuit. Points on the sphere are encoded as single-qubit states. The circuit prepares the state for one point, applies the inverse preparation for the other point across the \(n\) qubits, and measures the probability of obtaining the all-zero bitstring. In the ideal setting, this probability matches the analytical kernel above.

- **Circuit-validation pilot:** Uses PennyLane's analytic simulator (`shots=None`) to check that circuit probabilities, kernel matrices, and downstream predictions agree with the closed-form calculation.
- **Main parameter grid:** Uses the closed-form kernel rather than executing a circuit for every matrix entry, making the main sweep faster and reproducible.
- **Hardware:** No physical quantum hardware is used.

This comparison validates the implementation of the ideal kernel; it does not demonstrate quantum advantage, speedup, or hardware superiority.

## Experimental Design

### 1. Exact-Kernel Grid

The default settings are:

| Parameter | Default settings |
|---|---|
| Kernel copies / qubits (\(n\)) | 1, 2, 4, 8 |
| Target degrees (\(\ell\)) | Every integer from 0 through 10 |
| Training sizes (\(N\)) | 32, 64, 128, 256 |
| Ridge regularization (\(\tau\)) | \(10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\) |
| Target families | Zonal harmonics and random real-harmonic combinations |
| Training seeds | 20 |
| Independent random target realizations | 3 per degree for the random-harmonic family |
| Fixed held-out test set | 1,000 points on \(S^2\) |
| Fixed validation set | 400 points on \(S^2\) |

For a given seed and training-set size, the training points are shared across kernel copy counts and model families. The validation and test sets remain fixed across the grid. Each zonal target is evaluated once per degree, while the random-harmonic family uses multiple independent coefficient realizations.

### 2. Why Harmonic Degree Matters

The spherical-harmonic eigenvalues of the kernel follow this expression:

$$
\lambda_\ell^{(n)} =
\begin{cases}
\dfrac{(n!)^2}{(n-\ell)!(n+\ell+1)!}, & 0 \leq \ell \leq n,\\[6pt]
0, & \ell > n.
\end{cases}
$$

The exact kernel therefore has spectral support only through harmonic degree \(n\). Harmonic components above \(n\) lie outside its function space. However, a nonzero eigenvalue for \(\ell\leq n\) does not guarantee easy recovery: higher supported degrees can have very small eigenvalues, making them harder to learn under finite sampling and regularization.

### 3. Classical Baselines

We compare the fidelity kernel against two classical spherical-harmonic ridge-regression baselines:

1. **Matched-span baseline:** Uses harmonic features through degree \(n\), matching the nominal supported span of the fidelity kernel.
2. **Higher-capacity reference:** Uses harmonic features through degree 10, the maximum target degree in the grid.

These comparisons help distinguish limitations caused by spectral support from effects of regularization and sample size. The matched-span baseline does not use the same reproducing-kernel Hilbert space (RKHS) penalty as the fidelity kernel, and the higher-capacity baseline has direct access to the basis used to construct the synthetic targets. The baselines are diagnostic comparisons, not tests of quantum advantage.

### 4. Regularization and Model Selection

Test-error curves across the predefined `tau` grid are retained for sensitivity analysis. For model selection, each model family selects `tau` using the minimum validation MSE, then reports performance on the held-out test set. The test set is not used for model selection.

### 5. Outputs and Metrics

Runs save `results.csv`, `summary.csv`, and `config.json`, as well as diagnostic plots. The pilot also saves `pilot.json` when it is enabled.

Key metrics include:

- Test mean squared error (MSE) and normalized MSE (NMSE)
- Empirical target-amplitude recovery and correlation
- Population eigenvalues (\(\lambda_\ell\)), theoretical shrinkage (\(g_\ell\)), and effective dimension
- Validation-selected regularization and held-out test performance
- Comparisons with the harmonic-ridge baselines
- Recovery trends across target degree and training-set size

Because the kernel's spectrum is known analytically, these results illustrate its behavior in a controlled, finite-sample synthetic setting. They do not establish universal approximation.

## Main Outcomes

The experiment distinguishes between **representational support** and **learnability**:

- If \(\ell>n\), a pure harmonic component lies outside the exact kernel's spectral support. More training data cannot make that component representable by this finite-\(n\) kernel.
- If \(\ell\leq n\), the component is supported, but a small eigenvalue can still make recovery difficult, especially with limited data or strong regularization.
- Recovery as the training set grows can vary substantially across supported degrees.

The synthetic targets make it possible to isolate spectral effects under controlled conditions, but the findings do not automatically generalize to real-world spherical datasets.

## Installation and Testing

The following commands are intended for Windows PowerShell. Run them from the `spherical_fidelity_kernel_experiment1` directory.

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

## Running the Experiments

### Run the analytic circuit-validation pilot

```powershell
python -m experiments.experiment_1 --pilot
```

### Run a lightweight smoke test

This checks that the code runs; it is not intended to produce final results.

```powershell
python -m experiments.experiment_1 --quick
```

### Run the full default grid

```powershell
python -m experiments.experiment_1
```

### Optional flags

```powershell
python -m experiments.experiment_1 --no-quantum
python -m experiments.experiment_1 --copies 1 2 4 --degrees 0 1 2 3 4
python -m experiments.experiment_1 --seeds 0 1 2 --target-seeds 0 1
python -m experiments.experiment_1 --taus 0.00001 0.0001 0.001 0.01 0.1
```

Using `--no-quantum` skips the analytic circuit pilot while leaving the main exact-kernel grid enabled. The `--out results/smoke` option can be used to keep a new run's outputs separate from previous results.

## Cautions and Limitations

- Spectral support ends at \(\ell\leq n\), but support alone does not guarantee easy recovery.
- More training data cannot make an unsupported harmonic component representable by this exact finite-\(n\) kernel.
- Multiple synthetic target families test behavior across the chosen target types; they do not prove universal approximation.
- The main grid uses the closed-form expression for an ideal quantum fidelity kernel. The PennyLane pilot uses an analytic simulator, not physical quantum hardware.
- This experiment studies kernel expressivity and recoverability; it does not assess quantum advantage or hardware performance.

## Project Layout

```text
spherical_fidelity_kernel_experiment1/
├── experiments/
│   ├── __init__.py
│   └── experiment_1.py
├── src/
│   ├── __init__.py
│   ├── kernel.py
│   ├── metrics.py
│   ├── quantum_kernel.py
│   ├── regression.py
│   ├── spectrum.py
│   ├── sphere.py
│   └── targets.py
├── tests/
│   └── test_foundations.py
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

The `results/` directory is generated when experiments are run and is not included in the committed project layout by default.