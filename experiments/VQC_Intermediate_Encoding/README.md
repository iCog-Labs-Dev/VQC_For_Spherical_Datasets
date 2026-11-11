# VQC with Intermediate Data Encoding and Classical Baselines

This project compares a Variational Quantum Classifier (VQC) using repeated
intermediate RX-RZ data encoding with linear- and logistic-regression
baselines on the UCI Student Performance dataset.

## Project organization

```text
VQC_Intermediate_Encoding/
├── data/                         # Canonical CSV source (stored once)
├── shared/
│   ├── data.py                   # Shared loading, sampling, scaling, splitting
│   ├── optim.py                  # Shared NumPy Adam for classical baselines
│   └── plotting.py               # Shared academic blue plotting style
├── models/
│   ├── linear_regression/        # Linear model + MSE training
│   ├── logistic_regression/      # Logistic model + BCE training
│   └── vqc/
│       ├── encoding.py           # Feature encoding operations
│       ├── ansatz.py             # Trainable rotations and CNOT entanglers
│       ├── vqc.py                # Full QNode architecture and measurement
│       ├── training.py           # Quantum optimization loop
│       └── ...
├── findings/                     # Original research material and figures
├── results/                      # New training curves
└── requirements.txt
```

## Design decisions

- The CSV dataset is stored once under `data/`.
- `shared.data` owns common loading, feature selection, deterministic
  subsampling, standardization, and stratified train/test splitting.
- The shared loader supports signed labels (`-1`, `+1`) for the VQC and
  linear baseline, and binary labels (`0`, `1`) for logistic regression.
  This is only a label adapter; the feature preprocessing is shared.
- VQC-only clipping to `[-pi, pi]` remains in `models/vqc/datasets.py`.
  Quantum encoding is intentionally separate from classical preprocessing.
- The VQC's feature-encoding operations and trainable ansatz are separate
  modules. `vqc.py` composes them into the full circuit and performs
  measurement.
- Linear regression, logistic regression, and VQC retain distinct model,
  objective, and training implementations. The two classical baselines
  reuse the same Adam optimizer implementation.
- Figures use a consistent blue palette, restrained grid lines, and
  300-DPI saving for a more formal, academic presentation.

**Reproducibility note:** The original pipeline fitted `StandardScaler`
before splitting into training and test sets. The shared implementation
intentionally retains that behavior to avoid changing the experiment
during a structural refactor. It should be evaluated separately as a
potential data-leakage issue, not silently altered here.

## Installation

```bash
pip install -r requirements.txt
```

## Run the models

Run commands from the repository root:

```bash
python -m models.vqc.main
python -m models.linear_regression.main
python -m models.logistic_regression.main
```

Each model uses 100 sampled observations, the existing 80/20 stratified
split, four features, 50 epochs, batch size 8, and learning rate 0.05.
Model-specific training and objective functions remain separate.

## Preserved outputs

Existing PNGs and research files from the submitted project are retained
in their original relative locations under `findings/`, the baseline model
folders, and the project root. The refactored code writes newly generated
loss curves to `results/`; it does not overwrite the archived figures.
