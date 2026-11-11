"""Single source of truth for dataset loading and common preprocessing."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
FEATURE_COLUMNS = ["studytime", "failures", "absences", "G1"]
DEFAULT_SEED = 42

@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

def load_academic_performance_data(n_samples=200, seed=DEFAULT_SEED,
                                   label_encoding="signed"):
    """
    Load, sample, scale, and split the UCI Student Performance dataset.

    label_encoding:
        "signed": fail=-1, pass=+1 (VQC and linear regression)
        "binary": fail=0, pass=1 (logistic regression)

    The scaler is intentionally fitted before the train/test split to
    preserve the behavior of the original experiment implementations.
    """
    if label_encoding not in {"signed", "binary"}:
        raise ValueError("label_encoding must be 'signed' or 'binary'")

    np.random.seed(seed)
    print("Loading UCI Student Performance dataset...")
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df[FEATURE_COLUMNS].values
    passed = df["G3"].values >= 10
    y = passed.astype(float) if label_encoding == "binary" else np.where(passed, 1.0, -1.0)

    if n_samples is not None and n_samples < len(X):
        indices = np.random.choice(len(X), size=n_samples, replace=False)
        X, y = X[indices], y[indices]

    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=seed, stratify=y
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)
