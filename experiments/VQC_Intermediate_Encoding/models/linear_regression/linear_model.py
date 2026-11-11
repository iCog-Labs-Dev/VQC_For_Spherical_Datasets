import numpy as np
from .config import FEATURE_DIM

def linear_model(weights, x):
    w, b = weights[:-1], weights[-1]
    return np.dot(w, x) + b

def linear_model_batch(weights, X):
    w, b = weights[:-1], weights[-1]
    return X @ w + b

def init_weights():
    np.random.seed(42)
    return np.random.uniform(-1, 1, size=(FEATURE_DIM + 1,))

def describe_model(weights, sample_x):
    w, b = weights[:-1], weights[-1]
    terms = " + ".join(f"({wi:.3f} * x{i})" for i, wi in enumerate(w))
    return (f"y = {terms} + ({b:.3f})\n"
            f"Example input : {sample_x}\n"
            f"Example output: {linear_model(weights, sample_x):.4f}")
