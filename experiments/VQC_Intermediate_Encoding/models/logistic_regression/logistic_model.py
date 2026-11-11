import numpy as np
from .config import FEATURE_DIM

def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def logistic_model(weights, x):
    w, b = weights[:-1], weights[-1]
    return sigmoid(np.dot(w, x) + b)

def logistic_model_batch(weights, X):
    w, b = weights[:-1], weights[-1]
    return sigmoid(X @ w + b)

def init_weights():
    np.random.seed(42)
    return np.random.uniform(-1, 1, size=(FEATURE_DIM + 1,))

def describe_model(weights, sample_x):
    w, b = weights[:-1], weights[-1]
    terms = " + ".join(f"({wi:.3f} * x{i})" for i, wi in enumerate(w))
    return (f"p = sigmoid({terms} + ({b:.3f}))\n"
            f"Example input : {sample_x}\n"
            f"Example output: {logistic_model(weights, sample_x):.4f} (probability of Pass)")
