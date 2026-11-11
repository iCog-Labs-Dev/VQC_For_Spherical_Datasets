import numpy as np
from .logistic_model import logistic_model_batch
from shared.optim import AdamOptimizer
from .config import LEARNING_RATE, EPOCHS, BATCH_SIZE

EPS = 1e-12

def binary_cross_entropy_loss(weights, X_batch, y_batch):
    probs = np.clip(logistic_model_batch(weights, X_batch), EPS, 1 - EPS)
    return -np.mean(y_batch * np.log(probs) + (1 - y_batch) * np.log(1 - probs))

def bce_gradient(weights, X_batch, y_batch):
    n = len(X_batch)
    probs = logistic_model_batch(weights, X_batch)
    error = probs - y_batch
    grad_w = (1.0 / n) * (X_batch.T @ error)
    grad_b = (1.0 / n) * np.sum(error)
    return np.concatenate([grad_w, [grad_b]])

def calculate_accuracy(weights, X, y):
    probs = logistic_model_batch(weights, X)
    binary_preds = (probs >= 0.5).astype(float)
    return np.mean(binary_preds == y)

def train(weights, X, y):
    opt = AdamOptimizer(stepsize=LEARNING_RATE)
    loss_history = []
    num_samples = len(X)
    for epoch in range(EPOCHS):
        indices = np.random.permutation(num_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, num_samples, BATCH_SIZE):
            X_batch = X_shuffled[i:i + BATCH_SIZE]
            y_batch = y_shuffled[i:i + BATCH_SIZE]
            weights, loss = opt.step_and_cost(
                lambda w: binary_cross_entropy_loss(w, X_batch, y_batch),
                weights,
                lambda w: bce_gradient(w, X_batch, y_batch)
            )
        epoch_loss = binary_cross_entropy_loss(weights, X, y)
        loss_history.append(epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = calculate_accuracy(weights, X, y)
            print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {acc * 100:.1f}%")
    return weights, loss_history
