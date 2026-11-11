import numpy as np
from .linear_model import linear_model_batch
from shared.optim import AdamOptimizer
from .config import LEARNING_RATE, EPOCHS, BATCH_SIZE

def mean_squared_error_loss(weights, X_batch, y_batch):
    predictions = linear_model_batch(weights, X_batch)
    return np.mean((predictions - y_batch) ** 2)

def mse_gradient(weights, X_batch, y_batch):
    n = len(X_batch)
    w, b = weights[:-1], weights[-1]
    error = X_batch @ w + b - y_batch
    grad_w = (2.0 / n) * (X_batch.T @ error)
    grad_b = (2.0 / n) * np.sum(error)
    return np.concatenate([grad_w, [grad_b]])

def calculate_accuracy(weights, X, y):
    return np.mean(np.sign(linear_model_batch(weights, X)) == y)

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
                lambda w: mean_squared_error_loss(w, X_batch, y_batch),
                weights,
                lambda w: mse_gradient(w, X_batch, y_batch)
            )
        epoch_loss = mean_squared_error_loss(weights, X, y)
        loss_history.append(epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = calculate_accuracy(weights, X, y)
            print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {acc * 100:.1f}%")
    return weights, loss_history
