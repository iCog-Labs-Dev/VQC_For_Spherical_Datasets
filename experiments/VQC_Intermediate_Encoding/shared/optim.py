"""Shared NumPy Adam optimizer used by both classical baselines."""
import numpy as np

class AdamOptimizer:
    def __init__(self, stepsize=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = stepsize, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step_and_cost(self, cost_fn, weights, grad_fn):
        loss = cost_fn(weights)
        grad = grad_fn(weights)
        if self.m is None:
            self.m, self.v = np.zeros_like(weights), np.zeros_like(weights)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        new_weights = weights - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return new_weights, loss
