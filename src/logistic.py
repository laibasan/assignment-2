"""Task 3: sigmoid, likelihood, and batch gradient descent from scratch.

X contains an explicit first column of ones. The caller owns standardisation.
"""
import numpy as np


def sigmoid(z):
    # This identity is stable even at +/-1000; no overflow of exp(-z).
    z = np.asarray(z, dtype=float)
    exp_neg_abs = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exp_neg_abs), exp_neg_abs / (1 + exp_neg_abs))


def hypothesis(X, theta):
    return sigmoid(np.asarray(X) @ np.asarray(theta))


def cost(X, y, theta):
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)))


def fit_logistic(X, y, alpha, n_iters):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or not len(y):
        raise ValueError("X must be a matrix with one binary label per row")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonfinite features or nonbinary labels")
    if alpha <= 0 or n_iters <= 0:
        raise ValueError("alpha and n_iters must be positive")
    theta = np.zeros(X.shape[1])
    history = np.empty(n_iters)
    for iteration in range(n_iters):
        h = hypothesis(X, theta)
        theta += alpha * (X.T @ (y - h)) / len(y)
        history[iteration] = cost(X, y, theta)
    return theta, history


def predict(X, theta):
    return (hypothesis(X, theta) >= 0.5).astype(int)
