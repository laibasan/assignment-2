"""Task 5 metrics implemented directly with NumPy, rows = actual labels."""
import numpy as np


def _labels(y_true, y_pred):
    a, b = np.asarray(y_true), np.asarray(y_pred)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Expected equally sized, nonempty one-dimensional label arrays")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Labels must be binary")
    return a, b


def accuracy(y_true, y_pred):
    a, b = _labels(y_true, y_pred)
    return float(np.sum(a == b) / len(a))


def confusion_matrix(y_true, y_pred):
    a, b = _labels(y_true, y_pred)
    return np.array([[np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))],
                     [np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))]], dtype=int)
