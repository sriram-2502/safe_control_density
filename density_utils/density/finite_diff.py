import numpy as np


def finite_difference_grad(fn, x, eps=1e-3):
    """Central finite-difference gradient for small-dimensional inputs."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        x_f = x.copy()
        x_b = x.copy()
        x_f[i] += eps
        x_b[i] -= eps
        grad[i] = (fn(x_f) - fn(x_b)) / (2.0 * eps)
    return grad
