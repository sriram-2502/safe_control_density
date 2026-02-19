import numpy as np


def single_integrator_step(x, u, dt):
    """Euler step for x_dot = u."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    return x + dt * u
