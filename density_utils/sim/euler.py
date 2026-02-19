import numpy as np


def forward_euler(x, u, dt):
    """Forward Euler integrator."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    return x + dt * u
