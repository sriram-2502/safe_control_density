import numpy as np


def unicycle_step(state, v, omega, dt):
    """Euler step for unicycle dynamics."""
    state = np.asarray(state, dtype=float)
    x, y, theta = state
    x_next = x + dt * v * np.cos(theta)
    y_next = y + dt * v * np.sin(theta)
    theta_next = theta + dt * omega
    return np.array([x_next, y_next, theta_next], dtype=float)
