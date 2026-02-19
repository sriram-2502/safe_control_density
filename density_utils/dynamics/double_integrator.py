import numpy as np


def double_integrator_step(state, u, dt):
    """Euler step for x_dot = v, v_dot = u."""
    state = np.asarray(state, dtype=float)
    u = np.asarray(u, dtype=float)
    next_state = state.copy()
    next_state[:2] = state[:2] + state[2:] * dt
    next_state[2:] = state[2:] + u * dt
    return next_state
