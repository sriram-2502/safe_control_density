import numpy as np
import scipy.linalg as la

from density_utils.density import density_grad


def _lqr_gain(n, dt, q_lqr, r_lqr):
    """Compute discrete-time LQR gain for x_{k+1} = x_k + dt * u_k."""
    if np.isscalar(q_lqr):
        q = float(q_lqr) * np.eye(n)
    else:
        q = np.asarray(q_lqr, dtype=float)
    if np.isscalar(r_lqr):
        r = float(r_lqr) * np.eye(n)
    else:
        r = np.asarray(r_lqr, dtype=float)

    if q.shape != (n, n) or r.shape != (n, n):
        raise ValueError("q_lqr and r_lqr must be scalars or (n,n) arrays")

    a = np.eye(n, dtype=float)
    b = dt * np.eye(n, dtype=float)
    p = la.solve_discrete_are(a, b, q, r)
    bt_p = b.T @ p
    return np.linalg.solve(bt_p @ b + r, bt_p @ a)


def density_feedback_control(
    x,
    goal,
    alpha,
    obstacles,
    ctrl_multiplier=10.0,
    rad_from_goal=0.1,
    q_lqr=1.0,
    r_lqr=1.0,
    dt=0.02,
    saturation=2.0,
):
    """Density feedback control with a local LQR goal-tracking fallback."""
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if np.linalg.norm(x - goal) < rad_from_goal:
        k_lqr = _lqr_gain(x.size, dt, q_lqr, r_lqr)
        u = -k_lqr @ (x - goal)
    else:
        u = ctrl_multiplier * density_grad(x, goal, alpha, obstacles)

    max_u = np.max(np.abs(u))
    if max_u > saturation:
        u = u / max_u * saturation
    return u

