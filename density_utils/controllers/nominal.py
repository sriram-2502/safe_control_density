import numpy as np
import scipy.linalg as la

from density_utils.density import density_grad


def _lqr_gain(n, dt, q_lqr, r_lqr):
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


def _clip_by_bounds(u, u_min, u_max):
    u = np.asarray(u, dtype=float)
    u_min = np.asarray(u_min, dtype=float)
    u_max = np.asarray(u_max, dtype=float)
    if u_min.shape == ():
        u_min = np.full_like(u, float(u_min))
    if u_max.shape == ():
        u_max = np.full_like(u, float(u_max))
    return np.clip(u, u_min, u_max)


def _goal_direction_control(x, goal, ctrl_multiplier):
    direction = goal - x
    dist = np.linalg.norm(direction)
    if dist < 1e-12:
        return np.zeros_like(x)
    return ctrl_multiplier * direction / dist


def single_integrator_nominal_control(
    x,
    goal,
    alpha,
    obstacles,
    *,
    mode="goal",
    ctrl_multiplier=2.0,
    rad_from_goal=0.1,
    q_lqr=1.0,
    r_lqr=1.0,
    dt=0.02,
    u_min=-2.0,
    u_max=2.0,
    density_weight=1.0,
    goal_weight=0.5,
    lookahead_distance=1.0,
):
    """Reference control choices for single-integrator Density filter examples."""
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    obstacles = list(obstacles)
    mode = mode.lower().replace("-", "_")
    dist = np.linalg.norm(x - goal)

    if mode in ("goal", "straight", "straight_line"):
        if dist < rad_from_goal:
            k_lqr = _lqr_gain(x.size, dt, q_lqr, r_lqr)
            u = -k_lqr @ (x - goal)
        else:
            u = _goal_direction_control(x, goal, ctrl_multiplier)
    elif mode == "lqr":
        k_lqr = _lqr_gain(x.size, dt, q_lqr, r_lqr)
        u = -k_lqr @ (x - goal)
    elif mode in ("density", "density_feedback"):
        if dist < rad_from_goal:
            k_lqr = _lqr_gain(x.size, dt, q_lqr, r_lqr)
            u = -k_lqr @ (x - goal)
        else:
            u = ctrl_multiplier * density_grad(x, goal, alpha, obstacles)
    elif mode in ("density_blend", "blend"):
        if dist < rad_from_goal:
            k_lqr = _lqr_gain(x.size, dt, q_lqr, r_lqr)
            u = -k_lqr @ (x - goal)
        else:
            u_density = density_weight * density_grad(x, goal, alpha, obstacles)
            u_goal = goal_weight * _goal_direction_control(x, goal, 1.0)
            u = ctrl_multiplier * (u_density + u_goal)
    elif mode in ("pure_pursuit", "pursuit"):
        speed_scale = min(1.0, dist / max(float(lookahead_distance), 1e-12))
        u = speed_scale * _goal_direction_control(x, goal, ctrl_multiplier)
    else:
        raise ValueError(
            "mode must be one of: goal, lqr, density, density_blend, pure_pursuit"
        )

    return _clip_by_bounds(u, u_min, u_max)
