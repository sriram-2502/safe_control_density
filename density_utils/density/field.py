from dataclasses import dataclass

import numpy as np

from .bump import p_norm_bump
from .finite_diff import finite_difference_grad


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray
    r1: float
    r2: float
    p: float = 2.0
    scale: np.ndarray | None = None
    angle: float = 0.0


def density_value(x, goal, alpha, obstacles, min_dist=1e-3):
    """Compute density value g(x) * bump(x) for a list of obstacles."""
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    dist = max(np.linalg.norm(x - goal), min_dist)
    g = 1.0 / (dist ** (2.0 * alpha))

    bump_val = 1.0
    for obs in obstacles:
        bump_val *= p_norm_bump(
            x,
            obs.center,
            obs.r1,
            obs.r2,
            p=obs.p,
            scale=obs.scale,
            angle=obs.angle,
        )
    return g * bump_val


def density_grad(x, goal, alpha, obstacles, eps=1e-3):
    """Finite-difference gradient of the density field."""
    return finite_difference_grad(
        lambda x_eval: density_value(x_eval, goal, alpha, obstacles),
        x,
        eps=eps,
    )


def full_state_density_value(
    x,
    goal,
    alpha,
    obstacles,
    *,
    position_indices=(0, 1),
    P=None,
    min_v=1e-6,
):
    """Compute ``rho(x) = Phi(position) / V(x)^alpha``.

    ``Phi`` is the obstacle bump product evaluated on the position entries of
    the state, while ``V`` is a quadratic full-state Lyapunov function around
    ``goal``.  This matches the CDF construction used in the MPC-CDF examples
    while still supporting 2D obstacle geometry.
    """
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if goal.shape != x.shape:
        raise ValueError("goal must have the same dimension as x")

    if P is None:
        P = np.eye(x.size, dtype=float)
    else:
        P = np.asarray(P, dtype=float)
    if P.shape != (x.size, x.size):
        raise ValueError("P must have shape (state_dim, state_dim)")

    err = x - goal
    v = max(float(err @ P @ err), float(min_v))
    if isinstance(position_indices, slice):
        pos = x[position_indices]
    else:
        pos = x[np.asarray(position_indices, dtype=int)]

    phi = 1.0
    for obs in obstacles:
        phi *= p_norm_bump(
            pos,
            obs.center,
            obs.r1,
            obs.r2,
            p=obs.p,
            scale=obs.scale,
            angle=obs.angle,
        )
    return phi / (v ** float(alpha))


def full_state_density_grad(
    x,
    goal,
    alpha,
    obstacles,
    *,
    position_indices=(0, 1),
    P=None,
    min_v=1e-6,
    eps=1e-3,
):
    """Finite-difference gradient of the full-state CDF density."""
    return finite_difference_grad(
        lambda x_eval: full_state_density_value(
            x_eval,
            goal,
            alpha,
            obstacles,
            position_indices=position_indices,
            P=P,
            min_v=min_v,
        ),
        x,
        eps=eps,
    )
