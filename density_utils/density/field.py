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
