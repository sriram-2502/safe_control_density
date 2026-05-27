"""Discrete density-function safety filter for single-integrator examples.

The optimizer keeps the command close to a nominal single-integrator control
while enforcing the discrete MPC-CDF density residual for each obstacle.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize

from density_utils.controllers.solver_utils import require_solver
from density_utils.density import density_value
from density_utils.dynamics import single_integrator_step


@dataclass(frozen=True)
class DensityFilterResult:
    u: np.ndarray
    success: bool
    slack: np.ndarray
    objective: float
    message: str


def _as_bound_vector(value, control_dim, name):
    value = np.asarray(value, dtype=float)
    if value.shape == ():
        return np.full(control_dim, float(value))
    if value.shape != (control_dim,):
        raise ValueError(f"{name} must be a scalar or a vector with shape ({control_dim},)")
    return value


def _control_bounds(u_min, u_max, control_dim):
    lower = _as_bound_vector(u_min, control_dim, "u_min")
    upper = _as_bound_vector(u_max, control_dim, "u_max")
    if np.any(lower > upper):
        raise ValueError("u_min must be less than or equal to u_max componentwise")
    return list(zip(lower, upper))


def _clip_to_bounds(u, bounds):
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    return np.clip(u, lower, upper)


def _as_weight_matrix(weight, control_dim):
    if np.isscalar(weight):
        return float(weight) * np.eye(control_dim)
    weight = np.asarray(weight, dtype=float)
    if weight.shape != (control_dim, control_dim):
        raise ValueError("control_weight must be a scalar or a (control_dim, control_dim) array")
    return weight


def _eval_scalar(value, x, default):
    if value is None:
        return float(default)
    if callable(value):
        value = value(x)
    return float(value)


def _density_for_obstacle(density_fn, x, goal, alpha, obstacle):
    return float(density_fn(x, goal, alpha, [obstacle]))


def solve_discrete_density_filter(
    x,
    goal,
    alpha,
    obstacles,
    *,
    u_nom,
    dt,
    next_state_fn=single_integrator_step,
    u_min=-2.0,
    u_max=2.0,
    divergence=None,
    slack_weight=1e4,
    control_weight=1.0,
    min_density=1e-10,
    density_fn=density_value,
    solver="auto",
    return_info=False,
):
    """Solve one discrete MPC-CDF density-filter step for ``x_dot = u``.

    For each obstacle, the constraint is

        rho(x_next) - rho(x) + dt * div(F_d)(x) * rho(x) + s >= 0,

    where ``x_next = x + dt * u`` by default and ``s >= 0`` is penalized.
    A custom ``next_state_fn`` may be passed, but this controller expects the
    control and state to have the same dimension, as in the single-integrator
    examples.
    """
    require_solver(solver, ("scipy_slsqp",), controller="solve_discrete_density_filter")
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    u_nom = np.asarray(u_nom, dtype=float)
    obstacles = list(obstacles)

    if x.ndim != 1:
        raise ValueError("x must be a vector")
    if goal.shape != x.shape:
        raise ValueError("goal must have the same dimension as x")
    if u_nom.ndim != 1:
        raise ValueError("u_nom must be a vector")

    control_dim = u_nom.size
    bounds = _control_bounds(u_min, u_max, control_dim)
    u_ref = _clip_to_bounds(u_nom, bounds)

    if not obstacles:
        result = DensityFilterResult(
            u=u_ref,
            success=True,
            slack=np.zeros(0),
            objective=0.0,
            message="",
        )
        return result if return_info else result.u

    div_value = _eval_scalar(divergence, x, default=0.0)

    def next_state(u_eval):
        x_next = np.asarray(next_state_fn(x, u_eval, dt), dtype=float)
        if x_next.shape != x.shape:
            raise ValueError("next_state_fn must return a vector with the same shape as x")
        return x_next

    constraints = []
    slack_init = []

    for obstacle in obstacles:
        rho = _density_for_obstacle(density_fn, x, goal, alpha, obstacle)
        rho = max(rho, float(min_density))
        rho_next_ref = _density_for_obstacle(
            density_fn,
            next_state(u_ref),
            goal,
            alpha,
            obstacle,
        )
        residual_ref = rho_next_ref - rho + dt * div_value * rho
        slack_init.append(max(0.0, -residual_ref))

        slack_index = control_dim + len(constraints)

        def make_constraint(obstacle=obstacle, rho_current=rho, slack_index=slack_index):
            def constraint(z):
                rho_next = _density_for_obstacle(
                    density_fn,
                    next_state(z[:control_dim]),
                    goal,
                    alpha,
                    obstacle,
                )
                residual = rho_next - rho_current + dt * div_value * rho_current
                return residual + z[slack_index]

            return constraint

        constraints.append({"type": "ineq", "fun": make_constraint()})

    slack_init = np.asarray(slack_init, dtype=float)
    z0 = np.concatenate([u_ref, slack_init])
    w_u = _as_weight_matrix(control_weight, control_dim)
    slack_weight = float(slack_weight)

    def objective(z):
        u = z[:control_dim]
        slack = z[control_dim:]
        du = u - u_nom
        return 0.5 * float(du @ w_u @ du) + 0.5 * slack_weight * float(slack @ slack)

    def objective_jac(z):
        grad = np.zeros_like(z)
        grad[:control_dim] = w_u @ (z[:control_dim] - u_nom)
        grad[control_dim:] = slack_weight * z[control_dim:]
        return grad

    opt_bounds = bounds + [(0.0, None)] * len(constraints)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step",
            category=RuntimeWarning,
        )
        sol = minimize(
            objective,
            z0,
            jac=objective_jac,
            bounds=opt_bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 100, "disp": False},
        )

    z = sol.x if sol.success else z0
    u = _clip_to_bounds(z[:control_dim], bounds)
    slack = np.maximum(z[control_dim:], 0.0)
    result = DensityFilterResult(
        u=u,
        success=bool(sol.success),
        slack=slack,
        objective=float(objective(np.concatenate([u, slack]))),
        message=str(sol.message),
    )
    return result if return_info else result.u
