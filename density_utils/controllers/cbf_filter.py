"""Discrete-time CBF safety filter."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize

from density_utils.dynamics import single_integrator_step


@dataclass(frozen=True)
class CBFFilterResult:
    u: np.ndarray
    success: bool
    slack: np.ndarray
    clf_slack: float
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


def solve_cbf_filter(
    x,
    *,
    h_fns,
    dt,
    u_nom=None,
    next_state_fn=single_integrator_step,
    gamma=0.5,
    clf_fn=None,
    clf_rate=0.25,
    u_min=-2.0,
    u_max=2.0,
    slack_weight=1e6,
    clf_slack_weight=1e3,
    control_weight=1.0,
    return_info=False,
):
    """Solve one discrete-time CBF or CLF-CBF filter step.

    For each barrier ``h_i(x) >= 0``, the discrete-time condition is

        h_i(x_next) - (1 - gamma) h_i(x) + s_i >= 0,

    with ``s_i >= 0`` penalized in the objective. If ``clf_fn`` is provided,
    an additional relaxed discrete-time CLF condition is imposed:

        V(x_next) - (1 - clf_rate) V(x) <= delta,

    where ``delta >= 0`` is penalized. If ``u_nom`` is omitted, the objective
    minimizes control effort directly.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a vector")
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if clf_fn is not None and not 0.0 < float(clf_rate) <= 1.0:
        raise ValueError("clf_rate must be in (0, 1]")

    if u_nom is None:
        control_dim = x.size
        u_nom = np.zeros(control_dim, dtype=float)
    else:
        u_nom = np.asarray(u_nom, dtype=float)
        if u_nom.ndim != 1:
            raise ValueError("u_nom must be a vector")
        control_dim = u_nom.size
    bounds = _control_bounds(u_min, u_max, control_dim)
    u_ref = _clip_to_bounds(u_nom, bounds)
    h_fns = list(h_fns)

    if not h_fns and clf_fn is None:
        result = CBFFilterResult(
            u=u_ref,
            success=True,
            slack=np.zeros(0),
            clf_slack=0.0,
            objective=0.0,
            message="",
        )
        return result if return_info else result.u

    def next_state(u_eval):
        x_next = np.asarray(next_state_fn(x, u_eval, dt), dtype=float)
        if x_next.shape != x.shape:
            raise ValueError("next_state_fn must return a vector with the same shape as x")
        return x_next

    constraints = []
    slack_init = []
    gamma = float(gamma)
    for h_fn in h_fns:
        h_current = float(h_fn(x))
        h_next_ref = float(h_fn(next_state(u_ref)))
        residual_ref = h_next_ref - (1.0 - gamma) * h_current
        slack_init.append(max(0.0, -residual_ref))
        slack_index = control_dim + len(constraints)

        def make_constraint(h_fn=h_fn, h_current=h_current, slack_index=slack_index):
            def constraint(z):
                h_next = float(h_fn(next_state(z[:control_dim])))
                residual = h_next - (1.0 - gamma) * h_current
                return residual + z[slack_index]

            return constraint

        constraints.append({"type": "ineq", "fun": make_constraint()})

    slack_init = np.asarray(slack_init, dtype=float)
    cbf_slack_count = len(constraints)
    z_parts = [u_ref, slack_init]
    clf_slack_index = None
    if clf_fn is not None:
        v_current = float(clf_fn(x))
        v_next_ref = float(clf_fn(next_state(u_ref)))
        clf_residual_ref = v_next_ref - (1.0 - float(clf_rate)) * v_current
        clf_delta_init = max(0.0, clf_residual_ref)
        clf_slack_index = control_dim + cbf_slack_count
        z_parts.append(np.array([clf_delta_init], dtype=float))

        def clf_constraint(z, v_current=v_current):
            v_next = float(clf_fn(next_state(z[:control_dim])))
            return z[clf_slack_index] - (v_next - (1.0 - float(clf_rate)) * v_current)

        constraints.append({"type": "ineq", "fun": clf_constraint})

    z0 = np.concatenate(z_parts)
    w_u = _as_weight_matrix(control_weight, control_dim)
    slack_weight = float(slack_weight)
    clf_slack_weight = float(clf_slack_weight)

    def objective(z):
        u = z[:control_dim]
        cbf_slack = z[control_dim : control_dim + cbf_slack_count]
        du = u - u_nom
        cost = 0.5 * float(du @ w_u @ du)
        cost += 0.5 * slack_weight * float(cbf_slack @ cbf_slack)
        if clf_slack_index is not None:
            cost += 0.5 * clf_slack_weight * float(z[clf_slack_index] ** 2)
        return cost

    def objective_jac(z):
        grad = np.zeros_like(z)
        grad[:control_dim] = w_u @ (z[:control_dim] - u_nom)
        grad[control_dim : control_dim + cbf_slack_count] = (
            slack_weight * z[control_dim : control_dim + cbf_slack_count]
        )
        if clf_slack_index is not None:
            grad[clf_slack_index] = clf_slack_weight * z[clf_slack_index]
        return grad

    opt_bounds = bounds + [(0.0, None)] * cbf_slack_count
    if clf_slack_index is not None:
        opt_bounds.append((0.0, None))
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
    slack = np.maximum(z[control_dim : control_dim + cbf_slack_count], 0.0)
    clf_slack = max(float(z[clf_slack_index]), 0.0) if clf_slack_index is not None else 0.0
    z_obj = np.concatenate([u, slack])
    if clf_slack_index is not None:
        z_obj = np.concatenate([z_obj, np.array([clf_slack])])
    result = CBFFilterResult(
        u=u,
        success=bool(sol.success),
        slack=slack,
        clf_slack=clf_slack,
        objective=float(objective(z_obj)),
        message=str(sol.message),
    )
    return result if return_info else result.u
