"""Discrete-time CBF MPC controller."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize

from density_utils.controllers.density_filter import _as_weight_matrix, _clip_to_bounds, _control_bounds
from density_utils.controllers.solver_utils import require_solver
from density_utils.dynamics import single_integrator_step


@dataclass(frozen=True)
class CBFMPCResult:
    u: np.ndarray
    success: bool
    slack: np.ndarray
    objective: float
    message: str
    controls: np.ndarray
    states: np.ndarray


def _as_control_sequence(u_nom, horizon, control_dim, bounds):
    u_nom = np.asarray(u_nom, dtype=float)
    if u_nom.ndim == 1:
        if u_nom.size != control_dim:
            raise ValueError("u_nom has the wrong control dimension")
        sequence = np.repeat(u_nom[None, :], horizon, axis=0)
    elif u_nom.shape == (horizon, control_dim):
        sequence = u_nom.copy()
    else:
        raise ValueError("u_nom must have shape (control_dim,) or (horizon, control_dim)")
    return np.vstack([_clip_to_bounds(u, bounds) for u in sequence])


def _pack(controls, slack):
    return np.concatenate([controls.ravel(), slack.ravel()])


def _unpack(z, horizon, control_dim, num_barriers):
    control_size = horizon * control_dim
    controls = z[:control_size].reshape(horizon, control_dim)
    slack = z[control_size:].reshape(horizon, num_barriers)
    return controls, slack


def solve_cbf_mpc(
    x,
    goal,
    *,
    h_fns,
    u_nom,
    horizon,
    dt,
    next_state_fn=single_integrator_step,
    gamma=0.5,
    u_min=-2.0,
    u_max=2.0,
    slack_weight=1e6,
    slack_l1_weight=0.0,
    slack_max=None,
    control_weight=1.0,
    control_rate_weight=0.0,
    previous_control=None,
    state_weight=0.0,
    terminal_weight=0.0,
    initial_controls=None,
    solver="auto",
    return_info=False,
):
    """Solve a finite-horizon discrete-time CBF MPC problem.

    For each predicted step and barrier ``h_i(x) >= 0``, the constraint is

        h_i(x[k+1]) - (1 - gamma) h_i(x[k]) + s_i[k] >= 0,

    with ``s_i[k] >= 0`` penalized in the objective. The remaining objective
    terms mirror :func:`solve_density_mpc`: nominal-control tracking,
    control-rate smoothing, optional state/terminal goal costs, and slack
    penalties.
    """
    require_solver(solver, ("scipy_slsqp",), controller="solve_cbf_mpc")
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    h_fns = list(h_fns)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if x.ndim != 1:
        raise ValueError("x must be a vector")
    if goal.shape != x.shape:
        raise ValueError("goal must have the same dimension as x")
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")

    u_nom_arr = np.asarray(u_nom, dtype=float)
    control_dim = u_nom_arr.size if u_nom_arr.ndim == 1 else u_nom_arr.shape[-1]
    bounds = _control_bounds(u_min, u_max, control_dim)
    u_nom_seq = _as_control_sequence(u_nom_arr, horizon, control_dim, bounds)

    if not h_fns:
        result = CBFMPCResult(
            u=u_nom_seq[0],
            success=True,
            slack=np.zeros((horizon, 0)),
            objective=0.0,
            message="",
            controls=u_nom_seq,
            states=np.repeat(x[None, :], horizon + 1, axis=0),
        )
        return result if return_info else result.u

    num_barriers = len(h_fns)
    w_u = _as_weight_matrix(control_weight, control_dim)
    w_du = _as_weight_matrix(control_rate_weight, control_dim)
    w_x = _as_weight_matrix(state_weight, x.size)
    if previous_control is None:
        previous_control = u_nom_seq[0]
    previous_control = np.asarray(previous_control, dtype=float)
    if previous_control.shape != (control_dim,):
        raise ValueError("previous_control must have shape (control_dim,)")

    if initial_controls is None:
        controls0 = u_nom_seq.copy()
    else:
        controls0 = np.asarray(initial_controls, dtype=float)
        if controls0.shape != (horizon, control_dim):
            raise ValueError("initial_controls must have shape (horizon, control_dim)")
        controls0 = np.vstack([_clip_to_bounds(u, bounds) for u in controls0])

    def rollout(controls):
        states = [x]
        state = x
        for control in controls:
            state = np.asarray(next_state_fn(state, control, dt), dtype=float)
            if state.shape != x.shape:
                raise ValueError("next_state_fn must return a vector with the same shape as x")
            states.append(state)
        return np.asarray(states, dtype=float)

    gamma = float(gamma)
    slack0 = np.zeros((horizon, num_barriers), dtype=float)
    z0 = _pack(controls0, slack0)

    def objective(z):
        controls, slack = _unpack(z, horizon, control_dim, num_barriers)
        du = controls - u_nom_seq
        cost = 0.5 * sum(float(row @ w_u @ row) for row in du)
        states = None
        if np.any(w_du):
            rate_rows = np.vstack([controls[0] - previous_control, np.diff(controls, axis=0)])
            cost += 0.5 * sum(float(row @ w_du @ row) for row in rate_rows)
        if np.any(w_x):
            states = rollout(controls)
            state_err = states[1:] - goal
            cost += 0.5 * sum(float(row @ w_x @ row) for row in state_err)
        cost += slack_l1_weight * float(np.sum(slack))
        cost += 0.5 * slack_weight * float(np.sum(slack * slack))
        if terminal_weight > 0.0:
            if states is None:
                states = rollout(controls)
            terminal_err = states[-1] - goal
            cost += 0.5 * terminal_weight * float(terminal_err @ terminal_err)
        return cost

    def cbf_constraints(z):
        controls, slack = _unpack(z, horizon, control_dim, num_barriers)
        states = rollout(controls)
        values = []
        for k in range(horizon):
            for j, h_fn in enumerate(h_fns):
                h_current = float(h_fn(states[k]))
                h_next = float(h_fn(states[k + 1]))
                values.append(h_next - (1.0 - gamma) * h_current + slack[k, j])
        return np.asarray(values, dtype=float)

    constraints = {"type": "ineq", "fun": cbf_constraints}
    slack_upper = None if slack_max is None else float(slack_max)
    opt_bounds = bounds * horizon + [(0.0, slack_upper)] * (horizon * num_barriers)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step",
            category=RuntimeWarning,
        )
        sol = minimize(
            objective,
            z0,
            bounds=opt_bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-8, "maxiter": 120, "disp": False},
        )

    if sol.success:
        z = sol.x
    else:
        candidates = [z0]
        if getattr(sol, "x", None) is not None and np.all(np.isfinite(sol.x)):
            candidates.append(sol.x)

        def violation_score(z_candidate):
            residual = cbf_constraints(z_candidate)
            return float(np.sum(np.minimum(residual, 0.0) ** 2))

        z = min(candidates, key=lambda z_candidate: (violation_score(z_candidate), objective(z_candidate)))
    controls, slack = _unpack(z, horizon, control_dim, num_barriers)
    controls = np.vstack([_clip_to_bounds(u, bounds) for u in controls])
    slack = np.maximum(slack, 0.0)
    states = rollout(controls)
    result = CBFMPCResult(
        u=controls[0],
        success=bool(sol.success),
        slack=slack,
        objective=float(objective(_pack(controls, slack))),
        message=str(sol.message),
        controls=controls,
        states=states,
    )
    return result if return_info else result.u
