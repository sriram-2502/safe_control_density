"""Discrete density-function MPC controller.

This is the horizon version of the one-step density filter.  The optimizer
chooses a sequence of controls and nonnegative slacks while enforcing the
discrete density residual along the predicted rollout.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize

from density_utils.controllers.solver_utils import normalize_solver
from density_utils.controllers.density_filter import (
    _as_weight_matrix,
    _clip_to_bounds,
    _control_bounds,
    _density_for_obstacle,
    _eval_scalar,
)
from density_utils.density import density_value
from density_utils.dynamics import single_integrator_step


@dataclass(frozen=True)
class DensityMPCResult:
    u: np.ndarray
    success: bool
    slack: np.ndarray
    objective: float
    message: str
    controls: np.ndarray
    states: np.ndarray


_JAX_UNICYCLE_CACHE = {}


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


def _unpack(z, horizon, control_dim, num_obstacles):
    control_size = horizon * control_dim
    controls = z[:control_size].reshape(horizon, control_dim)
    slack = z[control_size:].reshape(horizon, num_obstacles)
    return controls, slack


def _obstacle_schedule(obstacles, horizon):
    obstacles = list(obstacles)
    if not obstacles:
        return [[] for _ in range(horizon + 1)], 0
    first = obstacles[0]
    if isinstance(first, (list, tuple)):
        schedule = [list(row) for row in obstacles]
        if len(schedule) == horizon:
            schedule.append(list(schedule[-1]))
        if len(schedule) != horizon + 1:
            raise ValueError("moving-obstacle schedules must have horizon + 1 entries")
        num_obstacles = len(schedule[0])
        if any(len(row) != num_obstacles for row in schedule):
            raise ValueError("all moving-obstacle schedule rows must have the same length")
        return schedule, num_obstacles
    return [obstacles for _ in range(horizon + 1)], len(obstacles)


def solve_density_mpc(
    x,
    goal,
    alpha,
    obstacles,
    *,
    solver="scipy_slsqp",
    **kwargs,
):
    """Solve a finite-horizon discrete density MPC problem.

    ``solver`` may be ``"scipy_slsqp"``, ``"jax_slsqp"``,
    ``"casadi_ipopt"``, or ``"auto"``. The SciPy backend is kept as the
    compatibility/default backend. The JAX backend uses automatic derivatives
    with SciPy SLSQP, which is convenient for simulation research.
    """
    solver = normalize_solver(solver)

    if solver == "scipy_slsqp":
        return _solve_density_mpc_scipy(x, goal, alpha, obstacles, **kwargs)
    if solver == "jax_slsqp":
        return _solve_density_mpc_jax(x, goal, alpha, obstacles, **kwargs)
    if solver == "casadi_ipopt":
        return _solve_density_mpc_casadi(x, goal, alpha, obstacles, **kwargs)
    raise ValueError(
        "solver must be one of 'scipy_slsqp', 'jax_slsqp', 'casadi_ipopt', or 'auto'"
    )


def _solve_density_mpc_scipy(
    x,
    goal,
    alpha,
    obstacles,
    *,
    u_nom,
    horizon,
    dt,
    next_state_fn=single_integrator_step,
    u_min=-2.0,
    u_max=2.0,
    divergence=None,
    slack_weight=1e4,
    slack_l1_weight=0.0,
    control_weight=1.0,
    control_rate_weight=0.0,
    previous_control=None,
    state_weight=0.0,
    terminal_weight=0.0,
    min_density=1e-10,
    density_fn=density_value,
    initial_controls=None,
    return_info=False,
):
    """Solve a finite-horizon discrete density MPC problem.

    For each horizon step and obstacle, the constraint is

        rho(x[k+1]) - rho(x[k])
        + dt * div(F_d)(x[k]) * rho(x[k])
        - dt * C[k] * rho(x[k]) >= 0.

    This mirrors the MPC-CDF examples: ``C[k]`` is a nonnegative density-rate
    decision variable, not an additive feasibility slack. The optimizer
    minimizes tracking of a nominal control sequence plus density-rate
    penalties and optional state/terminal quadratic goal penalties.
    """
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if x.ndim != 1:
        raise ValueError("x must be a vector")
    if goal.shape != x.shape:
        raise ValueError("goal must have the same dimension as x")

    u_nom_arr = np.asarray(u_nom, dtype=float)
    control_dim = u_nom_arr.size if u_nom_arr.ndim == 1 else u_nom_arr.shape[-1]
    bounds = _control_bounds(u_min, u_max, control_dim)
    u_nom_seq = _as_control_sequence(u_nom_arr, horizon, control_dim, bounds)

    obstacle_schedule, num_obstacles = _obstacle_schedule(obstacles, horizon)
    w_u = _as_weight_matrix(control_weight, control_dim)
    w_du = _as_weight_matrix(control_rate_weight, control_dim)
    w_x = _as_weight_matrix(state_weight, x.size)
    if previous_control is None:
        previous_control = u_nom_seq[0]
    previous_control = np.asarray(previous_control, dtype=float)
    if previous_control.shape != (control_dim,):
        raise ValueError("previous_control must have shape (control_dim,)")
    slack_weight = float(slack_weight)
    slack_l1_weight = float(slack_l1_weight)
    terminal_weight = float(terminal_weight)

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

    slack0 = np.zeros((horizon, num_obstacles), dtype=float)

    z0 = _pack(controls0, slack0)

    def objective(z):
        controls, slack = _unpack(z, horizon, control_dim, num_obstacles)
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

    def density_constraints(z):
        controls, slack = _unpack(z, horizon, control_dim, num_obstacles)
        states = rollout(controls)
        values = []
        for k in range(horizon):
            div_value = _eval_scalar(divergence, states[k], default=0.0)
            for j in range(num_obstacles):
                obstacle = obstacle_schedule[k][j]
                obstacle_next = obstacle_schedule[k + 1][j]
                rho = max(
                    _density_for_obstacle(density_fn, states[k], goal, alpha, obstacle),
                    float(min_density),
                )
                rho_next = _density_for_obstacle(density_fn, states[k + 1], goal, alpha, obstacle_next)
                density_transport = rho_next - rho + dt * div_value * rho
                values.append(density_transport - dt * slack[k, j] * rho)
        return np.asarray(values, dtype=float)

    constraints = () if num_obstacles == 0 else {"type": "ineq", "fun": density_constraints}

    opt_bounds = bounds * horizon + [(0.0, None)] * (horizon * num_obstacles)

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

    z = sol.x if np.all(np.isfinite(sol.x)) else z0
    controls, slack = _unpack(z, horizon, control_dim, num_obstacles)
    controls = np.vstack([_clip_to_bounds(u, bounds) for u in controls])
    slack = np.maximum(slack, 0.0)
    states = rollout(controls)
    result = DensityMPCResult(
        u=controls[0],
        success=bool(sol.success),
        slack=slack,
        objective=float(objective(_pack(controls, slack))),
        message=str(sol.message),
        controls=controls,
        states=states,
    )
    return result if return_info else result.u


def _casadi_p_norm_bump(ca, pos, obstacle):
    center = np.asarray(obstacle.center, dtype=float)
    dx0 = pos[0] - float(center[0])
    dx1 = pos[1] - float(center[1])

    if float(obstacle.angle):
        c = float(np.cos(-obstacle.angle))
        s = float(np.sin(-obstacle.angle))
        rot0 = c * dx0 - s * dx1
        rot1 = s * dx0 + c * dx1
        dx0, dx1 = rot0, rot1

    if obstacle.scale is not None:
        scale = np.asarray(obstacle.scale, dtype=float)
        if np.any(scale <= 0.0):
            raise ValueError("scale entries must be positive")
        dx0 = dx0 / float(scale[0])
        dx1 = dx1 / float(scale[1])

    p = float(obstacle.p)
    norm_p_p = ca.fabs(dx0) ** p + ca.fabs(dx1) ** p
    r1_p = float(obstacle.r1) ** p
    r2_p = float(obstacle.r2) ** p
    denom = max(r2_p - r1_p, 1e-12)
    m = (norm_p_p - r1_p) / denom

    eps = 1e-8
    m_safe = ca.fmin(ca.fmax(m, eps), 1.0 - eps)
    f = ca.exp(-1.0 / m_safe)
    g = ca.exp(-1.0 / (1.0 - m_safe))
    smooth = f / (f + g)
    return ca.if_else(m <= 0.0, 0.0, ca.if_else(m >= 1.0, 1.0, smooth))


def _casadi_pose_density(ca, state, goal, alpha, obstacles, *, min_density, theta_weight=0.05):
    dx = state[0] - float(goal[0])
    dy = state[1] - float(goal[1])
    dtheta = state[2] - float(goal[2])
    lyap = ca.fmax(dx * dx + dy * dy + float(theta_weight) * dtheta * dtheta, 1e-6)
    phi = 1.0
    pos = state[0:2]
    for obstacle in obstacles:
        phi = phi * _casadi_p_norm_bump(ca, pos, obstacle)
    return ca.fmax(phi / (lyap ** float(alpha)), float(min_density))


def _get_jax_unicycle_functions(horizon, num_obstacles):
    key = (int(horizon), int(num_obstacles))
    cached = _JAX_UNICYCLE_CACHE.get(key)
    if cached is not None:
        return cached

    import jax
    import jax.numpy as jnp

    horizon = int(horizon)
    num_obstacles = int(num_obstacles)
    control_dim = 2

    def unpack(z):
        controls = z[: horizon * control_dim].reshape((horizon, control_dim))
        slack = z[horizon * control_dim :].reshape((horizon, num_obstacles))
        return controls, slack

    def step_fn(state, control, dt):
        return jnp.array(
            [
                state[0] + dt * control[0] * jnp.cos(state[2]),
                state[1] + dt * control[0] * jnp.sin(state[2]),
                state[2] + dt * control[1],
            ],
            dtype=state.dtype,
        )

    def rollout(x0, controls, dt):
        def body(state, control):
            next_state = step_fn(state, control, dt)
            return next_state, next_state

        _, tail = jax.lax.scan(body, x0, controls)
        return jnp.vstack((x0[None, :], tail))

    def bump(pos, obstacle):
        center = obstacle[0:2]
        r1 = obstacle[2]
        r2 = obstacle[3]
        p = obstacle[4]
        scale = obstacle[5:7]
        angle = obstacle[7]
        dx = pos - center
        c = jnp.cos(-angle)
        s = jnp.sin(-angle)
        dx = jnp.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]]) / scale
        norm_p_p = jnp.sum(jnp.abs(dx) ** p)
        r1_p = r1**p
        r2_p = r2**p
        m = (norm_p_p - r1_p) / jnp.maximum(r2_p - r1_p, 1e-12)
        m_safe = jnp.clip(m, 1e-8, 1.0 - 1e-8)
        f = jnp.exp(-1.0 / m_safe)
        g = jnp.exp(-1.0 / (1.0 - m_safe))
        smooth = f / (f + g)
        return jnp.where(m <= 0.0, 0.0, jnp.where(m >= 1.0, 1.0, smooth))

    def pose_density(state, goal, alpha, obstacle, min_density):
        err = state - goal
        lyap = jnp.maximum(err[0] ** 2 + err[1] ** 2 + 0.05 * err[2] ** 2, 1e-6)
        phi = bump(state[:2], obstacle)
        return jnp.maximum(phi / (lyap**alpha), min_density)

    def objective_raw(z, params):
        controls, slack = unpack(z)
        x0 = params["x0"]
        goal = params["goal"]
        u_nom = params["u_nom"]
        previous_control = params["previous_control"]
        w_u = params["w_u"]
        w_du = params["w_du"]
        w_x = params["w_x"]
        dt = params["dt"]
        slack_l1_weight = params["slack_l1_weight"]
        slack_weight = params["slack_weight"]
        terminal_weight = params["terminal_weight"]
        use_rate = params["use_rate"]
        use_state = params["use_state"]

        du = controls - u_nom
        cost = 0.5 * jnp.sum(jnp.einsum("bi,ij,bj->b", du, w_u, du))
        rate_rows = jnp.vstack((controls[0] - previous_control, controls[1:] - controls[:-1]))
        rate_cost = 0.5 * jnp.sum(jnp.einsum("bi,ij,bj->b", rate_rows, w_du, rate_rows))
        states = rollout(x0, controls, dt)
        state_err = states[1:] - goal
        state_cost = 0.5 * jnp.sum(jnp.einsum("bi,ij,bj->b", state_err, w_x, state_err))
        terminal_err = states[-1] - goal
        terminal_cost = 0.5 * terminal_weight * jnp.dot(terminal_err, terminal_err)
        slack_cost = slack_l1_weight * jnp.sum(slack) + 0.5 * slack_weight * jnp.sum(slack * slack)
        return cost + use_rate * rate_cost + use_state * state_cost + terminal_cost + slack_cost

    def constraints_raw(z, params):
        controls, slack = unpack(z)
        states = rollout(params["x0"], controls, params["dt"])
        values = []
        for k in range(horizon):
            for j in range(num_obstacles):
                rho = pose_density(
                    states[k],
                    params["goal"],
                    params["alpha"],
                    params["obstacles"][j],
                    params["min_density"],
                )
                rho_next = pose_density(
                    states[k + 1],
                    params["goal"],
                    params["alpha"],
                    params["obstacles"][j],
                    params["min_density"],
                )
                density_transport = rho_next - rho + params["dt"] * params["divergence"] * rho
                values.append(density_transport - params["dt"] * slack[k, j] * rho)
        return jnp.asarray(values)

    value_and_grad = jax.jit(jax.value_and_grad(objective_raw))
    constraints = jax.jit(constraints_raw)
    constraints_jac = jax.jit(jax.jacfwd(constraints_raw))
    cached = value_and_grad, constraints, constraints_jac, rollout
    _JAX_UNICYCLE_CACHE[key] = cached
    return cached


def _solve_density_mpc_jax(
    x,
    goal,
    alpha,
    obstacles,
    *,
    u_nom,
    horizon,
    dt,
    next_state_fn=single_integrator_step,
    u_min=-2.0,
    u_max=2.0,
    divergence=None,
    slack_weight=1e4,
    slack_l1_weight=0.0,
    control_weight=1.0,
    control_rate_weight=0.0,
    previous_control=None,
    state_weight=0.0,
    terminal_weight=0.0,
    min_density=1e-10,
    density_fn=density_value,
    initial_controls=None,
    return_info=False,
    maxiter=80,
    ftol=1e-7,
):
    """JAX autodiff + SciPy SLSQP backend for unicycle density MPC."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import config as jax_config
    except ImportError as exc:
        raise ImportError("JAX is not installed. Install jax/jaxlib or use another solver.") from exc

    jax_config.update("jax_enable_x64", True)

    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    obstacles = list(obstacles)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if x.shape != (3,):
        raise ValueError("jax_slsqp backend currently supports 3D unicycle state [x, y, theta]")
    if goal.shape != x.shape:
        raise ValueError("goal must have the same dimension as x")

    u_nom_arr = np.asarray(u_nom, dtype=float)
    control_dim = u_nom_arr.size if u_nom_arr.ndim == 1 else u_nom_arr.shape[-1]
    if control_dim != 2:
        raise ValueError("jax_slsqp backend currently supports 2D controls [v, omega]")
    bounds = _control_bounds(u_min, u_max, control_dim)
    u_nom_seq = _as_control_sequence(u_nom_arr, horizon, control_dim, bounds)

    if not obstacles:
        result = DensityMPCResult(
            u=u_nom_seq[0],
            success=True,
            slack=np.zeros((horizon, 0)),
            objective=0.0,
            message="",
            controls=u_nom_seq,
            states=np.repeat(x[None, :], horizon + 1, axis=0),
        )
        return result if return_info else result.u

    num_obstacles = len(obstacles)
    w_u = _as_weight_matrix(control_weight, control_dim)
    w_du = _as_weight_matrix(control_rate_weight, control_dim)
    w_x = _as_weight_matrix(state_weight, x.size)
    if previous_control is None:
        previous_control = u_nom_seq[0]
    previous_control = np.asarray(previous_control, dtype=float)
    if previous_control.shape != (control_dim,):
        raise ValueError("previous_control must have shape (control_dim,)")

    if callable(divergence):
        div_const = float(divergence(x))
    else:
        div_const = _eval_scalar(divergence, x, default=0.0)

    if initial_controls is None:
        controls0 = u_nom_seq.copy()
    else:
        controls0 = np.asarray(initial_controls, dtype=float)
        if controls0.shape != (horizon, control_dim):
            raise ValueError("initial_controls must have shape (horizon, control_dim)")
        controls0 = np.vstack([_clip_to_bounds(u, bounds) for u in controls0])

    z0 = _pack(controls0, np.zeros((horizon, num_obstacles), dtype=float))
    opt_bounds = bounds * horizon + [(0.0, None)] * (horizon * num_obstacles)

    obstacle_params = []
    for obstacle in obstacles:
        center = np.asarray(obstacle.center, dtype=float)
        scale = np.ones(2, dtype=float) if obstacle.scale is None else np.asarray(obstacle.scale, dtype=float)
        if np.any(scale <= 0.0):
            raise ValueError("scale entries must be positive")
        obstacle_params.append(
            [
                center[0],
                center[1],
                float(obstacle.r1),
                float(obstacle.r2),
                float(obstacle.p),
                scale[0],
                scale[1],
                float(obstacle.angle),
            ]
        )

    objective_value_and_grad, constraints_fun, constraints_jac, _ = _get_jax_unicycle_functions(
        horizon,
        num_obstacles,
    )
    params = {
        "x0": jnp.asarray(x),
        "goal": jnp.asarray(goal),
        "u_nom": jnp.asarray(u_nom_seq),
        "previous_control": jnp.asarray(previous_control),
        "w_u": jnp.asarray(w_u),
        "w_du": jnp.asarray(w_du),
        "w_x": jnp.asarray(w_x),
        "obstacles": jnp.asarray(obstacle_params),
        "dt": jnp.asarray(float(dt)),
        "alpha": jnp.asarray(float(alpha)),
        "min_density": jnp.asarray(float(min_density)),
        "divergence": jnp.asarray(float(div_const)),
        "slack_l1_weight": jnp.asarray(float(slack_l1_weight)),
        "slack_weight": jnp.asarray(float(slack_weight)),
        "terminal_weight": jnp.asarray(float(terminal_weight)),
        "use_rate": jnp.asarray(float(np.any(w_du))),
        "use_state": jnp.asarray(float(np.any(w_x))),
    }

    def objective_np(z):
        value, _ = objective_value_and_grad(jnp.asarray(z, dtype=jnp.float64), params)
        return float(value)

    def objective_grad_np(z):
        _, grad = objective_value_and_grad(jnp.asarray(z, dtype=jnp.float64), params)
        return np.asarray(grad, dtype=float)

    def constraints_np(z):
        return np.asarray(constraints_fun(jnp.asarray(z, dtype=jnp.float64), params), dtype=float)

    def constraints_jac_np(z):
        return np.asarray(constraints_jac(jnp.asarray(z, dtype=jnp.float64), params), dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step",
            category=RuntimeWarning,
        )
        sol = minimize(
            objective_np,
            z0,
            jac=objective_grad_np,
            bounds=opt_bounds,
            constraints={"type": "ineq", "fun": constraints_np, "jac": constraints_jac_np},
            method="SLSQP",
            options={"ftol": float(ftol), "maxiter": int(maxiter), "disp": False},
        )

    z = sol.x if sol.success else z0
    controls, slack = _unpack(z, horizon, control_dim, num_obstacles)
    controls = np.vstack([_clip_to_bounds(u, bounds) for u in controls])
    slack = np.maximum(slack, 0.0)

    states = [x.copy()]
    state = x.copy()
    for control in controls:
        state = np.asarray(
            [
                state[0] + float(dt) * control[0] * np.cos(state[2]),
                state[1] + float(dt) * control[0] * np.sin(state[2]),
                state[2] + float(dt) * control[1],
            ],
            dtype=float,
        )
        states.append(state.copy())
    states = np.asarray(states, dtype=float)

    result = DensityMPCResult(
        u=controls[0],
        success=bool(sol.success),
        slack=slack,
        objective=float(objective_np(_pack(controls, slack))),
        message=str(sol.message),
        controls=controls,
        states=states,
    )
    return result if return_info else result.u


def _solve_density_mpc_casadi(
    x,
    goal,
    alpha,
    obstacles,
    *,
    u_nom,
    horizon,
    dt,
    next_state_fn=single_integrator_step,
    u_min=-2.0,
    u_max=2.0,
    divergence=None,
    slack_weight=1e4,
    slack_l1_weight=0.0,
    control_weight=1.0,
    control_rate_weight=0.0,
    previous_control=None,
    state_weight=0.0,
    terminal_weight=0.0,
    min_density=1e-10,
    density_fn=density_value,
    initial_controls=None,
    return_info=False,
    ipopt_options=None,
):
    """CasADi/IPOPT backend for unicycle density MPC.

    This backend intentionally keeps the same density transport constraint as
    the SciPy reference backend. It currently supports the unicycle state
    ``[x, y, theta]`` and control ``[v, omega]`` used by the examples.
    """
    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError(
            "CasADi is not installed. Install casadi or use solver='scipy_slsqp'."
        ) from exc

    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    obstacles = list(obstacles)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if x.shape != (3,):
        raise ValueError("casadi_ipopt backend currently supports 3D unicycle state [x, y, theta]")
    if goal.shape != x.shape:
        raise ValueError("goal must have the same dimension as x")

    u_nom_arr = np.asarray(u_nom, dtype=float)
    control_dim = u_nom_arr.size if u_nom_arr.ndim == 1 else u_nom_arr.shape[-1]
    if control_dim != 2:
        raise ValueError("casadi_ipopt backend currently supports 2D controls [v, omega]")
    bounds = _control_bounds(u_min, u_max, control_dim)
    u_nom_seq = _as_control_sequence(u_nom_arr, horizon, control_dim, bounds)

    if not obstacles:
        result = DensityMPCResult(
            u=u_nom_seq[0],
            success=True,
            slack=np.zeros((horizon, 0)),
            objective=0.0,
            message="",
            controls=u_nom_seq,
            states=np.repeat(x[None, :], horizon + 1, axis=0),
        )
        return result if return_info else result.u

    if density_fn is not density_value and getattr(density_fn, "__name__", "") not in {
        "density_value",
        "density_fn",
    }:
        # The unicycle examples pass a small local wrapper named density_fn.
        # We reproduce that pose density symbolically here.
        pass

    num_obstacles = len(obstacles)
    w_u = _as_weight_matrix(control_weight, control_dim)
    w_du = _as_weight_matrix(control_rate_weight, control_dim)
    w_x = _as_weight_matrix(state_weight, x.size)
    if previous_control is None:
        previous_control = u_nom_seq[0]
    previous_control = np.asarray(previous_control, dtype=float)
    if previous_control.shape != (control_dim,):
        raise ValueError("previous_control must have shape (control_dim,)")

    if callable(divergence):
        div_const = float(divergence(x))
    else:
        div_const = _eval_scalar(divergence, x, default=0.0)

    if initial_controls is None:
        controls0 = u_nom_seq.copy()
    else:
        controls0 = np.asarray(initial_controls, dtype=float)
        if controls0.shape != (horizon, control_dim):
            raise ValueError("initial_controls must have shape (horizon, control_dim)")
        controls0 = np.vstack([_clip_to_bounds(u, bounds) for u in controls0])

    opti = ca.Opti()
    X = opti.variable(3, horizon + 1)
    U = opti.variable(2, horizon)
    C = opti.variable(num_obstacles, horizon)

    opti.subject_to(X[:, 0] == x)
    opti.subject_to(C >= 0)
    for k in range(horizon):
        v = U[0, k]
        omega = U[1, k]
        x_next = ca.vertcat(
            X[0, k] + float(dt) * v * ca.cos(X[2, k]),
            X[1, k] + float(dt) * v * ca.sin(X[2, k]),
            X[2, k] + float(dt) * omega,
        )
        opti.subject_to(X[:, k + 1] == x_next)
        opti.subject_to(U[0, k] >= bounds[0][0])
        opti.subject_to(U[0, k] <= bounds[0][1])
        opti.subject_to(U[1, k] >= bounds[1][0])
        opti.subject_to(U[1, k] <= bounds[1][1])

        for j, obstacle in enumerate(obstacles):
            rho = _casadi_pose_density(ca, X[:, k], goal, alpha, [obstacle], min_density=min_density)
            rho_next = _casadi_pose_density(
                ca,
                X[:, k + 1],
                goal,
                alpha,
                [obstacle],
                min_density=min_density,
            )
            density_transport = rho_next - rho + float(dt) * div_const * rho
            opti.subject_to(density_transport - float(dt) * C[j, k] * rho >= 0)

    cost = 0.0
    for k in range(horizon):
        du = U[:, k] - ca.DM(u_nom_seq[k])
        cost += 0.5 * ca.mtimes([du.T, ca.DM(w_u), du])
        if np.any(w_du):
            prev_u = ca.DM(previous_control) if k == 0 else U[:, k - 1]
            rate = U[:, k] - prev_u
            cost += 0.5 * ca.mtimes([rate.T, ca.DM(w_du), rate])
        if np.any(w_x):
            err = X[:, k + 1] - ca.DM(goal)
            cost += 0.5 * ca.mtimes([err.T, ca.DM(w_x), err])
        if float(slack_l1_weight):
            cost += float(slack_l1_weight) * ca.sum1(C[:, k])
        if float(slack_weight):
            cost += 0.5 * float(slack_weight) * ca.sumsqr(C[:, k])

    if float(terminal_weight) > 0.0:
        terminal_err = X[:, horizon] - ca.DM(goal)
        cost += 0.5 * float(terminal_weight) * ca.dot(terminal_err, terminal_err)

    opti.minimize(cost)
    opti.set_initial(U, controls0.T)
    opti.set_initial(C, np.zeros((num_obstacles, horizon)))

    states0 = [x.copy()]
    state = x.copy()
    for control in controls0:
        state = np.array(
            [
                state[0] + float(dt) * control[0] * np.cos(state[2]),
                state[1] + float(dt) * control[0] * np.sin(state[2]),
                state[2] + float(dt) * control[1],
            ],
            dtype=float,
        )
        states0.append(state.copy())
    opti.set_initial(X, np.asarray(states0, dtype=float).T)

    options = {
        "verbose": False,
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.max_iter": 80,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-5,
        "ipopt.acceptable_obj_change_tol": 1e-6,
    }
    if ipopt_options:
        options.update(ipopt_options)
    opti.solver("ipopt", options)

    success = True
    message = "Solve_Succeeded"
    try:
        sol = opti.solve()
        controls = np.asarray(sol.value(U), dtype=float).T
        slack = np.asarray(sol.value(C), dtype=float).T
        states = np.asarray(sol.value(X), dtype=float).T
        objective = float(sol.value(cost))
    except RuntimeError as exc:
        success = False
        message = str(exc).splitlines()[0] if str(exc) else "IPOPT failed"
        controls = np.asarray(opti.debug.value(U), dtype=float).T
        slack = np.asarray(opti.debug.value(C), dtype=float).T
        states = np.asarray(opti.debug.value(X), dtype=float).T
        objective = float(opti.debug.value(cost))

    controls = np.vstack([_clip_to_bounds(u, bounds) for u in controls])
    slack = np.maximum(slack, 0.0)
    result = DensityMPCResult(
        u=controls[0],
        success=success,
        slack=slack,
        objective=objective,
        message=message,
        controls=controls,
        states=states,
    )
    return result if return_info else result.u
