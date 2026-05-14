"""Single-step density-function QP controller.

This is the receding-horizon MPC-CDF density constraint collapsed to a local
QP for control-affine dynamics, ``x_dot = f(x) + g(x) u``.  The controller
tracks a nominal command while enforcing a linearized CDF condition

    grad(rho_i(x)) @ (f(x) + g(x) u) + s_i >= cdf_rate * rho_i(x)

for each obstacle density ``rho_i``.  The nonnegative slack variables keep the
QP feasible at low-gradient points while still heavily penalizing violations.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from density_utils.density import density_grad, density_value, finite_difference_grad


@dataclass(frozen=True)
class DensityQPResult:
    u: np.ndarray
    success: bool
    slack: np.ndarray
    objective: float
    message: str


@dataclass(frozen=True)
class DoubleIntegratorQPResult:
    u: np.ndarray
    v_des: np.ndarray
    qp: DensityQPResult | None
    accel_success: bool
    accel_slack: np.ndarray
    using_lqr: bool


@dataclass(frozen=True)
class ControlAffineDynamics:
    """Evaluated ``x_dot = drift + control_matrix @ u`` data."""

    drift: np.ndarray
    control_matrix: np.ndarray


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


def _nominal_control(
    x,
    goal,
    drift,
    control_matrix,
    ctrl_multiplier,
    rad_from_goal,
    q_lqr,
    r_lqr,
    dt,
):
    n = x.size
    if np.linalg.norm(x - goal) < rad_from_goal:
        k_lqr = _lqr_gain(n, dt, q_lqr, r_lqr)
        desired_xdot = -k_lqr @ (x - goal)
    else:
        direction = goal - x
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            desired_xdot = np.zeros_like(x)
        else:
            desired_xdot = ctrl_multiplier * direction / norm

    return np.linalg.lstsq(control_matrix, desired_xdot - drift, rcond=None)[0]


def _as_weight_matrix(weight, n):
    if np.isscalar(weight):
        return float(weight) * np.eye(n)
    weight = np.asarray(weight, dtype=float)
    if weight.shape != (n, n):
        raise ValueError("control_weight must be a scalar or an (n,n) array")
    return weight


def _eval_vector(value, x, name, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"{name} must be provided")
        return np.asarray(default, dtype=float)
    if callable(value):
        value = value(x)
    value = np.asarray(value, dtype=float)
    if value.ndim != 1:
        raise ValueError(f"{name} must evaluate to a vector")
    return value


def _eval_matrix(value, x, name, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"{name} must be provided")
        return np.asarray(default, dtype=float)
    if callable(value):
        value = value(x)
    value = np.asarray(value, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must evaluate to a matrix")
    return value


def _evaluate_control_affine_dynamics(
    x,
    *,
    dynamics=None,
    A=None,
    B=None,
    f=None,
    g=None,
    u_dim=None,
):
    """Evaluate a linear or nonlinear control-affine model at ``x``.

    Supported forms:
    - omit everything for the single integrator ``x_dot = u``
    - pass ``A`` and ``B`` for ``x_dot = A @ x + B @ u``
    - pass ``f`` and ``g`` as arrays or callables for ``x_dot = f(x) + g(x)u``
    """
    n = x.size

    if dynamics is not None:
        if any(item is not None for item in (A, B, f, g)):
            raise ValueError("Pass either dynamics=... or A/B/f/g, not both")

        if isinstance(dynamics, ControlAffineDynamics):
            drift = np.asarray(dynamics.drift, dtype=float)
            control_matrix = np.asarray(dynamics.control_matrix, dtype=float)
            if drift.shape != (n,) or control_matrix.ndim != 2 or control_matrix.shape[0] != n:
                raise ValueError("dynamics has incompatible drift/control_matrix shapes")
            if u_dim is not None and control_matrix.shape[1] != int(u_dim):
                raise ValueError("u_dim does not match the control dimension")
            return ControlAffineDynamics(drift=drift, control_matrix=control_matrix)

        if callable(dynamics):
            evaluated = dynamics(x)
            if isinstance(evaluated, ControlAffineDynamics):
                return _evaluate_control_affine_dynamics(x, dynamics=evaluated, u_dim=u_dim)
            if not isinstance(evaluated, (tuple, list)) or len(evaluated) != 2:
                raise ValueError("callable dynamics must return (f, g) or ControlAffineDynamics")
            drift = _eval_vector(evaluated[0], x, "dynamics drift")
            control_matrix = _eval_matrix(evaluated[1], x, "dynamics control matrix")
            if drift.shape != (n,) or control_matrix.shape[0] != n:
                raise ValueError("callable dynamics returned incompatible shapes")
            if u_dim is not None and control_matrix.shape[1] != int(u_dim):
                raise ValueError("u_dim does not match the control dimension")
            return ControlAffineDynamics(drift=drift, control_matrix=control_matrix)

        if isinstance(dynamics, dict):
            A = dynamics.get("A")
            B = dynamics.get("B")
            f = dynamics.get("f")
            g = dynamics.get("g")
            u_dim = dynamics.get("u_dim", u_dim)
        elif isinstance(dynamics, (tuple, list)) and len(dynamics) == 2:
            A, B = dynamics
        else:
            raise ValueError(
                "dynamics must be a dict, an (A, B) tuple, a callable, "
                "or ControlAffineDynamics"
            )

    if A is not None:
        A = np.asarray(A, dtype=float)
        if A.shape != (n, n):
            raise ValueError("A must have shape (state_dim, state_dim)")
        linear_drift = A @ x
    else:
        linear_drift = np.zeros(n, dtype=float)

    if f is None:
        drift = linear_drift
    else:
        drift = _eval_vector(f, x, "f")
        if drift.shape != (n,):
            raise ValueError("f must evaluate to shape (state_dim,)")
        if A is not None:
            drift = drift + linear_drift

    if B is not None:
        B = np.asarray(B, dtype=float)
        if B.ndim != 2 or B.shape[0] != n:
            raise ValueError("B must have shape (state_dim, control_dim)")
        linear_control = B
    else:
        if g is None:
            if u_dim is None:
                u_dim = n
            linear_control = np.eye(n, int(u_dim), dtype=float)
        else:
            linear_control = None

    if g is None:
        control_matrix = linear_control
    else:
        control_matrix = _eval_matrix(g, x, "g")
        if control_matrix.shape[0] != n:
            raise ValueError("g must evaluate to shape (state_dim, control_dim)")
        if B is not None:
            if control_matrix.shape != linear_control.shape:
                raise ValueError("g and B must have the same shape when both are provided")
            control_matrix = control_matrix + linear_control

    if u_dim is not None and control_matrix.shape[1] != int(u_dim):
        raise ValueError("u_dim does not match the control dimension")

    return ControlAffineDynamics(drift=drift, control_matrix=control_matrix)


def _control_bounds(saturation, control_dim):
    if np.isscalar(saturation):
        sat = float(saturation)
        return [(-sat, sat)] * control_dim

    if isinstance(saturation, tuple) and len(saturation) == 2:
        lower = np.asarray(saturation[0], dtype=float)
        upper = np.asarray(saturation[1], dtype=float)
        if lower.shape == ():
            lower = np.full(control_dim, float(lower))
        if upper.shape == ():
            upper = np.full(control_dim, float(upper))
        if lower.shape != (control_dim,) or upper.shape != (control_dim,):
            raise ValueError("saturation bounds must match the control dimension")
        return list(zip(lower, upper))

    sat = np.asarray(saturation, dtype=float)
    if sat.shape != (control_dim,):
        raise ValueError("saturation must be a scalar, a vector, or a (lower, upper) tuple")
    return [(-float(val), float(val)) for val in sat]


def _clip_to_bounds(u, bounds):
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    return np.clip(u, lower, upper)


def _density_value_for_obstacle(density_fn, x, goal, alpha, obs):
    return float(density_fn(x, goal, alpha, [obs]))


def _density_grad_for_obstacle(density_fn, density_grad_fn, x, goal, alpha, obs):
    if density_grad_fn is not None:
        return np.asarray(density_grad_fn(x, goal, alpha, [obs]), dtype=float)
    return finite_difference_grad(
        lambda x_eval: _density_value_for_obstacle(density_fn, x_eval, goal, alpha, obs),
        x,
        eps=1e-3,
    )


def _eval_scalar(value, x, name, default=0.0):
    if value is None:
        return float(default)
    if callable(value):
        value = value(x)
    return float(value)


def solve_density_qp(
    x,
    goal,
    alpha,
    obstacles,
    *,
    dynamics=None,
    A=None,
    B=None,
    f=None,
    g=None,
    u_dim=None,
    u_nom=None,
    ctrl_multiplier=2.0,
    rad_from_goal=0.1,
    q_lqr=1.0,
    r_lqr=1.0,
    dt=0.02,
    saturation=2.0,
    cdf_rate=0.1,
    slack_weight=1e4,
    control_weight=1.0,
    min_density=1e-10,
    density_fn=None,
    density_grad_fn=None,
    constraint_mode="continuous",
    next_state_fn=None,
    divergence=None,
    return_info=False,
):
    """Solve a density-function safety-filter QP.

    By default this behaves like the original single-integrator controller,
    ``x_dot = u``.  Pass dynamics as:

    - ``dynamics=(A, B)`` or ``A=A, B=B`` for ``x_dot = A @ x + B @ u``
    - ``dynamics={"f": f, "g": g}`` or ``f=f, g=g`` for
      ``x_dot = f(x) + g(x)u``
    - ``dynamics=lambda x: (f_x, g_x)`` for a nonlinear model callback

    ``obstacles`` are enforced as separate density constraints, matching the
    obstacle loop in the MPC-CDF MATLAB examples.  By default the density is
    the repo's position density.  Pass ``density_fn`` and optionally
    ``density_grad_fn`` to use a full-state CDF such as ``Phi / V^alpha``.

    ``constraint_mode="continuous"`` enforces
    ``grad(rho) @ (f + g u) >= cdf_rate * rho``.  ``"discrete"`` enforces a
    one-step linearization of the MPC-CDF condition
    ``rho_next - rho + dt * div(f, g) * rho >= dt * cdf_rate * rho``.
    """
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if goal.shape != x.shape:
        raise ValueError("goal must have the same dimension as x")
    obstacles = list(obstacles)
    if density_fn is None:
        density_fn = density_value
    if density_grad_fn is None and density_fn is density_value:
        density_grad_fn = density_grad
    model = _evaluate_control_affine_dynamics(
        x,
        dynamics=dynamics,
        A=A,
        B=B,
        f=f,
        g=g,
        u_dim=u_dim,
    )
    drift = model.drift
    control_matrix = model.control_matrix
    control_dim = control_matrix.shape[1]
    bounds = _control_bounds(saturation, control_dim)

    if u_nom is None:
        u_nom = _nominal_control(
            x,
            goal,
            drift,
            control_matrix,
            ctrl_multiplier,
            rad_from_goal,
            q_lqr,
            r_lqr,
            dt,
        )
    else:
        u_nom = np.asarray(u_nom, dtype=float)

    if u_nom.shape != (control_dim,):
        raise ValueError("u_nom must have shape (control_dim,)")

    if not obstacles:
        u = _clip_to_bounds(u_nom, bounds)
        result = DensityQPResult(u=u, success=True, slack=np.zeros(0), objective=0.0, message="")
        return result if return_info else result.u

    a_rows = []
    b_vals = []
    if constraint_mode not in ("continuous", "discrete"):
        raise ValueError("constraint_mode must be 'continuous' or 'discrete'")
    u_ref = _clip_to_bounds(u_nom, bounds)
    for obs in obstacles:
        rho = _density_value_for_obstacle(density_fn, x, goal, alpha, obs)
        if rho <= min_density:
            continue
        if constraint_mode == "continuous":
            grad = _density_grad_for_obstacle(density_fn, density_grad_fn, x, goal, alpha, obs)
            if grad.shape != x.shape:
                raise ValueError("density_grad_fn must return shape (state_dim,)")
            if np.linalg.norm(grad) <= 1e-12:
                continue
            a_rows.append(grad @ control_matrix)
            b_vals.append(cdf_rate * rho - float(grad @ drift))
        else:
            if next_state_fn is None:
                next_state = lambda u_eval: x + dt * (drift + control_matrix @ u_eval)
            else:
                next_state = lambda u_eval: np.asarray(next_state_fn(x, u_eval, dt), dtype=float)

            rho_next_ref = _density_value_for_obstacle(
                density_fn,
                next_state(u_ref),
                goal,
                alpha,
                obs,
            )
            grad_u = finite_difference_grad(
                lambda u_eval: _density_value_for_obstacle(
                    density_fn,
                    next_state(u_eval),
                    goal,
                    alpha,
                    obs,
                ),
                u_ref,
                eps=1e-4,
            )
            if np.linalg.norm(grad_u) <= 1e-12:
                continue
            div_val = _eval_scalar(divergence, x, "divergence", default=0.0)
            a_rows.append(grad_u)
            b_vals.append(
                dt * cdf_rate * rho
                - (rho_next_ref - rho)
                - dt * div_val * rho
                + float(grad_u @ u_ref)
            )

    if not a_rows:
        u = _clip_to_bounds(u_nom, bounds)
        result = DensityQPResult(u=u, success=True, slack=np.zeros(0), objective=0.0, message="")
        return result if return_info else result.u

    a_mat = np.vstack(a_rows)
    b_vec = np.asarray(b_vals, dtype=float)
    m = b_vec.size
    w_u = _as_weight_matrix(control_weight, control_dim)
    slack_weight = float(slack_weight)

    u0 = _clip_to_bounds(u_nom, bounds)
    s0 = np.maximum(0.0, b_vec - a_mat @ u0)
    z0 = np.concatenate([u0, s0])

    def objective(z):
        u = z[:control_dim]
        slack = z[control_dim:]
        du = u - u_nom
        return 0.5 * float(du @ w_u @ du) + 0.5 * slack_weight * float(slack @ slack)

    def objective_jac(z):
        grad_z = np.zeros_like(z)
        grad_z[:control_dim] = w_u @ (z[:control_dim] - u_nom)
        grad_z[control_dim:] = slack_weight * z[control_dim:]
        return grad_z

    constraints = [
        {
            "type": "ineq",
            "fun": lambda z: a_mat @ z[:control_dim] + z[control_dim:] - b_vec,
            "jac": lambda z: np.hstack([a_mat, np.eye(m)]),
        }
    ]
    opt_bounds = bounds + [(0.0, None)] * m

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

    if sol.success:
        z = sol.x
    else:
        z = z0
    u = _clip_to_bounds(z[:control_dim], bounds)
    slack = np.maximum(z[control_dim:], 0.0)
    result = DensityQPResult(
        u=u,
        success=bool(sol.success),
        slack=slack,
        objective=float(objective(np.concatenate([u, slack]))),
        message=str(sol.message),
    )
    return result if return_info else result.u


def _double_integrator_lqr_gain(dt, q_lqr, r_lqr):
    if np.isscalar(q_lqr):
        q = float(q_lqr) * np.eye(4)
    else:
        q = np.asarray(q_lqr, dtype=float)
    if np.isscalar(r_lqr):
        r = float(r_lqr) * np.eye(2)
    else:
        r = np.asarray(r_lqr, dtype=float)
    if q.shape != (4, 4) or r.shape != (2, 2):
        raise ValueError("q_lqr must be (4,4) and r_lqr must be (2,2) for double integrator")

    a = np.block([[np.eye(2), dt * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    b = np.block([[np.zeros((2, 2))], [dt * np.eye(2)]])
    p = la.solve_discrete_are(a, b, q, r)
    bt_p = b.T @ p
    return np.linalg.solve(bt_p @ b + r, bt_p @ a)


def scaled_saturation(dist, saturation, rad_from_goal):
    if dist >= rad_from_goal:
        return float(saturation)
    decay_length = max(rad_from_goal / 3.0, 1e-6)
    return float(saturation) * (1.0 - np.exp(-dist / decay_length))


def _clip_by_inf_norm(u, saturation):
    max_u = np.max(np.abs(u))
    if max_u > saturation:
        return u / max_u * saturation
    return u


def _p_norm_distance(x, obs):
    dx = x - obs.center
    if obs.angle:
        c = np.cos(-obs.angle)
        s = np.sin(-obs.angle)
        dx = np.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]])
    if obs.scale is not None:
        dx = dx / obs.scale
    return np.sum(np.abs(dx) ** obs.p) ** (1.0 / obs.p)


def _distance_grad(x, obs, eps=1e-4):
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        x_f = x.copy()
        x_b = x.copy()
        x_f[i] += eps
        x_b[i] -= eps
        grad[i] = (_p_norm_distance(x_f, obs) - _p_norm_distance(x_b, obs)) / (2.0 * eps)
    norm = np.linalg.norm(grad)
    if norm < 1e-12:
        return None
    return grad / norm


def _solve_accel_filter(
    u_nom,
    pos,
    vel,
    obstacles,
    saturation,
    *,
    barrier_kp=4.0,
    barrier_kd=4.0,
    slack_weight=1e5,
):
    rows = []
    bounds = []
    for obs in obstacles:
        h = _p_norm_distance(pos, obs) - obs.r1
        normal = _distance_grad(pos, obs)
        if normal is None:
            continue
        v_radial = float(normal @ vel)
        rows.append(normal)
        bounds.append(-barrier_kd * v_radial - barrier_kp * h)

    if not rows:
        return _clip_by_inf_norm(u_nom, saturation), True, np.zeros(0)

    a_mat = np.vstack(rows)
    b_vec = np.asarray(bounds, dtype=float)
    m = b_vec.size
    u0 = _clip_by_inf_norm(u_nom, saturation)
    s0 = np.maximum(0.0, b_vec - a_mat @ u0)
    z0 = np.concatenate([u0, s0])

    def objective(z):
        du = z[:2] - u_nom
        slack = z[2:]
        return 0.5 * float(du @ du) + 0.5 * slack_weight * float(slack @ slack)

    def objective_jac(z):
        grad = np.zeros_like(z)
        grad[:2] = z[:2] - u_nom
        grad[2:] = slack_weight * z[2:]
        return grad

    constraints = [
        {
            "type": "ineq",
            "fun": lambda z: a_mat @ z[:2] + z[2:] - b_vec,
            "jac": lambda z: np.hstack([a_mat, np.eye(m)]),
        }
    ]
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
            bounds=[(-saturation, saturation), (-saturation, saturation)] + [(0.0, None)] * m,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 50, "disp": False},
        )
    z = sol.x if sol.success else z0
    u = _clip_by_inf_norm(z[:2], saturation)
    slack = np.maximum(z[2:], 0.0)
    return u, bool(sol.success), slack


def _add_tangent_escape(v_des, pos, goal, vel, obstacles, speed_limit, activation_dist=0.35):
    if not obstacles:
        return v_des

    closest = None
    for obs in obstacles:
        h = _p_norm_distance(pos, obs) - obs.r1
        normal = _distance_grad(pos, obs)
        if normal is None:
            continue
        if closest is None or h < closest[0]:
            closest = (h, normal)

    if closest is None:
        return v_des

    h, normal = closest
    if h >= activation_dist:
        return v_des

    goal_dir = goal - pos
    goal_norm = np.linalg.norm(goal_dir)
    if goal_norm < 1e-12:
        return v_des
    goal_dir = goal_dir / goal_norm
    tangent = goal_dir - float(goal_dir @ normal) * normal
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-8:
        tangent = np.array([-normal[1], normal[0]], dtype=float)
        if np.linalg.norm(vel) > 1e-8 and float(tangent @ vel) < 0.0:
            tangent = -tangent
    else:
        tangent = tangent / tangent_norm

    strength = np.clip((activation_dist - h) / activation_dist, 0.0, 1.0)
    outward = max(0.0, 0.15 - h)
    adjusted = v_des + 0.5 * speed_limit * strength * tangent + outward * normal
    return _clip_by_inf_norm(adjusted, speed_limit)


def solve_double_integrator_density_qp(
    state,
    goal,
    alpha,
    obstacles,
    prev_v_des,
    *,
    accel_obstacles=None,
    dt=0.01,
    ctrl_multiplier=2.0,
    rad_from_goal=1.0,
    q_lqr=None,
    r_lqr=1.0,
    saturation=1.0,
    k_backstep=4.0,
    cdf_rate=0.1,
    slack_weight=1e4,
    accel_slack_weight=1e5,
    barrier_kp=4.0,
    barrier_kd=4.0,
):
    """Density-QP adapter for ``pos_dot = vel, vel_dot = u``.

    The first-order ``solve_density_qp`` handles control-affine systems whose
    input appears in ``d rho / dt``.  A double integrator has relative degree
    two for position-only densities, so this adapter computes a safe desired
    velocity with ``solve_density_qp`` and tracks it with an acceleration QP.
    It lives in this module so callers still use the density-QP controller API
    instead of a separate controller package.
    """
    state = np.asarray(state, dtype=float)
    goal = np.asarray(goal, dtype=float)
    prev_v_des = np.asarray(prev_v_des, dtype=float)
    pos = state[:2]
    vel = state[2:]
    accel_obstacles = obstacles if accel_obstacles is None else accel_obstacles
    dist = np.linalg.norm(pos - goal)
    sat = scaled_saturation(dist, saturation, rad_from_goal)

    if q_lqr is None:
        q_lqr = np.diag([10.0, 10.0, 10.0, 10.0])

    if dist < rad_from_goal:
        k_lqr = _double_integrator_lqr_gain(dt, q_lqr, r_lqr)
        goal_state = np.array([goal[0], goal[1], 0.0, 0.0], dtype=float)
        u = -k_lqr @ (state - goal_state)
        u = _clip_by_inf_norm(u, sat)
        return DoubleIntegratorQPResult(
            u=u,
            v_des=np.zeros(2, dtype=float),
            qp=None,
            accel_success=True,
            accel_slack=np.zeros(0),
            using_lqr=True,
        )

    v_nom = ctrl_multiplier * density_grad(pos, goal, alpha, obstacles)
    v_nom = _clip_by_inf_norm(v_nom, ctrl_multiplier)
    qp = solve_density_qp(
        pos,
        goal,
        alpha,
        obstacles,
        dynamics={"B": np.eye(2), "u_dim": 2},
        u_nom=v_nom,
        ctrl_multiplier=ctrl_multiplier,
        rad_from_goal=0.0,
        q_lqr=1.0,
        r_lqr=1.0,
        dt=dt,
        saturation=ctrl_multiplier,
        cdf_rate=cdf_rate,
        slack_weight=slack_weight,
        return_info=True,
    )
    v_des = qp.u
    v_des = _add_tangent_escape(v_des, pos, goal, vel, accel_obstacles, ctrl_multiplier)
    v_des_dot = (v_des - prev_v_des) / dt
    u_nom = v_des_dot - k_backstep * (vel - v_des)
    u, accel_success, accel_slack = _solve_accel_filter(
        u_nom,
        pos,
        vel,
        accel_obstacles,
        sat,
        barrier_kp=barrier_kp,
        barrier_kd=barrier_kd,
        slack_weight=accel_slack_weight,
    )
    return DoubleIntegratorQPResult(
        u=u,
        v_des=v_des,
        qp=qp,
        accel_success=accel_success,
        accel_slack=accel_slack,
        using_lqr=False,
    )
