from pathlib import Path
import argparse
import sys

import numpy as np
from scipy.optimize import minimize

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import SOLVER_CHOICES
from density_utils.controllers.solver_utils import require_solver
from density_utils.density import Obstacle
from density_utils.sim import forward_euler
from density_utils.utils.timing import TimedBlock

from _plotting import plot_single_integrator_results


def _p_norm_power(x, obs):
    dx = x - obs.center
    if obs.angle:
        c = np.cos(-obs.angle)
        s = np.sin(-obs.angle)
        dx = np.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]])
    if obs.scale is not None:
        dx = dx / obs.scale
    return float(np.sum(np.abs(dx) ** obs.p))


def _p_norm_distance(x, obs):
    return _p_norm_power(x, obs) ** (1.0 / obs.p)


def _barrier_fn(obs):
    def h(x):
        return _p_norm_power(x, obs) - obs.r1 ** obs.p

    return h


def _clf_fn(goal):
    def v(x):
        err = x - goal
        return float(err @ err)

    return v


def _solve_clf_cbf_filter(
    x,
    goal,
    obs,
    *,
    gamma,
    clf_rate,
    u_min,
    u_max,
    cbf_slack_weight,
    clf_slack_weight,
    solver="auto",
):
    require_solver(solver, ("scipy_slsqp",), controller="_solve_clf_cbf_filter")
    x = np.asarray(x, dtype=float)
    goal = np.asarray(goal, dtype=float)
    rel_obs = x - obs.center
    h = float(rel_obs @ rel_obs - obs.r1 ** 2)
    grad_h = 2.0 * rel_obs

    err = x - goal
    v = float(err @ err)
    grad_v = 2.0 * err

    # z = [u_x, u_y, cbf_slack, clf_slack]
    u_goal = -0.5 * float(clf_rate) * err
    z0 = np.array(
        [
            np.clip(u_goal[0], u_min[0], u_max[0]),
            np.clip(u_goal[1], u_min[1], u_max[1]),
            0.0,
            max(0.0, grad_v @ u_goal + float(clf_rate) * v),
        ],
        dtype=float,
    )

    def objective(z):
        return (
            0.5 * float(z[0] ** 2 + z[1] ** 2)
            + 0.5 * float(cbf_slack_weight) * float(z[2] ** 2)
            + 0.5 * float(clf_slack_weight) * float(z[3] ** 2)
        )

    def objective_jac(z):
        return np.array(
            [
                z[0],
                z[1],
                float(cbf_slack_weight) * z[2],
                float(clf_slack_weight) * z[3],
            ],
            dtype=float,
        )

    def cbf_constraint(z):
        return grad_h @ z[:2] + float(gamma) * h + z[2]

    def cbf_constraint_jac(_z):
        return np.array([grad_h[0], grad_h[1], 1.0, 0.0], dtype=float)

    def clf_constraint(z):
        return -grad_v @ z[:2] - float(clf_rate) * v + z[3]

    def clf_constraint_jac(_z):
        return np.array([-grad_v[0], -grad_v[1], 0.0, 1.0], dtype=float)

    result = minimize(
        objective,
        z0,
        jac=objective_jac,
        bounds=[(u_min[0], u_max[0]), (u_min[1], u_max[1]), (0.0, None), (0.0, None)],
        constraints=[
            {"type": "ineq", "fun": cbf_constraint, "jac": cbf_constraint_jac},
            {"type": "ineq", "fun": clf_constraint, "jac": clf_constraint_jac},
        ],
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 100, "disp": False},
    )
    z = result.x if np.all(np.isfinite(result.x)) else z0
    return {
        "u": np.clip(z[:2], u_min, u_max),
        "cbf_slack": max(float(z[2]), 0.0),
        "clf_slack": max(float(z[3]), 0.0),
        "success": bool(result.success),
        "message": str(result.message),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run the simulation without opening plots.")
    parser.add_argument("--steps", type=int, default=4000, help="Maximum simulation steps.")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto", help="Optimizer backend.")
    parser.add_argument("--verbose", action="store_true", help="Print solver failure diagnostics.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Discrete-time CBF rate in (0, 1].")
    parser.add_argument("--clf-rate", type=float, default=0.20, help="Discrete-time CLF decrease rate in (0, 1].")
    args = parser.parse_args()

    dt = 0.1
    steps = args.steps
    rad_from_goal = 0.35
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    stop_when_stable = True
    u_max = np.array([2.0, 2.0])
    u_min = -u_max
    cbf_slack_weight = 1e6
    clf_slack_weight = 1e4
    animate = not args.no_plot
    save_animation = args.save_gif
    animation_stride = 10
    animation_fps = 20
    animation_format = "gif"
    animation_path = Path("animations") / f"single_integrator_static_cbf_filter.{animation_format}"

    agent_radius = 0.1
    start = np.array([-2.0, -1.0])
    goal = np.array([2.0, 1.1])
    obstacle = Obstacle(center=np.array([0.0, 0.0]), r1=0.6, r2=1.0, p=2.0)
    inflated_obstacle = Obstacle(
        center=obstacle.center,
        r1=obstacle.r1 + agent_radius,
        r2=obstacle.r2 + agent_radius,
        p=obstacle.p,
        scale=obstacle.scale,
        angle=obstacle.angle,
    )
    x = start.copy()
    traj = [x.copy()]
    controls = []
    slacks = []
    solver_failures = 0
    min_clearance = _p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1

    control_time = 0.0
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    stop_count = 0
    for step in range(steps):
        dist = np.linalg.norm(x - goal)
        with timer:
            filter_result = _solve_clf_cbf_filter(
                x,
                goal,
                inflated_obstacle,
                gamma=args.gamma,
                clf_rate=args.clf_rate,
                u_min=u_min,
                u_max=u_max,
                cbf_slack_weight=cbf_slack_weight,
                clf_slack_weight=clf_slack_weight,
                solver=args.solver,
            )
            u = filter_result["u"]
        control_time += timer.last
        controls.append(u.copy())
        slacks.append(filter_result["cbf_slack"])
        if not filter_result["success"]:
            solver_failures += 1

        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        clearance = _p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1
        min_clearance = min(min_clearance, clearance)

        if stop_when_stable:
            if dist < stop_tol:
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0
        if step % print_interval == 0:
            print(
                f"iter={step} dist_to_goal={dist:.3f} "
                f"clearance={clearance:.3f} "
                f"cbf_slack={slacks[-1]:.2e} "
                f"clf_slack={filter_result['clf_slack']:.2e}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros_like(traj[0]))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    traj = np.array(traj)
    controls = np.array(controls)
    slacks = np.array(slacks)

    def _format_duration(seconds):
        if seconds < 1.0:
            return f"{seconds * 1e3:.1f} ms"
        return f"{seconds:.2f} s"

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    summary = (
        "steps="
        f"{steps_taken} "
        f"gamma={args.gamma:.3f} "
        f"clf_rate={args.clf_rate:.3f} "
        f"sim_time={_format_duration(control_time)} "
        f"avg_iteration={_format_duration(avg_control)} "
        f"min_clearance={min_clearance:.4f} "
        f"max_cbf_slack={np.max(slacks):.2e}"
    )
    if args.verbose:
        summary += f" solver_failures={solver_failures}"
    print(summary)

    if not args.no_plot:
        plot_single_integrator_results(
            traj=traj,
            controls=controls,
            slacks=slacks,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=[obstacle],
            agent_radius=agent_radius,
            title=f"CLF-CBF filter (gamma={args.gamma:.2f})",
            animate=animate,
            save_animation=save_animation,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
        )


if __name__ == "__main__":
    main()
