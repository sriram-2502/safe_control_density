from pathlib import Path
import argparse
import sys

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import SOLVER_CHOICES, solve_cbf_filter
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock

from _plotting import plot_unicycle_results
from config import CONFIG
from density_filter import (
    _as_array,
    _control_bounds,
    _inflate_obstacles,
    _obstacle_from_config,
    _p_norm_distance,
    _unicycle_filter_step,
    _wrap_angle,
)


def _barrier_fn(obstacle):
    def h(state):
        distance = _p_norm_distance(np.asarray(state, dtype=float)[:2], obstacle)
        return distance * distance - float(obstacle.r1) ** 2

    return h


def _clf_fn(goal_state, theta_weight=0.05):
    def v(state):
        state = np.asarray(state, dtype=float)
        err = np.array(
            [
                state[0] - goal_state[0],
                state[1] - goal_state[1],
                _wrap_angle(state[2] - goal_state[2]),
            ],
            dtype=float,
        )
        return float(err[0] ** 2 + err[1] ** 2 + float(theta_weight) * err[2] ** 2)

    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    parser.add_argument("--steps", type=int, default=None, help="Override maximum simulation steps.")
    parser.add_argument("--gamma", type=float, default=0.85, help="Discrete-time CBF rate in (0, 1].")
    parser.add_argument("--clf-rate", type=float, default=0.20, help="Discrete-time CLF rate in (0, 1].")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto", help="Optimizer backend.")
    parser.add_argument("--verbose", action="store_true", help="Print solver failure diagnostics.")
    args = parser.parse_args()

    cfg = CONFIG
    sim_cfg = cfg["simulation"]
    scenario_cfg = cfg["scenario"]
    control_cfg = cfg["control"]
    animation_cfg = cfg["animation"]

    dt = float(sim_cfg["dt"])
    steps = args.steps if args.steps is not None else int(sim_cfg["feedback_steps"])
    stop_tol = float(sim_cfg.get("feedback_stop_tol", sim_cfg["stop_tol"]))
    v_max = float(control_cfg["v_max"])
    omega_max = float(control_cfg["omega_max"])
    u_min, u_max = _control_bounds(control_cfg)
    animation_stride = int(animation_cfg["stride"])
    animation_fps = int(animation_cfg["fps"])
    animation_path = Path("animations/unicycle_static_clf_cbf_filter.gif")

    agent_radius = float(scenario_cfg["agent_radius"])
    start = _as_array(scenario_cfg["start"])
    goal = _as_array(scenario_cfg["goal"])
    obstacles = [_obstacle_from_config(obs_cfg) for obs_cfg in scenario_cfg["obstacles"]]
    inflated_obstacles = _inflate_obstacles(obstacles, agent_radius)
    obstacle = obstacles[0]
    inflated_obstacle = inflated_obstacles[0]
    h_fns = [_barrier_fn(obs) for obs in inflated_obstacles]

    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    state = np.array([start[0], start[1], heading0], dtype=float)
    goal_state = np.array([goal[0], goal[1], heading0], dtype=float)
    clf_fn = _clf_fn(goal_state)

    traj = [state.copy()]
    controls = []
    cbf_slacks = []
    clf_slacks = []
    solver_failures = 0
    min_clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1

    control_time = 0.0
    timer = TimedBlock(enabled=True)
    print_interval = int(sim_cfg["print_interval"])

    print(
        f"running CLF-CBF filter gamma={args.gamma:.3f} "
        f"clf_rate={args.clf_rate:.3f} stop_tol={stop_tol:.3f} max_steps={steps}"
    )

    for step in range(steps):
        u_nom = np.zeros(2, dtype=float)
        with timer:
            filter_result = solve_cbf_filter(
                state,
                h_fns=h_fns,
                dt=dt,
                u_nom=u_nom,
                next_state_fn=_unicycle_filter_step,
                gamma=args.gamma,
                clf_fn=clf_fn,
                clf_rate=args.clf_rate,
                u_min=u_min,
                u_max=u_max,
                slack_weight=1e6,
                slack_max=0.0,
                clf_slack_weight=1e4,
                control_weight=np.diag([0.01, 0.01]),
                solver=args.solver,
                return_info=True,
            )
        control_time += timer.last
        control = filter_result.u
        controls.append(control.copy())
        cbf_slacks.append(float(np.max(filter_result.slack)) if filter_result.slack.size else 0.0)
        clf_slacks.append(float(filter_result.clf_slack))
        if not filter_result.success:
            solver_failures += 1

        state = unicycle_step(state, float(control[0]), float(control[1]), dt)
        state[2] = _wrap_angle(state[2])
        traj.append(state.copy())
        clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1
        min_clearance = min(min_clearance, clearance)
        dist = float(np.linalg.norm(state[:2] - goal))

        if step % print_interval == 0:
            parts = [
                f"iter={step}",
                f"dist_to_goal={dist:.3f}",
                f"clearance={clearance:.3f}",
                f"cbf_slack={cbf_slacks[-1]:.2e}",
                f"clf_slack={clf_slacks[-1]:.2e}",
                f"solve_ms={timer.last * 1e3:.2f}",
            ]
            if args.verbose:
                parts.append(f"failures={solver_failures}")
            print(" ".join(parts))
        if dist < stop_tol:
            print(f"stopping at iter={step} (close to goal, dist={dist:.4f})")
            break

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2, dtype=float))
    if len(cbf_slacks) < len(traj):
        cbf_slacks.append(cbf_slacks[-1] if cbf_slacks else 0.0)

    traj = np.asarray(traj, dtype=float)
    controls = np.asarray(controls, dtype=float)
    cbf_slacks = np.asarray(cbf_slacks, dtype=float)
    clf_slacks = np.asarray(clf_slacks, dtype=float)

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    summary = (
        "steps="
        f"{steps_taken} "
        f"gamma={args.gamma:.3f} "
        f"clf_rate={args.clf_rate:.3f} "
        f"sim_time={control_time:.2f} s "
        f"avg_iteration={avg_control * 1e3:.2f} ms "
        f"min_clearance={min_clearance:.4f} "
        f"max_cbf_slack={np.max(cbf_slacks):.2e} "
        f"max_clf_slack={np.max(clf_slacks):.2e}"
    )
    if args.verbose:
        summary += f" solver_failures={solver_failures}"
    print(summary)

    if not args.no_plot:
        plot_unicycle_results(
            traj=traj,
            controls=controls,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=[obstacle],
            agent_radius=agent_radius,
            title=f"Unicycle - Static Obstacle (CLF-CBF filter, gamma={args.gamma:.2f})",
            animate=not args.no_plot,
            save_animation=args.save_gif,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
            slacks=cbf_slacks,
        )


if __name__ == "__main__":
    main()
