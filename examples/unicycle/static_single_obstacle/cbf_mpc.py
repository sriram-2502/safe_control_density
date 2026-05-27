from pathlib import Path
import argparse
import sys

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import SOLVER_CHOICES, solve_cbf_mpc
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
from density_mpc import _shift_controls


def _barrier_fn(obstacle):
    def h(state):
        distance = _p_norm_distance(np.asarray(state, dtype=float)[:2], obstacle)
        return distance * distance - float(obstacle.r1) ** 2

    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    parser.add_argument("--steps", type=int, default=None, help="Override maximum simulation steps.")
    parser.add_argument("--horizon", type=int, default=10, help="MPC prediction horizon.")
    parser.add_argument("--gamma", type=float, default=0.85, help="Discrete-time CBF rate in (0, 1].")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto", help="Optimizer backend.")
    parser.add_argument("--dt", type=float, default=None, help="Override simulation integration step.")
    parser.add_argument(
        "--mpc-dt",
        type=float,
        default=None,
        help="Override MPC prediction step. Defaults to the simulation dt for CBF safety checks.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print solver failure diagnostics.")
    args = parser.parse_args()

    cfg = CONFIG
    sim_cfg = cfg["simulation"]
    scenario_cfg = cfg["scenario"]
    control_cfg = cfg["control"]
    animation_cfg = cfg["animation"]

    dt = float(args.dt) if args.dt is not None else float(sim_cfg["dt"])
    mpc_dt = float(args.mpc_dt) if args.mpc_dt is not None else dt
    steps = args.steps if args.steps is not None else int(sim_cfg["steps"])
    horizon = int(args.horizon)
    stop_tol = float(sim_cfg.get("mpc_stop_tol", sim_cfg["stop_tol"]))
    stop_steps = int(sim_cfg["stop_steps"])
    stop_when_stable = bool(sim_cfg.get("stop_when_stable", True))
    u_min, u_max = _control_bounds(control_cfg)
    animation_stride = int(animation_cfg["stride"])
    animation_fps = int(animation_cfg["fps"])
    animation_path = Path("animations/unicycle_static_cbf_mpc.gif")

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
    traj = [state.copy()]
    controls = []
    slacks = []
    solver_failures = 0
    min_clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1
    previous_controls = None
    previous_control = np.zeros(2, dtype=float)

    control_time = 0.0
    timer = TimedBlock(enabled=True)
    print_interval = int(sim_cfg["print_interval"])
    stop_count = 0

    for step in range(steps):
        u_nom = np.zeros((horizon, 2), dtype=float)
        initial_controls = _shift_controls(previous_controls)
        if initial_controls is None:
            initial_controls = np.repeat(previous_control[None, :], horizon, axis=0)

        with timer:
            mpc_result = solve_cbf_mpc(
                state,
                goal_state,
                h_fns=h_fns,
                u_nom=u_nom,
                horizon=horizon,
                dt=mpc_dt,
                next_state_fn=_unicycle_filter_step,
                gamma=args.gamma,
                u_min=u_min,
                u_max=u_max,
                slack_weight=1e6,
                slack_l1_weight=0.0,
                slack_max=0.0,
                control_weight=np.diag([0.01, 0.01]),
                control_rate_weight=1.0,
                previous_control=previous_control,
                state_weight=np.diag([30.0, 30.0, 10.0]),
                terminal_weight=1000.0,
                initial_controls=initial_controls,
                solver=args.solver,
                return_info=True,
            )
        control_time += timer.last
        previous_controls = mpc_result.controls
        control = mpc_result.u
        previous_control = control.copy()
        controls.append(control.copy())
        slacks.append(float(np.max(mpc_result.slack)) if mpc_result.slack.size else 0.0)
        if not mpc_result.success:
            solver_failures += 1

        state = unicycle_step(state, float(control[0]), float(control[1]), dt)
        state[2] = _wrap_angle(state[2])
        traj.append(state.copy())
        clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1
        min_clearance = min(min_clearance, clearance)

        post_dist = float(np.linalg.norm(state[:2] - goal))
        heading_error = abs(_wrap_angle(state[2] - goal_state[2]))
        if stop_when_stable:
            if post_dist < stop_tol and heading_error < np.deg2rad(5.0):
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0
        if step % print_interval == 0:
            print(
                f"iter={step} dist_to_goal={post_dist:.3f} heading_error={heading_error:.3f} "
                f"clearance={clearance:.3f} cbf_slack={slacks[-1]:.2e} "
                f"solve_ms={timer.last * 1e3:.1f}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2, dtype=float))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    traj = np.asarray(traj, dtype=float)
    controls = np.asarray(controls, dtype=float)
    slacks = np.asarray(slacks, dtype=float)

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    summary = (
        "steps="
        f"{steps_taken} "
        f"gamma={args.gamma:.3f} "
        f"sim_time={control_time:.2f} s "
        f"avg_iteration={avg_control * 1e3:.1f} ms "
        f"min_clearance={min_clearance:.4f} "
        f"max_cbf_slack={np.max(slacks):.2e}"
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
            title=f"Unicycle - Static Obstacle (CBF MPC, N={horizon})",
            animate=not args.no_plot,
            save_animation=args.save_gif,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
            slacks=slacks,
        )


if __name__ == "__main__":
    main()
