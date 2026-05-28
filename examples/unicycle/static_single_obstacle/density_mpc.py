from pathlib import Path
import argparse
import sys

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import SOLVER_CHOICES, single_integrator_nominal_control, solve_density_mpc
from density_utils.density import Obstacle
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock

from _plotting import add_animation_save_args, animation_save_paths, plot_unicycle_results, wants_animation_output
from config import CONFIG
from density_filter import (
    _as_array,
    _control_bounds,
    _inflate_obstacles,
    _obstacle_from_config,
    _p_norm_distance,
    _pose_density,
    _unicycle_filter_step,
    _unicycle_nominal_from_planar_ref,
    _wrap_angle,
)


def _nominal_sequence(
    *,
    state,
    goal,
    goal_state,
    alpha,
    obstacles,
    horizon,
    density_dt,
    ctrl_multiplier,
    rad_from_goal,
    stop_tol,
    q_lqr,
    r_lqr,
    v_max,
    omega_max,
    k_heading,
    mode,
):
    """Roll a nominal unicycle sequence used as the MPC tracking reference."""
    state_nom = np.asarray(state, dtype=float).copy()
    sequence = []
    for _ in range(horizon):
        pos = state_nom[:2]
        dist = float(np.linalg.norm(pos - goal))
        if dist < stop_tol:
            omega_nom = np.clip(k_heading * _wrap_angle(goal_state[2] - state_nom[2]), -omega_max, omega_max)
            control = np.array([0.0, omega_nom], dtype=float)
        else:
            planar_ref = single_integrator_nominal_control(
                pos,
                goal,
                alpha,
                obstacles,
                mode=mode,
                ctrl_multiplier=ctrl_multiplier,
                rad_from_goal=rad_from_goal,
                q_lqr=q_lqr,
                r_lqr=r_lqr,
                dt=density_dt,
                u_min=[-v_max, -v_max],
                u_max=[v_max, v_max],
            )
            control = _unicycle_nominal_from_planar_ref(state_nom, planar_ref, v_max, omega_max, k_heading)
        sequence.append(control)
        state_nom = unicycle_step(state_nom, float(control[0]), float(control[1]), density_dt)
        state_nom[2] = _wrap_angle(state_nom[2])
    return np.asarray(sequence, dtype=float)


def _shift_controls(controls):
    if controls is None:
        return None
    shifted = np.asarray(controls, dtype=float).copy()
    shifted[:-1] = shifted[1:]
    shifted[-1] = shifted[-2] if len(shifted) > 1 else shifted[-1]
    return shifted


def main():
    parser = argparse.ArgumentParser()
    add_animation_save_args(parser)
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    parser.add_argument("--steps", type=int, default=None, help="Override maximum simulation steps.")
    parser.add_argument("--horizon", type=int, default=7, help="MPC prediction horizon.")
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="MPC solver backend.",
    )
    parser.add_argument("--dt", type=float, default=None, help="Override simulation integration step.")
    parser.add_argument(
        "--density-dt",
        type=float,
        default=None,
        help="Override MPC prediction step. Defaults to --dt when set, otherwise config density_dt.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print solver failure diagnostics.")
    args = parser.parse_args()

    cfg = CONFIG
    sim_cfg = cfg["simulation"]
    scenario_cfg = cfg["scenario"]
    density_cfg = cfg["density"]
    control_cfg = cfg["control"]
    animation_cfg = cfg["animation"]

    dt = float(args.dt) if args.dt is not None else float(sim_cfg["dt"])
    if args.density_dt is not None:
        density_dt = float(args.density_dt)
    elif args.dt is not None:
        density_dt = dt
    else:
        density_dt = float(sim_cfg["density_dt"])
    steps = args.steps if args.steps is not None else int(sim_cfg["steps"])
    horizon = int(args.horizon)
    alpha = float(density_cfg["alpha"])
    stop_tol = float(sim_cfg.get("mpc_stop_tol", sim_cfg["stop_tol"]))
    stop_steps = int(sim_cfg["stop_steps"])
    stop_when_stable = bool(sim_cfg.get("stop_when_stable", True))
    v_max = float(control_cfg["v_max"])
    omega_max = float(control_cfg["omega_max"])
    k_heading = float(control_cfg["k_heading"])
    u_min, u_max = _control_bounds(control_cfg)
    slack_weight = float(density_cfg["slack_weight"])
    animate = not args.no_plot or wants_animation_output(args)
    animation_stride = int(animation_cfg["stride"])
    animation_fps = int(animation_cfg["fps"])
    animation_path = EXAMPLE_ROOT / "animations/unicycle_static_density_mpc.gif"
    animation_paths = animation_save_paths(animation_path, save_gif=args.save_gif, save_mp4=args.save_mp4)

    agent_radius = float(scenario_cfg["agent_radius"])
    start = _as_array(scenario_cfg["start"])
    goal = _as_array(scenario_cfg["goal"])
    obstacles = [_obstacle_from_config(obs_cfg) for obs_cfg in scenario_cfg["obstacles"]]
    inflated_obstacles = _inflate_obstacles(obstacles, agent_radius)
    obstacle = obstacles[0]
    inflated_obstacle = inflated_obstacles[0]

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

    def density_fn(state_eval, goal_eval, alpha_eval, obstacles_eval):
        return _pose_density(state_eval, goal_eval, alpha_eval, obstacles_eval)

    control_time = 0.0
    timer = TimedBlock(enabled=True)
    print_interval = int(sim_cfg["print_interval"])
    stop_count = 0

    for step in range(steps):
        pos = state[:2]
        dist = float(np.linalg.norm(pos - goal))
        u_nom = np.zeros((horizon, 2), dtype=float)
        initial_controls = _shift_controls(previous_controls)
        if initial_controls is None:
            initial_controls = np.repeat(previous_control[None, :], horizon, axis=0)

        with timer:
            mpc_result = solve_density_mpc(
                state,
                goal_state,
                alpha,
                [inflated_obstacle],
                solver=args.solver,
                u_nom=u_nom,
                horizon=horizon,
                dt=density_dt,
                next_state_fn=_unicycle_filter_step,
                u_min=u_min,
                u_max=u_max,
                divergence=0.0,
                slack_weight=0.0,
                slack_l1_weight=1.0,
                control_weight=np.diag([0.01, 0.01]),
                control_rate_weight=1.0,
                previous_control=previous_control,
                state_weight=np.diag([30.0, 30.0, 10.0]),
                terminal_weight=1000.0,
                density_fn=density_fn,
                initial_controls=initial_controls,
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
                f"clearance={clearance:.3f} slack={slacks[-1]:.2e} "
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
        f"sim_time={control_time:.2f} s "
        f"avg_iteration={avg_control * 1e3:.1f} ms "
        f"min_clearance={min_clearance:.4f} "
        f"max_slack={np.max(slacks):.2e}"
    )
    if args.verbose:
        summary += f" solver_failures={solver_failures}"
    print(summary)

    if not args.no_plot or wants_animation_output(args):
        plot_unicycle_results(
            traj=traj,
            controls=controls,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=[obstacle],
            agent_radius=agent_radius,
            title=f"Unicycle - Static Obstacle (Density MPC, N={horizon})",
            animate=animate,
            save_animation=False,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
            slacks=slacks,
            animation_paths=animation_paths,
            show_plot=not args.no_plot,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )


if __name__ == "__main__":
    main()
