from pathlib import Path
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import SOLVER_CHOICES, solve_cbf_filter, solve_cbf_mpc
from density_utils.dynamics import unicycle_step

from config import CONFIG
import compare_density_controllers as dashboard
from density_filter import (
    _as_array,
    _control_bounds,
    _inflate_obstacles,
    _obstacle_from_config,
    _p_norm_distance,
    _pose_density,
    _unicycle_filter_step,
    _wrap_angle,
)
from density_mpc import _shift_controls


dashboard.STYLE.clear()
dashboard.STYLE.update(
    {
        "filter": {"label": "Density filter", "color": "tab:blue"},
        "mpc": {"label": "Density MPC", "color": "tab:orange"},
        "clf_cbf_filter": {"label": "CLF-CBF filter", "color": "tab:red"},
        "cbf_mpc": {"label": "CBF MPC", "color": "tab:green"},
    }
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


def _simulate_clf_cbf_filter(
    *,
    cfg,
    start,
    goal,
    obstacle,
    inflated_obstacle,
    steps,
    gamma,
    clf_rate,
    solver,
    early_stop,
    verbose=False,
):
    sim_cfg = cfg["simulation"]
    control_cfg = cfg["control"]
    dt = float(sim_cfg["dt"])
    stop_tol = float(sim_cfg.get("mpc_stop_tol", sim_cfg["stop_tol"]))
    print_interval = int(sim_cfg["print_interval"])
    u_min, u_max = _control_bounds(control_cfg)
    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    goal_state = np.array([goal[0], goal[1], heading0], dtype=float)
    h_fns = [_barrier_fn(inflated_obstacle)]
    clf_fn = _clf_fn(goal_state)

    print(f"running CLF-CBF filter gamma={gamma:.3f} clf_rate={clf_rate:.3f} stop_tol={stop_tol:.3f}")
    state = np.array([start[0], start[1], heading0], dtype=float)
    traj = [state.copy()]
    controls = []
    density = [_pose_density(state, goal_state, float(cfg["density"]["alpha"]), [inflated_obstacle])]
    clearance = [_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1]
    slacks = []
    solve_times = []
    failures = 0

    for step in range(steps):
        solve_start = time.perf_counter()
        result = solve_cbf_filter(
            state,
            h_fns=h_fns,
            dt=dt,
            u_nom=np.zeros(2, dtype=float),
            next_state_fn=_unicycle_filter_step,
            gamma=gamma,
            clf_fn=clf_fn,
            clf_rate=clf_rate,
            u_min=u_min,
            u_max=u_max,
            slack_weight=1e6,
            slack_max=0.0,
            clf_slack_weight=1e4,
            control_weight=np.diag([0.01, 0.01]),
            solver=solver,
            return_info=True,
        )
        solve_times.append(time.perf_counter() - solve_start)
        if not result.success:
            failures += 1
        control = result.u
        controls.append(control.copy())
        slacks.append(float(np.max(result.slack)) if result.slack.size else 0.0)

        state = unicycle_step(state, float(control[0]), float(control[1]), dt)
        state[2] = _wrap_angle(state[2])
        traj.append(state.copy())
        density.append(_pose_density(state, goal_state, float(cfg["density"]["alpha"]), [inflated_obstacle]))
        clearance.append(_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1)
        dist = float(np.linalg.norm(state[:2] - goal))
        if step % print_interval == 0:
            dashboard._print_progress(
                "clf_cbf_filter",
                step,
                dist,
                clearance[-1],
                solve_times[-1],
                slack=slacks[-1],
                failures=failures,
                verbose=verbose,
            )
        if early_stop and dist < stop_tol:
            print(f"clf_cbf_filter stopping at iter={step} (close to goal, dist={dist:.4f})")
            break

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2, dtype=float))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)
    return dashboard._make_result(
        "clf_cbf_filter",
        dt,
        traj,
        controls,
        density,
        clearance,
        slacks,
        solve_times,
        failures,
        obstacle,
    )


def _simulate_cbf_mpc(
    *,
    cfg,
    start,
    goal,
    obstacle,
    inflated_obstacle,
    steps,
    horizon,
    gamma,
    solver,
    early_stop,
    verbose=False,
):
    sim_cfg = cfg["simulation"]
    control_cfg = cfg["control"]
    dt = float(sim_cfg["dt"])
    stop_tol = float(sim_cfg.get("mpc_stop_tol", sim_cfg["stop_tol"]))
    stop_steps = int(sim_cfg["stop_steps"])
    stop_when_stable = bool(sim_cfg.get("stop_when_stable", True))
    print_interval = int(sim_cfg["print_interval"])
    u_min, u_max = _control_bounds(control_cfg)
    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    goal_state = np.array([goal[0], goal[1], heading0], dtype=float)
    h_fns = [_barrier_fn(inflated_obstacle)]

    print(f"running CBF MPC horizon={horizon} gamma={gamma:.3f} stop_tol={stop_tol:.3f}")
    state = np.array([start[0], start[1], heading0], dtype=float)
    traj = [state.copy()]
    controls = []
    density = [_pose_density(state, goal_state, float(cfg["density"]["alpha"]), [inflated_obstacle])]
    clearance = [_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1]
    slacks = []
    solve_times = []
    failures = 0
    stop_count = 0
    previous_controls = None
    previous_control = np.zeros(2, dtype=float)

    for step in range(steps):
        initial_controls = _shift_controls(previous_controls)
        if initial_controls is None:
            initial_controls = np.repeat(previous_control[None, :], horizon, axis=0)
        solve_start = time.perf_counter()
        result = solve_cbf_mpc(
            state,
            goal_state,
            h_fns=h_fns,
            u_nom=np.zeros((horizon, 2), dtype=float),
            horizon=horizon,
            dt=dt,
            next_state_fn=_unicycle_filter_step,
            gamma=gamma,
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
            solver=solver,
            return_info=True,
        )
        solve_times.append(time.perf_counter() - solve_start)
        previous_controls = result.controls
        if not result.success:
            failures += 1
        control = result.u
        previous_control = control.copy()
        controls.append(control.copy())
        slacks.append(float(np.max(result.slack)) if result.slack.size else 0.0)

        state = unicycle_step(state, float(control[0]), float(control[1]), dt)
        state[2] = _wrap_angle(state[2])
        traj.append(state.copy())
        density.append(_pose_density(state, goal_state, float(cfg["density"]["alpha"]), [inflated_obstacle]))
        clearance.append(_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1)
        dist = float(np.linalg.norm(state[:2] - goal))
        heading_error = abs(_wrap_angle(state[2] - goal_state[2]))
        if step % print_interval == 0:
            dashboard._print_progress(
                "cbf_mpc",
                step,
                dist,
                clearance[-1],
                solve_times[-1],
                slack=slacks[-1],
                failures=failures,
                verbose=verbose,
            )
        if early_stop and stop_when_stable:
            if dist < stop_tol and heading_error < np.deg2rad(5.0):
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"cbf_mpc stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2, dtype=float))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)
    return dashboard._make_result(
        "cbf_mpc",
        dt,
        traj,
        controls,
        density,
        clearance,
        slacks,
        solve_times,
        failures,
        obstacle,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=int(CONFIG["simulation"]["steps"]),
        help="Maximum simulation steps for each controller.",
    )
    parser.add_argument("--density-horizon", type=int, default=5, help="Density MPC prediction horizon.")
    parser.add_argument("--cbf-horizon", type=int, default=10, help="CBF MPC prediction horizon.")
    parser.add_argument("--gamma", type=float, default=0.85, help="CBF rate for CLF-CBF filter and CBF MPC.")
    parser.add_argument("--clf-rate", type=float, default=0.20, help="CLF rate for the one-step CLF-CBF filter.")
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="Optimizer backend. Density MPC supports all choices; CBF controllers currently use scipy_slsqp.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=int(CONFIG["animation"]["stride"]),
        help="Dashboard GIF frame stride.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=int(CONFIG["animation"]["fps"]),
        help="Dashboard GIF playback frame rate.",
    )
    parser.add_argument(
        "--fixed-steps",
        action="store_true",
        help="Run each controller for exactly --steps iterations instead of stopping near the goal.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["density_filter", "density_mpc", "clf_cbf_filter", "cbf_mpc"],
        default=["density_filter", "density_mpc", "clf_cbf_filter", "cbf_mpc"],
        help="Controllers to include in the comparison.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print solver failure diagnostics.")
    parser.add_argument("--no-gif", action="store_true", help="Skip saving the dashboard GIF.")
    parser.add_argument("--save-mp4", action="store_true", help="Save the dashboard animation as compact MP4.")
    parser.add_argument("--mp4-crf", type=int, default=28, help="MP4 quality factor. Higher is smaller.")
    parser.add_argument("--mp4-preset", default="slow", help="ffmpeg x264 preset.")
    parser.add_argument("--no-show", action="store_true", help="Save outputs without opening matplotlib windows.")
    args = parser.parse_args()

    cfg = CONFIG
    scenario_cfg = cfg["scenario"]
    agent_radius = float(scenario_cfg["agent_radius"])
    start = _as_array(scenario_cfg["start"])
    goal = _as_array(scenario_cfg["goal"])
    obstacles = [_obstacle_from_config(obs_cfg) for obs_cfg in scenario_cfg["obstacles"]]
    inflated_obstacles = _inflate_obstacles(obstacles, agent_radius)
    obstacle = obstacles[0]
    inflated_obstacle = inflated_obstacles[0]
    early_stop = not args.fixed_steps

    results = []
    if "density_filter" in args.controllers:
        results.append(
            dashboard._simulate_filter(
                cfg=cfg,
                start=start,
                goal=goal,
                obstacle=obstacle,
                inflated_obstacle=inflated_obstacle,
                steps=args.steps,
                solver=args.solver,
                early_stop=early_stop,
                verbose=args.verbose,
            )
        )
    if "density_mpc" in args.controllers:
        results.append(
            dashboard._simulate_mpc(
                cfg=cfg,
                start=start,
                goal=goal,
                obstacle=obstacle,
                inflated_obstacle=inflated_obstacle,
                steps=args.steps,
                horizon=args.density_horizon,
                solver=args.solver,
                early_stop=early_stop,
                verbose=args.verbose,
            )
        )
    if "clf_cbf_filter" in args.controllers:
        results.append(
            _simulate_clf_cbf_filter(
                cfg=cfg,
                start=start,
                goal=goal,
                obstacle=obstacle,
                inflated_obstacle=inflated_obstacle,
                steps=args.steps,
                gamma=args.gamma,
                clf_rate=args.clf_rate,
                solver=args.solver,
                early_stop=early_stop,
                verbose=args.verbose,
            )
        )
    if "cbf_mpc" in args.controllers:
        results.append(
            _simulate_cbf_mpc(
                cfg=cfg,
                start=start,
                goal=goal,
                obstacle=obstacle,
                inflated_obstacle=inflated_obstacle,
                steps=args.steps,
                horizon=args.cbf_horizon,
                gamma=args.gamma,
                solver=args.solver,
                early_stop=early_stop,
                verbose=args.verbose,
            )
        )

    for result in results:
        dashboard._summarize(result, goal, verbose=args.verbose)

    output_dir = Path(__file__).resolve().parent / "comparison_results"
    xy_path = output_dir / "unicycle_static_safety_filters_xy.png"
    ts_path = output_dir / "unicycle_static_safety_filters_timeseries.png"
    gif_path = output_dir / "unicycle_static_safety_filters.gif"
    mp4_path = gif_path.with_suffix(".mp4")

    figures = [
        dashboard._plot_xy(results, start=start, goal=goal, agent_radius=agent_radius, path=xy_path),
        dashboard._plot_time_series(results, goal=goal, path=ts_path),
    ]
    animations_to_show = []
    if not args.no_gif:
        fig, ani = dashboard._save_dashboard_animation(
            results,
            start=start,
            goal=goal,
            agent_radius=agent_radius,
            path=gif_path,
            stride=args.stride,
            fps=args.fps,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )
        figures.append(fig)
        animations_to_show.append(ani)
    if args.save_mp4:
        fig, ani = dashboard._save_dashboard_animation(
            results,
            start=start,
            goal=goal,
            agent_radius=agent_radius,
            path=mp4_path,
            stride=args.stride,
            fps=args.fps,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )
        figures.append(fig)
        animations_to_show.append(ani)

    print(f"saved XY plot: {xy_path}")
    print(f"saved time-series plot: {ts_path}")
    if not args.no_gif:
        print(f"saved dashboard GIF: {gif_path}")
    if args.save_mp4:
        print(f"saved dashboard MP4: {mp4_path}")

    if args.no_show:
        for fig in figures:
            plt.close(fig)
    else:
        plt.show()
    _ = animations_to_show


if __name__ == "__main__":
    main()
