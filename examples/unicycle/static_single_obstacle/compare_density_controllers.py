from pathlib import Path
import argparse
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import (
    SOLVER_CHOICES,
    density_feedback_control,
    single_integrator_nominal_control,
    solve_density_mpc,
    solve_discrete_density_filter,
)
from density_utils.density import Obstacle, density_value
from density_utils.dynamics import unicycle_step
from density_utils.utils import plot_goal, plot_obstacle, plot_start

from _plotting import save_animation_file
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
from density_mpc import _nominal_sequence, _shift_controls


STYLE = {
    "feedback": {"label": "Density feedback", "color": "tab:blue"},
    "filter": {"label": "Density filter", "color": "tab:orange"},
    "mpc": {"label": "Density MPC", "color": "tab:green"},
}


def _pad_rows(values, target_len):
    values = np.asarray(values, dtype=float)
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return np.zeros((target_len, 2), dtype=float)
    return np.vstack([values, np.repeat(values[-1:], target_len - len(values), axis=0)])


def _pad_1d(values, target_len):
    values = np.asarray(values, dtype=float)
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return np.zeros(target_len, dtype=float)
    return np.concatenate([values, np.full(target_len - len(values), values[-1], dtype=float)])


def _axis_limits(*arrays, pad_fraction=0.08):
    data = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    data = data[np.isfinite(data)]
    if data.size == 0:
        return -1.0, 1.0
    lo = float(np.min(data))
    hi = float(np.max(data))
    pad = max(1e-3, (hi - lo) * pad_fraction if abs(hi - lo) > 1e-9 else max(abs(hi), 1.0) * pad_fraction)
    return lo - pad, hi + pad


def _triangle_points(center, heading, size):
    center = np.asarray(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    tip = center + size * 1.35 * forward
    left = center - size * 0.9 * forward + size * 0.65 * right
    right_pt = center - size * 0.9 * forward - size * 0.65 * right
    return np.stack([tip, left, right_pt], axis=0)


def _control_variation(controls, dt):
    controls = np.asarray(controls, dtype=float)
    if len(controls) < 2:
        return np.zeros(len(controls), dtype=float)
    du = np.linalg.norm(np.diff(controls, axis=0), axis=1) / max(float(dt), 1e-12)
    return np.concatenate([[0.0], du])


def _print_progress(name, step, dist, clearance, solve_time, *, slack=None, failures=None, verbose=False):
    parts = [
        f"{name} iter={step}",
        f"dist_to_goal={dist:.3f}",
        f"clearance={clearance:.3f}",
        f"solve_ms={solve_time * 1e3:.2f}",
    ]
    if slack is not None:
        parts.append(f"slack={slack:.2e}")
    if verbose and failures is not None:
        parts.append(f"failures={failures}")
    print(" ".join(parts))


def _simulate_feedback(*, cfg, start, goal, obstacle, inflated_obstacle, steps, early_stop):
    sim_cfg = cfg["simulation"]
    density_cfg = cfg["density"]
    control_cfg = cfg["control"]
    dt = float(sim_cfg["dt"])
    alpha = 0.4
    ctrl_multiplier = 1.0
    rad_from_goal = 0.01
    stop_tol = float(sim_cfg.get("feedback_stop_tol", sim_cfg["stop_tol"]))
    print_interval = int(sim_cfg["print_interval"])
    q_lqr = float(density_cfg["q_lqr"])
    r_lqr = float(density_cfg["r_lqr"])
    saturation = 4.0
    k_heading = 2.0
    v_max = float(control_cfg["v_max"])
    omega_max = float(control_cfg["omega_max"])

    print(f"running density feedback early_stop={early_stop} stop_tol={stop_tol:.3f} max_steps={steps}")
    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    state = np.array([start[0], start[1], heading0], dtype=float)
    tilde_prev = heading0
    traj = [state.copy()]
    controls = []
    density = [density_value(state[:2], goal, alpha, [inflated_obstacle])]
    clearance = [_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1]
    solve_times = []

    for step in range(steps):
        pos = state[:2]
        solve_start = time.perf_counter()
        planar_ref = density_feedback_control(
            pos,
            goal,
            alpha,
            [inflated_obstacle],
            ctrl_multiplier=ctrl_multiplier,
            rad_from_goal=rad_from_goal,
            q_lqr=q_lqr,
            r_lqr=r_lqr,
            dt=dt,
            saturation=saturation,
        )
        solve_times.append(time.perf_counter() - solve_start)
        v = min(float(np.linalg.norm(planar_ref)), v_max)
        tilde = float(np.arctan2(planar_ref[1], planar_ref[0])) if v > 1e-10 else tilde_prev
        tilde_dot = _wrap_angle(tilde - tilde_prev) / dt
        tilde_prev = tilde
        omega = float(np.clip(tilde_dot - k_heading * _wrap_angle(state[2] - tilde), -omega_max, omega_max))
        controls.append([v, omega])
        state = unicycle_step(state, v, omega, dt)
        state[2] = _wrap_angle(state[2])
        traj.append(state.copy())
        density.append(density_value(state[:2], goal, alpha, [inflated_obstacle]))
        clearance.append(_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1)
        post_dist = float(np.linalg.norm(state[:2] - goal))
        if step % print_interval == 0:
            _print_progress(
                "feedback",
                step,
                post_dist,
                clearance[-1],
                solve_times[-1],
            )
        if early_stop and post_dist < stop_tol:
            print(f"feedback stopping at iter={step} (close to goal, dist={post_dist:.4f})")
            break

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))
    return _make_result("feedback", dt, traj, controls, density, clearance, None, solve_times, 0, obstacle)


def _simulate_filter(*, cfg, start, goal, obstacle, inflated_obstacle, steps, solver, early_stop, verbose=False):
    sim_cfg = cfg["simulation"]
    density_cfg = cfg["density"]
    control_cfg = cfg["control"]
    dt = float(sim_cfg["dt"])
    density_dt = float(sim_cfg["density_dt"])
    alpha = float(density_cfg["alpha"])
    ctrl_multiplier = float(density_cfg["ctrl_multiplier"])
    rad_from_goal = float(density_cfg["rad_from_goal"])
    stop_tol = float(sim_cfg["stop_tol"])
    print_interval = int(sim_cfg["print_interval"])
    q_lqr = float(density_cfg["q_lqr"])
    r_lqr = float(density_cfg["r_lqr"])
    v_max = float(control_cfg["v_max"])
    omega_max = float(control_cfg["omega_max"])
    k_heading = float(control_cfg["k_heading"])
    u_min, u_max = _control_bounds(control_cfg)
    slack_weight = float(density_cfg["slack_weight"])
    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    goal_state = np.array([goal[0], goal[1], heading0], dtype=float)

    print(f"running density filter early_stop={early_stop} stop_tol={stop_tol:.3f}")
    state = np.array([start[0], start[1], heading0], dtype=float)
    traj = [state.copy()]
    controls = []
    density = [_pose_density(state, goal_state, alpha, [inflated_obstacle])]
    clearance = [_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1]
    slacks = []
    solve_times = []
    failures = 0

    def density_fn(state_eval, goal_eval, alpha_eval, obstacles_eval):
        return _pose_density(state_eval, goal_eval, alpha_eval, obstacles_eval)

    for step in range(steps):
        pos = state[:2]
        dist = float(np.linalg.norm(pos - goal))
        solve_start = time.perf_counter()
        if dist < stop_tol:
            omega_nom = np.clip(k_heading * _wrap_angle(goal_state[2] - state[2]), -omega_max, omega_max)
            u_nom = np.array([0.0, omega_nom], dtype=float)
        else:
            planar_ref = single_integrator_nominal_control(
                pos,
                goal,
                alpha,
                [inflated_obstacle],
                mode="density_blend",
                ctrl_multiplier=ctrl_multiplier,
                rad_from_goal=rad_from_goal,
                q_lqr=q_lqr,
                r_lqr=r_lqr,
                dt=density_dt,
                u_min=[-v_max, -v_max],
                u_max=[v_max, v_max],
            )
            u_nom = _unicycle_nominal_from_planar_ref(state, planar_ref, v_max, omega_max, k_heading)
        result = solve_discrete_density_filter(
            state,
            goal_state,
            alpha,
            [inflated_obstacle],
            u_nom=u_nom,
            dt=density_dt,
            next_state_fn=_unicycle_filter_step,
            u_min=u_min,
            u_max=u_max,
            divergence=0.0,
            slack_weight=slack_weight,
            density_fn=density_fn,
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
        density.append(_pose_density(state, goal_state, alpha, [inflated_obstacle]))
        clearance.append(_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1)
        post_dist = float(np.linalg.norm(state[:2] - goal))
        if step % print_interval == 0:
            _print_progress(
                "filter",
                step,
                post_dist,
                clearance[-1],
                solve_times[-1],
                slack=slacks[-1],
                failures=failures,
                verbose=verbose,
            )
        if early_stop and post_dist < stop_tol:
            print(f"filter stopping at iter={step} (close to goal, dist={post_dist:.4f})")
            break

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)
    return _make_result("filter", dt, traj, controls, density, clearance, slacks, solve_times, failures, obstacle)


def _simulate_mpc(*, cfg, start, goal, obstacle, inflated_obstacle, steps, horizon, solver, early_stop, verbose=False):
    sim_cfg = cfg["simulation"]
    density_cfg = cfg["density"]
    control_cfg = cfg["control"]
    dt = float(sim_cfg["dt"])
    density_dt = float(sim_cfg["density_dt"])
    alpha = float(density_cfg["alpha"])
    stop_tol = float(sim_cfg.get("mpc_stop_tol", sim_cfg["stop_tol"]))
    stop_steps = int(sim_cfg["stop_steps"])
    stop_when_stable = bool(sim_cfg.get("stop_when_stable", True))
    print_interval = int(sim_cfg["print_interval"])
    u_min, u_max = _control_bounds(control_cfg)
    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    goal_state = np.array([goal[0], goal[1], heading0], dtype=float)

    print(
        f"running density MPC horizon={horizon} solver={solver} "
        f"early_stop={early_stop} stop_tol={stop_tol:.3f} stop_steps={stop_steps}"
    )
    state = np.array([start[0], start[1], heading0], dtype=float)
    traj = [state.copy()]
    controls = []
    density = [_pose_density(state, goal_state, alpha, [inflated_obstacle])]
    clearance = [_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1]
    slacks = []
    solve_times = []
    failures = 0
    stop_count = 0
    previous_controls = None
    previous_control = np.zeros(2, dtype=float)

    def density_fn(state_eval, goal_eval, alpha_eval, obstacles_eval):
        return _pose_density(state_eval, goal_eval, alpha_eval, obstacles_eval)

    for step in range(steps):
        pos = state[:2]
        dist = float(np.linalg.norm(pos - goal))
        u_nom = np.zeros((horizon, 2), dtype=float)
        initial_controls = _shift_controls(previous_controls)
        if initial_controls is None:
            initial_controls = np.repeat(previous_control[None, :], horizon, axis=0)
        solve_start = time.perf_counter()
        result = solve_density_mpc(
            state,
            goal_state,
            alpha,
            [inflated_obstacle],
            solver=solver,
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
        density.append(_pose_density(state, goal_state, alpha, [inflated_obstacle]))
        clearance.append(_p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1)
        post_dist = float(np.linalg.norm(state[:2] - goal))
        heading_error = abs(_wrap_angle(state[2] - goal_state[2]))
        if step % print_interval == 0:
            _print_progress(
                "mpc",
                step,
                post_dist,
                clearance[-1],
                solve_times[-1],
                slack=slacks[-1],
                failures=failures,
                verbose=verbose,
            )
        if early_stop and stop_when_stable:
            if post_dist < stop_tol and heading_error < np.deg2rad(5.0):
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"mpc stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)
    return _make_result("mpc", dt, traj, controls, density, clearance, slacks, solve_times, failures, obstacle)


def _make_result(name, dt, traj, controls, density, clearance, slacks, solve_times, failures, obstacle):
    traj = np.asarray(traj, dtype=float)
    controls = _pad_rows(controls, len(traj))
    if slacks is None:
        slacks = np.zeros(len(traj), dtype=float)
    else:
        slacks = _pad_1d(slacks, len(traj))
    return {
        "name": name,
        "dt": float(dt),
        "traj": traj,
        "controls": controls,
        "density": _pad_1d(density, len(traj)),
        "clearance": _pad_1d(clearance, len(traj)),
        "slack": slacks,
        "solve_times": np.asarray(solve_times, dtype=float),
        "solver_failures": int(failures),
        "obstacle": obstacle,
        "variation": _control_variation(controls, dt),
    }


def _summarize(result, goal, *, verbose=False):
    solve_times = result["solve_times"]
    avg_ms = float(np.mean(solve_times) * 1e3) if solve_times.size else 0.0
    parts = [
        f"{STYLE[result['name']]['label']} steps={len(result['traj']) - 1}",
        f"final_dist={np.linalg.norm(result['traj'][-1, :2] - goal):.4f}",
        f"min_clearance={np.min(result['clearance']):.4f}",
        f"mean_control_variation={np.mean(result['variation']):.3f}",
        f"max_slack={np.max(result['slack']):.2e}",
        f"avg_solve_ms={avg_ms:.2f}",
    ]
    if verbose:
        parts.append(f"solver_failures={result['solver_failures']}")
    print(" ".join(parts))


def _plot_xy(results, *, start, goal, agent_radius, path):
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    plot_start(ax, start)
    plot_goal(ax, goal)
    base_obstacle = results[0]["obstacle"]
    plot_obstacle(
        ax,
        base_obstacle.center,
        base_obstacle.r1,
        base_obstacle.r2,
        p=base_obstacle.p,
        scale=base_obstacle.scale,
        angle=base_obstacle.angle,
        color="0.3",
        fill=True,
    )
    for result in results:
        style = STYLE[result["name"]]
        traj = result["traj"]
        ax.plot(traj[:, 0], traj[:, 1], color=style["color"], linewidth=2.2, label=style["label"])
        ax.scatter(traj[-1, 0], traj[-1, 1], color=style["color"], s=34, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _plot_time_series(results, *, goal, path):
    max_time = max((len(result["traj"]) - 1) * result["dt"] for result in results)
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.6), sharex=True)
    axes = axes.ravel()
    for result in results:
        style = STYLE[result["name"]]
        label = style["label"]
        color = style["color"]
        traj = result["traj"]
        controls = result["controls"]
        t = result["dt"] * np.arange(len(traj))
        distance = np.linalg.norm(traj[:, :2] - goal[None, :], axis=1)
        axes[0].plot(t, distance, color=color, label=label)
        axes[1].plot(t, controls[:, 0], color=color, label=label)
        axes[2].plot(t, controls[:, 1], color=color, label=label)
        axes[3].plot(t, result["variation"], color=color, label=label)
        axes[4].plot(t, result["density"], color=color, label=label)
        axes[5].plot(t, result["clearance"], color=color, label=label)
    labels = ["distance [m]", "v [m/s]", "omega [rad/s]", "||du/dt||", "rho(z)", "clearance [m]"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(0.0, max_time)
    axes[5].axhline(0.0, color="0.2", linewidth=1.0, linestyle="--")
    axes[4].set_xlabel("time [s]")
    axes[5].set_xlabel("time [s]")
    axes[0].legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _save_dashboard_animation(results, *, start, goal, agent_radius, path, stride, fps, mp4_crf=28, mp4_preset="slow"):
    max_time = max((len(result["traj"]) - 1) * result["dt"] for result in results)
    frame_times = np.arange(0.0, max_time + 1e-9, min(result["dt"] for result in results) * stride)
    if frame_times[-1] < max_time:
        frame_times = np.append(frame_times, max_time)

    sampled = []
    for result in results:
        time_grid = result["dt"] * np.arange(len(result["traj"]))
        traj = result["traj"]
        controls = result["controls"]
        traj_frame = np.column_stack(
            [
                np.interp(frame_times, time_grid, traj[:, 0]),
                np.interp(frame_times, time_grid, traj[:, 1]),
                np.interp(frame_times, time_grid, traj[:, 2]),
            ]
        )
        controls_frame = np.column_stack(
            [
                np.interp(frame_times, time_grid, controls[:, 0]),
                np.interp(frame_times, time_grid, controls[:, 1]),
            ]
        )
        sampled.append(
            {
                **result,
                "time_grid": time_grid,
                "traj_actual": traj,
                "traj_frame": traj_frame,
                "controls_frame": controls_frame,
                "distance": np.linalg.norm(traj_frame[:, :2] - goal[None, :], axis=1),
                "v": controls_frame[:, 0],
                "omega": controls_frame[:, 1],
                "variation_frame": np.interp(frame_times, time_grid, result["variation"]),
                "rho": np.interp(frame_times, time_grid, result["density"]),
                "clearance_frame": np.interp(frame_times, time_grid, result["clearance"]),
            }
        )

    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.0, 1.0])
    map_ax = fig.add_subplot(gs[:, 0])
    trace_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]

    plot_start(map_ax, start)
    plot_goal(map_ax, goal)
    base_obstacle = results[0]["obstacle"]
    plot_obstacle(
        map_ax,
        base_obstacle.center,
        base_obstacle.r1,
        base_obstacle.r2,
        p=base_obstacle.p,
        scale=base_obstacle.scale,
        angle=base_obstacle.angle,
        color="0.3",
        fill=True,
    )
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_xlabel("x [m]")
    map_ax.set_ylabel("y [m]")
    map_ax.grid(True, linestyle="--", alpha=0.35)

    map_lines = []
    agents = []
    for result in sampled:
        style = STYLE[result["name"]]
        line, = map_ax.plot([], [], color=style["color"], linewidth=2.1, label=style["label"])
        agent = patches.Polygon(
            _triangle_points(result["traj_frame"][0, :2], result["traj_frame"][0, 2], agent_radius),
            closed=True,
            facecolor=style["color"],
            edgecolor="k",
            linewidth=1.0,
            zorder=5,
        )
        map_ax.add_patch(agent)
        map_lines.append(line)
        agents.append(agent)
    map_ax.legend(loc="lower right", framealpha=0.92)

    series_specs = [
        ("distance", "distance [m]"),
        ("v", "v [m/s]"),
        ("omega", "omega [rad/s]"),
        ("variation_frame", "||du/dt||"),
        ("rho", "rho(z)"),
        ("clearance_frame", "clearance [m]"),
    ]
    dash_lines = []
    for ax, (key, ylabel) in zip(trace_axes, series_specs):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(frame_times[0], frame_times[-1])
        ax.set_ylim(_axis_limits(*(result[key] for result in sampled)))
        if key == "clearance_frame":
            ax.axhline(0.0, color="0.2", linewidth=0.9, linestyle="--")
        lines = []
        for result in sampled:
            line, = ax.plot([], [], color=STYLE[result["name"]]["color"], linewidth=1.6)
            lines.append(line)
        dash_lines.append((key, lines))
    trace_axes[-2].set_xlabel("time [s]")
    trace_axes[-1].set_xlabel("time [s]")
    fig.tight_layout()

    def init():
        for line in map_lines:
            line.set_data([], [])
        for _, lines in dash_lines:
            for line in lines:
                line.set_data([], [])
        return tuple(map_lines + agents + [line for _, lines in dash_lines for line in lines])

    def update(i):
        artists = []
        for idx, result in enumerate(sampled):
            traj = result["traj_frame"]
            actual_mask = result["time_grid"] <= frame_times[i]
            actual_xy = result["traj_actual"][actual_mask, :2]
            if actual_xy.size:
                line_xy = np.vstack([actual_xy, traj[i, :2]])
            else:
                line_xy = traj[i : i + 1, :2]
            map_lines[idx].set_data(line_xy[:, 0], line_xy[:, 1])
            agents[idx].set_xy(_triangle_points(traj[i, :2], traj[i, 2], agent_radius))
            artists.extend([map_lines[idx], agents[idx]])
        current_t = frame_times[: i + 1]
        for key, lines in dash_lines:
            for line, result in zip(lines, sampled):
                line.set_data(current_t, result[key][: i + 1])
                artists.append(line)
        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=range(len(frame_times)),
        interval=20,
        blit=True,
        repeat=False,
    )
    save_animation_file(ani, path, fps, mp4_crf=mp4_crf, mp4_preset=mp4_preset)
    return fig, ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=int(CONFIG["simulation"]["steps"]),
        help="Maximum simulation steps for filter and MPC.",
    )
    parser.add_argument(
        "--feedback-steps",
        type=int,
        default=int(CONFIG["simulation"].get("feedback_steps", CONFIG["simulation"]["steps"])),
        help="Maximum simulation steps for density feedback.",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Density MPC prediction horizon.")
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="Optimizer backend. Density MPC supports all choices; filters currently use scipy_slsqp.",
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
        help="Run each controller for exactly --steps iterations instead of stopping once stable near the goal.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["feedback", "filter", "mpc"],
        default=["feedback", "filter", "mpc"],
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

    results = []
    if "feedback" in args.controllers:
        results.append(
            _simulate_feedback(
                cfg=cfg,
                start=start,
                goal=goal,
                obstacle=obstacle,
                inflated_obstacle=inflated_obstacle,
                steps=args.feedback_steps,
                early_stop=not args.fixed_steps,
            )
        )
    if "filter" in args.controllers:
        results.append(
            _simulate_filter(
                cfg=cfg,
                start=start,
                goal=goal,
                obstacle=obstacle,
                inflated_obstacle=inflated_obstacle,
                steps=args.steps,
                solver=args.solver,
                early_stop=not args.fixed_steps,
                verbose=args.verbose,
            )
        )
    if "mpc" in args.controllers:
        results.append(
            _simulate_mpc(
                cfg=cfg,
                start=start,
                goal=goal,
                obstacle=obstacle,
                inflated_obstacle=inflated_obstacle,
                steps=args.steps,
                horizon=args.horizon,
                solver=args.solver,
                early_stop=not args.fixed_steps,
                verbose=args.verbose,
            )
        )
    for result in results:
        _summarize(result, goal, verbose=args.verbose)

    output_dir = Path(__file__).resolve().parent / "comparison_results"
    xy_path = output_dir / "unicycle_static_density_controllers_xy.png"
    ts_path = output_dir / "unicycle_static_density_controllers_timeseries.png"
    gif_path = output_dir / "unicycle_static_density_controllers.gif"
    mp4_path = gif_path.with_suffix(".mp4")

    figures = [
        _plot_xy(results, start=start, goal=goal, agent_radius=agent_radius, path=xy_path),
        _plot_time_series(results, goal=goal, path=ts_path),
    ]
    animations_to_show = []
    if not args.no_gif:
        fig, ani = _save_dashboard_animation(
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
        fig, ani = _save_dashboard_animation(
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
