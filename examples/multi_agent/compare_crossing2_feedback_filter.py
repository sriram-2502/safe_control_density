import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np


MULTI_AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MULTI_AGENT_ROOT.parents[1]
EXAMPLE_ROOT = MULTI_AGENT_ROOT.parents[0]
sys.path[:0] = [
    str(MULTI_AGENT_ROOT),
    str(REPO_ROOT),
    str(EXAMPLE_ROOT / "unicycle"),
    str(MULTI_AGENT_ROOT / "density_feedback"),
    str(MULTI_AGENT_ROOT / "density_filter"),
]

from density_feedback import _config
from density_feedback import collision_cone as fb_collision_cone
from density_feedback import reactive as fb_reactive
from density_feedback import velocity_obstacle as fb_velocity_obstacle
from density_filter import reactive as filter_reactive
from density_filter import collision_cone as filter_collision_cone
from density_filter import velocity_obstacle as filter_velocity_obstacle

from _plotting import AGENT_COLORS, save_animation_file
from density_utils.controllers import density_feedback_control, single_integrator_nominal_control, solve_discrete_density_filter
from density_utils.density import Obstacle
from density_utils.dynamics import unicycle_step


@dataclass
class RunResult:
    label: str
    family: str
    method: str
    traj: np.ndarray
    controls: np.ndarray
    max_dist: np.ndarray
    min_clearance: np.ndarray
    avg_ms: float
    status: str
    steps: int


METHOD_COLORS = {
    "Vanilla bump": "tab:blue",
    "Collision cone": "tab:green",
    "Velocity obstacle": "tab:red",
}

FAMILY_STYLES = {
    "Feedback": "-",
    "Filter": "--",
}


def _base_args():
    return argparse.Namespace(
        scenario="crossing2",
        agent_radius=0.12,
        start_offset=(0.0, 0.0),
        reactive_crossing6_offset=(0.039, 0.0),
        alpha=0.2,
        ctrl_multiplier=10.0,
        rad_from_goal=0.05,
        v_max=1.5,
        omega_max=2.5,
        k_heading=3.0,
        dt=0.01,
        filter_dt=0.1,
        filter_neighbors=2,
        filter_safety_margin=0.0,
        cone_density_margin=0.45,
        q_lqr=4.0,
        r_lqr=1.0,
        density_weight=1.0,
        goal_weight=0.5,
        nominal_smoothing=0.25,
        slack_weight=1e4,
        control_weight=1.0,
        solver="auto",
    )


def _finish_run(label, family, method, traj, controls, goals, agent_r1, control_time, steps):
    traj = np.asarray(traj, dtype=float)
    controls = np.asarray(controls, dtype=float)
    dists = np.linalg.norm(traj[:, :, :2] - goals[None, :, :], axis=2)
    clearances = np.asarray([_config.min_pair_clearance(states, agent_r1) for states in traj], dtype=float)
    final_dists = dists[-1]
    status = "success" if np.all(final_dists < _base_args().rad_from_goal) else "timeout"
    avg_ms = float(np.mean(control_time / max(steps, 1) * 1e3))
    return RunResult(
        label=label,
        family=family,
        method=method,
        traj=traj,
        controls=controls,
        max_dist=np.max(dists, axis=1),
        min_clearance=clearances,
        avg_ms=avg_ms,
        status=status,
        steps=steps,
    )


def _feedback_to_unicycle(state, planar_u, tilde_prev, args):
    v = min(float(np.linalg.norm(planar_u)), args.v_max)
    desired_heading = float(np.arctan2(planar_u[1], planar_u[0]))
    desired_heading_rate = _config.wrap_angle(desired_heading - tilde_prev) / args.dt
    omega = desired_heading_rate - args.k_heading * _config.wrap_angle(state[2] - desired_heading)
    return np.array([v, float(np.clip(omega, -args.omega_max, args.omega_max))]), desired_heading


def _collision_cone_feedback_to_unicycle(state, planar_u, previous_control, args):
    v = min(float(np.linalg.norm(planar_u)), args.v_max)
    desired_heading = float(np.arctan2(planar_u[1], planar_u[0]))
    omega = args.k_heading * _config.wrap_angle(desired_heading - state[2])
    omega = float(np.clip(omega, -args.omega_max, args.omega_max))
    v = float(
        np.clip(
            v,
            previous_control[0] - fb_collision_cone.SPEED_RATE_LIMIT * args.dt,
            previous_control[0] + fb_collision_cone.SPEED_RATE_LIMIT * args.dt,
        )
    )
    omega = float(
        np.clip(
            omega,
            previous_control[1] - fb_collision_cone.OMEGA_RATE_LIMIT * args.dt,
            previous_control[1] + fb_collision_cone.OMEGA_RATE_LIMIT * args.dt,
        )
    )
    return np.array([v, omega], dtype=float), desired_heading


def _simulate_feedback(method, starts, goals, headings, agent_r1, agent_r2, args, max_steps):
    states = np.hstack([starts, headings[:, None]])
    tilde_prev = headings.copy()
    prev_planar_vel = np.column_stack([np.cos(headings), np.sin(headings)]) * 0.2
    prev_planar_cmd = prev_planar_vel.copy()
    previous_control = np.zeros((states.shape[0], 2), dtype=float)
    num_agents = states.shape[0]
    traj = [states.copy()]
    controls = []
    control_time = np.zeros(num_agents, dtype=float)
    q_lqr = 1.0
    r_lqr = 1.0
    saturation = 4.0

    for step in range(max_steps):
        if np.all(np.linalg.norm(states[:, :2] - goals, axis=1) < args.rad_from_goal):
            break
        new_states = states.copy()
        new_planar_vel = np.zeros_like(prev_planar_vel)
        step_controls = np.zeros((num_agents, 2), dtype=float)
        for agent_idx in range(num_agents):
            other_indices = np.delete(np.arange(num_agents), agent_idx)
            start_time = time.perf_counter()
            if method == "Vanilla bump":
                planar_u = density_feedback_control(
                    states[agent_idx, :2],
                    goals[agent_idx],
                    args.alpha,
                    fb_reactive._agent_obstacles(states, agent_r1, agent_r2, agent_idx),
                    ctrl_multiplier=args.ctrl_multiplier,
                    rad_from_goal=args.rad_from_goal,
                    q_lqr=q_lqr,
                    r_lqr=r_lqr,
                    dt=args.dt,
                    saturation=saturation,
                )
            elif method == "Collision cone":
                planar_u = fb_collision_cone._collision_cone_density_control(
                    ego_pos=states[agent_idx, :2],
                    goal=goals[agent_idx],
                    ego_vel=prev_planar_vel[agent_idx],
                    other_positions=states[other_indices, :2],
                    other_velocities=prev_planar_vel[other_indices],
                    ego_r1=agent_r1[agent_idx],
                    other_r1=agent_r1[other_indices],
                    other_r2=agent_r2[other_indices],
                    alpha=args.alpha,
                    cone_margin=args.cone_density_margin,
                    ctrl_multiplier=args.ctrl_multiplier,
                    rad_from_goal=args.rad_from_goal,
                    q_lqr=q_lqr,
                    r_lqr=r_lqr,
                    dt=args.dt,
                    saturation=saturation,
                )
                planar_u = (
                    (1.0 - fb_collision_cone.PLANAR_COMMAND_SMOOTHING) * planar_u
                    + fb_collision_cone.PLANAR_COMMAND_SMOOTHING * prev_planar_cmd[agent_idx]
                )
                prev_planar_cmd[agent_idx] = planar_u
            else:
                extra_radius, close_margin, projection_passes = fb_velocity_obstacle.VO_SCENARIO_TUNING.get(
                    args.scenario,
                    (
                        fb_velocity_obstacle.VO_DEFAULT_EXTRA_RADIUS,
                        fb_velocity_obstacle.VO_DEFAULT_CLOSE_MARGIN,
                        fb_velocity_obstacle.VO_DEFAULT_PROJECTION_PASSES,
                    ),
                )
                planar_u = fb_velocity_obstacle._velocity_obstacle_density_control(
                    ego_pos=states[agent_idx, :2],
                    goal=goals[agent_idx],
                    agent_idx=agent_idx,
                    other_positions=states[other_indices, :2],
                    other_velocities=prev_planar_vel[other_indices],
                    ego_r1=agent_r1[agent_idx],
                    other_r1=agent_r1[other_indices],
                    other_r2=agent_r2[other_indices],
                    alpha=args.alpha,
                    vo_margin=args.cone_density_margin,
                    ctrl_multiplier=args.ctrl_multiplier,
                    rad_from_goal=args.rad_from_goal,
                    q_lqr=q_lqr,
                    r_lqr=r_lqr,
                    dt=args.dt,
                    saturation=saturation,
                    extra_radius=extra_radius,
                    close_margin=close_margin,
                    projection_passes=projection_passes,
                )
            control_time[agent_idx] += time.perf_counter() - start_time
            if method == "Collision cone":
                control, tilde_prev[agent_idx] = _collision_cone_feedback_to_unicycle(
                    states[agent_idx], planar_u, previous_control[agent_idx], args
                )
            else:
                control, tilde_prev[agent_idx] = _feedback_to_unicycle(
                    states[agent_idx], planar_u, tilde_prev[agent_idx], args
                )
            previous_control[agent_idx] = control
            step_controls[agent_idx] = control
            new_states[agent_idx] = unicycle_step(states[agent_idx], control[0], control[1], args.dt)
            new_states[agent_idx, 2] = _config.wrap_angle(new_states[agent_idx, 2])
            new_planar_vel[agent_idx] = control[0] * np.array(
                [np.cos(new_states[agent_idx, 2]), np.sin(new_states[agent_idx, 2])]
            )
        states = new_states
        prev_planar_vel = new_planar_vel
        controls.append(step_controls)
        traj.append(states.copy())

    return _finish_run(f"{method} feedback", "Feedback", method, traj, controls, goals, agent_r1, control_time, len(controls))


def _filter_obstacles(states, agent_r1, agent_r2, agent_idx, other_indices, safety_margin):
    return [
        Obstacle(
            center=states[other_idx, :2],
            r1=agent_r1[agent_idx] + agent_r1[other_idx] + safety_margin,
            r2=agent_r1[agent_idx] + agent_r2[other_idx] + safety_margin,
            p=2.0,
        )
        for other_idx in other_indices
    ]


def _simulate_filter(method, starts, goals, headings, agent_r1, agent_r2, args, max_steps):
    states = np.hstack([starts, headings[:, None]])
    goal_states = np.hstack([goals, headings[:, None]])
    num_agents = states.shape[0]
    previous_control = np.zeros((num_agents, 2), dtype=float)
    previous_planar = np.zeros((num_agents, 2), dtype=float)
    u_min = np.array([0.0, -args.omega_max], dtype=float)
    u_max = np.array([args.v_max, args.omega_max], dtype=float)
    traj = [states.copy()]
    controls = []
    control_time = np.zeros(num_agents, dtype=float)

    nominal_gain = 0.6 if method == "Velocity obstacle" else 0.8
    nominal_smoothing = 0.60 if method == "Velocity obstacle" else 0.55 if method == "Collision cone" else args.nominal_smoothing
    filter_safety_margin = 0.04 if method == "Velocity obstacle" else 0.08 if method == "Collision cone" else args.filter_safety_margin
    control_weight = 2.5 if method == "Velocity obstacle" else 2.0 if method == "Collision cone" else args.control_weight

    for step in range(max_steps):
        if np.all(np.linalg.norm(states[:, :2] - goals, axis=1) < args.rad_from_goal):
            break
        new_states = states.copy()
        planar_snapshot = previous_planar.copy()
        next_planar = previous_planar.copy()
        step_controls = np.zeros((num_agents, 2), dtype=float)
        for agent_idx in range(num_agents):
            other_indices = np.delete(np.arange(num_agents), agent_idx)
            obstacles = _filter_obstacles(states, agent_r1, agent_r2, agent_idx, other_indices, filter_safety_margin)
            other_velocities = planar_snapshot[other_indices].copy()
            if method == "Vanilla bump":
                planar_nominal = single_integrator_nominal_control(
                    states[agent_idx, :2],
                    goals[agent_idx],
                    args.alpha,
                    obstacles,
                    mode="density_blend",
                    ctrl_multiplier=args.ctrl_multiplier,
                    rad_from_goal=args.rad_from_goal,
                    q_lqr=args.q_lqr,
                    r_lqr=args.r_lqr,
                    dt=args.filter_dt,
                    u_min=[-args.v_max, -args.v_max],
                    u_max=[args.v_max, args.v_max],
                    density_weight=args.density_weight,
                    goal_weight=args.goal_weight,
                )
                nominal = filter_reactive._unicycle_nominal_from_planar_ref(
                    states[agent_idx], planar_nominal, args.v_max, args.omega_max, args.k_heading
                )
                nominal = filter_reactive._smooth_nominal(
                    nominal, previous_control[agent_idx], args.nominal_smoothing, args.v_max, args.omega_max
                )
                density_fn = filter_reactive._pose_density
                slack_weight = args.slack_weight
            elif method == "Collision cone":
                planar_nominal = filter_collision_cone._collision_cone_planar_nominal(
                    states[agent_idx, :2],
                    goals[agent_idx],
                    planar_snapshot[agent_idx],
                    obstacles,
                    other_velocities,
                    args.alpha,
                    args.cone_density_margin,
                    args.ctrl_multiplier,
                    args.rad_from_goal,
                    args.v_max,
                )
                nominal = filter_collision_cone._unicycle_nominal_from_planar_ref(
                    states[agent_idx], planar_nominal, args.v_max, args.omega_max, args.k_heading
                )
                nominal = filter_collision_cone._smooth_nominal(
                    nominal, previous_control[agent_idx], nominal_smoothing, args.v_max, args.omega_max
                )
                density_fn = lambda state_eval, goal_eval, alpha_eval, obstacles_eval, ov=other_velocities, nom=nominal: filter_collision_cone._collision_cone_density(
                    state_eval,
                    goal_eval,
                    alpha_eval,
                    obstacles_eval,
                    other_velocities=ov,
                    ego_speed=max(float(nom[0]), 0.2),
                    cone_margin=args.cone_density_margin,
                )
                slack_weight = args.slack_weight
            else:
                goal_planar = filter_velocity_obstacle._goal_planar_control(
                    states[agent_idx, :2], goals[agent_idx], nominal_gain, args.v_max, args.rad_from_goal
                )
                planar_nominal = filter_velocity_obstacle._vo_planar_nominal(
                    states[agent_idx, :2],
                    goals[agent_idx],
                    goal_planar,
                    obstacles,
                    other_velocities,
                    args.alpha,
                    args.cone_density_margin,
                    args.ctrl_multiplier,
                    args.rad_from_goal,
                    args.v_max,
                )
                nominal = filter_velocity_obstacle._unicycle_nominal_from_planar_ref(
                    states[agent_idx], planar_nominal, args.v_max, args.omega_max, args.k_heading
                )
                nominal = filter_velocity_obstacle._smooth_nominal(
                    nominal, previous_control[agent_idx], nominal_smoothing, args.v_max, args.omega_max
                )
                density_fn = lambda state_eval, goal_eval, alpha_eval, obstacles_eval, ov=other_velocities, nom=nominal: filter_velocity_obstacle._vo_density(
                    state_eval,
                    goal_eval,
                    alpha_eval,
                    obstacles_eval,
                    other_velocities=ov,
                    ego_speed=max(float(nom[0]), 0.2),
                    vo_margin=args.cone_density_margin,
                )
                slack_weight = max(args.slack_weight, 1e4)

            start_time = time.perf_counter()
            result = solve_discrete_density_filter(
                states[agent_idx],
                goal_states[agent_idx],
                args.alpha,
                obstacles,
                u_nom=nominal,
                dt=args.filter_dt,
                next_state_fn=filter_reactive._unicycle_filter_step,
                u_min=u_min,
                u_max=u_max,
                divergence=0.0,
                slack_weight=slack_weight,
                control_weight=control_weight,
                density_fn=density_fn,
                solver=args.solver,
                return_info=True,
            )
            control_time[agent_idx] += time.perf_counter() - start_time
            previous_control[agent_idx] = result.u
            step_controls[agent_idx] = result.u
            new_states[agent_idx] = unicycle_step(states[agent_idx], result.u[0], result.u[1], args.dt)
            new_states[agent_idx, 2] = _config.wrap_angle(new_states[agent_idx, 2])
            next_planar[agent_idx] = result.u[0] * np.array(
                [np.cos(new_states[agent_idx, 2]), np.sin(new_states[agent_idx, 2])]
            )

        states = new_states
        previous_planar = next_planar
        controls.append(step_controls)
        traj.append(states.copy())

    return _finish_run(f"{method} filter", "Filter", method, traj, controls, goals, agent_r1, control_time, len(controls))


def _pad_result(result, total_frames):
    repeat = total_frames - result.traj.shape[0]
    if repeat <= 0:
        return result
    traj = np.concatenate([result.traj, np.repeat(result.traj[-1:], repeat, axis=0)], axis=0)
    max_dist = np.concatenate([result.max_dist, np.repeat(result.max_dist[-1], repeat)])
    min_clearance = np.concatenate([result.min_clearance, np.repeat(result.min_clearance[-1], repeat)])
    if result.controls.size:
        controls = np.concatenate([result.controls, np.repeat(result.controls[-1:], repeat, axis=0)], axis=0)
    else:
        controls = result.controls
    return RunResult(
        label=result.label,
        family=result.family,
        method=result.method,
        traj=traj,
        controls=controls,
        max_dist=max_dist,
        min_clearance=min_clearance,
        avg_ms=result.avg_ms,
        status=result.status,
        steps=result.steps,
    )


def _triangle(center, heading, size):
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    return np.stack(
        [
            center + size * 1.3 * forward,
            center - size * 0.9 * forward + size * 0.6 * right,
            center - size * 0.9 * forward - size * 0.6 * right,
        ],
        axis=0,
    )


def _use_family_styles(results):
    families = {result.family for result in results}
    methods = {result.method for result in results}
    return len(families) > 1 and len(methods) == 1


def _line_style(result, use_family_styles):
    return FAMILY_STYLES[result.family] if use_family_styles else "-"


def _method_short_name(method):
    if method == "Velocity obstacle":
        return "VO"
    if method == "Collision cone":
        return "CC"
    return method


def _save_dashboard(results, starts, goals, agent_r1, path, stride, fps, title, hold_seconds):
    import matplotlib.pyplot as plt
    from matplotlib import animation, gridspec, patches

    hold_frames = max(0, int(round(float(hold_seconds) * float(fps))))
    frame_sequence = []
    for result_idx, result in enumerate(results):
        local_frames = np.arange(0, result.traj.shape[0], max(1, stride), dtype=int)
        if local_frames[-1] != result.traj.shape[0] - 1:
            local_frames = np.append(local_frames, result.traj.shape[0] - 1)
        frame_sequence.extend([(result_idx, 0)] * hold_frames)
        frame_sequence.extend((result_idx, int(frame)) for frame in local_frames)
        frame_sequence.extend([(result_idx, int(local_frames[-1]))] * hold_frames)

    fig = plt.figure(figsize=(12.0, 7.4), dpi=120)
    gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1.35, 1.0, 1.0], wspace=0.35, hspace=0.42)
    map_ax = fig.add_subplot(gs[:, 0])
    speed_ax = fig.add_subplot(gs[0, 1:])
    omega_ax = fig.add_subplot(gs[1, 1:])
    clear_ax = fig.add_subplot(gs[2, 1:])

    map_ax.set_title(title)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.grid(True, linestyle="--", alpha=0.35)
    all_points = np.concatenate([starts, goals, *[result.traj[:, :, :2].reshape(-1, 2) for result in results]], axis=0)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    width = float(np.max(all_points.max(axis=0) - all_points.min(axis=0)))
    half = 0.5 * max(width, 5.0) + 0.45
    map_ax.set_xlim(center[0] - half, center[0] + half)
    map_ax.set_ylim(center[1] - half, center[1] + half)
    map_ax.set_xlabel("x")
    map_ax.set_ylabel("y")

    for idx, start in enumerate(starts):
        map_ax.scatter(start[0], start[1], marker="o", s=55, color=AGENT_COLORS[idx], edgecolor="k", zorder=4)
    for idx, goal in enumerate(goals):
        map_ax.scatter(goal[0], goal[1], marker="*", s=130, color=AGENT_COLORS[idx], edgecolor="k", zorder=4)

    trail_lines = {}
    agent_patches = {}
    use_family_styles = _use_family_styles(results)
    for result in results:
        color = METHOD_COLORS[result.method]
        style = _line_style(result, use_family_styles)
        for agent_idx in range(starts.shape[0]):
            line, = map_ax.plot([], [], linestyle=style, linewidth=2.0, color=color, alpha=0.85)
            trail_lines[(result.label, agent_idx)] = line
            patch = patches.Polygon(
                _triangle(result.traj[0, agent_idx, :2], result.traj[0, agent_idx, 2], agent_r1[agent_idx]),
                closed=True,
                facecolor=color,
                edgecolor="k",
                linewidth=0.8,
                alpha=0.82 if result.family == "Feedback" else 0.45,
                zorder=5,
            )
            map_ax.add_patch(patch)
            agent_patches[(result.label, agent_idx)] = patch

    if use_family_styles:
        legend_handles = []
        for result in results:
            legend_handles.append(
                map_ax.plot(
                    [],
                    [],
                    color=METHOD_COLORS[result.method],
                    linestyle=FAMILY_STYLES[result.family],
                    linewidth=2.5,
                    label=f"{result.family} ({_method_short_name(result.method)})",
                )[0]
            )
    else:
        legend_handles = []
        methods_in_plot = []
        for result in results:
            if result.method not in methods_in_plot:
                methods_in_plot.append(result.method)
        for method in methods_in_plot:
            legend_handles.append(map_ax.plot([], [], color=METHOD_COLORS[method], linewidth=2.5, label=method)[0])
    map_ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    metric_axes = (speed_ax, omega_ax, clear_ax)
    max_time = max((result.traj.shape[0] - 1) * 0.01 for result in results)
    for ax in metric_axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(0.0, max_time)
    speed_ax.set_ylabel("v")
    omega_ax.set_ylabel("omega")
    clear_ax.set_ylabel("min clearance")
    clear_ax.set_xlabel("time [s]")
    clear_ax.axhline(0.0, color="k", linestyle=":", linewidth=1.0)

    metric_lines = {}
    for result in results:
        color = METHOD_COLORS[result.method]
        style = _line_style(result, use_family_styles)
        t = np.arange(result.traj.shape[0]) * 0.01
        speed = np.zeros((result.traj.shape[0], starts.shape[0]), dtype=float)
        omega = np.zeros((result.traj.shape[0], starts.shape[0]), dtype=float)
        if result.controls.size:
            speed[1 : result.controls.shape[0] + 1, :] = result.controls[:, :, 0]
            omega[1 : result.controls.shape[0] + 1, :] = result.controls[:, :, 1]
        for agent_idx in range(starts.shape[0]):
            alpha = 0.95 if agent_idx == 0 else 0.60
            metric_lines[(result.label, "speed", agent_idx)] = speed_ax.plot(
                [], [], color=color, linestyle=style, linewidth=1.8, alpha=alpha
            )[0]
            metric_lines[(result.label, "omega", agent_idx)] = omega_ax.plot(
                [], [], color=color, linestyle=style, linewidth=1.8, alpha=alpha
            )[0]
        metric_lines[(result.label, "clear")] = clear_ax.plot([], [], color=color, linestyle=style, linewidth=1.8)[0]
        metric_lines[(result.label, "data")] = (t, speed, omega, result.min_clearance)

    speed_ax.set_ylim(0.0, 1.65)
    max_abs_omega = max(float(np.max(np.abs(result.controls[:, :, 1]))) if result.controls.size else 0.0 for result in results)
    omega_ax.set_ylim(-max(2.6, max_abs_omega * 1.05), max(2.6, max_abs_omega * 1.05))
    clear_ax.set_ylim(
        min(0.0, min(float(np.min(result.min_clearance)) for result in results)) - 0.05,
        max(float(np.max(result.min_clearance)) for result in results) + 0.1,
    )

    status_text = fig.text(0.02, 0.02, "", fontsize=9, family="monospace")

    def update(frame_event):
        active_idx, frame = frame_event
        artists = []
        for result_idx, result in enumerate(results):
            local_frame = min(frame, result.traj.shape[0] - 1)
            for agent_idx in range(starts.shape[0]):
                is_active = result_idx == active_idx
                agent_patches[(result.label, agent_idx)].set_visible(is_active)
                if is_active:
                    trail_lines[(result.label, agent_idx)].set_visible(True)
                    trail_lines[(result.label, agent_idx)].set_data(
                        result.traj[: local_frame + 1, agent_idx, 0],
                        result.traj[: local_frame + 1, agent_idx, 1],
                    )
                    agent_patches[(result.label, agent_idx)].set_xy(
                        _triangle(result.traj[local_frame, agent_idx, :2], result.traj[local_frame, agent_idx, 2], agent_r1[agent_idx])
                    )
                elif result_idx < active_idx:
                    trail_lines[(result.label, agent_idx)].set_visible(True)
                    trail_lines[(result.label, agent_idx)].set_data(
                        result.traj[:, agent_idx, 0],
                        result.traj[:, agent_idx, 1],
                    )
                else:
                    trail_lines[(result.label, agent_idx)].set_visible(False)
                    trail_lines[(result.label, agent_idx)].set_data([], [])
                artists.extend([trail_lines[(result.label, agent_idx)], agent_patches[(result.label, agent_idx)]])
            if result_idx < active_idx:
                metric_frame = result.traj.shape[0] - 1
            elif result_idx == active_idx:
                metric_frame = local_frame
            else:
                metric_frame = -1
            t, speed, omega, min_clearance = metric_lines[(result.label, "data")]
            if metric_frame >= 0:
                for agent_idx in range(starts.shape[0]):
                    metric_lines[(result.label, "speed", agent_idx)].set_data(
                        t[: metric_frame + 1], speed[: metric_frame + 1, agent_idx]
                    )
                    metric_lines[(result.label, "omega", agent_idx)].set_data(
                        t[: metric_frame + 1], omega[: metric_frame + 1, agent_idx]
                    )
                metric_lines[(result.label, "clear")].set_data(t[: metric_frame + 1], min_clearance[: metric_frame + 1])
            else:
                for agent_idx in range(starts.shape[0]):
                    metric_lines[(result.label, "speed", agent_idx)].set_data([], [])
                    metric_lines[(result.label, "omega", agent_idx)].set_data([], [])
                metric_lines[(result.label, "clear")].set_data([], [])
            for agent_idx in range(starts.shape[0]):
                artists.extend(
                    [
                        metric_lines[(result.label, "speed", agent_idx)],
                        metric_lines[(result.label, "omega", agent_idx)],
                    ]
                )
            artists.append(metric_lines[(result.label, "clear")])
        active = results[active_idx]
        status_text.set_text(
            f"Active: {active.label} | status={active.status} | avg={active.avg_ms:.2f} ms | "
            f"run {active_idx + 1}/{len(results)}"
        )
        artists.append(status_text)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_sequence,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.94)
    save_animation_file(ani, path, fps)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", default=True, help="Save the dashboard GIF.")
    parser.add_argument("--no-gif", action="store_true", help="Skip saving the dashboard GIF.")
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hold-seconds", type=float, default=1.5, help="Initial pause before trajectories move.")
    args_cli = parser.parse_args(argv)

    args = _base_args()
    starts, goals, headings, agent_r1, agent_r2 = _config.make_scenario(args, "reactive_density_feedback")

    methods = ("Vanilla bump", "Collision cone", "Velocity obstacle")
    results = []
    for method in methods:
        results.append(_simulate_feedback(method, starts, goals, headings, agent_r1, agent_r2, args, args_cli.steps))
    for method in methods:
        results.append(_simulate_filter(method, starts, goals, headings, agent_r1, agent_r2, args, args_cli.steps))

    for result in results:
        print(
            f"{result.label}: status={result.status} steps={result.steps} "
            f"max_dist={result.max_dist[-1]:.3f} min_clearance={np.min(result.min_clearance):.3f} "
            f"avg_iteration_mean={result.avg_ms:.3f} [ms]"
        )

    if not args_cli.no_gif:
        output_dir = MULTI_AGENT_ROOT / "animations" / "comparison"
        comparisons = [
            (
                "crossing2_density_feedback_methods.gif",
                "Crossing 2: Density Feedback Methods",
                [result for result in results if result.family == "Feedback"],
            ),
            (
                "crossing2_density_filter_methods.gif",
                "Crossing 2: Density Filter Methods",
                [result for result in results if result.family == "Filter"],
            ),
            (
                "crossing2_velocity_obstacle_feedback_vs_filter.gif",
                "Crossing 2: Velocity Obstacle Feedback vs Filter",
                [result for result in results if result.method == "Velocity obstacle"],
            ),
            (
                "crossing2_collision_cone_feedback_vs_filter.gif",
                "Crossing 2: Collision Cone Feedback vs Filter",
                [result for result in results if result.method == "Collision cone"],
            ),
        ]
        for filename, title, subset in comparisons:
            _save_dashboard(
                subset,
                starts,
                goals,
                agent_r1,
                output_dir / filename,
                args_cli.stride,
                args_cli.fps,
                title,
                args_cli.hold_seconds,
            )


if __name__ == "__main__":
    main()
