import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


DYNAMIC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(DYNAMIC_ROOT), str(REPO_ROOT)]

from _config import (
    add_common_arguments,
    example_root,
    finalize_args,
    initialize_dynamic_obstacles,
    make_scenario,
    min_obstacle_clearance,
    step_dynamic_obstacles,
    wrap_angle,
)
from _plotting import animation_save_paths, plot_dynamic_obstacle_results, wants_animation_output

from density_utils.controllers import density_feedback_control
from density_utils.density import p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


METHOD = "collision_cone"
CONTROLLER = "density_feedback"


@dataclass(frozen=True)
class MovingObstacle:
    center: np.ndarray
    velocity: np.ndarray
    r1: float
    r2: float
    p: float = 2.0


def _smooth_scalar_bump(value, inner, outer):
    denom = max(float(outer) - float(inner), 1e-12)
    s = (float(value) - float(inner)) / denom
    if s <= 0.0:
        return 0.0
    if s >= 1.0:
        return 1.0
    f = np.exp(-1.0 / s)
    f_shift = np.exp(-1.0 / (1.0 - s))
    return f / (f + f_shift)


def _collision_cone_h(ego_pos, ego_vel, obs_pos, obs_vel, collision_radius):
    p_rel = np.asarray(obs_pos, dtype=float) - np.asarray(ego_pos, dtype=float)
    v_rel = np.asarray(obs_vel, dtype=float) - np.asarray(ego_vel, dtype=float)
    p_norm = float(np.linalg.norm(p_rel))
    v_norm = float(np.linalg.norm(v_rel))
    if p_norm <= float(collision_radius):
        return -np.inf
    if v_norm < 1e-8:
        return np.inf
    cos_phi = np.sqrt(max(p_norm * p_norm - float(collision_radius) ** 2, 0.0)) / p_norm
    return float(p_rel @ v_rel + p_norm * v_norm * cos_phi)


def _selected_indices(centers, radii, args, robot_pos):
    active = np.flatnonzero(np.asarray(radii, dtype=float) > 0.0)
    if active.size <= args.dynamic_neighbors:
        return active
    distances = np.linalg.norm(np.asarray(centers, dtype=float)[active] - robot_pos, axis=1)
    return active[np.argsort(distances)[: args.dynamic_neighbors]]


def _moving_obstacles(centers, radii, velocities, indices, robot_radius, sensing_margin):
    return [
        MovingObstacle(
            center=np.asarray(centers[idx], dtype=float),
            velocity=np.asarray(velocities[idx], dtype=float),
            r1=float(robot_radius + radii[idx]),
            r2=float(robot_radius + radii[idx] + sensing_margin),
        )
        for idx in indices
    ]


def _collision_cone_planar(state, goal, obstacles, ego_vel, args):
    nominal = density_feedback_control(
        state[:2],
        goal,
        args.alpha,
        [],
        ctrl_multiplier=args.ctrl_multiplier,
        rad_from_goal=args.rad_from_goal,
        q_lqr=1.0,
        r_lqr=1.0,
        dt=args.dt,
        saturation=4.0,
    )
    correction = np.zeros(2, dtype=float)
    for obs in obstacles:
        p_rel = obs.center - state[:2]
        p_norm = float(np.linalg.norm(p_rel))
        if p_norm < 1e-8:
            continue
        spatial_weight = 1.0 - p_norm_bump(state[:2], obs.center, obs.r1, obs.r2, p=obs.p)
        if spatial_weight <= 1e-8:
            continue
        h_value = _collision_cone_h(state[:2], ego_vel, obs.center, obs.velocity, obs.r1)
        risk_weight = 1.0 - _smooth_scalar_bump(h_value, -args.cone_density_margin, args.cone_density_margin)
        if risk_weight <= 1e-8:
            continue
        axis_away = -p_rel / p_norm
        tangent = np.array([-axis_away[1], axis_away[0]], dtype=float)
        if np.dot(tangent, nominal) < 0.0:
            tangent = -tangent
        correction += spatial_weight * risk_weight * (1.15 * axis_away + 0.65 * tangent)
    planar = nominal + args.ctrl_multiplier * correction
    max_abs = float(np.max(np.abs(planar)))
    if max_abs > 4.0:
        planar = planar / max_abs * 4.0
    return planar


def _unicycle_from_planar(state, planar_u, v_max, omega_max, k_heading):
    speed = float(np.linalg.norm(planar_u))
    if speed < 1e-10:
        return np.zeros(2, dtype=float)
    desired_heading = float(np.arctan2(planar_u[1], planar_u[0]))
    heading_error = wrap_angle(desired_heading - state[2])
    turn_gate = max(0.0, np.cos(heading_error))
    return np.array(
        [min(speed, float(v_max)) * turn_gate, float(np.clip(k_heading * heading_error, -omega_max, omega_max))],
        dtype=float,
    )


def _smooth_control(raw, previous, smoothing, v_max, omega_max):
    if np.linalg.norm(previous) < 1e-10:
        return raw
    smoothed = float(smoothing) * previous + (1.0 - float(smoothing)) * raw
    return np.array(
        [float(np.clip(smoothed[0], 0.0, v_max)), float(np.clip(smoothed[1], -omega_max, omega_max))],
        dtype=float,
    )


def _navigation_goal(scenario, state):
    if scenario.streaming is None:
        return scenario.goal
    path_dir = np.asarray(scenario.streaming.path_direction, dtype=float)
    progress = float((state[:2] - scenario.start) @ path_dir)
    final_progress = float((scenario.goal - scenario.start) @ path_dir)
    target_progress = min(progress + scenario.streaming.goal_lookahead, final_progress)
    return scenario.start + target_progress * path_dir


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = finalize_args(parser.parse_args(argv))
    scenario = make_scenario(args.scenario)

    state = np.array([scenario.start[0], scenario.start[1], scenario.heading], dtype=float)
    centers, radii, velocities, active_obstacles, obstacle_rng = initialize_dynamic_obstacles(scenario, state[:2])
    previous_planar = np.array([0.2 * np.cos(state[2]), 0.2 * np.sin(state[2])], dtype=float)
    previous_planar_cmd = previous_planar.copy()
    previous_control = np.zeros(2, dtype=float)

    suffix = "_local_frame" if args.follow_robot else ""
    animation_base_path = (
        example_root()
        / "animations"
        / CONTROLLER
        / f"dynamic_obstacles_{args.scenario}_{METHOD}_{CONTROLLER}{suffix}.gif"
    )
    save_paths = animation_save_paths(animation_base_path, save_gif=args.save_gif, save_mp4=args.save_mp4)
    want_plot_data = (not args.no_plot) or wants_animation_output(args)
    traj = [state.copy()] if want_plot_data else None
    obstacle_traj = [centers.copy()] if want_plot_data else None
    obstacle_vel_traj = [velocities.copy()] if want_plot_data else None
    obstacle_radii_traj = [radii.copy()] if want_plot_data else None
    controls = []
    clearances = [min_obstacle_clearance(state, centers, radii, args.robot_radius)]
    timer = TimedBlock(enabled=args.log_timing)
    control_time = 0.0

    for step in range(args.steps):
        goal = _navigation_goal(scenario, state)
        dist = float(np.linalg.norm(state[:2] - scenario.goal))
        if dist < args.rad_from_goal:
            print(f"stopping at iter={step} (robot within rad_from_goal)")
            break
        if args.print_interval > 0 and step % args.print_interval == 0:
            print(f"iter={step} dist={dist:.3f} clearance={clearances[-1]:.3f}")

        if step < scenario.start_delay_steps:
            control = np.zeros(2, dtype=float)
            previous_planar = np.zeros(2, dtype=float)
        else:
            idx = _selected_indices(centers, radii, args, state[:2])
            obstacles = _moving_obstacles(centers, radii, velocities, idx, args.robot_radius, args.sensing_margin)
            with timer:
                planar = _collision_cone_planar(state, goal, obstacles, previous_planar, args)
            control_time += timer.last
            planar = 0.25 * planar + 0.75 * previous_planar_cmd
            previous_planar_cmd = planar
            raw_control = _unicycle_from_planar(state, planar, args.v_max, args.omega_max, args.k_heading)
            control = _smooth_control(raw_control, previous_control, 0.65, args.v_max, args.omega_max)
            previous_control = control

        state = unicycle_step(state, float(control[0]), float(control[1]), args.dt)
        state[2] = wrap_angle(state[2])
        previous_planar = control[0] * np.array([np.cos(state[2]), np.sin(state[2])], dtype=float)
        centers, radii, velocities, active_obstacles, _ = step_dynamic_obstacles(
            centers,
            radii,
            velocities,
            active_obstacles,
            scenario,
            state[:2],
            args.dt,
            step + 1,
            obstacle_rng,
        )
        controls.append(control)
        clearances.append(min_obstacle_clearance(state, centers, radii, args.robot_radius))
        if want_plot_data:
            traj.append(state.copy())
            obstacle_traj.append(centers.copy())
            obstacle_vel_traj.append(velocities.copy())
            obstacle_radii_traj.append(radii.copy())
        if clearances[-1] < 0.0:
            print(f"collision at iter={step + 1}")
            break

    final_dist = float(np.linalg.norm(state[:2] - scenario.goal))
    status = "collision" if min(clearances) < 0.0 else "success" if final_dist < args.rad_from_goal else "timeout"
    steps_taken = len(controls)
    avg_ms = control_time / max(steps_taken, 1) * 1e3
    print(
        f"status={status} steps={steps_taken} final_dist={final_dist:.3f} "
        f"min_obstacle_clearance={min(clearances):.3f} avg_iteration_mean={avg_ms:.3f} [ms]"
    )

    if want_plot_data:
        plot_dynamic_obstacle_results(
            traj=np.asarray(traj, dtype=float),
            controls=np.asarray(controls, dtype=float),
            obstacle_traj=np.asarray(obstacle_traj, dtype=float),
            obstacle_vel_traj=np.asarray(obstacle_vel_traj, dtype=float),
            obstacle_radii=np.asarray(obstacle_radii_traj, dtype=float),
            clearances=np.asarray(clearances, dtype=float),
            dt=args.dt,
            start=scenario.start,
            goal=scenario.goal,
            robot_radius=args.robot_radius,
            title="Dynamic Obstacles - Collision Cone Density Feedback",
            xlim=scenario.xlim,
            ylim=scenario.ylim,
            save_paths=save_paths,
            show_plot=not args.no_plot,
            fps=args.animation_fps,
            animation_stride=args.animation_stride,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
            follow_robot=args.follow_robot,
            follow_size=(args.follow_width, args.follow_height),
        )


if __name__ == "__main__":
    main()
