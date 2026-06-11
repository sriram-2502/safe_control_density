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

from density_utils.controllers import SOLVER_CHOICES, density_feedback_control, solve_discrete_density_filter
from density_utils.density import p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


METHOD = "collision_cone"
CONTROLLER = "density_filter"


@dataclass(frozen=True)
class MovingObstacle:
    center: np.ndarray
    velocity: np.ndarray
    r1: float
    r2: float
    p: float = 2.0


def _pose_error(state, goal_state):
    return np.array(
        [state[0] - goal_state[0], state[1] - goal_state[1], wrap_angle(state[2] - goal_state[2])],
        dtype=float,
    )


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


def _obstacle_center(obs, state_eval, reference_state, dt):
    if reference_state is None or dt is None:
        return obs.center
    is_next = float(np.linalg.norm(state_eval[:2] - reference_state[:2])) > 1e-9
    return obs.center + obs.velocity * float(dt) if is_next else obs.center


def _ego_velocity(state_eval, reference_state, previous_planar, dt):
    if reference_state is None or dt is None:
        return previous_planar
    displacement = np.asarray(state_eval[:2], dtype=float) - np.asarray(reference_state[:2], dtype=float)
    if float(np.linalg.norm(displacement)) < 1e-9:
        return previous_planar
    return displacement / max(float(dt), 1e-9)


def _collision_cone_density(
    state,
    goal_state,
    alpha,
    obstacles,
    previous_planar,
    cone_margin,
    reference_state=None,
    dt=None,
):
    err = _pose_error(state, goal_state)
    lyap = max(float(err[0] ** 2 + err[1] ** 2 + 0.05 * err[2] ** 2), 1e-6)
    ego_vel = _ego_velocity(state, reference_state, previous_planar, dt)
    phi = 1.0
    for obs in obstacles:
        center = _obstacle_center(obs, state, reference_state, dt)
        spatial = p_norm_bump(state[:2], center, obs.r1, obs.r2, p=obs.p)
        h_value = _collision_cone_h(state[:2], ego_vel, center, obs.velocity, obs.r1)
        phi *= spatial * _smooth_scalar_bump(h_value, -cone_margin, cone_margin)
    return phi / (lyap ** float(alpha))


def _selected_indices(centers, radii, args, robot_pos):
    active = np.flatnonzero(np.asarray(radii, dtype=float) > 0.0)
    if active.size <= args.dynamic_neighbors:
        return active
    distances = np.linalg.norm(np.asarray(centers, dtype=float)[active] - robot_pos, axis=1)
    return active[np.argsort(distances)[: args.dynamic_neighbors]]


def _moving_obstacles(centers, radii, velocities, indices, robot_radius, sensing_margin, safety_margin):
    return [
        MovingObstacle(
            center=np.asarray(centers[idx], dtype=float),
            velocity=np.asarray(velocities[idx], dtype=float),
            r1=float(robot_radius + radii[idx] + safety_margin),
            r2=float(robot_radius + radii[idx] + sensing_margin + safety_margin),
        )
        for idx in indices
    ]


def _collision_cone_planar(state, goal, obstacles, ego_vel, args, predict_dt=0.0):
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
        center = obs.center + float(predict_dt) * obs.velocity
        p_rel = center - state[:2]
        p_norm = float(np.linalg.norm(p_rel))
        if p_norm < 1e-8:
            continue
        spatial_weight = 1.0 - p_norm_bump(state[:2], center, obs.r1, obs.r2, p=obs.p)
        if spatial_weight <= 1e-8:
            continue
        h_value = _collision_cone_h(state[:2], ego_vel, center, obs.velocity, obs.r1)
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


def _unicycle_filter_step(state, control, dt):
    return unicycle_step(state, float(control[0]), float(control[1]), dt)


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


def _smooth_nominal(raw, previous, smoothing, v_max, omega_max):
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
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto")
    parser.add_argument("--filter-dt", type=float, default=None)
    parser.add_argument("--filter-safety-margin", type=float, default=0.02)
    parser.add_argument("--slack-weight", type=float, default=1e4)
    parser.add_argument("--control-weight", type=float, default=0.6)
    parser.add_argument("--nominal-smoothing", type=float, default=0.15)
    args = finalize_args(parser.parse_args(argv))
    if args.filter_dt is None:
        args.filter_dt = args.dt
    if args.scenario == "closing_in":
        args.cone_density_margin = max(args.cone_density_margin, 0.8)
    scenario = make_scenario(args.scenario)

    state = np.array([scenario.start[0], scenario.start[1], scenario.heading], dtype=float)
    centers, radii, velocities, active_obstacles, obstacle_rng = initialize_dynamic_obstacles(scenario, state[:2])
    previous_control = np.zeros(2, dtype=float)
    previous_planar = np.array([0.2 * np.cos(state[2]), 0.2 * np.sin(state[2])], dtype=float)
    u_min = np.array([0.0, -args.omega_max], dtype=float)
    u_max = np.array([args.v_max, args.omega_max], dtype=float)

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
    solver_failures = 0
    max_slack = 0.0

    for step in range(args.steps):
        goal = _navigation_goal(scenario, state)
        goal_state = np.array([goal[0], goal[1], scenario.heading], dtype=float)
        dist = float(np.linalg.norm(state[:2] - scenario.goal))
        if dist < args.rad_from_goal:
            print(f"stopping at iter={step} (robot within rad_from_goal)")
            break
        if args.print_interval > 0 and step % args.print_interval == 0:
            print(f"iter={step} dist={dist:.3f} clearance={clearances[-1]:.3f}")

        if step < scenario.start_delay_steps:
            control = np.zeros(2, dtype=float)
            previous_planar = np.zeros(2, dtype=float)
            previous_control = control
        else:
            idx = _selected_indices(centers, radii, args, state[:2])
            moving_obstacles = _moving_obstacles(
                centers,
                radii,
                velocities,
                idx,
                args.robot_radius,
                args.sensing_margin,
                args.filter_safety_margin,
            )
            planar_nominal = _collision_cone_planar(
                state,
                goal,
                moving_obstacles,
                previous_planar,
                args,
                predict_dt=args.filter_dt,
            )
            nominal = _unicycle_from_planar(state, planar_nominal, args.v_max, args.omega_max, args.k_heading)
            nominal = _smooth_nominal(nominal, previous_control, args.nominal_smoothing, args.v_max, args.omega_max)

            def density_fn(state_eval, goal_eval, alpha_eval, obstacles_eval, reference_state=state.copy()):
                return _collision_cone_density(
                    state_eval,
                    goal_eval,
                    alpha_eval,
                    obstacles_eval,
                    previous_planar,
                    args.cone_density_margin,
                    reference_state=reference_state,
                    dt=args.filter_dt,
                )

            with timer:
                result = solve_discrete_density_filter(
                    state,
                    goal_state,
                    args.alpha,
                    moving_obstacles,
                    u_nom=nominal,
                    dt=args.filter_dt,
                    next_state_fn=_unicycle_filter_step,
                    u_min=u_min,
                    u_max=u_max,
                    divergence=0.0,
                    slack_weight=args.slack_weight,
                    control_weight=args.control_weight,
                    density_fn=density_fn,
                    solver=args.solver,
                    return_info=True,
                )
            control_time += timer.last
            solver_failures += int(not result.success)
            max_slack = max(max_slack, float(np.max(result.slack)) if result.slack.size else 0.0)
            control = result.u
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
        f"min_obstacle_clearance={min(clearances):.3f} max_slack={max_slack:.2e} "
        f"solver_failures={solver_failures} avg_iteration_mean={avg_ms:.3f} [ms]"
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
            title="Dynamic Obstacles - Collision Cone Density Filter",
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
