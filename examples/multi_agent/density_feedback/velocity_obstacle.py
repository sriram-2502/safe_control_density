import argparse
from pathlib import Path

import numpy as np

from _config import (
    add_common_arguments,
    finalize_args,
    make_scenario,
    min_pair_clearance,
    mode_label,
    wrap_angle,
)
from _plotting import animation_save_paths, wants_animation_output, _plot_multi_agent_results

from density_utils.controllers import density_feedback_control
from density_utils.density import Obstacle
from density_utils.density.bump import p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


INTERACTION_MODE = "velocity_obstacle_density_feedback"
VO_DEFAULT_EXTRA_RADIUS = 0.35
VO_DEFAULT_CLOSE_MARGIN = 0.18
VO_DEFAULT_PROJECTION_PASSES = 2
VO_SCENARIO_TUNING = {
    "crossing2": (0.45, 0.16, 2),
    "crossing4": (0.50, 0.18, 2),
}


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _rotate(vec, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _velocity_obstacle_project(
    candidate_vel,
    ego_pos,
    other_positions,
    other_velocities,
    ego_r1,
    other_r1,
    other_r2,
    vo_margin,
    agent_idx,
    extra_radius,
    projection_passes,
):
    projected = np.asarray(candidate_vel, dtype=float).copy()
    ego_pos = np.asarray(ego_pos, dtype=float)
    for _ in range(projection_passes):
        for other_pos, other_vel, r1, r2 in zip(other_positions, other_velocities, other_r1, other_r2):
            collision_r1 = float(ego_r1 + r1)
            influence_radius = max(float(ego_r1 + r2) + extra_radius, collision_r1 + extra_radius)
            spatial_bump = p_norm_bump(ego_pos, other_pos, collision_r1, influence_radius, p=2.0)
            spatial_weight = 1.0 - spatial_bump
            if spatial_weight <= 1e-8:
                continue

            p_rel = np.asarray(other_pos, dtype=float) - ego_pos
            p_norm = float(np.linalg.norm(p_rel))
            if p_norm <= collision_r1:
                projected -= spatial_weight * p_rel / max(p_norm, 1e-6)
                continue

            axis = p_rel / p_norm
            half_angle = np.arcsin(np.clip(collision_r1 / p_norm, 0.0, 1.0))
            left_boundary = _rotate(axis, half_angle)
            right_boundary = _rotate(axis, -half_angle)
            v_rel = projected - np.asarray(other_vel, dtype=float)

            left_safe_margin = _cross2(left_boundary, v_rel)
            right_safe_margin = _cross2(v_rel, right_boundary)
            safe_margin = max(left_safe_margin, right_safe_margin)
            if safe_margin >= vo_margin:
                continue

            if abs(left_safe_margin - right_safe_margin) < 1e-8:
                use_left = agent_idx % 2 == 0
            else:
                use_left = left_safe_margin > right_safe_margin
            if use_left:
                direction = np.array([-left_boundary[1], left_boundary[0]])
                margin = left_safe_margin
            else:
                direction = np.array([right_boundary[1], -right_boundary[0]])
                margin = right_safe_margin
            projected += spatial_weight * (vo_margin - margin) * direction
    return projected


def _velocity_obstacle_density_control(
    *,
    ego_pos,
    goal,
    agent_idx,
    other_positions,
    other_velocities,
    ego_r1,
    other_r1,
    other_r2,
    alpha,
    vo_margin,
    ctrl_multiplier,
    rad_from_goal,
    q_lqr,
    r_lqr,
    dt,
    saturation,
    extra_radius,
    close_margin,
    projection_passes,
):
    obstacles = [
        Obstacle(
            center=other_pos,
            r1=float(ego_r1 + r1),
            r2=float(ego_r1 + r2),
            p=2.0,
        )
        for other_pos, r1, r2 in zip(other_positions, other_r1, other_r2)
    ]
    reactive_nominal = density_feedback_control(
        ego_pos,
        goal,
        alpha,
        obstacles,
        ctrl_multiplier=ctrl_multiplier,
        rad_from_goal=rad_from_goal,
        q_lqr=q_lqr,
        r_lqr=r_lqr,
        dt=dt,
        saturation=saturation,
    )
    nominal = density_feedback_control(
        ego_pos,
        goal,
        alpha,
        [],
        ctrl_multiplier=ctrl_multiplier,
        rad_from_goal=rad_from_goal,
        q_lqr=q_lqr,
        r_lqr=r_lqr,
        dt=dt,
        saturation=saturation,
    )
    if np.linalg.norm(ego_pos - goal) < rad_from_goal:
        return nominal

    vo_velocity = _velocity_obstacle_project(
        nominal,
        ego_pos,
        other_positions,
        other_velocities,
        ego_r1,
        other_r1,
        other_r2,
        vo_margin,
        agent_idx,
        extra_radius,
        projection_passes,
    )

    close_weight = 0.0
    for other_pos, r1 in zip(other_positions, other_r1):
        collision_r1 = float(ego_r1 + r1)
        close_bump = p_norm_bump(ego_pos, other_pos, collision_r1, collision_r1 + close_margin, p=2.0)
        close_weight = max(close_weight, 1.0 - close_bump)
    planar_u = (1.0 - close_weight) * vo_velocity + close_weight * reactive_nominal
    max_u = np.max(np.abs(planar_u))
    if max_u > saturation:
        planar_u = planar_u / max_u * saturation
    return planar_u


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = finalize_args(parser.parse_args(argv))
    vo_extra_radius, vo_close_margin, vo_projection_passes = VO_SCENARIO_TUNING.get(
        args.scenario,
        (VO_DEFAULT_EXTRA_RADIUS, VO_DEFAULT_CLOSE_MARGIN, VO_DEFAULT_PROJECTION_PASSES),
    )

    starts, goals, headings, agent_r1, agent_r2 = make_scenario(args, INTERACTION_MODE)
    states = np.hstack([starts, headings[:, None]])
    tilde_prev = headings.copy()
    num_agents = states.shape[0]
    other_agent_indices = [np.delete(np.arange(num_agents), agent_idx) for agent_idx in range(num_agents)]
    prev_planar_vel = np.column_stack([np.cos(headings), np.sin(headings)]) * 0.2

    q_lqr = 1.0
    r_lqr = 1.0
    saturation = 4.0
    stop_tol = min(0.005, args.rad_from_goal)
    stop_steps = 500
    stop_count = 0

    animation_base_path = (
        Path(__file__).resolve().parents[1]
        / "animations"
        / "density_feedback"
        / f"multi_agent_unicycle_{args.scenario}_{INTERACTION_MODE}.gif"
    )
    save_paths = animation_save_paths(
        animation_base_path,
        save_gif=args.save_gif,
        save_mp4=args.save_mp4,
    )
    want_plot_data = (not args.no_plot) or wants_animation_output(args)
    animation_stride = max(1, args.animation_stride)
    stored_traj = [states.copy()] if want_plot_data else None
    control_time = np.zeros(num_agents, dtype=float)
    timers = [TimedBlock(enabled=args.log_timing) for _ in range(num_agents)]
    min_clearance = min_pair_clearance(states, agent_r1)

    for step in range(args.steps):
        dists = np.linalg.norm(states[:, :2] - goals, axis=1)
        if np.all(dists < args.rad_from_goal):
            print(f"stopping at iter={step} (all agents within rad_from_goal)")
            break
        if np.max(dists) < stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(f"stopping at iter={step} (stable within stop_tol)")
                break
        else:
            stop_count = 0
        if args.print_interval > 0 and step % args.print_interval == 0:
            print("iter=" + str(step) + " dists=" + np.array2string(dists, precision=3))

        new_states = states.copy()
        new_planar_vel = np.zeros_like(prev_planar_vel)
        for agent_idx in range(num_agents):
            other_indices = other_agent_indices[agent_idx]
            vo_density_neighbors = 2 if args.scenario == "swap8_opposite" else args.cone_density_neighbors
            if vo_density_neighbors > 0 and len(other_indices) > vo_density_neighbors:
                rel = states[other_indices, :2] - states[agent_idx, :2]
                nearest = np.argsort(np.linalg.norm(rel, axis=1))[:vo_density_neighbors]
                other_indices = other_indices[nearest]

            with timers[agent_idx]:
                planar_u = _velocity_obstacle_density_control(
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
                    extra_radius=vo_extra_radius,
                    close_margin=vo_close_margin,
                    projection_passes=vo_projection_passes,
                )
            control_time[agent_idx] += timers[agent_idx].last

            v = min(float(np.linalg.norm(planar_u)), args.v_max)
            desired_heading = float(np.arctan2(planar_u[1], planar_u[0]))
            desired_heading_rate = wrap_angle(desired_heading - tilde_prev[agent_idx]) / args.dt
            tilde_prev[agent_idx] = desired_heading
            omega = desired_heading_rate - args.k_heading * wrap_angle(states[agent_idx, 2] - desired_heading)
            omega = float(np.clip(omega, -args.omega_max, args.omega_max))
            new_states[agent_idx] = unicycle_step(states[agent_idx], v, omega, args.dt)
            new_states[agent_idx, 2] = wrap_angle(new_states[agent_idx, 2])
            new_planar_vel[agent_idx] = v * np.array([np.cos(new_states[agent_idx, 2]), np.sin(new_states[agent_idx, 2])])

        states = new_states
        prev_planar_vel = new_planar_vel
        min_clearance = min(min_clearance, min_pair_clearance(states, agent_r1))
        if want_plot_data and ((step + 1) % animation_stride == 0):
            stored_traj.append(states.copy())

    final_dists = np.linalg.norm(states[:, :2] - goals, axis=1)
    status = "success" if np.all(final_dists < args.rad_from_goal) else "timeout"
    steps_taken = step + 1 if "step" in locals() else 0
    avg_ms = control_time / max(steps_taken, 1) * 1e3
    print(
        f"status={status} steps={steps_taken} "
        f"max_dist={np.max(final_dists):.3f} mean_dist={np.mean(final_dists):.3f} "
        f"min_pair_clearance={min_clearance:.3f} "
        f"avg_iteration_mean={np.mean(avg_ms):.3f} [ms]"
    )

    if args.log_timing:
        for idx, timer in enumerate(timers, start=1):
            mean_ms, std_ms = timer.mean_std_ms()
            if mean_ms is not None:
                print(f"agent_{idx}_avg_iteration={mean_ms:.3f} [ms] std={std_ms:.3f} [ms]")

    if want_plot_data:
        if not np.allclose(stored_traj[-1], states):
            stored_traj.append(states.copy())
        _plot_multi_agent_results(
            traj=np.asarray(stored_traj, dtype=float),
            starts=starts,
            goals=goals,
            agent_r1=agent_r1,
            title=f"Multi-Agent Unicycle - {mode_label(INTERACTION_MODE)}",
            save_paths=save_paths,
            show_plot=not args.no_plot,
            fps=args.animation_fps,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )


if __name__ == "__main__":
    main()
