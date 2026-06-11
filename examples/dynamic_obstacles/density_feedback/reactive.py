import argparse
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
from density_utils.density import Obstacle
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


METHOD = "reactive"
CONTROLLER = "density_feedback"


def _nearest_indices(centers, radii, max_neighbors, robot_pos):
    active = np.flatnonzero(np.asarray(radii, dtype=float) > 0.0)
    if active.size <= max_neighbors:
        return active
    distances = np.linalg.norm(np.asarray(centers, dtype=float)[active] - robot_pos, axis=1)
    return active[np.argsort(distances)[:max_neighbors]]


def _obstacles_from_indices(centers, radii, indices, robot_radius, sensing_margin):
    return [
        Obstacle(
            center=np.asarray(centers[idx], dtype=float),
            r1=float(robot_radius + radii[idx]),
            r2=float(robot_radius + radii[idx] + sensing_margin),
            p=2.0,
        )
        for idx in indices
    ]


def _unicycle_from_planar(state, planar_u, v_max, omega_max, k_heading):
    speed = float(np.linalg.norm(planar_u))
    if speed < 1e-10:
        return np.zeros(2, dtype=float)
    desired_heading = float(np.arctan2(planar_u[1], planar_u[0]))
    heading_error = wrap_angle(desired_heading - state[2])
    turn_gate = max(0.0, np.cos(heading_error))
    return np.array(
        [
            min(speed, float(v_max)) * turn_gate,
            float(np.clip(k_heading * heading_error, -omega_max, omega_max)),
        ],
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
        else:
            idx = _nearest_indices(centers, radii, args.dynamic_neighbors, state[:2])
            obstacles = _obstacles_from_indices(centers, radii, idx, args.robot_radius, args.sensing_margin)
            with timer:
                planar_u = density_feedback_control(
                    state[:2],
                    goal,
                    args.alpha,
                    obstacles,
                    ctrl_multiplier=args.ctrl_multiplier,
                    rad_from_goal=args.rad_from_goal,
                    q_lqr=1.0,
                    r_lqr=1.0,
                    dt=args.dt,
                    saturation=4.0,
                )
            control_time += timer.last
            control = _unicycle_from_planar(state, planar_u, args.v_max, args.omega_max, args.k_heading)

        state = unicycle_step(state, float(control[0]), float(control[1]), args.dt)
        state[2] = wrap_angle(state[2])
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
            title="Dynamic Obstacles - Reactive Density Feedback",
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
