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
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


INTERACTION_MODE = "reactive_density_feedback"


def _agent_obstacles(states, agent_r1, agent_r2, agent_idx):
    obstacles = []
    for other_idx in range(states.shape[0]):
        if other_idx == agent_idx:
            continue
        obstacles.append(
            Obstacle(
                center=states[other_idx, :2],
                r1=agent_r1[agent_idx] + agent_r1[other_idx],
                r2=agent_r1[agent_idx] + agent_r2[other_idx],
                p=2.0,
            )
        )
    return obstacles


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = finalize_args(parser.parse_args(argv))

    starts, goals, headings, agent_r1, agent_r2 = make_scenario(args, INTERACTION_MODE)
    states = np.hstack([starts, headings[:, None]])
    tilde_prev = headings.copy()
    num_agents = states.shape[0]

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
        for agent_idx in range(num_agents):
            obstacles = _agent_obstacles(states, agent_r1, agent_r2, agent_idx)
            with timers[agent_idx]:
                planar_u = density_feedback_control(
                    states[agent_idx, :2],
                    goals[agent_idx],
                    args.alpha,
                    obstacles,
                    ctrl_multiplier=args.ctrl_multiplier,
                    rad_from_goal=args.rad_from_goal,
                    q_lqr=q_lqr,
                    r_lqr=r_lqr,
                    dt=args.dt,
                    saturation=saturation,
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

        states = new_states
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
