import argparse
from pathlib import Path
import sys

import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
MULTI_AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(MULTI_AGENT_ROOT), str(REPO_ROOT), str(EXAMPLE_ROOT / "unicycle"), str(MULTI_AGENT_ROOT / "density_feedback")]

from _config import add_common_arguments, finalize_args, make_scenario, min_pair_clearance, mode_label, wrap_angle
from _plotting import animation_save_paths, wants_animation_output, _plot_multi_agent_results

from density_utils.controllers import SOLVER_CHOICES, single_integrator_nominal_control, solve_discrete_density_filter
from density_utils.density import Obstacle, p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


INTERACTION_MODE = "density_filter_reactive"


def _pose_error(state, goal_state):
    return np.array(
        [state[0] - goal_state[0], state[1] - goal_state[1], wrap_angle(state[2] - goal_state[2])],
        dtype=float,
    )


def _pose_density(state, goal_state, alpha, obstacles, theta_weight=0.05, min_v=1e-6):
    err = _pose_error(state, goal_state)
    lyap = max(float(err[0] ** 2 + err[1] ** 2 + theta_weight * err[2] ** 2), min_v)
    phi = 1.0
    for obs in obstacles:
        phi *= p_norm_bump(state[:2], obs.center, obs.r1, obs.r2, p=obs.p, scale=obs.scale, angle=obs.angle)
    return phi / (lyap ** float(alpha))


def _unicycle_filter_step(state, control, dt):
    return unicycle_step(state, float(control[0]), float(control[1]), dt)


def _unicycle_nominal_from_planar_ref(state, planar_ref, v_max, omega_max, k_heading):
    speed = float(np.linalg.norm(planar_ref))
    if speed < 1e-10:
        return np.zeros(2, dtype=float)
    desired_heading = float(np.arctan2(planar_ref[1], planar_ref[0]))
    heading_error = wrap_angle(desired_heading - state[2])
    turn_gate = max(0.0, np.cos(heading_error))
    return np.array(
        [
            min(speed, float(v_max)) * turn_gate,
            float(np.clip(k_heading * heading_error, -omega_max, omega_max)),
        ],
        dtype=float,
    )


def _smooth_nominal(raw, previous, smoothing, v_max, omega_max):
    if np.linalg.norm(previous) < 1e-10:
        return raw
    smoothed = float(smoothing) * previous + (1.0 - float(smoothing)) * raw
    return np.array(
        [
            float(np.clip(smoothed[0], 0.0, v_max)),
            float(np.clip(smoothed[1], -omega_max, omega_max)),
        ],
        dtype=float,
    )


def _nearest_indices(states, agent_idx, max_neighbors):
    other_indices = np.delete(np.arange(states.shape[0]), agent_idx)
    if max_neighbors <= 0 or len(other_indices) <= max_neighbors:
        return other_indices
    rel = states[other_indices, :2] - states[agent_idx, :2]
    return other_indices[np.argsort(np.linalg.norm(rel, axis=1))[:max_neighbors]]


def _agent_obstacles(states, agent_r1, agent_r2, agent_idx, other_indices, safety_margin):
    return [
        Obstacle(
            center=states[other_idx, :2],
            r1=agent_r1[agent_idx] + agent_r1[other_idx] + safety_margin,
            r2=agent_r1[agent_idx] + agent_r2[other_idx] + safety_margin,
            p=2.0,
        )
        for other_idx in other_indices
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto")
    parser.add_argument("--filter-neighbors", type=int, default=2)
    parser.add_argument("--filter-dt", type=float, default=0.1)
    parser.add_argument("--filter-safety-margin", type=float, default=0.0)
    parser.add_argument("--slack-weight", type=float, default=1e4)
    parser.add_argument("--control-weight", type=float, default=1.0)
    parser.add_argument("--nominal-smoothing", type=float, default=0.25)
    parser.add_argument("--density-weight", type=float, default=1.0)
    parser.add_argument("--goal-weight", type=float, default=0.5)
    parser.add_argument("--q-lqr", type=float, default=4.0)
    parser.add_argument("--r-lqr", type=float, default=1.0)
    args = finalize_args(parser.parse_args(argv))

    filter_neighbors = args.filter_neighbors
    control_weight = args.control_weight
    nominal_smoothing = args.nominal_smoothing
    density_weight = args.density_weight
    goal_weight = args.goal_weight
    if args.scenario == "crossing4":
        filter_neighbors = max(filter_neighbors, 3)

    starts, goals, headings, agent_r1, agent_r2 = make_scenario(args, INTERACTION_MODE)
    states = np.hstack([starts, headings[:, None]])
    goal_states = np.hstack([goals, headings[:, None]])
    num_agents = states.shape[0]
    previous_control = np.zeros((num_agents, 2), dtype=float)
    u_min = np.array([0.0, -args.omega_max], dtype=float)
    u_max = np.array([args.v_max, args.omega_max], dtype=float)

    animation_base_path = (
        Path(__file__).resolve().parents[1]
        / "animations"
        / "density_filter"
        / f"multi_agent_unicycle_{args.scenario}_{INTERACTION_MODE}.gif"
    )
    save_paths = animation_save_paths(animation_base_path, save_gif=args.save_gif, save_mp4=args.save_mp4)
    want_plot_data = (not args.no_plot) or wants_animation_output(args)
    stored_traj = [states.copy()] if want_plot_data else None
    animation_stride = max(1, args.animation_stride)
    control_time = np.zeros(num_agents, dtype=float)
    timers = [TimedBlock(enabled=args.log_timing) for _ in range(num_agents)]
    solver_failures = 0
    max_slack = 0.0
    min_clearance = min_pair_clearance(states, agent_r1)

    for step in range(args.steps):
        dists = np.linalg.norm(states[:, :2] - goals, axis=1)
        if np.all(dists < args.rad_from_goal):
            print(f"stopping at iter={step} (all agents within rad_from_goal)")
            break
        if args.print_interval > 0 and step % args.print_interval == 0:
            print("iter=" + str(step) + " dists=" + np.array2string(dists, precision=3))

        new_states = states.copy()
        for agent_idx in range(num_agents):
            other_indices = _nearest_indices(states, agent_idx, filter_neighbors)
            obstacles = _agent_obstacles(
                states, agent_r1, agent_r2, agent_idx, other_indices, args.filter_safety_margin
            )
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
                density_weight=density_weight,
                goal_weight=goal_weight,
            )
            nominal = _unicycle_nominal_from_planar_ref(
                states[agent_idx], planar_nominal, args.v_max, args.omega_max, args.k_heading
            )
            nominal = _smooth_nominal(
                nominal,
                previous_control[agent_idx],
                nominal_smoothing,
                args.v_max,
                args.omega_max,
            )
            with timers[agent_idx]:
                result = solve_discrete_density_filter(
                    states[agent_idx],
                    goal_states[agent_idx],
                    args.alpha,
                    obstacles,
                    u_nom=nominal,
                    dt=args.filter_dt,
                    next_state_fn=_unicycle_filter_step,
                    u_min=u_min,
                    u_max=u_max,
                    divergence=0.0,
                    slack_weight=args.slack_weight,
                    control_weight=control_weight,
                    density_fn=_pose_density,
                    solver=args.solver,
                    return_info=True,
                )
            control_time[agent_idx] += timers[agent_idx].last
            previous_control[agent_idx] = result.u
            max_slack = max(max_slack, float(np.max(result.slack)) if result.slack.size else 0.0)
            if not result.success:
                solver_failures += 1
            control = previous_control[agent_idx]
            new_states[agent_idx] = unicycle_step(states[agent_idx], float(control[0]), float(control[1]), args.dt)
            new_states[agent_idx, 2] = wrap_angle(new_states[agent_idx, 2])

        states = new_states
        min_clearance = min(min_clearance, min_pair_clearance(states, agent_r1))
        if want_plot_data and ((step + 1) % animation_stride == 0):
            stored_traj.append(states.copy())

    final_dists = np.linalg.norm(states[:, :2] - goals, axis=1)
    status = "collision" if min_clearance < 0.0 else "success" if np.all(final_dists < args.rad_from_goal) else "timeout"
    steps_taken = step + 1 if "step" in locals() else 0
    avg_ms = control_time / max(steps_taken, 1) * 1e3
    print(
        f"status={status} steps={steps_taken} "
        f"max_dist={np.max(final_dists):.3f} mean_dist={np.mean(final_dists):.3f} "
        f"min_pair_clearance={min_clearance:.3f} max_slack={max_slack:.2e} "
        f"solver_failures={solver_failures} avg_iteration_mean={np.mean(avg_ms):.3f} [ms]"
    )

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
