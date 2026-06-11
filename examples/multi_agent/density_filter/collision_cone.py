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

from density_utils.controllers import SOLVER_CHOICES, solve_discrete_density_filter
from density_utils.density import Obstacle, p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


INTERACTION_MODE = "density_filter_collision_cone"


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


def _collision_cone_h(ego_pos, ego_vel, other_pos, other_vel, collision_radius):
    p_rel = np.asarray(other_pos, dtype=float) - np.asarray(ego_pos, dtype=float)
    v_rel = np.asarray(other_vel, dtype=float) - np.asarray(ego_vel, dtype=float)
    p_norm = float(np.linalg.norm(p_rel))
    v_norm = float(np.linalg.norm(v_rel))
    if p_norm <= float(collision_radius):
        return -np.inf
    if v_norm < 1e-8:
        return np.inf
    cos_phi = np.sqrt(max(p_norm * p_norm - float(collision_radius) ** 2, 0.0)) / p_norm
    return float(p_rel @ v_rel + p_norm * v_norm * cos_phi)


def _collision_cone_density(
    state,
    goal_state,
    alpha,
    obstacles,
    *,
    other_velocities,
    ego_speed,
    cone_margin,
    theta_weight=0.05,
    min_v=1e-6,
):
    err = _pose_error(state, goal_state)
    lyap = max(float(err[0] ** 2 + err[1] ** 2 + theta_weight * err[2] ** 2), min_v)
    ego_vel = float(ego_speed) * np.array([np.cos(state[2]), np.sin(state[2])], dtype=float)
    phi = 1.0
    for obs, other_vel in zip(obstacles, other_velocities):
        spatial = p_norm_bump(state[:2], obs.center, obs.r1, obs.r2, p=obs.p, scale=obs.scale, angle=obs.angle)
        h_value = _collision_cone_h(state[:2], ego_vel, obs.center, other_vel, obs.r1)
        cone = _smooth_scalar_bump(h_value, -cone_margin, cone_margin)
        phi *= spatial * cone
    return phi / (lyap ** float(alpha))


def _collision_cone_planar_density(
    ego_pos,
    goal,
    ego_vel,
    obstacles,
    other_velocities,
    alpha,
    cone_margin,
    min_dist=1e-3,
):
    dist = max(float(np.linalg.norm(np.asarray(ego_pos, dtype=float) - np.asarray(goal, dtype=float))), min_dist)
    phi = 1.0 / (dist ** (2.0 * float(alpha)))
    for obs, other_vel in zip(obstacles, other_velocities):
        spatial = p_norm_bump(ego_pos, obs.center, obs.r1, obs.r2, p=obs.p, scale=obs.scale, angle=obs.angle)
        h_value = _collision_cone_h(ego_pos, ego_vel, obs.center, other_vel, obs.r1)
        cone = _smooth_scalar_bump(h_value, -cone_margin, cone_margin)
        phi *= spatial * cone
    return phi


def _finite_difference_grad(fn, x, eps=5e-3):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in range(x.size):
        step = np.zeros_like(x)
        step[idx] = eps
        grad[idx] = (fn(x + step) - fn(x - step)) / (2.0 * eps)
    return grad


def _collision_cone_planar_nominal(
    ego_pos,
    goal,
    ego_vel,
    obstacles,
    other_velocities,
    alpha,
    cone_margin,
    ctrl_multiplier,
    rad_from_goal,
    saturation,
):
    goal_vec = np.asarray(goal, dtype=float) - np.asarray(ego_pos, dtype=float)
    goal_dist = float(np.linalg.norm(goal_vec))
    if goal_dist < rad_from_goal:
        return np.zeros(2, dtype=float)
    grad = _finite_difference_grad(
        lambda pos: _collision_cone_planar_density(
            pos,
            goal,
            ego_vel,
            obstacles,
            other_velocities,
            alpha,
            cone_margin,
        ),
        ego_pos,
    )
    goal_dir = goal_vec / max(goal_dist, 1e-12)
    planar = float(ctrl_multiplier) * (grad + 0.35 * goal_dir)
    max_abs = float(np.max(np.abs(planar)))
    if max_abs > saturation:
        planar = planar / max_abs * saturation
    return planar


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
        [min(speed, float(v_max)) * turn_gate, float(np.clip(k_heading * heading_error, -omega_max, omega_max))],
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


def _danger_scores(states, previous_planar, agent_r1, other_indices, agent_idx, margin):
    scores = []
    for other_idx in other_indices:
        h_value = _collision_cone_h(
            states[agent_idx, :2],
            previous_planar[agent_idx],
            states[other_idx, :2],
            previous_planar[other_idx],
            agent_r1[agent_idx] + agent_r1[other_idx],
        )
        distance = float(np.linalg.norm(states[other_idx, :2] - states[agent_idx, :2]))
        if np.isneginf(h_value):
            danger = np.inf
        elif np.isposinf(h_value):
            danger = 1.0 / max(distance, 1e-6)
        else:
            danger = max(0.0, float(margin) - h_value) + 0.05 / max(distance, 1e-6)
        scores.append(danger)
    return np.asarray(scores, dtype=float)


def _active_indices(states, previous_planar, agent_r1, agent_idx, max_neighbors, margin):
    other_indices = np.delete(np.arange(states.shape[0]), agent_idx)
    if max_neighbors <= 0 or len(other_indices) <= max_neighbors:
        return other_indices
    rel = states[other_indices, :2] - states[agent_idx, :2]
    nearest_order = np.argsort(np.linalg.norm(rel, axis=1))
    danger_order = np.argsort(-_danger_scores(states, previous_planar, agent_r1, other_indices, agent_idx, margin))
    selected = list(nearest_order[: max(1, max_neighbors // 2)])
    for idx in danger_order:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_neighbors:
            break
    return other_indices[np.asarray(selected[:max_neighbors], dtype=int)]


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
    args = finalize_args(parser.parse_args(argv))

    filter_safety_margin = args.filter_safety_margin
    slack_weight = args.slack_weight
    control_weight = args.control_weight
    nominal_smoothing = args.nominal_smoothing
    if args.scenario == "crossing2":
        filter_safety_margin = max(filter_safety_margin, 0.08)
        slack_weight = max(slack_weight, 1e4)
        control_weight = max(control_weight, 2.0)
        nominal_smoothing = max(nominal_smoothing, 0.55)

    starts, goals, headings, agent_r1, agent_r2 = make_scenario(args, INTERACTION_MODE)
    filter_neighbors = args.filter_neighbors
    if args.scenario in {"crossing6", "swap8_opposite"}:
        filter_neighbors = max(filter_neighbors, 3)
    states = np.hstack([starts, headings[:, None]])
    goal_states = np.hstack([goals, headings[:, None]])
    num_agents = states.shape[0]
    previous_planar = np.zeros((num_agents, 2), dtype=float)
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
        planar_snapshot = previous_planar.copy()
        next_planar = previous_planar.copy()
        for agent_idx in range(num_agents):
            other_indices = _active_indices(
                states, planar_snapshot, agent_r1, agent_idx, filter_neighbors, args.cone_density_margin
            )
            obstacles = _agent_obstacles(
                states, agent_r1, agent_r2, agent_idx, other_indices, filter_safety_margin
            )
            other_velocities = planar_snapshot[other_indices].copy()
            planar_nominal = _collision_cone_planar_nominal(
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

            def density_fn(state_eval, goal_eval, alpha_eval, obstacles_eval, other_velocities=other_velocities, nominal=nominal):
                return _collision_cone_density(
                    state_eval,
                    goal_eval,
                    alpha_eval,
                    obstacles_eval,
                    other_velocities=other_velocities,
                    ego_speed=max(float(nominal[0]), 0.2),
                    cone_margin=args.cone_density_margin,
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
                    slack_weight=slack_weight,
                    control_weight=control_weight,
                    density_fn=density_fn,
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
            next_planar[agent_idx] = float(control[0]) * np.array(
                [np.cos(new_states[agent_idx, 2]), np.sin(new_states[agent_idx, 2])]
            )

        previous_planar = next_planar
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
