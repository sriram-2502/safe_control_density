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
from density_utils.density.bump import p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


INTERACTION_MODE = "collision_cone_density_feedback"
PLANAR_COMMAND_SMOOTHING = 0.75
SPEED_RATE_LIMIT = 2.0
OMEGA_RATE_LIMIT = 8.0


def _finite_difference_grad(fn, x, eps=1e-3):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in range(x.size):
        step = np.zeros_like(x)
        step[idx] = eps
        grad[idx] = (fn(x + step) - fn(x - step)) / (2.0 * eps)
    return grad


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


def _collision_cone_density_value(
    ego_pos,
    goal,
    ego_vel,
    other_positions,
    other_velocities,
    ego_r1,
    other_r1,
    other_r2,
    alpha,
    cone_margin,
):
    ego_pos = np.asarray(ego_pos, dtype=float)
    goal = np.asarray(goal, dtype=float)
    dist = max(float(np.linalg.norm(ego_pos - goal)), 1e-3)
    density = 1.0 / (dist ** (2.0 * float(alpha)))
    for other_pos, other_vel, r1, r2 in zip(other_positions, other_velocities, other_r1, other_r2):
        collision_r1 = float(ego_r1 + r1)
        collision_r2 = float(ego_r1 + r2)
        spatial_bump = p_norm_bump(ego_pos, other_pos, collision_r1, collision_r2, p=2.0)
        h_value = _collision_cone_h(ego_pos, ego_vel, other_pos, other_vel, collision_r1)
        cone_bump = _smooth_scalar_bump(h_value, -cone_margin, cone_margin)
        density *= spatial_bump * cone_bump
    return density


def _collision_cone_density_control(
    *,
    ego_pos,
    goal,
    ego_vel,
    other_positions,
    other_velocities,
    ego_r1,
    other_r1,
    other_r2,
    alpha,
    cone_margin,
    ctrl_multiplier,
    rad_from_goal,
    q_lqr,
    r_lqr,
    dt,
    saturation,
):
    if np.linalg.norm(ego_pos - goal) < rad_from_goal:
        return density_feedback_control(
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
    grad = _finite_difference_grad(
        lambda pos: _collision_cone_density_value(
            pos,
            goal,
            ego_vel,
            other_positions,
            other_velocities,
            ego_r1,
            other_r1,
            other_r2,
            alpha,
            cone_margin,
        ),
        ego_pos,
    )
    planar_u = float(ctrl_multiplier) * grad
    max_u = np.max(np.abs(planar_u))
    if max_u > saturation:
        planar_u = planar_u / max_u * saturation
    return planar_u


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = finalize_args(parser.parse_args(argv))

    starts, goals, headings, agent_r1, agent_r2 = make_scenario(args, INTERACTION_MODE)
    states = np.hstack([starts, headings[:, None]])
    tilde_prev = headings.copy()
    num_agents = states.shape[0]
    other_agent_indices = [np.delete(np.arange(num_agents), agent_idx) for agent_idx in range(num_agents)]
    prev_planar_vel = np.column_stack([np.cos(headings), np.sin(headings)]) * 0.2
    prev_planar_cmd = prev_planar_vel.copy()
    prev_control = np.zeros((num_agents, 2), dtype=float)

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
            cone_density_neighbors = 2 if args.scenario == "swap8_opposite" else args.cone_density_neighbors
            if cone_density_neighbors > 0 and len(other_indices) > cone_density_neighbors:
                rel = states[other_indices, :2] - states[agent_idx, :2]
                nearest = np.argsort(np.linalg.norm(rel, axis=1))[:cone_density_neighbors]
                other_indices = other_indices[nearest]

            with timers[agent_idx]:
                planar_u = _collision_cone_density_control(
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
            control_time[agent_idx] += timers[agent_idx].last

            planar_u = (
                (1.0 - PLANAR_COMMAND_SMOOTHING) * planar_u
                + PLANAR_COMMAND_SMOOTHING * prev_planar_cmd[agent_idx]
            )
            prev_planar_cmd[agent_idx] = planar_u

            v = min(float(np.linalg.norm(planar_u)), args.v_max)
            desired_heading = float(np.arctan2(planar_u[1], planar_u[0]))
            tilde_prev[agent_idx] = desired_heading
            omega = args.k_heading * wrap_angle(desired_heading - states[agent_idx, 2])
            omega = float(np.clip(omega, -args.omega_max, args.omega_max))
            v = float(
                np.clip(
                    v,
                    prev_control[agent_idx, 0] - SPEED_RATE_LIMIT * args.dt,
                    prev_control[agent_idx, 0] + SPEED_RATE_LIMIT * args.dt,
                )
            )
            omega = float(
                np.clip(
                    omega,
                    prev_control[agent_idx, 1] - OMEGA_RATE_LIMIT * args.dt,
                    prev_control[agent_idx, 1] + OMEGA_RATE_LIMIT * args.dt,
                )
            )
            prev_control[agent_idx] = np.array([v, omega], dtype=float)
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
