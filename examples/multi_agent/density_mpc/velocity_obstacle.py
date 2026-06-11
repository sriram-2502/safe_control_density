import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
MULTI_AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(MULTI_AGENT_ROOT), str(REPO_ROOT), str(EXAMPLE_ROOT / "unicycle"), str(MULTI_AGENT_ROOT / "density_feedback")]

from _config import add_common_arguments, finalize_args, make_scenario, min_pair_clearance, mode_label, wrap_angle
from _plotting import animation_save_paths, wants_animation_output, _plot_multi_agent_results

from density_utils.controllers import (
    SOLVER_CHOICES,
    single_integrator_nominal_control,
    solve_density_mpc,
    solve_discrete_density_filter,
)
from density_utils.density import Obstacle, p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock


INTERACTION_MODE = "density_mpc"
NOMINAL_MODE = "velocity_obstacle"


def _load_density_feedback_module(name):
    path = MULTI_AGENT_ROOT / "density_feedback" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"multi_agent_density_feedback_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COLLISION_CONE = None
_VELOCITY_OBSTACLE = None


def _agent_obstacle_schedule(
    states,
    planar_snapshot,
    agent_r1,
    agent_r2,
    agent_idx,
    max_neighbors,
    safety_margin,
    horizon,
    dt,
):
    other_indices = np.delete(np.arange(states.shape[0]), agent_idx)
    if max_neighbors > 0 and len(other_indices) > max_neighbors:
        rel = states[other_indices, :2] - states[agent_idx, :2]
        other_indices = other_indices[np.argsort(np.linalg.norm(rel, axis=1))[:max_neighbors]]
    schedule = []
    for k in range(int(horizon) + 1):
        centers = states[other_indices, :2] + float(k) * float(dt) * planar_snapshot[other_indices]
        schedule.append(
            [
                Obstacle(
                    center=centers[row_idx],
                    r1=agent_r1[agent_idx] + agent_r1[other_idx] + safety_margin,
                    r2=agent_r1[agent_idx] + agent_r2[other_idx] + safety_margin,
                    p=2.0,
                )
                for row_idx, other_idx in enumerate(other_indices)
            ]
        )
    return schedule, other_indices


def _shift_controls(controls):
    if controls is None:
        return None
    shifted = np.asarray(controls, dtype=float).copy()
    if shifted.shape[0] > 1:
        shifted[:-1] = shifted[1:]
        shifted[-1] = shifted[-2]
    return shifted


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


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _rotate(vec, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)


def _vo_safe_margin(ego_pos, ego_vel, other_pos, other_vel, collision_radius):
    p_rel = np.asarray(other_pos, dtype=float) - np.asarray(ego_pos, dtype=float)
    p_norm = float(np.linalg.norm(p_rel))
    if p_norm <= collision_radius:
        return -np.inf
    axis = p_rel / p_norm
    half_angle = np.arcsin(np.clip(collision_radius / p_norm, 0.0, 1.0))
    left_boundary = _rotate(axis, half_angle)
    right_boundary = _rotate(axis, -half_angle)
    v_rel = np.asarray(ego_vel, dtype=float) - np.asarray(other_vel, dtype=float)
    return max(_cross2(left_boundary, v_rel), _cross2(v_rel, right_boundary))


def _velocity_aware_pose_density(
    state,
    goal_state,
    alpha,
    obstacles,
    *,
    other_velocities,
    ego_speed,
    mode,
    margin,
    theta_weight=0.05,
    min_v=1e-6,
):
    err = _pose_error(state, goal_state)
    lyap = max(float(err[0] ** 2 + err[1] ** 2 + theta_weight * err[2] ** 2), min_v)
    ego_vel = float(ego_speed) * np.array([np.cos(state[2]), np.sin(state[2])], dtype=float)
    phi = 1.0
    for obs, other_vel in zip(obstacles, other_velocities):
        spatial = p_norm_bump(state[:2], obs.center, obs.r1, obs.r2, p=obs.p, scale=obs.scale, angle=obs.angle)
        if mode == "collision_cone":
            value = _collision_cone_h(state[:2], ego_vel, obs.center, other_vel, obs.r1)
        elif mode == "velocity_obstacle":
            value = _vo_safe_margin(state[:2], ego_vel, obs.center, other_vel, obs.r1)
        else:
            raise ValueError(f"unknown velocity-aware density mode: {mode}")
        phi *= spatial * _smooth_scalar_bump(value, -margin, margin)
    return phi / (lyap ** float(alpha))


def _mpc_density_fn(nominal_mode, other_velocities, ego_speed, args):
    if nominal_mode == "reactive":
        return _pose_density

    def density_fn(state_eval, goal_eval, alpha_eval, obstacles_eval):
        return _velocity_aware_pose_density(
            state_eval,
            goal_eval,
            alpha_eval,
            obstacles_eval,
            other_velocities=other_velocities,
            ego_speed=max(float(ego_speed), 0.2),
            mode=nominal_mode,
            margin=args.cone_density_margin,
        )

    return density_fn


def _nominal_control(mode, pos, goal, agent_idx, other_positions, other_velocities, ego_r1, other_r1, other_r2, obstacles, previous_planar, args):
    global _COLLISION_CONE, _VELOCITY_OBSTACLE
    if mode == "reactive":
        return single_integrator_nominal_control(
            pos,
            goal,
            args.alpha,
            obstacles,
            mode="density_blend",
            ctrl_multiplier=args.ctrl_multiplier,
            rad_from_goal=args.rad_from_goal,
            q_lqr=args.q_lqr,
            r_lqr=args.r_lqr,
            dt=args.mpc_dt,
            u_min=[-args.v_max, -args.v_max],
            u_max=[args.v_max, args.v_max],
        )
    if mode == "collision_cone":
        if _COLLISION_CONE is None:
            _COLLISION_CONE = _load_density_feedback_module("collision_cone")
        return _COLLISION_CONE._collision_cone_density_control(
            ego_pos=pos,
            goal=goal,
            ego_vel=previous_planar[agent_idx],
            other_positions=other_positions,
            other_velocities=other_velocities,
            ego_r1=ego_r1,
            other_r1=other_r1,
            other_r2=other_r2,
            alpha=args.alpha,
            cone_margin=args.cone_density_margin,
            ctrl_multiplier=args.ctrl_multiplier,
            rad_from_goal=args.rad_from_goal,
            q_lqr=args.q_lqr,
            r_lqr=args.r_lqr,
            dt=args.mpc_dt,
            saturation=args.v_max,
        )
    if mode == "velocity_obstacle":
        if _VELOCITY_OBSTACLE is None:
            _VELOCITY_OBSTACLE = _load_density_feedback_module("velocity_obstacle")
        extra_radius, close_margin, projection_passes = _VELOCITY_OBSTACLE.VO_SCENARIO_TUNING.get(
            args.scenario,
            (
                _VELOCITY_OBSTACLE.VO_DEFAULT_EXTRA_RADIUS,
                _VELOCITY_OBSTACLE.VO_DEFAULT_CLOSE_MARGIN,
                _VELOCITY_OBSTACLE.VO_DEFAULT_PROJECTION_PASSES,
            ),
        )
        return _VELOCITY_OBSTACLE._velocity_obstacle_density_control(
            ego_pos=pos,
            goal=goal,
            agent_idx=agent_idx,
            other_positions=other_positions,
            other_velocities=other_velocities,
            ego_r1=ego_r1,
            other_r1=other_r1,
            other_r2=other_r2,
            alpha=args.alpha,
            vo_margin=args.cone_density_margin,
            ctrl_multiplier=args.ctrl_multiplier,
            rad_from_goal=args.rad_from_goal,
            q_lqr=args.q_lqr,
            r_lqr=args.r_lqr,
            dt=args.mpc_dt,
            saturation=args.v_max,
            extra_radius=extra_radius,
            close_margin=close_margin,
            projection_passes=projection_passes,
        )
    raise ValueError(f"unknown nominal mode: {mode}")


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


def _nominal_sequence(mode, state, goal, agent_idx, other_positions, other_velocities, ego_r1, other_r1, other_r2, obstacles, previous_planar, horizon, dt, args):
    state_nom = np.asarray(state, dtype=float).copy()
    seq = []
    for _ in range(horizon):
        u = _nominal_control(
            mode,
            state_nom[:2],
            goal,
            agent_idx,
            other_positions,
            other_velocities,
            ego_r1,
            other_r1,
            other_r2,
            obstacles,
            previous_planar,
            args,
        )
        unicycle_u = _unicycle_nominal_from_planar_ref(
            state_nom, u, args.v_max, args.omega_max, args.k_heading
        )
        seq.append(unicycle_u)
        state_nom = unicycle_step(state_nom, float(unicycle_u[0]), float(unicycle_u[1]), dt)
        state_nom[2] = wrap_angle(state_nom[2])
    return np.asarray(seq, dtype=float)


def _unicycle_mpc_step(state, control, dt):
    return unicycle_step(state, float(control[0]), float(control[1]), dt)


def _min_obstacle_clearance(pos, obstacles):
    if not obstacles:
        return np.inf
    clearances = [float(np.linalg.norm(np.asarray(pos, dtype=float) - obs.center) - obs.r1) for obs in obstacles]
    return min(clearances)


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto")
    parser.add_argument("--mpc-neighbors", type=int, default=1)
    parser.add_argument("--mpc-safety-margin", type=float, default=0.0)
    parser.add_argument("--mpc-dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--control-period", type=int, default=1)
    parser.add_argument("--slack-l1-weight", type=float, default=1.0)
    parser.add_argument("--control-weight", type=float, default=0.05)
    parser.add_argument("--control-rate-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=30.0)
    parser.add_argument("--heading-state-weight", type=float, default=10.0)
    parser.add_argument("--terminal-weight", type=float, default=500.0)
    parser.add_argument("--q-lqr", type=float, default=4.0)
    parser.add_argument("--r-lqr", type=float, default=1.0)
    parser.add_argument("--polish-slack-weight", type=float, default=1e4)
    parser.add_argument("--polish-control-weight", type=float, default=5.0)
    parser.add_argument("--polish-activation-margin", type=float, default=0.25)
    parser.add_argument("--verbose", action="store_true", help="Print per-interval MPC solver diagnostics.")
    args = finalize_args(parser.parse_args(argv))

    if args.scenario == "crossing2":
        args.mpc_safety_margin = max(args.mpc_safety_margin, 0.24)
        args.cone_density_margin = max(args.cone_density_margin, 0.75)
        args.polish_activation_margin = max(args.polish_activation_margin, 0.40)
        args.slack_l1_weight = max(args.slack_l1_weight, 200.0)
        args.control_weight = max(args.control_weight, 0.12)
        args.terminal_weight = max(args.terminal_weight, 1000.0)

    nominal_mode = NOMINAL_MODE
    interaction_mode = f"{INTERACTION_MODE}_{nominal_mode}"
    starts, goals, headings, agent_r1, agent_r2 = make_scenario(args, interaction_mode)
    states = np.hstack([starts, headings[:, None]])
    goal_states = np.hstack([goals, headings[:, None]])
    num_agents = states.shape[0]
    previous_planar = np.zeros((num_agents, 2), dtype=float)
    previous_control = np.zeros((num_agents, 2), dtype=float)
    previous_sequences = [None for _ in range(num_agents)]
    u_min = np.array([0.0, -args.omega_max], dtype=float)
    u_max = np.array([args.v_max, args.omega_max], dtype=float)

    animation_base_path = (
        Path(__file__).resolve().parents[1]
        / "animations"
        / "density_mpc"
        / f"multi_agent_unicycle_{args.scenario}_{interaction_mode}.gif"
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
        do_print = args.print_interval > 0 and step % args.print_interval == 0

        new_states = states.copy()
        planar_snapshot = previous_planar.copy()
        next_planar = previous_planar.copy()
        solve_this_step = step % max(1, args.control_period) == 0
        step_solve_time = 0.0
        step_failures = 0
        step_max_slack = 0.0
        step_polished = 0
        for agent_idx in range(num_agents):
            if solve_this_step:
                obstacle_schedule, other_indices = _agent_obstacle_schedule(
                    states,
                    planar_snapshot,
                    agent_r1,
                    agent_r2,
                    agent_idx,
                    args.mpc_neighbors,
                    args.mpc_safety_margin,
                    args.horizon,
                    args.mpc_dt,
                )
                obstacles = obstacle_schedule[0]
                nominal_seq = _nominal_sequence(
                    nominal_mode,
                    states[agent_idx],
                    goals[agent_idx],
                    agent_idx,
                    states[other_indices, :2],
                    planar_snapshot[other_indices],
                    agent_r1[agent_idx],
                    agent_r1[other_indices],
                    agent_r2[other_indices],
                    obstacles,
                    planar_snapshot,
                    args.horizon,
                    args.mpc_dt,
                    args,
                )
                density_fn = _mpc_density_fn(
                    nominal_mode,
                    planar_snapshot[other_indices].copy(),
                    max(float(nominal_seq[0, 0]), float(previous_control[agent_idx, 0]), 0.2),
                    args,
                )
                initial_controls = _shift_controls(previous_sequences[agent_idx])
                if initial_controls is None:
                    initial_controls = nominal_seq
                with timers[agent_idx]:
                    result = solve_density_mpc(
                        states[agent_idx],
                        goal_states[agent_idx],
                        args.alpha,
                        obstacle_schedule,
                        solver=args.solver,
                        u_nom=nominal_seq,
                        horizon=args.horizon,
                        dt=args.mpc_dt,
                        next_state_fn=_unicycle_mpc_step,
                        u_min=u_min,
                        u_max=u_max,
                        divergence=0.0,
                        slack_weight=0.0,
                        slack_l1_weight=args.slack_l1_weight,
                        control_weight=args.control_weight,
                        control_rate_weight=args.control_rate_weight,
                        previous_control=previous_control[agent_idx],
                        state_weight=np.diag([args.state_weight, args.state_weight, args.heading_state_weight]),
                        terminal_weight=args.terminal_weight,
                        density_fn=density_fn,
                        initial_controls=initial_controls,
                        return_info=True,
                    )
                control_time[agent_idx] += timers[agent_idx].last
                step_solve_time += timers[agent_idx].last
                candidate_next = _unicycle_mpc_step(states[agent_idx], result.u, args.mpc_dt)
                polish_obstacles = obstacle_schedule[min(1, len(obstacle_schedule) - 1)]
                needs_polish = _min_obstacle_clearance(candidate_next[:2], polish_obstacles) < args.polish_activation_margin
                polish = None
                if needs_polish:
                    polish_start = time.perf_counter()
                    polish = solve_discrete_density_filter(
                        states[agent_idx],
                        goal_states[agent_idx],
                        args.alpha,
                        polish_obstacles,
                        u_nom=result.u,
                        dt=args.mpc_dt,
                        next_state_fn=_unicycle_mpc_step,
                        u_min=u_min,
                        u_max=u_max,
                        divergence=0.0,
                        slack_weight=args.polish_slack_weight,
                        control_weight=args.polish_control_weight,
                        density_fn=density_fn,
                        solver=args.solver,
                        return_info=True,
                    )
                    polish_elapsed = time.perf_counter() - polish_start
                    control_time[agent_idx] += polish_elapsed
                    step_solve_time += polish_elapsed
                    step_polished += 1
                    previous_control[agent_idx] = polish.u
                else:
                    previous_control[agent_idx] = result.u
                result_controls = result.controls.copy()
                result_controls[0] = previous_control[agent_idx]
                previous_sequences[agent_idx] = result_controls
                max_result_slack = float(np.max(result.slack)) if result.slack.size else 0.0
                max_polish_slack = float(np.max(polish.slack)) if polish is not None and polish.slack.size else 0.0
                step_max_slack = max(step_max_slack, max_result_slack, max_polish_slack)
                max_slack = max(max_slack, max_result_slack, max_polish_slack)
                if not result.success or (polish is not None and not polish.success):
                    solver_failures += 1
                    step_failures += 1

            v = float(previous_control[agent_idx, 0])
            omega = float(previous_control[agent_idx, 1])
            new_states[agent_idx] = unicycle_step(states[agent_idx], v, omega, args.dt)
            new_states[agent_idx, 2] = wrap_angle(new_states[agent_idx, 2])
            next_planar[agent_idx] = v * np.array(
                [np.cos(new_states[agent_idx, 2]), np.sin(new_states[agent_idx, 2])],
                dtype=float,
            )

        previous_planar = next_planar
        states = new_states
        min_clearance = min(min_clearance, min_pair_clearance(states, agent_r1))
        if do_print:
            current_clearance = min_pair_clearance(states, agent_r1)
            parts = [
                f"iter={step}",
                "dists=" + np.array2string(dists, precision=3),
                f"solve_ms={step_solve_time * 1e3:.3f}",
                f"clearance={current_clearance:.3f}",
                f"slack={step_max_slack:.2e}",
            ]
            if args.verbose:
                parts.extend(
                    [
                        f"failures={solver_failures}",
                        f"step_failures={step_failures}",
                        f"polished={step_polished}",
                    ]
                )
            print(" ".join(parts))
        if want_plot_data and ((step + 1) % animation_stride == 0):
            stored_traj.append(states.copy())

    final_dists = np.linalg.norm(states[:, :2] - goals, axis=1)
    if min_clearance < 0.0:
        status = "collision"
    elif np.all(final_dists < args.rad_from_goal):
        status = "success"
    else:
        status = "timeout"
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
            title=f"Multi-Agent Unicycle - {mode_label(interaction_mode)}",
            save_paths=save_paths,
            show_plot=not args.no_plot,
            fps=args.animation_fps,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )


if __name__ == "__main__":
    main()
