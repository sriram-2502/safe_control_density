from pathlib import Path
import argparse
import sys

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import SOLVER_CHOICES, single_integrator_nominal_control, solve_discrete_density_filter
from density_utils.density import Obstacle, p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock

from _plotting import add_animation_save_args, animation_save_paths, plot_unicycle_results, wants_animation_output
from config import CONFIG


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _p_norm_distance(x, obs):
    dx = x - obs.center
    if obs.angle:
        c = np.cos(-obs.angle)
        s = np.sin(-obs.angle)
        dx = np.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]])
    if obs.scale is not None:
        dx = dx / obs.scale
    return np.sum(np.abs(dx) ** obs.p) ** (1.0 / obs.p)


def _as_array(value):
    return np.asarray(value, dtype=float)


def _obstacle_from_config(config):
    angle = config.get("angle", 0.0)
    if "angle_deg" in config:
        angle = np.deg2rad(config["angle_deg"])
    scale = config.get("scale")
    return Obstacle(
        center=_as_array(config["center"]),
        r1=float(config["r1"]),
        r2=float(config["r2"]),
        p=float(config.get("p", 2.0)),
        scale=None if scale is None else _as_array(scale),
        angle=float(angle),
    )


def _inflate_obstacles(obstacles, agent_radius):
    return [
        Obstacle(
            center=obs.center,
            r1=obs.r1 + agent_radius,
            r2=obs.r2 + agent_radius,
            p=obs.p,
            scale=obs.scale,
            angle=obs.angle,
        )
        for obs in obstacles
    ]


def _control_bounds(control_cfg):
    v_max = float(control_cfg["v_max"])
    omega_max = float(control_cfg["omega_max"])
    return np.array([0.0, -omega_max]), np.array([v_max, omega_max])


def _pose_error(state, goal_state):
    return np.array(
        [
            state[0] - goal_state[0],
            state[1] - goal_state[1],
            _wrap_angle(state[2] - goal_state[2]),
        ],
        dtype=float,
    )


def _pose_density(state, goal_state, alpha, obstacles, theta_weight=0.05, min_v=1e-6):
    err = _pose_error(state, goal_state)
    lyap = max(float(err[0] ** 2 + err[1] ** 2 + theta_weight * err[2] ** 2), min_v)
    phi = 1.0
    for obs in obstacles:
        phi *= p_norm_bump(
            state[:2],
            obs.center,
            obs.r1,
            obs.r2,
            p=obs.p,
            scale=obs.scale,
            angle=obs.angle,
        )
    return phi / (lyap ** float(alpha))


def _unicycle_filter_step(state, control, dt):
    return unicycle_step(state, float(control[0]), float(control[1]), dt)


def _unicycle_nominal_from_planar_ref(state, planar_ref, v_max, omega_max, k_heading):
    speed = float(np.linalg.norm(planar_ref))
    if speed < 1e-10:
        return np.zeros(2, dtype=float)

    desired_heading = float(np.arctan2(planar_ref[1], planar_ref[0]))
    heading_error = _wrap_angle(desired_heading - state[2])
    turn_gate = max(0.0, np.cos(heading_error))
    v_nom = min(speed, float(v_max)) * turn_gate
    omega_nom = float(np.clip(k_heading * heading_error, -omega_max, omega_max))
    return np.array([v_nom, omega_nom], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    add_animation_save_args(parser)
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    parser.add_argument("--steps", type=int, default=None, help="Override maximum simulation steps.")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto", help="Optimizer backend.")
    parser.add_argument("--verbose", action="store_true", help="Print solver failure diagnostics.")
    args = parser.parse_args()

    cfg = CONFIG
    sim_cfg = cfg["simulation"]
    scenario_cfg = cfg["scenario"]
    density_cfg = cfg["density"]
    control_cfg = cfg["control"]
    animation_cfg = cfg["animation"]

    dt = float(sim_cfg["dt"])
    density_dt = float(sim_cfg["density_dt"])
    steps = args.steps if args.steps is not None else int(sim_cfg["steps"])
    alpha = float(density_cfg["alpha"])
    ctrl_multiplier = float(density_cfg["ctrl_multiplier"])
    rad_from_goal = float(density_cfg["rad_from_goal"])
    stop_tol = float(sim_cfg["stop_tol"])
    stop_steps = int(sim_cfg["stop_steps"])
    stop_when_stable = bool(sim_cfg.get("stop_when_stable", True))
    q_lqr = float(density_cfg["q_lqr"])
    r_lqr = float(density_cfg["r_lqr"])
    v_max = float(control_cfg["v_max"])
    omega_max = float(control_cfg["omega_max"])
    k_heading = float(control_cfg["k_heading"])
    u_min, u_max = _control_bounds(control_cfg)
    slack_weight = float(density_cfg["slack_weight"])
    animate = not args.no_plot or wants_animation_output(args)
    animation_stride = int(animation_cfg["stride"])
    animation_fps = int(animation_cfg["fps"])
    animation_path = EXAMPLE_ROOT / animation_cfg["path"]
    animation_paths = animation_save_paths(animation_path, save_gif=args.save_gif, save_mp4=args.save_mp4)

    agent_radius = float(scenario_cfg["agent_radius"])
    start = _as_array(scenario_cfg["start"])
    goal = _as_array(scenario_cfg["goal"])
    obstacles = [_obstacle_from_config(obs_cfg) for obs_cfg in scenario_cfg["obstacles"]]
    inflated_obstacles = _inflate_obstacles(obstacles, agent_radius)
    obstacle = obstacles[0]
    inflated_obstacle = inflated_obstacles[0]

    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    state = np.array([start[0], start[1], heading0], dtype=float)
    goal_state = np.array([goal[0], goal[1], heading0], dtype=float)
    traj = [state.copy()]
    controls = []
    slacks = []
    solver_failures = 0
    min_clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1

    control_time = 0.0
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = int(sim_cfg["print_interval"])
    stop_count = 0

    def density_fn(state_eval, goal_eval, alpha_eval, obstacles_eval):
        return _pose_density(state_eval, goal_eval, alpha_eval, obstacles_eval)

    for step in range(steps):
        pos = state[:2]
        dist = np.linalg.norm(pos - goal)
        with timer:
            if dist < stop_tol:
                omega_nom = np.clip(
                    k_heading * _wrap_angle(goal_state[2] - state[2]),
                    -omega_max,
                    omega_max,
                )
                u_nom = np.array([0.0, omega_nom], dtype=float)
            else:
                planar_ref = single_integrator_nominal_control(
                    pos,
                    goal,
                    alpha,
                    [inflated_obstacle],
                    mode="density_blend",
                    ctrl_multiplier=ctrl_multiplier,
                    rad_from_goal=rad_from_goal,
                    q_lqr=q_lqr,
                    r_lqr=r_lqr,
                    dt=density_dt,
                    u_min=[-v_max, -v_max],
                    u_max=[v_max, v_max],
                )
                u_nom = _unicycle_nominal_from_planar_ref(
                    state,
                    planar_ref,
                    v_max,
                    omega_max,
                    k_heading,
                )
            filter_result = solve_discrete_density_filter(
                state,
                goal_state,
                alpha,
                [inflated_obstacle],
                u_nom=u_nom,
                dt=density_dt,
                next_state_fn=_unicycle_filter_step,
                u_min=u_min,
                u_max=u_max,
                divergence=0.0,
                slack_weight=slack_weight,
                density_fn=density_fn,
                solver=args.solver,
                return_info=True,
            )
            control = filter_result.u
        control_time += timer.last
        controls.append(control.copy())
        slacks.append(float(np.max(filter_result.slack)) if filter_result.slack.size else 0.0)
        if not filter_result.success:
            solver_failures += 1

        state = unicycle_step(state, float(control[0]), float(control[1]), dt)
        state[2] = _wrap_angle(state[2])
        traj.append(state.copy())
        clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1
        min_clearance = min(min_clearance, clearance)

        heading_error = abs(_wrap_angle(state[2] - goal_state[2]))
        if stop_when_stable:
            if dist < stop_tol and heading_error < np.deg2rad(5.0):
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0
        if step % print_interval == 0:
            print(
                f"iter={step} dist_to_goal={dist:.3f} heading_error={heading_error:.3f} "
                f"clearance={clearance:.3f} slack={slacks[-1]:.2e}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2, dtype=float))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    traj = np.array(traj)
    controls = np.array(controls)
    slacks = np.array(slacks)

    def _format_duration(seconds):
        if seconds < 1.0:
            return f"{seconds * 1e3:.1f} ms"
        return f"{seconds:.2f} s"

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    summary = (
        "steps="
        f"{steps_taken} "
        f"sim_time={_format_duration(control_time)} "
        f"avg_iteration={_format_duration(avg_control)} "
        f"min_clearance={min_clearance:.4f} "
        f"max_slack={np.max(slacks):.2e}"
    )
    if args.verbose:
        summary += f" solver_failures={solver_failures}"
    print(summary)

    if not args.no_plot or wants_animation_output(args):
        plot_unicycle_results(
            traj=traj,
            controls=controls,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=[obstacle],
            agent_radius=agent_radius,
            title="Unicycle - Static Obstacle (Density filter)",
            animate=animate,
            save_animation=False,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
            slacks=slacks,
            animation_paths=animation_paths,
            show_plot=not args.no_plot,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )


if __name__ == "__main__":
    main()
