from pathlib import Path
import argparse
import sys

import numpy as np

from density_utils.controllers import single_integrator_nominal_control, solve_discrete_density_qp
from density_utils.density import Obstacle
from density_utils.dynamics import single_integrator_step
from density_utils.sim import forward_euler
from density_utils.utils.timing import TimedBlock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plotting import plot_single_integrator_results


def _angle_wrap(angle):
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


def calculate_fov_points(position, heading, fov_angle, cam_range):
    half_fov = fov_angle / 2.0
    left_angle = heading - half_fov
    right_angle = heading + half_fov
    left_point = (
        position[0] + cam_range * np.cos(left_angle),
        position[1] + cam_range * np.sin(left_angle),
    )
    right_point = (
        position[0] + cam_range * np.cos(right_angle),
        position[1] + cam_range * np.sin(right_angle),
    )
    return left_point, right_point


def detect_sensed_obstacles(pos, heading, obstacles, cam_range, fov_angle):
    """Range + FOV filtering based on robot heading."""
    pos = np.asarray(pos, dtype=float)
    sensed = []
    for obs in obstacles:
        rel = obs.center - pos
        dist = np.linalg.norm(rel)
        if dist > cam_range:
            continue
        if fov_angle < 2.0 * np.pi:
            angle_to_obs = np.arctan2(rel[1], rel[0])
            if abs(_angle_wrap(angle_to_obs - heading)) > fov_angle / 2.0:
                continue
        sensed.append((dist, obs))
    sensed.sort(key=lambda item: item[0])
    return [obs for _, obs in sensed]


def sample_obstacle_boundary(obs, num=120):
    """Sample obstacle boundary for visualization."""
    theta = np.linspace(0.0, 2.0 * np.pi, num=num, endpoint=True)
    c = np.cos(theta)
    s = np.sin(theta)
    p = float(obs.p)
    x = np.sign(c) * (np.abs(c) ** (2.0 / p))
    y = np.sign(s) * (np.abs(s) ** (2.0 / p))
    pts = np.stack([x, y], axis=1) * obs.r1
    if obs.scale is not None:
        pts = pts * np.asarray(obs.scale, dtype=float)[None, :]
    if obs.angle:
        ca = np.cos(obs.angle)
        sa = np.sin(obs.angle)
        rot = np.array([[ca, -sa], [sa, ca]])
        pts = pts @ rot.T
    return pts + obs.center[None, :]


def _make_obstacles(agent_radius):
    big_r1 = 0.5
    small_r1 = 0.2
    sensing_margin = 0.1
    big_r2 = big_r1 + sensing_margin
    small_r2 = small_r1 + sensing_margin

    obstacles = [
        Obstacle(center=np.array([-0.6, 0.6]), r1=big_r1, r2=big_r2, p=4.0, angle=np.deg2rad(30.0)),
        Obstacle(center=np.array([1.0, 0.3]), r1=big_r1, r2=big_r2, p=8.0, angle=np.deg2rad(60.0)),
        Obstacle(center=np.array([-1.8, 1.2]), r1=small_r1, r2=small_r2, p=2.0, scale=np.array([1.6, 0.8])),
        Obstacle(center=np.array([0.0, 1.6]), r1=small_r1, r2=small_r2, p=4.0, angle=np.deg2rad(90.0)),
        Obstacle(center=np.array([-1.4, -0.6]), r1=small_r1, r2=small_r2, p=2.0),
        Obstacle(center=np.array([-1.6, -1.4]), r1=small_r1, r2=small_r2, p=2.0),
        Obstacle(center=np.array([0.4, -1.6]), r1=small_r1, r2=small_r2, p=2.0, scale=np.array([0.7, 1.5])),
        Obstacle(center=np.array([1.8, -1.2]), r1=small_r1, r2=small_r2, p=2.0),
        Obstacle(center=np.array([-0.2, -0.8]), r1=small_r1, r2=small_r2, p=2.0, scale=np.array([1.4, 1.0])),
        Obstacle(center=np.array([1.0, 1.6]), r1=small_r1, r2=small_r2, p=2.0),
    ]

    inflated_obstacles = [
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
    return obstacles, inflated_obstacles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run the simulation without opening plots.")
    parser.add_argument("--steps", type=int, default=12000, help="Maximum simulation steps.")
    parser.add_argument(
        "--u-nom",
        default="density",
        choices=("goal", "lqr", "density", "density_blend", "pure_pursuit"),
        help="Nominal controller used as the Density-QP reference.",
    )
    args = parser.parse_args()

    dt = 0.01
    steps = args.steps
    alpha = 0.4
    ctrl_multiplier = 2.0
    rad_from_goal = 0.35
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 250
    stop_when_stable = True
    q_lqr = 4.0
    r_lqr = 1.0
    u_max = np.array([2.0, 2.0])
    u_min = -u_max
    slack_weight = 1e4
    animate = not args.no_plot
    save_animation = args.save_gif
    animation_stride = 20
    animation_fps = 20
    animation_format = "gif"
    animation_path = Path("animations") / f"single_integrator_multi_local_qp.{animation_format}"

    cam_range = 1.0
    fov_angle = np.deg2rad(80.0)
    max_sensed = 5
    linger_steps = 200

    agent_radius = 0.1
    start = np.array([-2.1, -2.1])
    goal = np.array([2.0, 2.0])
    obstacles, inflated_obstacles = _make_obstacles(agent_radius)

    x = start.copy()
    traj = [x.copy()]
    controls = []
    headings = []
    slacks = []
    sensed_counts = []
    buffered_counts = []
    sensed_buffer = {}
    solver_failures = 0
    min_clearance = min(_p_norm_distance(x, obs) - obs.r1 for obs in inflated_obstacles)

    control_time = 0.0
    control_times = []
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    stop_count = 0
    for step in range(steps):
        heading = np.arctan2(goal[1] - x[1], goal[0] - x[0])
        sensed = detect_sensed_obstacles(x, heading, inflated_obstacles, cam_range, fov_angle)
        sensed = sensed[:max_sensed]
        for obs in sensed:
            sensed_buffer[id(obs)] = linger_steps
        for obs_id in list(sensed_buffer.keys()):
            sensed_buffer[obs_id] -= 1
            if sensed_buffer[obs_id] <= 0:
                sensed_buffer.pop(obs_id)
        buffered = [obs for obs in inflated_obstacles if id(obs) in sensed_buffer]

        dist = np.linalg.norm(x - goal)
        with timer:
            u_nom = single_integrator_nominal_control(
                x,
                goal,
                alpha,
                buffered,
                mode=args.u_nom,
                ctrl_multiplier=ctrl_multiplier,
                rad_from_goal=rad_from_goal,
                q_lqr=q_lqr,
                r_lqr=r_lqr,
                dt=dt,
                u_min=u_min,
                u_max=u_max,
            )
            qp = solve_discrete_density_qp(
                x,
                goal,
                alpha,
                buffered,
                u_nom=u_nom,
                next_state_fn=single_integrator_step,
                dt=dt,
                u_min=u_min,
                u_max=u_max,
                divergence=0.0,
                slack_weight=slack_weight,
                return_info=True,
            )
            u = qp.u
        dt_control = timer.last
        control_time += dt_control
        if log_timing and dist >= rad_from_goal:
            control_times.append(dt_control)
        headings.append(float(np.arctan2(u[1], u[0])) if np.linalg.norm(u) > 1e-6 else heading)
        controls.append(u.copy())
        slacks.append(float(np.max(qp.slack)) if qp.slack.size else 0.0)
        sensed_counts.append(len(sensed))
        buffered_counts.append(len(buffered))
        if not qp.success:
            solver_failures += 1

        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        clearance = min(_p_norm_distance(x, obs) - obs.r1 for obs in inflated_obstacles)
        min_clearance = min(min_clearance, clearance)

        if stop_when_stable:
            if dist < stop_tol:
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0

        if step % print_interval == 0:
            dist = np.linalg.norm(x - goal)
            print(
                f"iter={step} dist_to_goal={dist:.3f} clearance={clearance:.3f} "
                f"sensed={len(sensed)} buffered={len(buffered)} slack={slacks[-1]:.2e}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros_like(traj[0]))
    if len(headings) < len(traj):
        headings.append(headings[-1] if headings else 0.0)
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    traj = np.array(traj)
    controls = np.array(controls)
    headings = np.array(headings)
    slacks = np.array(slacks)

    def _format_duration(seconds):
        if seconds < 1.0:
            return f"{seconds * 1e3:.1f} ms"
        return f"{seconds:.2f} s"

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    print(
        "steps="
        f"{steps_taken} "
        f"sim_time={_format_duration(control_time)} "
        f"avg_iteration={_format_duration(avg_control)} "
        f"min_clearance={min_clearance:.4f} "
        f"max_slack={np.max(slacks):.2e} "
        f"solver_failures={solver_failures}"
    )
    if sensed_counts:
        print(
            f"max_sensed={max(sensed_counts)} "
            f"max_buffered={max(buffered_counts) if buffered_counts else 0}"
        )
    if log_timing:
        mean_ms, std_ms = timer.mean_std_ms()
        if mean_ms is not None:
            print(f"avg_iteration={mean_ms:.3f} [ms] std={std_ms:.3f} [ms]")
        if control_times:
            control_times = np.array(control_times, dtype=float)
            mean_ms = control_times.mean() * 1e3
            std_ms = control_times.std() * 1e3
            print(f"avg_iteration_outside_goal={mean_ms:.3f} [ms] std={std_ms:.3f} [ms]")

    if not args.no_plot:
        plot_single_integrator_results(
            traj=traj,
            controls=controls,
            slacks=slacks,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=obstacles,
            agent_radius=agent_radius,
            title="Single Integrator - Local Sensing Density QP",
            animate=animate,
            save_animation=save_animation,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
            headings=headings,
            inflated_obstacles=inflated_obstacles,
            fov_angle=fov_angle,
            cam_range=cam_range,
            max_sensed=max_sensed,
            animation_interval=15,
        )


if __name__ == "__main__":
    main()
