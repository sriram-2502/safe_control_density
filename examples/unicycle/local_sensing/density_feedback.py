from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from density_utils.controllers import density_feedback_control
from density_utils.density import Obstacle
from density_utils.dynamics import unicycle_step
from density_utils.utils import plot_goal, plot_obstacle, plot_start
from density_utils.utils.timing import TimedBlock


def _angle_wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _triangle_points(center, heading, size):
    c = np.array(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    tip = c + size * 1.3 * forward
    left = c - size * 0.9 * forward + size * 0.6 * right
    right_pt = c - size * 0.9 * forward - size * 0.6 * right
    return np.stack([tip, left, right_pt], axis=0)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    args = parser.parse_args()

    dt = 0.001
    steps = 40000
    alpha = 0.4
    ctrl_multiplier = 4.0
    rad_from_goal = 0.01
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    stop_when_stable = True
    q_lqr = 4.0
    r_lqr = 1.0
    saturation = 4.0
    k_heading = 6.0
    v_max = 2.0
    omega_max = 4.0
    animate = True
    save_animation = args.save_gif
    animation_stride = 50
    animation_fps = 15
    animation_format = "gif"
    animation_path = Path("animations") / f"unicycle_multi_local.{animation_format}"

    # Local sensing settings (safe_control-like)
    cam_range = 1.0
    fov_angle = np.deg2rad(80.0)
    max_sensed = 5
    linger_steps = 1000

    agent_radius = 0.1
    start = np.array([-2.1, -2.1])
    goal = np.array([2.0, 2.0])

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

    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    state = np.array([start[0], start[1], heading0], dtype=float)
    tilde_prev = heading0
    traj = [state.copy()]

    sensed_buffer = {}
    control_time = 0.0
    control_times = []
    controls = []
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    stop_count = 0
    for step in range(steps):
        heading = state[2]
        sensed = detect_sensed_obstacles(
            state[:2], heading, inflated_obstacles, cam_range, fov_angle
        )
        sensed = sensed[:max_sensed]
        for obs in sensed:
            sensed_buffer[id(obs)] = linger_steps
        for obs_id in list(sensed_buffer.keys()):
            sensed_buffer[obs_id] -= 1
            if sensed_buffer[obs_id] <= 0:
                sensed_buffer.pop(obs_id)
        buffered = [obs for obs in inflated_obstacles if id(obs) in sensed_buffer]

        dist = np.linalg.norm(state[:2] - goal)
        with timer:
            u = density_feedback_control(
                state[:2],
                goal,
                alpha,
                buffered,
                ctrl_multiplier=ctrl_multiplier,
                rad_from_goal=rad_from_goal,
                q_lqr=q_lqr,
                r_lqr=r_lqr,
                dt=dt,
                saturation=saturation,
            )
        dt_control = timer.last
        control_time += dt_control
        if log_timing and dist >= rad_from_goal:
            control_times.append(dt_control)
        v = float(np.linalg.norm(u))
        v = min(v, v_max)
        tilde = float(np.arctan2(u[1], u[0]))
        tilde_dot = _angle_wrap(tilde - tilde_prev) / dt
        tilde_prev = tilde
        omega = tilde_dot - k_heading * _angle_wrap(state[2] - tilde)
        omega = float(np.clip(omega, -omega_max, omega_max))
        controls.append([v, omega])
        state = unicycle_step(state, v, omega, dt)
        traj.append(state.copy())
        if stop_when_stable:
            if dist < stop_tol:
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0

        if step % print_interval == 0:
            dist = np.linalg.norm(state[:2] - goal)
            print(
                f"iter={step} dist_to_goal={dist:.3f} sensed={len(sensed)} buffered={len(buffered)}"
            )
        if np.linalg.norm(state[:2] - goal) < rad_from_goal:
            print(f"stopping at iter={step} (within rad_from_goal)")
            break

    traj = np.array(traj)
    controls = np.array(controls, dtype=float)

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    if avg_control < 1.0:
        avg_str = f"{avg_control * 1e3:.1f} ms"
    else:
        avg_str = f"{avg_control:.2f} s"
    print(f"steps={steps_taken} avg_iteration={avg_str}")
    if log_timing:
        mean_ms, std_ms = timer.mean_std_ms()
        if mean_ms is not None:
            print(f"avg_iteration={mean_ms:.3f} [ms] std={std_ms:.3f} [ms]")
        if control_times:
            control_times = np.array(control_times, dtype=float)
            mean_ms = control_times.mean() * 1e3
            std_ms = control_times.std() * 1e3
            print(f"avg_iteration_outside_goal={mean_ms:.3f} [ms] std={std_ms:.3f} [ms]")

    t_state = dt * np.arange(len(traj))
    t_u = dt * np.arange(len(controls))
    fig_ts, axes = plt.subplots(3, 2, figsize=(9, 7))
    axes[0, 0].plot(t_state, traj[:, 0], linewidth=1.8, label="x [m]")
    axes[0, 1].plot(t_state, traj[:, 1], linewidth=1.8, label="y [m]")
    axes[1, 0].plot(t_state, traj[:, 2], linewidth=1.8, label="theta [rad]")
    axes[1, 1].plot(t_u, controls[:, 0], linewidth=1.8, label="v [m/s]")
    axes[2, 0].plot(t_u, controls[:, 1], linewidth=1.8, label="omega [rad/s]")
    axes[2, 1].axis("off")
    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel("time [s]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_start(ax, start)
    plot_goal(ax, goal)
    for obs in obstacles:
        plot_obstacle(
            ax,
            obs.center,
            obs.r1,
            obs.r2,
            p=obs.p,
            scale=obs.scale,
            angle=obs.angle,
            color="0.3",
            fill=True,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Unicycle - Local Sensing Density")
    ax.grid(True, linestyle="--", alpha=0.4)

    if animate:
        line, = ax.plot([], [], color="tab:blue", linewidth=2)
        agent = patches.Polygon(
            _triangle_points(traj[0, :2], traj[0, 2], agent_radius),
            closed=True,
            facecolor="tab:blue",
            edgecolor="k",
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(agent)
        fov_poly = patches.Polygon(
            np.zeros((3, 2)),
            closed=True,
            edgecolor="darkorange",
            facecolor="gold",
            linestyle="--",
            linewidth=2.5,
            alpha=0.25,
            zorder=6,
        )
        ax.add_patch(fov_poly)
        boundary_points = [sample_obstacle_boundary(obs) for obs in obstacles]
        sensed_edges = []
        for _ in obstacles:
            edge_line, = ax.plot([], [], color="tab:orange", linewidth=1.5, zorder=3)
            sensed_edges.append(edge_line)

        def init():
            line.set_data([], [])
            heading_angle = traj[0, 2]
            fov_left, fov_right = calculate_fov_points(
                traj[0, :2], heading_angle, fov_angle, cam_range
            )
            fov_pts = np.array(
                [[traj[0, 0], traj[0, 1]], fov_left, fov_right], dtype=float
            )
            fov_poly.set_xy(fov_pts)
            for edge_line in sensed_edges:
                edge_line.set_data([], [])
            return line, agent, fov_poly, *sensed_edges

        def update(i):
            line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            agent.set_xy(_triangle_points(traj[i, :2], traj[i, 2], agent_radius))
            heading_angle = traj[i, 2]
            fov_left, fov_right = calculate_fov_points(
                traj[i, :2], heading_angle, fov_angle, cam_range
            )
            fov_pts = np.array(
                [[traj[i, 0], traj[i, 1]], fov_left, fov_right], dtype=float
            )
            fov_poly.set_xy(fov_pts)
            sensed = detect_sensed_obstacles(
                traj[i, :2], heading_angle, inflated_obstacles, cam_range, fov_angle
            )
            sensed = sensed[:max_sensed]
            sensed_ids = {id(obs) for obs in sensed}
            for idx, edge_line in enumerate(sensed_edges):
                pts = boundary_points[idx]
                rel = pts - traj[i, :2]
                dists = np.linalg.norm(rel, axis=1)
                angles = np.arctan2(rel[:, 1], rel[:, 0])
                ang_diff = np.abs(_angle_wrap(angles - heading_angle))
                mask = (dists <= cam_range) & (ang_diff <= fov_angle / 2.0)
                if id(inflated_obstacles[idx]) in sensed_ids and np.any(mask):
                    edge_line.set_data(pts[mask, 0], pts[mask, 1])
                else:
                    edge_line.set_data([], [])
            return line, agent, fov_poly, *sensed_edges

        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=range(0, len(traj), animation_stride),
            interval=15,
            blit=True,
            repeat=False,
        )
        if save_animation:
            animation_path.parent.mkdir(parents=True, exist_ok=True)
            ani.save(animation_path, writer=animation.PillowWriter(fps=animation_fps))
    else:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()








