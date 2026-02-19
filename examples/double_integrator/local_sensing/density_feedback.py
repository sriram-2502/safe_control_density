from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib import animation, patches

from density_utils.density import Obstacle, density_grad
from density_utils.utils import plot_goal, plot_obstacle, plot_start
from density_utils.utils.timing import TimedBlock


def _angle_wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _heading_from_velocity(vel, fallback):
    speed = np.linalg.norm(vel)
    if speed < 1e-6:
        return fallback
    return float(np.arctan2(vel[1], vel[0]))


def _lqr_gain(dt, q_lqr, r_lqr):
    if np.isscalar(q_lqr):
        q = float(q_lqr) * np.eye(4)
    else:
        q = np.asarray(q_lqr, dtype=float)
    if np.isscalar(r_lqr):
        r = float(r_lqr) * np.eye(2)
    else:
        r = np.asarray(r_lqr, dtype=float)
    if q.shape != (4, 4) or r.shape != (2, 2):
        raise ValueError("q_lqr must be (4,4) and r_lqr must be (2,2) for double integrator")
    a = np.block([[np.eye(2), dt * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    b = np.block([[np.zeros((2, 2))], [dt * np.eye(2)]])
    p = la.solve_discrete_are(a, b, q, r)
    bt_p = b.T @ p
    return np.linalg.solve(bt_p @ b + r, bt_p @ a)


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
    steps = 50000
    alpha = 0.2
    ctrl_multiplier = 2.0
    rad_from_goal = 1.0
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    stop_when_stable = True
    q_lqr = np.diag([10.0, 10.0, 10.0, 10.0])
    r_lqr = 1.0
    saturation = 2.0
    k_backstep = 4.0
    animate = True
    save_animation = args.save_gif
    animation_stride = 50
    animation_fps = 15
    animation_format = "gif"
    animation_path = Path("animations") / f"double_integrator_multi_local.{animation_format}"

    # Local sensing settings (safe_control-like)
    cam_range = 1.0
    fov_angle = np.deg2rad(80.0)
    max_sensed = 5
    linger_steps = 8000

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

    def _scaled_saturation(dist):
        if dist >= rad_from_goal:
            return saturation
        decay_length = max(rad_from_goal / 3.0, 1e-6)
        scale = 1.0 - np.exp(-dist / decay_length)
        return saturation * scale

    def backstepping_control(pos, vel, prev_v_des, obstacles_local, sat):
        grad = density_grad(pos, goal, alpha, obstacles_local)
        v_des = ctrl_multiplier * grad
        v_des_dot = (v_des - prev_v_des) / dt
        u = v_des_dot - k_backstep * (vel - v_des)
        max_u = np.max(np.abs(u))
        if max_u > sat:
            u = u / max_u * sat
        return u, v_des

    k_lqr = _lqr_gain(dt, q_lqr, r_lqr)
    goal_state = np.array([goal[0], goal[1], 0.0, 0.0], dtype=float)
    state = np.array([start[0], start[1], 0.0, 0.0], dtype=float)
    traj = [state.copy()]

    controls = []
    sensed_buffer = {}
    control_time = 0.0
    control_times = []
    v_des_prev = np.zeros(2, dtype=float)
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    lqr_triggered = False
    stop_count = 0
    for step in range(steps):
        dist = np.linalg.norm(state[:2] - goal)
        sat = _scaled_saturation(dist)
        goal_heading = np.arctan2(goal[1] - state[1], goal[0] - state[0])
        heading = _heading_from_velocity(state[2:], goal_heading)
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

        with timer:
            if dist < rad_from_goal:
                if not lqr_triggered:
                    lqr_triggered = True
                u = -k_lqr @ (state - goal_state)
                max_u = np.max(np.abs(u))
                if max_u > sat:
                    u = u / max_u * sat
            else:
                u, v_des_prev = backstepping_control(state[:2], state[2:], v_des_prev, buffered, sat)
        dt_control = timer.last
        control_time += dt_control
        if log_timing and dist >= rad_from_goal:
            control_times.append(dt_control)
        controls.append(u.copy())
        state[:2] = state[:2] + state[2:] * dt
        state[2:] = state[2:] + u * dt
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

    traj = np.array(traj)
    controls = np.array(controls)

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
    axes[1, 0].plot(t_state, traj[:, 2], linewidth=1.8, label="v_x [m/s]")
    axes[1, 1].plot(t_state, traj[:, 3], linewidth=1.8, label="v_y [m/s]")
    axes[2, 0].plot(t_u, controls[:, 0], linewidth=1.8, label="u_x [m/s^2]")
    axes[2, 1].plot(t_u, controls[:, 1], linewidth=1.8, label="u_y [m/s^2]")
    for ax in axes.ravel():
        ax.set_xlabel("time [s]")
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax.has_data():
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
    ax.set_title("Double Integrator - Local Sensing Density")
    ax.grid(True, linestyle="--", alpha=0.4)

    if animate:
        line, = ax.plot([], [], color="tab:blue", linewidth=2)
        agent = patches.Circle(traj[0], agent_radius, color="tab:blue", zorder=4)
        ax.add_patch(agent)
        heading = ax.quiver(
            traj[0, 0],
            traj[0, 1],
            0.0,
            0.0,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="tab:orange",
            width=0.006,
            zorder=5,
        )
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
            goal_heading = np.arctan2(goal[1] - traj[0, 1], goal[0] - traj[0, 0])
            heading_angle = _heading_from_velocity(traj[0, 2:], goal_heading)
            speed = np.linalg.norm(traj[0, 2:])
            if speed < 1e-6:
                ux, uy = 0.0, 0.0
            else:
                heading_len = 0.35
                ux, uy = traj[0, 2:] / speed * heading_len
            heading.set_offsets([traj[0, 0], traj[0, 1]])
            heading.set_UVC([ux], [uy])
            fov_left, fov_right = calculate_fov_points(
                traj[0, :2], heading_angle, fov_angle, cam_range
            )
            fov_pts = np.array(
                [[traj[0, 0], traj[0, 1]], fov_left, fov_right], dtype=float
            )
            fov_poly.set_xy(fov_pts)
            for edge_line in sensed_edges:
                edge_line.set_data([], [])
            return line, agent, heading, fov_poly, *sensed_edges

        def update(i):
            line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            agent.center = (traj[i, 0], traj[i, 1])
            goal_heading = np.arctan2(goal[1] - traj[i, 1], goal[0] - traj[i, 0])
            heading_angle = _heading_from_velocity(traj[i, 2:], goal_heading)
            speed = np.linalg.norm(traj[i, 2:])
            if speed < 1e-6:
                ux, uy = 0.0, 0.0
            else:
                heading_len = 0.35
                ux, uy = traj[i, 2:] / speed * heading_len
            heading.set_offsets([traj[i, 0], traj[i, 1]])
            heading.set_UVC([ux], [uy])
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
            return line, agent, heading, fov_poly, *sensed_edges

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







