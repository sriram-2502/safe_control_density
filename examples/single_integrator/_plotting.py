from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from density_utils.utils import plot_goal, plot_obstacle, plot_start


def _angle_wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _calculate_fov_points(position, heading, fov_angle, cam_range):
    half_fov = fov_angle / 2.0
    left_angle = heading - half_fov
    right_angle = heading + half_fov
    return (
        (
            position[0] + cam_range * np.cos(left_angle),
            position[1] + cam_range * np.sin(left_angle),
        ),
        (
            position[0] + cam_range * np.cos(right_angle),
            position[1] + cam_range * np.sin(right_angle),
        ),
    )


def _sample_obstacle_boundary(obs, num=120):
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


def _detect_sensed_obstacles(pos, heading, obstacles, cam_range, fov_angle):
    sensed = []
    for obs in obstacles:
        rel = obs.center - pos
        dist = np.linalg.norm(rel)
        if dist > cam_range:
            continue
        angle_to_obs = np.arctan2(rel[1], rel[0])
        if abs(_angle_wrap(angle_to_obs - heading)) > fov_angle / 2.0:
            continue
        sensed.append((dist, obs))
    sensed.sort(key=lambda item: item[0])
    return [obs for _, obs in sensed]


def _plot_time_series(traj, controls, dt, slacks=None):
    rows = 3 if slacks is not None else 2
    fig, axes = plt.subplots(rows, 2, figsize=(8, 7 if rows == 3 else 6))
    axes = np.asarray(axes)
    t_state = dt * np.arange(len(traj))
    t_u = dt * np.arange(len(controls))

    axes[0, 0].plot(t_state, traj[:, 0], linewidth=1.8, label="x [m]")
    axes[0, 1].plot(t_state, traj[:, 1], linewidth=1.8, label="y [m]")
    axes[1, 0].plot(t_u, controls[:, 0], linewidth=1.8, label="u_x [m/s]")
    axes[1, 1].plot(t_u, controls[:, 1], linewidth=1.8, label="u_y [m/s]")
    if slacks is not None:
        axes[2, 0].plot(t_u, slacks, linewidth=1.8, label="slack")
        axes[2, 1].axis("off")

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel("time [s]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")


def plot_single_integrator_results(
    *,
    traj,
    controls,
    dt,
    start,
    goal,
    obstacles,
    agent_radius,
    title,
    animate,
    save_animation,
    animation_path,
    animation_stride,
    animation_fps,
    slacks=None,
    headings=None,
    inflated_obstacles=None,
    fov_angle=None,
    cam_range=None,
    max_sensed=5,
    animation_interval=20,
):
    """Plot common single-integrator state/control traces and plan-view animation."""
    traj = np.asarray(traj, dtype=float)
    controls = np.asarray(controls, dtype=float)
    slacks = None if slacks is None else np.asarray(slacks, dtype=float)
    show_fov = headings is not None and inflated_obstacles is not None and fov_angle is not None and cam_range is not None

    _plot_time_series(traj, controls, dt, slacks=slacks)

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
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if not animate:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=2)
        plt.tight_layout()
        plt.show()
        return

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

    artists = [line, agent, heading]
    if show_fov:
        headings = np.asarray(headings, dtype=float)
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
        boundary_points = [_sample_obstacle_boundary(obs) for obs in obstacles]
        sensed_edges = []
        for _ in obstacles:
            edge_line, = ax.plot([], [], color="tab:orange", linewidth=1.5, zorder=3)
            sensed_edges.append(edge_line)
        artists.extend([fov_poly, *sensed_edges])
    else:
        fov_poly = None
        boundary_points = []
        sensed_edges = []

    def _control_arrow(i):
        u = controls[i]
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            return 0.0, 0.0
        return u / u_norm * 0.35

    def init():
        line.set_data([], [])
        if show_fov:
            heading_angle = headings[0]
            fov_left, fov_right = _calculate_fov_points(traj[0], heading_angle, fov_angle, cam_range)
            fov_poly.set_xy(np.array([[traj[0, 0], traj[0, 1]], fov_left, fov_right], dtype=float))
            for edge_line in sensed_edges:
                edge_line.set_data([], [])
        return tuple(artists)

    def update(i):
        line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
        agent.center = (traj[i, 0], traj[i, 1])
        ux, uy = _control_arrow(i)
        heading.set_offsets([traj[i, 0], traj[i, 1]])
        heading.set_UVC([ux], [uy])

        if show_fov:
            heading_angle = headings[i]
            fov_left, fov_right = _calculate_fov_points(traj[i], heading_angle, fov_angle, cam_range)
            fov_poly.set_xy(np.array([[traj[i, 0], traj[i, 1]], fov_left, fov_right], dtype=float))
            sensed = _detect_sensed_obstacles(traj[i], heading_angle, inflated_obstacles, cam_range, fov_angle)[:max_sensed]
            sensed_ids = {id(obs) for obs in sensed}
            for idx, edge_line in enumerate(sensed_edges):
                pts = boundary_points[idx]
                rel = pts - traj[i]
                dists = np.linalg.norm(rel, axis=1)
                angles = np.arctan2(rel[:, 1], rel[:, 0])
                angle_diff = np.abs(_angle_wrap(angles - heading_angle))
                mask = (dists <= cam_range) & (angle_diff <= fov_angle / 2.0)
                if id(inflated_obstacles[idx]) in sensed_ids and np.any(mask):
                    edge_line.set_data(pts[mask, 0], pts[mask, 1])
                else:
                    edge_line.set_data([], [])

        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=range(0, len(traj), animation_stride),
        interval=animation_interval,
        blit=True,
        repeat=False,
    )
    if save_animation:
        animation_path = Path(animation_path)
        animation_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if animation_path.suffix == ".mp4":
                writer = animation.FFMpegWriter(fps=animation_fps)
            else:
                writer = animation.PillowWriter(fps=animation_fps)
            ani.save(animation_path, writer=writer)
        except Exception:
            if animation_path.suffix == ".mp4":
                fallback = animation_path.with_suffix(".gif")
                ani.save(fallback, writer=animation.PillowWriter(fps=animation_fps))
            else:
                raise

    plt.tight_layout()
    plt.show()
