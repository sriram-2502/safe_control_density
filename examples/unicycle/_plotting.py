from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from density_utils.utils import plot_goal, plot_obstacle, plot_start

VIDEO_FIGSIZE = (6.0, 6.0)
VIDEO_DPI = 120
MP4_CRF = 28
MP4_PRESET = "slow"


def _angle_wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _triangle_points(center, heading, size):
    c = np.asarray(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    tip = c + size * 1.3 * forward
    left = c - size * 0.9 * forward + size * 0.6 * right
    right_pt = c - size * 0.9 * forward - size * 0.6 * right
    return np.stack([tip, left, right_pt], axis=0)


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
        if fov_angle < 2.0 * np.pi:
            angle_to_obs = np.arctan2(rel[1], rel[0])
            if abs(_angle_wrap(angle_to_obs - heading)) > fov_angle / 2.0:
                continue
        sensed.append((dist, obs))
    sensed.sort(key=lambda item: item[0])
    return [obs for _, obs in sensed]


def add_animation_save_args(parser):
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--save-mp4", action="store_true", help="Save a compact MP4 animation.")
    parser.add_argument(
        "--mp4-crf",
        type=int,
        default=MP4_CRF,
        help="MP4 quality factor. Higher is smaller; 26-30 is useful for slides/web.",
    )
    parser.add_argument(
        "--mp4-preset",
        default=MP4_PRESET,
        help="ffmpeg x264 preset used for MP4 export.",
    )


def animation_save_paths(base_path, *, save_gif=False, save_mp4=False):
    base_path = Path(base_path)
    paths = []
    if save_gif:
        paths.append(base_path.with_suffix(".gif"))
    if save_mp4:
        paths.append(base_path.with_suffix(".mp4"))
    return paths


def wants_animation_output(args):
    return bool(getattr(args, "save_gif", False) or getattr(args, "save_mp4", False))


def _obstacle_extent(obs):
    radius = float(obs.r2)
    if obs.scale is not None:
        radius *= float(np.max(np.asarray(obs.scale, dtype=float)))
    return radius


def _set_standard_plan_limits(ax, traj, start, goal, obstacles, extra_radius=0.0):
    points = [np.asarray(traj[:, :2], dtype=float), np.asarray([start, goal], dtype=float)]
    mins = []
    maxs = []
    for pts in points:
        mins.append(np.min(pts, axis=0))
        maxs.append(np.max(pts, axis=0))
    for obs in obstacles:
        radius = _obstacle_extent(obs)
        mins.append(np.asarray(obs.center, dtype=float) - radius)
        maxs.append(np.asarray(obs.center, dtype=float) + radius)
    lower = np.min(np.asarray(mins, dtype=float), axis=0)
    upper = np.max(np.asarray(maxs, dtype=float), axis=0)
    center = 0.5 * (lower + upper)
    width = float(np.max(upper - lower))
    half_width = 0.5 * max(width + 2.0 * extra_radius, 1.0)
    padding = max(0.10 * half_width, 0.15)
    half_width += padding
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)


def _animation_writer(path, fps, *, mp4_crf=MP4_CRF, mp4_preset=MP4_PRESET):
    path = Path(path)
    if path.suffix == ".mp4":
        return animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=-1,
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(int(mp4_crf)),
                "-preset",
                str(mp4_preset),
                "-movflags",
                "+faststart",
            ],
        )
    return animation.PillowWriter(fps=fps)


def save_animation_file(ani, path, fps, *, mp4_crf=MP4_CRF, mp4_preset=MP4_PRESET):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=_animation_writer(path, fps, mp4_crf=mp4_crf, mp4_preset=mp4_preset), dpi=VIDEO_DPI)
    print(f"saved animation to {path}")


def _plot_time_series(traj, controls, dt, slacks=None):
    rows = 3
    fig, axes = plt.subplots(rows, 2, figsize=(9, 7))
    t_state = dt * np.arange(len(traj))
    t_u = dt * np.arange(len(controls))

    axes[0, 0].plot(t_state, traj[:, 0], linewidth=1.8, label="x [m]")
    axes[0, 1].plot(t_state, traj[:, 1], linewidth=1.8, label="y [m]")
    axes[1, 0].plot(t_state, traj[:, 2], linewidth=1.8, label="theta [rad]")
    axes[1, 1].plot(t_u, controls[:, 0], linewidth=1.8, label="v [m/s]")
    axes[2, 0].plot(t_u, controls[:, 1], linewidth=1.8, label="omega [rad/s]")
    if slacks is None:
        axes[2, 1].axis("off")
    else:
        axes[2, 1].plot(t_u, slacks, linewidth=1.8, label="slack")

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel("time [s]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")


def plot_unicycle_results(
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
    inflated_obstacles=None,
    fov_angle=None,
    cam_range=None,
    fov_active=None,
    max_sensed=5,
    animation_interval=20,
    animation_paths=None,
    show_plot=True,
    mp4_crf=MP4_CRF,
    mp4_preset=MP4_PRESET,
):
    """Plot common unicycle traces and plan-view animation."""
    traj = np.asarray(traj, dtype=float)
    controls = np.asarray(controls, dtype=float)
    slacks = None if slacks is None else np.asarray(slacks, dtype=float)
    fov_active = None if fov_active is None else np.asarray(fov_active, dtype=bool)
    show_fov = inflated_obstacles is not None and fov_angle is not None and cam_range is not None

    if show_plot:
        _plot_time_series(traj, controls, dt, slacks=slacks)

    fig, ax = plt.subplots(figsize=VIDEO_FIGSIZE, dpi=VIDEO_DPI)
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
    _set_standard_plan_limits(
        ax,
        traj,
        start,
        goal,
        obstacles,
        extra_radius=float(cam_range) if show_fov else float(agent_radius),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if not animate:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=2)
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.92)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        return

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
    artists = [line, agent]

    if show_fov:
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

    def set_fov_style(i):
        if not show_fov or fov_active is None:
            return
        is_active = bool(fov_active[min(i, len(fov_active) - 1)])
        if is_active:
            fov_poly.set_edgecolor("darkred")
            fov_poly.set_facecolor("red")
            fov_poly.set_alpha(0.28)
            fov_poly.set_linewidth(2.8)
        else:
            fov_poly.set_edgecolor("darkorange")
            fov_poly.set_facecolor("gold")
            fov_poly.set_alpha(0.25)
            fov_poly.set_linewidth(2.5)

    def init():
        line.set_data([], [])
        if show_fov:
            set_fov_style(0)
            left, right = _calculate_fov_points(traj[0, :2], traj[0, 2], fov_angle, cam_range)
            fov_poly.set_xy(np.array([[traj[0, 0], traj[0, 1]], left, right], dtype=float))
            for edge_line in sensed_edges:
                edge_line.set_data([], [])
        return tuple(artists)

    def update(i):
        line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
        agent.set_xy(_triangle_points(traj[i, :2], traj[i, 2], agent_radius))
        if show_fov:
            set_fov_style(i)
            left, right = _calculate_fov_points(traj[i, :2], traj[i, 2], fov_angle, cam_range)
            fov_poly.set_xy(np.array([[traj[i, 0], traj[i, 1]], left, right], dtype=float))
            sensed = _detect_sensed_obstacles(
                traj[i, :2],
                traj[i, 2],
                inflated_obstacles,
                cam_range,
                fov_angle,
            )[:max_sensed]
            sensed_ids = {id(obs) for obs in sensed}
            for idx, edge_line in enumerate(sensed_edges):
                pts = boundary_points[idx]
                rel = pts - traj[i, :2]
                dists = np.linalg.norm(rel, axis=1)
                angles = np.arctan2(rel[:, 1], rel[:, 0])
                angle_diff = np.abs(_angle_wrap(angles - traj[i, 2]))
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
    paths = []
    if save_animation:
        paths.append(Path(animation_path))
    if animation_paths is not None:
        paths.extend(Path(path) for path in animation_paths)

    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.92)
    for path in paths:
        try:
            save_animation_file(ani, path, animation_fps, mp4_crf=mp4_crf, mp4_preset=mp4_preset)
        except Exception:
            if path.suffix == ".mp4":
                fallback = path.with_suffix(".gif")
                save_animation_file(ani, fallback, animation_fps)
            else:
                raise

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
