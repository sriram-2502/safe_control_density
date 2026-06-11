from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

from density_utils.utils import plot_goal, plot_start


VIDEO_FIGSIZE = (6.4, 6.0)
VIDEO_DPI = 96
MP4_CRF = 30
MP4_PRESET = "slow"


def add_animation_save_args(parser):
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--save-mp4", action="store_true", help="Save a compact MP4 animation.")
    parser.add_argument("--follow-robot", action="store_true", help="Center the animation view on the robot.")
    parser.add_argument("--follow-width", type=float, default=9.0, help="Local-frame animation width in meters.")
    parser.add_argument("--follow-height", type=float, default=6.5, help="Local-frame animation height in meters.")
    parser.add_argument(
        "--mp4-crf",
        type=int,
        default=MP4_CRF,
        help="MP4 quality factor. Higher is smaller; 28-32 is useful for git/web previews.",
    )
    parser.add_argument("--mp4-preset", default=MP4_PRESET)


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


def _triangle_points(center, heading, size):
    center = np.asarray(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    return np.stack(
        [
            center + size * 1.35 * forward,
            center - size * 0.9 * forward + size * 0.65 * right,
            center - size * 0.9 * forward - size * 0.65 * right,
        ],
        axis=0,
    )


def _plot_time_series(traj, controls, clearances, dt):
    t_state = dt * np.arange(len(traj))
    t_u = dt * np.arange(len(controls))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=False)
    axes = axes.ravel()
    axes[0].plot(t_state, traj[:, 0], linewidth=1.8, label="x [m]")
    axes[1].plot(t_state, traj[:, 1], linewidth=1.8, label="y [m]")
    axes[2].plot(t_u, controls[:, 0], linewidth=1.8, label="v [m/s]")
    axes[2].plot(t_u, controls[:, 1], linewidth=1.8, label="omega [rad/s]")
    axes[3].plot(t_state, clearances, linewidth=1.8, label="min clearance [m]")
    axes[3].axhline(0.0, color="k", linestyle=":", linewidth=1.0)
    for ax in axes:
        ax.set_xlabel("time [s]")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
    fig.tight_layout()


def plot_dynamic_obstacle_results(
    *,
    traj,
    controls,
    obstacle_traj,
    obstacle_vel_traj,
    obstacle_radii,
    clearances,
    dt,
    start,
    goal,
    robot_radius,
    title,
    xlim,
    ylim,
    save_paths,
    show_plot,
    fps,
    animation_stride,
    mp4_crf=MP4_CRF,
    mp4_preset=MP4_PRESET,
    follow_robot=False,
    follow_size=(9.0, 6.5),
):
    traj = np.asarray(traj, dtype=float)
    controls = np.asarray(controls, dtype=float)
    obstacle_traj = np.asarray(obstacle_traj, dtype=float)
    obstacle_vel_traj = np.asarray(obstacle_vel_traj, dtype=float)
    obstacle_radii = np.asarray(obstacle_radii, dtype=float)
    if obstacle_radii.ndim == 1:
        obstacle_radii = np.repeat(obstacle_radii[None, :], obstacle_traj.shape[0], axis=0)
    clearances = np.asarray(clearances, dtype=float)

    if show_plot:
        _plot_time_series(traj, controls, clearances, dt)

    fig, ax = plt.subplots(figsize=VIDEO_FIGSIZE, dpi=VIDEO_DPI)
    plot_start(ax, start)
    plot_goal(ax, goal)
    follow_width, follow_height = follow_size
    if follow_robot:
        ax.set_xlim(traj[0, 0] - 0.5 * follow_width, traj[0, 0] + 0.5 * follow_width)
        ax.set_ylim(traj[0, 1] - 0.5 * follow_height, traj[0, 1] + 0.5 * follow_height)
    else:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    line, = ax.plot([], [], color="tab:blue", linewidth=2.0, label="robot")
    robot_patch = patches.Polygon(
        _triangle_points(traj[0, :2], traj[0, 2], robot_radius),
        closed=True,
        facecolor="tab:blue",
        edgecolor="k",
        linewidth=0.9,
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(robot_patch)

    obstacle_patches = []
    velocity_arrows = []
    for idx, radius in enumerate(obstacle_radii[0]):
        patch = patches.Circle(
            obstacle_traj[0, idx],
            radius,
            edgecolor="0.2",
            facecolor="0.65",
            alpha=0.75,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(patch)
        obstacle_patches.append(patch)
        arrow = patches.FancyArrowPatch(
            obstacle_traj[0, idx],
            obstacle_traj[0, idx] + obstacle_vel_traj[0, idx],
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color="tab:orange",
            zorder=4,
        )
        ax.add_patch(arrow)
        velocity_arrows.append(arrow)

    status_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, family="monospace")
    ax.legend(loc="upper left", fontsize=8)

    frames = range(0, len(traj), max(1, int(animation_stride)))

    def update(frame):
        if follow_robot:
            center = traj[frame, :2]
            ax.set_xlim(center[0] - 0.5 * follow_width, center[0] + 0.5 * follow_width)
            ax.set_ylim(center[1] - 0.5 * follow_height, center[1] + 0.5 * follow_height)
        line.set_data(traj[: frame + 1, 0], traj[: frame + 1, 1])
        robot_patch.set_xy(_triangle_points(traj[frame, :2], traj[frame, 2], robot_radius))
        for idx, patch in enumerate(obstacle_patches):
            center = obstacle_traj[frame, idx]
            patch.center = center
            patch.radius = obstacle_radii[frame, idx]
            patch.set_alpha(0.75 if obstacle_radii[frame, idx] > 0.0 else 0.0)
            velocity_arrows[idx].set_positions(center, center + obstacle_vel_traj[frame, idx])
            velocity_arrows[idx].set_alpha(1.0 if obstacle_radii[frame, idx] > 0.0 else 0.0)
        status_text.set_text(f"t={frame * dt:5.2f} s | clearance={clearances[frame]: .3f} m")
        return [line, robot_patch, status_text, *obstacle_patches, *velocity_arrows]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=not follow_robot,
        repeat=False,
    )
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.10, top=0.92)
    for path in save_paths:
        save_animation_file(ani, path, fps, mp4_crf=mp4_crf, mp4_preset=mp4_preset)
    if show_plot:
        plt.show()
    else:
        plt.close("all")
