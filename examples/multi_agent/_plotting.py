from pathlib import Path

import numpy as np

from density_utils.utils import plot_goal, plot_start


MP4_CRF = 28
MP4_PRESET = "slow"
VIDEO_DPI = 120
VIDEO_FIGSIZE = (6.0, 6.0)
AGENT_COLORS = (
    "tab:purple",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:brown",
    "tab:pink",
    "tab:cyan",
    "tab:olive",
    "0.35",
    "goldenrod",
)


def _triangle_points(center, heading, size):
    center = np.asarray(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    tip = center + size * 1.3 * forward
    left = center - size * 0.9 * forward + size * 0.6 * right
    right_pt = center - size * 0.9 * forward - size * 0.6 * right
    return np.stack([tip, left, right_pt], axis=0)


def add_animation_save_args(parser):
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--save-mp4", action="store_true", help="Save a compact MP4 animation.")
    parser.add_argument(
        "--mp4-crf",
        type=int,
        default=MP4_CRF,
        help="MP4 quality factor. Higher is smaller; 26-30 is useful for slides/web.",
    )
    parser.add_argument("--mp4-preset", default=MP4_PRESET, help="ffmpeg x264 preset used for MP4 export.")


def animation_save_paths(base_path, *, save_gif=False, save_mp4=False):
    base_path = Path(base_path)
    paths = []
    if save_gif:
        paths.append(base_path.with_suffix(".gif"))
    if save_mp4:
        paths.append(base_path.with_suffix(".mp4"))
    return paths


def wants_animation_output(args):
    return bool(args.save_gif or args.save_mp4)


def _animation_writer(path, fps, *, mp4_crf=MP4_CRF, mp4_preset=MP4_PRESET):
    from matplotlib import animation

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


def _plot_multi_agent_results(
    *,
    traj,
    starts,
    goals,
    agent_r1,
    title,
    save_paths,
    show_plot,
    fps,
    mp4_crf,
    mp4_preset,
):
    import matplotlib.pyplot as plt
    from matplotlib import animation, patches

    traj = np.asarray(traj, dtype=float)
    fig, ax = plt.subplots(figsize=VIDEO_FIGSIZE, dpi=VIDEO_DPI)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    all_points = np.concatenate([traj[:, :, :2].reshape(-1, 2), starts, goals], axis=0)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    width = float(np.max(all_points.max(axis=0) - all_points.min(axis=0)))
    half_width = 0.5 * max(width, 5.0) + 0.4
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)

    colors = [AGENT_COLORS[idx % len(AGENT_COLORS)] for idx in range(starts.shape[0])]
    for idx, start in enumerate(starts):
        plot_start(ax, start, color=colors[idx])
    for idx, goal in enumerate(goals):
        plot_goal(ax, goal, color=colors[idx])

    agent_patches = []
    trail_lines = []
    for idx in range(starts.shape[0]):
        tri = _triangle_points(traj[0, idx, :2], traj[0, idx, 2], agent_r1[idx])
        agent = patches.Polygon(
            tri,
            closed=True,
            facecolor=colors[idx],
            edgecolor="k",
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(agent)
        agent_patches.append(agent)
        line, = ax.plot([], [], linewidth=2.0, color=colors[idx])
        trail_lines.append(line)

    def init():
        for line in trail_lines:
            line.set_data([], [])
        return (*agent_patches, *trail_lines)

    def update(frame_idx):
        for idx in range(traj.shape[1]):
            agent_patches[idx].set_xy(_triangle_points(traj[frame_idx, idx, :2], traj[frame_idx, idx, 2], agent_r1[idx]))
            trail_lines[idx].set_data(traj[: frame_idx + 1, idx, 0], traj[: frame_idx + 1, idx, 1])
        return (*agent_patches, *trail_lines)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=range(traj.shape[0]),
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.92)
    for path in save_paths:
        save_animation_file(ani, path, fps, mp4_crf=mp4_crf, mp4_preset=mp4_preset)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
