from pathlib import Path
import argparse
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.density import Obstacle
from density_utils.sim import forward_euler
from density_utils.utils import plot_goal, plot_obstacle, plot_start

from cbf_filter import _p_norm_distance, _solve_clf_cbf_filter


def _pad_rows(values, target_len):
    values = np.asarray(values, dtype=float)
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return np.zeros((target_len, 2), dtype=float)
    return np.vstack([values, np.repeat(values[-1:], target_len - len(values), axis=0)])


def _pad_1d(values, target_len):
    values = np.asarray(values, dtype=float)
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return np.zeros(target_len, dtype=float)
    return np.concatenate([values, np.full(target_len - len(values), values[-1], dtype=float)])


def _control_arrow(controls, i, length=0.32):
    if len(controls) == 0:
        return 0.0, 0.0
    u = np.asarray(controls[min(i, len(controls) - 1)], dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-8:
        return 0.0, 0.0
    return u / u_norm * length


def _axis_limits(*arrays, pad_fraction=0.08):
    data = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    data = data[np.isfinite(data)]
    if data.size == 0:
        return -1.0, 1.0
    lo = float(np.min(data))
    hi = float(np.max(data))
    pad = max(1e-3, (hi - lo) * pad_fraction if abs(hi - lo) > 1e-9 else max(abs(hi), 1.0) * pad_fraction)
    return lo - pad, hi + pad


def _parse_gammas(gamma_text):
    gammas = [float(item.strip()) for item in gamma_text.split(",") if item.strip()]
    if not gammas:
        raise ValueError("provide at least one gamma value")
    return gammas


def _simulate_gamma(
    *,
    gamma,
    clf_rate,
    start,
    goal,
    inflated_obstacle,
    dt,
    steps,
    stop_tol,
    stop_steps,
    u_min,
    u_max,
    cbf_slack_weight,
    clf_slack_weight,
):
    print(f"running CLF-CBF filter gamma={gamma:.3f}")
    x = start.copy()
    traj = [x.copy()]
    controls = []
    cbf_slacks = []
    clf_slacks = []
    clearances = [_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1]
    solve_times = []
    solver_failures = 0
    stop_count = 0
    print_interval = max(1, min(250, steps // 8))

    for step in range(steps):
        dist = float(np.linalg.norm(x - goal))
        solve_start = time.perf_counter()
        filter_result = _solve_clf_cbf_filter(
            x,
            goal,
            inflated_obstacle,
            gamma=gamma,
            clf_rate=clf_rate,
            u_min=u_min,
            u_max=u_max,
            cbf_slack_weight=cbf_slack_weight,
            clf_slack_weight=clf_slack_weight,
        )
        solve_time = time.perf_counter() - solve_start
        solve_times.append(solve_time)
        if not filter_result["success"]:
            solver_failures += 1

        u = filter_result["u"]
        controls.append(u.copy())
        cbf_slacks.append(filter_result["cbf_slack"])
        clf_slacks.append(filter_result["clf_slack"])
        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        clearances.append(_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1)

        if dist < stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(
                    f"gamma={gamma:.3f} iter={step} dist={dist:.4f} "
                    f"clearance={clearances[-1]:.4f} avg_solve_ms={np.mean(solve_times) * 1e3:.3f} stable"
                )
                break
        else:
            stop_count = 0

        if step % print_interval == 0:
            print(
                f"gamma={gamma:.3f} iter={step} dist={dist:.4f} "
                f"clearance={clearances[-1]:.4f} "
                f"cbf_slack={cbf_slacks[-1]:.2e} clf_slack={clf_slacks[-1]:.2e} "
                f"solve_ms={solve_time * 1e3:.3f}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))
    if len(cbf_slacks) < len(traj):
        cbf_slacks.append(cbf_slacks[-1] if cbf_slacks else 0.0)
    if len(clf_slacks) < len(traj):
        clf_slacks.append(clf_slacks[-1] if clf_slacks else 0.0)

    traj = np.asarray(traj, dtype=float)
    controls = np.asarray(controls, dtype=float)
    cbf_slacks = np.asarray(cbf_slacks, dtype=float)
    clf_slacks = np.asarray(clf_slacks, dtype=float)
    clearances = np.asarray(clearances, dtype=float)
    solve_times = np.asarray(solve_times, dtype=float)

    print(
        f"gamma={gamma:.3f} steps={len(traj) - 1} "
        f"final_dist={np.linalg.norm(traj[-1] - goal):.4f} "
        f"min_clearance={np.min(clearances):.4f} "
        f"max_cbf_slack={np.max(cbf_slacks):.2e} "
        f"max_clf_slack={np.max(clf_slacks):.2e} "
        f"avg_solve_ms={np.mean(solve_times) * 1e3:.3f} "
        f"solver_failures={solver_failures}"
    )

    return {
        "gamma": gamma,
        "traj": traj,
        "controls": controls,
        "cbf_slack": cbf_slacks,
        "clf_slack": clf_slacks,
        "clearance": clearances,
        "solve_times": solve_times,
        "solver_failures": solver_failures,
    }


def _plot_xy(results, *, start, goal, obstacle, path, colors):
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    plot_start(ax, start)
    plot_goal(ax, goal)
    plot_obstacle(
        ax,
        obstacle.center,
        obstacle.r1,
        obstacle.r2,
        p=obstacle.p,
        scale=obstacle.scale,
        angle=obstacle.angle,
        color="0.3",
        fill=True,
    )
    for color, result in zip(colors, results):
        gamma = result["gamma"]
        traj = result["traj"]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.1, label=fr"$\gamma={gamma:.2f}$")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=34, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("CLF-CBF filter gamma sweep")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _plot_time_series(results, *, goal, dt, path, colors):
    max_len = max(len(result["traj"]) for result in results)
    t = dt * np.arange(max_len)

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.6), sharex=True)
    axes = axes.ravel()
    for color, result in zip(colors, results):
        label = fr"$\gamma={result['gamma']:.2f}$"
        traj = _pad_rows(result["traj"], max_len)
        controls = _pad_rows(result["controls"], max_len)
        distance = np.linalg.norm(traj - goal[None, :], axis=1)
        speed = np.linalg.norm(controls, axis=1)
        clearance = _pad_1d(result["clearance"], max_len)
        cbf_slack = _pad_1d(result["cbf_slack"], max_len)
        clf_slack = _pad_1d(result["clf_slack"], max_len)

        axes[0].plot(t, traj[:, 0], color=color, label=label)
        axes[1].plot(t, traj[:, 1], color=color, label=label)
        axes[2].plot(t, distance, color=color, label=label)
        axes[3].plot(t, speed, color=color, label=label)
        axes[4].plot(t, clearance, color=color, label=label)
        axes[5].plot(t, clf_slack, color=color, label=label)

    labels = ["x [m]", "y [m]", "distance to goal [m]", "||u|| [m/s]", "clearance [m]", "CLF slack"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[4].axhline(0.0, color="0.2", linewidth=1.0, linestyle="--")
    axes[4].set_xlabel("time [s]")
    axes[5].set_xlabel("time [s]")
    axes[0].legend(loc="upper right", ncols=min(3, len(results)), framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _save_dashboard_animation(results, *, start, goal, obstacle, agent_radius, dt, path, stride, fps, colors):
    max_len = max(len(result["traj"]) for result in results)
    padded = []
    for result in results:
        traj = _pad_rows(result["traj"], max_len)
        controls = _pad_rows(result["controls"], max_len)
        padded.append(
            {
                **result,
                "traj_pad": traj,
                "controls_pad": controls,
                "x": traj[:, 0],
                "y": traj[:, 1],
                "u_x": controls[:, 0],
                "u_y": controls[:, 1],
                "clearance_pad": _pad_1d(result["clearance"], max_len),
                "clf_slack_pad": _pad_1d(result["clf_slack"], max_len),
            }
        )

    frames = list(range(0, max_len, stride))
    if frames[-1] != max_len - 1:
        frames.append(max_len - 1)

    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.0, 1.0])
    map_ax = fig.add_subplot(gs[:, 0])
    trace_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]

    plot_start(map_ax, start)
    plot_goal(map_ax, goal)
    plot_obstacle(
        map_ax,
        obstacle.center,
        obstacle.r1,
        obstacle.r2,
        p=obstacle.p,
        scale=obstacle.scale,
        angle=obstacle.angle,
        color="0.3",
        fill=True,
    )
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_xlabel("x [m]")
    map_ax.set_ylabel("y [m]")
    map_ax.set_title("CLF-CBF filter gamma sweep")
    map_ax.grid(True, linestyle="--", alpha=0.35)

    map_lines = []
    agents = []
    headings = []
    for color, result in zip(colors, padded):
        line, = map_ax.plot([], [], color=color, linewidth=2.1, label=fr"$\gamma={result['gamma']:.2f}$")
        agent = patches.Circle(result["traj_pad"][0], agent_radius, color=color, alpha=0.9, zorder=5)
        heading = map_ax.quiver(
            result["traj_pad"][0, 0],
            result["traj_pad"][0, 1],
            0.0,
            0.0,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.005,
            zorder=6,
        )
        map_ax.add_patch(agent)
        map_lines.append(line)
        agents.append(agent)
        headings.append(heading)
    map_ax.legend(loc="lower right", framealpha=0.92)

    series_specs = [
        ("x", "x [m]"),
        ("y", "y [m]"),
        ("u_x", "u_x [m/s]"),
        ("u_y", "u_y [m/s]"),
        ("clearance_pad", "clearance [m]"),
        ("clf_slack_pad", "CLF slack"),
    ]
    t = dt * np.arange(max_len)
    dash_lines = []
    for ax, (key, ylabel) in zip(trace_axes, series_specs):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(_axis_limits(*(result[key] for result in padded)))
        if key == "clearance_pad":
            ax.axhline(0.0, color="0.2", linewidth=0.9, linestyle="--")
        lines = []
        for color, result in zip(colors, padded):
            line, = ax.plot([], [], color=color, linewidth=1.6)
            lines.append(line)
        dash_lines.append((key, lines))
    trace_axes[-2].set_xlabel("time [s]")
    trace_axes[-1].set_xlabel("time [s]")
    fig.tight_layout()

    def init():
        for line in map_lines:
            line.set_data([], [])
        for _, lines in dash_lines:
            for line in lines:
                line.set_data([], [])
        return tuple(map_lines + agents + headings + [line for _, lines in dash_lines for line in lines])

    def update(i):
        artists = []
        for idx, result in enumerate(padded):
            traj = result["traj_pad"]
            controls = result["controls_pad"]
            map_lines[idx].set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            agents[idx].center = (traj[i, 0], traj[i, 1])
            ux, uy = _control_arrow(controls, i)
            headings[idx].set_offsets([traj[i, 0], traj[i, 1]])
            headings[idx].set_UVC([ux], [uy])
            artists.extend([map_lines[idx], agents[idx], headings[idx]])

        current_t = t[: i + 1]
        for key, lines in dash_lines:
            for line, result in zip(lines, padded):
                line.set_data(current_t, result[key][: i + 1])
                artists.append(line)
        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=20,
        blit=True,
        repeat=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    return fig, ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gammas",
        default="0.50,0.625,0.75,0.875,1.00",
        help="Comma-separated CBF gamma values to sweep.",
    )
    parser.add_argument("--clf-rate", type=float, default=0.20, help="CLF decrease rate.")
    parser.add_argument("--steps", type=int, default=4000, help="Maximum simulation steps per gamma.")
    parser.add_argument("--stride", type=int, default=8, help="Animation frame stride.")
    parser.add_argument("--fps", type=int, default=18, help="GIF playback frame rate.")
    parser.add_argument("--no-gif", action="store_true", help="Skip saving the dashboard GIF.")
    parser.add_argument("--no-show", action="store_true", help="Save outputs without opening matplotlib windows.")
    args = parser.parse_args()

    dt = 0.1
    start = np.array([-2.0, -1.0])
    goal = np.array([2.0, 1.1])
    agent_radius = 0.1
    obstacle = Obstacle(center=np.array([0.0, 0.0]), r1=0.6, r2=1.0, p=2.0)
    inflated_obstacle = Obstacle(
        center=obstacle.center,
        r1=obstacle.r1 + agent_radius,
        r2=obstacle.r2 + agent_radius,
        p=obstacle.p,
        scale=obstacle.scale,
        angle=obstacle.angle,
    )
    u_max = np.array([2.0, 2.0])
    u_min = -u_max
    stop_tol = 0.005
    stop_steps = 500
    cbf_slack_weight = 1e6
    clf_slack_weight = 1e4
    gammas = _parse_gammas(args.gammas)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(gammas)))

    results = [
        _simulate_gamma(
            gamma=gamma,
            clf_rate=args.clf_rate,
            start=start,
            goal=goal,
            inflated_obstacle=inflated_obstacle,
            dt=dt,
            steps=args.steps,
            stop_tol=stop_tol,
            stop_steps=stop_steps,
            u_min=u_min,
            u_max=u_max,
            cbf_slack_weight=cbf_slack_weight,
            clf_slack_weight=clf_slack_weight,
        )
        for gamma in gammas
    ]

    output_dir = Path(__file__).resolve().parent / "comparison_results"
    xy_path = output_dir / "single_integrator_clf_cbf_filter_gamma_sweep_xy.png"
    ts_path = output_dir / "single_integrator_clf_cbf_filter_gamma_sweep_timeseries.png"
    gif_path = output_dir / "single_integrator_clf_cbf_filter_gamma_sweep.gif"

    figures = [
        _plot_xy(results, start=start, goal=goal, obstacle=obstacle, path=xy_path, colors=colors),
        _plot_time_series(results, goal=goal, dt=dt, path=ts_path, colors=colors),
    ]
    animations_to_show = []
    if not args.no_gif:
        fig, ani = _save_dashboard_animation(
            results,
            start=start,
            goal=goal,
            obstacle=obstacle,
            agent_radius=agent_radius,
            dt=dt,
            path=gif_path,
            stride=args.stride,
            fps=args.fps,
            colors=colors,
        )
        figures.append(fig)
        animations_to_show.append(ani)

    print(f"saved XY plot: {xy_path}")
    print(f"saved time-series plot: {ts_path}")
    if not args.no_gif:
        print(f"saved dashboard GIF: {gif_path}")

    if not args.no_show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    _ = animations_to_show


if __name__ == "__main__":
    main()
