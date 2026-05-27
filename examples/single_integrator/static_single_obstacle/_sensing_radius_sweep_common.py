from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import (
    SOLVER_CHOICES,
    density_feedback_control,
    single_integrator_nominal_control,
    solve_discrete_density_filter,
)
from density_utils.density import Obstacle, density_value
from density_utils.dynamics import single_integrator_step
from density_utils.sim import forward_euler
from density_utils.utils import plot_goal, plot_obstacle, plot_start

from cbf_filter import _p_norm_distance


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


def _axis_limits(*arrays, pad_fraction=0.08):
    data = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    data = data[np.isfinite(data)]
    if data.size == 0:
        return -1.0, 1.0
    lo = float(np.min(data))
    hi = float(np.max(data))
    pad = max(1e-3, (hi - lo) * pad_fraction if abs(hi - lo) > 1e-9 else max(abs(hi), 1.0) * pad_fraction)
    return lo - pad, hi + pad


def _control_arrow(controls, i, length=0.32):
    if len(controls) == 0:
        return 0.0, 0.0
    u = np.asarray(controls[min(i, len(controls) - 1)], dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-8:
        return 0.0, 0.0
    return u / u_norm * length


def _plot_sensing_ring(ax, obstacle, *, color):
    ring = patches.Circle(
        obstacle.center,
        obstacle.r2,
        fill=False,
        edgecolor=color,
        linestyle="--",
        linewidth=1.4,
        alpha=0.95,
    )
    ax.add_patch(ring)


def _parse_radii(radius_text):
    radii = [float(item.strip()) for item in radius_text.split(",") if item.strip()]
    if not radii:
        raise ValueError("provide at least one sensing radius")
    if any(radius <= 0.6 for radius in radii):
        raise ValueError("each sensing radius must be larger than the obstacle inner radius r1=0.6")
    return radii


def _make_obstacles(sensing_radius, agent_radius):
    obstacle = Obstacle(center=np.array([0.0, 0.0]), r1=0.6, r2=sensing_radius, p=2.0)
    inflated_obstacle = Obstacle(
        center=obstacle.center,
        r1=obstacle.r1 + agent_radius,
        r2=obstacle.r2 + agent_radius,
        p=obstacle.p,
        scale=obstacle.scale,
        angle=obstacle.angle,
    )
    return obstacle, inflated_obstacle


def _simulate_density_feedback(
    *,
    sensing_radius,
    start,
    goal,
    agent_radius,
    steps,
):
    obstacle, inflated_obstacle = _make_obstacles(sensing_radius, agent_radius)
    dt = 0.01
    alpha = 0.4
    ctrl_multiplier = 6.0
    rad_from_goal = 1.0
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    q_lqr = 4.0
    r_lqr = 1.0
    saturation = 4.0

    print(f"running density feedback sensing_radius={sensing_radius:.3f}")
    x = start.copy()
    traj = [x.copy()]
    controls = []
    density = [density_value(x, goal, alpha, [inflated_obstacle])]
    clearances = [_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1]
    solve_times = []
    stop_count = 0
    print_interval = max(1, min(500, steps // 8))

    for step in range(steps):
        dist = float(np.linalg.norm(x - goal))
        solve_start = time.perf_counter()
        u = density_feedback_control(
            x,
            goal,
            alpha,
            [inflated_obstacle],
            ctrl_multiplier=ctrl_multiplier,
            rad_from_goal=rad_from_goal,
            q_lqr=q_lqr,
            r_lqr=r_lqr,
            dt=dt,
            saturation=saturation,
        )
        solve_times.append(time.perf_counter() - solve_start)
        controls.append(u.copy())
        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        density.append(density_value(x, goal, alpha, [inflated_obstacle]))
        clearances.append(_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1)

        if dist < stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(
                    f"feedback r2={sensing_radius:.3f} iter={step} dist={dist:.4f} "
                    f"clearance={clearances[-1]:.4f} avg_eval_ms={np.mean(solve_times) * 1e3:.3f} stable"
                )
                break
        else:
            stop_count = 0
        if step % print_interval == 0:
            print(
                f"feedback r2={sensing_radius:.3f} iter={step} dist={dist:.4f} "
                f"rho={density[-1]:.3e} clearance={clearances[-1]:.4f}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))

    return {
        "controller": "feedback",
        "sensing_radius": sensing_radius,
        "dt": dt,
        "traj": np.asarray(traj, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "density": np.asarray(density, dtype=float),
        "clearance": np.asarray(clearances, dtype=float),
        "slack": np.zeros(len(traj), dtype=float),
        "solve_times": np.asarray(solve_times, dtype=float),
        "solver_failures": 0,
        "obstacle": obstacle,
    }


def _simulate_density_filter(
    *,
    sensing_radius,
    start,
    goal,
    agent_radius,
    steps,
    u_nom_mode,
    solver,
):
    obstacle, inflated_obstacle = _make_obstacles(sensing_radius, agent_radius)
    dt = 0.1
    alpha = 0.1
    ctrl_multiplier = 1.0
    rad_from_goal = 0.35
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    q_lqr = 4.0
    r_lqr = 1.0
    u_max = np.array([2.0, 2.0])
    u_min = -u_max
    slack_weight = 1e4

    print(f"running density filter sensing_radius={sensing_radius:.3f}")
    x = start.copy()
    traj = [x.copy()]
    controls = []
    density = [density_value(x, goal, alpha, [inflated_obstacle])]
    clearances = [_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1]
    slacks = []
    solve_times = []
    solver_failures = 0
    stop_count = 0
    print_interval = max(1, min(250, steps // 8))

    for step in range(steps):
        dist = float(np.linalg.norm(x - goal))
        solve_start = time.perf_counter()
        u_nom = single_integrator_nominal_control(
            x,
            goal,
            alpha,
            [inflated_obstacle],
            mode=u_nom_mode,
            ctrl_multiplier=ctrl_multiplier,
            rad_from_goal=rad_from_goal,
            q_lqr=q_lqr,
            r_lqr=r_lqr,
            dt=dt,
            u_min=u_min,
            u_max=u_max,
        )
        filter_result = solve_discrete_density_filter(
            x,
            goal,
            alpha,
            [inflated_obstacle],
            u_nom=u_nom,
            next_state_fn=single_integrator_step,
            dt=dt,
            u_min=u_min,
            u_max=u_max,
            divergence=0.0,
            slack_weight=slack_weight,
            solver=solver,
            return_info=True,
        )
        solve_times.append(time.perf_counter() - solve_start)
        if not filter_result.success:
            solver_failures += 1
        u = filter_result.u
        controls.append(u.copy())
        slacks.append(float(np.max(filter_result.slack)) if filter_result.slack.size else 0.0)
        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        density.append(density_value(x, goal, alpha, [inflated_obstacle]))
        clearances.append(_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1)

        if dist < stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(
                    f"filter r2={sensing_radius:.3f} iter={step} dist={dist:.4f} "
                    f"clearance={clearances[-1]:.4f} slack={slacks[-1]:.2e} "
                    f"avg_solve_ms={np.mean(solve_times) * 1e3:.3f} stable"
                )
                break
        else:
            stop_count = 0
        if step % print_interval == 0:
            print(
                f"filter r2={sensing_radius:.3f} iter={step} dist={dist:.4f} "
                f"rho={density[-1]:.3e} clearance={clearances[-1]:.4f} "
                f"slack={slacks[-1]:.2e} solve_ms={solve_times[-1] * 1e3:.3f}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    return {
        "controller": "filter",
        "sensing_radius": sensing_radius,
        "dt": dt,
        "traj": np.asarray(traj, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "density": np.asarray(density, dtype=float),
        "clearance": np.asarray(clearances, dtype=float),
        "slack": np.asarray(slacks, dtype=float),
        "solve_times": np.asarray(solve_times, dtype=float),
        "solver_failures": solver_failures,
        "obstacle": obstacle,
    }


def _summarize(result, goal, *, verbose=False):
    traj = result["traj"]
    solve_times = result["solve_times"]
    summary = (
        f"{result['controller']} r2={result['sensing_radius']:.3f} "
        f"steps={len(traj) - 1} final_dist={np.linalg.norm(traj[-1] - goal):.4f} "
        f"min_clearance={np.min(result['clearance']):.4f} "
        f"max_rho={np.max(result['density']):.3e} "
        f"max_slack={np.max(result['slack']):.2e} "
        f"avg_eval_ms={np.mean(solve_times) * 1e3:.3f}"
    )
    if verbose:
        summary += f" solver_failures={result['solver_failures']}"
    print(summary)


def _plot_xy(results, *, start, goal, agent_radius, path, colors, title):
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    plot_start(ax, start)
    plot_goal(ax, goal)
    base_obstacle = results[0]["obstacle"]
    plot_obstacle(
        ax,
        base_obstacle.center,
        base_obstacle.r1,
        None,
        p=base_obstacle.p,
        scale=base_obstacle.scale,
        angle=base_obstacle.angle,
        color="0.3",
        fill=True,
    )
    for result, color in zip(results, colors):
        obstacle = result["obstacle"]
        _plot_sensing_ring(ax, obstacle, color=color)
    for color, result in zip(colors, results):
        traj = result["traj"]
        radius = result["sensing_radius"]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.1, label=fr"$r_2={radius:.2f}$")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=34, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _plot_time_series(results, *, goal, path, colors, title):
    max_time = max((len(result["traj"]) - 1) * result["dt"] for result in results)
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.6), sharex=True)
    axes = axes.ravel()
    for color, result in zip(colors, results):
        label = fr"$r_2={result['sensing_radius']:.2f}$"
        traj = result["traj"]
        controls = _pad_rows(result["controls"], len(traj))
        t = result["dt"] * np.arange(len(traj))
        distance = np.linalg.norm(traj - goal[None, :], axis=1)
        speed = np.linalg.norm(controls, axis=1)

        axes[0].plot(t, traj[:, 0], color=color, label=label)
        axes[1].plot(t, traj[:, 1], color=color, label=label)
        axes[2].plot(t, distance, color=color, label=label)
        axes[3].plot(t, speed, color=color, label=label)
        axes[4].plot(t, result["density"], color=color, label=label)
        axes[5].plot(t, result["clearance"], color=color, label=label)

    labels = ["x [m]", "y [m]", "distance to goal [m]", "||u|| [m/s]", "rho(x)", "clearance [m]"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(0.0, max_time)
    axes[5].axhline(0.0, color="0.2", linewidth=1.0, linestyle="--")
    axes[4].set_xlabel("time [s]")
    axes[5].set_xlabel("time [s]")
    axes[0].legend(loc="upper right", ncols=min(3, len(results)), framealpha=0.92)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _save_dashboard_animation(results, *, start, goal, agent_radius, path, stride, fps, colors, title):
    max_time = max((len(result["traj"]) - 1) * result["dt"] for result in results)
    frame_times = np.arange(0.0, max_time + 1e-9, min(result["dt"] for result in results) * stride)
    if frame_times[-1] < max_time:
        frame_times = np.append(frame_times, max_time)

    padded = []
    for result in results:
        time_grid = result["dt"] * np.arange(len(result["traj"]))
        traj = result["traj"]
        controls = _pad_rows(result["controls"], len(traj))
        sampled_traj = np.column_stack(
            [
                np.interp(frame_times, time_grid, traj[:, 0]),
                np.interp(frame_times, time_grid, traj[:, 1]),
            ]
        )
        sampled_controls = np.column_stack(
            [
                np.interp(frame_times, time_grid, controls[:, 0]),
                np.interp(frame_times, time_grid, controls[:, 1]),
            ]
        )
        padded.append(
            {
                **result,
                "traj_frame": sampled_traj,
                "controls_frame": sampled_controls,
                "x": sampled_traj[:, 0],
                "y": sampled_traj[:, 1],
                "u_x": sampled_controls[:, 0],
                "u_y": sampled_controls[:, 1],
                "rho": np.interp(frame_times, time_grid, result["density"]),
                "clearance_frame": np.interp(frame_times, time_grid, result["clearance"]),
            }
        )

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
    base_obstacle = results[0]["obstacle"]
    plot_obstacle(
        map_ax,
        base_obstacle.center,
        base_obstacle.r1,
        None,
        p=base_obstacle.p,
        scale=base_obstacle.scale,
        angle=base_obstacle.angle,
        color="0.3",
        fill=True,
    )
    for result, color in zip(results, colors):
        obstacle = result["obstacle"]
        _plot_sensing_ring(map_ax, obstacle, color=color)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_xlabel("x [m]")
    map_ax.set_ylabel("y [m]")
    map_ax.set_title(title)
    map_ax.grid(True, linestyle="--", alpha=0.35)

    map_lines = []
    agents = []
    headings = []
    for color, result in zip(colors, padded):
        line, = map_ax.plot([], [], color=color, linewidth=2.1, label=fr"$r_2={result['sensing_radius']:.2f}$")
        agent = patches.Circle(result["traj_frame"][0], agent_radius, color=color, alpha=0.9, zorder=5)
        heading = map_ax.quiver(
            result["traj_frame"][0, 0],
            result["traj_frame"][0, 1],
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
        ("rho", "rho(x)"),
        ("clearance_frame", "clearance [m]"),
    ]
    dash_lines = []
    for ax, (key, ylabel) in zip(trace_axes, series_specs):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(frame_times[0], frame_times[-1])
        ax.set_ylim(_axis_limits(*(result[key] for result in padded)))
        if key == "clearance_frame":
            ax.axhline(0.0, color="0.2", linewidth=0.9, linestyle="--")
        lines = []
        for color in colors:
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
            traj = result["traj_frame"]
            controls = result["controls_frame"]
            map_lines[idx].set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            agents[idx].center = (traj[i, 0], traj[i, 1])
            ux, uy = _control_arrow(controls, i)
            headings[idx].set_offsets([traj[i, 0], traj[i, 1]])
            headings[idx].set_UVC([ux], [uy])
            artists.extend([map_lines[idx], agents[idx], headings[idx]])

        current_t = frame_times[: i + 1]
        for key, lines in dash_lines:
            for line, result in zip(lines, padded):
                line.set_data(current_t, result[key][: i + 1])
                artists.append(line)
        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=range(len(frame_times)),
        interval=20,
        blit=True,
        repeat=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    return fig, ani


def _run_controller(
    controller,
    *,
    radii,
    start,
    goal,
    agent_radius,
    steps_feedback,
    steps_filter,
    u_nom_mode,
    output_dir,
    no_gif,
    stride,
    fps,
    no_show,
    solver="auto",
    verbose=False,
):
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(radii)))
    if controller == "feedback":
        results = [
            _simulate_density_feedback(
                sensing_radius=radius,
                start=start,
                goal=goal,
                agent_radius=agent_radius,
                steps=steps_feedback,
            )
            for radius in radii
        ]
        label = "density_feedback"
        title = "Density feedback sensing-radius sweep"
    else:
        results = [
            _simulate_density_filter(
                sensing_radius=radius,
                start=start,
                goal=goal,
                agent_radius=agent_radius,
                steps=steps_filter,
                u_nom_mode=u_nom_mode,
                solver=solver,
            )
            for radius in radii
        ]
        label = "density_filter"
        title = "Density filter sensing-radius sweep"

    for result in results:
        _summarize(result, goal, verbose=verbose)

    xy_path = output_dir / f"single_integrator_{label}_sensing_radius_sweep_xy.png"
    ts_path = output_dir / f"single_integrator_{label}_sensing_radius_sweep_timeseries.png"
    gif_path = output_dir / f"single_integrator_{label}_sensing_radius_sweep.gif"

    figures = [
        _plot_xy(results, start=start, goal=goal, agent_radius=agent_radius, path=xy_path, colors=colors, title=title),
        _plot_time_series(results, goal=goal, path=ts_path, colors=colors, title=title),
    ]
    animations_to_show = []
    if not no_gif:
        fig, ani = _save_dashboard_animation(
            results,
            start=start,
            goal=goal,
            agent_radius=agent_radius,
            path=gif_path,
            stride=stride,
            fps=fps,
            colors=colors,
            title=title,
        )
        figures.append(fig)
        animations_to_show.append(ani)

    print(f"saved {controller} XY plot: {xy_path}")
    print(f"saved {controller} time-series plot: {ts_path}")
    if not no_gif:
        print(f"saved {controller} dashboard GIF: {gif_path}")

    if no_show:
        for fig in figures:
            plt.close(fig)
    return animations_to_show
