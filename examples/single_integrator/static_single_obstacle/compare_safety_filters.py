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

from density_utils.controllers import (
    density_feedback_control,
    single_integrator_nominal_control,
    solve_discrete_density_filter,
)
from density_utils.density import Obstacle, density_value
from density_utils.dynamics import single_integrator_step
from density_utils.sim import forward_euler
from density_utils.utils import plot_goal, plot_obstacle, plot_start

from cbf_filter import _p_norm_distance, _solve_clf_cbf_filter


STYLE = {
    "density_feedback": {"label": "Density feedback", "color": "tab:blue"},
    "density_filter": {"label": "Density filter", "color": "tab:orange"},
    "clf_cbf_filter": {"label": "CLF-CBF filter", "color": "tab:green"},
}


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


def _simulate_density_feedback(*, start, goal, inflated_obstacle, dt, steps, alpha, stop_tol, stop_steps):
    print("running density feedback agent")
    x = start.copy()
    traj = [x.copy()]
    controls = []
    density = [density_value(x, goal, alpha, [inflated_obstacle])]
    clearances = [_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1]
    solve_times = []
    stop_count = 0
    print_interval = max(1, min(100, steps // 10))

    for step in range(steps):
        dist = float(np.linalg.norm(x - goal))
        solve_start = time.perf_counter()
        u = density_feedback_control(
            x,
            goal,
            alpha,
            [inflated_obstacle],
            ctrl_multiplier=3.0,
            rad_from_goal=0.35,
            q_lqr=4.0,
            r_lqr=1.0,
            dt=dt,
            saturation=2.0,
        )
        solve_times.append(time.perf_counter() - solve_start)
        controls.append(u.copy())
        x = single_integrator_step(x, u, dt)
        traj.append(x.copy())
        density.append(density_value(x, goal, alpha, [inflated_obstacle]))
        clearances.append(_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1)

        if dist < stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(f"density_feedback iter={step} dist={dist:.4f} clearance={clearances[-1]:.4f} stable")
                break
        else:
            stop_count = 0
        if step % print_interval == 0:
            print(
                f"density_feedback iter={step} dist={dist:.4f} "
                f"rho={density[-1]:.4e} clearance={clearances[-1]:.4f}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))

    return {
        "name": "density_feedback",
        "traj": np.asarray(traj, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "density": np.asarray(density, dtype=float),
        "clearance": np.asarray(clearances, dtype=float),
        "slack": np.zeros(len(traj), dtype=float),
        "solve_times": np.asarray(solve_times, dtype=float),
        "solver_failures": 0,
    }


def _simulate_density_filter(*, start, goal, inflated_obstacle, dt, steps, alpha, stop_tol, stop_steps):
    print("running density filter agent")
    x = start.copy()
    traj = [x.copy()]
    controls = []
    density = [density_value(x, goal, alpha, [inflated_obstacle])]
    clearances = [_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1]
    slacks = []
    solve_times = []
    solver_failures = 0
    u_max = np.array([2.0, 2.0])
    u_min = -u_max
    stop_count = 0
    print_interval = max(1, min(100, steps // 10))

    for step in range(steps):
        dist = float(np.linalg.norm(x - goal))
        u_nom = single_integrator_nominal_control(
            x,
            goal,
            alpha,
            [inflated_obstacle],
            mode="density_blend",
            ctrl_multiplier=3.0,
            rad_from_goal=0.35,
            q_lqr=4.0,
            r_lqr=1.0,
            dt=dt,
            u_min=u_min,
            u_max=u_max,
        )
        solve_start = time.perf_counter()
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
            slack_weight=1e4,
            return_info=True,
        )
        solve_times.append(time.perf_counter() - solve_start)
        if not filter_result.success:
            solver_failures += 1
        u = filter_result.u
        controls.append(u.copy())
        slacks.append(float(np.max(filter_result.slack)) if filter_result.slack.size else 0.0)
        x = single_integrator_step(x, u, dt)
        traj.append(x.copy())
        density.append(density_value(x, goal, alpha, [inflated_obstacle]))
        clearances.append(_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1)

        if dist < stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(
                    f"density_filter iter={step} dist={dist:.4f} clearance={clearances[-1]:.4f} "
                    f"slack={slacks[-1]:.2e} stable"
                )
                break
        else:
            stop_count = 0
        if step % print_interval == 0:
            print(
                f"density_filter iter={step} dist={dist:.4f} solve_ms={solve_times[-1] * 1e3:.3f} "
                f"rho={density[-1]:.4e} clearance={clearances[-1]:.4f} slack={slacks[-1]:.2e}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    return {
        "name": "density_filter",
        "traj": np.asarray(traj, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "density": np.asarray(density, dtype=float),
        "clearance": np.asarray(clearances, dtype=float),
        "slack": np.asarray(slacks, dtype=float),
        "solve_times": np.asarray(solve_times, dtype=float),
        "solver_failures": solver_failures,
    }


def _simulate_clf_cbf_filter(*, start, goal, inflated_obstacle, dt, steps, alpha, stop_tol, stop_steps, gamma, clf_rate):
    print("running CLF-CBF filter agent")
    x = start.copy()
    traj = [x.copy()]
    controls = []
    density = [density_value(x, goal, alpha, [inflated_obstacle])]
    clearances = [_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1]
    cbf_slacks = []
    clf_slacks = []
    solve_times = []
    solver_failures = 0
    u_max = np.array([2.0, 2.0])
    u_min = -u_max
    stop_count = 0
    print_interval = max(1, min(100, steps // 10))

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
            cbf_slack_weight=1e6,
            clf_slack_weight=1e4,
        )
        solve_times.append(time.perf_counter() - solve_start)
        if not filter_result["success"]:
            solver_failures += 1
        u = filter_result["u"]
        controls.append(u.copy())
        cbf_slacks.append(filter_result["cbf_slack"])
        clf_slacks.append(filter_result["clf_slack"])
        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        density.append(density_value(x, goal, alpha, [inflated_obstacle]))
        clearances.append(_p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1)

        if dist < stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(
                    f"clf_cbf_filter iter={step} dist={dist:.4f} clearance={clearances[-1]:.4f} "
                    f"cbf_slack={cbf_slacks[-1]:.2e} clf_slack={clf_slacks[-1]:.2e} stable"
                )
                break
        else:
            stop_count = 0
        if step % print_interval == 0:
            print(
                f"clf_cbf_filter iter={step} dist={dist:.4f} solve_ms={solve_times[-1] * 1e3:.3f} "
                f"rho={density[-1]:.4e} clearance={clearances[-1]:.4f} "
                f"cbf_slack={cbf_slacks[-1]:.2e} clf_slack={clf_slacks[-1]:.2e}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2))
    if len(cbf_slacks) < len(traj):
        cbf_slacks.append(cbf_slacks[-1] if cbf_slacks else 0.0)
    if len(clf_slacks) < len(traj):
        clf_slacks.append(clf_slacks[-1] if clf_slacks else 0.0)

    return {
        "name": "clf_cbf_filter",
        "traj": np.asarray(traj, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "density": np.asarray(density, dtype=float),
        "clearance": np.asarray(clearances, dtype=float),
        "slack": np.asarray(cbf_slacks, dtype=float),
        "clf_slack": np.asarray(clf_slacks, dtype=float),
        "solve_times": np.asarray(solve_times, dtype=float),
        "solver_failures": solver_failures,
    }


def _plot_xy(results, *, start, goal, obstacle, path):
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
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
    for result in results:
        style = STYLE[result["name"]]
        traj = result["traj"]
        ax.plot(traj[:, 0], traj[:, 1], color=style["color"], linewidth=2.2, label=style["label"])
        ax.scatter(traj[-1, 0], traj[-1, 1], color=style["color"], s=44, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Single integrator static obstacle: safety filters")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _plot_time_series(results, *, dt, path):
    max_len = max(len(result["traj"]) for result in results)
    t = dt * np.arange(max_len)
    fig, axes = plt.subplots(4, 2, figsize=(11.2, 10.0), sharex=True)
    axes = axes.ravel()

    for result in results:
        style = STYLE[result["name"]]
        label = style["label"]
        color = style["color"]
        traj = _pad_rows(result["traj"], max_len)
        controls = _pad_rows(result["controls"], max_len)
        speed = np.linalg.norm(controls, axis=1)
        axes[0].plot(t, traj[:, 0], label=label, color=color)
        axes[1].plot(t, traj[:, 1], label=label, color=color)
        axes[2].plot(t, controls[:, 0], label=label, color=color)
        axes[3].plot(t, controls[:, 1], label=label, color=color)
        axes[4].plot(t, speed, label=label, color=color)
        axes[5].plot(t, _pad_1d(result["clearance"], max_len), label=label, color=color)
        axes[6].plot(t, _pad_1d(result["density"], max_len), label=label, color=color)
        axes[7].plot(t, _pad_1d(result["slack"], max_len), label=label, color=color)

    labels = ["x [m]", "y [m]", "u_x [m/s]", "u_y [m/s]", "||u|| [m/s]", "clearance [m]", "rho(x)", "safety slack"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[5].axhline(0.0, color="0.2", linewidth=1.0, linestyle="--")
    axes[6].set_xlabel("time [s]")
    axes[7].set_xlabel("time [s]")
    axes[0].legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    return fig


def _save_dashboard_animation(results, *, start, goal, obstacle, agent_radius, dt, path, stride, fps):
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
                "rho": _pad_1d(result["density"], max_len),
                "clearance_pad": _pad_1d(result["clearance"], max_len),
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
    map_ax.set_title("Safety filter comparison")
    map_ax.grid(True, linestyle="--", alpha=0.35)

    map_lines = []
    agents = []
    headings = []
    for result in padded:
        style = STYLE[result["name"]]
        traj = result["traj_pad"]
        line, = map_ax.plot([], [], color=style["color"], linewidth=2.1, label=style["label"])
        agent = patches.Circle(traj[0], agent_radius, color=style["color"], alpha=0.9, zorder=5)
        heading = map_ax.quiver(
            traj[0, 0],
            traj[0, 1],
            0.0,
            0.0,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=style["color"],
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
        ("clearance_pad", "clearance [m]"),
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
        for result in padded:
            line, = ax.plot([], [], color=STYLE[result["name"]]["color"], linewidth=1.6)
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


def _print_summary(result):
    solve_ms = result["solve_times"] * 1e3
    avg_solve_ms = float(np.mean(solve_ms)) if solve_ms.size else 0.0
    max_solve_ms = float(np.max(solve_ms)) if solve_ms.size else 0.0
    print(
        f"{result['name']} "
        f"steps={len(result['traj']) - 1} "
        f"min_clearance={np.min(result['clearance']):.4f} "
        f"final_density={result['density'][-1]:.4e} "
        f"max_slack={np.max(result['slack']):.2e} "
        f"solver_failures={result['solver_failures']} "
        f"avg_solve_ms={avg_solve_ms:.3f} "
        f"max_solve_ms={max_solve_ms:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gif", action="store_true", help="Skip saving the comparison GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Save figures without opening plot windows.")
    parser.add_argument("--steps", type=int, default=4000, help="Maximum simulation steps.")
    parser.add_argument("--gamma", type=float, default=0.85, help="CLF-CBF filter CBF rate.")
    parser.add_argument("--clf-rate", type=float, default=0.20, help="CLF-CBF filter CLF rate.")
    parser.add_argument("--stride", type=int, default=8, help="Dashboard GIF frame stride.")
    parser.add_argument("--fps", type=int, default=18, help="Dashboard GIF playback frame rate.")
    args = parser.parse_args()

    dt = 0.02
    alpha = 0.4
    stop_tol = 0.01
    stop_steps = 100
    agent_radius = 0.1
    start = np.array([-2.0, -1.0])
    goal = np.array([2.0, 1.1])
    obstacle = Obstacle(center=np.array([0.0, 0.0]), r1=0.6, r2=1.0, p=2.0)
    inflated_obstacle = Obstacle(
        center=obstacle.center,
        r1=obstacle.r1 + agent_radius,
        r2=obstacle.r2 + agent_radius,
        p=obstacle.p,
        scale=obstacle.scale,
        angle=obstacle.angle,
    )

    results = [
        _simulate_density_feedback(
            start=start,
            goal=goal,
            inflated_obstacle=inflated_obstacle,
            dt=dt,
            steps=args.steps,
            alpha=alpha,
            stop_tol=stop_tol,
            stop_steps=stop_steps,
        ),
        _simulate_density_filter(
            start=start,
            goal=goal,
            inflated_obstacle=inflated_obstacle,
            dt=dt,
            steps=args.steps,
            alpha=alpha,
            stop_tol=stop_tol,
            stop_steps=stop_steps,
        ),
        _simulate_clf_cbf_filter(
            start=start,
            goal=goal,
            inflated_obstacle=inflated_obstacle,
            dt=dt,
            steps=args.steps,
            alpha=alpha,
            stop_tol=stop_tol,
            stop_steps=stop_steps,
            gamma=args.gamma,
            clf_rate=args.clf_rate,
        ),
    ]

    output_dir = Path(__file__).resolve().parent / "comparison_results"
    xy_path = output_dir / "single_integrator_static_safety_filters_xy.png"
    ts_path = output_dir / "single_integrator_static_safety_filters_timeseries.png"
    gif_path = output_dir / "single_integrator_static_safety_filters.gif"

    figures = [
        _plot_xy(results, start=start, goal=goal, obstacle=obstacle, path=xy_path),
        _plot_time_series(results, dt=dt, path=ts_path),
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
        )
        figures.append(fig)
        animations_to_show.append(ani)

    for result in results:
        _print_summary(result)
    print(f"saved {xy_path}")
    print(f"saved {ts_path}")
    if not args.no_gif:
        print(f"saved {gif_path}")

    if not args.no_plot:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    _ = animations_to_show


if __name__ == "__main__":
    main()
