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


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _triangle_points(center, heading, size):
    c = np.array(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    tip = c + size * 1.3 * forward
    left = c - size * 0.9 * forward + size * 0.6 * right
    right_pt = c - size * 0.9 * forward - size * 0.6 * right
    return np.stack([tip, left, right_pt], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    args = parser.parse_args()

    dt = 0.01
    steps = 4000
    alpha = 0.4
    ctrl_multiplier = 6.0
    rad_from_goal = 0.01
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    stop_when_stable = True
    q_lqr = 4.0
    r_lqr = 1.0
    saturation = 4.0
    k_heading = 2.0
    v_max = 2.0
    omega_max = 3.0
    animate = True
    save_animation = args.save_gif
    animation_stride = 10
    animation_fps = 20
    animation_format = "gif"
    animation_path = Path("animations") / f"unicycle_static.{animation_format}"

    agent_radius = 0.1
    start = np.array([-2.0, -1.0])
    goal = np.array([2.0, 1.0])
    obstacle = Obstacle(center=np.array([0.0, 0.0]), r1=0.6, r2=1.0, p=2.0)
    inflated_obstacle = Obstacle(
        center=obstacle.center,
        r1=obstacle.r1 + agent_radius,
        r2=obstacle.r2 + agent_radius,
        p=obstacle.p,
        scale=obstacle.scale,
    )

    heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    state = np.array([start[0], start[1], heading0], dtype=float)
    tilde_prev = heading0
    traj = [state.copy()]

    control_time = 0.0
    control_times = []
    controls = []
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    stop_count = 0
    for step in range(steps):
        pos = state[:2]
        dist = np.linalg.norm(pos - goal)
        with timer:
            u = density_feedback_control(
                pos,
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
        dt_control = timer.last
        control_time += dt_control
        if log_timing and dist >= rad_from_goal:
            control_times.append(dt_control)
        v = float(np.linalg.norm(u))
        v = min(v, v_max)
        tilde = float(np.arctan2(u[1], u[0]))
        tilde_dot = _wrap_angle(tilde - tilde_prev) / dt
        tilde_prev = tilde
        omega = tilde_dot - k_heading * _wrap_angle(state[2] - tilde)
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
            print(f"iter={step} dist_to_goal={dist:.3f}")
        if np.linalg.norm(state[:2] - goal) < rad_from_goal:
            break

    traj = np.array(traj)
    controls = np.array(controls, dtype=float)

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
        f"avg_iteration={_format_duration(avg_control)}"
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
    plot_obstacle(
        ax,
        obstacle.center,
        obstacle.r1,
        obstacle.r2,
        p=obstacle.p,
        angle=obstacle.angle,
        color="0.3",
        fill=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Unicycle - Static Obstacle (Density Feedback)")
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

        def init():
            line.set_data([], [])
            return line, agent

        def update(i):
            line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            agent.set_xy(_triangle_points(traj[i, :2], traj[i, 2], agent_radius))
            return line, agent

        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=range(0, len(traj), animation_stride),
            interval=20,
            blit=True,
            repeat=False,
        )
        if save_animation:
            animation_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if animation_format == "mp4":
                    writer = animation.FFMpegWriter(fps=animation_fps)
                else:
                    writer = animation.PillowWriter(fps=animation_fps)
                ani.save(animation_path, writer=writer)
            except Exception:
                if animation_format == "mp4":
                    fallback = animation_path.with_suffix(".gif")
                    ani.save(fallback, writer=animation.PillowWriter(fps=animation_fps))
                else:
                    raise
    else:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()





