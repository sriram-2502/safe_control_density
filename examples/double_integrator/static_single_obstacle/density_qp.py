from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from density_utils.controllers import solve_density_qp
from density_utils.density import (
    Obstacle,
    finite_difference_grad,
    full_state_density_grad,
    full_state_density_value,
)
from density_utils.utils import plot_goal, plot_obstacle, plot_start
from density_utils.utils.timing import TimedBlock


def _p_norm_distance(x, obs):
    dx = x - obs.center
    if obs.angle:
        c = np.cos(-obs.angle)
        s = np.sin(-obs.angle)
        dx = np.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]])
    if obs.scale is not None:
        dx = dx / obs.scale
    return np.sum(np.abs(dx) ** obs.p) ** (1.0 / obs.p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run the simulation without opening plots.")
    parser.add_argument("--steps", type=int, default=8000, help="Maximum simulation steps.")
    args = parser.parse_args()

    dt = 0.01
    density_dt = 0.25
    steps = args.steps
    alpha = 0.1
    ctrl_multiplier = 0.8
    rad_from_goal = 1.0
    stop_tol = 0.01
    stop_steps = 500
    stop_when_stable = True
    saturation = 2.0
    kp = 1.0
    kd = 1.6
    cdf_rate = 0.1
    slack_weight = 1e7
    animate = not args.no_plot
    save_animation = args.save_gif
    animation_stride = 10
    animation_fps = 20
    animation_format = "gif"
    animation_path = Path("animations") / f"double_integrator_static_qp.{animation_format}"

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
    control_margin = 0.6
    control_obstacle = Obstacle(
        center=inflated_obstacle.center,
        r1=inflated_obstacle.r1,
        r2=inflated_obstacle.r2 + control_margin,
        p=inflated_obstacle.p,
        scale=inflated_obstacle.scale,
        angle=inflated_obstacle.angle,
    )

    state = np.array([start[0], start[1], 0.0, 0.0], dtype=float)
    goal_state = np.array([goal[0], goal[1], 0.0, 0.0], dtype=float)
    A = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    density_P = np.array(
        [
            [1.0, 0.0, 0.35, 0.0],
            [0.0, 1.0, 0.0, 0.35],
            [0.35, 0.0, 0.5, 0.0],
            [0.0, 0.35, 0.0, 0.5],
        ],
        dtype=float,
    )

    def cdf_density(x_eval, goal_eval, alpha_eval, obstacles_eval):
        return full_state_density_value(
            x_eval,
            goal_eval,
            alpha_eval,
            obstacles_eval,
            position_indices=(0, 1),
            P=density_P,
        )

    def cdf_density_grad(x_eval, goal_eval, alpha_eval, obstacles_eval):
        return full_state_density_grad(
            x_eval,
            goal_eval,
            alpha_eval,
            obstacles_eval,
            position_indices=(0, 1),
            P=density_P,
        )

    def double_integrator_step(x_eval, u_eval, dt_eval):
        x_next = np.asarray(x_eval, dtype=float).copy()
        u_eval = np.asarray(u_eval, dtype=float)
        x_next[:2] = x_next[:2] + x_next[2:] * dt_eval + 0.5 * u_eval * dt_eval**2
        x_next[2:] = x_next[2:] + u_eval * dt_eval
        return x_next

    def density_ascent_nominal(x_eval, dist):
        if dist < rad_from_goal:
            return kp * (goal - x_eval[:2]) - kd * x_eval[2:]
        grad_u = finite_difference_grad(
            lambda u_eval: cdf_density(
                double_integrator_step(x_eval, u_eval, density_dt),
                goal_state,
                alpha,
                [control_obstacle],
            ),
            np.zeros(2, dtype=float),
            eps=1e-3,
        )
        max_grad = np.max(np.abs(grad_u))
        if max_grad < 1e-12:
            return kp * (goal - x_eval[:2]) - kd * x_eval[2:]
        return ctrl_multiplier * grad_u / max_grad

    traj = [state.copy()]
    controls = []
    slacks = []
    solver_failures = 0
    min_clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1

    control_time = 0.0
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    stop_count = 0
    for step in range(steps):
        dist = np.linalg.norm(state[:2] - goal)
        with timer:
            u_nom = density_ascent_nominal(state, dist)
            result = solve_density_qp(
                state,
                goal_state,
                alpha,
                [control_obstacle],
                dynamics=(A, B),
                u_nom=u_nom,
                dt=density_dt,
                rad_from_goal=rad_from_goal,
                saturation=saturation,
                cdf_rate=cdf_rate,
                slack_weight=slack_weight,
                density_fn=cdf_density,
                density_grad_fn=cdf_density_grad,
                constraint_mode="discrete",
                next_state_fn=double_integrator_step,
                return_info=True,
            )
            u = result.u
        control_time += timer.last
        controls.append(u.copy())
        slacks.append(float(np.max(result.slack)) if result.slack.size else 0.0)
        if not result.success:
            solver_failures += 1

        state[:2] = state[:2] + state[2:] * dt
        state[2:] = state[2:] + u * dt
        traj.append(state.copy())
        clearance = _p_norm_distance(state[:2], inflated_obstacle) - inflated_obstacle.r1
        min_clearance = min(min_clearance, clearance)

        speed = np.linalg.norm(state[2:])
        if stop_when_stable:
            if dist < stop_tol and speed < stop_tol:
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0
        if step % print_interval == 0:
            print(
                f"iter={step} dist_to_goal={dist:.3f} speed={speed:.3f} "
                f"clearance={clearance:.3f} slack={slacks[-1]:.2e}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2, dtype=float))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    traj = np.array(traj)
    controls = np.array(controls)
    slacks = np.array(slacks)

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
        f"avg_iteration={_format_duration(avg_control)} "
        f"min_clearance={min_clearance:.4f} "
        f"max_slack={np.max(slacks):.2e} "
        f"solver_failures={solver_failures}"
    )

    if args.no_plot:
        return

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
    ax.set_title("Double Integrator - Static Obstacle (Density QP)")
    ax.grid(True, linestyle="--", alpha=0.4)

    if animate:
        line, = ax.plot([], [], color="tab:blue", linewidth=2)
        agent = patches.Circle(traj[0, :2], agent_radius, color="tab:blue", zorder=4)
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

        def init():
            line.set_data([], [])
            return line, agent, heading

        def update(i):
            line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            agent.center = (traj[i, 0], traj[i, 1])
            vel = traj[i, 2:]
            speed = np.linalg.norm(vel)
            if speed < 1e-6:
                ux, uy = 0.0, 0.0
            else:
                heading_len = 0.35
                ux, uy = vel / speed * heading_len
            heading.set_offsets([traj[i, 0], traj[i, 1]])
            heading.set_UVC([ux], [uy])
            return line, agent, heading

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
