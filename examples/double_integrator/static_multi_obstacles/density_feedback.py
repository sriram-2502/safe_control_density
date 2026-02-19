from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib import animation, patches

from density_utils.density import Obstacle, density_grad
from density_utils.utils import plot_goal, plot_obstacle, plot_start
from density_utils.utils.timing import TimedBlock


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
    animation_path = Path("animations") / f"double_integrator_multi.{animation_format}"

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

    def p_norm_distance(x, obs):
        dx = x - obs.center
        if obs.angle:
            c = np.cos(-obs.angle)
            s = np.sin(-obs.angle)
            dx = np.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]])
        if obs.scale is not None:
            dx = dx / obs.scale
        return np.sum(np.abs(dx) ** obs.p) ** (1.0 / obs.p)

    if len(obstacles) != 10:
        raise ValueError("Expected 10 fixed obstacles.")

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

    for pt_name, pt in [("start", start), ("goal", goal)]:
        for obs in inflated_obstacles:
            if p_norm_distance(pt, obs) <= obs.r2:
                raise ValueError(f"{pt_name} is inside an obstacle sensing region")

    def _scaled_saturation(dist):
        if dist >= rad_from_goal:
            return saturation
        decay_length = max(rad_from_goal / 3.0, 1e-6)
        scale = 1.0 - np.exp(-dist / decay_length)
        return saturation * scale

    def backstepping_control(pos, vel, prev_v_des, sat):
        grad = density_grad(pos, goal, alpha, inflated_obstacles)
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
        with timer:
            if dist < rad_from_goal:
                if not lqr_triggered:
                    lqr_triggered = True
                u = -k_lqr @ (state - goal_state)
                max_u = np.max(np.abs(u))
                if max_u > sat:
                    u = u / max_u * sat
            else:
                u, v_des_prev = backstepping_control(state[:2], state[2:], v_des_prev, sat)
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
            print(f"iter={step} dist_to_goal={dist:.3f}")

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros_like(traj[0]))

    traj = np.array(traj)
    controls = np.array(controls)
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
    ax.set_title("Double Integrator - Multiple Obstacles (Density Feedback)")
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
            interval=15,
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






