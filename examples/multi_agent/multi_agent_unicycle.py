import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from density_utils.controllers import density_feedback_control
from density_utils.density import Obstacle
from density_utils.dynamics import unicycle_step
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
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10,
        choices=[2, 4, 6, 10],
        help="Number of agents on the circle (2, 4, 6, or 10).",
    )
    parser.add_argument(
        "--scenario1",
        action="store_true",
        help="Match multiagent_unicycle_scenario1.m parameters.",
    )
    parser.add_argument(
        "--scenario2",
        action="store_true",
        help="Match multiagent_unicycle_scenario2.m parameters.",
    )
    parser.add_argument(
        "--scenario3",
        action="store_true",
        help="8-agent square scenario with offsets.",
    )
    parser.add_argument(
        "--start-offset",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        metavar=("DX", "DY"),
        help="XY offset applied to all start positions.",
    )
    args = parser.parse_args()

    dt = 0.01
    steps = 40000
    alpha = 0.2
    ctrl_multiplier = 10.0
    rad_from_goal = 0.05
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    stop_when_stable = True
    q_lqr = 1.0
    r_lqr = 1.0
    saturation = 4.0
    k_heading = 3.0
    v_max = 1.5
    omega_max = 2.5
    agent_radius = 0.12
    sensing_margin = 0.4

    animate = True
    save_animation = args.save_gif
    animation_stride = 4
    animation_fps = 20
    animation_path = Path("animations") / "multi_agent_unicycle.gif"

    if args.scenario1 or args.scenario2 or args.scenario3:
        env_size = 5.0
        offset = 0.1
        if args.scenario3:
            starts = np.array(
                [
                    [env_size, env_size],
                    [-env_size, -env_size],
                    [env_size, -env_size],
                    [-env_size, env_size],
                    [0.0, env_size],
                    [0.0, -env_size],
                    [-env_size, 0.0],
                    [env_size, 0.0],
                ],
                dtype=float,
            )
            goals = np.array(
                [
                    [-env_size - offset, -env_size + offset],
                    [env_size + offset, env_size + offset],
                    [-env_size, env_size + offset],
                    [env_size, -env_size - offset],
                    [-offset, -env_size],
                    [offset, env_size],
                    [env_size + offset, 0.0 + offset],
                    [-env_size - offset, 0.0 - offset],
                ],
                dtype=float,
            )
            headings = np.array(
                [
                    np.pi / 4,
                    3 * np.pi / 4,
                    -np.pi / 4,
                    -3 * np.pi / 4,
                    -np.pi / 2,
                    np.pi / 2,
                    0.0,
                    np.pi,
                ]
            )
        elif args.scenario2:
            starts = np.array(
                [
                    [env_size, env_size],
                    [-env_size, -env_size],
                    [env_size, -env_size],
                    [-env_size, env_size],
                    [0.0, env_size],
                    [0.0, -env_size],
                ],
                dtype=float,
            )
            goals = np.array(
                [
                    [-env_size - offset, -env_size],
                    [env_size, env_size + offset],
                    [-env_size, env_size + offset],
                    [env_size, -env_size - offset],
                    [-offset, -env_size],
                    [offset, env_size],
                ],
                dtype=float,
            )
            headings = np.array(
                [np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4, -np.pi / 2, np.pi / 2]
            )
        else:
            starts = np.array(
                [
                    [env_size, env_size],
                    [-env_size, -env_size],
                    [env_size, -env_size],
                    [-env_size, env_size],
                ],
                dtype=float,
            )
            goals = np.array(
                [
                    [-env_size - offset, -env_size],
                    [env_size, env_size + offset],
                    [-env_size, env_size + offset],
                    [env_size, -env_size - offset],
                ],
                dtype=float,
            )
            headings = np.array([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4])
        if args.scenario3:
            agent_r1 = np.full(8, 0.5, dtype=float)
            agent_r2 = np.full(8, 2.0, dtype=float)
        elif args.scenario2:
            agent_r1 = np.full(6, 0.5, dtype=float)
            agent_r2 = np.full(6, 2.0, dtype=float)
        else:
            agent_r1 = np.full(4, 0.5, dtype=float)
            agent_r2 = np.full(4, 2.0, dtype=float)
    else:
        num_agents = args.num_agents
        circle_radius = 2.0
        angles = np.linspace(0.0, 2.0 * np.pi, num_agents, endpoint=False)
        starts = np.stack(
            [circle_radius * np.cos(angles), circle_radius * np.sin(angles)], axis=1
        )
        starts = starts + np.array(args.start_offset, dtype=float)
        goals = -starts.copy()
        headings = np.arctan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
        agent_r1 = np.full(starts.shape[0], agent_radius, dtype=float)
        agent_r2 = np.full(starts.shape[0], agent_radius + sensing_margin, dtype=float)
    states = np.hstack([starts, headings[:, None]])
    tilde_prev = headings.copy()

    traj = [states.copy()]
    control_time = np.zeros(states.shape[0], dtype=float)
    timers = [TimedBlock() for _ in range(states.shape[0])]

    print_interval = 500
    stop_count = 0
    for step in range(steps):
        new_states = states.copy()
        for j in range(states.shape[0]):
            pos = states[j, :2]
            goal = goals[j]

            obstacles = []
            for k in range(states.shape[0]):
                if k == j:
                    continue
                obstacles.append(
                    Obstacle(
                        center=states[k, :2],
                        r1=agent_r1[k],
                        r2=agent_r2[k],
                        p=2.0,
                    )
                )

            with timers[j]:
                u = density_feedback_control(
                    pos,
                    goal,
                    alpha,
                    obstacles,
                    ctrl_multiplier=ctrl_multiplier,
                    rad_from_goal=rad_from_goal,
                    q_lqr=q_lqr,
                    r_lqr=r_lqr,
                    dt=dt,
                    saturation=saturation,
                )
            control_time[j] += timers[j].samples[-1]
            v = float(np.linalg.norm(u))
            v = min(v, v_max)

            tilde = float(np.arctan2(u[1], u[0]))
            tilde_dot = _wrap_angle(tilde - tilde_prev[j]) / dt
            tilde_prev[j] = tilde

            theta = states[j, 2]
            omega = tilde_dot - k_heading * _wrap_angle(theta - tilde)
            omega = float(np.clip(omega, -omega_max, omega_max))

            new_states[j] = unicycle_step(states[j], v, omega, dt)

        states = new_states
        traj.append(states.copy())
        if stop_when_stable:
            if dist < stop_tol:
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0

        dists = np.linalg.norm(states[:, :2] - goals, axis=1)
        if step % print_interval == 0:
            print("iter=" + str(step) + " dists=" + np.array2string(dists, precision=3))

        if np.all(dists < rad_from_goal):
            print(f"stopping at iter={step} (all agents within rad_from_goal)")
            print("final_dists=" + np.array2string(dists, precision=3))
            break

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    avg_control_ms = avg_control * 1e3
    if log_timing:
        for idx, timer in enumerate(timers, start=1):
            mean_ms, std_ms = timer.mean_std_ms()
            if mean_ms is not None:
                print(f"agent_{idx}_avg_iteration={mean_ms:.3f} [ms] std={std_ms:.3f} [ms]")
    else:
        for idx, avg_ms in enumerate(avg_control_ms, start=1):
            print(f"agent_{idx}_avg_iteration={avg_ms:.3f} [ms]")

    traj = np.array(traj)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Unicycle Intersection - Multi-Agent Density Feedback")
    ax.grid(True, linestyle="--", alpha=0.4)

    colors = plt.cm.tab10(np.linspace(0, 1, starts.shape[0]))
    for idx, goal in enumerate(goals):
        ax.scatter(
            starts[idx, 0],
            starts[idx, 1],
            s=90,
            c=[colors[idx]],
            marker="*",
            edgecolor="k",
            zorder=3,
        )
    # Skip start markers to keep the plot clean.

    if animate:
        agent_patches = []
        heading_quivers = []
        trail_lines = []
        for idx in range(starts.shape[0]):
            tri = _triangle_points(traj[0, idx, :2], traj[0, idx, 2], agent_r1[idx])
            agent = patches.Polygon(tri, closed=True, color=colors[idx], zorder=4)
            ax.add_patch(agent)
            agent_patches.append(agent)
            heading = ax.quiver(
                traj[0, idx, 0],
                traj[0, idx, 1],
                np.cos(traj[0, idx, 2]),
                np.sin(traj[0, idx, 2]),
                angles="xy",
                scale_units="xy",
                scale=2.0,
                color=colors[idx],
                width=0.004,
                zorder=5,
            )
            heading_quivers.append(heading)
            line, = ax.plot([], [], linewidth=1.5, color=colors[idx])
            trail_lines.append(line)

        def init():
            for line in trail_lines:
                line.set_data([], [])
            return (*agent_patches, *heading_quivers, *trail_lines)

        def update(i):
            for idx in range(traj.shape[1]):
                tri = _triangle_points(traj[i, idx, :2], traj[i, idx, 2], agent_r1[idx])
                agent_patches[idx].set_xy(tri)
                heading_quivers[idx].set_offsets([traj[i, idx, 0], traj[i, idx, 1]])
                heading_quivers[idx].set_UVC(
                    [np.cos(traj[i, idx, 2])],
                    [np.sin(traj[i, idx, 2])],
                )
                trail_lines[idx].set_data(traj[: i + 1, idx, 0], traj[: i + 1, idx, 1])
            return (*agent_patches, *heading_quivers, *trail_lines)

        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=range(0, traj.shape[0], animation_stride),
            interval=20,
            blit=True,
            repeat=False,
        )
        if save_animation:
            animation_path.parent.mkdir(parents=True, exist_ok=True)
            ani.save(animation_path, writer=animation.PillowWriter(fps=animation_fps))
    else:
        for idx in range(traj.shape[1]):
            ax.plot(traj[:, idx, 0], traj[:, idx, 1], linewidth=1.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()





