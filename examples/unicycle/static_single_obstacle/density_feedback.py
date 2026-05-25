from pathlib import Path
import argparse
import sys

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import density_feedback_control
from density_utils.density import Obstacle
from density_utils.dynamics import unicycle_step
from density_utils.utils.timing import TimedBlock

from _plotting import plot_unicycle_results


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    args = parser.parse_args()

    dt = 0.01
    steps = 4000
    alpha = 0.4
    ctrl_multiplier = 3.0
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
    animate = not args.no_plot
    save_animation = args.save_gif
    animation_stride = 10
    animation_fps = 30
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

    if not args.no_plot:
        plot_unicycle_results(
            traj=traj,
            controls=controls,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=[obstacle],
            agent_radius=agent_radius,
            title="Unicycle - Static Obstacle (Density Feedback)",
            animate=animate,
            save_animation=save_animation,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
        )


if __name__ == "__main__":
    main()





