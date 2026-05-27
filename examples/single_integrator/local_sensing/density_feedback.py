from pathlib import Path
import argparse
import sys

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import density_feedback_control
from density_utils.density import Obstacle
from density_utils.sim import forward_euler
from density_utils.utils.timing import TimedBlock

from _plotting import plot_single_integrator_results


def _angle_wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def detect_sensed_obstacles(pos, heading, obstacles, cam_range, fov_angle):
    """Range + FOV filtering based on robot heading."""
    pos = np.asarray(pos, dtype=float)
    sensed = []
    for obs in obstacles:
        rel = obs.center - pos
        dist = np.linalg.norm(rel)
        if dist > cam_range:
            continue
        if fov_angle < 2.0 * np.pi:
            angle_to_obs = np.arctan2(rel[1], rel[0])
            if abs(_angle_wrap(angle_to_obs - heading)) > fov_angle / 2.0:
                continue
        sensed.append((dist, obs))
    sensed.sort(key=lambda item: item[0])
    return [obs for _, obs in sensed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    args = parser.parse_args()

    dt = 0.001
    steps = 40000
    alpha = 0.4
    ctrl_multiplier = 4.0
    rad_from_goal = 0.01
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    stop_when_stable = True
    q_lqr = 4.0
    r_lqr = 1.0
    saturation = 4.0
    animate = not args.no_plot
    save_animation = args.save_gif
    animation_stride = 100
    animation_fps = 20
    animation_format = "gif"
    animation_path = Path("animations") / f"single_integrator_multi_local.{animation_format}"

    # Local sensing settings (safe_control-like)
    cam_range = 1.0
    fov_angle = np.deg2rad(80.0)
    max_sensed = 5
    linger_steps = 1000

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

    x = start.copy()
    traj = [x.copy()]

    controls = []
    headings = []
    sensed_buffer = {}
    control_time = 0.0
    control_times = []
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    stop_count = 0
    for step in range(steps):
        heading = np.arctan2(goal[1] - x[1], goal[0] - x[0])
        sensed = detect_sensed_obstacles(x, heading, inflated_obstacles, cam_range, fov_angle)
        sensed = sensed[:max_sensed]
        for obs in sensed:
            sensed_buffer[id(obs)] = linger_steps
        for obs_id in list(sensed_buffer.keys()):
            sensed_buffer[obs_id] -= 1
            if sensed_buffer[obs_id] <= 0:
                sensed_buffer.pop(obs_id)
        buffered = [obs for obs in inflated_obstacles if id(obs) in sensed_buffer]

        dist = np.linalg.norm(x - goal)
        with timer:
            u = density_feedback_control(
                x,
                goal,
                alpha,
                buffered,
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
        headings.append(float(np.arctan2(u[1], u[0])) if np.linalg.norm(u) > 1e-6 else heading)
        controls.append(u.copy())
        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        if stop_when_stable:
            if dist < stop_tol:
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0

        if step % print_interval == 0:
            dist = np.linalg.norm(x - goal)
            print(
                f"iter={step} dist_to_goal={dist:.3f} sensed={len(sensed)} buffered={len(buffered)}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros_like(traj[0]))
    if len(headings) < len(traj):
        headings.append(headings[-1] if headings else 0.0)

    traj = np.array(traj)
    controls = np.array(controls)
    headings = np.array(headings)

    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    if avg_control < 1.0:
        avg_str = f"{avg_control * 1e3:.1f} ms"
    else:
        avg_str = f"{avg_control:.2f} s"
    print(f"steps={steps_taken} avg_iteration={avg_str}")
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
        plot_single_integrator_results(
            traj=traj,
            controls=controls,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=obstacles,
            agent_radius=agent_radius,
            title="Single Integrator - Local Sensing Density Feedback",
            animate=animate,
            save_animation=save_animation,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
            headings=headings,
            inflated_obstacles=inflated_obstacles,
            fov_angle=fov_angle,
            cam_range=cam_range,
            max_sensed=max_sensed,
            animation_interval=15,
        )


if __name__ == "__main__":
    main()








