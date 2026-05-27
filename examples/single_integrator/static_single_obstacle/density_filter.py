from pathlib import Path
import argparse
import sys

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(EXAMPLE_ROOT)]

from density_utils.controllers import SOLVER_CHOICES, single_integrator_nominal_control, solve_discrete_density_filter
from density_utils.density import Obstacle
from density_utils.dynamics import single_integrator_step
from density_utils.sim import forward_euler
from density_utils.utils.timing import TimedBlock

from _plotting import plot_single_integrator_results


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
    parser.add_argument("--steps", type=int, default=4000, help="Maximum simulation steps.")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto", help="Optimizer backend.")
    parser.add_argument("--verbose", action="store_true", help="Print solver failure diagnostics.")
    parser.add_argument(
        "--u-nom",
        default="density",
        choices=("goal", "lqr", "density", "density_blend", "pure_pursuit"),
        help="Nominal controller used as the Density filter reference.",
    )
    args = parser.parse_args()

    dt = 0.1
    steps = args.steps
    alpha = 0.1
    ctrl_multiplier = 1.0
    rad_from_goal = 0.35
    stop_tol = min(0.005, rad_from_goal)
    stop_steps = 500
    stop_when_stable = True
    q_lqr = 4.0
    r_lqr = 1.0
    u_max = np.array([2.0, 2.0])
    u_min = -u_max
    slack_weight = 1e4
    animate = not args.no_plot
    save_animation = args.save_gif
    animation_stride = 10
    animation_fps = 20
    animation_format = "gif"
    animation_path = Path("animations") / f"single_integrator_static_filter.{animation_format}"

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

    x = start.copy()
    traj = [x.copy()]
    controls = []
    slacks = []
    solver_failures = 0
    min_clearance = _p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1

    control_time = 0.0
    control_times = []
    log_timing = False
    timer = TimedBlock(enabled=log_timing)
    print_interval = 500
    stop_count = 0
    for step in range(steps):
        dist = np.linalg.norm(x - goal)
        with timer:
            u_nom = single_integrator_nominal_control(
                x,
                goal,
                alpha,
                [inflated_obstacle],
                mode=args.u_nom,
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
                solver=args.solver,
                return_info=True,
            )
            u = filter_result.u
        dt_control = timer.last
        control_time += dt_control
        if log_timing and dist >= rad_from_goal:
            control_times.append(dt_control)
        controls.append(u.copy())
        slacks.append(float(np.max(filter_result.slack)) if filter_result.slack.size else 0.0)
        if not filter_result.success:
            solver_failures += 1

        x = forward_euler(x, u, dt)
        traj.append(x.copy())
        clearance = _p_norm_distance(x, inflated_obstacle) - inflated_obstacle.r1
        min_clearance = min(min_clearance, clearance)

        if stop_when_stable:
            if dist < stop_tol:
                stop_count += 1
                if stop_count >= stop_steps:
                    print(f"stopping at iter={step} (stable within stop_tol)")
                    break
            else:
                stop_count = 0
        if step % print_interval == 0:
            print(
                f"iter={step} dist_to_goal={dist:.3f} "
                f"clearance={clearance:.3f} slack={slacks[-1]:.2e}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros_like(traj[0]))
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
    summary = (
        "steps="
        f"{steps_taken} "
        f"sim_time={_format_duration(control_time)} "
        f"avg_iteration={_format_duration(avg_control)} "
        f"min_clearance={min_clearance:.4f} "
        f"max_slack={np.max(slacks):.2e}"
    )
    if args.verbose:
        summary += f" solver_failures={solver_failures}"
    print(summary)
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
            slacks=slacks,
            dt=dt,
            start=start,
            goal=goal,
            obstacles=[obstacle],
            agent_radius=agent_radius,
            title="Single Integrator - Static Obstacle (Density filter)",
            animate=animate,
            save_animation=save_animation,
            animation_path=animation_path,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
        )

if __name__ == "__main__":
    main()
