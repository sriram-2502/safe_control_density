from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCENARIO_CHOICES = ("closing_in", "dense_flow")
METHOD_LABELS = {
    "reactive": "Reactive bump",
    "collision_cone": "Collision cone",
    "velocity_obstacle": "Velocity obstacle",
}


@dataclass(frozen=True)
class DynamicObstacle:
    center: np.ndarray
    radius: float
    velocity: np.ndarray
    bounds: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    start: np.ndarray
    goal: np.ndarray
    heading: float
    obstacles: tuple[DynamicObstacle, ...]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    steps: int
    start_delay_steps: int = 0
    streaming: None = None


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _obs(center, radius, velocity, bounds=None):
    return DynamicObstacle(
        center=np.asarray(center, dtype=float),
        radius=float(radius),
        velocity=np.asarray(velocity, dtype=float),
        bounds=bounds,
    )


def _surrounding_obstacles(center, radius, angles_deg, obstacle_radius, speed, tangent_speed, bounds):
    obstacles = []
    center = np.asarray(center, dtype=float)
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        radial = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        tangent = np.array([-radial[1], radial[0]], dtype=float)
        obs_center = center + float(radius) * radial
        obs_velocity = -float(speed) * radial + float(tangent_speed) * tangent
        obstacles.append(_obs(obs_center, obstacle_radius, obs_velocity, bounds))
    return tuple(obstacles)


def make_scenario(name):
    if name == "closing_in":
        bounds = (-4.0, 7.0, -4.0, 4.0)
        obstacles = _surrounding_obstacles(
            center=(0.0, 0.0),
            radius=3.20,
            angles_deg=(0, 45, 90, 135, 180, 225, 270, 315),
            obstacle_radius=0.37,
            speed=0.16,
            tangent_speed=0.06,
            bounds=bounds,
        )
        return Scenario(
            name=name,
            start=np.array([0.0, 0.0], dtype=float),
            goal=np.array([5.8, 0.8], dtype=float),
            heading=float(np.arctan2(0.8, 5.8)),
            obstacles=obstacles,
            xlim=(-4.0, 7.0),
            ylim=(-4.0, 4.0),
            steps=950,
            start_delay_steps=0,
        )
    if name == "dense_flow":
        bounds = (0.0, 16.0, 0.0, 12.0)
        obstacles = (
            _obs((4.0, 3.0), 0.34, (0.24, 0.08), bounds),
            _obs((4.8, 8.4), 0.34, (0.42, -0.28), bounds),
            _obs((5.5, 6.5), 0.36, (0.33, 0.16), bounds),
            _obs((6.8, 0.8), 0.28, (0.22, 0.20), bounds),
            _obs((7.4, 1.3), 0.30, (0.20, 0.18), bounds),
            _obs((7.6, 8.9), 0.36, (0.36, -0.38), bounds),
            _obs((8.3, 6.9), 0.33, (-0.30, -0.36), bounds),
            _obs((8.6, 4.3), 0.34, (-0.34, 0.42), bounds),
            _obs((9.7, 6.8), 0.36, (-0.38, -0.34), bounds),
            _obs((10.1, 3.7), 0.33, (-0.32, 0.38), bounds),
            _obs((10.7, 2.8), 0.34, (-0.36, 0.32), bounds),
            _obs((11.6, 8.0), 0.34, (-0.42, -0.30), bounds),
            _obs((12.6, 5.1), 0.36, (-0.34, 0.38), bounds),
            _obs((12.4, 10.4), 0.32, (-0.38, -0.12), bounds),
        )
        return Scenario(
            name=name,
            start=np.array([6.0, 4.0], dtype=float),
            goal=np.array([13.6, 8.8], dtype=float),
            heading=float(np.arctan2(4.8, 7.6)),
            obstacles=obstacles,
            xlim=(0.0, 16.0),
            ylim=(0.0, 12.0),
            steps=1100,
        )
    raise ValueError(f"unknown scenario: {name}")


def obstacle_arrays(obstacles):
    centers = np.asarray([obs.center for obs in obstacles], dtype=float)
    radii = np.asarray([obs.radius for obs in obstacles], dtype=float)
    velocities = np.asarray([obs.velocity for obs in obstacles], dtype=float)
    return centers, radii, velocities


def initialize_dynamic_obstacles(scenario, robot_pos=None):
    centers, radii, velocities = obstacle_arrays(scenario.obstacles)
    active = radii > 0.0
    return centers, radii, velocities, active, None


def step_obstacles(centers, velocities, obstacles, dt):
    new_centers = np.asarray(centers, dtype=float).copy()
    new_velocities = np.asarray(velocities, dtype=float).copy()
    for idx, obs in enumerate(obstacles):
        new_centers[idx] += new_velocities[idx] * float(dt)
        if obs.bounds is None:
            continue
        xmin, xmax, ymin, ymax = obs.bounds
        radius = float(obs.radius)
        if new_centers[idx, 0] < xmin + radius:
            new_centers[idx, 0] = xmin + radius
            new_velocities[idx, 0] = abs(new_velocities[idx, 0])
        elif new_centers[idx, 0] > xmax - radius:
            new_centers[idx, 0] = xmax - radius
            new_velocities[idx, 0] = -abs(new_velocities[idx, 0])
        if new_centers[idx, 1] < ymin + radius:
            new_centers[idx, 1] = ymin + radius
            new_velocities[idx, 1] = abs(new_velocities[idx, 1])
        elif new_centers[idx, 1] > ymax - radius:
            new_centers[idx, 1] = ymax - radius
            new_velocities[idx, 1] = -abs(new_velocities[idx, 1])
    return new_centers, new_velocities


def step_dynamic_obstacles(centers, radii, velocities, active, scenario, robot_pos, dt, step, rng):
    new_centers, new_velocities = step_obstacles(centers, velocities, scenario.obstacles, dt)
    return new_centers, radii, new_velocities, active, 0


def min_obstacle_clearance(state, centers, radii, robot_radius):
    radii = np.asarray(radii, dtype=float)
    active = radii > 0.0
    if not np.any(active):
        return float("inf")
    distances = np.linalg.norm(np.asarray(centers, dtype=float)[active] - np.asarray(state[:2], dtype=float), axis=1)
    return float(np.min(distances - radii[active] - float(robot_radius)))


def add_common_arguments(parser):
    from _plotting import add_animation_save_args

    add_animation_save_args(parser)
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="closing_in")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    parser.add_argument("--steps", type=int, default=None, help="Maximum simulation steps.")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--ctrl-multiplier", type=float, default=30.0)
    parser.add_argument("--rad-from-goal", type=float, default=0.12)
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--sensing-margin", type=float, default=0.75)
    parser.add_argument("--v-max", type=float, default=1.25)
    parser.add_argument("--omega-max", type=float, default=2.0)
    parser.add_argument("--k-heading", type=float, default=3.0)
    parser.add_argument("--cone-density-margin", type=float, default=0.45)
    parser.add_argument("--dynamic-neighbors", type=int, default=3)
    parser.add_argument("--animation-stride", type=int, default=5)
    parser.add_argument("--animation-fps", type=int, default=20)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--log-timing", action="store_true", help="Keep per-iteration timing samples.")


def finalize_args(args):
    scenario = make_scenario(args.scenario)
    if args.steps is None:
        args.steps = scenario.steps
    return args


def example_root():
    return Path(__file__).resolve().parent
