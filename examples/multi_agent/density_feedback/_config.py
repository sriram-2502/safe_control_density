from pathlib import Path
import argparse
import sys

import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
MULTI_AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(MULTI_AGENT_ROOT), str(REPO_ROOT), str(EXAMPLE_ROOT / "unicycle")]

from _plotting import add_animation_save_args


SCENARIO_CHOICES = ("crossing2", "crossing4", "crossing6", "swap8", "swap8_opposite", "swap10", "swap12")
SCENARIO_ALIASES = {
    "pair_crossing": "crossing2",
    "lanes6": "crossing6",
    "circle": "swap8",
    "scenario1": "crossing4",
    "scenario2": "crossing6",
    "scenario3": "swap8",
}
LEGACY_FLAG_SCENARIOS = {"scenario1": "crossing4", "scenario2": "crossing6", "scenario3": "swap8"}

MODE_LABELS = {
    "reactive_density_feedback": "Reactive Density Feedback",
    "collision_cone_density_feedback": "Collision Cone Density Feedback",
    "velocity_obstacle_density_feedback": "Velocity Obstacle Density Feedback",
    "density_filter": "Density Filter",
    "density_mpc": "Density MPC",
    "density_filter_reactive": "Density Filter - Reactive",
    "density_filter_collision_cone": "Density Filter - Collision Cone",
    "density_filter_velocity_obstacle": "Density Filter - Velocity Obstacle",
    "density_mpc_reactive": "Density MPC - Reactive",
    "density_mpc_collision_cone": "Density MPC - Collision Cone",
    "density_mpc_velocity_obstacle": "Density MPC - Velocity Obstacle",
}


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def mode_label(mode):
    return MODE_LABELS.get(mode, mode.replace("_", " ").title())


def directed_scenario(starts, goals, agent_radius, sensing_margin):
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    headings = np.arctan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
    agent_r1 = np.full(starts.shape[0], agent_radius, dtype=float)
    agent_r2 = np.full(starts.shape[0], agent_radius + sensing_margin, dtype=float)
    return starts, goals, headings, agent_r1, agent_r2


def swap_scenario(num_agents, start_offset, agent_radius):
    circle_radius = 3.35 if num_agents >= 12 else 2.2 + 0.16 * max(0, num_agents - 8)
    shift = 2 if num_agents >= 12 else 3
    angles = np.linspace(0.0, 2.0 * np.pi, num_agents, endpoint=False)
    starts = np.stack([circle_radius * np.cos(angles), circle_radius * np.sin(angles)], axis=1)
    starts = starts + np.asarray(start_offset, dtype=float)
    goals = np.roll(starts, shift=shift, axis=0)
    sensing_margin = 0.28 if num_agents <= 8 else 0.24
    return directed_scenario(starts, goals, agent_radius, sensing_margin)


def swap_opposite_scenario(num_agents, start_offset, agent_radius):
    circle_radius = 3.15
    angles = np.linspace(0.0, 2.0 * np.pi, num_agents, endpoint=False)
    starts = np.stack([circle_radius * np.cos(angles), circle_radius * np.sin(angles)], axis=1)
    starts = starts + np.asarray(start_offset, dtype=float)
    goals = np.roll(starts, shift=num_agents // 2, axis=0)
    return directed_scenario(starts, goals, agent_radius, 0.22)


def make_scenario(args, interaction_mode):
    scenario = SCENARIO_ALIASES.get(args.scenario, args.scenario)
    if scenario == "crossing2":
        return directed_scenario(
            starts=[[-2.4, -1.0], [2.4, 1.0]],
            goals=[[2.4, 1.0], [-2.4, -1.0]],
            agent_radius=args.agent_radius,
            sensing_margin=0.26,
        )
    if scenario == "crossing4":
        if interaction_mode == "density_filter_reactive":
            return directed_scenario(
                starts=[[-2.0, -1.65], [2.08, 1.59], [-2.0, 1.67], [2.08, -1.49]],
                goals=[[2.0, 1.65], [-2.08, -1.59], [2.0, -1.67], [-2.08, 1.49]],
                agent_radius=args.agent_radius,
                sensing_margin=0.32,
            )
        return directed_scenario(
            starts=[[-2.0, -1.65], [2.0, 1.65], [-2.0, 1.55], [2.0, -1.55]],
            goals=[[2.0, 1.65], [-2.0, -1.65], [2.0, -1.55], [-2.0, 1.55]],
            agent_radius=args.agent_radius,
            sensing_margin=0.32,
        )
    if scenario == "crossing6":
        right_offset = (
            np.asarray(args.reactive_crossing6_offset, dtype=float)
            if interaction_mode == "reactive_density_feedback"
            else np.zeros(2, dtype=float)
        )
        return directed_scenario(
            starts=[
                [-2.5, -1.20],
                [-2.5, 0.00],
                [-2.5, 1.20],
                [2.5 + right_offset[0], -1.20 + right_offset[1]],
                [2.5 + right_offset[0], 0.00 + right_offset[1]],
                [2.5 + right_offset[0], 1.20 + right_offset[1]],
            ],
            goals=[
                [2.5, 1.20],
                [2.5, 0.00],
                [2.5, -1.20],
                [-2.5 + right_offset[0], 1.20 + right_offset[1]],
                [-2.5 + right_offset[0], 0.00 + right_offset[1]],
                [-2.5 + right_offset[0], -1.20 + right_offset[1]],
            ],
            agent_radius=args.agent_radius,
            sensing_margin=0.30,
        )
    if scenario.startswith("swap"):
        if scenario.endswith("_opposite"):
            return swap_opposite_scenario(int(scenario.removeprefix("swap").removesuffix("_opposite")), args.start_offset, args.agent_radius)
        return swap_scenario(int(scenario.removeprefix("swap")), args.start_offset, args.agent_radius)
    raise ValueError(f"unknown scenario: {args.scenario}")


def add_common_arguments(parser):
    add_animation_save_args(parser)
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    parser.add_argument("--steps", type=int, default=40000, help="Maximum simulation steps.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--scenario", choices=(*SCENARIO_CHOICES, *SCENARIO_ALIASES.keys()), default="crossing2")
    parser.add_argument("--cone-density-margin", type=float, default=0.45)
    parser.add_argument("--cone-density-neighbors", type=int, default=0)
    parser.add_argument("--reactive-crossing6-offset", nargs=2, type=float, default=(0.039, 0.0), metavar=("DX", "DY"))
    parser.add_argument("--scenario1", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario2", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario3", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-agents", type=int, default=8, choices=(2, 4, 6, 8, 10, 12))
    parser.add_argument("--start-offset", nargs=2, type=float, default=(0.0, 0.0), metavar=("DX", "DY"))
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--ctrl-multiplier", type=float, default=10.0)
    parser.add_argument("--rad-from-goal", type=float, default=0.05)
    parser.add_argument("--v-max", type=float, default=1.5)
    parser.add_argument("--omega-max", type=float, default=2.5)
    parser.add_argument("--k-heading", type=float, default=3.0)
    parser.add_argument("--agent-radius", type=float, default=0.12)
    parser.add_argument("--sensing-margin", type=float, default=0.4)
    parser.add_argument("--animation-stride", type=int, default=4)
    parser.add_argument("--animation-fps", type=int, default=20)
    parser.add_argument("--print-interval", type=int, default=500)
    parser.add_argument("--log-timing", action="store_true", help="Keep per-iteration timing samples.")


def finalize_args(args):
    for legacy_name, scenario_name in LEGACY_FLAG_SCENARIOS.items():
        if getattr(args, legacy_name):
            args.scenario = scenario_name
    args.scenario = SCENARIO_ALIASES.get(args.scenario, args.scenario)
    return args


def min_pair_clearance(states, agent_r1):
    min_clearance = np.inf
    for i in range(states.shape[0]):
        for j in range(i + 1, states.shape[0]):
            distance = float(np.linalg.norm(states[i, :2] - states[j, :2]))
            clearance = distance - float(agent_r1[i] + agent_r1[j])
            min_clearance = min(min_clearance, clearance)
    return min_clearance
