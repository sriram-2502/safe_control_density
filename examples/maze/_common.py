from pathlib import Path
import argparse
from dataclasses import dataclass
import heapq
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from density_utils.controllers import solve_density_mpc, solve_discrete_density_filter
from density_utils.density import Obstacle, p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils import plot_goal, plot_start


@dataclass(frozen=True)
class RectangleObstacle:
    center: np.ndarray
    half_extents: np.ndarray
    r1: float
    r2: float


ASCII_MAPS = {
    "wide": (
        "###############################",
        "#.............................#",
        "#...S.........................#",
        "#........########.............#",
        "#........#......#.............#",
        "#........#......#.....####....#",
        "#........#......#.....#.......#",
        "#........########.....#.......#",
        "#.....................#.......#",
        "#....................#...G....#",
        "#.............................#",
        "#.............................#",
        "###############################",
    ),
    "wide_s": (
        "###############################",
        "#.............................#",
        "#...S....###########..........#",
        "#........#.........#..........#",
        "#........#.........#..........#",
        "#.......................###...#",
        "#........##########......#....#",
        "#.........................#...#",
        "#.....#####...................#",
        "#..........#.......#####......#",
        "#..........#..........G.......#",
        "#.............................#",
        "###############################",
    ),
    "multi_room": (
        "###############################",
        "#.............................#",
        "#...S.........................#",
        "#........##...................#",
        "#........#......######........#",
        "#..............#......#.......#",
        "#....###.......#......#.......#",
        "#......#.......#..............#",
        "#......#.......######.........#",
        "#.......#.....................#",
        "#........#####.......###......#",
        "#.......................#..G..#",
        "#.............................#",
        "###############################",
    ),
    "rooms": (
        "###############################",
        "#.............................#",
        "#...S...######......#####.....#",
        "#.....#......#.....#.....#....#",
        "#............#...........#....#",
        "#....#.......#####.......#....#",
        "#....#...................#....#",
        "#.....######.......######.....#",
        "#............#................#",
        "#............#......####......#",
        "#.......................#..G..#",
        "#.............................#",
        "###############################",
    ),
    "narrow_s": (
        "###############################",
        "#S............................#",
        "#.....###########.............#",
        "#.....#.........#.............#",
        "#.....#.........#.....#####...#",
        "#...............#.....#.......#",
        "#........########.....#.......#",
        "#.....................#.......#",
        "#.....#####...........#.......#",
        "#.........#.....#######.......#",
        "#.........#...............G...#",
        "#.............................#",
        "###############################",
    ),
    "clutter_s": (
        "###############################",
        "#.............................#",
        "#...S....#########............#",
        "#........#.......#....####....#",
        "#........#............#.......#",
        "#....#####............#.......#",
        "#................######.......#",
        "#.......######........#.......#",
        "#.......#.............#.......#",
        "#.......#.....#####...........#",
        "#.............#.........G.....#",
        "#.............................#",
        "###############################",
    ),
    "clutter_rooms": (
        "###############################",
        "#.............................#",
        "#...S...#####.......#####.....#",
        "#.......#...#.......#...#.....#",
        "#...........#...........#.....#",
        "#....###....#####.......#.....#",
        "#....#..................#.....#",
        "#....#.....######.......#.....#",
        "#..........#............#.....#",
        "#..........#.....#####........#",
        "#...............#........G....#",
        "#.............................#",
        "###############################",
    ),
    "clutter_zigzag": (
        "###############################",
        "#.............................#",
        "#...S.....#######.............#",
        "#.........#.....#....#####....#",
        "#...............#....#........#",
        "#....######.....#....#........#",
        "#.........#..........####.....#",
        "#.........#.....####..........#",
        "#....####.......#.............#",
        "#.......#.......#.....#####...#",
        "#.......#...............G.....#",
        "#.............................#",
        "###############################",
    ),
}

MAZE_DISPLAY_NAMES = {
    "wide": "Maze 1",
    "wide_s": "Maze 2",
    "multi_room": "Maze 3",
    "rooms": "Maze 4",
    "narrow_s": "Maze 5",
    "clutter_s": "Maze 6",
    "clutter_rooms": "Maze 7",
    "clutter_zigzag": "Maze 8",
}

CONTROLLER_DISPLAY_NAMES = {
    "density_feedback": "Density feedback",
    "density_filter": "Density filter",
    "density_mpc": "Density MPC",
    "nominal": "Nominal",
}


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def triangle_points(center, heading, size):
    center = np.asarray(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    tip = center + size * 1.3 * forward
    left = center - size * 0.9 * forward + size * 0.6 * right
    right_pt = center - size * 0.9 * forward - size * 0.6 * right
    return np.stack([tip, left, right_pt], axis=0)


def parse_ascii_map(name, resolution):
    rows = ASCII_MAPS[name]
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        lengths = ", ".join(str(len(row)) for row in rows)
        raise ValueError(f"map rows must have equal width; got lengths: {lengths}")
    grid = np.array([[char == "#" for char in row] for row in rows], dtype=bool)
    start = goal = None
    for r, row in enumerate(rows):
        for c, char in enumerate(row):
            point = np.array([c * resolution, (len(rows) - 1 - r) * resolution], dtype=float)
            if char == "S":
                start = point
            elif char == "G":
                goal = point
    if start is None or goal is None:
        raise ValueError("map must contain S and G")
    return grid, start, goal


def inflate_grid(grid, inflation_cells):
    inflated = grid.copy()
    rows, cols = grid.shape
    for r, c in np.argwhere(grid):
        r0 = max(0, r - inflation_cells)
        r1 = min(rows, r + inflation_cells + 1)
        c0 = max(0, c - inflation_cells)
        c1 = min(cols, c + inflation_cells + 1)
        inflated[r0:r1, c0:c1] = True
    return inflated


def world_to_cell(point, grid_shape, resolution):
    col = int(round(point[0] / resolution))
    row = grid_shape[0] - 1 - int(round(point[1] / resolution))
    return row, col


def cell_to_world(cell, grid_shape, resolution):
    row, col = cell
    return np.array([col * resolution, (grid_shape[0] - 1 - row) * resolution], dtype=float)


def astar(grid, start, goal, resolution, traversal_cost=None, allow_diagonal=True):
    start_cell = world_to_cell(start, grid.shape, resolution)
    goal_cell = world_to_cell(goal, grid.shape, resolution)
    rows, cols = grid.shape

    def heuristic(cell):
        return np.linalg.norm(np.subtract(cell, goal_cell))

    open_heap = [(heuristic(start_cell), 0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}
    closed = set()
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        moves.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            return [cell_to_world(cell, grid.shape, resolution) for cell in reversed(cells)]
        closed.add(current)

        for dr, dc in moves:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc]:
                continue
            if allow_diagonal and dr and dc and (grid[current[0] + dr, current[1]] or grid[current[0], current[1] + dc]):
                continue
            neighbour = (nr, nc)
            step_cost = np.hypot(dr, dc)
            if traversal_cost is not None:
                step_cost *= 0.5 * (traversal_cost[current] + traversal_cost[neighbour])
            tentative = current_cost + step_cost
            if tentative < g_score.get(neighbour, np.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(neighbour), tentative, neighbour))
    raise RuntimeError("A* could not find a path")


def clearance_traversal_cost(grid, resolution, obstacles, preferred_clearance, weight):
    if weight <= 0.0 or preferred_clearance <= 0.0:
        return None
    cost = np.ones(grid.shape, dtype=float)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col]:
                continue
            point = cell_to_world((row, col), grid.shape, resolution)
            clearance = min(rectangle_clearance(point, obs) for obs in obstacles)
            penalty = max(0.0, (preferred_clearance - clearance) / preferred_clearance)
            cost[row, col] += weight * penalty * penalty
    return cost


def has_line_of_sight(grid, p0, p1, resolution):
    dist = np.linalg.norm(p1 - p0)
    steps = max(2, int(np.ceil(dist / (0.25 * resolution))))
    for alpha in np.linspace(0.0, 1.0, steps):
        row, col = world_to_cell((1.0 - alpha) * p0 + alpha * p1, grid.shape, resolution)
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]) or grid[row, col]:
            return False
    return True


def has_clear_line_of_sight(p0, p1, obstacles, min_clearance, resolution):
    dist = np.linalg.norm(p1 - p0)
    steps = max(2, int(np.ceil(dist / (0.25 * resolution))))
    for alpha in np.linspace(0.0, 1.0, steps):
        point = (1.0 - alpha) * p0 + alpha * p1
        if min(rectangle_clearance(point, obs) for obs in obstacles) < min_clearance:
            return False
    return True


def simplify_path(path, grid, resolution, obstacles=None, min_clearance=0.0):
    if len(path) <= 2:
        return np.asarray(path, dtype=float)
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        next_anchor = anchor + 1
        for candidate in range(len(path) - 1, anchor, -1):
            has_los = has_line_of_sight(grid, path[anchor], path[candidate], resolution)
            has_clearance = (
                obstacles is None
                or has_clear_line_of_sight(
                    path[anchor],
                    path[candidate],
                    obstacles,
                    min_clearance,
                    resolution,
                )
            )
            if has_los and has_clearance:
                next_anchor = candidate
                break
        simplified.append(path[next_anchor])
        anchor = next_anchor
    return np.asarray(simplified, dtype=float)


def resample_path(path, max_spacing):
    path = np.asarray(path, dtype=float)
    if len(path) <= 1 or max_spacing <= 0.0:
        return path

    resampled = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-12:
            continue
        pieces = max(1, int(np.ceil(segment_length / max_spacing)))
        for piece in range(1, pieces + 1):
            resampled.append(start + piece / pieces * segment)
    return np.asarray(resampled, dtype=float)


def path_is_valid(path, grid, resolution, obstacles=None, min_clearance=0.0):
    path = np.asarray(path, dtype=float)
    for start, end in zip(path[:-1], path[1:]):
        if not has_line_of_sight(grid, start, end, resolution):
            return False
        if obstacles is not None and not has_clear_line_of_sight(start, end, obstacles, min_clearance, resolution):
            return False
    return True


def chaikin_smooth_path(path, grid, resolution, obstacles=None, min_clearance=0.0, iterations=2):
    smoothed = np.asarray(path, dtype=float)
    for _ in range(max(0, int(iterations))):
        if len(smoothed) <= 2:
            break
        candidate = [smoothed[0]]
        for start, end in zip(smoothed[:-1], smoothed[1:]):
            candidate.append(0.75 * start + 0.25 * end)
            candidate.append(0.25 * start + 0.75 * end)
        candidate.append(smoothed[-1])
        candidate = np.asarray(candidate, dtype=float)
        if not path_is_valid(candidate, grid, resolution, obstacles=obstacles, min_clearance=min_clearance):
            break
        smoothed = candidate
    return smoothed


def occupied_cell_centers(grid, resolution):
    return np.asarray([cell_to_world(tuple(cell), grid.shape, resolution) for cell in np.argwhere(grid)], dtype=float)


def wall_rectangles_from_grid(grid):
    remaining = grid.copy()
    rectangles = []
    rows, cols = remaining.shape
    for row in range(rows):
        for col in range(cols):
            if not remaining[row, col]:
                continue
            col_end = col
            while col_end < cols and remaining[row, col_end]:
                col_end += 1
            row_end = row + 1
            while row_end < rows and np.all(remaining[row_end, col:col_end]):
                row_end += 1
            remaining[row:row_end, col:col_end] = False
            rectangles.append((row, row_end, col, col_end))
    return rectangles


def rectangles_to_obstacles(rectangles, grid_shape, resolution, robot_radius, transition_width):
    obstacles = []
    rows = grid_shape[0]
    for row0, row1, col0, col1 in rectangles:
        x_min = (col0 - 0.5) * resolution
        x_max = (col1 - 0.5) * resolution
        y_min = (rows - row1 - 0.5) * resolution
        y_max = (rows - row0 - 0.5) * resolution
        center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=float)
        half_extents = np.array([(x_max - x_min) / 2.0, (y_max - y_min) / 2.0], dtype=float)
        obstacles.append(
            RectangleObstacle(
                center=center,
                half_extents=half_extents,
                r1=float(robot_radius),
                r2=float(robot_radius + transition_width),
            )
        )
    return obstacles


def rectangle_clearance(point, obstacle):
    q = np.abs(np.asarray(point, dtype=float) - obstacle.center) - obstacle.half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(float(q[0]), float(q[1])), 0.0)
    return outside + inside


def clearance_bump(clearance, r1, r2):
    if clearance <= r1:
        return 0.0
    if clearance >= r2:
        return 1.0
    m = (clearance - r1) / max(r2 - r1, 1e-12)
    f = np.exp(-1.0 / m)
    f_shift = np.exp(-1.0 / (1.0 - m))
    return float(f / (f + f_shift))


def pose_density(state, goal_state, alpha, obstacles, theta_weight=0.05, min_v=1e-6):
    err = np.array(
        [state[0] - goal_state[0], state[1] - goal_state[1], wrap_angle(state[2] - goal_state[2])],
        dtype=float,
    )
    lyap = max(float(err[0] ** 2 + err[1] ** 2 + theta_weight * err[2] ** 2), min_v)
    phi = 1.0
    for obs in obstacles:
        if isinstance(obs, RectangleObstacle):
            phi *= clearance_bump(rectangle_clearance(state[:2], obs), obs.r1, obs.r2)
        else:
            phi *= p_norm_bump(state[:2], obs.center, obs.r1, obs.r2, p=obs.p)
    return phi / (lyap ** float(alpha))


def position_density(position, goal, alpha, obstacles, min_dist=1e-3):
    position = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    dist = max(float(np.linalg.norm(position - goal)), float(min_dist))
    phi = 1.0
    for obs in obstacles:
        if isinstance(obs, RectangleObstacle):
            phi *= clearance_bump(rectangle_clearance(position, obs), obs.r1, obs.r2)
        else:
            phi *= p_norm_bump(position, obs.center, obs.r1, obs.r2, p=obs.p)
    return phi / (dist ** (2.0 * float(alpha)))


def position_density_grad(position, goal, alpha, obstacles, eps=1e-3):
    grad = np.zeros(2, dtype=float)
    for idx in range(2):
        step = np.zeros(2, dtype=float)
        step[idx] = eps
        grad[idx] = (
            position_density(position + step, goal, alpha, obstacles)
            - position_density(position - step, goal, alpha, obstacles)
        ) / (2.0 * eps)
    return grad


def unicycle_filter_step(state, control, dt):
    return unicycle_step(state, float(control[0]), float(control[1]), dt)


def shift_controls(controls):
    if controls is None:
        return None
    shifted = np.asarray(controls, dtype=float).copy()
    shifted[:-1] = shifted[1:]
    shifted[-1] = shifted[-2] if len(shifted) > 1 else shifted[-1]
    return shifted


def waypoint_control(state, waypoints, target_index, v_max, omega_max, k_heading, lookahead):
    while target_index < len(waypoints) - 1 and np.linalg.norm(waypoints[target_index] - state[:2]) < lookahead:
        target_index += 1
    target = waypoints[target_index]
    desired_heading = np.arctan2(target[1] - state[1], target[0] - state[0])
    heading_error = wrap_angle(desired_heading - state[2])
    turn_gate = max(0.0, np.cos(heading_error))
    distance = np.linalg.norm(target - state[:2])
    v = min(v_max, 1.5 * distance) * turn_gate
    omega = np.clip(k_heading * heading_error, -omega_max, omega_max)
    return np.array([v, omega], dtype=float), target_index


def path_lookahead_goal(position, waypoints, lookahead_distance):
    best_segment = 0
    best_t = 0.0
    best_dist = np.inf
    for idx in range(len(waypoints) - 1):
        a = waypoints[idx]
        b = waypoints[idx + 1]
        segment = b - a
        length_sq = float(segment @ segment)
        if length_sq <= 1e-12:
            continue
        t = float(np.clip(((position - a) @ segment) / length_sq, 0.0, 1.0))
        projection = a + t * segment
        dist = float(np.linalg.norm(position - projection))
        if dist < best_dist:
            best_dist = dist
            best_segment = idx
            best_t = t

    remaining = float(lookahead_distance)
    idx = best_segment
    a = waypoints[idx]
    b = waypoints[idx + 1]
    segment = b - a
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 1e-12:
        return waypoints[min(idx + 1, len(waypoints) - 1)], min(idx + 1, len(waypoints) - 1)
    point = a + best_t * segment
    distance_to_segment_end = (1.0 - best_t) * segment_length
    while remaining > distance_to_segment_end and idx < len(waypoints) - 2:
        remaining -= distance_to_segment_end
        idx += 1
        a = waypoints[idx]
        b = waypoints[idx + 1]
        segment = b - a
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-12:
            distance_to_segment_end = 0.0
            continue
        point = a
        distance_to_segment_end = segment_length
    if segment_length > 1e-12:
        point = point + min(remaining, distance_to_segment_end) / segment_length * segment
    target_index = min(idx + 1, len(waypoints) - 1)
    return point, target_index


def nominal_path_control(state, target, v_max, omega_max, k_heading):
    target_vec = target - state[:2]
    desired_heading = np.arctan2(target_vec[1], target_vec[0])
    heading_error = wrap_angle(desired_heading - state[2])
    turn_gate = max(0.0, np.cos(heading_error))
    distance = np.linalg.norm(target_vec)
    return np.array(
        [
            min(v_max, 1.5 * distance) * turn_gate,
            np.clip(k_heading * heading_error, -omega_max, omega_max),
        ],
        dtype=float,
    )


def simulate_baseline_trajectory(
    start,
    goal,
    waypoints,
    *,
    dt,
    steps,
    v_max,
    omega_max,
    k_heading,
    path_lookahead,
    goal_tolerance,
):
    start_heading = np.arctan2(waypoints[1, 1] - waypoints[0, 1], waypoints[1, 0] - waypoints[0, 0])
    state = np.array([start[0], start[1], start_heading], dtype=float)
    traj = [state.copy()]
    target_indices = [1]
    target_index = 1
    for _ in range(steps):
        target, target_index = path_lookahead_goal(state[:2], waypoints, path_lookahead)
        control = nominal_path_control(state, target, v_max, omega_max, k_heading)
        state = unicycle_step(state, float(control[0]), float(control[1]), dt)
        state[2] = wrap_angle(state[2])
        traj.append(state.copy())
        target_indices.append(target_index)
        if np.linalg.norm(state[:2] - goal) < goal_tolerance:
            break
    return np.asarray(traj, dtype=float), np.asarray(target_indices, dtype=int)


def baseline_lookahead_goal(position, baseline_traj, lookahead_distance):
    points = baseline_traj[:, :2]
    closest = int(np.argmin(np.linalg.norm(points - position, axis=1)))
    travelled = 0.0
    idx = closest
    while idx < len(points) - 1 and travelled < lookahead_distance:
        step = float(np.linalg.norm(points[idx + 1] - points[idx]))
        travelled += step
        idx += 1
    return points[idx], idx


def local_wall_obstacles(state, wall_centers, resolution, sensing_radius, max_obstacles, robot_radius):
    distances = np.linalg.norm(wall_centers - state[:2], axis=1)
    local_indices = np.argsort(distances)[:max_obstacles]
    local_indices = [idx for idx in local_indices if distances[idx] < sensing_radius]
    cell_radius = 0.5 * resolution
    return [
        Obstacle(
            center=wall_centers[idx],
            r1=cell_radius + robot_radius,
            r2=cell_radius + robot_radius + 0.45 * resolution,
            p=4.0,
        )
        for idx in local_indices
    ]


def local_rectangle_obstacles(state, rectangle_obstacles, sensing_radius, max_obstacles):
    scored = []
    for obstacle in rectangle_obstacles:
        clearance = rectangle_clearance(state[:2], obstacle)
        if clearance < sensing_radius:
            scored.append((clearance, obstacle))
    scored.sort(key=lambda item: item[0])
    return [obstacle for _, obstacle in scored[:max_obstacles]]


def min_wall_clearance(point, wall_centers, resolution, robot_radius):
    cell_radius = 0.5 * resolution
    return float(np.min(np.linalg.norm(wall_centers - point, axis=1)) - cell_radius - robot_radius)


def min_rectangle_clearance(point, rectangle_obstacles, robot_radius):
    return float(min(rectangle_clearance(point, obs) for obs in rectangle_obstacles) - robot_radius)


def make_result(
    *,
    status,
    steps,
    dist_to_goal,
    dist_to_target,
    min_clearance,
    waypoints,
    solver_failures,
    max_slack,
    grid,
    raw_path,
    simplified_path,
    baseline_traj,
    traj,
    target_indices,
    target_points,
    resolution,
    robot_radius,
    rectangles,
    controller,
    map_name,
):
    return {
        "status": status,
        "steps": steps,
        "dist_to_goal": dist_to_goal,
        "dist_to_target": dist_to_target,
        "min_clearance": min_clearance,
        "waypoints": len(waypoints),
        "solver_failures": solver_failures,
        "max_slack": max_slack,
        "grid": grid,
        "raw_path": np.asarray(raw_path, dtype=float),
        "simplified_path": np.asarray(simplified_path, dtype=float),
        "baseline_traj": np.asarray(baseline_traj, dtype=float),
        "traj": np.asarray(traj, dtype=float),
        "target_indices": np.asarray(target_indices, dtype=int),
        "target_points": np.asarray(target_points, dtype=float),
        "resolution": resolution,
        "robot_radius": robot_radius,
        "rectangles": rectangles,
        "controller": controller,
        "map_name": map_name,
    }


def run(args):
    grid, start, goal = parse_ascii_map(args.map, args.resolution)
    planning_grid = inflate_grid(grid, int(np.floor(args.robot_radius / args.resolution)))
    wall_centers = occupied_cell_centers(grid, args.resolution)
    rectangles = wall_rectangles_from_grid(grid)
    rectangle_obstacles = rectangles_to_obstacles(
        rectangles,
        grid.shape,
        args.resolution,
        args.robot_radius,
        args.transition_width,
    )
    traversal_cost = clearance_traversal_cost(
        planning_grid,
        args.resolution,
        rectangle_obstacles,
        args.preferred_clearance,
        args.clearance_cost,
    )
    raw_path = astar(
        planning_grid,
        start,
        goal,
        args.resolution,
        traversal_cost=traversal_cost,
        allow_diagonal=args.allow_diagonal_planning,
    )
    if args.smooth_path:
        waypoints = chaikin_smooth_path(
            raw_path,
            planning_grid,
            args.resolution,
            obstacles=rectangle_obstacles,
            min_clearance=args.smooth_clearance,
            iterations=args.smooth_iterations,
        )
    elif args.simplify_path:
        waypoints = simplify_path(
            raw_path,
            planning_grid,
            args.resolution,
            obstacles=rectangle_obstacles,
            min_clearance=args.waypoint_clearance,
        )
    else:
        waypoints = np.asarray(raw_path, dtype=float)
    waypoints = resample_path(waypoints, args.waypoint_spacing)
    final_target = waypoints[-1]
    baseline_traj, _ = simulate_baseline_trajectory(
        start,
        goal,
        waypoints,
        dt=args.dt,
        steps=args.baseline_steps,
        v_max=args.baseline_v_max,
        omega_max=args.baseline_omega_max,
        k_heading=args.baseline_k_heading,
        path_lookahead=args.baseline_lookahead,
        goal_tolerance=args.goal_tolerance,
    )

    start_heading = np.arctan2(waypoints[1, 1] - waypoints[0, 1], waypoints[1, 0] - waypoints[0, 0])
    state = np.array([start[0], start[1], start_heading], dtype=float)
    target_index = 1
    min_clearance = min_rectangle_clearance(state[:2], rectangle_obstacles, args.robot_radius)
    solver_failures = 0
    max_slack = 0.0
    traj = [state.copy()]
    target_indices = [target_index]
    target_points = [waypoints[target_index].copy()]
    previous_control = np.zeros(2, dtype=float)
    previous_controls = None

    for step in range(args.steps):
        if args.reference_mode == "baseline":
            path_target, target_index = baseline_lookahead_goal(
                state[:2],
                baseline_traj,
                args.baseline_tracking_lookahead,
            )
        elif args.waypoint_mode == "delayed":
            while (
                target_index < len(waypoints) - 1
                and np.linalg.norm(waypoints[target_index] - state[:2]) < args.waypoint_switch_radius
            ):
                target_index += 1
            path_target = waypoints[target_index]
        else:
            path_target, target_index = path_lookahead_goal(
                state[:2],
                waypoints,
                args.path_lookahead,
            )
        target_vec = path_target - state[:2]
        desired_heading = np.arctan2(target_vec[1], target_vec[0])
        heading_error = wrap_angle(desired_heading - state[2])
        turn_gate = max(0.0, np.cos(heading_error))
        distance = np.linalg.norm(target_vec)
        u_nom = np.array(
            [
                min(args.v_max, 1.5 * distance) * turn_gate,
                np.clip(args.k_heading * heading_error, -args.omega_max, args.omega_max),
            ],
            dtype=float,
        )
        control = u_nom
        if args.controller in {"density_feedback", "density_filter", "density_mpc"}:
            if args.obstacle_model == "rectangles":
                obstacles = local_rectangle_obstacles(
                    state,
                    rectangle_obstacles,
                    args.sensing_radius,
                    args.max_obstacles,
                )
            else:
                obstacles = local_wall_obstacles(
                    state,
                    wall_centers,
                    args.resolution,
                    args.sensing_radius,
                    args.max_obstacles,
                    args.robot_radius,
            )
            if obstacles:
                local_target = goal if args.density_goal_mode == "final" else path_target
                local_heading = np.arctan2(local_target[1] - state[1], local_target[0] - state[0])
                local_goal_state = np.array([local_target[0], local_target[1], local_heading], dtype=float)
                if args.controller == "density_feedback":
                    if distance > 1e-10:
                        planar_nom = min(args.v_max, 1.5 * distance) * target_vec / distance
                    else:
                        planar_nom = np.zeros(2, dtype=float)
                    planar_ref = planar_nom + args.density_gain * position_density_grad(
                        state[:2],
                        local_target,
                        args.alpha,
                        obstacles,
                    )
                    speed = float(np.linalg.norm(planar_ref))
                    if speed > args.v_max:
                        planar_ref = planar_ref / speed * args.v_max
                        speed = args.v_max
                    if speed < 1e-10:
                        result = None
                        control = np.zeros(2, dtype=float)
                    else:
                        desired_heading = np.arctan2(planar_ref[1], planar_ref[0])
                        heading_error = wrap_angle(desired_heading - state[2])
                        control = np.array(
                            [
                                min(args.v_max, speed) * max(0.0, np.cos(heading_error)),
                                np.clip(args.k_heading * heading_error, -args.omega_max, args.omega_max),
                            ],
                            dtype=float,
                        )
                        result = None
                elif args.controller == "density_filter":
                    result = solve_discrete_density_filter(
                        state,
                        local_goal_state,
                        args.alpha,
                        obstacles,
                        u_nom=u_nom,
                        dt=args.density_dt,
                        next_state_fn=unicycle_filter_step,
                        u_min=np.array([0.0, -args.omega_max]),
                        u_max=np.array([args.v_max, args.omega_max]),
                        divergence=0.0,
                        slack_weight=args.slack_weight,
                        density_fn=pose_density,
                        return_info=True,
                    )
                else:
                    initial_controls = shift_controls(previous_controls)
                    if initial_controls is None:
                        initial_controls = np.repeat(previous_control[None, :], args.horizon, axis=0)
                    result = solve_density_mpc(
                        state,
                        local_goal_state,
                        args.alpha,
                        obstacles,
                        solver="scipy_slsqp",
                        u_nom=np.zeros((args.horizon, 2), dtype=float),
                        horizon=args.horizon,
                        dt=args.density_dt,
                        next_state_fn=unicycle_filter_step,
                        u_min=np.array([0.0, -args.omega_max]),
                        u_max=np.array([args.v_max, args.omega_max]),
                        divergence=0.0,
                        slack_weight=0.0,
                        slack_l1_weight=1.0,
                        control_weight=np.diag([0.01, 0.01]),
                        control_rate_weight=1.0,
                        previous_control=previous_control,
                        state_weight=np.diag([30.0, 30.0, 10.0]),
                        terminal_weight=1000.0,
                        density_fn=pose_density,
                        initial_controls=initial_controls,
                        return_info=True,
                    )
                    previous_controls = result.controls
                if result is not None:
                    control = result.u
                    solver_failures += int(not result.success)
                    if result.slack.size:
                        max_slack = max(max_slack, float(np.max(result.slack)))
                previous_control = control.copy()
            else:
                previous_control = control.copy()
                previous_controls = None
        else:
            previous_control = control.copy()
            previous_controls = None

        state = unicycle_step(state, float(control[0]), float(control[1]), args.dt)
        state[2] = wrap_angle(state[2])
        traj.append(state.copy())
        target_indices.append(target_index)
        target_points.append(path_target.copy())
        clearance = min_rectangle_clearance(state[:2], rectangle_obstacles, args.robot_radius)
        min_clearance = min(min_clearance, clearance)
        dist_to_goal = float(np.linalg.norm(state[:2] - goal))
        dist_to_target = float(np.linalg.norm(state[:2] - final_target))

        if args.verbose and step % args.print_interval == 0:
            print(
                f"iter={step} dist_to_target={dist_to_target:.3f} dist_to_goal={dist_to_goal:.3f} "
                f"target={target_index}/{len(waypoints)-1} clearance={clearance:.3f}"
            )
        if clearance < 0.0:
            return make_result(
                status="collision",
                steps=step + 1,
                dist_to_goal=dist_to_goal,
                dist_to_target=dist_to_target,
                min_clearance=min_clearance,
                waypoints=waypoints,
                solver_failures=solver_failures,
                max_slack=max_slack,
                grid=grid,
                raw_path=raw_path,
                simplified_path=waypoints,
                baseline_traj=baseline_traj,
                traj=traj,
                target_indices=target_indices,
                target_points=target_points,
                resolution=args.resolution,
                robot_radius=args.robot_radius,
                rectangles=rectangles,
                controller=args.controller,
                map_name=args.map,
            )
        if dist_to_target < args.goal_tolerance:
            return make_result(
                status="success",
                steps=step + 1,
                dist_to_goal=dist_to_goal,
                dist_to_target=dist_to_target,
                min_clearance=min_clearance,
                waypoints=waypoints,
                solver_failures=solver_failures,
                max_slack=max_slack,
                grid=grid,
                raw_path=raw_path,
                simplified_path=waypoints,
                baseline_traj=baseline_traj,
                traj=traj,
                target_indices=target_indices,
                target_points=target_points,
                resolution=args.resolution,
                robot_radius=args.robot_radius,
                rectangles=rectangles,
                controller=args.controller,
                map_name=args.map,
            )

    return make_result(
        status="timeout",
        steps=args.steps,
        dist_to_goal=float(np.linalg.norm(state[:2] - goal)),
        dist_to_target=float(np.linalg.norm(state[:2] - final_target)),
        min_clearance=min_clearance,
        waypoints=waypoints,
        solver_failures=solver_failures,
        max_slack=max_slack,
        grid=grid,
        raw_path=raw_path,
        simplified_path=waypoints,
        baseline_traj=baseline_traj,
        traj=traj,
        target_indices=target_indices,
        target_points=target_points,
        resolution=args.resolution,
        robot_radius=args.robot_radius,
        rectangles=rectangles,
        controller=args.controller,
        map_name=args.map,
    )


def plot_result(result, *, show=False, save_gif=False, save_mp4=False, stride=2, fps=20):
    import matplotlib.pyplot as plt
    from matplotlib import animation, patches

    grid = result["grid"]
    resolution = result["resolution"]
    traj = result["traj"]
    raw_path = result["raw_path"]
    simplified_path = result["simplified_path"]
    target_points = result["target_points"]
    baseline_traj = result["baseline_traj"]
    robot_radius = result["robot_radius"]
    rectangles = result["rectangles"]

    height, width = grid.shape
    xlim = (-0.5 * resolution, (width - 0.5) * resolution)
    ylim = (-0.5 * resolution, (height - 0.5) * resolution)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    maze_name = MAZE_DISPLAY_NAMES.get(result["map_name"], result["map_name"])
    controller_name = CONTROLLER_DISPLAY_NAMES.get(result["controller"], result["controller"])
    ax.set_title(f"{maze_name} - {controller_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.4)

    for row, col in np.argwhere(grid):
        center = cell_to_world((row, col), grid.shape, resolution)
        rect = patches.Rectangle(
            center - 0.5 * resolution,
            resolution,
            resolution,
            facecolor="0.3",
            edgecolor="0.3",
            linewidth=0.2,
        )
        ax.add_patch(rect)

    for row0, row1, col0, col1 in rectangles:
        x_min = (col0 - 0.5) * resolution
        x_max = (col1 - 0.5) * resolution
        y_min = (grid.shape[0] - row1 - 0.5) * resolution
        y_max = (grid.shape[0] - row0 - 0.5) * resolution
        ax.add_patch(
            patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="none",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.35,
            )
        )

    plot_start(ax, traj[0, :2])
    plot_goal(ax, simplified_path[-1])
    ax.plot(raw_path[:, 0], raw_path[:, 1], color="0.65", linewidth=1.0, linestyle=":", label="A*")
    ax.plot(
        simplified_path[:, 0],
        simplified_path[:, 1],
        color="tab:orange",
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=3,
        label="waypoints",
    )
    if len(baseline_traj) > 1:
        ax.plot(
            baseline_traj[:, 0],
            baseline_traj[:, 1],
            color="tab:green",
            linewidth=1.4,
            alpha=0.8,
            label="baseline",
        )
    executed_line, = ax.plot([], [], color="tab:blue", linewidth=2.0, label="executed")
    target_marker, = ax.plot([], [], marker="x", color="tab:orange", markersize=8, linestyle="None", label="target")
    robot_patch = patches.Polygon(
        triangle_points(traj[0, :2], traj[0, 2], max(robot_radius, 0.06)),
        closed=True,
        facecolor="tab:blue",
        edgecolor="k",
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(robot_patch)
    ax.legend(loc="upper right")

    frame_indices = list(range(0, len(traj), max(1, stride)))
    if frame_indices[-1] != len(traj) - 1:
        frame_indices.append(len(traj) - 1)

    def update(frame_index):
        state = traj[frame_index]
        executed_line.set_data(traj[: frame_index + 1, 0], traj[: frame_index + 1, 1])
        robot_patch.set_xy(triangle_points(state[:2], state[2], max(robot_radius, 0.06)))
        target = target_points[min(frame_index, len(target_points) - 1)]
        target_marker.set_data([target[0]], [target[1]])
        return executed_line, robot_patch, target_marker

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=True)
    output_dir = Path(__file__).resolve().parent / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"maze_{result['map_name']}_{result['controller']}_{result['status']}"
    if save_gif:
        path = output_dir / f"{stem}.gif"
        ani.save(path, writer=animation.PillowWriter(fps=fps))
        print(f"saved animation to {path}")
    if save_mp4:
        path = output_dir / f"{stem}.mp4"
        try:
            ani.save(path, writer=animation.FFMpegWriter(fps=fps))
            print(f"saved animation to {path}")
        except Exception as exc:
            fallback = path.with_suffix(".gif")
            ani.save(fallback, writer=animation.PillowWriter(fps=fps))
            print(f"mp4 save failed ({exc}); saved animation to {fallback}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(
    default_controller="density_feedback",
    expose_controller=True,
    default_waypoint_mode="continuous",
    default_v_max=2.0,
    default_omega_max=3.0,
    default_smooth_path=True,
    default_waypoint_spacing=0.12,
    default_waypoint_clearance=0.0,
    default_density_goal_mode="final",
    default_reference_mode="path",
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", choices=tuple(ASCII_MAPS), default="wide")
    if expose_controller:
        parser.add_argument(
            "--controller",
            choices=("nominal", "density_feedback", "density_filter", "density_mpc"),
            default=default_controller,
        )
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--density-dt", type=float, default=0.1)
    parser.add_argument("--resolution", type=float, default=0.15)
    parser.add_argument(
        "--allow-diagonal-planning",
        action="store_true",
        help="Allow diagonal A* moves. Disabled by default to keep maze waypoints corridor-aligned.",
    )
    parser.add_argument("--robot-radius", type=float, default=0.035)
    parser.add_argument("--v-max", type=float, default=default_v_max)
    parser.add_argument("--omega-max", type=float, default=default_omega_max)
    parser.add_argument("--k-heading", type=float, default=4.0)
    parser.add_argument("--lookahead", type=float, default=0.20)
    parser.add_argument("--path-lookahead", type=float, default=0.45)
    parser.add_argument(
        "--reference-mode",
        choices=("path", "baseline"),
        default=default_reference_mode,
        help="Track direct path targets or a nominal baseline trajectory.",
    )
    parser.add_argument(
        "--waypoint-mode",
        choices=("continuous", "delayed"),
        default=default_waypoint_mode,
        help="Use a moving path-lookahead target or hold each waypoint until reached.",
    )
    parser.add_argument("--waypoint-switch-radius", type=float, default=0.06)
    parser.add_argument(
        "--waypoint-clearance",
        type=float,
        default=default_waypoint_clearance,
        help="Minimum clearance required for path shortcut waypoints.",
    )
    parser.add_argument(
        "--waypoint-spacing",
        type=float,
        default=default_waypoint_spacing,
        help="Maximum spacing between consecutive waypoints after path simplification.",
    )
    parser.add_argument(
        "--smooth-path",
        dest="smooth_path",
        action="store_true",
        default=default_smooth_path,
        help="Use a smoothed A* reference path for lookahead tracking.",
    )
    parser.add_argument(
        "--no-smooth-path",
        dest="smooth_path",
        action="store_false",
        help="Disable path smoothing and use shortcut/raw waypoints.",
    )
    parser.add_argument("--smooth-iterations", type=int, default=2, help="Number of Chaikin path smoothing passes.")
    parser.add_argument(
        "--smooth-clearance",
        type=float,
        default=0.05,
        help="Minimum wall clearance required for smoothed path segments.",
    )
    parser.add_argument(
        "--preferred-clearance",
        type=float,
        default=0.18,
        help="Planner clearance that receives no extra A* traversal penalty.",
    )
    parser.add_argument(
        "--clearance-cost",
        type=float,
        default=0.0,
        help="A* penalty weight for cells below the preferred clearance.",
    )
    parser.add_argument(
        "--simplify-path",
        dest="simplify_path",
        action="store_true",
        default=True,
        help="Use line-of-sight path simplification.",
    )
    parser.add_argument(
        "--no-simplify-path",
        dest="simplify_path",
        action="store_false",
        help="Use the raw A* grid path.",
    )
    parser.add_argument("--goal-tolerance", type=float, default=0.12)
    parser.add_argument("--baseline-steps", type=int, default=900)
    parser.add_argument("--baseline-v-max", type=float, default=0.35)
    parser.add_argument("--baseline-omega-max", type=float, default=2.8)
    parser.add_argument("--baseline-k-heading", type=float, default=5.0)
    parser.add_argument("--baseline-lookahead", type=float, default=0.28)
    parser.add_argument("--baseline-tracking-lookahead", type=float, default=0.18)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--density-gain", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=7, help="Density MPC prediction horizon.")
    parser.add_argument(
        "--obstacle-model",
        choices=("rectangles", "cells"),
        default="rectangles",
        help="Obstacle representation used by density controllers.",
    )
    parser.add_argument(
        "--density-goal-mode",
        choices=("final", "path"),
        default=default_density_goal_mode,
        help="Use the final goal or the current path target in the density objective.",
    )
    parser.add_argument("--transition-width", type=float, default=0.025, help="Rectangle density transition width.")
    parser.add_argument("--sensing-radius", type=float, default=0.38)
    parser.add_argument("--max-obstacles", type=int, default=6)
    parser.add_argument("--slack-weight", type=float, default=1e4)
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument("--plot", action="store_true", help="Show the maze trajectory plot/animation.")
    parser.add_argument("--no-plot", action="store_true", help="Run headless without showing the animation.")
    parser.add_argument("--save-gif", action="store_true", help="Save an animated GIF.")
    parser.add_argument("--save-mp4", action="store_true", help="Save an animated MP4 when ffmpeg is available.")
    parser.add_argument("--animation-stride", type=int, default=2, help="Use every Nth state in the animation.")
    parser.add_argument("--animation-fps", type=int, default=20, help="Animation frames per second.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not expose_controller:
        args.controller = default_controller

    summary = run(args)
    print(
        "status={status} steps={steps} dist_to_target={dist_to_target:.3f} dist_to_goal={dist_to_goal:.3f} "
        "min_clearance={min_clearance:.3f} waypoints={waypoints} "
        "solver_failures={solver_failures} max_slack={max_slack:.2e}".format(**summary)
    )
    show_animation = (not args.no_plot) or args.plot
    if show_animation or args.save_gif or args.save_mp4:
        plot_result(
            summary,
            show=show_animation,
            save_gif=args.save_gif,
            save_mp4=args.save_mp4,
            stride=args.animation_stride,
            fps=args.animation_fps,
        )


if __name__ == "__main__":
    main()
