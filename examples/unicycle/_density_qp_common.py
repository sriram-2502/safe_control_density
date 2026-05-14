from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from density_utils.controllers import density_feedback_control, solve_density_qp
from density_utils.density import Obstacle, finite_difference_grad, p_norm_bump
from density_utils.dynamics import unicycle_step
from density_utils.utils import plot_goal, plot_obstacle, plot_start
from density_utils.utils.timing import TimedBlock


def angle_wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def p_norm_distance(x, obs):
    dx = x - obs.center
    if obs.angle:
        c = np.cos(-obs.angle)
        s = np.sin(-obs.angle)
        dx = np.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]])
    if obs.scale is not None:
        dx = dx / obs.scale
    return np.sum(np.abs(dx) ** obs.p) ** (1.0 / obs.p)


def p_norm_distance_grad(x, obs, eps=1e-4):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in range(x.size):
        x_f = x.copy()
        x_b = x.copy()
        x_f[idx] += eps
        x_b[idx] -= eps
        grad[idx] = (p_norm_distance(x_f, obs) - p_norm_distance(x_b, obs)) / (2.0 * eps)
    norm = np.linalg.norm(grad)
    if norm < 1e-12:
        return None
    return grad / norm


def nearest_obstacles(pos, obstacles, max_count=4):
    if len(obstacles) <= max_count:
        return list(obstacles)
    ranked = sorted(obstacles, key=lambda obs: p_norm_distance(pos, obs) - obs.r1)
    return ranked[:max_count]


def triangle_points(center, heading, size):
    c = np.array(center, dtype=float)
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)])
    tip = c + size * 1.3 * forward
    left = c - size * 0.9 * forward + size * 0.6 * right
    right_pt = c - size * 0.9 * forward - size * 0.6 * right
    return np.stack([tip, left, right_pt], axis=0)


def calculate_fov_points(position, heading, fov_angle, cam_range):
    half_fov = fov_angle / 2.0
    left_angle = heading - half_fov
    right_angle = heading + half_fov
    left_point = (
        position[0] + cam_range * np.cos(left_angle),
        position[1] + cam_range * np.sin(left_angle),
    )
    right_point = (
        position[0] + cam_range * np.cos(right_angle),
        position[1] + cam_range * np.sin(right_angle),
    )
    return left_point, right_point


def detect_sensed_obstacles(pos, heading, obstacles, cam_range, fov_angle):
    pos = np.asarray(pos, dtype=float)
    sensed = []
    for obs in obstacles:
        rel = obs.center - pos
        dist = np.linalg.norm(rel)
        if dist > cam_range:
            continue
        if fov_angle < 2.0 * np.pi:
            angle_to_obs = np.arctan2(rel[1], rel[0])
            if abs(angle_wrap(angle_to_obs - heading)) > fov_angle / 2.0:
                continue
        sensed.append((dist, obs))
    sensed.sort(key=lambda item: item[0])
    return [obs for _, obs in sensed]


def sample_obstacle_boundary(obs, num=120):
    theta = np.linspace(0.0, 2.0 * np.pi, num=num, endpoint=True)
    c = np.cos(theta)
    s = np.sin(theta)
    p = float(obs.p)
    x = np.sign(c) * (np.abs(c) ** (2.0 / p))
    y = np.sign(s) * (np.abs(s) ** (2.0 / p))
    pts = np.stack([x, y], axis=1) * obs.r1
    if obs.scale is not None:
        pts = pts * np.asarray(obs.scale, dtype=float)[None, :]
    if obs.angle:
        ca = np.cos(obs.angle)
        sa = np.sin(obs.angle)
        rot = np.array([[ca, -sa], [sa, ca]])
        pts = pts @ rot.T
    return pts + obs.center[None, :]


def make_multi_obstacles(agent_radius):
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
    return obstacles, inflate_obstacles(obstacles, agent_radius)


def inflate_obstacles(obstacles, agent_radius, r2_margin=0.0):
    return [
        Obstacle(
            center=obs.center,
            r1=obs.r1 + agent_radius,
            r2=obs.r2 + agent_radius + r2_margin,
            p=obs.p,
            scale=obs.scale,
            angle=obs.angle,
        )
        for obs in obstacles
    ]


def unicycle_pose_error(state, goal_state):
    state = np.asarray(state, dtype=float)
    goal_state = np.asarray(goal_state, dtype=float)
    return np.array(
        [
            state[0] - goal_state[0],
            state[1] - goal_state[1],
            angle_wrap(state[2] - goal_state[2]),
        ],
        dtype=float,
    )


def unicycle_pose_density_value(
    state,
    goal_state,
    alpha,
    obstacles,
    *,
    theta_weight=0.25,
    min_v=1e-6,
):
    state = np.asarray(state, dtype=float)
    goal_state = np.asarray(goal_state, dtype=float)
    err = unicycle_pose_error(state, goal_state)
    v_lyap = max(
        float(err[0] ** 2 + err[1] ** 2 + theta_weight * err[2] ** 2),
        float(min_v),
    )

    phi = 1.0
    for obs in obstacles:
        phi *= p_norm_bump(
            state[:2],
            obs.center,
            obs.r1,
            obs.r2,
            p=obs.p,
            scale=obs.scale,
            angle=obs.angle,
        )
    return phi / (v_lyap ** float(alpha))


def unicycle_pose_density_grad(state, goal_state, alpha, obstacles, theta_weight=0.25):
    return finite_difference_grad(
        lambda state_eval: unicycle_pose_density_value(
            state_eval,
            goal_state,
            alpha,
            obstacles,
            theta_weight=theta_weight,
        ),
        state,
        eps=1e-3,
    )


def unicycle_dynamics(state):
    theta = float(state[2])
    return np.zeros(3, dtype=float), np.array(
        [
            [np.cos(theta), 0.0],
            [np.sin(theta), 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def pure_pursuit_nominal(state, goal, v_max, omega_max, k_heading, slow_radius):
    return pure_pursuit_to_target_nominal(
        state,
        goal,
        goal,
        v_max,
        omega_max,
        k_heading,
        slow_radius,
    )


def pure_pursuit_to_target_nominal(
    state,
    target,
    goal,
    v_max,
    omega_max,
    k_heading,
    slow_radius,
    min_turn_gate=0.0,
):
    direction = target - state[:2]
    dist = np.linalg.norm(direction)
    if dist < 1e-8:
        return np.zeros(2, dtype=float)

    desired_heading = float(np.arctan2(direction[1], direction[0]))
    heading_error = angle_wrap(desired_heading - state[2])
    goal_dist = np.linalg.norm(goal - state[:2])
    speed_scale = min(1.0, goal_dist / max(slow_radius, 1e-6))
    turn_gate = max(float(min_turn_gate), np.cos(heading_error))
    v_nom = v_max * speed_scale * turn_gate
    omega_nom = float(np.clip(k_heading * heading_error, -omega_max, omega_max))
    return np.array([v_nom, omega_nom], dtype=float)


def _segment_projection(pos, goal, point):
    path = goal - pos
    length_sq = float(path @ path)
    if length_sq < 1e-12:
        return pos.copy(), 0.0
    t = float(np.clip(((point - pos) @ path) / length_sq, 0.0, 1.0))
    return pos + t * path, t


def _blocking_obstacle(pos, goal, obstacles, block_margin):
    best = None
    best_score = np.inf
    for obs in obstacles:
        closest, t = _segment_projection(pos, goal, obs.center)
        current_clearance = p_norm_distance(pos, obs) - obs.r2
        proximity_margin = min(0.08, block_margin)
        if current_clearance <= proximity_margin and t < 0.98:
            score = current_clearance - block_margin
            if score < best_score:
                best = (obs, t)
                best_score = score
            continue
        if t <= 0.02 or t >= 0.98:
            continue
        clearance = p_norm_distance(closest, obs) - obs.r2
        if clearance > block_margin:
            continue
        dist_to_obs = max(p_norm_distance(pos, obs) - obs.r2, 0.0)
        score = clearance + 0.1 * dist_to_obs
        if score < best_score:
            best = (obs, t)
            best_score = score
    return best


def obstacle_aware_nominal(
    state,
    goal,
    obstacles,
    v_max,
    omega_max,
    k_heading,
    slow_radius,
    *,
    block_margin=0.2,
    waypoint_margin=0.25,
    side_preference=1.0,
):
    pos = state[:2]
    path = goal - pos
    path_norm = np.linalg.norm(path)
    if path_norm < 1e-8:
        return np.zeros(2, dtype=float)

    blocker = _blocking_obstacle(pos, goal, obstacles, block_margin)
    if blocker is None:
        return pure_pursuit_nominal(
            state,
            goal,
            v_max,
            omega_max,
            k_heading,
            slow_radius,
        )

    obs, _ = blocker
    path_dir = path / path_norm
    side_axis = np.array([-path_dir[1], path_dir[0]], dtype=float)
    rel_center = obs.center - pos
    center_side = float(path_dir[0] * rel_center[1] - path_dir[1] * rel_center[0])
    if abs(center_side) < 0.15:
        side = float(np.sign(side_preference) or 1.0)
    else:
        side = -float(np.sign(center_side))

    clearance_radius = obs.r2 + waypoint_margin
    target = obs.center + side * clearance_radius * side_axis
    target = target + 0.35 * clearance_radius * path_dir
    if float((target - pos) @ path_dir) < 0.2:
        target = pos + 0.5 * path_dir + side * clearance_radius * side_axis

    return pure_pursuit_to_target_nominal(
        state,
        target,
        goal,
        v_max,
        omega_max,
        k_heading,
        slow_radius,
        min_turn_gate=0.15,
    )


def feedback_reference_nominal(
    state,
    goal,
    alpha,
    obstacles,
    v_max,
    omega_max,
    k_heading,
    *,
    ctrl_multiplier,
    rad_from_goal,
    dt,
    saturation,
):
    planar_ref = density_feedback_control(
        state[:2],
        goal,
        alpha,
        obstacles,
        ctrl_multiplier=ctrl_multiplier,
        rad_from_goal=rad_from_goal,
        q_lqr=4.0,
        r_lqr=1.0,
        dt=dt,
        saturation=saturation,
    )
    speed = float(np.linalg.norm(planar_ref))
    if speed < 1e-8:
        return np.zeros(2, dtype=float)
    desired_heading = float(np.arctan2(planar_ref[1], planar_ref[0]))
    heading_error = angle_wrap(desired_heading - state[2])
    turn_gate = max(0.1, np.cos(heading_error))
    v_nom = min(speed, float(v_max)) * turn_gate
    omega_nom = float(np.clip(k_heading * heading_error, -omega_max, omega_max))
    return np.array([v_nom, omega_nom], dtype=float)


def unicycle_preview_step(state, control, dt):
    state = np.asarray(state, dtype=float)
    v = float(control[0])
    omega = float(control[1])
    theta = float(state[2])
    if abs(omega) < 1e-8:
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
    else:
        theta_next = theta + omega * dt
        dx = v / omega * (np.sin(theta_next) - np.sin(theta))
        dy = v / omega * (np.cos(theta) - np.cos(theta_next))
    return np.array(
        [
            state[0] + dx,
            state[1] + dy,
            angle_wrap(theta + omega * dt),
        ],
        dtype=float,
    )


def clearance_guard_control(
    control,
    state,
    goal,
    obstacles,
    v_max,
    omega_max,
    clearance_rate=6.0,
    turn_gain=4.0,
):
    v, omega = np.asarray(control, dtype=float)
    v_lower = 0.0
    v_upper = float(v_max)
    heading = np.array([np.cos(state[2]), np.sin(state[2])], dtype=float)
    closest_obs = None
    closest_h = np.inf

    for obs in obstacles:
        h = p_norm_distance(state[:2], obs) - obs.r1
        if h < closest_h:
            closest_h = h
            closest_obs = obs
        grad = p_norm_distance_grad(state[:2], obs)
        if grad is None:
            continue
        a = float(grad @ heading)
        b = -float(clearance_rate) * h
        if abs(a) < 1e-10:
            continue
        limit = b / a
        if a > 0.0:
            v_lower = max(v_lower, limit)
        else:
            v_upper = min(v_upper, limit)

    if v_lower > v_upper:
        v_safe = 0.0
    else:
        v_safe = float(np.clip(v, v_lower, v_upper))
    omega_safe = float(np.clip(omega, -omega_max, omega_max))

    if v_safe < 0.05 and closest_obs is not None and closest_h < 0.12:
        normal = state[:2] - closest_obs.center
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-8:
            normal = normal / normal_norm
            tangent = np.array([-normal[1], normal[0]], dtype=float)
            goal_dir = goal - state[:2]
            if float(tangent @ goal_dir) < 0.0:
                tangent = -tangent
            desired_heading = float(np.arctan2(tangent[1], tangent[0]))
            heading_error = angle_wrap(desired_heading - state[2])
            tangent_omega = float(np.clip(turn_gain * heading_error, -omega_max, omega_max))
            if abs(tangent_omega) > abs(omega_safe):
                omega_safe = tangent_omega
    return np.array([v_safe, omega_safe], dtype=float)


def format_duration(seconds):
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def scenario_config(name):
    if name == "static_single":
        agent_radius = 0.1
        start = np.array([-2.0, -1.0])
        goal = np.array([2.0, 1.1])
        heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
        start_pose = np.array([start[0], start[1], heading0], dtype=float)
        goal_pose = np.array([goal[0], goal[1], heading0], dtype=float)
        obstacles = [Obstacle(center=np.array([0.0, 0.0]), r1=0.6, r2=1.0, p=2.0)]
        return {
            "dt": 0.01,
            "preview_dt": 0.35,
            "constraint_mode": "continuous",
            "steps": 4000,
            "alpha": 0.4,
            "rad_from_goal": 0.01,
            "obstacle_drop_radius": 0.2,
            "stop_steps": 500,
            "v_max": 2.0,
            "omega_max": 3.0,
            "k_heading": 2.0,
            "slow_radius": 0.8,
            "nominal_mode": "density_reference",
            "nominal_ctrl_multiplier": 6.0,
            "nominal_vector_saturation": 4.0,
            "nominal_block_margin": 0.2,
            "nominal_waypoint_margin": 0.25,
            "avoidance_side": 1.0,
            "max_qp_obstacles": 4,
            "clearance_rate": 6.0,
            "theta_weight": 0.02,
            "theta_stop_tol": np.deg2rad(5.0),
            "animation_stride": 10,
            "animation_fps": 20,
            "animation_path": Path("animations") / "unicycle_static_qp.gif",
            "agent_radius": agent_radius,
            "start": start,
            "goal": goal,
            "start_pose": start_pose,
            "goal_pose": goal_pose,
            "obstacles": obstacles,
            "inflated_obstacles": inflate_obstacles(obstacles, agent_radius),
            "title": "Unicycle - Static Obstacle (Density QP)",
            "local_sensing": False,
        }

    if name in ("static_multi", "local_sensing"):
        agent_radius = 0.1
        obstacles, inflated = make_multi_obstacles(agent_radius)
        start = np.array([-2.1, -2.1])
        goal = np.array([2.0, 2.1])
        heading0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
        start_pose = np.array([start[0], start[1], heading0], dtype=float)
        goal_pose = np.array([goal[0], goal[1], heading0], dtype=float)
        return {
            "dt": 0.001,
            "preview_dt": 0.2,
            "constraint_mode": "continuous",
            "steps": 40000,
            "alpha": 0.4,
            "rad_from_goal": 0.01,
            "obstacle_drop_radius": 0.6,
            "stop_steps": 500,
            "v_max": 2.0,
            "omega_max": 4.0,
            "k_heading": 6.0 if name == "local_sensing" else 4.0,
            "slow_radius": 0.8,
            "nominal_mode": "density_reference",
            "nominal_ctrl_multiplier": 4.0,
            "nominal_vector_saturation": 4.0,
            "nominal_block_margin": 0.35,
            "nominal_waypoint_margin": 0.4,
            "avoidance_side": 1.0,
            "max_qp_obstacles": 4,
            "clearance_rate": 6.0,
            "theta_weight": 0.02,
            "theta_stop_tol": np.deg2rad(5.0),
            "animation_stride": 50,
            "animation_fps": 15,
            "animation_path": Path("animations") / f"unicycle_{name}_qp.gif",
            "agent_radius": agent_radius,
            "start": start,
            "goal": goal,
            "start_pose": start_pose,
            "goal_pose": goal_pose,
            "obstacles": obstacles,
            "inflated_obstacles": inflated,
            "title": (
                "Unicycle - Multiple Obstacles (Density QP)"
                if name == "static_multi"
                else "Unicycle - Local Sensing (Density QP)"
            ),
            "local_sensing": name == "local_sensing",
            "cam_range": 1.0,
            "fov_angle": np.deg2rad(80.0),
            "max_sensed": 5,
            "linger_steps": 1000,
        }

    raise ValueError(f"unknown scenario: {name}")


def run_unicycle_density_qp(name, args):
    cfg = scenario_config(name)
    dt = cfg["dt"]
    preview_dt = cfg["preview_dt"]
    constraint_mode = cfg["constraint_mode"]
    steps = args.steps if args.steps is not None else cfg["steps"]
    alpha = cfg["alpha"]
    rad_from_goal = cfg["rad_from_goal"]
    obstacle_drop_radius = cfg["obstacle_drop_radius"]
    stop_steps = cfg["stop_steps"]
    v_max = cfg["v_max"]
    omega_max = cfg["omega_max"]
    k_heading = cfg["k_heading"]
    slow_radius = cfg["slow_radius"]
    nominal_mode = cfg["nominal_mode"]
    nominal_ctrl_multiplier = cfg["nominal_ctrl_multiplier"]
    nominal_vector_saturation = cfg["nominal_vector_saturation"]
    nominal_block_margin = cfg["nominal_block_margin"]
    nominal_waypoint_margin = cfg["nominal_waypoint_margin"]
    avoidance_side = cfg["avoidance_side"]
    max_qp_obstacles = cfg["max_qp_obstacles"]
    clearance_rate = cfg["clearance_rate"]
    theta_weight = cfg["theta_weight"]
    theta_stop_tol = cfg["theta_stop_tol"]
    cdf_rate = 0.1
    slack_weight = 1e4
    animate = not args.no_plot
    save_animation = args.save_gif

    start = cfg["start"]
    goal = cfg["goal"]
    obstacles = cfg["obstacles"]
    inflated_obstacles = cfg["inflated_obstacles"]
    for pt_name, pt in [("start", start), ("goal", goal)]:
        for obs in inflated_obstacles:
            if p_norm_distance(pt, obs) <= obs.r2:
                raise ValueError(f"{pt_name} is inside an obstacle sensing region")

    state = cfg["start_pose"].copy()
    goal_state = cfg["goal_pose"].copy()
    traj = [state.copy()]
    controls = []
    slacks = []
    sensed_counts = []
    buffered_counts = []
    solver_failures = 0
    min_clearance = min(p_norm_distance(state[:2], obs) - obs.r1 for obs in inflated_obstacles)
    sensed_buffer = {}

    control_time = 0.0
    timer = TimedBlock(enabled=False)
    print_interval = 500
    stop_count = 0
    stop_tol = min(0.005, rad_from_goal)

    for step in range(steps):
        if cfg["local_sensing"]:
            sensed = detect_sensed_obstacles(
                state[:2],
                state[2],
                inflated_obstacles,
                cfg["cam_range"],
                cfg["fov_angle"],
            )[: cfg["max_sensed"]]
            for obs in sensed:
                sensed_buffer[id(obs)] = cfg["linger_steps"]
            for obs_id in list(sensed_buffer.keys()):
                sensed_buffer[obs_id] -= 1
                if sensed_buffer[obs_id] <= 0:
                    sensed_buffer.pop(obs_id)
            active_obstacles = [obs for obs in inflated_obstacles if id(obs) in sensed_buffer]
        else:
            sensed = []
            active_obstacles = inflated_obstacles

        dist = np.linalg.norm(state[:2] - goal)
        theta_error = angle_wrap(goal_state[2] - state[2])
        qp_obstacles = [] if dist < obstacle_drop_radius else active_obstacles
        with timer:
            if dist < rad_from_goal:
                u_nom = np.array(
                    [
                        0.0,
                        float(np.clip(k_heading * theta_error, -omega_max, omega_max)),
                    ],
                    dtype=float,
                )
            else:
                nominal_obstacles = nearest_obstacles(
                    state[:2],
                    qp_obstacles,
                    max_count=max_qp_obstacles,
                )
                if nominal_mode == "density_reference" and nominal_obstacles:
                    u_nom = feedback_reference_nominal(
                        state,
                        goal,
                        alpha,
                        nominal_obstacles,
                        v_max,
                        omega_max,
                        k_heading,
                        ctrl_multiplier=nominal_ctrl_multiplier,
                        rad_from_goal=rad_from_goal,
                        dt=dt,
                        saturation=nominal_vector_saturation,
                    )
                else:
                    u_nom = obstacle_aware_nominal(
                        state,
                        goal,
                        nominal_obstacles,
                        v_max,
                        omega_max,
                        k_heading,
                        slow_radius,
                        block_margin=nominal_block_margin,
                        waypoint_margin=nominal_waypoint_margin,
                        side_preference=avoidance_side,
                    )
            unicycle_qp_obstacles = nearest_obstacles(
                state[:2],
                qp_obstacles,
                max_count=max_qp_obstacles,
            )
            qp = solve_density_qp(
                state,
                goal_state,
                alpha,
                unicycle_qp_obstacles,
                dynamics=unicycle_dynamics,
                u_nom=u_nom,
                saturation=(np.array([0.0, -omega_max]), np.array([v_max, omega_max])),
                cdf_rate=cdf_rate,
                slack_weight=slack_weight,
                density_fn=lambda x_eval, goal_eval, alpha_eval, obstacles_eval: (
                    unicycle_pose_density_value(
                        x_eval,
                        goal_eval,
                        alpha_eval,
                        obstacles_eval,
                        theta_weight=theta_weight,
                    )
                ),
                density_grad_fn=lambda x_eval, goal_eval, alpha_eval, obstacles_eval: (
                    unicycle_pose_density_grad(
                        x_eval,
                        goal_eval,
                        alpha_eval,
                        obstacles_eval,
                        theta_weight=theta_weight,
                    )
                ),
                constraint_mode=constraint_mode,
                next_state_fn=unicycle_preview_step,
                dt=preview_dt,
                return_info=True,
            )
            safe_control = clearance_guard_control(
                qp.u,
                state,
                goal,
                unicycle_qp_obstacles,
                v_max,
                omega_max,
                clearance_rate=clearance_rate,
            )
            v, omega = safe_control

        control_time += timer.last
        controls.append([v, omega])
        slack = 0.0
        if qp.slack.size:
            slack = max(slack, float(np.max(qp.slack)))
        slacks.append(slack)
        sensed_counts.append(len(sensed))
        buffered_counts.append(len(active_obstacles))
        if not qp.success:
            solver_failures += 1

        state = unicycle_step(state, float(v), float(omega), dt)
        state[2] = angle_wrap(state[2])
        traj.append(state.copy())
        clearance = min(p_norm_distance(state[:2], obs) - obs.r1 for obs in inflated_obstacles)
        min_clearance = min(min_clearance, clearance)

        if dist < stop_tol and abs(theta_error) < theta_stop_tol:
            stop_count += 1
            if stop_count >= stop_steps:
                print(f"stopping at iter={step} (stable within stop_tol)")
                break
        else:
            stop_count = 0
        if np.linalg.norm(state[:2] - goal) < rad_from_goal and abs(
            angle_wrap(goal_state[2] - state[2])
        ) < theta_stop_tol:
            print(f"stopping at iter={step} (within rad_from_goal)")
            break

        if step % print_interval == 0:
            extra = ""
            if cfg["local_sensing"]:
                extra = f" sensed={len(sensed)} buffered={len(active_obstacles)}"
            print(
                f"iter={step} dist_to_goal={dist:.3f} clearance={clearance:.3f} "
                f"theta_error={theta_error:.3f} slack={slack:.2e}{extra}"
            )

    if len(controls) < len(traj):
        controls.append(controls[-1] if controls else np.zeros(2, dtype=float))
    if len(slacks) < len(traj):
        slacks.append(slacks[-1] if slacks else 0.0)

    traj = np.array(traj)
    controls = np.array(controls, dtype=float)
    slacks = np.array(slacks, dtype=float)
    steps_taken = len(traj) - 1
    avg_control = control_time / max(steps_taken, 1)
    print(
        "steps="
        f"{steps_taken} "
        f"sim_time={format_duration(control_time)} "
        f"avg_iteration={format_duration(avg_control)} "
        f"min_clearance={min_clearance:.4f} "
        f"max_slack={np.max(slacks):.2e} "
        f"solver_failures={solver_failures}"
    )

    if args.no_plot:
        return

    plot_unicycle_results(
        cfg,
        traj,
        controls,
        slacks,
        sensed_counts,
        buffered_counts,
        save_animation,
        animate,
    )


def plot_unicycle_results(
    cfg,
    traj,
    controls,
    slacks,
    sensed_counts,
    buffered_counts,
    save_animation,
    animate,
):
    dt = cfg["dt"]
    t_state = dt * np.arange(len(traj))
    t_u = dt * np.arange(len(controls))
    fig_ts, axes = plt.subplots(3, 2, figsize=(9, 7))
    axes[0, 0].plot(t_state, traj[:, 0], linewidth=1.8, label="x [m]")
    axes[0, 1].plot(t_state, traj[:, 1], linewidth=1.8, label="y [m]")
    axes[1, 0].plot(t_state, traj[:, 2], linewidth=1.8, label="theta [rad]")
    axes[1, 1].plot(t_u, controls[:, 0], linewidth=1.8, label="v [m/s]")
    axes[2, 0].plot(t_u, controls[:, 1], linewidth=1.8, label="omega [rad/s]")
    axes[2, 1].plot(t_state[: len(slacks)], slacks, linewidth=1.8, label="slack")
    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel("time [s]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_start(ax, cfg["start"])
    plot_goal(ax, cfg["goal"])
    for obs in cfg["obstacles"]:
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
    ax.set_title(cfg["title"])
    ax.grid(True, linestyle="--", alpha=0.4)

    if animate:
        line, = ax.plot([], [], color="tab:blue", linewidth=2)
        agent = patches.Polygon(
            triangle_points(traj[0, :2], traj[0, 2], cfg["agent_radius"]),
            closed=True,
            facecolor="tab:blue",
            edgecolor="k",
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(agent)
        artists = [line, agent]
        fov_poly = None
        sensed_edges = []
        boundary_points = []
        if cfg["local_sensing"]:
            fov_poly = patches.Polygon(
                np.zeros((3, 2)),
                closed=True,
                edgecolor="darkorange",
                facecolor="gold",
                linestyle="--",
                linewidth=2.5,
                alpha=0.25,
                zorder=6,
            )
            ax.add_patch(fov_poly)
            artists.append(fov_poly)
            boundary_points = [sample_obstacle_boundary(obs) for obs in cfg["obstacles"]]
            for _ in cfg["obstacles"]:
                edge_line, = ax.plot([], [], color="tab:orange", linewidth=1.5, zorder=3)
                sensed_edges.append(edge_line)
            artists.extend(sensed_edges)

        def update(i):
            line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            agent.set_xy(triangle_points(traj[i, :2], traj[i, 2], cfg["agent_radius"]))
            if cfg["local_sensing"]:
                fov_left, fov_right = calculate_fov_points(
                    traj[i, :2], traj[i, 2], cfg["fov_angle"], cfg["cam_range"]
                )
                fov_poly.set_xy(np.array([[traj[i, 0], traj[i, 1]], fov_left, fov_right]))
                sensed = detect_sensed_obstacles(
                    traj[i, :2],
                    traj[i, 2],
                    cfg["inflated_obstacles"],
                    cfg["cam_range"],
                    cfg["fov_angle"],
                )[: cfg["max_sensed"]]
                sensed_ids = {id(obs) for obs in sensed}
                for idx, edge_line in enumerate(sensed_edges):
                    pts = boundary_points[idx]
                    if id(cfg["inflated_obstacles"][idx]) in sensed_ids:
                        edge_line.set_data(pts[:, 0], pts[:, 1])
                    else:
                        edge_line.set_data([], [])
            return artists

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(0, len(traj), cfg["animation_stride"]),
            interval=20,
            blit=True,
            repeat=False,
        )
        if save_animation:
            cfg["animation_path"].parent.mkdir(parents=True, exist_ok=True)
            ani.save(cfg["animation_path"], writer=animation.PillowWriter(fps=cfg["animation_fps"]))
    else:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=2)

    plt.tight_layout()
    plt.show()
