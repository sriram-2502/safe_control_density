import argparse
import sys
from pathlib import Path

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANIMATION_DIR = REPO_ROOT / "animations"
DEFAULT_ANIMATION_PATH = ANIMATION_DIR / "density_mpc.gif"

from density_utils.racing import ClosedTrack, dynamic_bicycle_step, pid_tracking_control
from density_utils.racing.plotting import car_patch, plot_track

from config import (
    BICYCLE_PARAMS,
    DENSITY_COST_WEIGHT,
    DENSITY_GAMMA,
    DENSITY_MIN,
    DENSITY_SLACK_WEIGHT,
    DT,
    INITIAL_CONTROL,
    INITIAL_CURVILINEAR_STATE,
    MATRIX_A,
    MATRIX_B,
    MAX_SOLVER_ITER,
    MPC_HORIZON,
    NUM_STEPS,
    OBSTACLE_MARGIN,
    OBSTACLE_SUPERELLIPSE_DEGREE,
    OBSTACLE_TRANSITION,
    OBSTACLES,
    Q_HEADING,
    Q_LATERAL,
    Q_SPEED,
    R_ACCEL,
    R_ACCEL_RATE,
    R_STEER,
    R_STEER_RATE,
    SYSTEM_LIMITS,
    TARGET_LATERAL_ERROR,
    TARGET_SPEED,
    TRACK_FILE,
    TRACK_MARGIN,
    TRACK_TRANSITION,
    TRACK_WIDTH,
    USE_LTI_MODEL,
)


def global_state_from_curvilinear(track, xcurv):
    x, y = track.get_global_position(xcurv[4], xcurv[5])
    psi = track.get_orientation(xcurv[4], xcurv[5]) + xcurv[3]
    return np.array([xcurv[0], xcurv[1], xcurv[2], psi, x, y], dtype=float)


def predict_obstacles(track, obstacles, start_time, horizon):
    times = start_time + DT * np.arange(horizon + 1)
    predictions = []
    for obs in obstacles:
        states = []
        for t in times:
            s = obs["initial_s"] + obs["speed"] * t
            ey = obs["initial_ey"]
            x, y = track.get_global_position(s, ey)
            psi = track.get_orientation(s, ey)
            states.append([x, y, psi, s, ey])
        predictions.append({"config": obs, "states": np.asarray(states, dtype=float)})
    return predictions


def smooth_scalar_bump(value, r1, r2):
    value = float(value)
    if value <= r1:
        return 0.0
    if value >= r2:
        return 1.0
    m = (value - r1) / max(r2 - r1, 1e-12)
    f = np.exp(-1.0 / m)
    g = np.exp(-1.0 / (1.0 - m))
    return float(f / (f + g))


def track_density(ey):
    clearance = TRACK_WIDTH - abs(float(ey))
    return smooth_scalar_bump(
        clearance,
        TRACK_MARGIN,
        TRACK_MARGIN + TRACK_TRANSITION,
    )


def safety_density(track, xcurv, obstacle_predictions, k):
    rho = track_density(xcurv[5])
    for obs_pred in obstacle_predictions:
        rho *= obstacle_density(track, xcurv, obs_pred, k)
    return max(float(rho), 0.0)


def signed_lap_difference(s_ego, s_obs, lap_length):
    ds = float(s_ego) - float(s_obs)
    return (ds + 0.5 * lap_length) % lap_length - 0.5 * lap_length


def obstacle_density(track, xcurv, obs_pred, k):
    z = obstacle_superellipse_value(track, xcurv, obs_pred, k)
    return smooth_scalar_bump(
        z,
        1.0 + OBSTACLE_MARGIN,
        1.0 + OBSTACLE_MARGIN + OBSTACLE_TRANSITION,
    )


def obstacle_superellipse_value(track, xcurv, obs_pred, k):
    obs_cfg = obs_pred["config"]
    obs_state = obs_pred["states"][k]
    ds = signed_lap_difference(xcurv[4], obs_state[3], track.lap_length)
    dey = float(xcurv[5] - obs_state[4])
    safe_length = 0.5 * (obs_cfg["length"] + 0.4)
    safe_width = 0.5 * (obs_cfg["width"] + 0.2)
    degree = OBSTACLE_SUPERELLIPSE_DEGREE
    return (abs(ds) / safe_length) ** degree + (abs(dey) / safe_width) ** degree


def step_state(track, xcurv, xglob, control):
    if USE_LTI_MODEL:
        xcurv_next = MATRIX_A @ xcurv + MATRIX_B @ control
        xcurv_next[0] = np.clip(xcurv_next[0], SYSTEM_LIMITS.v_min, SYSTEM_LIMITS.v_max)
        xglob_next = global_state_from_curvilinear(track, xcurv_next)
        return xcurv_next, xglob_next

    curvature = track.get_curvature(xcurv[4])
    xcurv_next, xglob_next = dynamic_bicycle_step(
        xcurv,
        xglob,
        control,
        curvature,
        DT,
        params=BICYCLE_PARAMS,
    )
    xcurv_next[0] = np.clip(xcurv_next[0], SYSTEM_LIMITS.v_min, SYSTEM_LIMITS.v_max)
    xglob_next = global_state_from_curvilinear(track, xcurv_next)
    return xcurv_next, xglob_next


def split_decision(decision, horizon):
    decision = np.asarray(decision, dtype=float)
    u_seq = decision[: 2 * horizon].reshape(horizon, 2)
    slack = decision[2 * horizon :]
    return u_seq, slack


def rollout(track, xcurv0, decision, horizon):
    u_seq, slack = split_decision(decision, horizon)
    xcurv = np.asarray(xcurv0, dtype=float).copy()
    xglob = global_state_from_curvilinear(track, xcurv)
    curv_states = [xcurv.copy()]
    glob_states = [xglob.copy()]

    for u in u_seq:
        xcurv, xglob = step_state(track, xcurv, xglob, u)
        curv_states.append(xcurv.copy())
        glob_states.append(xglob.copy())

    return np.asarray(curv_states), np.asarray(glob_states), u_seq, slack


def mpc_objective(u_flat, track, xcurv0, obstacle_predictions, prev_control):
    curv_states, _, u_seq, slack = rollout(track, xcurv0, u_flat, MPC_HORIZON)
    cost = 0.0
    last_u = np.asarray(prev_control, dtype=float)

    for k in range(MPC_HORIZON):
        state = curv_states[k + 1]
        control = u_seq[k]
        du = control - last_u
        rho = safety_density(track, state, obstacle_predictions, k + 1)
        cost += Q_SPEED * (state[0] - TARGET_SPEED) ** 2
        cost += Q_HEADING * state[3] ** 2
        cost += Q_LATERAL * (state[5] - TARGET_LATERAL_ERROR) ** 2
        cost += R_STEER * control[0] ** 2 + R_ACCEL * control[1] ** 2
        cost += R_STEER_RATE * du[0] ** 2 + R_ACCEL_RATE * du[1] ** 2
        cost += DENSITY_COST_WEIGHT * (1.0 - rho) ** 2
        last_u = control

    cost += DENSITY_SLACK_WEIGHT * float(np.sum(slack**2))
    return float(cost)


def density_constraints(u_flat, track, xcurv0, obstacle_predictions):
    curv_states, _, _, slack = rollout(track, xcurv0, u_flat, MPC_HORIZON)
    values = []
    rho_prev = safety_density(track, curv_states[0], obstacle_predictions, 0)
    for k in range(MPC_HORIZON):
        rho_next = safety_density(track, curv_states[k + 1], obstacle_predictions, k + 1)
        values.append(rho_next + slack[2 * k] - DENSITY_MIN)
        values.append(rho_next + slack[2 * k + 1] - (1.0 - DENSITY_GAMMA) * rho_prev)
        for obs_pred in obstacle_predictions:
            z_next = obstacle_superellipse_value(track, curv_states[k + 1], obs_pred, k + 1)
            values.append(z_next - (1.0 + OBSTACLE_MARGIN))
        rho_prev = rho_next
    return np.asarray(values, dtype=float)


def solve_mpc(track, xcurv, obstacle_predictions, prev_control, warm_start=None):
    if warm_start is None:
        u0_controls = np.tile(
            pid_tracking_control(
                xcurv,
                target_speed=TARGET_SPEED,
                target_lateral=TARGET_LATERAL_ERROR,
                limits=SYSTEM_LIMITS,
            ),
            MPC_HORIZON,
        )
        u0 = np.concatenate([u0_controls, np.zeros(2 * MPC_HORIZON)])
    else:
        u0 = warm_start.copy()

    bounds = []
    for _ in range(MPC_HORIZON):
        bounds.append((-SYSTEM_LIMITS.delta_max, SYSTEM_LIMITS.delta_max))
        bounds.append((-SYSTEM_LIMITS.a_max, SYSTEM_LIMITS.a_max))
    for _ in range(2 * MPC_HORIZON):
        bounds.append((0.0, None))

    result = minimize(
        mpc_objective,
        u0,
        args=(track, xcurv, obstacle_predictions, prev_control),
        method="SLSQP",
        bounds=bounds,
        constraints={
            "type": "ineq",
            "fun": density_constraints,
            "args": (track, xcurv, obstacle_predictions),
        },
        options={"maxiter": MAX_SOLVER_ITER, "ftol": 1e-4, "disp": False},
    )

    if result.success:
        u_seq, slack = split_decision(result.x, MPC_HORIZON)
    else:
        u_seq, slack = split_decision(u0, MPC_HORIZON)
    shifted_controls = np.vstack([u_seq[1:], u_seq[-1:]]).reshape(-1)
    shifted_slack = np.zeros_like(slack)
    shifted = np.concatenate([shifted_controls, shifted_slack])
    return u_seq[0], shifted, result


def simulate(num_steps=NUM_STEPS):
    track = ClosedTrack(np.loadtxt(TRACK_FILE, delimiter=","), track_width=TRACK_WIDTH)
    xcurv = INITIAL_CURVILINEAR_STATE.copy()
    xglob = global_state_from_curvilinear(track, xcurv)
    prev_control = INITIAL_CONTROL.copy()
    warm_start = None

    curv_log = [xcurv.copy()]
    glob_log = [xglob.copy()]
    control_log = []
    density_log = []
    obstacle_log = []
    solver_success = []

    for step in range(num_steps):
        time = step * DT
        obstacle_predictions = predict_obstacles(track, OBSTACLES, time, MPC_HORIZON)
        control, warm_start, result = solve_mpc(
            track,
            xcurv,
            obstacle_predictions,
            prev_control,
            warm_start=warm_start,
        )

        xcurv, xglob = step_state(track, xcurv, xglob, control)

        current_obstacles = predict_obstacles(track, OBSTACLES, time + DT, 0)
        rho = safety_density(track, xcurv, current_obstacles, 0)
        curv_log.append(xcurv.copy())
        glob_log.append(xglob.copy())
        control_log.append(control.copy())
        density_log.append(rho)
        obstacle_log.append(current_obstacles)
        solver_success.append(bool(result.success))
        prev_control = control

    return {
        "track": track,
        "curv": np.asarray(curv_log),
        "glob": np.asarray(glob_log),
        "controls": np.asarray(control_log),
        "density": np.asarray(density_log),
        "obstacles": obstacle_log,
        "solver_success": np.asarray(solver_success, dtype=bool),
    }


def draw_obstacle(ax, obs_state, obs_config, color):
    xglob = np.array([0.0, 0.0, 0.0, obs_state[2], obs_state[0], obs_state[1]])
    patch = car_patch(
        xglob,
        length=obs_config["length"],
        width=obs_config["width"],
        facecolor=color,
        edgecolor="black",
    )
    ax.add_patch(patch)
    return patch


def animate_result(result, save_path=None, show=True):
    track = result["track"]
    glob_log = result["glob"]
    obstacle_log = result["obstacles"]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_track(ax, track, center_line=True, color="0.25")
    trajectory, = ax.plot([], [], color="tab:green", linewidth=2.0)
    ego = car_patch(glob_log[0], facecolor="tab:green", edgecolor="black")
    ax.add_patch(ego)
    obstacle_patches = [
        draw_obstacle(ax, obs["states"][0], obs["config"], "tab:red")
        for obs in obstacle_log[0]
    ]
    ax.scatter(glob_log[0, 4], glob_log[0, 5], color="tab:blue", s=35, label="start")
    density_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    ax.set_title("Density MPC-CDF Racing Example")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    def update(frame):
        trajectory.set_data(glob_log[: frame + 1, 4], glob_log[: frame + 1, 5])
        ego.set_xy(car_patch(glob_log[frame], facecolor="tab:green").get_xy())
        obs_index = min(frame, len(obstacle_log) - 1)
        for patch, obs in zip(obstacle_patches, obstacle_log[obs_index]):
            patch.set_xy(
                car_patch(
                    np.array([0.0, 0.0, 0.0, obs["states"][0, 2], obs["states"][0, 0], obs["states"][0, 1]]),
                    length=obs["config"]["length"],
                    width=obs["config"]["width"],
                    facecolor="tab:red",
                ).get_xy()
            )
        if frame == 0:
            density_text.set_text("")
        else:
            rho = result["density"][frame - 1]
            density_text.set_text(f"density = {rho:.2f}")
        return (trajectory, ego, density_text, *obstacle_patches)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(glob_log),
        interval=50,
        blit=True,
        repeat=False,
    )
    if save_path is not None:
        ani.save(save_path, writer="pillow", fps=20)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


def main():
    parser = argparse.ArgumentParser(description="Density MPC-CDF racing example.")
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--save-animation", type=Path, default=None)
    parser.add_argument("--no-save-animation", action="store_true")
    parser.add_argument(
        "--save-default-animation",
        action="store_true",
        help="Deprecated: default behavior already saves to animations/density_mpc.gif.",
    )
    args = parser.parse_args()

    result = simulate(num_steps=args.steps)
    save_path = args.save_animation or DEFAULT_ANIMATION_PATH
    if args.no_save_animation:
        save_path = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    animate_result(
        result,
        save_path=save_path,
        show=not args.no_animation,
    )

    final = result["curv"][-1]
    print(
        "Final curvilinear state: "
        f"vx={final[0]:.3f}, epsi={final[3]:.3f}, s={final[4]:.3f}, ey={final[5]:.3f}"
    )
    print(f"Minimum density: {np.min(result['density']):.3f}")
    print(f"Solver success rate: {np.mean(result['solver_success']):.2%}")


if __name__ == "__main__":
    main()
