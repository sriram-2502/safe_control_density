import argparse
import sys
from time import perf_counter
from pathlib import Path

from matplotlib import animation
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANIMATION_DIR = Path(__file__).resolve().parents[1] / "animations"
DEFAULT_ANIMATION_PATH = ANIMATION_DIR / "density_mpc.gif"
DEFAULT_DIAGNOSTIC_PATH = ANIMATION_DIR / "density_mpc_state_controls.png"

from density_utils.racing import ClosedTrack, dynamic_bicycle_step, pid_tracking_control
from density_utils.racing.plotting import car_patch, plot_track

from config import (
    BICYCLE_PARAMS,
    DENSITY_COST_WEIGHT,
    DENSITY_MIN,
    DENSITY_SLACK_WEIGHT,
    DENSITY_TRANSPORT_DIVERGENCE,
    DT,
    ENFORCE_HARD_SUPERELLIPSE,
    INITIAL_CONTROL,
    INITIAL_CURVILINEAR_STATE,
    MATRIX_A,
    MATRIX_B,
    MAX_SOLVER_ITER,
    MPC_HORIZON,
    NUM_STEPS,
    OBSTACLE_DENSITY_MODE,
    OBSTACLE_DENSITY_SHARPNESS,
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
from density_utils.controllers import SOLVER_CHOICES
from density_utils.controllers.solver_utils import require_solver


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


def safety_density(track, xcurv, obstacle_predictions, k, density_mode=None):
    rho = track_density(xcurv[5])
    for obs_pred in obstacle_predictions:
        rho *= obstacle_density(track, xcurv, obs_pred, k, density_mode=density_mode)
    return max(float(rho), 0.0)


def signed_lap_difference(s_ego, s_obs, lap_length):
    ds = float(s_ego) - float(s_obs)
    return (ds + 0.5 * lap_length) % lap_length - 0.5 * lap_length


def obstacle_density(track, xcurv, obs_pred, k, density_mode=None):
    if density_mode is None:
        density_mode = OBSTACLE_DENSITY_MODE
    z = obstacle_superellipse_value(track, xcurv, obs_pred, k)
    z_safe = 1.0 + OBSTACLE_MARGIN
    z_outer = z_safe + OBSTACLE_TRANSITION
    if density_mode == "bump":
        return smooth_scalar_bump(z, z_safe, z_outer)
    if density_mode != "sigmoid":
        raise ValueError(
            f"unknown obstacle density mode={density_mode!r}; "
            "expected 'sigmoid' or 'bump'"
        )
    z_mid = 0.5 * (z_safe + z_outer)
    scaled = (z - z_mid) / max(z_outer - z_safe, 1e-12)
    return float(1.0 / (1.0 + np.exp(-OBSTACLE_DENSITY_SHARPNESS * scaled)))


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


def mpc_objective(
    u_flat, track, xcurv0, obstacle_predictions, prev_control, density_mode=None
):
    curv_states, _, u_seq, slack = rollout(track, xcurv0, u_flat, MPC_HORIZON)
    cost = 0.0
    last_u = np.asarray(prev_control, dtype=float)

    for k in range(MPC_HORIZON):
        state = curv_states[k + 1]
        control = u_seq[k]
        du = control - last_u
        rho = safety_density(
            track, state, obstacle_predictions, k + 1, density_mode=density_mode
        )
        cost += Q_SPEED * (state[0] - TARGET_SPEED) ** 2
        cost += Q_HEADING * state[3] ** 2
        cost += Q_LATERAL * (state[5] - TARGET_LATERAL_ERROR) ** 2
        cost += R_STEER * control[0] ** 2 + R_ACCEL * control[1] ** 2
        cost += R_STEER_RATE * du[0] ** 2 + R_ACCEL_RATE * du[1] ** 2
        cost += DENSITY_COST_WEIGHT * (1.0 - rho) ** 2
        last_u = control

    cost += DENSITY_SLACK_WEIGHT * float(np.sum(slack**2))
    return float(cost)


def density_constraints(u_flat, track, xcurv0, obstacle_predictions, density_mode=None):
    curv_states, _, _, slack = rollout(track, xcurv0, u_flat, MPC_HORIZON)
    values = []
    for k in range(MPC_HORIZON):
        rho = safety_density(
            track, curv_states[k], obstacle_predictions, k, density_mode=density_mode
        )
        rho_next = safety_density(
            track,
            curv_states[k + 1],
            obstacle_predictions,
            k + 1,
            density_mode=density_mode,
        )
        density_transport = (
            (rho_next - rho)
            + DT * DENSITY_TRANSPORT_DIVERGENCE * rho
            - DT * slack[k] * rho
        )
        values.append(density_transport)
        values.append(rho_next - DENSITY_MIN)
        if ENFORCE_HARD_SUPERELLIPSE:
            for obs_pred in obstacle_predictions:
                z_next = obstacle_superellipse_value(
                    track, curv_states[k + 1], obs_pred, k + 1
                )
                values.append(z_next - (1.0 + OBSTACLE_MARGIN))
    return np.asarray(values, dtype=float)


def solve_mpc(
    track,
    xcurv,
    obstacle_predictions,
    prev_control,
    warm_start=None,
    density_mode=None,
    solver="auto",
):
    require_solver(solver, ("scipy_slsqp",), controller="racing density_mpc")
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
        u0 = np.concatenate([u0_controls, np.zeros(MPC_HORIZON)])
    else:
        u0 = warm_start.copy()

    bounds = []
    for _ in range(MPC_HORIZON):
        bounds.append((-SYSTEM_LIMITS.delta_max, SYSTEM_LIMITS.delta_max))
        bounds.append((-SYSTEM_LIMITS.a_max, SYSTEM_LIMITS.a_max))
    for _ in range(MPC_HORIZON):
        bounds.append((0.0, None))

    mpc_start = perf_counter()
    result = minimize(
        mpc_objective,
        u0,
        args=(track, xcurv, obstacle_predictions, prev_control, density_mode),
        method="SLSQP",
        bounds=bounds,
        constraints={
            "type": "ineq",
            "fun": density_constraints,
            "args": (track, xcurv, obstacle_predictions, density_mode),
        },
        options={"maxiter": MAX_SOLVER_ITER, "ftol": 1e-4, "disp": False},
    )
    solve_time = perf_counter() - mpc_start

    if result.success:
        u_seq, slack = split_decision(result.x, MPC_HORIZON)
        predicted_curv, predicted_glob, _, _ = rollout(
            track, xcurv, result.x, MPC_HORIZON
        )
    else:
        u_seq, slack = split_decision(u0, MPC_HORIZON)
        predicted_curv, predicted_glob, _, _ = rollout(
            track, xcurv, u0, MPC_HORIZON
        )
    shifted_controls = np.vstack([u_seq[1:], u_seq[-1:]]).reshape(-1)
    shifted_slack = np.zeros_like(slack)
    shifted = np.concatenate([shifted_controls, shifted_slack])
    return u_seq[0], shifted, result, solve_time, predicted_glob


def obstacle_distances(xglob, obstacle_predictions):
    ego_xy = np.asarray(xglob[4:6], dtype=float)
    return np.array(
        [
            np.linalg.norm(ego_xy - np.asarray(obs["states"][0, :2], dtype=float))
            for obs in obstacle_predictions
        ],
        dtype=float,
    )


def simulate(num_steps=NUM_STEPS, density_mode=None, solver="auto", print_progress=True):
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
    solve_time_log = []
    obstacle_distance_log = []
    prediction_log = []

    for step in range(num_steps):
        time = step * DT
        obstacle_predictions = predict_obstacles(track, OBSTACLES, time, MPC_HORIZON)
        control, warm_start, result, solve_time, predicted_glob = solve_mpc(
            track,
            xcurv,
            obstacle_predictions,
            prev_control,
            warm_start=warm_start,
            density_mode=density_mode,
            solver=solver,
        )

        xcurv, xglob = step_state(track, xcurv, xglob, control)

        current_obstacles = predict_obstacles(track, OBSTACLES, time + DT, 0)
        rho = safety_density(track, xcurv, current_obstacles, 0, density_mode=density_mode)
        distances = obstacle_distances(xglob, current_obstacles)
        curv_log.append(xcurv.copy())
        glob_log.append(xglob.copy())
        control_log.append(control.copy())
        density_log.append(rho)
        obstacle_log.append(current_obstacles)
        solver_success.append(bool(result.success))
        solve_time_log.append(solve_time)
        obstacle_distance_log.append(distances)
        prediction_log.append(predicted_glob.copy())
        if print_progress:
            status = "ok" if result.success else "fallback"
            print(
                f"step {step + 1:03d}/{num_steps:03d} | "
                f"solve_time={solve_time:.3f}s | {status}"
            )
        prev_control = control

    return {
        "track": track,
        "curv": np.asarray(curv_log),
        "glob": np.asarray(glob_log),
        "controls": np.asarray(control_log),
        "density": np.asarray(density_log),
        "obstacle_distances": np.asarray(obstacle_distance_log),
        "obstacles": obstacle_log,
        "predictions": np.asarray(prediction_log),
        "solver_success": np.asarray(solver_success, dtype=bool),
        "solve_times": np.asarray(solve_time_log, dtype=float),
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
    controls = result["controls"]
    density = result["density"]
    obstacle_log = result["obstacles"]
    prediction_log = result["predictions"]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(right=0.80)
    plot_track(
        ax,
        track,
        center_line=False,
        color="0.25",
        center_color="black",
        center_linestyle="--",
    )
    trajectory, = ax.plot([], [], color="tab:green", linewidth=2.0)
    prediction_line, = ax.plot(
        [],
        [],
        color="black",
        linewidth=1.6,
        alpha=0.9,
        label="MPC prediction",
    )
    ego = car_patch(glob_log[0], facecolor="tab:green", edgecolor="black")
    ax.add_patch(ego)
    obstacle_patches = [
        draw_obstacle(ax, obs["states"][0], obs["config"], "tab:red")
        for obs in obstacle_log[0]
    ]
    ax.set_title("Density MPC-CDF Racing Example")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    ax_wheel = fig.add_axes([0.805, 0.64, 0.17, 0.18])
    ax_wheel.set_aspect("equal", adjustable="box")
    ax_wheel.set_xlim(-1.35, 1.35)
    ax_wheel.set_ylim(-1.45, 1.45)
    ax_wheel.axis("off")
    ax_wheel.text(0.0, 1.34, "steer", ha="center", va="center", fontsize=8)
    wheel = patches.Circle((0.0, 0.0), 0.9, fill=False, color="0.2", linewidth=1.4)
    ax_wheel.add_patch(wheel)
    wheel_pointer, = ax_wheel.plot([0.0, 0.0], [0.0, 0.75], color="tab:purple", linewidth=2.0)
    wheel_text = ax_wheel.text(0.0, -1.08, "+0 deg", ha="center", va="top", fontsize=7)

    ax_bars = fig.add_axes([0.805, 0.13, 0.17, 0.23])
    bar_labels = ["accel", "brake", "density"]
    y_pos = np.arange(len(bar_labels))
    bar_colors = ["tab:green", "tab:red", "tab:blue"]
    bars = ax_bars.barh(
        y_pos,
        np.zeros(len(bar_labels)),
        height=0.55,
        color=bar_colors,
        alpha=0.85,
    )
    ax_bars.set_yticks(y_pos, labels=bar_labels, fontsize=7)
    ax_bars.set_xlim(0.0, 1.0)
    ax_bars.set_xticks([0.0, 0.5, 1.0])
    ax_bars.tick_params(axis="x", labelsize=7)
    ax_bars.set_title("MPC", fontsize=8, pad=2)
    ax_bars.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax_bars.invert_yaxis()
    bar_texts = [
        ax_bars.text(0.03, y, "0.00", va="center", ha="left", fontsize=7)
        for y in y_pos
    ]

    def update(frame):
        trajectory.set_data(glob_log[: frame + 1, 4], glob_log[: frame + 1, 5])
        ego.set_xy(car_patch(glob_log[frame], facecolor="tab:green").get_xy())
        pred_index = min(frame, len(prediction_log) - 1)
        prediction_line.set_data(
            prediction_log[pred_index, :, 4],
            prediction_log[pred_index, :, 5],
        )
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
            delta = 0.0
            accel = 0.0
            rho = density[0] if len(density) else 1.0
        else:
            control_index = min(frame - 1, len(controls) - 1)
            delta = controls[control_index, 0]
            accel = controls[control_index, 1]
            rho = density[control_index]

        telemetry_values = np.array(
            [
                np.clip(max(accel, 0.0) / SYSTEM_LIMITS.a_max, 0.0, 1.0),
                np.clip(max(-accel, 0.0) / SYSTEM_LIMITS.a_max, 0.0, 1.0),
                np.clip(rho, 0.0, 1.0),
            ],
            dtype=float,
        )
        raw_text = [
            f"{max(accel, 0.0):.2f}",
            f"{max(-accel, 0.0):.2f}",
            f"{rho:.2f}",
        ]
        steer_norm = np.clip(delta / SYSTEM_LIMITS.delta_max, -1.0, 1.0)
        wheel_angle = np.pi / 2.0 - steer_norm * np.pi / 2.0
        wheel_pointer.set_data(
            [0.0, 0.78 * np.cos(wheel_angle)],
            [0.0, 0.78 * np.sin(wheel_angle)],
        )
        wheel_text.set_text(f"{np.degrees(delta):+.0f} deg")
        for bar, value, text_obj, label in zip(
            bars, telemetry_values, bar_texts, raw_text
        ):
            bar.set_width(value)
            text_obj.set_x(min(value + 0.03, 0.88))
            text_obj.set_ha("left")
            text_obj.set_text(label)

        return (
            trajectory,
            prediction_line,
            ego,
            *obstacle_patches,
            wheel_pointer,
            wheel_text,
            *bars,
            *bar_texts,
        )

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


def plot_state_control_diagnostics(result, save_path=None, show=False):
    curv = result["curv"]
    controls = result["controls"]
    density = result["density"]
    obstacle_distances = result["obstacle_distances"]

    t_state = DT * np.arange(len(curv))
    t_control = DT * np.arange(len(controls))

    fig, axes = plt.subplots(5, 2, figsize=(11, 12), sharex=False)
    axes = np.asarray(axes)

    state_labels = [
        (0, r"$v_x$ [m/s]"),
        (1, r"$v_y$ [m/s]"),
        (2, r"$\omega_z$ [rad/s]"),
        (3, r"$e_\psi$ [rad]"),
        (4, r"$s$ [m]"),
        (5, r"$e_y$ [m]"),
    ]
    for ax, (idx, label) in zip(axes[:3].ravel(), state_labels):
        ax.plot(t_state, curv[:, idx], linewidth=1.8, color="tab:blue", label=label)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[3, 0].plot(
        t_control,
        controls[:, 0],
        linewidth=1.8,
        color="tab:green",
        label=r"$\delta$ [rad]",
    )
    axes[3, 0].plot(
        t_control,
        controls[:, 1],
        linewidth=1.8,
        color="tab:orange",
        label=r"$a$ [m/s$^2$]",
    )
    axes[3, 0].set_ylabel("control")
    axes[3, 0].set_xlabel("time [s]")
    axes[3, 0].grid(True, linestyle="--", alpha=0.35)
    axes[3, 0].legend(loc="best")

    axes[3, 1].plot(
        t_control,
        density,
        linewidth=1.8,
        color="tab:red",
        label=r"$\rho(x)$",
    )
    axes[3, 1].set_ylabel("density")
    axes[3, 1].set_xlabel("time [s]")
    axes[3, 1].grid(True, linestyle="--", alpha=0.35)
    axes[3, 1].legend(loc="best")

    for obs_idx, obs in enumerate(OBSTACLES):
        ax = axes[4, obs_idx]
        ax.plot(
            t_control,
            obstacle_distances[:, obs_idx],
            linewidth=1.8,
            color=f"C{obs_idx}",
            label=f"distance to {obs['name']} [m]",
        )
        ax.set_ylabel("distance [m]")
        ax.set_xlabel("time [s]")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    for ax in axes[4, len(OBSTACLES):]:
        ax.axis("off")

    axes[0, 0].set_title("Density MPC-CDF States")
    axes[0, 1].set_title("Track Coordinates and Diagnostics")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Density MPC-CDF racing example.")
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--save-animation", type=Path, default=None)
    parser.add_argument("--no-save-animation", action="store_true")
    parser.add_argument("--save-diagnostics", type=Path, default=None)
    parser.add_argument("--no-save-diagnostics", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--show-diagnostics", action="store_true")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto", help="Optimizer backend.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--density-mode",
        choices=("sigmoid", "bump"),
        default=OBSTACLE_DENSITY_MODE,
        help="Obstacle density mapping used in the MPC-CDF constraint.",
    )
    parser.add_argument(
        "--save-default-animation",
        action="store_true",
        help="Deprecated: default behavior already saves to this example's animations/density_mpc.gif.",
    )
    args = parser.parse_args()

    result = simulate(
        num_steps=args.steps,
        density_mode=args.density_mode,
        solver=args.solver,
        print_progress=not args.quiet,
    )
    save_path = args.save_animation or DEFAULT_ANIMATION_PATH
    if args.no_save_animation:
        save_path = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    diagnostic_path = args.save_diagnostics or DEFAULT_DIAGNOSTIC_PATH
    if args.no_save_diagnostics:
        diagnostic_path = None
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    plot_state_control_diagnostics(
        result,
        save_path=diagnostic_path,
        show=(not args.no_diagnostics) or args.show_diagnostics,
    )

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
    print(f"Mean solve time: {np.mean(result['solve_times']):.3f}s")
    print(f"Max solve time: {np.max(result['solve_times']):.3f}s")


if __name__ == "__main__":
    main()
