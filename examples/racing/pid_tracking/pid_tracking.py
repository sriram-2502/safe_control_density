import argparse
import sys
from pathlib import Path

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from density_utils.racing import ClosedTrack, dynamic_bicycle_step, pid_tracking_control
from density_utils.racing.plotting import car_patch, plot_track

from config import (
    BICYCLE_PARAMS,
    DT,
    INITIAL_CURVILINEAR_STATE,
    INTEGRATION_DT,
    NUM_STEPS,
    SYSTEM_LIMITS,
    TARGET_LATERAL_ERROR,
    TARGET_SPEED,
    TRACK_FILE,
    TRACK_WIDTH,
)


def global_state_from_curvilinear(track, xcurv):
    x, y = track.get_global_position(xcurv[4], xcurv[5])
    psi = track.get_orientation(xcurv[4], xcurv[5]) + xcurv[3]
    return np.array([xcurv[0], xcurv[1], xcurv[2], psi, x, y], dtype=float)


def run_pid_tracking(num_steps=NUM_STEPS):
    track_spec = np.loadtxt(TRACK_FILE, delimiter=",")
    track = ClosedTrack(track_spec, track_width=TRACK_WIDTH)
    xcurv = INITIAL_CURVILINEAR_STATE.copy()
    xglob = global_state_from_curvilinear(track, xcurv)

    substeps = max(1, int(round(DT / INTEGRATION_DT)))
    dt_inner = DT / substeps

    curv_log = [xcurv.copy()]
    glob_log = [xglob.copy()]
    ctrl_log = []

    for _ in range(num_steps):
        control = pid_tracking_control(
            xcurv,
            target_speed=TARGET_SPEED,
            target_lateral=TARGET_LATERAL_ERROR,
            limits=SYSTEM_LIMITS,
        )
        ctrl_log.append(control.copy())

        for _ in range(substeps):
            curvature = track.get_curvature(xcurv[4])
            xcurv, xglob = dynamic_bicycle_step(
                xcurv,
                xglob,
                control,
                curvature,
                dt_inner,
                params=BICYCLE_PARAMS,
            )
            xcurv[0] = np.clip(xcurv[0], SYSTEM_LIMITS.v_min, SYSTEM_LIMITS.v_max)

        # The curvilinear state is the source of truth for track-relative plots.
        xglob = global_state_from_curvilinear(track, xcurv)
        curv_log.append(xcurv.copy())
        glob_log.append(xglob.copy())

    return track, np.asarray(curv_log), np.asarray(glob_log), np.asarray(ctrl_log)


def plot_result(track, curv_log, glob_log, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_track(ax, track, center_line=True, color="0.25")
    ax.plot(glob_log[:, 4], glob_log[:, 5], color="tab:green", linewidth=2.0)
    ax.scatter(glob_log[0, 4], glob_log[0, 5], color="tab:blue", s=40, label="start")
    ax.scatter(glob_log[-1, 4], glob_log[-1, 5], color="tab:red", s=40, label="end")
    ax.add_patch(car_patch(glob_log[-1], facecolor="tab:green", edgecolor="black"))
    ax.set_title("PID Tracking on L-shaped Racing Track")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_result(track, glob_log, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_track(ax, track, center_line=True, color="0.25")
    trajectory, = ax.plot([], [], color="tab:green", linewidth=2.0)
    car = car_patch(glob_log[0], facecolor="tab:green", edgecolor="black")
    ax.add_patch(car)
    ax.scatter(glob_log[0, 4], glob_log[0, 5], color="tab:blue", s=40, label="start")
    ax.set_title("PID Tracking on L-shaped Racing Track")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    def update(frame):
        trajectory.set_data(glob_log[: frame + 1, 4], glob_log[: frame + 1, 5])
        car.set_xy(car_patch(glob_log[frame], facecolor="tab:green").get_xy())
        return trajectory, car

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(glob_log),
        interval=40,
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
    parser = argparse.ArgumentParser(
        description="Baseline PID tracking example for the racing environment."
    )
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--save-animation", type=Path, default=None)
    parser.add_argument("--static", action="store_true", help="Show a static trajectory plot.")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    track, curv_log, glob_log, ctrl_log = run_pid_tracking(num_steps=args.steps)
    if args.static or args.save is not None:
        plot_result(track, curv_log, glob_log, save_path=args.save, show=args.static)
    if not args.no_animation or args.save_animation is not None:
        animate_result(
            track,
            glob_log,
            save_path=args.save_animation,
            show=not args.no_animation,
        )

    final = curv_log[-1]
    print(
        "Final curvilinear state: "
        f"vx={final[0]:.3f}, epsi={final[3]:.3f}, s={final[4]:.3f}, ey={final[5]:.3f}"
    )
    print(f"Mean |ey|: {np.mean(np.abs(curv_log[:, 5])):.3f}")
    print(f"Max |delta|: {np.max(np.abs(ctrl_log[:, 0])):.3f}")


if __name__ == "__main__":
    main()
