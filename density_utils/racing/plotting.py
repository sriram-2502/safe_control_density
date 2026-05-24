import numpy as np
from matplotlib import patches


def plot_track(ax, track, *, center_line=True, color="tab:blue"):
    _, left, center, right = track.sample_boundaries()
    ax.plot(left[:, 0], left[:, 1], color=color, linewidth=2.0)
    ax.plot(right[:, 0], right[:, 1], color=color, linewidth=2.0)
    if center_line:
        ax.plot(center[:, 0], center[:, 1], "--", color="tab:red", linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def car_patch(xglob, *, length=0.4, width=0.2, facecolor="none", edgecolor="black"):
    xglob = np.asarray(xglob, dtype=float)
    psi, x, y = xglob[3], xglob[4], xglob[5]
    forward = np.array([np.cos(psi), np.sin(psi)])
    left = np.array([-np.sin(psi), np.cos(psi)])
    center = np.array([x, y])
    corners = np.array(
        [
            center + 0.5 * length * forward + 0.5 * width * left,
            center + 0.5 * length * forward - 0.5 * width * left,
            center - 0.5 * length * forward - 0.5 * width * left,
            center - 0.5 * length * forward + 0.5 * width * left,
        ]
    )
    return patches.Polygon(
        corners,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=4,
    )
