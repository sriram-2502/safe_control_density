import numpy as np


def _pnorm_points(center, radius, p=2.0, num=200, scale=None, angle=0.0):
    """Parametric p-norm contour (superellipse) for plotting."""
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, num=num, endpoint=True)
    c = np.cos(theta)
    s = np.sin(theta)
    p = float(p)
    x = np.sign(c) * (np.abs(c) ** (2.0 / p))
    y = np.sign(s) * (np.abs(s) ** (2.0 / p))
    pts = np.stack([x, y], axis=1) * radius
    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        pts = pts * scale[None, :]
    if angle:
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
    return pts + center[None, :]


def plot_obstacle(
    ax,
    center,
    r1,
    r2=None,
    color="0.5",
    p=2.0,
    scale=None,
    angle=0.0,
    fill=False,
    fill_alpha=0.15,
):
    inner = _pnorm_points(center, r1, p=p, scale=scale, angle=angle)
    ax.plot(inner[:, 0], inner[:, 1], color=color, linewidth=2)
    if fill:
        ax.fill(inner[:, 0], inner[:, 1], color=color, alpha=fill_alpha)
    if r2 is not None:
        outer = _pnorm_points(center, r2, p=p, scale=scale, angle=angle)
        ax.plot(outer[:, 0], outer[:, 1], color=color, linestyle="--", linewidth=1)


def plot_start(ax, start, color="tab:blue"):
    ax.scatter(start[0], start[1], s=80, c=color, marker="o", edgecolor="k", zorder=3)


def plot_goal(ax, goal, color="tab:green"):
    ax.scatter(goal[0], goal[1], s=80, c=color, marker="*", edgecolor="k", zorder=3)
