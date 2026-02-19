import numpy as np


def p_norm_bump(x, center, r1, r2, p=2, scale=None, angle=0.0):
    """Smooth bump based on a p-norm ball with inner/outer radii r1/r2."""
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    dx = x - center
    if angle:
        c = np.cos(-angle)
        s = np.sin(-angle)
        dx = np.array([c * dx[0] - s * dx[1], s * dx[0] + c * dx[1]])
    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        if np.any(scale <= 0.0):
            raise ValueError("scale entries must be positive")
        dx = dx / scale
    norm_p_p = np.sum(np.abs(dx) ** p)
    denom = max(r2 ** p - r1 ** p, 1e-12)
    m = (norm_p_p - r1 ** p) / denom

    if m <= 0.0:
        return 0.0
    if m >= 1.0:
        return 1.0

    f = np.exp(-1.0 / m)
    f_shift = np.exp(-1.0 / (1.0 - m))
    return f / (f + f_shift)
