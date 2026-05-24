from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BicycleParams:
    m: float = 2.366
    lf: float = 0.1377
    lr: float = 0.1203
    Iz: float = 0.0278
    Df: float = 0.8 * 2.366 * 9.81 / 2.0
    Cf: float = 1.25
    Bf: float = 1.0
    Dr: float = 0.8 * 2.366 * 9.81 / 2.0
    Cr: float = 1.25
    Br: float = 1.0


@dataclass(frozen=True)
class SystemLimits:
    delta_max: float = 0.5
    a_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 10.0


def pid_tracking_control(xcurv, target_speed=0.8, target_lateral=0.0, limits=None):
    """PID controller used as the first car-racing sanity check."""
    if limits is None:
        limits = SystemLimits()
    xcurv = np.asarray(xcurv, dtype=float)
    delta = -0.6 * (xcurv[5] - target_lateral) - 0.9 * xcurv[3]
    accel = 1.5 * (target_speed - xcurv[0])
    return np.array(
        [
            np.clip(delta, -limits.delta_max, limits.delta_max),
            np.clip(accel, -limits.a_max, limits.a_max),
        ],
        dtype=float,
    )


def dynamic_bicycle_step(xcurv, xglob, control, curvature, dt, params=None):
    """One Euler step of the dynamic bicycle model.

    State convention follows ``car-racing``:

    ``xcurv = [vx, vy, wz, epsi, s, ey]``
    ``xglob = [vx, vy, wz, psi, X, Y]``
    ``control = [delta, acceleration]``
    """
    if params is None:
        params = BicycleParams()
    xcurv = np.asarray(xcurv, dtype=float)
    xglob = np.asarray(xglob, dtype=float)
    delta, accel = np.asarray(control, dtype=float)

    vx, vy, wz, epsi, s, ey = xcurv
    psi, X, Y = xglob[3], xglob[4], xglob[5]
    vx_safe = max(abs(vx), 1e-3) * (1.0 if vx >= 0.0 else -1.0)

    alpha_f = delta - np.arctan2(vy + params.lf * wz, vx_safe)
    alpha_r = -np.arctan2(vy - params.lr * wz, vx_safe)
    fyf = 2.0 * params.Df * np.sin(params.Cf * np.arctan(params.Bf * alpha_f))
    fyr = 2.0 * params.Dr * np.sin(params.Cr * np.arctan(params.Br * alpha_r))

    xglob_next = np.zeros_like(xglob)
    xglob_next[0] = vx + dt * (accel - fyf * np.sin(delta) / params.m + wz * vy)
    xglob_next[1] = vy + dt * ((fyf * np.cos(delta) + fyr) / params.m - wz * vx)
    xglob_next[2] = wz + dt * (
        (params.lf * fyf * np.cos(delta) - params.lr * fyr) / params.Iz
    )
    xglob_next[3] = psi + dt * wz
    xglob_next[4] = X + dt * (vx * np.cos(psi) - vy * np.sin(psi))
    xglob_next[5] = Y + dt * (vx * np.sin(psi) + vy * np.cos(psi))

    denom = max(1.0 - curvature * ey, 1e-3)
    xcurv_next = np.zeros_like(xcurv)
    xcurv_next[:3] = xglob_next[:3]
    xcurv_next[3] = epsi + dt * (
        wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) * curvature / denom
    )
    xcurv_next[4] = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / denom)
    xcurv_next[5] = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
    return xcurv_next, xglob_next
