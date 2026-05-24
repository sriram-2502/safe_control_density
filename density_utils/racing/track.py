import numpy as np


def wrap_angle(angle):
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _sign(value):
    return 1.0 if value >= 0.0 else -1.0


def _segment_index(point_and_tangent, lap_length, s):
    s_wrapped = float(s) % float(lap_length)
    starts = point_and_tangent[:, 3]
    lengths = point_and_tangent[:, 4]
    mask = (s_wrapped >= starts) & (s_wrapped <= starts + lengths + 1e-9)
    indices = np.where(mask)[0]
    if indices.size == 0:
        return point_and_tangent.shape[0] - 1, s_wrapped
    return int(indices[0]), s_wrapped


class ClosedTrack:
    """Closed curvilinear track from ``car-racing`` segment specs.

    The input ``spec`` has one row per segment:

    ``[segment_length, turn_radius]``.

    ``turn_radius == 0`` means a straight segment; otherwise the signed radius
    defines a circular arc.  The generated table columns are
    ``[x_end, y_end, psi_end, s_start, segment_length, curvature]``.
    """

    def __init__(self, spec, track_width=0.8):
        self.spec = np.asarray(spec, dtype=float)
        self.width = float(track_width)
        self.point_and_tangent = self._build_point_and_tangent(self.spec)
        self.lap_length = float(
            self.point_and_tangent[-1, 3] + self.point_and_tangent[-1, 4]
        )

    @staticmethod
    def _build_point_and_tangent(spec):
        point_and_tangent = np.zeros((spec.shape[0] + 1, 6), dtype=float)
        for i, (length, radius) in enumerate(spec):
            if i == 0:
                x_start = y_start = psi_start = s_start = 0.0
            else:
                x_start, y_start, psi_start = point_and_tangent[i - 1, :3]
                s_start = point_and_tangent[i - 1, 3] + point_and_tangent[i - 1, 4]

            if radius == 0.0:
                x_end = x_start + length * np.cos(psi_start)
                y_end = y_start + length * np.sin(psi_start)
                psi_end = psi_start
                curvature = 0.0
            else:
                direction = _sign(radius)
                abs_radius = abs(radius)
                center_x = x_start + abs_radius * np.cos(
                    psi_start + direction * np.pi / 2.0
                )
                center_y = y_start + abs_radius * np.sin(
                    psi_start + direction * np.pi / 2.0
                )
                span = length / abs_radius
                psi_end = wrap_angle(psi_start + span * direction)
                angle_normal = wrap_angle(direction * np.pi / 2.0 + psi_start)
                angle = -(np.pi - abs(angle_normal)) * _sign(angle_normal)
                x_end = center_x + abs_radius * np.cos(angle + direction * span)
                y_end = center_y + abs_radius * np.sin(angle + direction * span)
                curvature = 1.0 / radius

            point_and_tangent[i, :] = np.array(
                [x_end, y_end, psi_end, s_start, length, curvature],
                dtype=float,
            )

        x_last, y_last = point_and_tangent[-2, :2]
        closing_length = np.hypot(x_last, y_last)
        point_and_tangent[-1, :] = np.array(
            [
                0.0,
                0.0,
                0.0,
                point_and_tangent[-2, 3] + point_and_tangent[-2, 4],
                closing_length,
                0.0,
            ],
            dtype=float,
        )
        return point_and_tangent

    def get_global_position(self, s, ey):
        idx, s_wrapped = _segment_index(self.point_and_tangent, self.lap_length, s)
        segment = self.point_and_tangent[idx]
        if segment[5] == 0.0:
            x_end, y_end, psi, s_start, length, _ = segment
            if idx == 0:
                x_start, y_start = 0.0, 0.0
            else:
                x_start, y_start = self.point_and_tangent[idx - 1, :2]
            ds = s_wrapped - s_start
            xy = (1.0 - ds / length) * np.array([x_start, y_start])
            xy += (ds / length) * np.array([x_end, y_end])
            xy += ey * np.array([np.cos(psi + np.pi / 2.0), np.sin(psi + np.pi / 2.0)])
            return float(xy[0]), float(xy[1])

        radius = 1.0 / segment[5]
        direction = _sign(radius)
        abs_radius = abs(radius)
        if idx == 0:
            x_start, y_start, psi_start = 0.0, 0.0, 0.0
        else:
            x_start, y_start, psi_start = self.point_and_tangent[idx - 1, :3]
        center = np.array(
            [
                x_start + abs_radius * np.cos(psi_start + direction * np.pi / 2.0),
                y_start + abs_radius * np.sin(psi_start + direction * np.pi / 2.0),
            ]
        )
        span = (s_wrapped - segment[3]) / abs_radius
        angle_normal = wrap_angle(direction * np.pi / 2.0 + psi_start)
        angle = -(np.pi - abs(angle_normal)) * _sign(angle_normal)
        xy = center + (abs_radius - direction * ey) * np.array(
            [np.cos(angle + direction * span), np.sin(angle + direction * span)]
        )
        return float(xy[0]), float(xy[1])

    def get_orientation(self, s, ey=0.0):
        idx, s_wrapped = _segment_index(self.point_and_tangent, self.lap_length, s)
        segment = self.point_and_tangent[idx]
        if segment[5] == 0.0:
            return float(segment[2])
        radius = 1.0 / segment[5]
        direction = _sign(radius)
        if idx == 0:
            psi_start = 0.0
        else:
            psi_start = self.point_and_tangent[idx - 1, 2]
        span = (s_wrapped - segment[3]) / abs(radius)
        angle_normal = wrap_angle(direction * np.pi / 2.0 + psi_start)
        angle = -(np.pi - abs(angle_normal)) * _sign(angle_normal)
        return wrap_angle(angle + direction * span + np.pi / 2.0)

    def get_curvature(self, s):
        idx, _ = _segment_index(self.point_and_tangent, self.lap_length, s)
        return float(self.point_and_tangent[idx, 5])

    def sample_centerline(self, n=500):
        s_grid = np.linspace(0.0, self.lap_length, int(n), endpoint=True)
        xy = np.array([self.get_global_position(s, 0.0) for s in s_grid])
        return s_grid, xy

    def sample_boundaries(self, n=500):
        s_grid = np.linspace(0.0, self.lap_length, int(n), endpoint=True)
        left = np.array([self.get_global_position(s, self.width) for s in s_grid])
        center = np.array([self.get_global_position(s, 0.0) for s in s_grid])
        right = np.array([self.get_global_position(s, -self.width) for s in s_grid])
        return s_grid, left, center, right
