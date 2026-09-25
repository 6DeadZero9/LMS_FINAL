"""EKF on unicycle state [x, y, yaw, v, gyro_bias]."""

from __future__ import annotations

import numpy as np

from sensor_engineering.tooling import wrap_angle

# state indices
X, Y, YAW, V, BG = 0, 1, 2, 3, 4
N = 5


def _f(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    px, py, yaw, v, bg = map(float, x)
    ax, wz = float(u[0]), float(u[1])
    return np.array(
        [px + v * np.cos(yaw) * dt, py + v * np.sin(yaw) * dt, wrap_angle(yaw + (wz - bg) * dt), v + ax * dt, bg]
    )


def _F(x: np.ndarray, dt: float) -> np.ndarray:
    yaw, v = float(x[YAW]), float(x[V])
    F = np.eye(N)
    F[X, YAW] = -v * np.sin(yaw) * dt
    F[X, V] = np.cos(yaw) * dt
    F[Y, YAW] = v * np.cos(yaw) * dt
    F[Y, V] = np.sin(yaw) * dt
    F[YAW, BG] = -dt
    return F


def process_noise(dt: float, sig_ax: float, sig_wz: float, sig_bg: float) -> np.ndarray:
    q = np.zeros((N, N))
    q[X, X] = q[Y, Y] = 1e-5
    q[YAW, YAW] = (sig_wz * dt) ** 2
    q[V, V] = (sig_ax * dt) ** 2
    q[BG, BG] = (sig_bg * np.sqrt(dt)) ** 2
    return q


class ExtendedKalmanFilter:
    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.x = np.array([x, y, yaw, 0.0, 0.0], dtype=float)
        self.P = np.diag([0.05**2, 0.05**2, 0.05**2, 0.1**2, 0.02**2])

    def _correct(self, innov: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R + np.eye(H.shape[0]) * 1e-9
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x[YAW] = wrap_angle(self.x[YAW])

    def predict(self, u: np.ndarray, dt: float, q: np.ndarray) -> None:
        F = _F(self.x, dt)
        self.x = _f(self.x, u, dt)
        self.P = F @ self.P @ F.T + q
        self.P = 0.5 * (self.P + self.P.T)

    def update_speed(self, v_meas: float, r_v: float) -> None:
        H = np.zeros((1, N))
        H[0, V] = 1.0
        self._correct(np.array([v_meas - self.x[V]]), H, np.diag([r_v**2]))

    def update_landmark(
        self,
        z_range: float,
        z_bearing: float,
        landmark_xy: tuple[float, float],
        r_range: float,
        r_bearing: float,
        gate: float,
    ) -> bool:
        dx = landmark_xy[0] - self.x[X]
        dy = landmark_xy[1] - self.x[Y]
        q = max(dx * dx + dy * dy, 1e-6)
        rng = max(np.sqrt(q), 1e-3)
        innov = np.array([z_range - rng, wrap_angle(z_bearing - (np.atan2(dy, dx) - self.x[YAW]))])
        H = np.zeros((2, N))
        H[0, X], H[0, Y] = -dx / rng, -dy / rng
        H[1, X], H[1, Y], H[1, YAW] = dy / q, -dx / q, -1.0
        R = np.diag([r_range**2, r_bearing**2])
        S = H @ self.P @ H.T + R
        try:
            if float(innov.T @ np.linalg.solve(S, innov)) > gate:
                return False
        except np.linalg.LinAlgError:
            return False
        self._correct(innov, H, R)
        return True

    def pose(self) -> tuple[float, float, float, float]:
        return float(self.x[X]), float(self.x[Y]), float(self.x[YAW]), float(self.x[V])
