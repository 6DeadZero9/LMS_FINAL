"""Shared helpers: angles, suites, path, pursuit, LiDAR match, metrics, ROS msg utils."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- sensor suites -----------------------------------------------------------------

MODES = ('imu', 'imu_odom', 'imu_lidar', 'imu_odom_lidar')

MODE_LABELS = {
    'imu': 'IMU only',
    'imu_odom': 'IMU + wheel odom',
    'imu_lidar': 'IMU + LiDAR',
    'imu_odom_lidar': 'IMU + odom + LiDAR',
}

MODE_COLORS = {
    'imu': '#d62728',
    'imu_odom': '#ff7f0e',
    'imu_lidar': '#1f77b4',
    'imu_odom_lidar': '#2ca02c',
}

G = 9.80665  # m/s^2


def parse_mode(text: str) -> str | None:
    mode = text.strip().split('|')[-1].strip()
    return mode if mode in MODES else None


def mode_uses_odom(mode: str) -> bool:
    return 'odom' in mode


def mode_uses_lidar(mode: str) -> bool:
    return 'lidar' in mode


# --- angles ------------------------------------------------------------------------

def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# --- ROS / time helpers ------------------------------------------------------------

def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def odom_sample(msg) -> tuple[float, float, float, float]:
    q = msg.pose.pose.orientation
    return (
        stamp_to_sec(msg.header.stamp),
        float(msg.pose.pose.position.x),
        float(msg.pose.pose.position.y),
        quaternion_to_yaw(q.x, q.y, q.z, q.w),
    )


def make_odom(stamp, x: float, y: float, yaw: float, speed: float = 0.0):
    from nav_msgs.msg import Odometry

    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = 'map'
    msg.child_frame_id = 'base_footprint'
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.twist.twist.linear.x = float(speed)
    return msg


def body_forward_accel(ax: float, qx: float, qy: float, qz: float, qw: float) -> float:
    """Body-x acceleration with gravity removed using attitude quaternion."""
    if qx * qx + qy * qy + qz * qz + qw * qw < 0.5:
        return ax
    # conjugate rotate of world (0, 0, G) into the body frame
    tx = 2.0 * ((-qy) * G - (-qz) * 0.0)
    ty = 2.0 * ((-qz) * 0.0 - (-qx) * G)
    tz = 2.0 * ((-qx) * 0.0 - (-qy) * 0.0)
    gx = qw * tx + ((-qy) * tz - (-qz) * ty)
    return ax - gx


# --- path + pursuit ----------------------------------------------------------------

def rectangle_loop(x_min: float, x_max: float, y_min: float, y_max: float, spacing: float) -> np.ndarray:
    if spacing <= 0.0 or x_max <= x_min or y_max <= y_min:
        raise ValueError('invalid path bounds or spacing')
    corners = (
        (x_min, y_min, 0.0),
        (x_max, y_min, math.pi / 2.0),
        (x_max, y_max, math.pi),
        (x_min, y_max, -math.pi / 2.0),
    )
    samples: list[tuple[float, float, float]] = []
    for i, (x0, y0, yaw) in enumerate(corners):
        x1, y1, _ = corners[(i + 1) % 4]
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.floor(length / spacing)))
        for step in range(steps):
            t = step / steps
            samples.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), yaw))
    samples.append((x_min, y_min, -math.pi / 2.0))
    return np.asarray(samples, dtype=float)


def mean_distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> float:
    pts = np.asarray(points, float)
    line = np.asarray(polyline, float)
    if pts.size == 0 or line.size == 0:
        return float('nan')
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return float(np.mean(np.min(np.linalg.norm(pts[:, None, :] - line[None, :, :], axis=2), axis=1)))


def pure_pursuit(
    x: float,
    y: float,
    yaw: float,
    path_xy: np.ndarray,
    lookahead: float,
    speed: float,
    max_yaw_rate: float,
    min_index: int = 0,
) -> tuple[float, float, int, bool]:
    path = np.asarray(path_xy, float)
    if path.ndim != 2 or path.shape[0] < 2 or lookahead <= 0.0:
        return 0.0, 0.0, 0, True
    position = np.array([x, y], float)
    min_index = int(np.clip(min_index, 0, path.shape[0] - 1))
    closest = min_index + int(np.argmin(np.sum((path[min_index:] - position) ** 2, axis=1)))
    closest = min(closest, min_index + 8)
    finished = (
        closest >= path.shape[0] - 3
        and float(np.linalg.norm(path[-1] - position)) <= 0.25
        and min_index >= max(1, (3 * path.shape[0]) // 4)
    )
    if finished:
        return 0.0, 0.0, closest, True
    target = path.shape[0] - 1
    for i in range(closest, path.shape[0]):
        if float(np.linalg.norm(path[i] - position)) >= lookahead:
            target = i
            break
    dx, dy = float(path[target, 0] - x), float(path[target, 1] - y)
    c, s = np.cos(yaw), np.sin(yaw)
    x_r, y_r = c * dx + s * dy, -s * dx + c * dy
    dist = float(np.hypot(x_r, y_r))
    if dist < 1e-3:
        return 0.0, 0.0, closest, False
    yaw_rate = float(np.clip(speed * 2.0 * y_r / (dist * dist), -max_yaw_rate, max_yaw_rate))
    if x_r < 0.0:
        turn = max_yaw_rate if y_r >= 0.0 else -max_yaw_rate
        return float(0.35 * speed), float(turn), closest, False
    return float(speed), yaw_rate, closest, False


# --- LiDAR poles -------------------------------------------------------------------

@dataclass(frozen=True)
class Cluster:
    x: float
    y: float


@dataclass(frozen=True)
class LandmarkFix:
    range_m: float
    bearing_rad: float
    map_xy: tuple[float, float]


def clusters_from_ranges(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float = 0.12,
    range_max: float = 3.4,
    gap_m: float = 0.25,
    min_points: int = 2,
) -> list[Cluster]:
    values = np.asarray(ranges, float).reshape(-1)
    clusters: list[Cluster] = []
    bucket: list[tuple[float, float]] = []

    def flush() -> None:
        if len(bucket) >= min_points:
            pts = np.asarray(bucket, float)
            clusters.append(Cluster(float(pts[:, 0].mean()), float(pts[:, 1].mean())))
        bucket.clear()

    prev = None
    for i, rng in enumerate(values):
        if not np.isfinite(rng) or rng < range_min or rng > range_max:
            flush()
            prev = None
            continue
        angle = angle_min + i * angle_increment
        pt = (rng * np.cos(angle), rng * np.sin(angle))
        if prev is not None and float(np.hypot(pt[0] - prev[0], pt[1] - prev[1])) > gap_m:
            flush()
        bucket.append(pt)
        prev = pt
    flush()
    return clusters


def match_clusters(
    clusters: list[Cluster],
    landmarks: list[tuple[float, float, float]],
    pose: tuple[float, float, float],
    scan_offset: tuple[float, float] = (-0.032, 0.0),
    range_gate: float = 0.35,
    bearing_gate: float = 0.40,
) -> list[LandmarkFix]:
    if not clusters or not landmarks:
        return []
    px, py, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    preds = []
    for lx, ly, _ in landmarks:
        dx, dy = lx - px, ly - py
        xb, yb = c * dx + s * dy, -s * dx + c * dy
        preds.append((float(np.hypot(xb, yb)), float(np.arctan2(yb, xb))))

    used: set[int] = set()
    fixes: list[LandmarkFix] = []
    for cluster in clusters:
        surface = np.array([cluster.x + scan_offset[0], cluster.y + scan_offset[1]])
        norm = float(np.linalg.norm(surface))
        if norm < 1e-3:
            continue
        best_i, best_score, best = -1, float('inf'), None
        for i, (lx, ly, radius) in enumerate(landmarks):
            if i in used:
                continue
            center = surface + (radius / norm) * surface
            mr, mb = float(np.linalg.norm(center)), float(np.arctan2(center[1], center[0]))
            pr, pb = preds[i]
            score = ((mr - pr) / range_gate) ** 2 + (float(wrap_angle(mb - pb)) / bearing_gate) ** 2
            if score < best_score:
                best_i, best_score, best = i, score, LandmarkFix(mr, mb, (lx, ly))
        if best is not None and best_score <= 2.5:
            used.add(best_i)
            fixes.append(best)
    return fixes


# --- metrics -----------------------------------------------------------------------

def ate_rmse(est_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    est, gt = np.asarray(est_xy, float), np.asarray(gt_xy, float)
    if est.size == 0 or est.shape != gt.shape:
        return float('nan')
    err = est - gt
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def yaw_rmse(est_yaw: np.ndarray, gt_yaw: np.ndarray) -> float:
    est = np.asarray(est_yaw, float).reshape(-1)
    gt = np.asarray(gt_yaw, float).reshape(-1)
    if est.size == 0 or est.shape != gt.shape:
        return float('nan')
    err = wrap_angle(est - gt)
    return float(np.sqrt(np.mean(err * err)))


def associate_nearest(
    est_times: np.ndarray,
    gt_times: np.ndarray,
    gt_values: np.ndarray,
    max_dt: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    est_times = np.asarray(est_times, float).reshape(-1)
    gt_times = np.asarray(gt_times, float).reshape(-1)
    gt_values = np.asarray(gt_values, float)
    if est_times.size == 0 or gt_times.size == 0:
        width = gt_values.shape[1] if gt_values.ndim == 2 else 1
        return np.array([], dtype=int), np.empty((0, width))
    kept_est, kept_gt = [], []
    for i, stamp in enumerate(est_times):
        j = int(np.argmin(np.abs(gt_times - stamp)))
        if abs(gt_times[j] - stamp) <= max_dt:
            kept_est.append(i)
            kept_gt.append(gt_values[j])
    if not kept_gt:
        width = gt_values.shape[1] if gt_values.ndim == 2 else 1
        return np.array([], dtype=int), np.empty((0, width))
    return np.asarray(kept_est, dtype=int), np.vstack(kept_gt)


def score_trajectories(gt_rows, est_rows) -> tuple[float, float, int]:
    if len(gt_rows) < 5 or len(est_rows) < 5:
        return float('nan'), float('nan'), 0
    gt, est = np.asarray(gt_rows, float), np.asarray(est_rows, float)
    idx, matched = associate_nearest(est[:, 0], gt[:, 0], gt[:, 1:4])
    if idx.size < 5:
        return float('nan'), float('nan'), 0
    kept = est[idx]
    return ate_rmse(kept[:, 1:3], matched[:, 0:2]), yaw_rmse(kept[:, 3], matched[:, 2]), int(idx.size)


# --- results dumps (CSV + PNG under results/) --------------------------------------

def project_root() -> 'Path':
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


def finish_timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime('%Y%m%d_%H%M%S')


def results_dirs(root: 'Path | None' = None) -> tuple['Path', 'Path']:
    from pathlib import Path

    base = (root or project_root()) / 'results'
    csv_dir = base / 'csv'
    img_dir = base / 'images'
    csv_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, img_dir


def _write_traj_csv(path: 'Path', rows) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write('t,x,y,yaw\n')
        for row in rows:
            f.write(f'{float(row[0]):.6f},{float(row[1]):.6f},{float(row[2]):.6f},{float(row[3]):.6f}\n')


def dump_run_results(
    gt_rows,
    est_rows: dict,
    nis_rows: dict | None = None,
    plan=None,
    stamp: str | None = None,
) -> str:
    """Write one finish-timestamped CSV set and one PNG per graphic under results/."""
    import csv

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stamp = stamp or finish_timestamp()
    csv_dir, img_dir = results_dirs()
    nis_rows = nis_rows or {}

    ate_vals: list[float] = []
    yaw_vals: list[float] = []
    nis_means: list[float] = []

    metrics_path = csv_dir / f'{stamp}_metrics_summary.csv'
    with open(metrics_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['mode', 'ate_rmse_m', 'yaw_rmse_rad', 'samples', 'nis_mean', 'nis_accepted', 'nis_total']
        )
        for mode in MODES:
            ate, yaw_err, n = score_trajectories(gt_rows, est_rows.get(mode, []))
            nis_list = list(nis_rows.get(mode, []))
            accepted = [float(r[1]) for r in nis_list if len(r) >= 3 and bool(r[2])]
            nis_mean = float(np.mean(accepted)) if accepted else float('nan')
            writer.writerow(
                [
                    mode,
                    f'{ate:.6f}' if not math.isnan(ate) else '',
                    f'{yaw_err:.6f}' if not math.isnan(yaw_err) else '',
                    n,
                    f'{nis_mean:.6f}' if accepted else '',
                    len(accepted),
                    len(nis_list),
                ]
            )
            ate_vals.append(0.0 if math.isnan(ate) else ate)
            yaw_vals.append(0.0 if math.isnan(yaw_err) else yaw_err)
            nis_means.append(nis_mean)

    if gt_rows:
        _write_traj_csv(csv_dir / f'{stamp}_trajectory_gt.csv', gt_rows)
    for mode in MODES:
        rows = est_rows.get(mode) or []
        if rows:
            _write_traj_csv(csv_dir / f'{stamp}_trajectory_{mode}.csv', rows)
        nis_list = list(nis_rows.get(mode, []))
        if nis_list:
            with open(csv_dir / f'{stamp}_nis_{mode}.csv', 'w', encoding='utf-8') as f:
                f.write('t,nis,accepted\n')
                for row in nis_list:
                    f.write(f'{float(row[0]):.6f},{float(row[1]):.6f},{int(bool(row[2]))}\n')

    labels = [MODE_LABELS[m] for m in MODES]
    colors = [MODE_COLORS[m] for m in MODES]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    if plan:
        p = np.asarray(plan, float)
        ax.plot(p[:, 0], p[:, 1], color='#888', ls='--', lw=1, label='Plan')
    if gt_rows:
        g = np.asarray(gt_rows, float)
        ax.plot(g[:, 1], g[:, 2], color='black', lw=2, label='Ground truth')
    for mode in MODES:
        rows = est_rows.get(mode) or []
        if not rows:
            continue
        a = np.asarray(rows, float)
        ax.plot(a[:, 1], a[:, 2], color=MODE_COLORS[mode], lw=1.6, label=MODE_LABELS[mode])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('EKF trajectories')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(img_dir / f'{stamp}_trajectories.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.bar(labels, ate_vals, color=colors)
    ax.set_title('ATE RMSE (m)')
    ax.tick_params(axis='x', labelrotation=15)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / f'{stamp}_ate_rmse.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.bar(labels, yaw_vals, color=colors)
    ax.set_title('Yaw RMSE (rad)')
    ax.tick_params(axis='x', labelrotation=15)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / f'{stamp}_yaw_rmse.png')
    plt.close(fig)

    has_nis = any(not math.isnan(v) for v in nis_means)
    if has_nis:
        plot_vals = [0.0 if math.isnan(v) else v for v in nis_means]
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        ax.bar(labels, plot_vals, color=colors)
        ax.axhline(2.0, color='#444', ls='--', lw=1, label='E[NIS]=2 (2-DoF)')
        ax.set_title('Mean landmark NIS (accepted updates)')
        ax.tick_params(axis='x', labelrotation=15)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        fig.savefig(img_dir / f'{stamp}_nis_mean.png')
        plt.close(fig)

        for mode in MODES:
            nis_list = list(nis_rows.get(mode, []))
            if not nis_list:
                continue
            arr = np.asarray(nis_list, float)
            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
            ax.plot(arr[:, 0] - arr[0, 0], arr[:, 1], color=MODE_COLORS[mode], lw=1.0)
            ax.axhline(2.0, color='#444', ls='--', lw=1)
            ax.set_title(f'NIS over time — {MODE_LABELS[mode]}')
            ax.set_xlabel('t (s, relative)')
            ax.set_ylabel('NIS')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(img_dir / f'{stamp}_nis_timeseries_{mode}.png')
            plt.close(fig)

    return stamp
