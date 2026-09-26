"""Pick an EKF sensor suite and plot ATE against Gazebo."""

from __future__ import annotations

import json
import threading
from collections import deque

import yaml
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String

from sensor_engineering.tooling import (
    MODE_COLORS,
    MODE_LABELS,
    MODES,
    dump_run_results,
    finish_timestamp,
    mean_distance_to_polyline,
    odom_sample,
    parse_mode,
    score_trajectories,
)


class FusionDashboard(Node):
    def __init__(self) -> None:
        super().__init__('fusion_dashboard')
        self.declare_parameter('fusion_config', '')
        fusion_path = self.get_parameter('fusion_config').get_parameter_value().string_value
        with open(fusion_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.mode = parse_mode(str(cfg['initial_selection'])) or 'imu_odom_lidar'
        self._lock = threading.Lock()
        self.gt: deque = deque(maxlen=4000)
        self.est = {m: deque(maxlen=4000) for m in MODES}
        self.plan = None
        self.metrics = {}
        self._dumped = False
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.selection_pub = self.create_publisher(String, '/fusion/selection', latched)
        self.create_subscription(Odometry, '/ground_truth/odom', self._on_gt, qos_profile_sensor_data)
        self.create_subscription(Path, '/ground_truth/path', self._on_path, latched)
        self.create_subscription(String, '/fusion/metrics', self._on_metrics, 10)
        for mode in MODES:
            self.create_subscription(Odometry, f'/fusion/{mode}/odom', lambda msg, m=mode: self._on_est(m, msg), 10)
        self._publish_selection()

    def _publish_selection(self) -> None:
        msg = String()
        msg.data = self.mode
        self.selection_pub.publish(msg)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._publish_selection()

    def _on_gt(self, msg: Odometry) -> None:
        with self._lock:
            self.gt.append(odom_sample(msg))

    def _on_est(self, mode: str, msg: Odometry) -> None:
        with self._lock:
            self.est[mode].append(odom_sample(msg))

    def _on_path(self, msg: Path) -> None:
        with self._lock:
            self.plan = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]

    def _on_metrics(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self.metrics = payload.get('modes', {})

    def snapshot(self):
        with self._lock:
            return (
                self.mode,
                list(self.gt),
                {m: list(v) for m, v in self.est.items()},
                None if self.plan is None else list(self.plan),
                dict(self.metrics),
            )

    def dump_results(self, reason: str = 'manual') -> str | None:
        if self._dumped:
            return None
        mode, gt_rows, est_rows, plan, _metrics = self.snapshot()
        del mode
        if len(gt_rows) < 5:
            return None
        stamp = finish_timestamp()
        try:
            dump_run_results(gt_rows, est_rows, nis_rows=None, plan=plan, stamp=stamp)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Dashboard results dump failed ({reason}): {exc}')
            return None
        self._dumped = True
        self.get_logger().info(f'Dashboard plots/CSV written under results/ (stamp={stamp}, reason={reason})')
        return stamp


def run_dashboard(node: FusionDashboard) -> None:
    import os
    import sys

    import numpy as np

    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        print('fusion_dashboard needs DISPLAY. Use start_gui:=false for headless.', file=sys.stderr)
        sys.exit(1)

    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title('SENSOR_ENGINEERING_FINAL — EKF suites')
    root.geometry('1100x720')

    controls = ttk.Frame(root, padding=8)
    controls.pack(side=tk.TOP, fill=tk.X)
    mode_var = tk.StringVar(value=node.mode)
    box = ttk.LabelFrame(controls, text='EKF sensor suite', padding=6)
    box.pack(side=tk.LEFT, padx=6)
    for name in MODES:
        ttk.Radiobutton(box, text=MODE_LABELS[name], value=name, variable=mode_var).pack(anchor=tk.W)
    status = ttk.Label(controls, text='Waiting for Gazebo…', wraplength=420, justify=tk.LEFT)
    status.pack(side=tk.LEFT, padx=12)

    fig = Figure(figsize=(10, 6), dpi=100)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])
    ax_traj, ax_ate, ax_yaw = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    mode_var.trace_add('write', lambda *_: node.set_mode(mode_var.get()))

    def on_close() -> None:
        node.dump_results('window_close')
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)

    def refresh() -> None:
        mode, gt_rows, est_rows, plan, metrics = node.snapshot()
        del mode
        ax_traj.clear()
        ax_ate.clear()
        ax_yaw.clear()
        labels = [MODE_LABELS[m] for m in MODES]
        colors = [MODE_COLORS[m] for m in MODES]
        ate_vals, yaw_vals = [], []
        if plan:
            p = np.asarray(plan, float)
            ax_traj.plot(p[:, 0], p[:, 1], color='#888', ls='--', lw=1, label='Plan')
        if gt_rows:
            g = np.asarray(gt_rows, float)
            ax_traj.plot(g[:, 1], g[:, 2], color='black', lw=2, label='Ground truth')
        for m in MODES:
            rows = est_rows[m]
            ate, yaw_err, _ = score_trajectories(gt_rows, rows)
            ate_vals.append(0.0 if np.isnan(ate) else ate)
            yaw_vals.append(0.0 if np.isnan(yaw_err) else yaw_err)
            if rows:
                a = np.asarray(rows, float)
                active = m == mode_var.get()
                ax_traj.plot(
                    a[:, 1],
                    a[:, 2],
                    color=MODE_COLORS[m],
                    lw=2.4 if active else 1.1,
                    alpha=1 if active else 0.75,
                    label=MODE_LABELS[m],
                )
        ax_traj.set_aspect('equal', adjustable='box')
        if plan:
            pad = 0.75
            ax_traj.set_xlim(float(p[:, 0].min()) - pad, float(p[:, 0].max()) + pad)
            ax_traj.set_ylim(float(p[:, 1].min()) - pad, float(p[:, 1].max()) + pad)
        ax_traj.set_title('EKF trajectories (thick = selected suite)')
        ax_traj.set_xlabel('x (m)')
        ax_traj.set_ylabel('y (m)')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend(loc='upper right', fontsize=8)
        ax_ate.bar(labels, ate_vals, color=colors)
        ax_ate.set_title('ATE RMSE (m)')
        ax_ate.tick_params(axis='x', labelrotation=15)
        ax_ate.grid(True, axis='y', alpha=0.3)
        ax_yaw.bar(labels, yaw_vals, color=colors)
        ax_yaw.set_title('Yaw RMSE (rad)')
        ax_yaw.tick_params(axis='x', labelrotation=15)
        ax_yaw.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        canvas.draw_idle()

        ate, yaw_err, n = score_trajectories(gt_rows, est_rows[mode_var.get()])
        path_err = float('nan')
        if plan and gt_rows:
            path_err = mean_distance_to_polyline(np.asarray(gt_rows, float)[-200:, 1:3], np.asarray(plan, float))
        nis_mean = metrics.get(mode_var.get(), {}).get('nis_mean')
        nis_txt = f', NIS {nis_mean:.2f}' if isinstance(nis_mean, (int, float)) else ''
        status.configure(
            text=(
                f'EKF / {MODE_LABELS[mode_var.get()]}\n'
                f'ATE {ate:.3f} m, yaw {yaw_err:.3f} rad ({n}){nis_txt}\n'
                f'Path error {path_err:.3f} m'
            )
        )
        root.after(250, refresh)

    root.after(250, refresh)
    root.mainloop()


def main() -> None:
    import rclpy

    rclpy.init()
    node = FusionDashboard()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        run_dashboard(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.dump_results('shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
