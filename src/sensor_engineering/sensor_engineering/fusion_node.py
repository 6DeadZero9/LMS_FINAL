"""Run one EKF per sensor suite and publish the selected pose."""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

from sensor_engineering.ekf import ExtendedKalmanFilter, process_noise
from sensor_engineering.tooling import (
    MODES,
    associate_nearest,
    ate_rmse,
    body_forward_accel,
    clusters_from_ranges,
    make_odom,
    match_clusters,
    mode_uses_lidar,
    mode_uses_odom,
    odom_sample,
    parse_mode,
    stamp_to_sec,
    yaw_rmse,
)


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__('fusion_node')
        self.declare_parameter('course_config', '')
        self.declare_parameter('fusion_config', '')
        course_path = self.get_parameter('course_config').get_parameter_value().string_value
        fusion_path = self.get_parameter('fusion_config').get_parameter_value().string_value
        with open(course_path, encoding='utf-8') as f:
            course = yaml.safe_load(f)
        with open(fusion_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        noise, ekf = cfg['noise'], cfg['ekf']
        self.rng = np.random.default_rng(int(noise['seed']))
        self.accel_std = float(noise['accel_std'])
        self.gyro_std = float(noise['gyro_std'])
        self.gyro_bias = float(noise['gyro_bias'])
        self.odom_v_std = float(noise['odom_v_std'])
        self.sig_ax = float(ekf['sig_ax'])
        self.sig_wz = float(ekf['sig_wz'])
        self.sig_bg = float(ekf['sig_bg'])
        self.r_v = float(ekf['r_odom_v'])
        self.r_range = float(ekf['r_range'])
        self.r_bearing = float(ekf['r_bearing'])
        self.lidar_gate = float(ekf['lidar_gate'])
        off = course['scan_frame_offset']
        self.scan_offset = (float(off['x']), float(off['y']))
        self.landmarks = [(float(p['x']), float(p['y']), float(p['radius'])) for p in course['landmarks']]
        spawn = course['spawn']
        self._spawn = (float(spawn['x']), float(spawn['y']), float(spawn['yaw']))

        self.mode = parse_mode(str(cfg['initial_selection'])) or 'imu_odom_lidar'
        self.filters: dict[str, ExtendedKalmanFilter] = {}
        self.ready = False
        self.last_imu_time: float | None = None
        self.last_publish = 0.0
        self.gt_history: deque = deque(maxlen=8000)
        self.est_history = {m: deque(maxlen=8000) for m in MODES}

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pubs = {m: self.create_publisher(Odometry, f'/fusion/{m}/odom', 10) for m in MODES}
        self.active_pub = self.create_publisher(Odometry, '/fusion/active/odom', 10)
        self.metrics_pub = self.create_publisher(String, '/fusion/metrics', 10)
        self.create_subscription(String, '/fusion/selection', self._on_selection, latched)
        self.create_subscription(Odometry, '/ground_truth/odom', self._on_gt, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self.create_timer(0.5, self._publish_metrics)
        self.get_logger().info(f'EKF active suite: {self.mode}')

    def _on_selection(self, msg: String) -> None:
        mode = parse_mode(msg.data)
        if mode and mode != self.mode:
            self.mode = mode
            self.get_logger().info(f'Active suite: {mode}')

    def _ensure(self, x: float, y: float, yaw: float) -> None:
        if self.ready:
            return
        self.filters = {m: ExtendedKalmanFilter(x, y, yaw) for m in MODES}
        self.ready = True

    def _on_gt(self, msg: Odometry) -> None:
        sample = odom_sample(msg)
        self._ensure(sample[1], sample[2], sample[3])
        self.gt_history.append(sample)

    def _on_imu(self, msg: Imu) -> None:
        if not self.ready:
            self._ensure(*self._spawn)
        stamp = stamp_to_sec(msg.header.stamp)
        q = msg.orientation
        ax = body_forward_accel(
            float(msg.linear_acceleration.x), float(q.x), float(q.y), float(q.z), float(q.w)
        ) + float(self.rng.normal(0.0, self.accel_std))
        wz = float(msg.angular_velocity.z) + self.gyro_bias + float(self.rng.normal(0.0, self.gyro_std))
        if self.last_imu_time is None:
            self.last_imu_time = stamp
            return
        dt = stamp - self.last_imu_time
        self.last_imu_time = stamp
        if dt <= 0.0 or dt > 0.2:
            return
        qn = process_noise(dt, self.sig_ax, self.sig_wz, self.sig_bg)
        u = np.array([ax, wz])
        for filt in self.filters.values():
            filt.predict(u, dt, qn)
        if stamp - self.last_publish >= 0.05:
            self.last_publish = stamp
            self._publish(msg.header.stamp)

    def _on_odom(self, msg: Odometry) -> None:
        if not self.ready:
            return
        v = float(msg.twist.twist.linear.x) + float(self.rng.normal(0.0, self.odom_v_std))
        for mode, filt in self.filters.items():
            if mode_uses_odom(mode):
                filt.update_speed(v, self.r_v)

    def _on_scan(self, msg: LaserScan) -> None:
        if not self.ready:
            return
        clusters = clusters_from_ranges(
            np.asarray(msg.ranges, dtype=float),
            float(msg.angle_min),
            float(msg.angle_increment),
            range_min=max(0.12, float(msg.range_min)),
            range_max=min(3.4, float(msg.range_max)),
        )
        for mode, filt in self.filters.items():
            if not mode_uses_lidar(mode):
                continue
            for fix in match_clusters(clusters, self.landmarks, filt.pose()[:3], self.scan_offset):
                filt.update_landmark(
                    fix.range_m, fix.bearing_rad, fix.map_xy, self.r_range, self.r_bearing, self.lidar_gate
                )

    def _publish(self, stamp) -> None:
        t = stamp_to_sec(stamp)
        for mode, filt in self.filters.items():
            x, y, yaw, speed = filt.pose()
            self.est_history[mode].append((t, x, y, yaw))
            msg = make_odom(stamp, x, y, yaw, speed)
            self.pubs[mode].publish(msg)
            if mode == self.mode:
                self.active_pub.publish(msg)

    def _publish_metrics(self) -> None:
        if len(self.gt_history) < 5:
            return
        gt = np.asarray(self.gt_history, dtype=float)
        payload = {'selection': self.mode, 'modes': {}}
        for mode in MODES:
            samples = np.asarray(self.est_history[mode], dtype=float)
            if samples.shape[0] < 5:
                continue
            idx, matched = associate_nearest(samples[:, 0], gt[:, 0], gt[:, 1:4])
            if idx.size < 5:
                continue
            est = samples[idx]
            payload['modes'][mode] = {
                'ate_rmse': ate_rmse(est[:, 1:3], matched[:, 0:2]),
                'yaw_rmse': yaw_rmse(est[:, 3], matched[:, 2]),
                'samples': int(idx.size),
            }
        out = String()
        out.data = json.dumps(payload)
        self.metrics_pub.publish(out)


def main() -> None:
    import rclpy

    rclpy.init()
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
