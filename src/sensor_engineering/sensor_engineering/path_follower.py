"""Pure pursuit along the planned square using Gazebo ground-truth pose."""

from __future__ import annotations

import yaml
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool

from sensor_engineering.tooling import quaternion_to_yaw, pure_pursuit


class PathFollower(Node):
    def __init__(self) -> None:
        super().__init__('path_follower')
        self.declare_parameter('fusion_config', '')
        self.declare_parameter('course_config', '')
        fusion_path = self.get_parameter('fusion_config').get_parameter_value().string_value
        with open(fusion_path, 'r', encoding='utf-8') as handle:
            fusion = yaml.safe_load(handle)
        nav = fusion['navigation']
        self.speed = float(nav['linear_speed'])
        self.lookahead = float(nav['lookahead'])
        self.max_yaw_rate = float(nav['max_yaw_rate'])
        rate = float(nav['command_rate_hz'])

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.path_xy = None
        self.pose = None
        self.progress = 0
        self.done = False
        self._last_log = 0.0
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.done_pub = self.create_publisher(Bool, '/navigation/lap_done', latched)
        self.create_subscription(Path, '/ground_truth/path', self._on_path, latched)
        self.create_subscription(Odometry, '/ground_truth/odom', self._on_pose, qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self._command, clock=Clock(clock_type=ClockType.SYSTEM_TIME))
        self.get_logger().info('Path follower using /ground_truth/odom; one lap then stop')

    def _on_path(self, message: Path) -> None:
        if not message.poses:
            return
        new_path = [
            (float(pose.pose.position.x), float(pose.pose.position.y)) for pose in message.poses
        ]
        # Ground truth republishes the same path every second. Do not reset progress.
        if self.path_xy is not None and len(self.path_xy) == len(new_path):
            self.path_xy = new_path
            return
        self.path_xy = new_path
        self.progress = 0
        self.done = False

    def _on_pose(self, message: Odometry) -> None:
        q = message.pose.pose.orientation
        self.pose = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            quaternion_to_yaw(q.x, q.y, q.z, q.w),
        )
        self._command()

    def _command(self) -> None:
        import numpy as np

        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        if self.done:
            self.cmd_pub.publish(message)
            return
        if self.path_xy is None or self.pose is None:
            self.cmd_pub.publish(message)
            now = time_wall()
            if now - self._last_log > 2.0:
                self._last_log = now
                missing = []
                if self.path_xy is None:
                    missing.append('path')
                if self.pose is None:
                    missing.append('ground_truth pose')
                self.get_logger().warning('Holding robot: waiting for ' + ' and '.join(missing))
            return
        x, y, yaw = self.pose
        linear, angular, self.progress, finished = pure_pursuit(
            x,
            y,
            yaw,
            np.asarray(self.path_xy, dtype=float),
            self.lookahead,
            self.speed,
            self.max_yaw_rate,
            self.progress,
        )
        if finished:
            self.done = True
            linear, angular = 0.0, 0.0
            self.get_logger().info('Finished one square lap; stopping')
            done_msg = Bool()
            done_msg.data = True
            self.done_pub.publish(done_msg)
        message.twist.linear.x = linear
        message.twist.angular.z = angular
        self.cmd_pub.publish(message)


def time_wall() -> float:
    import time

    return time.monotonic()


def main() -> None:
    import rclpy

    rclpy.init()
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
