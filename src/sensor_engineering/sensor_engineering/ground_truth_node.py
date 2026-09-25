"""Publish the planned rectangle and Gazebo's pose of the burger."""

from __future__ import annotations

import yaml
from geometry_msgs.msg import PoseStamped
from gz.msgs.pose_v_pb2 import Pose_V
from gz.transport import Node as GzNode
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from sensor_engineering.tooling import rectangle_loop, yaw_to_quaternion


class GroundTruthNode(Node):
    def __init__(self) -> None:
        super().__init__('ground_truth_node')
        self.declare_parameter('course_config', '')
        course_path = self.get_parameter('course_config').get_parameter_value().string_value
        with open(course_path, 'r', encoding='utf-8') as handle:
            course = yaml.safe_load(handle)

        self.robot_name = str(course['robot_name'])
        world_name = str(course['world_name'])
        path_cfg = course['path']
        self.path = rectangle_loop(
            float(path_cfg['x_min']),
            float(path_cfg['x_max']),
            float(path_cfg['y_min']),
            float(path_cfg['y_max']),
            float(path_cfg['spacing']),
        )
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.path_pub = self.create_publisher(Path, '/ground_truth/path', latched)
        self.odom_pub = self.create_publisher(Odometry, '/ground_truth/odom', 10)
        self.traveled_pub = self.create_publisher(Path, '/ground_truth/traveled', latched)
        self._announced = False
        self._traveled = Path()
        self._traveled.header.frame_id = 'map'
        self._last_xy: tuple[float, float] | None = None
        self._gz = GzNode()
        topic = f'/world/{world_name}/dynamic_pose/info'
        if not self._gz.subscribe(Pose_V, topic, self._on_pose):
            raise RuntimeError(f'Could not subscribe to {topic}')
        self.create_timer(1.0, self._publish_path)
        self._publish_path()
        self.get_logger().info(f'Waiting for Gazebo pose of {self.robot_name} on {topic}')

    def _publish_path(self) -> None:
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'map'
        for x, y, yaw in self.path:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _on_pose(self, message: Pose_V) -> None:
        if not self._announced:
            names = ', '.join(pose.name for pose in message.pose) or '(empty)'
            self.get_logger().info(f'Gazebo entities: {names}')
            self._announced = True
        selected = next((pose for pose in message.pose if pose.name == self.robot_name), None)
        if selected is None:
            return
        position = selected.position
        orientation = selected.orientation
        stamp = self.get_clock().now().to_msg()
        header = message.header.stamp
        if header.sec or header.nsec:
            stamp.sec = int(header.sec)
            stamp.nanosec = int(header.nsec)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = float(position.x)
        odom.pose.pose.position.y = float(position.y)
        odom.pose.pose.position.z = float(position.z)
        odom.pose.pose.orientation.x = float(orientation.x)
        odom.pose.pose.orientation.y = float(orientation.y)
        odom.pose.pose.orientation.z = float(orientation.z)
        odom.pose.pose.orientation.w = float(orientation.w)
        self.odom_pub.publish(odom)

        xy = (float(position.x), float(position.y))
        if self._last_xy is None or (xy[0] - self._last_xy[0]) ** 2 + (xy[1] - self._last_xy[1]) ** 2 > 0.0025:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = 'map'
            pose.pose = odom.pose.pose
            self._traveled.header.stamp = stamp
            self._traveled.poses.append(pose)
            if len(self._traveled.poses) > 4000:
                self._traveled.poses = self._traveled.poses[-4000:]
            self.traveled_pub.publish(self._traveled)
            self._last_xy = xy


def main() -> None:
    import rclpy

    rclpy.init()
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
