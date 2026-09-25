#!/usr/bin/env python3
"""Launch Gazebo (burger in world), bridges, EKF fusion, path follow, GUI."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    os.environ.setdefault('TURTLEBOT3_MODEL', 'burger')
    share = Path(get_package_share_directory('sensor_engineering'))
    turtlebot = Path(get_package_share_directory('turtlebot3_gazebo'))
    ros_gz_sim = Path(get_package_share_directory('ros_gz_sim'))
    course_config = share / 'config' / 'course.yaml'
    fusion_config = share / 'config' / 'fusion.yaml'
    world = share / 'worlds' / 'sensor_course.sdf'
    bridge_yaml = turtlebot / 'params' / 'turtlebot3_burger_bridge.yaml'

    gz_gui = LaunchConfiguration('gz_gui').perform(context) == 'true'
    gz_args = f'-r -v2 {world}' if gz_gui else f'-r -s -v2 {world}'
    course_params = {'use_sim_time': True, 'course_config': str(course_config)}
    fusion_params = {
        'use_sim_time': True,
        'course_config': str(course_config),
        'fusion_config': str(fusion_config),
    }

    actions = [
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', str(turtlebot / 'models')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(ros_gz_sim / 'launch' / 'gz_sim.launch.py')),
            launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(turtlebot / 'launch' / 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items(),
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['--ros-args', '-p', f'config_file:={bridge_yaml}'],
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(package='sensor_engineering', executable='ground_truth_node', output='screen', parameters=[course_params]),
        Node(package='sensor_engineering', executable='fusion_node', output='screen', parameters=[fusion_params]),
        Node(package='sensor_engineering', executable='path_follower', output='screen', parameters=[fusion_params]),
    ]
    if LaunchConfiguration('start_gui').perform(context) == 'true':
        actions.append(
            Node(package='sensor_engineering', executable='fusion_dashboard', output='screen', parameters=[fusion_params])
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('gz_gui', default_value='true'),
        DeclareLaunchArgument('start_gui', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
