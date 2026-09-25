# SENSOR_ENGINEERING_FINAL

Final project for robot_dreams course on sensor engineering. The project is implemented with ROS stack that includes: ROS2/Gazeebo/Python3. ROS was chosen due to the familiarity and longing to learn in further.

## Architecture

One ROS package (`sensor_engineering`): Gazebo world with TurtleBot3 burger, ground-truth path, **EKF** under four sensor suites, pure-pursuit square drive, and a small dashboard.

| Suite | Sensors |
|-------|---------|
| `imu` | IMU prediction only |
| `imu_odom` | IMU + wheel speed |
| `imu_lidar` | IMU + pole range/bearing |
| `imu_odom_lidar` | all three |

State `[x, y, yaw, v, gyro_bias]`. Pure pursuit drives one 2 m square from Gazebo pose and stops. Filters only estimate and score.

Code: `ekf.py` (filter), `tooling.py` (angles, suites, path, pursuit, LiDAR, metrics), plus the four ROS node modules.

## Setup and launch

```bash
./setup_environment.sh
```

```bash
export TURTLEBOT3_MODEL=burger
colcon build --symlink-install
source install/setup.bash
ros2 launch sensor_engineering sensor_course.launch.py
```

`gz_gui:=false` / `start_gui:=false` hide Gazebo / the dashboard.
