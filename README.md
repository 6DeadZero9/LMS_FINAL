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

## Results dumps

When the robot finishes one square lap (or when `fusion_node` shuts down), metrics and plots are written under:

- `results/csv/` — ATE/yaw summary, per-suite trajectories, per-suite NIS time series
- `results/images/` — trajectories, ATE bar, yaw bar, mean NIS, NIS time series

Every file is prefixed with the finish timestamp (`YYYYMMDD_HHMMSS_…`). LiDAR measurement noise is `r_range=0.01` m / `r_bearing=0.02` rad so landmark NIS should sit near 2 for a consistent filter. Uses `np.arctan2` (NumPy 1.26 / Ubuntu 24.04).
