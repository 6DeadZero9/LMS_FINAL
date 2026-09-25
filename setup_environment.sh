#!/usr/bin/env bash
# Install ROS 2, Gazebo, TurtleBot3, and the Python libraries this project imports.
# Re-running replaces the marked block in ~/.bashrc.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEGIN_MARK="# >>> SENSOR_ENGINEERING_FINAL >>>"
END_MARK="# <<< SENSOR_ENGINEERING_FINAL <<<"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

# shellcheck disable=SC1091
source /etc/os-release
case "${VERSION_ID}" in
  24.04) ROS_DISTRO="jazzy" ;;
  26.04) ROS_DISTRO="lyrical" ;;
  *)
    echo "Ubuntu ${VERSION_ID} is not supported. Use 24.04 (Jazzy) or 26.04 (Lyrical)." >&2
    exit 1
    ;;
esac

$SUDO apt-get update
$SUDO apt-get install -y curl ca-certificates
$SUDO curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
  | $SUDO tee /etc/apt/sources.list.d/ros2.list > /dev/null
$SUDO apt-get update
$SUDO apt-get install -y \
  "ros-${ROS_DISTRO}-desktop" \
  "ros-${ROS_DISTRO}-ros-gz" \
  "ros-${ROS_DISTRO}-turtlebot3" \
  "ros-${ROS_DISTRO}-turtlebot3-gazebo" \
  python3-colcon-common-extensions \
  python3-numpy \
  python3-matplotlib \
  python3-tk \
  python3-yaml

tmp="$(mktemp)"
if [[ -f "${HOME}/.bashrc" ]]; then
  awk -v begin="${BEGIN_MARK}" -v end="${END_MARK}" '
    $0 == begin { skip = 1; next }
    $0 == end { skip = 0; next }
    skip != 1 { print }
  ' "${HOME}/.bashrc" > "${tmp}"
else
  : > "${tmp}"
fi
cat >> "${tmp}" << EOF
${BEGIN_MARK}
source /opt/ros/${ROS_DISTRO}/setup.bash
if [ -f "${ROOT}/install/setup.bash" ]; then
  source "${ROOT}/install/setup.bash"
fi
export TURTLEBOT3_MODEL=burger
${END_MARK}
EOF
mv "${tmp}" "${HOME}/.bashrc"

echo "Installed ROS 2 ${ROS_DISTRO}, Gazebo (ros_gz), TurtleBot3, numpy, matplotlib, tk, yaml."
echo "Open a new terminal, then from ${ROOT}:"
echo "  colcon build --symlink-install"
echo "  ros2 launch sensor_engineering sensor_course.launch.py"
