#!/usr/bin/env bash
# D3-G 보드에서 git pull 후 빌드까지 한 번에 실행합니다.
#
# Usage:
#   ./scripts/board_sync.sh          # pull + init + build
#   ./scripts/board_sync.sh --no-pull  # init + build only (로컬 변경 테스트)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DO_PULL=1

for arg in "$@"; do
  case "${arg}" in
    --no-pull) DO_PULL=0 ;;
    -h|--help)
      echo "Usage: $0 [--no-pull]"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

cd "${ROOT}"

if [ "${DO_PULL}" -eq 1 ]; then
  echo "[SEA-Me] git pull..."
  git pull --ff-only
fi

echo "[SEA-Me] init workspace..."
"${ROOT}/scripts/init_workspace.sh"

if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "[SEA-Me] ERROR: ROS2 Humble not found at /opt/ros/humble/setup.bash" >&2
  exit 1
fi

# ROS setup scripts reference unset vars; incompatible with set -u.
set +u
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
set -u

echo "[SEA-Me] colcon build..."
colcon build --symlink-install

set +u
# shellcheck source=/dev/null
source "${ROOT}/install/setup.bash"
set -u

echo ""
echo "[SEA-Me] Ready."
echo "  Manual:  ros2 launch inference manual_driving.launch.py"
echo "  Auto:    ros2 launch inference auto_driving.launch.py"
