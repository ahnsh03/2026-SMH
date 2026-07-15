#!/usr/bin/env bash
# 실차·대회 브랜치용: pull → Kit 링크 → 보드 필수 패키지만 빌드.
#
# Usage:
#   ./scripts/board_race_sync.sh           # pull + init + build
#   ./scripts/board_race_sync.sh --no-pull  # init + build only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DO_PULL=1

for arg in "$@"; do
  case "${arg}" in
    --no-pull) DO_PULL=0 ;;
    -h|--help)
      echo "Usage: $0 [--no-pull]"
      echo "See BOARD.md for the race board workflow."
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
  echo "[SEA-Me board] git pull --ff-only..."
  git pull --ff-only
fi

echo "[SEA-Me board] init workspace..."
"${ROOT}/scripts/board_init_workspace.sh"

if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "[SEA-Me board] ERROR: ROS2 Humble not found at /opt/ros/humble/setup.bash" >&2
  exit 1
fi

set +u
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
set -u

# Board runtime: Kit packages + team inference. Skip dracer_sim / limo build chain.
echo "[SEA-Me board] colcon build (packages-up-to inference)..."
colcon build --symlink-install --packages-up-to inference

set +u
# shellcheck source=/dev/null
source "${ROOT}/install/setup.bash"
set -u

echo ""
echo "[SEA-Me board] Ready. (see BOARD.md)"
echo "  Manual:  ros2 launch inference manual_driving.launch.py"
echo "  Auto:    ros2 launch inference auto_driving.launch.py"
echo "  Route:   ros2 launch inference auto_driving.launch.py route_mode:=in"
