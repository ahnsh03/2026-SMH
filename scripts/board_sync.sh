#!/usr/bin/env bash
# D3-G 보드에서 공유 브랜치(board)를 pull한 뒤 빌드까지 한 번에 실행합니다.
#
# Usage:
#   ./scripts/board_sync.sh          # board 브랜치로 checkout + pull + init + build
#   ./scripts/board_sync.sh --no-pull  # init + build only (로컬 변경 테스트)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DO_PULL=1
BOARD_BRANCH="${BOARD_BRANCH:-board}"

for arg in "$@"; do
  case "${arg}" in
    --no-pull) DO_PULL=0 ;;
    -h|--help)
      echo "Usage: $0 [--no-pull]"
      echo "  Syncs and builds the shared D3-G branch '${BOARD_BRANCH}'."
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
  echo "[SEA-Me] fetch + checkout ${BOARD_BRANCH}..."
  git fetch origin "${BOARD_BRANCH}"
  # Shared board branch — create local tracking ref if this clone is new.
  if git show-ref --verify --quiet "refs/heads/${BOARD_BRANCH}"; then
    git checkout "${BOARD_BRANCH}"
  else
    git checkout -B "${BOARD_BRANCH}" "origin/${BOARD_BRANCH}"
  fi
  echo "[SEA-Me] git pull --ff-only origin ${BOARD_BRANCH}..."
  git pull --ff-only origin "${BOARD_BRANCH}"
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
echo "[SEA-Me] Ready (branch: ${BOARD_BRANCH})."
echo "  Manual:  ros2 launch inference manual_driving.launch.py"
echo "  Auto:    ros2 launch inference auto_driving.launch.py"
