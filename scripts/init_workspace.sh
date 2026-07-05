#!/usr/bin/env bash
# D-Racer-Kit 공식 패키지 + 팀 inference 패키지를 하나의 colcon 워크스페이스로 구성합니다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
VENDOR="${PROJECT_ROOT}/external/D-Racer-Kit"
BRANCH="release/v1.0.0"
REPO="https://github.com/topst-development/D-Racer-Kit.git"

echo "[SEA-Me] workspace root: ${ROOT}"
echo "[SEA-Me] project root: ${PROJECT_ROOT}"
echo "[SEA-Me] vendor: ${VENDOR}"

if [ ! -d "${VENDOR}/.git" ]; then
  echo "[SEA-Me] Cloning D-Racer-Kit (${BRANCH})..."
  mkdir -p "${PROJECT_ROOT}/external"
  git clone --branch "${BRANCH}" --depth 1 "${REPO}" "${VENDOR}"
else
  echo "[SEA-Me] D-Racer-Kit already present at ${VENDOR}"
fi

OFFICIAL_PKGS=(
  camera control joystick monitor opencv battery topst_utils
  battery_msgs control_msgs joystick_msgs
)

mkdir -p "${ROOT}/src"

for pkg in "${OFFICIAL_PKGS[@]}"; do
  src="${VENDOR}/src/${pkg}"
  dst="${ROOT}/src/${pkg}"
  if [ ! -d "${src}" ]; then
    echo "[SEA-Me] WARNING: missing official package ${pkg}"
    continue
  fi
  if [ -e "${dst}" ]; then
    echo "[SEA-Me] skip ${pkg} (already exists)"
  else
    ln -sfn "../../external/D-Racer-Kit/src/${pkg}" "${dst}"
    echo "[SEA-Me] linked ${pkg}"
  fi
done

# vehicle config
if [ -f "${VENDOR}/src/config/vehicle_config.yaml" ] && [ ! -e "${ROOT}/src/config" ]; then
  mkdir -p "${ROOT}/src"
  ln -sfn "../../external/D-Racer-Kit/src/config" "${ROOT}/src/config"
  echo "[SEA-Me] linked config"
fi

echo ""
echo "[SEA-Me] Done. Next steps:"
echo "  source /opt/ros/humble/setup.bash"
echo "  cd ${ROOT} && colcon build --symlink-install"
echo "  source install/setup.bash"
