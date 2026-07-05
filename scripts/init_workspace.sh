#!/usr/bin/env bash
# D-Racer-Kit 공식 패키지 + 팀 inference 패키지를 하나의 colcon 워크스페이스로 구성합니다.
#
# D-Racer-Kit 위치 (우선순위):
#   1. <repo>/external/D-Racer-Kit     ← D3-G 보드 단독 clone (권장)
#   2. <repo>/../external/D-Racer-Kit ← PC 상위 프로젝트 (2026-seame-hackathon)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="release/v1.0.0"
REPO="https://github.com/topst-development/D-Racer-Kit.git"

resolve_vendor_dir() {
  local in_repo="${ROOT}/external/D-Racer-Kit"
  local parent="${ROOT}/../external/D-Racer-Kit"

  if [ -d "${in_repo}/.git" ]; then
    echo "${in_repo}"
    return
  fi
  if [ -d "${parent}/.git" ]; then
    echo "${parent}"
    return
  fi
  echo "${in_repo}"
}

VENDOR="$(resolve_vendor_dir)"

echo "[SEA-Me] workspace root: ${ROOT}"
echo "[SEA-Me] D-Racer-Kit:    ${VENDOR}"

if [ ! -d "${VENDOR}/.git" ]; then
  echo "[SEA-Me] Cloning D-Racer-Kit (${BRANCH})..."
  mkdir -p "$(dirname "${VENDOR}")"
  git clone --branch "${BRANCH}" --depth 1 "${REPO}" "${VENDOR}"
else
  echo "[SEA-Me] D-Racer-Kit already present"
fi

OFFICIAL_PKGS=(
  camera control joystick monitor opencv battery topst_utils
  battery_msgs control_msgs joystick_msgs
)

mkdir -p "${ROOT}/src"

link_path() {
  local target="$1"
  local link_name="$2"
  local link_dir
  link_dir="$(dirname "${link_name}")"
  mkdir -p "${link_dir}"
  local rel
  rel="$(realpath --relative-to="${link_dir}" "${target}")"
  ln -sfn "${rel}" "${link_name}"
}

for pkg in "${OFFICIAL_PKGS[@]}"; do
  src="${VENDOR}/src/${pkg}"
  dst="${ROOT}/src/${pkg}"
  if [ ! -d "${src}" ]; then
    echo "[SEA-Me] WARNING: missing official package ${pkg}"
    continue
  fi
  link_path "${src}" "${dst}"
  echo "[SEA-Me] linked ${pkg}"
done

config_src="${VENDOR}/src/config"
config_dst="${ROOT}/src/config"
if [ -d "${config_src}" ]; then
  link_path "${config_src}" "${config_dst}"
  echo "[SEA-Me] linked config"
fi

echo ""
echo "[SEA-Me] Done. Next:"
echo "  source /opt/ros/humble/setup.bash"
echo "  cd ${ROOT} && colcon build --symlink-install"
echo "  source install/setup.bash"
