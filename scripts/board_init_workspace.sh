#!/usr/bin/env bash
# 실차·대회용: 공식 D-Racer-Kit + 팀 inference 만 colcon 워크스페이스로 구성.
# Gazebo / limo_car 는 보드에서 불필요 — 있으면 링크, 없으면 건너뜀.
#
# D-Racer-Kit: <repo>/external/D-Racer-Kit (없거나 비정상면 clone)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${DRACER_KIT_BRANCH:-release/v1.0.0}"
REPO="https://github.com/topst-development/D-Racer-Kit.git"
VENDOR="${ROOT}/external/D-Racer-Kit"

echo "[SEA-Me board] workspace root: ${ROOT}"
echo "[SEA-Me board] D-Racer-Kit:    ${VENDOR} (branch ${BRANCH})"

if [ -L "${VENDOR}" ]; then
  echo "[SEA-Me board] using symlinked Kit → $(readlink -f "${VENDOR}" 2>/dev/null || readlink "${VENDOR}")"
elif [ ! -d "${VENDOR}/.git" ]; then
  if [ -e "${VENDOR}" ] && [ ! -d "${VENDOR}/.git" ]; then
    echo "[SEA-Me board] Removing incomplete ${VENDOR}..."
    rm -rf "${VENDOR}" 2>/dev/null || {
      echo "[SEA-Me board] ERROR: cannot remove ${VENDOR}. Run: sudo rm -rf ${VENDOR}"
      exit 1
    }
  fi
  echo "[SEA-Me board] Cloning D-Racer-Kit (${BRANCH})..."
  mkdir -p "$(dirname "${VENDOR}")"
  git clone --branch "${BRANCH}" --depth 1 "${REPO}" "${VENDOR}"
else
  echo "[SEA-Me board] D-Racer-Kit already present"
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
    echo "[SEA-Me board] WARNING: missing official package ${pkg}"
    continue
  fi
  link_path "${src}" "${dst}"
  echo "[SEA-Me board] linked ${pkg}"
done

# Optional: keep PC/sim trees happy if vendor limo exists; board does not require it.
LIMO_CAR_SRC="${ROOT}/vendor/limo_car"
if [ -f "${LIMO_CAR_SRC}/package.xml" ]; then
  link_path "${LIMO_CAR_SRC}" "${ROOT}/src/limo_car"
  echo "[SEA-Me board] linked limo_car (optional vendor)"
else
  echo "[SEA-Me board] skip limo_car (not needed on D3-G)"
fi

config_src="${VENDOR}/src/config"
config_dst="${ROOT}/src/config"
team_config="${ROOT}/config/vehicle_config.yaml"
mkdir -p "${config_dst}"
if [ -f "${team_config}" ]; then
  link_path "${team_config}" "${config_dst}/vehicle_config.yaml"
  echo "[SEA-Me board] linked team vehicle_config"
elif [ -d "${config_src}" ]; then
  link_path "${config_src}" "${config_dst}"
  echo "[SEA-Me board] linked Kit default config"
else
  echo "[SEA-Me board] WARNING: no vehicle_config found"
fi

if [ ! -d "${ROOT}/src/inference" ]; then
  echo "[SEA-Me board] ERROR: missing src/inference (team race package)" >&2
  exit 1
fi

echo ""
echo "[SEA-Me board] Init done. Build with:"
echo "  ./scripts/board_race_sync.sh --no-pull"
echo "  # or: colcon build --symlink-install --packages-up-to inference"
