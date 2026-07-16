#!/usr/bin/env bash
# 실차·대회용: 공식 D-Racer-Kit + 팀 inference 만 colcon 워크스페이스로 구성.
# Gazebo / limo_car / vendor 없음.
#
# D-Racer-Kit: <repo>/external/D-Racer-Kit (symlink 또는 clone)
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

# Apply board-only Kit patches (idempotent). See patches/README.md.
PATCH_DIR="${ROOT}/patches"
KIT_GIT=0
if [ -d "${VENDOR}/.git" ] || [ -L "${VENDOR}" ]; then
  KIT_GIT=1
fi
if [ -d "${PATCH_DIR}" ] && [ "${KIT_GIT}" -eq 1 ]; then
  apply_patch() {
    local patch_file="$1"
    local name
    name="$(basename "${patch_file}")"
    if [ ! -f "${patch_file}" ]; then
      return 0
    fi
    if git -C "${VENDOR}" apply --check "${patch_file}" >/dev/null 2>&1; then
      git -C "${VENDOR}" apply "${patch_file}"
      echo "[SEA-Me board] applied patch ${name}"
    elif git -C "${VENDOR}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
      echo "[SEA-Me board] patch already applied: ${name}"
    else
      echo "[SEA-Me board] WARNING: could not apply ${name} (Kit tree mismatch?)"
    fi
  }
  apply_patch "${PATCH_DIR}/camera-native-caps.patch"
  apply_patch "${PATCH_DIR}/camera-v4l2-controls.patch"
  apply_patch "${PATCH_DIR}/control-steer-invert.patch"
  python3 "${PATCH_DIR}/apply_monitor_bev_labels.py" "${VENDOR}" || true
  python3 "${PATCH_DIR}/apply_camera_quiet_logs.py" "${VENDOR}" || true
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

config_dst="${ROOT}/src/config"
team_config="${ROOT}/config/vehicle_config.yaml"
mkdir -p "${config_dst}"
if [ -f "${team_config}" ]; then
  link_path "${team_config}" "${config_dst}/vehicle_config.yaml"
  echo "[SEA-Me board] linked team vehicle_config"
else
  echo "[SEA-Me board] WARNING: missing config/vehicle_config.yaml"
fi

if [ ! -d "${ROOT}/src/inference" ]; then
  echo "[SEA-Me board] ERROR: missing src/inference (team race package)" >&2
  exit 1
fi

echo ""
echo "[SEA-Me board] Init done. Build with:"
echo "  ./scripts/board_race_sync.sh --no-pull"
