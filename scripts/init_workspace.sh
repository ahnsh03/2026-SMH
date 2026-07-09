#!/usr/bin/env bash
# D-Racer-Kit 공식 패키지 + 팀 inference 패키지를 하나의 colcon 워크스페이스로 구성합니다.
#
# D-Racer-Kit: <repo>/external/D-Racer-Kit (없으면 자동 clone)
# LIMO 모델:  <repo>/vendor/limo_car (레포에 포함, 추가 clone 불필요)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="release/v1.0.0"
REPO="https://github.com/topst-development/D-Racer-Kit.git"

VENDOR="${ROOT}/external/D-Racer-Kit"

echo "[SEA-Me] workspace root: ${ROOT}"
echo "[SEA-Me] D-Racer-Kit:    ${VENDOR}"

if [ ! -d "${VENDOR}/.git" ]; then
  if [ -e "${VENDOR}" ] && [ ! -d "${VENDOR}/.git" ]; then
    echo "[SEA-Me] Removing incomplete ${VENDOR} (empty dir or broken mount)..."
    rm -rf "${VENDOR}" 2>/dev/null || {
      echo "[SEA-Me] ERROR: cannot remove ${VENDOR} (permission?). Run:"
      echo "  sudo rm -rf ${VENDOR}"
      exit 1
    }
  fi
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

LIMO_CAR_SRC="${ROOT}/vendor/limo_car"
if [ -f "${LIMO_CAR_SRC}/package.xml" ]; then
  link_path "${LIMO_CAR_SRC}" "${ROOT}/src/limo_car"
  echo "[SEA-Me] linked limo_car (vendor)"
else
  echo "[SEA-Me] ERROR: missing vendor/limo_car — re-clone the team repo"
  exit 1
fi

config_src="${VENDOR}/src/config"
config_dst="${ROOT}/src/config"
team_config="${ROOT}/config/vehicle_config.yaml"
mkdir -p "${config_dst}"
if [ -f "${team_config}" ]; then
  link_path "${team_config}" "${config_dst}/vehicle_config.yaml"
  echo "[SEA-Me] linked team vehicle_config (config/vehicle_config.yaml)"
elif [ -d "${config_src}" ]; then
  link_path "${config_src}" "${config_dst}"
  echo "[SEA-Me] linked config (D-Racer-Kit default)"
else
  echo "[SEA-Me] WARNING: no vehicle_config found"
fi

echo ""
echo "[SEA-Me] Done. Next:"
echo "  source /opt/ros/humble/setup.bash"
echo "  cd ${ROOT} && colcon build --symlink-install"
echo "  source install/setup.bash"
