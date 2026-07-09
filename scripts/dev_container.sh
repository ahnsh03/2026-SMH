#!/usr/bin/env bash
# PC(WSL) 개발 컨테이너 — Ubuntu 22.04 + ROS2 Humble (호스트가 24.04/26.04여도 동일).
#
# Usage:
#   ./scripts/dev_container.sh build              # 이미지 빌드 (Gazebo 포함)
#   ./scripts/dev_container.sh shell              # 빌드 전용 셸
#   ./scripts/dev_container.sh build-inference    # inference 빌드
#   ./scripts/dev_container.sh build-sim          # dracer_sim + inference 빌드
#   ./scripts/dev_container.sh sim-shell          # Gazebo GUI용 셸
#   ./scripts/dev_container.sh sim                # Gazebo 자율주행 시뮬 실행
#   ./scripts/dev_container.sh sim-bringup        # Gazebo + 브리지만
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
fi

READONLY_BUILD_SIM='
set -eo pipefail
source /workspace/scripts/sim_gpu_env.sh 2>/dev/null || true
./scripts/init_workspace.sh
if [ -f build/limo_car/CMakeCache.txt ] && grep -q external/limo_ros2 build/limo_car/CMakeCache.txt 2>/dev/null; then
  echo "[SEA-Me] Cleaning stale limo_car build cache (vendor path migration)..."
  rm -rf build/limo_car install/limo_car
fi
if ! python3 -c "import flask" 2>/dev/null; then
  apt-get update -qq && apt-get install -y -qq python3-flask
fi
set +u
source /opt/ros/humble/setup.bash
set -u
colcon build --symlink-install --packages-up-to dracer_sim limo_car inference monitor joystick topst_utils opencv
set +u
source install/setup.bash
set -u
'

run_dev() {
  "${COMPOSE[@]}" run --rm dev "$@"
}

run_sim() {
  "${COMPOSE[@]}" run --rm sim "$@"
}

ensure_sim_display() {
  if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:0
  fi
}

ensure_gazebo() {
  if "${COMPOSE[@]}" run --rm dev bash -lc 'command -v gzserver >/dev/null 2>&1 || command -v gazebo >/dev/null 2>&1'; then
    return 0
  fi
  echo "[SEA-Me] Gazebo가 이미지에 없습니다. 자동 설치를 시작합니다 (~5-10분, 1회)..."
  install_gazebo_in_image
}

install_gazebo_in_image() {
  if docker ps -a --format '{{.Names}}' | grep -qx 'smh-gazebo-install'; then
    if docker exec smh-gazebo-install bash -lc 'command -v gzserver >/dev/null 2>&1 || command -v gazebo >/dev/null 2>&1'; then
      echo "[SEA-Me] 이전 Gazebo 설치 컨테이너에서 설치가 완료된 것을 확인했습니다. 이미지에 반영합니다..."
      docker commit smh-gazebo-install 2026-smh-dev:latest
      docker rm -f smh-gazebo-install
      echo "[SEA-Me] Gazebo 설치 완료 → 2026-smh-dev:latest"
      return 0
    fi
    echo "[SEA-Me] 이전 smh-gazebo-install 컨테이너가 남아 있어 제거 후 재시도합니다..."
    docker rm -f smh-gazebo-install
  fi

  echo "[SEA-Me] Gazebo apt 설치 시작 (~230MB, 5-10분). 네트워크 오류 시 자동 재시도..."
  "${COMPOSE[@]}" run -d --name smh-gazebo-install dev sleep 7200 >/dev/null
  docker exec smh-gazebo-install bash -lc '
    set -e
    success=0
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
      echo "=== gazebo apt attempt ${attempt}/10 ==="
      apt-get update -qq
      if apt-get install -y --no-install-recommends --fix-missing \
          ros-humble-gazebo-ros-pkgs ros-humble-robot-state-publisher gazebo \
          libgl1-mesa-dri libgl1-mesa-glx mesa-utils; then
        success=1
        break
      fi
      echo "Hash mismatch or network error — retry in 30s..."
      sleep 30
    done
    test "${success}" -eq 1
    command -v gzserver >/dev/null || command -v gazebo >/dev/null
    gzserver --version 2>/dev/null | head -1 || gazebo --version 2>/dev/null | head -1
  '
  docker commit smh-gazebo-install 2026-smh-dev:latest
  docker rm -f smh-gazebo-install
  echo "[SEA-Me] Gazebo 설치 완료 → 2026-smh-dev:latest"
}

usage() {
  cat <<'EOF'
Usage: ./scripts/dev_container.sh <command>

Commands:
  build            Docker 이미지 빌드 (Gazebo 제외, apt 재시도 포함)
  install-gazebo   Gazebo + gazebo-ros-pkgs 1회 설치 (Hash mismatch 자동 재시도)
  shell            dev 컨테이너 bash (빌드·코드 작업)
  init             D-Racer-Kit clone + src/ 심볼릭 링크
  build-inference  colcon build --packages-up-to inference
  build-sim        colcon build (dracer_sim + inference + joystick + monitor)
  check            inference.pipeline import 검증 (CI와 동일)
  sim-shell        sim 컨테이너 bash (Gazebo GUI, WSLg/X11)
  sim-bringup      Gazebo + 트랙 + 브리지 (추가 인자 전달 가능)
  check-gpu        OpenGL 렌더러 확인 (CPU/GPU 판정)
  check-rviz       rviz2 설치 여부 확인
  verify-sim       시뮬 토픽·카메라 동작 검증 (sim 실행 중)
  sim              sim-bringup + inference 자율주행
  sim-manual       sim-bringup + 조이스틱 수동주행
  help             이 도움말

WSL 24.04/26.04: 호스트에 ros-humble 설치하지 말고 Docker만 사용하세요.

Examples:
  ./scripts/dev_container.sh build
  ./scripts/dev_container.sh build-sim
  ./scripts/dev_container.sh sim
EOF
}

cmd="${1:-shell}"

case "${cmd}" in
  build)
    # WSL/Docker Desktop에서 BuildKit TLS 오류 시 legacy builder 사용
    DOCKER_BUILDKIT=0 "${COMPOSE[@]}" build
    if ! docker compose run --rm dev bash -lc 'command -v gzserver >/dev/null 2>&1 || command -v gazebo >/dev/null 2>&1' 2>/dev/null; then
      echo ""
      echo "[SEA-Me] 베이스 이미지 빌드 완료. Gazebo가 없으면 다음을 실행하세요:"
      echo "  ./scripts/dev_container.sh install-gazebo   # 1회, Hash mismatch 시 자동 재시도"
    fi
    ;;
  shell)
    run_dev bash
    ;;
  init)
    run_dev bash -lc './scripts/init_workspace.sh'
    ;;
  build-inference)
    run_dev bash -lc '
      set -eo pipefail
      ./scripts/init_workspace.sh
      set +u
      source /opt/ros/humble/setup.bash
      set -u
      colcon build --symlink-install --packages-up-to inference
    '
    ;;
  build-sim)
    run_dev bash -lc "${READONLY_BUILD_SIM}"
    ;;
  install-gazebo)
    install_gazebo_in_image
    ;;
  check)
    run_dev bash -lc '
      set -eo pipefail
      ./scripts/init_workspace.sh
      set +u
      source /opt/ros/humble/setup.bash
      set -u
      colcon build --symlink-install --packages-up-to inference
      set +u
      source install/setup.bash
      set -u
      python3 -c "from inference.pipeline import fuse_control, run_perception; print(\"ok\")"
    '
    ;;
  sim-shell)
    ensure_sim_display
    ensure_gazebo
    run_sim bash -lc '
      set +u
      source /opt/ros/humble/setup.bash
      if [ -f install/setup.bash ]; then source install/setup.bash; fi
      set -u
      exec bash
    '
    ;;
  sim-bringup)
    ensure_sim_display
    ensure_gazebo
    shift || true
    sim_extra_args="$*"
    run_sim bash -lc "${READONLY_BUILD_SIM}
      exec ros2 launch dracer_sim sim_bringup.launch.py ${sim_extra_args}"
    ;;
  sim)
    ensure_sim_display
    ensure_gazebo
    run_sim bash -lc "${READONLY_BUILD_SIM}
      exec ros2 launch dracer_sim sim_auto_driving.launch.py"
    ;;
  sim-manual)
    ensure_sim_display
    ensure_gazebo
    run_sim bash -lc "${READONLY_BUILD_SIM}
      exec ros2 launch dracer_sim sim_manual_driving.launch.py"
    ;;
  check-gpu)
    bash "${ROOT}/scripts/check_sim_gpu.sh"
    ;;
  check-rviz)
    bash "${ROOT}/scripts/check_sim_rviz.sh"
    ;;
  verify-sim)
    bash "${ROOT}/scripts/verify_sim.sh"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    run_dev bash -lc "$*"
    ;;
esac
