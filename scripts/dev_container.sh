#!/usr/bin/env bash
# PC(WSL) 개발 컨테이너 — Ubuntu 22.04 + ROS2 Humble
#
# 시뮬 개발 (컨테이너 1개 + 터미널 2개):
#   ./scripts/dev_container.sh sim-up          # 2026-smh-sim 생성·시작 (1회)
#   ./scripts/dev_container.sh sim-bringup     # 터미널1: Gazebo+브리지 (Ctrl+C → launch만 종료)
#   docker exec -it 2026-smh-sim bash          # 터미널2: 코드 빌드·inference 실행
#   ./scripts/dev_container.sh sim-down        # 컨테이너 제거
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SIM_CONTAINER_NAME="${SMH_SIM_CONTAINER:-2026-smh-sim}"

setup_compose() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[SEA-Me] docker 명령을 찾을 수 없습니다.

WSL2 + Docker Desktop 사용 시:
  1. Windows에서 Docker Desktop 실행
  2. Settings → Resources → WSL Integration
  3. 사용 중인 배포판(예: Ubuntu) 토글 ON → Apply & Restart
  4. WSL 터미널을 닫았다가 다시 열기

확인:
  docker --version
  docker compose version
  docker ps
EOF
    exit 1
  fi

  if ! docker info >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[SEA-Me] Docker 데몬에 연결할 수 없습니다.

  - Docker Desktop이 Windows에서 실행 중인지 확인
  - WSL Integration이 이 배포판에 켜져 있는지 확인
  - 터미널 재시작 후: docker ps
EOF
    exit 1
  fi

  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
    return 0
  fi

  if command -v docker-compose >/dev/null 2>&1 && docker-compose version >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
    return 0
  fi

  cat >&2 <<'EOF'
[SEA-Me] docker compose / docker-compose를 사용할 수 없습니다.

Docker Desktop을 최신으로 업데이트하거나 WSL Integration을 다시 켜 주세요.
확인: docker compose version
EOF
  exit 1
}

setup_compose

READONLY_BUILD_SIM='
set -eo pipefail
source /workspace/scripts/sim_gpu_env.sh 2>/dev/null || true
./scripts/init_workspace.sh
if [ -f build/limo_car/CMakeCache.txt ] && grep -q external/limo_ros2 build/limo_car/CMakeCache.txt 2>/dev/null; then
  echo "[SEA-Me] Cleaning stale limo_car build cache (vendor path migration)..."
  rm -rf build/limo_car install/limo_car
fi
if [ -d build/dracer_sim/launch/__pycache__ ] || find build/dracer_sim -path "*/__pycache__" -print -quit 2>/dev/null | grep -q .; then
  echo "[SEA-Me] Cleaning stale dracer_sim build cache (__pycache__)..."
  rm -rf build/dracer_sim install/dracer_sim
fi
if ! python3 -c "import flask" 2>/dev/null; then
  apt-get update -qq && apt-get install -y -qq python3-flask
fi
set +u
source /opt/ros/humble/setup.bash
set -u
python3 scripts/prepare_mission_signs.py
python3 scripts/prepare_bev_calib_mat.py
colcon build --symlink-install --packages-up-to dracer_sim limo_car inference monitor joystick topst_utils opencv
set +u
source install/setup.bash
set -u
'

SIM_EXEC_PREAMBLE='
set +u
source /workspace/scripts/sim_gpu_env.sh 2>/dev/null || true
source /opt/ros/humble/setup.bash
if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi
'

# bash -lc "…${READONLY_BUILD_SIM}…" 는 스크립트 안의 set 이 $s 로 잘려 et 가 됨 → stdin 으로 전달

run_dev() {
  "${COMPOSE[@]}" run --rm dev "$@"
}

ensure_sim_display() {
  if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:0
  fi
}

sim_container_running() {
  docker ps --format '{{.Names}}' | grep -qx "${SIM_CONTAINER_NAME}"
}

sim_container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "${SIM_CONTAINER_NAME}"
}

sim_up() {
  local quiet="${1:-}"
  ensure_sim_display
  ensure_gazebo
  if sim_container_running; then
    echo "[SEA-Me] ${SIM_CONTAINER_NAME} already running"
    return 0
  fi
  if sim_container_exists; then
    docker start "${SIM_CONTAINER_NAME}" >/dev/null
    echo "[SEA-Me] Started ${SIM_CONTAINER_NAME}"
    return 0
  fi
  "${COMPOSE[@]}" run -d --name "${SIM_CONTAINER_NAME}" sim sleep infinity >/dev/null
  echo "[SEA-Me] Created ${SIM_CONTAINER_NAME}"
  if [ -z "${quiet}" ]; then
    cat <<EOF

다음 단계:
  터미널1  ./scripts/dev_container.sh sim-bringup

  터미널2  docker exec -it ${SIM_CONTAINER_NAME} bash
           source /opt/ros/humble/setup.bash && source install/setup.bash

  종료     ./scripts/dev_container.sh sim-down
EOF
  fi
}

sim_down() {
  if sim_container_exists; then
    docker rm -f "${SIM_CONTAINER_NAME}" >/dev/null
    echo "[SEA-Me] Removed ${SIM_CONTAINER_NAME}"
  else
    echo "[SEA-Me] ${SIM_CONTAINER_NAME} not found"
  fi
}

sim_exec_launch() {
  local launch_pkg="$1"
  local launch_file="$2"
  shift 2
  local remote="/tmp/smh-launch-$$.sh"
  sim_up quiet
  {
    printf '%s\n' "$READONLY_BUILD_SIM"
    printf '%s\n' "$SIM_EXEC_PREAMBLE"
    printf 'exec ros2 launch %q %q' "$launch_pkg" "$launch_file"
    if [ "$#" -gt 0 ]; then
      printf ' %q' "$@"
    fi
    printf '\n'
  } | docker exec -i "${SIM_CONTAINER_NAME}" tee "${remote}" > /dev/null
  # stdin 파이프와 -t 는 함께 쓸 수 없음 → 파일로 쓴 뒤 -it 로 실행
  if [ -t 0 ]; then
    docker exec -it "${SIM_CONTAINER_NAME}" bash "${remote}"
  else
    docker exec -i "${SIM_CONTAINER_NAME}" bash "${remote}"
  fi
  docker exec -i "${SIM_CONTAINER_NAME}" rm -f "${remote}" 2>/dev/null || true
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

run_in_sim_or_dev() {
  local script="$1"
  if sim_container_running; then
    printf '%s\n' "$script" | docker exec -i "${SIM_CONTAINER_NAME}" bash -s
  else
    printf '%s\n' "$script" | "${COMPOSE[@]}" run -T --rm dev bash -s
  fi
}

run_build_sim() {
  run_in_sim_or_dev "${READONLY_BUILD_SIM}"
}

usage() {
  cat <<EOF
Usage: ./scripts/dev_container.sh <command>

시뮬 개발 (컨테이너 1개: ${SIM_CONTAINER_NAME}):
  sim-up           시뮬 컨테이너 생성·시작 (백그라운드, sleep)
  sim-down         시뮬 컨테이너 중지·삭제
  sim-bringup      build-sim + Gazebo launch (Ctrl+C 시 launch만 종료, 컨테이너 유지)
  lane-tune        인지 모드 튜너 안내 (Gazebo 미기동 — bringup 후 컨테이너에서 실행)
  sim              build-sim + 자율주행 launch
  sim-manual       build-sim + 수동주행 launch
  verify-sim       토픽 검증 (bringup 실행 중)

워크스페이스 (${SIM_CONTAINER_NAME} 실행 중이면 같은 컨테이너, 없으면 일회성 dev):
  init             D-Racer-Kit clone + 링크
  build-sim        dracer_sim + inference colcon build
  build-inference  inference만 colcon build
  check            CI와 동일 inference.pipeline 검증

이미지·호스트 (컨테이너 밖):
  build            Docker 이미지 빌드 (Dockerfile)
  install-gazebo   Gazebo 1회 설치 (이미지에 반영)
  check-gpu        GPU 렌더링 확인

Examples:
  ./scripts/dev_container.sh sim-up
  ./scripts/dev_container.sh sim-bringup
  docker exec -it ${SIM_CONTAINER_NAME} bash
  ./scripts/dev_container.sh sim-bringup use_camera_view:=false headless:=true
  # 인지 검증 (Gazebo 재기동 금지):
  #   source /opt/ros/humble/setup.bash && source install/setup.bash
  #   python3 scripts/vision_tune/tune_lane_detect.py --mode white
EOF
}

cmd="${1:-help}"

case "${cmd}" in
  build)
    DOCKER_BUILDKIT=0 "${COMPOSE[@]}" build
    if ! docker compose run --rm dev bash -lc 'command -v gzserver >/dev/null 2>&1 || command -v gazebo >/dev/null 2>&1' 2>/dev/null; then
      echo ""
      echo "[SEA-Me] 베이스 이미지 빌드 완료. Gazebo가 없으면:"
      echo "  ./scripts/dev_container.sh install-gazebo"
    fi
    ;;
  sim-up)
    sim_up
    ;;
  sim-down)
    sim_down
    ;;
  init)
    run_in_sim_or_dev '
      set -eo pipefail
      ./scripts/init_workspace.sh
    '
    ;;
  build-inference)
    run_in_sim_or_dev '
      set -eo pipefail
      ./scripts/init_workspace.sh
      set +u
      source /opt/ros/humble/setup.bash
      set -u
      colcon build --symlink-install --packages-up-to inference
    '
    ;;
  build-sim)
    run_build_sim
    ;;
  install-gazebo)
    install_gazebo_in_image
    ;;
  check)
    run_in_sim_or_dev '
      set -eo pipefail
      ./scripts/init_workspace.sh
      set +u
      source /opt/ros/humble/setup.bash
      set -u
      colcon build --symlink-install --packages-up-to inference
      set +u
      source install/setup.bash
      set -u
      python3 -c "from inference.pipeline import MainPlanner, fuse_control; print(\"ok\")"
    '
    ;;
  sim-bringup)
    ensure_sim_display
    shift || true
    sim_exec_launch dracer_sim sim_bringup.launch.py "$@"
    ;;
  lane-tune)
    cat <<EOF
[SEA-Me] 인지 모드 튜너 — Gazebo를 이 명령으로 켜지 않습니다.

  터미널1:  ./scripts/dev_container.sh sim-bringup
  터미널2:  docker exec -it ${SIM_CONTAINER_NAME} bash
            source /opt/ros/humble/setup.bash && source install/setup.bash
            python3 scripts/vision_tune/tune_lane_detect.py --mode white

  키 1–9 / 0: white yellow dash dash_left dash_right fork fork_left fork_right red crossing
  문서: docs/lane-perception-topic.md §6.2 · scripts/vision_tune/README.md

  ❌ LANE_VISUALIZE=… ros2 launch dracer_sim sim_auto_driving.launch.py
     (bringup이 이미 있으면 Gazebo가 하나 더 뜸)
EOF
    ;;
  sim)
    ensure_sim_display
    sim_exec_launch dracer_sim sim_auto_driving.launch.py
    ;;
  sim-manual)
    ensure_sim_display
    sim_exec_launch dracer_sim sim_manual_driving.launch.py
    ;;
  check-gpu)
    bash "${ROOT}/scripts/check_sim_gpu.sh"
    ;;
  verify-sim)
    bash "${ROOT}/scripts/verify_sim.sh"
    ;;
  shell|sim-shell|sim-exec|check-rviz)
    echo "[SEA-Me] '${cmd}' 는 제거되었습니다." >&2
    echo "  시뮬: ./scripts/dev_container.sh sim-up → sim-bringup" >&2
    echo "  셸:  docker exec -it ${SIM_CONTAINER_NAME} bash" >&2
    echo "  빌드: ./scripts/dev_container.sh build-sim" >&2
    exit 1
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    run_in_sim_or_dev "set -eo pipefail
$*"
    ;;
esac
