#!/usr/bin/env bash
# 시뮬 토픽·노드 동작 검증 (2026-smh-sim 실행 중, sim-bringup launch 동작 중)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SIM_CONTAINER_NAME="${SMH_SIM_CONTAINER:-2026-smh-sim}"

if ! docker ps --format '{{.Names}}' | grep -qx "${SIM_CONTAINER_NAME}"; then
  echo "[SEA-Me] ${SIM_CONTAINER_NAME} not running."
  echo "  ./scripts/dev_container.sh sim-up"
  echo "  ./scripts/dev_container.sh sim-bringup   # 다른 터미널"
  exit 1
fi

echo "[SEA-Me] Simulation interface verification (${SIM_CONTAINER_NAME})"
echo ""

docker exec "${SIM_CONTAINER_NAME}" bash -lc '
  set -e
  source /workspace/scripts/sim_gpu_env.sh 2>/dev/null || true
  source /opt/ros/humble/setup.bash
  if [ -f install/setup.bash ]; then source install/setup.bash; fi

  check_topic() {
    local topic="$1"
    local type="$2"
    if timeout 8 ros2 topic list 2>/dev/null | grep -qx "${topic}"; then
      echo "[OK] topic ${topic}"
      return 0
    fi
    echo "[FAIL] missing topic ${topic} (expected ${type})"
    return 1
  }

  fail=0
  check_topic /camera/image/compressed sensor_msgs/msg/CompressedImage || fail=1
  check_topic /camera/image_raw sensor_msgs/msg/Image || fail=1
  check_topic /control control_msgs/msg/Control || fail=1
  check_topic /cmd_vel geometry_msgs/msg/Twist || fail=1
  check_topic /joint_states sensor_msgs/msg/JointState || fail=1
  check_topic /robot_description std_msgs/msg/String || fail=1

  if timeout 5 ros2 topic echo /camera/image/compressed --once >/dev/null 2>&1; then
    echo "[OK] /camera/image/compressed publishes data"
  else
    echo "[WARN] /camera/image/compressed no data yet (sim-bringup running?)"
    fail=1
  fi

  if timeout 5 ros2 topic echo /camera/image_raw --once >/dev/null 2>&1; then
    echo "[OK] /camera/image_raw publishes data"
  else
    echo "[WARN] /camera/image_raw no data yet"
    fail=1
  fi

  exit "${fail}"
' || {
  echo ""
  echo "sim-bringup이 다른 터미널에서 실행 중인지 확인하세요."
  exit 1
}

echo ""
echo "[SEA-Me] Verification passed."
