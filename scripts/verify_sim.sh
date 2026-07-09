#!/usr/bin/env bash
# 시뮬 토픽·노드 동작 검증 (sim-bringup 실행 중 또는 sim-shell에서)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
fi

echo "[SEA-Me] Simulation interface verification"
echo ""

"${COMPOSE[@]}" run --rm sim bash -lc '
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
    echo "[WARN] /camera/image/compressed no data yet (Gazebo running?)"
    fail=1
  fi

  if timeout 5 ros2 topic echo /camera/image_raw --once >/dev/null 2>&1; then
    echo "[OK] /camera/image_raw publishes data (RViz)"
  else
    echo "[WARN] /camera/image_raw no data yet"
    fail=1
  fi

  exit "${fail}"
' || {
  echo ""
  echo "시뮬이 꺼져 있으면 먼저 다른 터미널에서:"
  echo "  ./scripts/dev_container.sh sim-bringup"
  exit 1
}

echo ""
echo "[SEA-Me] Verification passed."
