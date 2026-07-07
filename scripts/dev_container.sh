#!/usr/bin/env bash
# PC(WSL) 개발 컨테이너 헬퍼 — Ubuntu 22.04 + ROS2 Humble 환경을 팀원 간 통일합니다.
#
# Usage:
#   ./scripts/dev_container.sh build            # 이미지 빌드
#   ./scripts/dev_container.sh shell            # 컨테이너 bash 진입 (기본)
#   ./scripts/dev_container.sh init             # init_workspace.sh 실행
#   ./scripts/dev_container.sh build-inference  # CI와 동일한 inference 빌드
#   ./scripts/dev_container.sh check            # pipeline import 검증
#   ./scripts/dev_container.sh <cmd...>         # 임의 명령 실행
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
fi

run_dev() {
  "${COMPOSE[@]}" run --rm dev "$@"
}

usage() {
  cat <<'EOF'
Usage: ./scripts/dev_container.sh <command>

Commands:
  build            Docker 이미지 빌드 (2026-smh-dev:latest)
  shell            개발 컨테이너 bash 셸 진입
  init             D-Racer-Kit clone + src/ 심볼릭 링크
  build-inference  colcon build --packages-up-to inference
  check            inference.pipeline import 검증 (CI와 동일)
  help             이 도움말

Examples:
  ./scripts/dev_container.sh build
  ./scripts/dev_container.sh shell
  ./scripts/dev_container.sh build-inference
EOF
}

cmd="${1:-shell}"

case "${cmd}" in
  build)
    "${COMPOSE[@]}" build
    ;;
  shell)
    run_dev bash
    ;;
  init)
    run_dev bash -lc './scripts/init_workspace.sh'
    ;;
  build-inference)
    run_dev bash -lc '
      set -euo pipefail
      ./scripts/init_workspace.sh
      source /opt/ros/humble/setup.bash
      colcon build --symlink-install --packages-up-to inference
    '
    ;;
  check)
    run_dev bash -lc '
      set -euo pipefail
      ./scripts/init_workspace.sh
      source /opt/ros/humble/setup.bash
      colcon build --symlink-install --packages-up-to inference
      source install/setup.bash
      python3 -c "from inference.pipeline import fuse_control, run_perception; print(\"ok\")"
    '
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    run_dev bash -lc "$*"
    ;;
esac
