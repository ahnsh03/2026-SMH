#!/usr/bin/env bash
# sim Docker 이미지에 rviz2 설치 여부 확인
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
fi

"${COMPOSE[@]}" run --rm dev bash -lc '
  source /opt/ros/humble/setup.bash
  if command -v rviz2 >/dev/null; then
    echo "[OK] rviz2: $(command -v rviz2)"
    rviz2 --help 2>&1 | head -1
    exit 0
  fi
  echo "[MISSING] rviz2 not in image. Run: ./scripts/dev_container.sh build"
  exit 1
'
