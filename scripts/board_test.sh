#!/usr/bin/env bash
# Feature unit tests for board-race (no ROS overlay required for most).
#
# Usage:
#   ./scripts/board_test.sh              # all unit + integration
#   ./scripts/board_test.sh fork         # perception/fork only
#   ./scripts/board_test.sh planner
#   ./scripts/board_test.sh blob
#   ./scripts/board_test.sh signs
#   ./scripts/board_test.sh aruco
#   ./scripts/board_test.sh integration
#   ./scripts/board_test.sh -k path_lost # pytest -k filter
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="${ROOT}/src/inference"
cd "${PKG}"

FEATURE="${1:-}"
EXTRA=()
case "${FEATURE}" in
  ''|all)
    TARGETS=(test/unit test/integration)
    ;;
  fork)
    TARGETS=(test/unit/perception/fork)
    ;;
  blob|perception)
    TARGETS=(test/unit/perception/blob)
    ;;
  planner)
    TARGETS=(test/unit/planner)
    ;;
  signs|sign|traffic)
    TARGETS=(test/unit/signs)
    ;;
  aruco)
    TARGETS=(test/unit/aruco)
    ;;
  integration|adapters)
    TARGETS=(test/integration)
    ;;
  -k)
    TARGETS=(test/unit test/integration)
    EXTRA=("$@")
    FEATURE=""
    ;;
  *)
    # Pass through as pytest args (path or -k …)
    TARGETS=()
    EXTRA=("$@")
    ;;
esac

if [[ -n "${FEATURE}" && "${FEATURE}" != -k ]]; then
  shift || true
  EXTRA+=("$@")
fi

export PYTHONPATH="${PKG}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v pytest >/dev/null 2>&1; then
  PYTEST=(pytest)
elif python3 -m pytest --version >/dev/null 2>&1; then
  PYTEST=(python3 -m pytest)
else
  echo "pytest not found. Install: pip install pytest  (or use docker 2026-smh-dev)" >&2
  exit 1
fi

echo "[board_test] ${PYTEST[*]} ${TARGETS[*]:-} ${EXTRA[*]:-}"
exec "${PYTEST[@]}" ${TARGETS[@]+"${TARGETS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
