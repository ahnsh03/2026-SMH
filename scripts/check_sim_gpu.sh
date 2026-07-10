#!/usr/bin/env bash
# WSL Docker sim 컨테이너에서 OpenGL 렌더러 확인 (Gazebo GPU 여부).
# llvmpipe / softpipe / SVGA3D → CPU 소프트웨어 렌더링 (렉 원인)
# D3D12 / NVIDIA / AMD / Intel → GPU 가속 가능
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SIM_CONTAINER_NAME="${SMH_SIM_CONTAINER:-2026-smh-sim}"

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
fi

GPU_CHECK_SCRIPT='
set -e
source /workspace/scripts/sim_gpu_env.sh
export DISPLAY=${DISPLAY:-:0}

if ! command -v glxinfo >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq mesa-utils
fi

echo "=== 장치 ==="
ls -la /dev/dxg 2>/dev/null || echo "/dev/dxg: 없음"
ls -la /dev/dri 2>/dev/null || echo "/dev/dri: 없음 (WSL D3D12만 사용 가능)"
ls /usr/lib/wsl/lib/libd3d12.so /usr/lib/wsl/lib/libnvidia-gpucomp.so* 2>/dev/null | head -3 || true

echo ""
echo "=== GPU 환경변수 ==="
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "MESA_LOADER_DRIVER_OVERRIDE=${MESA_LOADER_DRIVER_OVERRIDE:-unset}"
echo "GALLIUM_DRIVER=${GALLIUM_DRIVER:-unset}"
echo "MESA_D3D12_DEFAULT_ADAPTER_NAME=${MESA_D3D12_DEFAULT_ADAPTER_NAME:-unset}"

echo ""
echo "=== OpenGL (glxinfo) ==="
glxinfo -B 2>&1 | grep -E "direct rendering|OpenGL vendor|OpenGL renderer|OpenGL version" || true

renderer="$(glxinfo -B 2>/dev/null | grep "OpenGL renderer" || true)"
echo ""
if echo "${renderer}" | grep -qiE "llvmpipe|softpipe|SVGA3D|software"; then
  echo ">>> 판정: CPU 소프트웨어 렌더링 (Gazebo GUI 렉 예상)"
  echo ">>> 조치: ./scripts/dev_container.sh build 후 재시도"
  echo ">>> 대안: ./scripts/dev_container.sh sim-bringup headless:=true"
  exit 1
elif echo "${renderer}" | grep -qiE "D3D12|NVIDIA|AMD|Radeon|Intel"; then
  echo ">>> 판정: GPU 가속 렌더링 (D3D12)"
  exit 0
else
  echo ">>> 판정: 불명확 — 위 renderer 문자열을 확인하세요"
  exit 2
fi
'

echo "[SEA-Me] OpenGL 렌더러 확인 (DISPLAY=${DISPLAY:-:0})"
echo ""

if docker ps --format '{{.Names}}' | grep -qx "${SIM_CONTAINER_NAME}"; then
  echo "[SEA-Me] using ${SIM_CONTAINER_NAME}"
  printf '%s\n' "$GPU_CHECK_SCRIPT" | docker exec -i "${SIM_CONTAINER_NAME}" bash -s
else
  echo "[SEA-Me] ${SIM_CONTAINER_NAME} not running — one-off sim container (same GPU settings)"
  printf '%s\n' "$GPU_CHECK_SCRIPT" | "${COMPOSE[@]}" run -T --rm sim bash -s
fi
