# WSL2 Docker에서 Gazebo/RViz OpenGL GPU(D3D12) 사용 — microsoft/wslg container 가이드
# https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=0
export LIBVA_DRIVER_NAME=d3d12
export MESA_LOADER_DRIVER_OVERRIDE=d3d12
export GALLIUM_DRIVER=d3d12
# NVIDIA GPU (호스트 /usr/lib/wsl/lib에 libcuda 있으면 NVIDIA 우선)
if ls /usr/lib/wsl/lib/libnvidia*.so* >/dev/null 2>&1; then
  export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
elif ls /usr/lib/wsl/lib/libd3d12*.so* >/dev/null 2>&1; then
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-}"
fi
