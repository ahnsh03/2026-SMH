# Ubuntu 22.04 + ROS2 Humble (가벼운 베이스, ~3분)
# Gazebo는 Hash mismatch 회피를 위해 별도 1회 설치: ./scripts/dev_container.sh install-gazebo
# GPU(Mesa D3D12) + rviz2는 이미지에 포함
FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble

# WSL/Docker Desktop에서 apt Hash Sum mismatch 시 자동 재시도 (최대 10회)
RUN set -eux; \
    packages=" \
      build-essential ca-certificates curl git \
      python3-colcon-common-extensions python3-numpy python3-opencv \
      python3-pip python3-flask \
      libgl1-mesa-dri libgl1-mesa-glx libglvnd0 libegl1 libglx0 mesa-utils \
      ros-humble-cv-bridge ros-humble-robot-state-publisher \
      ros-humble-rviz2 ros-humble-xacro \
    "; \
    success=0; \
    for attempt in 1 2 3 4 5 6 7 8 9 10; do \
      echo "=== apt attempt ${attempt}/10 ==="; \
      rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
      apt-get update -qq && \
      apt-get install -y --no-install-recommends --fix-missing ${packages} && \
      success=1 && break || \
      echo "apt failed — retry in 30s..." && sleep 30; \
    done; \
    test "${success}" -eq 1; \
    rm -rf /var/lib/apt/lists/*

# WSLg container GPU defaults (docker-compose sim service에서도 설정)
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    LIBGL_ALWAYS_SOFTWARE=0 \
    LIBVA_DRIVER_NAME=d3d12 \
    MESA_LOADER_DRIVER_OVERRIDE=d3d12 \
    GALLIUM_DRIVER=d3d12 \
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

WORKDIR /workspace

RUN cat >> /etc/bash.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
if [ -f /workspace/scripts/sim_gpu_env.sh ]; then
  source /workspace/scripts/sim_gpu_env.sh
fi
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi
EOF

CMD ["/bin/bash"]
