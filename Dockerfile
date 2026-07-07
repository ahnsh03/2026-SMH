# Team dev image — matches D3-G (Ubuntu 22.04 + ROS2 Humble) and GitHub Actions CI.
FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble

# Build tools + Python deps used by inference / opencv packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-opencv \
    python3-pip \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Auto-source ROS (and workspace overlay when present) in interactive shells.
RUN cat >> /etc/bash.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi
EOF

CMD ["/bin/bash"]
