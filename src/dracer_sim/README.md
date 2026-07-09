# Gazebo simulator for D-Racer / LIMO (SEA:ME 2026-SMH)

D-Racer 실기와 **동일한 ROS2 토픽**으로 개발·검증하기 위한 Gazebo Classic 시뮬레이터입니다.  
기본 로봇은 **LIMO Ackermann** (`limo_ros2`의 `limo_car`)이며, 경량 D-Racer 박스 모델도 선택할 수 있습니다.

## 인터페이스 (실기와 동일)

설정 요약: `config/sim_interface.yaml`

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | 320×180 JPEG (C920e FOV, 16:9) |
| `/control` | `control_msgs/Control` | throttle / steering (-1~1) |
| `/battery_status` | `battery_msgs/Battery` | 시뮬 스텁 (80%) |
| `/joint_states` | `sensor_msgs/JointState` | RViz 로봇 관절 (Gazebo) |
| `/robot_description` | `std_msgs/String` | RViz RobotModel |

하드웨어 `camera_node`, `control_node`, `battery_node`는 **실행하지 않습니다**.

## 사전 요구사항

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-robot-state-publisher
```

`init_workspace.sh`가 D-Racer-Kit을 자동 clone하고, `vendor/limo_car`(레포 포함)를 링크합니다.

## 실행 (Docker — 팀 PC 시뮬)

```bash
# 팀 표준: dev_container.sh (상세: docs/simulation-setup.md)
./scripts/dev_container.sh sim-bringup   # Gazebo + RViz
./scripts/dev_container.sh sim           # + inference

# 컨테이너 안에서 직접 launch (sim-shell)
ros2 launch dracer_sim sim_bringup.launch.py
ros2 launch dracer_sim sim_auto_driving.launch.py
```

`init_workspace.sh` + colcon은 Docker 없이 22.04 네이티브에서도 가능하지만, **팀 표준은 dev_container.sh** 입니다.

## 트랙

- 텍스처: `models/track_plane/materials/textures/track_cw_real.png` (팀 CW 트랙)
- **실제 크기**: 이미지 가로 전체 **12.0 m**, 세로 **8.9975 m** (1211×908 px 비율)
- 스폰 기본값: `spawn_x=2.6`, `spawn_y=-3.92`, `spawn_yaw=-3.14`
- RViz2: `sim_bringup` 기본 `use_rviz:=true` (D-Racer 320×180 카메라)

```bash
ros2 launch dracer_sim sim_bringup.launch.py spawn_x:=0.0 spawn_y:=-3.6 spawn_yaw:=1.57
```

상세 WSL 설치·팀원 재현: [docs/simulation-setup.md](../../docs/simulation-setup.md)

## 카메라 파이프라인 (C920e 16:9)

| 단계 | 해상도 | 비고 |
|------|--------|------|
| C920e 네이티브 (USB) | 1920×1080 | 16:9 |
| Gazebo 렌더 | **640×360** | `limo_dracer_sim.xacro` |
| `sim_camera_republish` | **320×180** | 균일 다운스케일 + JPEG |
| 토픽 | `/camera/image/compressed`, `/camera/image_raw` (RViz) | |

실기: `config/vehicle_config.yaml`의 `IMAGE_HEIGHT: 180` — `init_workspace.sh` 후 `camera_node`가 동일 해상도로 publish합니다.

## 튜닝

| 파일 | 내용 |
|------|------|
| `config/camera_republish.yaml` | 출력 해상도·JPEG 품질 |
| `config/control_bridge.yaml` | 최대 속도·조향각·휠베이스 (LIMO: 0.24 m) |
| `urdf/limo_dracer_sim.xacro` | LIMO + C920e 카메라 (FOV 1.229 rad, 640×360) |
| `urdf/dracer_sim.urdf` | 경량 박스 모델 (`robot:=dracer`) |

카메라 마운트 각도는 실차 측정 후 `camera_joint` origin을 맞추세요.

## 검증

```bash
# 카메라 토픽
ros2 topic hz /camera/image/compressed

# 수동 제어 (조이스틱 없을 때)
ros2 topic pub /control control_msgs/msg/Control "{steering: 0.0, throttle: 0.3}" -r 10
```
