# Gazebo simulator for D-Racer / LIMO (SEA:ME 2026-SMH)

D-Racer 실기와 **동일한 ROS2 토픽**으로 개발·검증하기 위한 Gazebo Classic 시뮬레이터입니다.  
기본 로봇은 **LIMO Ackermann** (`vendor/limo_car`, 레포 포함)이며, 경량 D-Racer 박스 모델도 선택할 수 있습니다.

## 인터페이스 (실기와 동일)

설정 요약: `config/sim_interface.yaml`

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | 320×180 JPEG (C920e FOV, 16:9) |
| `/control` | `control_msgs/Control` | throttle / steering (-1~1) |
| `/battery_status` | `battery_msgs/Battery` | 시뮬 스텁 (80%) |
| `/joint_states` | `sensor_msgs/JointState` | Gazebo 관절 |
| `/robot_description` | `std_msgs/String` | (디버그용, 기본 시각화 없음) |

하드웨어 `camera_node`, `control_node`, `battery_node`는 **실행하지 않습니다**.

## 사전 요구사항

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-robot-state-publisher
```

`init_workspace.sh`가 D-Racer-Kit을 자동 clone하고, `vendor/limo_car`(레포 포함)를 링크합니다.

## 실행 (Docker — 팀 PC 시뮬)

상세: [docs/simulation-setup.md](../../docs/simulation-setup.md) §4 · **직접 명령**: [§4.8](../../docs/simulation-setup.md#48-직접-명령어-치트시트-스크립트-없이)

```bash
# 컨테이너 생성 (호스트)
docker compose run -d --name 2026-smh-sim sim sleep infinity

# 터미널 1 — 시뮬 (컨테이너 안)
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch dracer_sim sim_bringup.launch.py

# 터미널 2 — inference (컨테이너 안, 별도 셸)
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run inference inference_node --ros-args -p use_sim_time:=true

# 또는 스크립트: ./scripts/dev_container.sh sim-bringup
```

`init_workspace.sh` + colcon은 Docker 없이 22.04 네이티브에서도 가능하지만, **팀 표준은 dev_container.sh** 입니다.

## 트랙

- 텍스처: `models/track_plane/materials/textures/track_cw_real.png` (팀 CW 트랙)
- **실제 크기**: 이미지 가로 전체 **12.0 m**, 세로 **8.9975 m** (1211×908 px 비율)
- 스폰 기본값: `spawn_x=2.6`, `spawn_y=-3.92`, `spawn_yaw=-3.14`
- **카메라 프리뷰**: `sim_bringup` 기본 `use_camera_view:=true` (320×180 → 창 640×360)

### 미션 표지판 (갈림길 · ArUco)

기본 월드 `worlds/track_cw.world`에 **3개 모두** 고정 배치됩니다 (별도 맵·launch 인자 없음).

| 모델 | 실물 크기 | 판자 | 대회 연계 |
|------|-----------|------|-----------|
| `turn_sign_left` | 파란 원 **Ø 20 cm** | 흰 원판 **Ø 21 cm** | 갈림길 좌회전 |
| `turn_sign_right` | 파란 원 **Ø 20 cm** | 흰 원판 **Ø 21 cm** | 갈림길 우회전 |
| `aruco_stop_sign` | 마커 **15 cm** | 흰 사각 **18 cm** | 동적 장애물 정지 (DICT_6X6_50 ID 3) |

**현재 배치 (2026-07-11)**

| 표지판 | x | y | yaw | 비고 |
|--------|---|---|-----|------|
| 좌회전 | -3.14 | 3.71 | -90° | 시계방향 90° → -X 향함 |
| 우회전 | -3.00 | 3.71 | -90° | 좌회전과 나란히 |
| ArUco | 4.5 | 0.0 | +180° | +Y 향함 (기본 -Y의 반대) |

**높이 (아랫선 기준)**  
트랙 바닥(`track_plane`)은 **z = 0.01 m (1 cm)**. 표지판 **아랫선**은 트랙 위 **10 cm** (`z = 0.11`).

| 표지판 | 링크 중심 z (world) | 계산 |
|--------|---------------------|------|
| 좌/우회전 | **0.215** | 0.11 + 반지름 0.105 |
| ArUco | **0.20** | 0.11 + 높이/2 0.09 |

위치·회전 수정: `config/mission_signs.yaml` 편집 후 **`worlds/track_cw.world`의 `<pose>`도 같이** 맞출 것 (yaml은 참고용, 실제 스폰은 world 파일).

**에셋·텍스처 (clone만으로 사용 가능)**

```
src/dracer_sim/
├── assets/signs/              # 원본 PNG (Git 포함)
│   ├── trun_left.png
│   ├── trun_right.png
│   └── ArUco_stop.png
├── models/
│   ├── turn_sign_left/
│   ├── turn_sign_right/
│   └── aruco_stop_sign/
└── scripts/prepare_mission_signs.py   # build-sim 시 자동 실행
```

- 좌/우 텍스처: Ø21 cm 원형 마스크 + 흰 배경 (`alpha_blend`)
- 모델마다 **서로 다른 PNG 파일명** 사용 (Gazebo `sign.png` 전역 캐시 버그 회피)
- 원본 PNG 수정 후: `python3 scripts/prepare_mission_signs.py` → `build-sim`

상세 튜닝 가이드: [docs/simulation-setup.md § 미션 표지판](../../docs/simulation-setup.md#미션-표지판-갈림길--aruco)

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
| 토픽 | `/camera/image/compressed`, `/camera/image_raw` (OpenCV 프리뷰) | |

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
