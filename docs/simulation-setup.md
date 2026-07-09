# Gazebo 시뮬 — 팀원 PC 재현 가이드

> **이 문서만 따라하면** `git clone` 한 대로 Docker에서 LIMO + CW 트랙 시뮬을 실행할 수 있습니다.  
> 트러블슈팅·GPU: [simulation.md](./simulation.md) · 패키지 상세: [../src/dracer_sim/README.md](../src/dracer_sim/README.md)

마지막 업데이트: 2026-07-10

---

## 0. 한눈에 보기

| 항목 | 내용 |
|------|------|
| 필요 환경 | Windows 11 + WSL2 + **Docker Desktop** (WSLg 권장) |
| 호스트 ROS | **설치 불필요** (WSL 24.04/26.04도 OK) |
| 로봇 모델 | LIMO Ackermann (`vendor/limo_car`) |
| 트랙 | CW 팀 트랙, 가로 **12 m** |
| 카메라 | C920e FOV, **320×180** JPEG (16:9) |
| 시각화 | **Gazebo** (3D) + **RViz2** (카메라·로봇 모델) |
| 웹 모니터 | 시뮬 기본 **OFF** (실기용 D-Racer monitor) |

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH && chmod +x scripts/*.sh
./scripts/dev_container.sh build
./scripts/dev_container.sh install-gazebo   # 최초 1회
./scripts/dev_container.sh init
./scripts/dev_container.sh build-sim
./scripts/dev_container.sh sim-bringup
```

---

## 1. 무엇이 시뮬되는가

| 항목 | 내용 |
|------|------|
| 로봇 | LIMO Ackermann (`vendor/limo_car`, ROS2 Humble) |
| 트랙 | CW 팀 트랙 텍스처 (`track_cw_real.png`), 가로 **12 m** |
| 인터페이스 | **D-Racer 실기와 동일 토픽** (`src/dracer_sim/config/sim_interface.yaml`) |
| 참고 레포 | `ahns_limo_sim`은 ROS1·라이다 등 **미사용** — 트랙 plane·Ackermann만 참고 |

### 아키텍처

```
Gazebo (LIMO + 트랙 + C920e 카메라 640×360)
    │
    ├─ /gazebo/camera/image_raw
    │       └─ sim_camera_republish → /camera/image/compressed (320×180 JPEG)
    │                              → /camera/image_raw (RViz용)
    │
    ├─ /control (Control) ← sim_control_bridge ← inference 또는 수동 pub
    │       └─ /cmd_vel → Ackermann Gazebo plugin
    │
    └─ /battery_status ← sim_battery_stub (80% 고정)

RViz2: RobotModel (/robot_description) + Image (/camera/image_raw)
```

### D-Racer 호환 토픽

| 토픽 | 타입 | 용도 |
|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | **inference_node** 입력 (320×180 JPEG) |
| `/camera/image_raw` | `sensor_msgs/Image` | RViz 카메라 뷰 |
| `/control` | `control_msgs/Control` | throttle / steering (-1~1) |
| `/battery_status` | `battery_msgs/Battery` | monitor 스텁 |
| `/joint_states` | `sensor_msgs/JointState` | RViz 로봇 관절 |
| `/odom` | `nav_msgs/Odometry` | Gazebo Ackermann 플러그인 |

내부 전용: `/cmd_vel` ← `sim_control_bridge`

---

## 2. 사전 조건 (Windows + WSL)

### 2.1 필수 소프트웨어

1. **Windows 11** (WSLg 내장 — Gazebo/RViz GUI에 유리)
2. **WSL2** Ubuntu 배포판 (22.04 / 24.04 / 26.04 모두 가능)
3. **Docker Desktop** — 설치 후 실행 상태 유지
4. Docker Desktop → **Settings → Resources → WSL Integration** → 사용 중인 배포판 **ON**

### 2.2 확인 명령 (WSL 터미널)

```bash
docker --version
docker compose version
echo $DISPLAY    # 보통 :0 (WSLg)
```

### 2.3 주의

- 호스트 WSL에 `sudo apt install ros-humble-*` 하지 마세요. **Humble은 Ubuntu 22.04 전용**이며, 팀은 Docker 안에서만 ROS를 씁니다.
- `external/D-Racer-Kit/`은 Git에 없습니다. `./scripts/dev_container.sh init`이 자동 clone합니다.
- `vendor/limo_car/`는 **레포에 포함**되어 있습니다. 별도 clone 불필요.

---

## 3. 최초 설치 (처음 clone한 팀원)

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
```

| 순서 | 명령 | 소요 | 설명 |
|------|------|------|------|
| 1 | `./scripts/dev_container.sh build` | ~3–8분 | Ubuntu 22.04 + Humble 베이스 이미지 |
| 2 | `./scripts/dev_container.sh install-gazebo` | ~5–10분 | Gazebo + gazebo-ros-pkgs (**최초 1회**) |
| 3 | `./scripts/dev_container.sh init` | ~30초 | D-Racer-Kit clone + `src/` 링크 + 팀 `vehicle_config` |
| 4 | `./scripts/dev_container.sh build-sim` | ~1–2분 | `dracer_sim` + `inference` + 의존 패키지 빌드 |
| 5 | `./scripts/dev_container.sh check-gpu` | 수초 | GPU 렌더링 확인 (선택, 권장) |

> `sim-bringup` 실행 시 Gazebo가 없으면 **자동 설치**를 시도하지만, 네트워크 안정을 위해 `install-gazebo`를 먼저 실행하는 것을 권장합니다.

### init이 하는 일

- `external/D-Racer-Kit` clone (`release/v1.0.0`)
- `src/camera`, `src/control`, `src/inference` 등 심볼릭 링크
- `vendor/limo_car` → `src/limo_car` 링크
- **`config/vehicle_config.yaml`** → `src/config/vehicle_config.yaml` (팀 카메라 320×180)

---

## 4. 시뮬 실행

### 4.1 기본 (Gazebo + RViz)

```bash
./scripts/dev_container.sh sim-bringup
```

**정상이면 다음이 보입니다:**

| 창 | 내용 |
|----|------|
| **Gazebo** | CW 트랙 바닥 + LIMO 본체·바퀴 mesh |
| **RViz2** | 로봇 모델 + D-Racer 카메라 영상 (320×180) |

RViz TF 표시는 기본 **OFF**입니다. 필요 시 Displays → TF 를 켜세요.

### 4.2 자율주행 (inference 포함)

```bash
./scripts/dev_container.sh sim
```

`sim_bringup` + `inference_node` + 조이스틱 노드(캘리브레이션 off).  
모듈 개발 시 카메라 토픽과 `/control` 출력을 시뮬에서 바로 검증할 수 있습니다.

### 4.3 수동 주행 (USB 조이스틱)

```bash
./scripts/dev_container.sh sim-manual
```

WSL USB 패스스루 설정이 필요할 수 있습니다. 조이스틱 없이 테스트:

```bash
ros2 topic pub /control control_msgs/msg/Control "{steering: 0.0, throttle: 0.3}" -r 10
```

### 4.4 launch 옵션

`sim-bringup` 뒤에 ROS launch 인자를 그대로 전달할 수 있습니다.

```bash
./scripts/dev_container.sh sim-bringup headless:=true
./scripts/dev_container.sh sim-bringup use_rviz:=false
./scripts/dev_container.sh sim-bringup use_monitor:=true   # 웹 모니터 (Docker Flask 이슈 가능)
./scripts/dev_container.sh sim-bringup spawn_x:=0.0 spawn_y:=-3.6
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `headless` | `false` | `true`면 Gazebo 3D 창 끔 (물리·카메라는 동작) |
| `use_rviz` | `true` | RViz2 카메라·로봇 모델 |
| `use_monitor` | `false` | D-Racer 웹 모니터 (시뮬에서는 보통 불필요) |
| `robot` | `limo` | `dracer` = 경량 박스 모델 |
| `spawn_x/y/z/yaw` | 2.6 / -3.92 / 0.15 / -3.14 | 트랙 위 스폰 위치 |

---

## 5. 동작 검증

시뮬 실행 중 **다른 WSL 터미널**에서:

```bash
./scripts/dev_container.sh verify-sim
```

기대 출력 예:

```
[OK] topic /camera/image/compressed
[OK] topic /camera/image_raw
[OK] /camera/image/compressed publishes data
[SEA-Me] Verification passed.
```

수동 확인:

```bash
ros2 topic hz /camera/image/compressed    # ~30 Hz 목표
ros2 topic echo /camera/image_raw --once
ros2 topic hz /joint_states
```

GPU 확인 (렉이 심할 때):

```bash
./scripts/dev_container.sh check-gpu
# OpenGL renderer: D3D12 (NVIDIA ...) → OK
# llvmpipe → CPU 렌더링 (headless:=true 권장)
```

---

## 6. `dev_container.sh` 명령 전체

```bash
./scripts/dev_container.sh help
```

| 명령 | 용도 |
|------|------|
| `build` | Docker 이미지 빌드 |
| `install-gazebo` | Gazebo 1회 설치 |
| `init` | D-Racer-Kit + 워크스페이스 링크 |
| `build-inference` | inference만 빌드 |
| `build-sim` | dracer_sim + inference 빌드 |
| `check` | CI와 동일 import 검증 |
| `shell` | dev 컨테이너 bash (코드 편집·빌드) |
| `sim-shell` | sim 컨테이너 bash (Gazebo GUI 환경) |
| `sim-bringup` | Gazebo + 트랙 + 브리지 + RViz |
| `sim` | bringup + inference 자율주행 |
| `sim-manual` | bringup + 조이스틱 수동주행 |
| `check-gpu` | OpenGL 렌더러 확인 |
| `check-rviz` | rviz2 설치 확인 |
| `verify-sim` | 시뮬 토픽 검증 (sim 실행 중) |

---

## 7. 레포 구조 (시뮬 관련)

```
2026-SMH/
├── Dockerfile                  # 베이스 이미지 (Gazebo 제외, Mesa+rviz2)
├── docker-compose.yml          # dev / sim 서비스, WSL GPU 마운트
├── config/
│   └── vehicle_config.yaml     # 팀 카메라 320×180 (init → src/config 링크)
├── scripts/
│   ├── dev_container.sh        # ★ 모든 PC 작업의 진입점
│   ├── init_workspace.sh
│   ├── verify_sim.sh
│   ├── check_sim_gpu.sh
│   └── sim_gpu_env.sh
├── vendor/limo_car/            # LIMO URDF + mesh (Git 포함)
├── external/D-Racer-Kit/       # init 시 clone (Git 제외)
└── src/
    ├── dracer_sim/             # ★ Gazebo 시뮬 패키지
    │   ├── launch/             # sim_bringup, sim_auto_driving …
    │   ├── urdf/               # limo_dracer_sim.xacro
    │   ├── models/track_plane/ # CW 트랙 텍스처
    │   ├── worlds/track_cw.world
    │   ├── config/             # camera, control, sim_interface
    │   └── rviz/sim_camera.rviz
    └── inference/              # 팀 자율주행 (시뮬에서 동일 코드 실행)
```

---

## 8. inference 모듈 개발 워크플로

```bash
# 1) 모듈 코드 수정 (예: modules/lane_detection.py)
# 2) PR 전 로컬 검증
./scripts/dev_container.sh check

# 3) 시뮬에서 perception 루프 확인
./scripts/dev_container.sh build-sim
./scripts/dev_container.sh sim

# 4) merge 후 실차 최종 확인 (D3-G)
./scripts/board_sync.sh
ros2 launch inference auto_driving.launch.py
```

시뮬은 **카메라·토픽·inference 파이프라인** 검증용입니다. 모터·조향 모델은 LIMO Ackermann이라 실차와 다릅니다.

---

## 9. 카메라 설정 (320×180)

| 환경 | 설정 파일 | 해상도 |
|------|-----------|--------|
| 실기 (D3-G) | `config/vehicle_config.yaml` | 320×180 (16:9) |
| 시뮬 Gazebo | `urdf/limo_dracer_sim.xacro` | 640×360 렌더 |
| 시뮬 출력 | `config/camera_republish.yaml` | 320×180 JPEG |

C920e 네이티브는 **1920×1080 (16:9)** 입니다. 주최측 기본 320×160(2:1) 대신 팀은 왜곡 없는 **320×180**을 사용합니다.  
상세: [hardware-camera.md](./hardware-camera.md)

---

## 10. 자주 하는 실수

| 증상 | 원인 | 해결 |
|------|------|------|
| `ros2: command not found` | 호스트 WSL에서 직접 실행 | `./scripts/dev_container.sh sim-bringup` |
| RViz에 로봇 mesh 없음 | 구 빌드 / mesh URI | `build-sim` 후 재실행 |
| Gazebo에 카메라만 떠 있음 | Gazebo가 `package://` mesh 미지원 | `build-sim` 후 재실행 (`file://` URDF spawn) |
| Gazebo 없음 | 베이스 이미지만 빌드 | `install-gazebo` 1회 |
| apt Hash Sum mismatch | WSL Docker 미러 | `build`·`install-gazebo` 재시도 (자동) |
| 렉 심함 | CPU 소프트웨어 렌더링 | `check-gpu`, `headless:=true` |
| monitor_node 죽음 | Docker Flask 버전 충돌 | 시뮬 기본 OFF — RViz 사용 |
| 트랙 텍스처 없음 | `build-sim` 안 함 | `build-sim` 후 재실행 |
| `control_msgs` 없음 | init 안 함 | `./scripts/dev_container.sh init` |

---

## 11. 일상 워크플로

```bash
git pull
./scripts/dev_container.sh init          # vehicle_config·링크 갱신
./scripts/dev_container.sh build-sim   # dracer_sim / inference 변경 후
./scripts/dev_container.sh sim-bringup # 튜닝·카메라 확인
./scripts/dev_container.sh sim         # inference 통합 테스트
```

실차 최종 확인: D3-G에서 `./scripts/board_sync.sh` → `ros2 launch inference auto_driving.launch.py`

---

## 12. 관련 문서

| 문서 | 내용 |
|------|------|
| [simulation.md](./simulation.md) | GPU·GUI 트러블슈팅 |
| [dev-environment.md](./dev-environment.md) | Docker·CI·보드 워크플로 |
| [hardware-camera.md](./hardware-camera.md) | C920e 스펙 |
| [board-workflow.md](./board-workflow.md) | D3-G 실차 개발 |
| [collaboration.md](./collaboration.md) | 브랜치·PR 규칙 |
