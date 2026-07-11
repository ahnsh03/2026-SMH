# Gazebo 시뮬레이터 (D-Racer)

> **팀원 재현 가이드**: [simulation-setup.md](./simulation-setup.md) ★  
> 마지막 업데이트: 2026-07-11  
> 패키지 상세: [`../src/dracer_sim/README.md`](../src/dracer_sim/README.md)  
> PC 개발 환경: [dev-environment.md](./dev-environment.md) ★  
> 카메라 스펙: [hardware-camera.md](./hardware-camera.md)

팀 CW 트랙 이미지를 Gazebo 바닥에 깔고, **실기와 동일한 토픽**으로 `inference`를 검증합니다.

## 한 줄 요약

| 항목 | 내용 |
|------|------|
| 패키지 | `src/dracer_sim` + `vendor/limo_car` (레포 포함) |
| 기본 로봇 | **LIMO Ackermann** (`robot:=limo`, 기본값) |
| 트랙 크기 | 이미지 가로 **12.0 m** × 세로 **8.9975 m** (plane UV, 왜곡 없음) |
| 미션 표지판 | 좌/우회전 + ArUco **3종** (`track_cw.world` 고정 배치) |
| 실행 환경 | **Docker** — 컨테이너 `2026-smh-sim` 1개 + 터미널 2개 |
| 명령 (터미널1) | `./scripts/dev_container.sh sim-bringup` |
| 셸 (터미널2) | `docker exec -it 2026-smh-sim bash` |

---

## 개발 워크플로 (터미널 2개)

상세: [simulation-setup.md §4](./simulation-setup.md#4-일상-개발-워크플로-컨테이너-1개--터미널-2개)

```
터미널1  sim-bringup     →  Gazebo + /camera/* + /control 브리지
터미널2  docker exec     →  inference 빌드·실행 (use_sim_time:=true)
```

- 터미널1 **Ctrl+C**: launch만 종료, 컨테이너 유지
- 터미널2에서 `sim_bringup` launch **재실행 금지** (Gazebo 중복)
- 통합 테스트만: `./scripts/dev_container.sh sim`
- **스크립트 없이 docker/ros2만**: [simulation-setup.md §4.8](./simulation-setup.md#48-직접-명령어-치트시트-스크립트-없이)

---

## WSL 26.04 / 24.04 — Docker로만 실행

ROS2 **Humble**은 Ubuntu **22.04 전용**입니다. WSL 26.04에서 아래는 **실패가 정상**입니다.

```bash
sudo apt install ros-humble-desktop   # ❌ Unable to locate package
```

팀 표준: **Docker Desktop** + `2026-SMH` dev 이미지 (내부는 22.04 + Humble + Gazebo).

---

## 빠른 시작 (Docker)

```bash
cd ~/projects/2026-seame-hackathon/2026-SMH   # 본인 clone 경로

chmod +x scripts/*.sh

# 1) 이미지 빌드 (가벼움, Gazebo 제외 — Hash mismatch 회피)
./scripts/dev_container.sh build

# Gazebo가 없을 때만 1회 (apt 재시도 포함)
./scripts/dev_container.sh install-gazebo

# 2) 워크스페이스 초기화 (D-Racer-Kit 자동 clone + vendor/limo_car 링크)
./scripts/dev_container.sh init

# 3) 시뮬 개발 (터미널 2개)
./scripts/dev_container.sh sim-up
./scripts/dev_container.sh sim-bringup      # 터미널1
docker exec -it 2026-smh-sim bash           # 터미널2 → inference

# 통합 테스트 한 번에: ./scripts/dev_container.sh sim
```

| 명령 | 설명 |
|------|------|
| `sim-up` / `sim-down` | `2026-smh-sim` 생성·삭제 |
| `sim-bringup` | **터미널1**: Gazebo + 브리지 + 카메라 프리뷰 |
| `docker exec -it 2026-smh-sim bash` | **터미널2**: inference 개발 셸 |
| `sim` | bringup + inference (한 터미널) |
| `sim-manual` | bringup + 조이스틱 수동 |

---

## 트랙 스케일

| 항목 | 값 |
|------|-----|
| 이미지 | `track_cw_real.png` (1211 × 908 px) |
| 실제 가로 | **12.0 m** (대회 도면 기준) |
| Gazebo 평면 | 12.0 m × 8.9975 m (`<plane>`, box 아님 — 세로 늘어남 방지) |
| 트랙 바닥 z | **0.01 m** (`track_plane` pose, 지면 z=0 위 1 cm) |
| 기본 스폰 | `spawn_x=2.6`, `spawn_y=-3.92`, `spawn_yaw=-3.14`, LIMO `spawn_z=0.15` |

미션 표지판(좌/우회전·ArUco) 배치·크기·높이: [simulation-setup.md § 미션 표지판](./simulation-setup.md#미션-표지판-갈림길--aruco)

---

## 사전 조건

1. **Docker Desktop** 설치 + WSL Integration (26.04 배포판 활성화)
2. **Windows 11** + WSLg 권장 (Gazebo GUI)
3. Docker Desktop이 **실행 중**이어야 함

```bash
docker --version
docker compose version
```

---

## 동작 확인 (`sim-bringup` 실행 중, 터미널2 `docker exec` 또는 호스트)

컨테이너 안 또는 `network_mode: host`이므로 **WSL 호스트 터미널**에서도:

```bash
ros2 topic hz /camera/image/compressed
ros2 topic pub /control control_msgs/msg/Control "{steering: 0.0, throttle: 0.25}" -r 10
```

웹 모니터: `http://127.0.0.1:5000` — 시뮬 기본 **OFF** (`use_monitor:=true`로 켤 수 있으나 Docker Flask 버전 이슈 가능)

카메라 프리뷰 (`sim_camera_preview`, 기본 ON): `/camera/image_raw` 320×180 → 창 640×360 · 검증: `./scripts/dev_container.sh verify-sim`

---

## Gazebo 렉 / GPU

WSL2 Docker에서 **GPU가 컨테이너에 전달되지 않으면** Gazebo GUI(`gzclient`)가 **CPU 소프트웨어 렌더링**으로 동작해 렉이 큽니다.

| 확인 | 내용 |
|------|------|
| 호스트 | `/dev/dxg` 있음 → WSL GPU 가능 |
| 이전 컨테이너 | `/dev/dri` 없음 → GPU 미사용(소프트웨어 렌더링) |

`docker-compose.yml` `sim` 서비스에 WSLg GPU 설정이 포함되어 있습니다 ([Microsoft WSLg container 가이드](https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md)):

- `/dev/dxg`, `/usr/lib/wsl` 마운트
- `MESA_LOADER_DRIVER_OVERRIDE=d3d12`, `GALLIUM_DRIVER=d3d12`, `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`

**이미지 재빌드 필수** (`Dockerfile`에 Mesa GL 패키지 포함):

```bash
./scripts/dev_container.sh build
./scripts/dev_container.sh check-gpu
```

`OpenGL renderer string: D3D12 (NVIDIA ...)` 이면 GPU 사용 중입니다.  
`llvmpipe` 이면 CPU 렌더링입니다.

렉이 계속되면:

```bash
# Gazebo 3D 창 없이 (물리·카메라·프리뷰 유지)
ros2 launch dracer_sim sim_bringup.launch.py headless:=true
```

`headless:=true`면 Gazebo GUI는 끄고, **OpenCV 카메라 프리뷰**(`/camera/image_raw` 320×180)로 카메라를 보는 것을 권장합니다.

---

## GUI 트러블슈팅

| 증상 | 조치 |
|------|------|
| `Unable to locate package ros-humble-*` (호스트) | **정상** — 호스트에 설치하지 말고 Docker 사용 |
| Gazebo 창 안 뜸 | Docker Desktop 재시작, `echo $DISPLAY` → `:0` 확인 |
| `cannot open display` | Win11 WSLg 업데이트 / Win10은 VcXsrv + DISPLAY 설정 |
| 텍스처 없음 | `build-sim` 후 `sim` 재실행 |
| 표지판 3개가 같아 보임 | 예전 `sign.png` 캐시 — `build-sim` 후 Gazebo 완전 재시작 |
| 표지판 이미지 안 보임 | `prepare_mission_signs.py` 실행 여부 확인, `killall gzserver gzclient` 후 재실행 |
| `control_msgs` 없음 | `./scripts/dev_container.sh init` 먼저 |

---

## 카메라 (320×180, 16:9)

C920e 네이티브는 **1920×1080 (16:9)** 입니다. 팀은 주최측 기본 320×160(2:1, 세로 눌림) 대신 **320×180**을 사용합니다.

| 환경 | 설정 |
|------|------|
| 실기 | `config/vehicle_config.yaml` → `IMAGE_WIDTH: 320`, `IMAGE_HEIGHT: 180` |
| 시뮬 Gazebo | 640×360 렌더 (`limo_dracer_sim.xacro`) |
| 시뮬 출력 | `sim_camera_republish` → 320×180 JPEG |

`git pull` 후 `init` + `build-sim`을 다시 실행하세요.

---

## 실기 vs 시뮬

| | 실기 (D3-G) | Docker Gazebo |
|--|-------------|---------------|
| 카메라 | `camera_node` | `sim_camera_republish` |
| 구동 | `control_node` (I2C) | `sim_control_bridge` |
| inference | 동일 | 동일 |
| `/control` 규약 | steering −1=좌/+1=우, throttle −1=후/+1=전 | **동일** (브릿지가 Gazebo 부호·조향각 변환) |
| E-Stop | `control_node` 래치 | `sim_control_bridge` 래치 (`joystick`) |

역학(가속·최대조향 체감)은 LIMO Ackermann이라 RC 실차와 다를 수 있으나, **토픽·부호·E-Stop 계약은 실차에 맞춤**.

실차 최종 튜닝은 D3-G에서 `./scripts/board_sync.sh` 후 실행.

---

## Ubuntu 22.04 WSL (선택)

22.04 WSL만 **네이티브** Humble 설치도 가능하지만, 팀 통일을 위해 **Docker 사용을 권장**합니다.
