# Gazebo 시뮬 — 팀원 PC 재현 가이드

> **이 문서만 따라하면** `git clone` 한 대로 Docker에서 LIMO + CW 트랙 시뮬을 실행할 수 있습니다.  
> 트러블슈팅·GPU: [simulation.md](./simulation.md) · 패키지 상세: [../src/dracer_sim/README.md](../src/dracer_sim/README.md)

마지막 업데이트: 2026-07-11

---

## 0. 한눈에 보기

| 항목 | 내용 |
|------|------|
| 필요 환경 | Windows 11 + WSL2 + **Docker Desktop** (WSLg 권장) |
| 호스트 ROS | **설치 불필요** (WSL 24.04/26.04도 OK) |
| 로봇 모델 | LIMO Ackermann (`vendor/limo_car`) |
| 트랙 | CW 팀 트랙, 가로 **12 m** |
| 미션 표지판 | 좌회전 · 우회전 · ArUco 정지 마커 (**3종**, 월드에 고정) |
| 카메라 | C920e FOV, **320×180** JPEG (16:9) |
| 시각화 | **Gazebo** (3D). OpenCV 카메라/BEV는 **`view:=none` 기본 OFF** (`cam`/`bev`/`both`로 켬). 자율 창은 **`viz:=lane`** |
| 웹 모니터 | 시뮬 기본 **OFF** |
| **개발 방식** | **컨테이너 1개** (`2026-smh-sim`) + **터미널 2개** (bringup / sim-auto) |
| **노드 구분** | [lane-perception-topic.md §2](./lane-perception-topic.md) ★ 시뮬 vs 실차 인벤토리 |
| **코스 색** | Out=**흰만** · In=**노란 우선** — [lane-occlusion-fork-strategy.md §0](./lane-occlusion-fork-strategy.md) |

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH && chmod +x scripts/*.sh
./scripts/dev_container.sh build
./scripts/dev_container.sh install-gazebo   # 최초 1회
./scripts/dev_container.sh init
./scripts/dev_container.sh sim-up           # 컨테이너 생성 (1회)
# Out 갈림 실험
./scripts/dev_container.sh sim-bringup spawn_pose:=out_fork view:=none
./scripts/dev_container.sh sim-auto route_mode:=out forced_turn:=left viz:=lane
```

### 시뮬에서 쓰는 노드 / 쓰지 않는 노드

| 구분 | 노드 | 설명 |
|------|------|------|
| **자율 코어 (필수)** | `inference_node` | MainPlanner → `/control` (+ 검증 토픽) |
| **시뮬 브리지 (필수)** | `sim_control_bridge`, `sim_camera_republish`, Gazebo·spawn | 실차 camera/control 대용 |
| **안전 (권장)** | `joystick_node` | E-Stop |
| **선택** | `sim_camera_preview` | 로컬 카메라 창 |
| **선택·기본 OFF** | `monitor_node` | 웹 UI. 실차 SSH 관측용. 시뮬 자율에 불필요 |
| **레거시 (실행 금지)** | `lane_control_node` | MainPlanner와 `/control` 충돌 |
| **실차만 (시뮬 ❌)** | `camera_node`, `control_node`, `battery_node` | 하드웨어 |

전체 표: [lane-perception-topic.md §2](./lane-perception-topic.md) · [main-planner.md](./main-planner.md)

---

## 1. 무엇이 시뮬되는가

| 항목 | 내용 |
|------|------|
| 로봇 | LIMO Ackermann (`vendor/limo_car`, ROS2 Humble) |
| 트랙 | CW 팀 트랙 텍스처 (`track_cw_real.png`), 가로 **12 m**, 바닥 z=**0.01 m** |
| 미션 표지판 | 좌/우회전(Ø21 cm 원판) + ArUco(15 cm) — `track_cw.world`에 포함 |
| 인터페이스 | **D-Racer 실기와 동일 토픽** (`src/dracer_sim/config/sim_interface.yaml`) |
| 참고 레포 | `ahns_limo_sim`은 ROS1·라이다 등 **미사용** — 트랙 plane·Ackermann만 참고 |

### 아키텍처

```
Gazebo (LIMO + CW 트랙 + 미션 표지판 3종 + C920e 카메라 640×360)
    │
    ├─ /gazebo/camera/image_raw
    │       └─ sim_camera_republish → /camera/image/compressed (320×180 JPEG)
    │                              → /camera/image_raw (프리뷰 창용)
    │
    ├─ /control ← inference_node (MainPlanner)
    │       └─ sim_control_bridge → /cmd_vel → Ackermann Gazebo plugin
    ├─ /perception/lane, /debug/* ← inference_node (검증용)
    │
    └─ /battery_status ← sim_battery_stub (80% 고정)

카메라/BEV 프리뷰: bringup 기본 **`view:=none`**. 켤 때 `view:=cam|bev|both`.  
자율 인지 창: **`viz:=lane`** → `Lane / Fork Perception` 1개 (`viz:=off|debug|all`).

현재 구조: [main-planner.md](./main-planner.md) · [lane-perception-topic.md](./lane-perception-topic.md) ★
```

### D-Racer 호환 토픽

| 토픽 | 타입 | 용도 |
|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | **inference_node** 입력 (320×180 JPEG) |
| `/camera/image_raw` | `sensor_msgs/Image` | 카메라 프리뷰 창 |
| `/perception/lane` | `lane_msgs/LaneDetections` | 검증·기록 (MainPlanner는 프레임 직접 사용) |
| `/control` | `control_msgs/Control` | throttle (−1=후/+1=전) / steering (−1=좌/+1=우) |
| `/battery_status` | `battery_msgs/Battery` | monitor 스텁 |
| `/joint_states` | `sensor_msgs/JointState` | Gazebo 관절 |
| `/odom` | `nav_msgs/Odometry` | Gazebo Ackermann 플러그인 |

내부 전용: `/cmd_vel` ← `sim_control_bridge`

`/control` 규약(실차와 동일): **steering −1=좌 / +1=우**, **throttle −1=후 / +1=전**.  
`sim_control_bridge`가 Gazebo Ackermann용으로 `angular.z`(조향각 rad, +Z=좌)에 **부호 반전**해 전달하고, E-Stop은 `joystick`에서 실차 `control_node`처럼 래치합니다.  
자율주행: `sim_auto_driving.launch.py` → **`inference_node` 하나**가 `/control`을 발행합니다. `lane_control_node`는 실행하지 마세요.

---

## 2. 사전 조건 (Windows + WSL)

### 2.1 필수 소프트웨어

1. **Windows 11** (WSLg 내장 — Gazebo·카메라 프리뷰 GUI에 유리)
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
- `vendor/limo_car/`는 **레포에 포함**되어 있습니다 (mesh ~100MB, clone 1–2분 더 걸릴 수 있음). 별도 clone 불필요.

---

## 3. 최초 설치 (처음 clone한 팀원)

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
```

| 순서 | 명령 | 소요 | 설명 |
|------|------|------|------|
| 1 | `./scripts/dev_container.sh build` | ~3–8분 | **Docker 이미지** 빌드 (`Dockerfile` → `2026-smh-dev:latest`) |
| 2 | `./scripts/dev_container.sh install-gazebo` | ~5–10분 | 이미지 안에 Gazebo apt 설치 (**최초 1회**) |
| 3 | `./scripts/dev_container.sh init` | ~30초 | D-Racer-Kit clone + `src/` 링크 + 팀 `vehicle_config` |
| 4 | `./scripts/dev_container.sh build-sim` | ~1–2분 | **ROS 워크스페이스** `colcon build` (dracer_sim, inference 등) |
| 5 | `./scripts/dev_container.sh check-gpu` | 수초 | GPU 렌더링 확인 (선택, 권장) |

> **「빌드」가 두 가지입니다** — 헷갈리기 쉬워서 구분합니다.  
> - **이미지 빌드** (`build`): `Dockerfile` 수정·최초 clone 시만. 시뮬을 매일 켤 때 **안 함**.  
> - **워크스페이스 빌드** (`build-sim` / `colcon build`): Python·launch·URDF 등 **레포 코드**를 바꾼 뒤. `sim-bringup`이 **자동**으로 해 줌.

> `sim-bringup` 실행 시 Gazebo가 없으면 **자동 설치**를 시도하지만, 네트워크 안정을 위해 `install-gazebo`를 먼저 실행하는 것을 권장합니다.

### init이 하는 일

- `external/D-Racer-Kit` clone (`release/v1.0.0`)
- `src/camera`, `src/control`, `src/inference` 등 심볼릭 링크
- `vendor/limo_car` → `src/limo_car` 링크
- **`config/vehicle_config.yaml`** → `src/config/vehicle_config.yaml` (팀 카메라 320×180)

---

## 4. 일상 개발 워크플로 (컨테이너 1개 + 터미널 2개)

### 4.0 왜 이렇게 하나

예전에는 `docker compose run --rm`으로 Gazebo를 띄웠기 때문에 **launch를 끄면 컨테이너도 같이 사라졌습니다.**  
모듈을 고치고 inference를 다시 돌릴 때마다 전체를 재시작해야 했고, 컨테이너가 2개면 `network_mode` 차이로 ROS 토픽이 안 맞을 수도 있었습니다.

지금은 다음 원칙으로 통일합니다.

| 원칙 | 내용 |
|------|------|
| **컨테이너 1개** | 이름 고정 `2026-smh-sim`, `network_mode: host` |
| **터미널 1** | Gazebo·브리지·카메라 프리뷰 (`sim-bringup`) — **유지** |
| **터미널 2** | 자율 스택 (`sim-auto` / `sim_auto_stack`) + **인지 오버레이** — 껐다 켜기 |
| **빌드·검증** | `sim-up` 중이면 `build-sim` / `check`도 **같은 컨테이너** (없을 때만 일회성 dev) |
| **launch만 끔** | T1 Ctrl+C → Gazebo 종료 / T2 Ctrl+C → 자율만 종료, **컨테이너 유지** |
| **셸 진입** | `docker exec -it 2026-smh-sim bash` 한 줄이면 충분 (별도 sh 래퍼 없음) |

```
┌─────────────────────────────────────────────────────────────┐
│  Docker 컨테이너: 2026-smh-sim  (sleep infinity, 백그라운드) │
│  network_mode: host  →  WSL과 같은 ROS 도메인               │
├──────────────────────────┬──────────────────────────────────┤
│  터미널 1 (호스트 WSL)    │  터미널 2 (호스트 WSL)            │
│  sim-bringup             │  sim-auto route_mode:=out       │
│  ├ build-sim (자동)      │  ├ joystick + inference_node    │
│  ├ Gazebo + 트랙         │  └ Lane/Fork Perception 창      │
│  ├ 토픽 브리지           │  Ctrl+C → 자율만 종료 (Gazebo 유지)│
│  └ 카메라 프리뷰 창      │                                   │
│  Ctrl+C → Gazebo만 종료  │  (또는 docker exec 셸에서 직접)  │
└──────────────────────────┴──────────────────────────────────┘
         sim-down 으로 컨테이너 전체 삭제 (하루 작업 끝)
```

### 4.1 컨테이너 생성·종료

고정 컨테이너 이름: **`2026-smh-sim`** (`SMH_SIM_CONTAINER` 환경변수로 변경 가능)

```bash
# 컨테이너 생성·시작 (PC 켜고 처음 1회, 또는 sim-down 이후)
./scripts/dev_container.sh sim-up

# 하루 작업 끝 — 컨테이너 삭제
./scripts/dev_container.sh sim-down
```

`sim-up`은 컨테이너 안에서 `sleep infinity`만 돌립니다. Gazebo는 아직 안 뜹니다.

직접 명령·launch·수동 워크플로 전체: **[§4.8 직접 명령어 치트시트](#48-직접-명령어-치트시트-스크립트-없이)**

```bash
# 생성
docker compose run -d --name 2026-smh-sim sim sleep infinity

# 중지됐을 때 재시작
docker start 2026-smh-sim

# 삭제
docker rm -f 2026-smh-sim
```

> `sim-bringup`을 처음 실행할 때 컨테이너가 없으면 **자동으로 `sim-up`까지** 합니다.  
> 매일 `sim-up`을 먼저 칠 필요는 없지만, 터미널 역할을 나누려면 `sim-up` → `sim-bringup` 순서가 읽기 쉽습니다.

### 4.2 터미널 1 — 시뮬 실행 (`sim-bringup`)

**역할**: Gazebo 월드, D-Racer 토픽 브리지, 카메라 프리뷰를 켜고 끕니다.

```bash
./scripts/dev_container.sh sim-bringup
```

내부적으로 다음을 **한 번에** 수행합니다.

1. `sim-up` (컨테이너 없으면 생성)
2. **워크스페이스 빌드** (`build-sim`) — `init_workspace` + `colcon build` ← **Dockerfile 빌드 아님**
3. `ros2 launch dracer_sim sim_bringup.launch.py`

**정상이면 다음이 보입니다:**

| 창 | 내용 |
|----|------|
| **Gazebo** | CW 트랙 바닥 + LIMO + **좌/우회전·ArUco 표지판** |
| **D-Racer Camera** | 320×180 영상, 기본 창 **640×360** (16:9, 2배) |

**중지 방법**

- **Ctrl+C** → launch 프로세스만 종료. **`2026-smh-sim` 컨테이너는 살아 있음**
- 다시 켤 때: `sim-bringup` 재실행 **또는** 컨테이너 안에서 `ros2 launch dracer_sim sim_bringup.launch.py` ([§4.8](#48-직접-명령어-치트시트-스크립트-없이))
- 월드·스폰 위치를 바꿨거나 Gazebo가 이상하면 터미널1에서 launch 재실행
- 터미널2의 inference 셸은 그대로 두고 시뮬만 재시작 가능

창 크기 변경 (16:9 유지):

```bash
./scripts/dev_container.sh sim-bringup camera_view_width:=800 camera_view_height:=450
./scripts/dev_container.sh sim-bringup use_camera_view:=false   # 카메라 프리뷰 끔
./scripts/dev_container.sh sim-bringup use_bev_view:=false      # Metric IPM BEV 창만 끔
```

### 4.3 터미널 2 — 코드 개발·실행 (`docker exec`)

**역할**: `src/inference/modules/` 수정, 빌드, **인지+제어** 실행.

```bash
docker exec -it 2026-smh-sim bash
```

컨테이너에 들어간 뒤 ROS 환경을 source합니다 (매 셸마다 1회).

```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash    # 또는 cd /workspace && source install/setup.bash
```

자율주행 (**권장** — Gazebo는 T1에 두고 자율만 토글):

```bash
# 호스트에서
./scripts/dev_container.sh sim-auto route_mode:=out forced_turn:=left viz:=lane
# viz:=off|lane|debug|all  · forced_turn 시 표지 무시
# 또는 컨테이너 안
ros2 launch dracer_sim sim_auto_stack.launch.py route_mode:=out
```

`Lane / Fork Perception` 창이 같이 뜹니다. 오버레이가 페인트에 맞고 차만 늦게 따라가면 **제어 반응**, 오버레이가 이미 페인트에서 벗어나면 **인지** 이슈입니다. (키 `0`=all, `1`=left, `2`=right)

올인원 (bringup+자율, **끄면 Gazebo도 같이 종료**):

```bash
ros2 launch dracer_sim sim_auto_driving.launch.py route_mode:=out spawn_pose:=out_fork
```

인지 노드만 (토픽 디버그용 — **차는 안 움직임**):

```bash
ros2 run inference inference_node --ros-args -p use_sim_time:=true
```

인지 오버레이만 (주행 중 별도 창):

```bash
ros2 run inference lane_preview_node --ros-args -p use_sim_time:=true
```

또는 실차용 launch를 시뮬에서 쓸 때(하드웨어 노드 포함 — 보통 비권장):

```bash
ros2 launch inference auto_driving.launch.py use_sim_time:=true
```

**주의**

- 터미널2에서 **`ros2 launch dracer_sim sim_bringup.launch.py`를 또 실행하지 마세요.** Gazebo·카메라 창이 **중복**으로 뜹니다. 시뮬은 **터미널1만** 담당합니다.
- `exit`로 셸만 빠져나옵니다. 컨테이너는 계속 실행 중입니다.
- 호스트(WSL)에서 코드를 편집해도 됩니다. `/workspace`는 레포가 **볼륨 마운트**되어 있어 컨테이너 안에서 바로 반영됩니다.

### 4.4 일상 개발 루프 (코드 수정 → 빌드 → 재실행)

| 단계 | 터미널 | 할 일 |
|------|--------|--------|
| 1 | 1 | `sim-bringup` 으로 시뮬 켜기 (이미 켜져 있으면 생략) |
| 2 | 2 | `docker exec -it 2026-smh-sim bash` |
| 3 | 호스트 | Cursor/에디터로 `src/inference/modules/*.py` 수정 |
| 4 | 2 | 빌드 후 inference 재실행 (아래 참고) |
| 5 | 1 | Gazebo에서 주행·카메라·표지판 확인 |
| 6 | 호스트 (선택) | `./scripts/dev_container.sh verify-sim` |

**터미널2에서 빌드** — `2026-smh-sim`이 떠 있는 동안 (**ROS 워크스페이스** 빌드, Dockerfile 아님):

```bash
# 컨테이너 안에서
cd /workspace
colcon build --symlink-install --packages-select inference
source install/setup.bash
ros2 run inference inference_node --ros-args -p use_sim_time:=true
```

**호스트에서 빌드** (같은 컨테이너에 반영):

```bash
./scripts/dev_container.sh build-sim
# 이후 터미널2에서 inference만 다시 실행
```

inference를 멈출 때: 터미널2에서 **Ctrl+C**.

### 4.5 자율주행 한 번에 (`sim`)

시뮬 + inference를 **한 터미널**에서 통합 테스트할 때:

```bash
./scripts/dev_container.sh sim
```

`sim_bringup` + `inference_node` + 조이스틱 노드(캘리브레이션 off).  
**모듈 개발 중**에는 터미널 2개 방식(§4.2–4.4)이 더 편합니다.

### 4.6 수동 주행 (USB 조이스틱)

```bash
./scripts/dev_container.sh sim-manual
```

WSL USB 패스스루 설정이 필요할 수 있습니다. 조이스틱 없이 테스트 (터미널2 또는 호스트):

```bash
ros2 topic pub /control control_msgs/msg/Control "{steering: 0.0, throttle: 0.3}" -r 10
```

### 4.7 launch 옵션

`sim-bringup` 뒤에 ROS launch 인자를 그대로 전달할 수 있습니다.

```bash
./scripts/dev_container.sh sim-bringup headless:=true
./scripts/dev_container.sh sim-bringup use_camera_view:=false
./scripts/dev_container.sh sim-bringup use_monitor:=true   # 웹 모니터 (Docker Flask 이슈 가능)
./scripts/dev_container.sh sim-bringup spawn_pose:=out_fork
./scripts/dev_container.sh sim-bringup spawn_pose:=custom spawn_x:=0.0 spawn_y:=-3.6 spawn_yaw:=1.57
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `headless` | `false` | `true`면 Gazebo 3D 창 끔 (물리·카메라는 동작) |
| `view` | `none` | OpenCV 창: `none` \| `cam` \| `bev` \| `both` (우선) |
| `use_camera_view` | `false` | 레거시 — `view:=` 없을 때만 |
| `use_bev_view` | `false` | 레거시 — `view:=` 없을 때만 |
| `bev_view_scale` | `2.0` | BEV 창 배율 |
| `camera_view_width` | `640` | 프리뷰 창 가로 (16:9) |
| `camera_view_height` | `360` | 프리뷰 창 세로 |
| `use_monitor` | `false` | D-Racer **웹 모니터**. 실차 SSH 관측용. 시뮬 자율주행엔 **불필요**(기본 OFF). 켜도 로직은 동작 |
| `robot` | `limo` | `dracer` = 경량 박스 모델 |
| `spawn_pose` | `start` | 미션 구간 프리셋 ([§5.0](#50-미션-구간-스폰-pose-limo)) |
| `spawn_x/y/z/yaw` | (custom용) | `spawn_pose:=custom`일 때만 사용 |

### 4.8 직접 명령어 치트시트 (스크립트 없이)

`./scripts/dev_container.sh` 없이 **docker / ros2 명령만**으로 같은 작업을 할 때의 참고표입니다.  
레포 루트(`2026-SMH/`)에서 실행한다고 가정합니다.

#### 「빌드」 종류 — Docker 이미지 vs ROS 워크스페이스

팀 문서에서 **빌드**는 맥락에 따라 다릅니다. 시뮬 launch 직전에 말하는 빌드는 **아래 ③**입니다.

| 구분 | 명령 예 | 무엇을 하나 | 언제 하나 |
|------|---------|-------------|-----------|
| ① **Docker 이미지** | `./scripts/dev_container.sh build`<br>`docker compose build` | `Dockerfile` → `2026-smh-dev:latest` 이미지 생성 | **최초 clone**, `Dockerfile` / 베이스 apt 변경 시 |
| ② **Gazebo 설치** | `./scripts/dev_container.sh install-gazebo` | 이미지 **안에** Gazebo apt 패키지 추가 | **최초 1회** (이미지에 Gazebo 없을 때) |
| ③ **ROS 워크스페이스** | `./scripts/dev_container.sh build-sim`<br>`colcon build …` | `src/` 코드 → `build/`, `install/` | **코드 수정 후**, 또는 최초 `init` 다음 |
| — launch만 | `ros2 launch dracer_sim …` | Gazebo·노드 실행 | ③이 끝난 뒤 (또는 코드 안 바꿨으면 ③ 생략) |

- `sim-bringup` = ③ 워크스페이스 빌드 + launch **한꺼번에**
- `ros2 launch`만 직접 치면 = ③을 **직접** 해야 함 (`Dockerfile` 다시 빌드할 필요 **없음**)
- `modules/lane_detection.py` 등 **Python만** 고쳤으면 ③에서 `inference`만 다시 빌드해도 됨

#### `ros2 launch`만 직접 할 때 — 빌드를 언제 다시 하나

| 상황 | 워크스페이스 빌드(③) 필요? | 할 일 |
|------|---------------------------|--------|
| 시뮬만 껐다 켬 (코드 변경 없음) | **아니오** | `source install/setup.bash` 후 `ros2 launch …` |
| `src/inference/` Python 수정 | **예** (inference만) | `colcon build --packages-select inference` → `source install/setup.bash` |
| `src/dracer_sim/` (launch, URDF, world) 수정 | **예** (dracer_sim) | `colcon build --packages-select dracer_sim` → `source install/setup.bash` |
| 처음 clone / `init` 안 함 / `install/` 없음 | **예** (전체) | 아래 §4.8 「최초 워크스페이스 빌드」 전체 |
| `Dockerfile` 수정 | ① 이미지 빌드 | `./scripts/dev_container.sh build` (드묾) |

#### 스크립트 ↔ 직접 명령 대응

| 하고 싶은 일 | 스크립트 | 직접 명령 |
|--------------|----------|-----------|
| **Docker 이미지** 빌드 | `build` | `docker compose build` |
| Gazebo apt 설치 | `install-gazebo` | (수동 비권장 — 스크립트 사용) |
| 시뮬 컨테이너 생성 | `sim-up` | `docker compose run -d --name 2026-smh-sim sim sleep infinity` |
| 컨테이너 상태 확인 | — | `docker ps -a --filter name=2026-smh-sim` |
| 중지된 컨테이너 시작 | (sim-up이 자동) | `docker start 2026-smh-sim` |
| 컨테이너 셸 진입 | — | `docker exec -it 2026-smh-sim bash` |
| 워크스페이스 링크 | `init` | 컨테이너 안 `./scripts/init_workspace.sh` |
| **ROS 워크스페이스** 빌드 | `build-sim` | 컨테이너 안 `colcon build …` (아래 §3) |
| Gazebo 시뮬 실행 | `sim-bringup` | 컨테이너 안 `ros2 launch dracer_sim sim_bringup.launch.py` |
| 시뮬만 끄기 | Ctrl+C (launch 터미널) | launch 실행 중인 셸에서 **Ctrl+C** |
| 컨테이너 삭제 | `sim-down` | `docker rm -f 2026-smh-sim` |
| 토픽 검증 | `verify-sim` | 호스트 `./scripts/verify_sim.sh` (편의 스크립트) |

> `sim-bringup`을 다시 실행해도 **`2026-smh-sim`은 새로 만들지 않습니다.**  
> 실행 중이면 그대로 쓰고, 중지됐으면 `docker start`만 합니다.

#### 1) 컨테이너 생성 (최초 1회)

```bash
cd ~/projects/2026-seame-hackathon/2026-SMH   # 본인 clone 경로

# WSLg GUI (보통 자동)
export DISPLAY=${DISPLAY:-:0}

docker compose run -d --name 2026-smh-sim sim sleep infinity
```

확인:

```bash
docker ps --filter name=2026-smh-sim
# STATUS 가 Up 이면 OK
```

이미 있으면 `Error: container name already in use` → **삭제하지 말고** `docker start 2026-smh-sim` 또는 그대로 `docker exec` 사용.

#### 2) 컨테이너 진입

```bash
docker exec -it 2026-smh-sim bash
```

프롬프트가 `root@…:/workspace#` 형태면 성공. 이후 명령은 **컨테이너 안**에서 실행합니다.

ROS 환경 (셸 열 때마다 1회):

```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
```

#### 3) 워크스페이스 빌드 (컨테이너 안) — `sim-bringup`이 하는 「빌드」

**Dockerfile / `docker compose build`가 아닙니다.** 레포의 ROS 패키지를 `colcon`으로 컴파일·설치하는 단계입니다.  
`sim-bringup` 없이 `ros2 launch`만 직접 실행할 때는 **launch 전에** 이 단계를 직접 해야 합니다.

**최초 1회 (전체)** — `build-sim`과 동일:

```bash
cd /workspace
./scripts/init_workspace.sh
python3 scripts/prepare_mission_signs.py
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to dracer_sim limo_car inference monitor joystick topst_utils opencv
source install/setup.bash
```

**코드 수정 후 (일상)** — 바꾼 패키지만:

```bash
cd /workspace
source /opt/ros/humble/setup.bash

# inference 모듈만 고쳤을 때 (가장 흔함)
colcon build --symlink-install --packages-select inference
source install/setup.bash

# dracer_sim (launch, URDF, world) 고쳤을 때
colcon build --symlink-install --packages-select dracer_sim
source install/setup.bash
```

**호스트에서** 같은 워크스페이스 빌드 (`2026-smh-sim` 실행 중이면 그 컨테이너 안에서 빌드됨):

```bash
./scripts/dev_container.sh build-sim
```

빌드가 끝난 뒤에야 `ros2 launch` / `ros2 run`이 최신 코드를 씁니다.

#### 4) 시뮬 launch 실행 (컨테이너 안)

launch 파일 위치: `src/dracer_sim/launch/`

| launch 파일 | 용도 |
|-------------|------|
| `sim_bringup.launch.py` | Gazebo + 트랙 + 브리지 + 카메라 프리뷰 (**기본**) |
| `sim_auto_stack.launch.py` | **bringup 없이** joystick + inference + 인지 오버레이 (실험 토글용) |
| `sim_auto_driving.launch.py` | bringup + auto stack 올인원 |
| `sim_manual_driving.launch.py` | bringup + 조이스틱 수동주행 |

**기본 시뮬** (터미널 1 — 이 셸은 launch 전용으로 두는 것을 권장):

```bash
ros2 launch dracer_sim sim_bringup.launch.py
```

옵션 예:

```bash
ros2 launch dracer_sim sim_bringup.launch.py headless:=true
ros2 launch dracer_sim sim_bringup.launch.py use_camera_view:=false
ros2 launch dracer_sim sim_bringup.launch.py camera_view_width:=800 camera_view_height:=450
```

**종료**: launch가 돌아가는 터미널에서 **Ctrl+C** → Gazebo·브리지만 종료, **`2026-smh-sim` 컨테이너는 유지**.

**다시 켜기** (코드 변경 없음 → 워크스페이스 빌드 생략):

```bash
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash
ros2 launch dracer_sim sim_bringup.launch.py
```

**다시 켜기** (코드 수정 있음 → launch **전에** colcon, [위 §3](#3-워크스페이스-빌드-컨테이너-안--sim-bringup이-하는-빌드) 참고):

```bash
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash
cd /workspace
colcon build --symlink-install --packages-select inference   # 예시
source install/setup.bash
ros2 launch dracer_sim sim_bringup.launch.py
```

#### 5) 터미널 2 — 인지+제어 (컨테이너 안, 별도 셸)

시뮬 bringup이 **이미 터미널 1에서 돌아가는 중**일 때, **새 WSL 터미널**에서:

```bash
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash
ros2 launch dracer_sim sim_auto_driving.launch.py
# inference_node(MainPlanner) (STEER_TRIM=0, use_sim_time=true)
```

확인:

```bash
ros2 topic echo /perception/lane --once
ros2 topic echo /control --once
```

#### 6) 컨테이너 삭제 (하루 작업 끝)

```bash
docker rm -f 2026-smh-sim
```

#### 전체 수동 워크플로 예시 (터미널 2개)

```bash
# === 호스트 WSL — 최초 1회 ===
cd 2026-SMH
docker compose run -d --name 2026-smh-sim sim sleep infinity

# === 터미널 1 — 시뮬 ===
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash
cd /workspace && ./scripts/init_workspace.sh
colcon build --symlink-install --packages-up-to dracer_sim inference
source install/setup.bash
ros2 launch dracer_sim sim_bringup.launch.py
# Ctrl+C 로 시뮬만 끔

# === 터미널 2 — inference (시뮬 켜진 상태) ===
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash
ros2 run inference inference_node --ros-args -p use_sim_time:=true

# === 호스트 — 작업 끝 ===
docker rm -f 2026-smh-sim
```

---

## 5. 미션 표지판 (갈림길 · ArUco)

대회 미션 검증용 표지판 **3종**이 기본 월드에 포함됩니다. **별도 맵 파일·launch 인자 없이** `sim-bringup` / `sim`만 실행하면 함께 로드됩니다.

### 5.0 미션 구간 스폰 pose (LIMO)

미션 구간별로 LIMO를 바로 스폰하려면 `spawn_pose:=…` 를 사용한다.  
SSOT: [`src/dracer_sim/config/spawn_poses.yaml`](../src/dracer_sim/config/spawn_poses.yaml)

```bash
./scripts/dev_container.sh sim-bringup spawn_pose:=start              # 기본 출발점
./scripts/dev_container.sh sim-bringup spawn_pose:=inout_fork
./scripts/dev_container.sh sim-bringup spawn_pose:=in_roundabout_entry
./scripts/dev_container.sh sim-bringup spawn_pose:=obstacle
./scripts/dev_container.sh sim spawn_pose:=out_fork route_mode:=out
# 수동 좌표
./scripts/dev_container.sh sim-bringup spawn_pose:=custom spawn_x:=0.0 spawn_y:=-3.6 spawn_yaw:=1.57
```

| `spawn_pose` | 구간 | x | y | yaw (rad) |
|--------------|------|---|---|-----------|
| `start` | 출발점 (**기본**) | 2.6 | −3.92 | −π |
| `inout_fork` | In/Out 코스 분기 | −0.3 | −3.92 | −π |
| `in_roundabout_entry` | In · 회전교차로 진입 직전 | −1.97 | −2.15 | π/2 |
| `in_roundabout_exit` | In · **탈출 분기** (유지 vs 탈출) | −0.9 | 1.39 | 0.15 |
| `in_out_merge` | In → Out 합류 | 0.45 | 2.45 | π/2 |
| `out_fork` | Out · **갈림** (유도선 없는 진짜 갈림) | −4.4 | 3.72 | 0 |
| `out_fork_merge_left` | Out 갈림 합류 (왼쪽 갈래) | −1.14 | 4.05 | 0 |
| `out_fork_merge_right` | Out 갈림 합류 (오른쪽 갈래) | −1.14 | 3.38 | 0 |
| `out_in_merge` | Out → In 합류 | 0 | 3.71 | 0 |
| `obstacle` | 동적 장애물 구간 | 4.7 | 2.97 | −1.13 |
| `custom` | 수동 | `spawn_x/y/z/yaw` | | |

좌표·yaw를 고치면 **yaml만** 수정한 뒤 `sim-bringup`을 다시 실행하면 된다 (월드 파일 수정 불필요).

**런타임 텔레포트 (Gazebo 재기동 없이):** bringup이 떠 있는 동안 다른 터미널에서 프리셋으로 옮긴다.

```bash
./scripts/dev_container.sh teleport --list
./scripts/dev_container.sh teleport in_roundabout_exit
./scripts/dev_container.sh teleport out_fork
./scripts/dev_container.sh teleport custom --x 0.0 --y -3.6 --yaw 1.57
# 컨테이너 안:
#   python3 scripts/teleport_spawn_pose.py in_roundabout_exit
```

`/set_entity_state`로 `limo`(기본) 모델 pose·twist를 설정한다. `robot:=dracer`면 `--entity dracer_sim`.

### 5.1 무엇이 배치되나

| Gazebo 모델 | 대회 의미 | 실물 크기 | 판자 |
|-------------|-----------|-----------|------|
| `turn_sign_left` | 갈림길 좌회전 | 파란 원 **Ø 20 cm** | 흰 원 **Ø 21 cm** |
| `turn_sign_right` | 갈림길 우회전 | 파란 원 **Ø 20 cm** | 흰 원 **Ø 21 cm** |
| `aruco_stop_sign` | 동적 장애물 정지 | 마커 **15 cm** | 흰 사각 **18 cm** (DICT_6X6_50 **ID 3**) |

### 5.2 현재 위치·방향 (2026-07-11)

좌표는 Gazebo world 기준 (트랙 중심 원점, 가로 12 m).

| 표지판 | x | y | 링크 중심 z | yaw | 향함 |
|--------|---|---|-------------|-----|------|
| 좌회전 | -3.14 | 3.71 | 0.215 | -90° | **-X** (옆) |
| 우회전 | -3.00 | 3.71 | 0.215 | -90° | **-X** |
| ArUco | 4.5 | 0.0 | 0.20 | +180° | **+Y** |

> yaw는 위에서 내려다본 기준. 기본 모델은 **-Y** 면에 텍스처가 붙어 있음.

### 5.3 높이 계산 (아랫선 기준)

트랙 텍스처 평면(`track_plane`)은 **z = 0.01 m** (지면 z=0보다 1 cm 위).  
**10 cm는 트랙 바닥 높이가 아니라**, 트랙 위로 띄우는 **여유 높이**입니다.

```
표지판 아랫선 z = track_surface_z + clearance
                 = 0.01 + 0.10 = 0.11 m
```

| 표지판 | world pose의 z (링크 **중심**) | 계산 |
|--------|-------------------------------|------|
| 좌/우회전 | **0.215** | 0.11 + 원판 반지름 0.105 |
| ArUco | **0.20** | 0.11 + 판 높이/2 0.09 |

`worlds/track_cw.world`의 `<include><pose>`에 **중심 z**를 넣습니다.  
참고 수치·수식은 `src/dracer_sim/config/mission_signs.yaml`에도 동일하게 적어 두었습니다.

### 5.4 위치·방향 바꾸는 방법

1. Gazebo에서 대략적인 x, y, yaw 확인 (Insert / 좌표 표시)
2. **`worlds/track_cw.world`** 의 해당 `<pose>x y z roll pitch yaw</pose>` 수정
3. **`config/mission_signs.yaml`** 도 같이 갱신 (팀 참고용)
4. `build-sim`은 world 파일만 바꿨다면 생략 가능, **Gazebo 재시작** 필수

yaw 빠른 참고 (기본 텍스처 면 = -Y):

| 목표 방향 | yaw (rad) | yaw (deg) |
|-----------|-----------|-----------|
| -Y (출발 쪽) | 0 | 0° |
| -X (옆) | -π/2 | -90° (시계방향 90°) |
| +Y | π | 180° |
| +X | π/2 | 90° |

### 5.5 이미지·텍스처 (팀원 clone)

외부 `data/` 폴더 **불필요**. 레포 안에 모두 포함됩니다.

```
src/dracer_sim/
├── assets/signs/                    # 원본 PNG (Git)
│   ├── trun_left.png
│   ├── trun_right.png
│   └── ArUco_stop.png
├── models/
│   ├── turn_sign_left/              # SDF + materials/
│   ├── turn_sign_right/
│   └── aruco_stop_sign/
├── worlds/track_cw.world            # ★ 실제 스폰 위치
├── config/mission_signs.yaml        # 크기·좌표 참고
└── (repo root) scripts/prepare_mission_signs.py
```

`./scripts/dev_container.sh build-sim` 시 `prepare_mission_signs.py`가 자동 실행되어:

- 좌/우: Ø21 cm **원형** 흰 배경 + Ø20 cm 표지 (PNG 알파)
- ArUco: 15 cm 마커를 18 cm 흰 사각 판에 합성
- 모델별 **고유 파일명** (`turn_sign_left.png` 등) — Gazebo 전역 캐시 충돌 방지

PNG 원본을 바꿨을 때만:

```bash
python3 scripts/prepare_mission_signs.py
./scripts/dev_container.sh build-sim
```

### 5.6 inference 연동

| 모듈 | 담당 | 시뮬에서 확인 |
|------|------|----------------|
| `modules/traffic_sign.py` | 장원정 | 좌/우 표지판 YOLO |
| `modules/aruco/detector.py` | 안승현 | ArUco ID 검출 |
| `modules/aruco/stop_logic.py` | 박성준 | ID 3 정지 판단 |

```bash
./scripts/dev_container.sh sim
ros2 topic echo /debug/aruco    # ArUco 디버그 (auto launch 시)
```

---

## 6. 동작 검증

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

## 7. `dev_container.sh` 명령 전체

```bash
./scripts/dev_container.sh help
```

| 명령 | 용도 |
|------|------|
| `build` | **Docker 이미지** 빌드 (`Dockerfile`) |
| `install-gazebo` | Gazebo 1회 설치 (이미지 안) |
| `init` | D-Racer-Kit + 링크 (`sim-up` 중이면 같은 컨테이너) |
| `build-inference` | inference만 `colcon build` (같은 컨테이너 우선) |
| `build-sim` | **ROS 워크스페이스** `colcon build` (같은 컨테이너 우선) |
| `check` | CI와 동일 import 검증 (같은 컨테이너 우선) |
| `sim-up` | `2026-smh-sim` 생성·시작 (백그라운드) |
| `sim-down` | 시뮬 컨테이너 삭제 |
| `sim-bringup` | **터미널1**: build-sim + Gazebo only |
| `sim-auto` | **터미널2**: 자율 스택+인지창 (bringup 위에서 토글) |
| `sim` | bringup + auto stack 올인원 |
| `sim-manual` | bringup + 조이스틱 수동주행 |
| `check-gpu` | OpenGL 렌더러 확인 (`2026-smh-sim` 우선) |
| `verify-sim` | 토픽 검증 (bringup 실행 중, 호스트에서) |

**터미널2 셸** (스크립트 없음): `docker exec -it 2026-smh-sim bash`

---

## 8. 레포 구조 (시뮬 관련)

```
2026-SMH/
├── Dockerfile                  # 베이스 이미지 (Gazebo 제외, Mesa + OpenCV 프리뷰)
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
    │   ├── assets/signs/       # 표지판 원본 PNG
    │   ├── models/
    │   │   ├── track_plane/    # CW 트랙 텍스처
    │   │   ├── turn_sign_left/
    │   │   ├── turn_sign_right/
    │   │   └── aruco_stop_sign/
    │   ├── worlds/track_cw.world
    │   ├── config/             # camera, control, mission_signs, sim_interface
    └── inference/              # 팀 자율주행 (시뮬에서 동일 코드 실행)
```

---

## 9. inference 모듈 개발 워크플로

```bash
# 1) 모듈 코드 수정 (예: modules/lane_detection.py)
# 2) PR 전 로컬 검증
./scripts/dev_container.sh check

# 3) 시뮬에서 인지→제어 closed-loop 확인
./scripts/dev_container.sh build-sim
# bringup 후:
ros2 launch dracer_sim sim_auto_driving.launch.py

# 4) merge 후 실차 최종 확인 (D3-G)
./scripts/board_sync.sh
ros2 launch inference auto_driving.launch.py
```

시뮬은 **카메라·토픽·인지→`/control`** 검증용입니다 (실차와 동일 메시지).  
시뮬만 `STEER_TRIM=0`·`use_sim_time=true`. 역학(가속도·선회)은 LIMO라 RC와 다를 수 있음.  
구조: [lane-perception-topic.md](./lane-perception-topic.md) · 기하: [vehicle-geometry.md](./vehicle-geometry.md).

---

## 10. 카메라 설정 (320×180)

| 환경 | 설정 파일 | 해상도 |
|------|-----------|--------|
| 실기 (D3-G) | `config/vehicle_config.yaml` | 320×180 (16:9) |
| 시뮬 Gazebo | `urdf/limo_dracer_sim.xacro` | 640×360 렌더 |
| 시뮬 출력 | `config/camera_republish.yaml` | 320×180 JPEG |

C920e 네이티브는 **1920×1080 (16:9)** 입니다. 주최측 기본 320×160(2:1) 대신 팀은 왜곡 없는 **320×180**을 사용합니다.  
상세: [hardware-camera.md](./hardware-camera.md)

---

## 11. 자주 하는 실수

| 증상 | 원인 | 해결 |
|------|------|------|
| `ros2: command not found` | 호스트 WSL에서 직접 실행 | `./scripts/dev_container.sh sim-bringup` |
| 카메라 프리뷰 검은 화면 | 토픽 미수신 / 구 빌드 | `build-sim` 후 재실행, `verify-sim` |
| Gazebo에 카메라만 떠 있음 | Gazebo가 `package://` mesh 미지원 | `build-sim` 후 재실행 (`file://` URDF spawn) |
| Gazebo 없음 | 베이스 이미지만 빌드 | `install-gazebo` 1회 |
| apt Hash Sum mismatch | WSL Docker 미러 | `build`·`install-gazebo` 재시도 (자동) |
| 렉 심함 | CPU 소프트웨어 렌더링 | `check-gpu`, `headless:=true` |
| monitor_node 죽음 | Docker Flask 버전 충돌 | 시뮬 기본 OFF — OpenCV 카메라 프리뷰 사용 |
| 트랙 텍스처 없음 | `build-sim` 안 함 | `build-sim` 후 재실행 |
| 표지판 3개가 전부 좌회전 | Gazebo `sign.png` 캐시 | `build-sim` + Gazebo 완전 재시작 |
| 표지판 판만 보이고 그림 없음 | plane 뒷면 / 텍스처 미로드 | 최신 `dracer_sim` pull 후 `build-sim`, `killall gzserver gzclient` |
| Gazebo·카메라 창 2세트 | bringup을 두 터미널에서 동시 실행 | 터미널1만 `sim-bringup`, 터미널2는 `docker exec` |
| `control_msgs` 없음 | init 안 함 | `./scripts/dev_container.sh init` |

---

## 12. 일상 워크플로

```bash
git pull
./scripts/dev_container.sh init          # vehicle_config·링크 갱신
./scripts/dev_container.sh build-sim   # dracer_sim / inference 변경 후
./scripts/dev_container.sh sim-bringup # 튜닝·카메라 확인
./scripts/dev_container.sh sim         # inference 통합 테스트
```

실차 최종 확인: D3-G에서 `./scripts/board_sync.sh` → `ros2 launch inference auto_driving.launch.py`

---

## 13. 관련 문서

| 문서 | 내용 |
|------|------|
| [simulation.md](./simulation.md) | GPU·GUI 트러블슈팅 |
| [dev-environment.md](./dev-environment.md) | Docker·CI·보드 워크플로 |
| [hardware-camera.md](./hardware-camera.md) | C920e 스펙 |
| [board-workflow.md](./board-workflow.md) | D3-G 실차 개발 |
| [collaboration.md](./collaboration.md) | 브랜치·PR 규칙 |
