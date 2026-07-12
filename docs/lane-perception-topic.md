# 차선 인지·제어 토픽 구조 (팀 SSOT)

> **필독**: PR 전에 이 문서를 읽고, 담당 모듈이 **어디에 꽂히는지** 확인하세요.  
> 관련: [lane-drive-strategy.md](./lane-drive-strategy.md) · [collaboration.md](./collaboration.md) · [roles.md](./roles.md)

인지(`inference_node`)와 임시 제어(`lane_control_node`)를 **ROS 토픽으로 분리**한다.  
**시뮬(Gazebo)과 실차(D3-G)는 같은 토픽·같은 메시지·같은 planner를 쓴다.** 환경 차이는 launch 파라미터뿐이다.

---

## 0. 한 줄 요약

```
카메라 → inference_node(인지만) → /perception/lane
      → lane_control_node(조향) → /control → (시뮬 bridge | 실차 control_node)
```

| 하면 안 되는 것 | 올바른 방법 |
|----------------|-------------|
| `inference_node`에서 `/control` 발행 | 제어는 `lane_control_node` (또는 향후 미션 planner) |
| `lane_detection.detect()`가 조향 반환 | 인지는 `LaneDetections`만 (polyline m) |
| planner가 원태 dataclass / msg를 직접 import | `types.LaneDetections` + `lane_adapters` |
| `ros2 run inference inference_node`만으로 자율주행 | 인지+제어 **둘 다** launch (아래 §4) |
| 사다리꼴 `warp_bev`를 런타임에 복구 | BEV SSOT = **Metric IPM** (`config/lane_vision.yaml`) |

---

## 1. 런타임 노드 구조 (토픽 분리)

```
/camera/image/compressed
        │
        ▼
┌──── inference_node (인지 전용) ────┐
│  lane_detection.detect() 직접 호출 │
│  aruco_detection.detect()          │
│  ※ pipeline.run_perception 우회    │
└───┬───────────────────────┬────────┘
    │                       │
    ▼                       ▼
/perception/lane      /debug/aruco
(lane_msgs)           (std_msgs/String)
    │
    ▼
┌──── lane_control_node (임시 판제) ─┐
│  detections_from_msg()             │
│  lane_planner.step() → LaneResult  │
│  lane_timeout → throttle=0         │
└───────────────┬────────────────────┘
                ▼
            /control
                │
        ┌───────┴────────┐
        ▼                ▼
  sim_control_bridge   control_node (실차)
  → /cmd_vel           → PCA9685 액추에이터
```

노드 전체 목록(필수/선택/`monitor`): **§2**

| 노드 | 구독 | 발행 | 환경 |
|------|------|------|------|
| `inference_node` | `/camera/image/compressed` | `/perception/lane`, `/debug/aruco` | 공통 |
| `lane_control_node` | `/perception/lane` | `/control` | 공통 |
| `sim_control_bridge` | `/control` | `/cmd_vel` | **시뮬만** |
| `control_node` | `/control` | (하드웨어) | **실차만** |
| `monitor_node` | 카메라 등 | 웹 UI | 실차 관측용(선택) · 시뮬 기본 OFF |

### 타입 3층 (이름만 비슷 — 헷갈리지 말 것)

| 층 | 위치 | 용도 |
|----|------|------|
| A. 원태 모듈 dataclass | `modules/lane_detection.py` | `detect()` 내부·반환 |
| B. ROS msg | `lane_msgs/LaneDetections` | `/perception/lane` 와이어 |
| C. 팀 SSOT | `inference.types.LaneDetections` | **planner·테스트·양서준 공통 입력** |

변환:

- A → C: `inference.lane_adapters.detections_from_module`
- B → C: `inference.lane_adapters.detections_from_msg`
- A → B: `inference_node.publish_lane_detections` (기존)

**판제 코드는 C만 사용하세요.** A/B를 planner에 직접 넣지 마세요.

---

## 2. 노드 인벤토리 — 시뮬 vs 실차

개발할 때 **“이 노드가 어디에 필요한지”** 를 먼저 구분하세요.  
자율주행 로직(인지→조향)은 공통이고, **카메라·액추에이터·웹 모니터**만 환경이 갈립니다.

### 2.1 한눈에 보기

| 노드 | 시뮬 | 실차(D3-G) | 역할 |
|------|:----:|:----------:|------|
| `inference_node` | **필수** | **필수** | 카메라 → `/perception/lane`, `/debug/aruco` |
| `lane_control_node` | **필수** | **필수** | `/perception/lane` → `/control` |
| `joystick_node` | 권장 | 권장 | 게임패드·**E-Stop** (없으면 비상 정지 불가) |
| `sim_control_bridge` | **필수** | ❌ | `/control` → Gazebo `/cmd_vel` |
| `sim_camera_republish` | **필수** | ❌ | Gazebo 카메라 → `/camera/image/compressed` |
| `sim_battery_stub` | 권장 | ❌ | `/battery_status` 스텁 (실차 `battery_node` 대용) |
| `robot_state_publisher` / spawn | **필수** | ❌ | Gazebo 로봇 스폰·TF |
| `sim_camera_preview` | 선택 | ❌ | 로컬 카메라 창 (기본 ON in bringup) |
| `camera_node` | ❌ | **필수** | USB 카메라 |
| `control_node` | ❌ | **필수** | `/control` → PCA9685 조향·스로틀 |
| `battery_node` | ❌ | **필수** | 실배터리 토픽 |
| `monitor_node` | 선택(기본 OFF) | 선택(launch에 포함) | **웹 UI** — SSH/PC에서 보드·영상 확인용 |

범례: **필수** = closed-loop 주행에 필요 · **권장** = 안전/편의 · **선택** = 없어도 차는 움직임 · ❌ = 그 환경에 넣지 않음

### 2.2 공통 — 팀 자율주행 코어 (시뮬·실차 동일)

이 두 노드(+ 조이스틱 E-Stop)만 맞으면 **토픽 계약은 같다.**

```
/camera/image/compressed
        → inference_node → /perception/lane
        → lane_control_node → /control
joystick_node → (E-Stop 래치; 수동 조이스틱 모드는 실차 control 설정)
```

| 노드 | 패키지 | 비고 |
|------|--------|------|
| `inference_node` | `inference` | 인지 전용. `/control` 안 씀 |
| `lane_control_node` | `inference` | 임시 P/EMA. 시뮬은 trim=0 |
| `joystick_node` | `joystick` | 이름 `gamepad_publisher`. E-Stop용 |

`inference_node`만 띄우면 **인지 토픽만** 나오고 차는 안 움직인다.

### 2.3 시뮬 전용 — Gazebo 스택

Launch: `sim_bringup.launch.py` (또는 `sim_auto_driving`이 이를 include).

| 노드 | 필수? | 하는 일 |
|------|-------|---------|
| Gazebo (`gzserver`/`gzclient`) | 필수 | 월드·물리 |
| `robot_state_publisher` | 필수 | URDF / TF |
| `spawn_robot` (`spawn_entity.py`) | 필수 | 모델 스폰 |
| `sim_control_bridge` | 필수 | `/control` → `/cmd_vel` (실차 `control_node` 대용) |
| `sim_camera_republish` | 필수 | 시뮬 카메라 → D-Racer와 같은 `/camera/image/compressed` |
| `sim_battery_stub` | 권장 | 배터리 토픽 스텁 |
| `sim_camera_preview` | 선택 | OpenCV/Qt 프리뷰 창 (`use_camera_view:=false`로 끔) |
| `monitor_node` | **불필요(기본 OFF)** | 아래 §2.5 |

실차 하드웨어 노드(`camera_node`, `control_node`, `battery_node`)를  
시뮬에서 같이 띄우지 마세요. (`inference auto_driving.launch.py`를 시뮬에 그대로 쓰면 하드웨어 노드까지 올라감 → **비권장**)

### 2.4 실차 전용 — D3-G 하드웨어

Launch: `inference/auto_driving.launch.py`

| 노드 | 필수? | 하는 일 |
|------|-------|---------|
| `camera_node` | 필수 | USB → `/camera/image/compressed` |
| `control_node` | 필수 | `/control` → 모터 (시뮬 bridge 대용) |
| `battery_node` | 필수* | 실배터리 (`*`모니터·안전 로그용; 조향 자체엔 불필요할 수 있음) |
| `monitor_node` | 선택 | 웹에서 보드 확인 (§2.5) — 현재 launch에 **포함** |
| `joystick_node` | 권장 | E-Stop |

시뮬 전용 노드(`sim_*`, Gazebo spawn)는 보드에서 실행하지 않음.

### 2.5 `monitor_node` — 언제 쓰나

| | 설명 |
|--|------|
| **용도** | PC/노트북 브라우저로 D-Racer 상태·카메라 확인 (보드에 SSH 연결해 둔 뒤 `http://<보드IP>:5000` 등) |
| **자율주행** | **없어도** `inference`→`lane_control`→액추에이터는 동작 |
| **시뮬** | 기본 **OFF** (`sim_auto_driving` → `use_monitor:=false`). Gazebo 프리뷰·`topic echo`면 충분 |
| **실차** | `auto_driving.launch.py`에 포함되어 있음. 켜 둬도 부담이 크지 않으면 그대로 둬도 됨 |
| **켤 때(시뮬)** | `ros2 launch dracer_sim sim_bringup.launch.py use_monitor:=true` |

정리: **모니터 = 원격 관측용 UI.** 차선/조향 로직 개발에 필수가 아니다.  
시뮬에서 헷갈리면 끄고, 실차에서 웹으로 보고 싶으면 켠다.

### 2.6 Launch ↔ 올라가는 노드

| Launch | 환경 | 포함 노드 (요약) |
|--------|------|------------------|
| `dracer_sim/sim_bringup.launch.py` | 시뮬 | Gazebo + bridge + camera republish + control bridge (+ preview). **monitor 기본 OFF** |
| `dracer_sim/sim_auto_driving.launch.py` | 시뮬 | bringup + `joystick` + `inference_node` + `lane_control_node` |
| `dracer_sim/sim_manual_driving.launch.py` | 시뮬 | bringup + 조이스틱 수동 (자율 없음) |
| `inference/auto_driving.launch.py` | **실차** | camera + control + joystick + battery + **monitor** + inference + lane_control |

권장 명령은 §5.

---

## 3. 시뮬 vs 실차 — 토픽·설정 차이

| 항목 | 시뮬 | 실차 |
|------|------|------|
| 토픽 이름 | 동일 (`/perception/lane`, `/control`) | 동일 |
| 메시지 | `lane_msgs`, `control_msgs` | 동일 |
| BEV | Metric IPM (`lane_vision.yaml`) | 동일 YAML |
| 제어 게인 | `config/lane_control.yaml` | 동일 파일 (게인 튜닝은 시뮬→실차) |
| Launch | `sim_auto_driving.launch.py` | `auto_driving.launch.py` |
| `use_sim_time` | **true** | false (기본) |
| `STEER_TRIM` | **강제 0** | `vehicle_config.yaml` |
| `/control` 소비자 | `sim_control_bridge` | `control_node` |
| 카메라 소스 | Gazebo → `sim_camera_republish` | `camera_node` |
| `monitor_node` | 기본 OFF | launch에 포함 (관측용) |
| 시각화 | `LANE_VISUALIZE=1`만 | 보드 headless 기본 OFF |

실차 trim이 시뮬에 들어가면 직진이 한쪽으로 기운다 → **시뮬 launch만** trim=0.

---

## 4. 메시지 계약 (`lane_msgs/LaneDetections`)

좌표계: `base_link`, x 전방, y 왼쪽, z=0, 단위 **m**.

```
std_msgs/Header header               # frame_id="base_link"
LaneMarking[] lanes
bool white_visible / yellow_visible / left_visible / right_visible
float32 white_confidence / … (색·측별)
geometry_msgs/Point32[] white_centerline / yellow_centerline
bool yellow_crossing_line
bool fork_active
RoadBranch[] branches                # branch_id 0=가장 왼쪽
sensor_msgs/Image drivable_area      # BEV mono8
float32 meters_per_pixel
float32 x_forward_max
```

### drivable_area 그리드 → base_link

```
x = x_forward_max - row * meters_per_pixel
y = ((width - 1) / 2 - col) * meters_per_pixel
```

현재 임시 planner(`lane_planner`)는 **흰 L/R polyline**만 사용한다.  
`fork_active` / `branches` / centerline / drivable는 **미션 planner(양서준 등)용으로 이미 전달**된다 — 필드 이름·단위를 바꾸지 말 것.

---

## 5. Launch · 실행 방법

### 시뮬 (권장)

**한 번에** (bringup + 인지 + 제어 — 보통 이걸 씀):

```bash
ros2 launch dracer_sim sim_auto_driving.launch.py
# Gazebo 스택 + joystick + inference_node + lane_control_node
# use_sim_time=true, STEER_TRIM=0, monitor 기본 OFF
```

**월드만 먼저** 켠 뒤, 코어만 따로 (bringup이 이미 떠 있을 때):

```bash
# 터미널1
ros2 launch dracer_sim sim_bringup.launch.py
# monitor 켜려면: use_monitor:=true

# 터미널2 — sim_auto_driving 을 또 치지 말 것 (Gazebo 중복)
ros2 run inference inference_node --ros-args -p use_sim_time:=true
ros2 run inference lane_control_node --ros-args \
  -p use_sim_time:=true -p steer_trim_override:=true -p steer_trim:=0.0
```

확인:

```bash
ros2 node list
# 있어야: inference_node, lane_control_node, sim_control_bridge, …
# 없어야: camera_node, control_node (실차 하드웨어)

ros2 topic echo /perception/lane --once
ros2 topic echo /control --once
```

`inference_node`만 띄우면 **차가 안 움직인다.**

### 실차 (보드)

```bash
cd ~/2026-SMH
./scripts/board_sync.sh
source install/setup.bash
ros2 launch inference auto_driving.launch.py
# camera_node + control_node + battery_node + monitor_node
# + joystick + inference_node + lane_control_node
# STEER_TRIM = vehicle_config
```

웹 모니터: `http://<보드IP>:5000` (또는 `vehicle_config`의 `WEB_HOST`/`WEB_PORT`).  
주행 로직만 보면 되고 모니터가 거슬리면 launch에서 빼도 **조향·인지 동작은 동일**하다.

### 디버그 시각화 (로컬만)

```bash
LANE_VISUALIZE=1 ros2 run inference inference_node
```

보드/SSH에서는 **켜지 마세요** (OpenCV 창·성능).

---

## 6. 인지 파이프라인 요약 (`modules/lane_detection.py`)

- **BEV SSOT:** `config/lane_vision.yaml` → `metric_ipm` + `scripts/vision_tune/metric_ipm.py`
  - 전방 ≈0.22~1.5 m, 횡 ±0.77 m, m/px=0.004
  - 사다리꼴 `bev_roi` / `tune_bev_roi.py`는 **시각 참고 툴만**
- **HSV:** YAML `hsv:` (`tune_hsv.py`로 시뮬·실차 각각 맞춘 뒤 저장)
- **출력:** 조향 없음. `LaneDetections` + fork/branches
- **담당:** 장원태 — 알고리즘·HSV·마스크. **Metric IPM 계약·msg 필드 삭제/개명은 팀장과 합의**

---

## 7. 임시 제어 (`lane_control_node` + `lane_planner`)

- 구독 → `detections_from_msg` → `LanePlanner.step` (P + EMA + rate limit)
- throttle = `cruise_throttle * throttle_scale`
- `/perception/lane`가 `lane_timeout_sec`(기본 0.5s) 이상 없으면 **throttle=0**
- 게인: `config/lane_control.yaml` (`tune_lane_control.py`)
- **양서준 Pure Pursuit / MainPlanner는 이 노드에 넣지 않음** — 별도 브랜치에서 `/perception/lane` 구독 노드로 교체·합류 PR

---

## 8. `pipeline.py`는 무엇인가

| 경로 | 용도 |
|------|------|
| **토픽 분리 (기본 ROS)** | `inference_node` + `lane_control_node` |
| **단프로세스 / pytest / CI** | `pipeline.run_perception` → adapter → planner → `fuse_control` |

런타임 launch는 pipeline을 **쓰지 않는다.**  
모듈 단위 테스트·import 검증용으로만 유지한다.  
`fuse_control` 우선순위(ArUco→빨간불→회전교차로→차선)는 **향후 미션 통합 시** control 쪽으로 옮길 예정.

---

## 9. 팀원별 PR 전 체크리스트

### 공통

- [ ] `main`에서 feature 브랜치 생성
- [ ] [collaboration.md](./collaboration.md) 담당 파일만 수정
- [ ] `/control`을 인지 모듈에서 발행하지 않음
- [ ] 시뮬 테스트 시 **하드웨어 노드**(`camera_node`/`control_node`)를 요구하지 않음
- [ ] 시뮬 또는 보드에서 build 성공

### 장원태 (`lane_detection.py`)

- [ ] 반환은 인지 `LaneDetections` (조향 필드에 의미 있는 값 넣지 않음)
- [ ] BEV는 Metric IPM 유지 (사다리꼴 런타임 복구 금지)
- [ ] `VISUALIZE` 기본 False / `LANE_VISUALIZE`만 사용
- [ ] msg에 이미 있는 필드명·단위(m, base_link) 유지
- [ ] 가능하면 `ros2 topic echo /perception/lane --once`로 발행 확인

### 양서준 (경로·PP / roundabout)

- [ ] 입력은 `types.LaneDetections` 또는 `detections_from_msg`
- [ ] `/perception/lane` 구독 → `/control` 발행 패턴을 따를 것
- [ ] 임시 `lane_control_node`와 **동시에** `/control`을 쓰지 말 것 (launch에서 하나만)
- [ ] `fork_active` / `branches` 활용 시 msg 계약 준수
- [ ] `types.py` / `lane_adapters.py` 변경 필요 시 팀장과 먼저 합의

### 장원정 (`traffic_sign.py`)

- [ ] 현재 ROS 경로에는 아직 미연결 — 모듈 API(`TrafficResult`) 유지
- [ ] `/control` 직접 발행 금지 (통합은 팀장)

### 박성준 / ArUco

- [ ] 인지는 `inference_node`가 `/debug/aruco` 발행
- [ ] 정지를 `/control`에 넣으려면 control 쪽 합류 (임시 노드는 lane-only)

### 안승현 (통합·시뮬)

- [ ] `types` / `adapters` / `lane_control_node` / launch / YAML SSOT
- [ ] 시뮬·실차 **노드 세트**와 launch 파라미터 차이 유지

---

## 10. 빌드 · 패키지 의존

```bash
# lane_msgs 포함 상위까지
colcon build --symlink-install --packages-up-to inference
source install/setup.bash
```

워크스페이스 루트에 `config/lane_vision.yaml`, `scripts/vision_tune/metric_ipm.py`가 있어야  
`lane_detection` import가 성공한다 (`board_sync` / 전체 clone 기준).

단위 테스트 (컨테이너):

```bash
python3 -m pytest src/inference/test/test_lane_planner.py \
                  src/inference/test/test_lane_adapters.py -q
```

---

## 11. 관련 파일 맵

| 경로 | 역할 |
|------|------|
| `modules/lane_detection.py` | 인지 (원태) |
| `modules/lane_planner.py` | 임시 P/EMA 조향 (승현) |
| `types.py` | SSOT dataclass |
| `lane_adapters.py` | module/msg → types |
| `inference_node.py` | 인지 ROS 노드 |
| `lane_control_node.py` | 제어 ROS 노드 |
| `pipeline.py` | 단프로세스/테스트 |
| `config/lane_vision.yaml` | Metric IPM + HSV |
| `config/lane_control.yaml` | planner 게인 |
| `src/lane_msgs/` | ROS 인터페이스 |
| `launch/auto_driving.launch.py` | **실차** 노드 세트 |
| `dracer_sim/.../sim_bringup.launch.py` | **시뮬** 월드·브리지 |
| `dracer_sim/.../sim_auto_driving.launch.py` | **시뮬** 자율 (bringup+코어) |
