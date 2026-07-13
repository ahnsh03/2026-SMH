# 차선 인지·제어 토픽 구조 (팀 SSOT)

> **2026-07-13 구조 변경:** 현재 auto-driving 런타임은
> `inference_node → pipeline.MainPlanner → /control` 단일 프로세스 구조다.
> `/perception/lane`은 RViz·로그·외부 검증을 위해 계속 발행하지만 planner의
> 연산 입력으로 재구독하지 않는다. 이 문서의 `lane_control_node` 설명은
> 이전 토픽 분리 구조의 호환·참고 자료이며, 현재 제어 SSOT는
> [main-planner.md](./main-planner.md)다.

> **필독**: PR 전에 이 문서를 읽고, 담당 모듈이 **어디에 꽂히는지** 확인하세요.  
> 관련: [lane-drive-strategy.md](./lane-drive-strategy.md) · [collaboration.md](./collaboration.md) · [roles.md](./roles.md)

인지 모듈은 독립적인 결과를 반환하고 `MainPlanner`가 같은 프레임에서 판단과
최종 제어를 통합한다. 시뮬과 실차는 같은 planner/YAML을 사용하며 환경 차이는
launch 파라미터와 actuator consumer뿐이다.

---

## 0. 한 줄 요약

```
카메라 → inference_node(MainPlanner 포함) → /control
      └→ /perception/lane, /debug/* (검증용)
```

| 하면 안 되는 것 | 올바른 방법 |
|----------------|-------------|
| `inference_node`와 `lane_control_node` 동시 실행 | `/control` publisher는 `inference_node` 하나만 실행 |
| `lane_detection.detect()`가 조향 반환 | 인지는 `LaneDetections`만 (polyline m) |
| 외부 노드가 모듈 dataclass를 토픽처럼 사용 | ROS msg와 `lane_adapters` 사용 |
| `ros2 run inference inference_node`만으로 자율주행 | 현재는 가능. 시뮬 bridge/실차 control consumer는 별도 필요 |
| 사다리꼴 `warp_bev`를 런타임에 복구 | BEV SSOT = **Metric IPM** (`config/lane_vision.yaml`) |

---

## 1. 현재 런타임 노드 구조

```
/camera/image/compressed
        │
        ▼
┌────────────── inference_node ──────────────┐
│ MainPlanner.step(frame)                    │
│  ├─ lane_detection.detect()               │
│  ├─ traffic_sign.detect()                 │
│  ├─ aruco_detection.detect()              │
│  └─ PP + heading + CTE + mission FSM      │
└───────┬──────────────────────┬─────────────┘
        ▼                      ▼
    /control       /perception/lane, /debug/*
   (실제 제어)             (검증·기록)
        │
        ▼
 (sim bridge | 실차 control_node)
```

노드 전체 목록(필수/선택/`monitor`): **§2**

| 노드 | 구독 | 발행 | 환경 |
|------|------|------|------|
| `inference_node` | `/camera/image/compressed` | `/control`, `/perception/lane`, `/debug/*` | 공통 |
| `lane_control_node` | `/perception/lane` | `/control` | 이전 호환용, auto launch 미사용 |
| `sim_control_bridge` | `/control` | `/cmd_vel` | **시뮬만** |
| `control_node` | `/control` | (하드웨어) | **실차만** |
| `monitor_node` | 카메라 등 | 웹 UI | 실차 관측용(선택) · 시뮬 기본 OFF |

### 타입 3층 (이름만 비슷 — 헷갈리지 말 것)

| 층 | 위치 | 용도 |
|----|------|------|
| A. 원태 모듈 dataclass | `modules/lane_detection.py` | `detect()` 내부·반환 |
| B. ROS msg | `lane_msgs/LaneDetections` | `/perception/lane` 와이어 |
| C. 공통 타입 | `inference.types.LaneDetections` | 토픽 adapter·외부 planner 호환 입력 |

변환:

- A → C: `inference.lane_adapters.detections_from_module`
- B → C: `inference.lane_adapters.detections_from_msg`
- A → B: `inference_node.publish_lane_detections` (기존)

현재 `MainPlanner`는 같은 프로세스에서 A를 직접 받아 불필요한 직렬화와 토픽
지연을 피한다. B는 관측·검증용이고 C/adapter는 외부 소비자와 이전 분리형
planner 호환을 위해 유지한다.

---

## 2. 노드 인벤토리 — 시뮬 vs 실차

개발할 때 **“이 노드가 어디에 필요한지”** 를 먼저 구분하세요.  
자율주행 로직(인지→조향)은 공통이고, **카메라·액추에이터·웹 모니터**만 환경이 갈립니다.

### 2.1 한눈에 보기

| 노드 | 시뮬 | 실차(D3-G) | 역할 |
|------|:----:|:----------:|------|
| `inference_node` | **필수** | **필수** | 인지·판단·제어 → `/control`, 검증 토픽 |
| `lane_control_node` | ❌ | ❌ | 이전 호환용. MainPlanner와 동시 실행 금지 |
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

`inference_node`와 actuator consumer(+ 조이스틱 E-Stop)가 핵심이다.

```
/camera/image/compressed
        → inference_node/MainPlanner → /control
        └→ /perception/lane, /debug/*
joystick_node → (E-Stop 래치; 수동 조이스틱 모드는 실차 control 설정)
```

| 노드 | 패키지 | 비고 |
|------|--------|------|
| `inference_node` | `inference` | 인지·미션 FSM·PP와 최종 `/control` 소유 |
| `lane_control_node` | `inference` | 이전 P/EMA 호환 노드. 현재 실행하지 않음 |
| `joystick_node` | `joystick` | 이름 `gamepad_publisher`. E-Stop용 |

시뮬에서는 `sim_control_bridge`, 실차에서는 `control_node`가 함께 떠 있어야
`/control`이 실제 차량 운동으로 이어진다.

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
| **자율주행** | **없어도** `inference/MainPlanner`→액추에이터는 동작 |
| **시뮬** | 기본 **OFF** (`sim_auto_driving` → `use_monitor:=false`). Gazebo 프리뷰·`topic echo`면 충분 |
| **실차** | `auto_driving.launch.py`에 포함되어 있음. 켜 둬도 부담이 크지 않으면 그대로 둬도 됨 |
| **켤 때(시뮬)** | `ros2 launch dracer_sim sim_bringup.launch.py use_monitor:=true` |

정리: **모니터 = 원격 관측용 UI.** 차선/조향 로직 개발에 필수가 아니다.  
시뮬에서 헷갈리면 끄고, 실차에서 웹으로 보고 싶으면 켠다.

### 2.6 Launch ↔ 올라가는 노드

| Launch | 환경 | 포함 노드 (요약) |
|--------|------|------------------|
| `dracer_sim/sim_bringup.launch.py` | 시뮬 | Gazebo + bridge + camera republish + control bridge (+ preview). **monitor 기본 OFF** |
| `dracer_sim/sim_auto_driving.launch.py` | 시뮬 | bringup + `joystick` + `inference_node` |
| `dracer_sim/sim_manual_driving.launch.py` | 시뮬 | bringup + 조이스틱 수동 (자율 없음) |
| `inference/auto_driving.launch.py` | **실차** | camera + control + joystick + battery + **monitor** + inference |

권장 명령은 §5.

---

## 3. 시뮬 vs 실차 — 토픽·설정 차이

| 항목 | 시뮬 | 실차 |
|------|------|------|
| 토픽 이름 | 동일 (`/perception/lane`, `/control`) | 동일 |
| 메시지 | `lane_msgs`, `control_msgs` | 동일 |
| BEV | Metric IPM (`lane_vision.yaml`) | 동일 YAML |
| 제어 게인 | `config/main_planner.yaml` | 동일 파일 (게인 튜닝은 시뮬→실차) |
| Launch | `sim_auto_driving.launch.py` | `auto_driving.launch.py` |
| `use_sim_time` | **true** | false (기본) |
| `STEER_TRIM` | **강제 0** | `vehicle_config.yaml` |
| `/control` 소비자 | `sim_control_bridge` | `control_node` |
| 카메라 소스 | Gazebo → `sim_camera_republish` | `camera_node` |
| `monitor_node` | 기본 OFF | launch에 포함 (관측용) |
| 시각화 | `LANE_VISUALIZE=off\|control\|on` (기본 off) | 보드 headless 기본 off |

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

현재 임시 planner(`lane_planner`, 레거시)는 **흰 L/R polyline**만 사용했다.  
`MainPlanner`는 `fork_active` / `branches` / centerline / drivable를 **이미 소비**한다 — 필드 이름·단위를 바꾸지 말 것.

---

## 5. Launch · 실행 방법

### 시뮬 (권장)

**한 번에** (bringup + 인지 + MainPlanner — 보통 이걸 씀):

```bash
ros2 launch dracer_sim sim_auto_driving.launch.py
# Gazebo 스택 + joystick + inference_node(MainPlanner)
# use_sim_time=true, STEER_TRIM=0, monitor 기본 OFF
# lane_control_node 는 포함되지 않음 — 실행하지 말 것
```

**월드만 먼저** 켠 뒤, 코어만 따로 (bringup이 이미 떠 있을 때):

```bash
# 터미널1
ros2 launch dracer_sim sim_bringup.launch.py
# monitor 켜려면: use_monitor:=true

# 터미널2 — sim_auto_driving 을 또 치지 말 것 (Gazebo 중복)
ros2 run inference inference_node --ros-args -p use_sim_time:=true
# MainPlanner가 같은 프로세스에서 /control 발행
# lane_control_node 를 추가로 실행하지 말 것
```

확인:

```bash
ros2 node list
# 있어야: inference_node, sim_control_bridge, …
# 없어야: lane_control_node, camera_node, control_node (실차 하드웨어)

ros2 topic echo /perception/lane --once
ros2 topic echo /control --once
```

`inference_node`만 띄우면 `/control`은 나오지만, **시뮬에서 차가 움직이려면** bringup의 `sim_control_bridge`가 필요하다.

### 실차 (보드)

```bash
cd ~/2026-SMH
./scripts/board_sync.sh
source install/setup.bash
ros2 launch inference auto_driving.launch.py
# camera_node + control_node + battery_node + monitor_node
# + joystick + inference_node(MainPlanner)
# STEER_TRIM = vehicle_config
# lane_control_node 미포함
```

웹 모니터: `http://<보드IP>:5000` (또는 `vehicle_config`의 `WEB_HOST`/`WEB_PORT`).  
주행 로직만 보면 되고 모니터가 거슬리면 launch에서 빼도 **조향·인지 동작은 동일**하다.

### 디버그 시각화 (로컬만 · DISPLAY 필요)

**권장 (Gazebo 중복 방지):** bringup만 켠 뒤 모드 튜너로 검증한다.
`sim_auto_driving`을 다시 띄우면 Gazebo가 **하나 더** 뜬다.

```bash
# 터미널1 — 이미 실행 중이면 그대로
./scripts/dev_container.sh sim-bringup

# 터미널2 — 컨테이너 안
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 scripts/vision_tune/tune_lane_detect.py --mode white
# 키 1–7: white / yellow / fork / fork_left / fork_right / red / crossing
```

오프라인: `--image path.png` 또는 `--folder data/captures/sim`.

레거시(전체 OpenCV 창, **launch가 Gazebo를 포함**하므로 bringup과 중복 주의):

```bash
# bringup이 없을 때만
LANE_VISUALIZE=control ros2 launch dracer_sim sim_auto_driving.launch.py
LANE_VISUALIZE=on ros2 launch dracer_sim sim_auto_driving.launch.py
```

| 값 | 의미 |
|----|------|
| `off` / 미설정 | 창 없음 (기본) |
| `control` | 경계·분기 등 주행 관련 창 (`white_boundaries`, `yellow_boundaries`, `road_branches`) |
| `on` / `1` | 전체 창 |

보드/SSH에서는 **켜지 마세요** (OpenCV 창·성능). 순차 검증은 **§6.2**.

---

## 6. 인지 파이프라인 · 시각화 검증 (`modules/lane_detection.py`)

### 6.1 원태 인지 상태 브리핑 (현재 main)

파일 하나: `modules/lane_detection.py` (~3.8k줄). BEV = Metric IPM (`config/lane_vision.yaml`). **조향 없음.**

```
camera frame
  → HSV white / yellow / black / red
  → Metric IPM warp_mask
  → fill_road_surface_holes + yellow dash connect
  → crossing line detect + fill holes
  → build_global_boundary_course L/R → LaneMarking (white/yellow, side_hint)
  → build_road_branches_cells → fork_active + RoadBranch
  → LaneDetections
```

**이미 잘 된 것 (#26+#27):**

- 흰/노란 **마스크 분리** (`white_hsv` / `yellow_hsv`)
- L/R을 **별도 경계 배열**로 추적 (`white_left/right`, `yellow_left/right`) → `LaneMarking.side_hint`
- 갈림길: half-split → **셀 추적** + 같은 색만 분기 + 깜빡임 완화
- 한쪽 선만 보일 때: IPM `valid` 마스크 + 반대색 반박 + 기울기 기반 폭 `1/cosθ`
- 점선 연결, 가로 실선 catch-22 수정, 성능 ~55ms/frame
- 시각화: `LANE_VISUALIZE=off|control|on` (기본 off) · **검증 주 경로 = `tune_lane_detect.py`**

**아직 약하거나 검증이 필요한 것 (승현 임시 합류):**

1. 곡선에서 한쪽 선이 FOV 밖으로 나갈 때 L/R ID가 뒤집히는지
2. 갈림길에서 **바깥 선 미검출** 시 branch/L-R이 헷갈리는지
3. `fork_active` rising이 BEV 1.5 m 한계로 늦게 뜨는지 (planner 체감과 연결)
4. 실차 HSV vs 시뮬 HSV
5. 빨간 장애물 차로(`red_road`) 커버리지·미션 힌트

**담당:** **안승현(임시)** — 갈림길·곡선·한쪽선 L/R. **조향·MainPlanner FSM은 건드리지 않음.**  
장원태 복귀 후 공동 소유·핸드오프. Metric IPM 계약·msg 필드 삭제/개명은 팀장과 합의.

### 6.2 모드 튜너 순차 검증 (`tune_lane_detect.py`)

구현·버그픽스 **전에** 모드별로 관측 로그를 남긴다. DISPLAY가 있는 PC에서, **sim-bringup만** 켠 뒤 튜너 실행.

| 순서 | `--mode` / 키 | 확인 질문 |
|------|---------------|-----------|
| 1 | `white` (`1`) | 흰 마스크·L(빨강)/R(파랑) 경계가 맞는가 |
| 2 | `yellow` (`2`) | 노란 경계·점선 연결이 합리적인가 |
| 3 | `dash` (`3`) | 분기/합류 점선(노랑·흰)이 분리·연결되는가 |
| 4 | `dash_left` / `dash_right` (`4`/`5`) | 선택한 갈래 쪽 점선만 남고 반대 고어 선은 빠지는가 |
| 5 | `fork` (`6`) | 갈림길에서 branch 2개가 안정적인가 |
| 6 | `fork_left` / `fork_right` (`7`/`8`) | 좌·우 갈래를 따로 구분할 수 있는가 |
| 7 | `red` (`9`) | 동적 장애물 빨간 차로 커버리지가 뜨는가 |
| 8 | `crossing` (`0`) | 가로 정지선/진입선이 경계를 오염시키지 않는가 |

```bash
# ❌ bringup이 이미 있을 때 sim_auto_driving 재실행 금지 (Gazebo 중복)
# ✅
python3 scripts/vision_tune/tune_lane_detect.py --mode white
```

트랙바는 모드별 HSV·`detect_tune` 스칼라. `s` → `config/lane_vision.yaml` (`hsv:` + `detect_tune:`).

버그픽스는 `feature/seunghyun-lane-fork-audit` 등에서 모드 1→6 검증 후 진행.

### 6.3 파이프라인 요약 (파라미터)

- **BEV SSOT:** `config/lane_vision.yaml` → `metric_ipm` + `scripts/vision_tune/metric_ipm.py`
  - 전방 ≈0.22~1.5 m, 횡 ±0.77 m, m/px=0.004
  - 사다리꼴 `bev_roi` / `tune_bev_roi.py`는 **시각 참고 툴만**
- **HSV:** YAML `hsv:` (`tune_hsv.py` / `tune_lane_detect.py`)
- **detect_tune:** crossing coverage / branch separation / red hue wrap (`tune_lane_detect.py`)
- **출력:** 조향 없음. `LaneDetections` + fork/branches · 디버그는 `detect_with_debug` / `LaneDebugFrame`

---
## 7. 이전 호환 제어 (`lane_control_node` + `lane_planner`)

- 구독 → `detections_from_msg` → `LanePlanner.step` (P + EMA + rate limit)
- throttle = `cruise_throttle * throttle_scale`
- `/perception/lane`가 `lane_timeout_sec`(기본 0.5s) 이상 없으면 **throttle=0**
- 게인: `config/lane_control.yaml` (`tune_lane_control.py`)
- **현재 auto-driving launch에서는 사용하지 않으며 MainPlanner와 동시 실행하지 않는다.**
- 신규 기능·튜닝을 이 경로에 넣지 말 것. 제어 SSOT는 [main-planner.md](./main-planner.md).

---

## 8. `pipeline.py`는 무엇인가

| 경로 | 용도 |
|------|------|
| **현재 ROS 런타임** | `inference_node` 내부 `MainPlanner.step(frame)` |
| **검증 토픽** | `/perception/lane`, `/debug/aruco`, `/debug/planner` |
| **호환 API** | `fuse_control` (기존 단위 테스트용) |

런타임 미션 우선순위와 경로 선택은 모두 `MainPlanner`가 소유한다.
`fuse_control`은 이전 호환 테스트를 위해 유지하지만 auto-driving 경로에서는
사용하지 않는다.

---

## 9. 팀원별 PR 전 체크리스트

### 공통

- [ ] `main`에서 feature 브랜치 생성
- [ ] [collaboration.md](./collaboration.md) 담당 파일만 수정
- [ ] 인지 모듈은 결과만 반환하고 최종 `/control`은 `MainPlanner`만 발행함
- [ ] 시뮬 테스트 시 **하드웨어 노드**(`camera_node`/`control_node`)를 요구하지 않음
- [ ] 시뮬 또는 보드에서 build 성공

### 안승현(임시) / 장원태 (`lane_detection.py`)

- [ ] 반환은 인지 `LaneDetections` (조향 필드에 의미 있는 값 넣지 않음)
- [ ] BEV는 Metric IPM 유지 (사다리꼴 런타임 복구 금지)
- [ ] `LANE_VISUALIZE` 기본 off / 검증은 `tune_lane_detect.py` (bringup만) · `control`·`on`은 레거시
- [ ] msg에 이미 있는 필드명·단위(m, base_link) 유지
- [ ] 갈림길·곡선 작업 시 §6.2 모드 순서로 관측 로그
- [ ] **MainPlanner / `main_planner.yaml`을 인지 PR에 섞지 않음**
- [ ] 가능하면 `ros2 topic echo /perception/lane --once`로 발행 확인

### 양서준 (MainPlanner / PP / In·Out)

- [ ] 입력은 모듈 `LaneDetections` (같은 프로세스) 또는 `types`/`adapters`
- [ ] `/control`은 `inference_node`의 MainPlanner만 발행
- [ ] `lane_control_node`와 **동시에** `/control`을 쓰지 말 것
- [ ] `fork_active` / `branches` 활용 시 msg 계약 준수
- [ ] `types.py` / `lane_adapters.py` 변경 필요 시 팀장과 먼저 합의

### 장원정 (`traffic_sign.py`)

- [ ] 현재 ROS 경로에는 아직 미연결 — 모듈 API(`TrafficResult`) 유지
- [ ] `/control` 직접 발행 금지 (통합은 팀장)

### 박성준 / ArUco

- [ ] 인지는 `inference_node`가 `/debug/aruco` 발행
- [ ] 정지를 `/control`에 넣으려면 MainPlanner/통합 쪽 합류

### 안승현 (통합·시뮬)

- [ ] `types` / `adapters` / launch / YAML SSOT · Metric IPM 잠금
- [ ] 시뮬·실차 **노드 세트**와 launch 파라미터 차이 유지
- [ ] 레거시 `lane_control_node`를 auto launch에 되넣지 않음

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
| `modules/lane_detection.py` | 인지 (승현 임시 / 원태) |
| `pipeline.py` | **MainPlanner** (양서준) — 런타임 조향 |
| `config/main_planner.yaml` | PP·FSM 게인 |
| `modules/lane_planner.py` | 레거시 P/EMA (실행 금지) |
| `types.py` | SSOT dataclass |
| `lane_adapters.py` | module/msg → types |
| `inference_node.py` | ROS 노드 (MainPlanner 호스트) |
| `lane_control_node.py` | 레거시 제어 노드 (실행 금지) |
| `config/lane_vision.yaml` | Metric IPM + HSV |
| `config/lane_control.yaml` | 레거시 planner 게인 |
| `src/lane_msgs/` | ROS 인터페이스 |
| `launch/auto_driving.launch.py` | **실차** 노드 세트 |
| `dracer_sim/.../sim_bringup.launch.py` | **시뮬** 월드·브리지 |
| `dracer_sim/.../sim_auto_driving.launch.py` | **시뮬** 자율 (bringup+inference_node) |
