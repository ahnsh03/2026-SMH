# Main Planner · Pure Pursuit 통합 가이드

## 목적과 데이터 흐름

`pipeline.py`가 차선·신호등/표지판·ArUco 인지 결과를 같은 카메라
프레임에서 직접 받아 최종 조향과 속도를 결정한다. 내부 연산은 ROS 토픽을
재구독하지 않으며, `/perception/lane`과 `/debug/*`는 검증·기록용이다.

```
/camera/image/compressed
        ▼
MainPlanner.step(frame)
  ├─ traffic_sign.detect()     → TrafficResult   # Out fork arm 전에 표지 먼저
  ├─ aruco_detection.detect()  → ArucoResult
  └─ lane_detection.detect(enable_fork=…) → LaneDetections
        ▼
Pure Pursuit / mask_p + mission state → ControlCommand → /control
```

기존 `modules/roundabout.py` override는 제거했다. 일반 주행, **Out 갈림**,
**In 회전교차로(유지·탈출 분기)** 의 상태와 최종 제어권은 `MainPlanner`가 소유한다.

용어(갈림·갈래·보조 코스 등): [lane-occlusion-fork-strategy.md §0](./lane-occlusion-fork-strategy.md).
코드의 `branches` / `fork_active` / `*_alt_*`는 **와이어 식별자**로 유지한다.

### 코스 ↔ 차선 색

| `route.mode` | 코스 | **추종 차선 색** | 비고 |
|--------------|------|------------------|------|
| `out` | Out (S자·**Out 갈림**) | **흰색** | `white_centerline` / 흰 갈래(`branches`) |
| `in` | In (회전·**In 탈출 분기**) | **노란색** | `yellow_centerline` / 노란 갈래(`branches`) |

인지는 흰·노란을 동시에 내보낸다. 플래너가 코스에 맞는 색을 경로로 고른다.

| 코스 | 추종 | 금지/폴백 |
|------|------|-----------|
| **Out** | **흰만** (`white_centerline` / 흰·`road_split` 갈래) | 노란 경로·노란 갈래 **금지** (`prefer_yellow` Out에서 강제 False) |
| **In** | **노란이 있으면 노란 우선**, 없으면 흰 | 미션: 흰(진입) → 노란(원) → 흰(합류). “없으면 last resort 노란”이 아님 |

표지 없이 방향 고정(시뮬): `forced_turn:=left|right` → 카메라 표지 **방향·rank만** 고정, 로그 `sign_ignored(forced=…)`.  
**주의:** `forced_turn`은 기본적으로 OUT **갈림 인지(`enable_fork`)를 랩 내내 켜지 않는다.** 표지 없이 fork만 보고 싶으면 `route.out_fork_forced_turn_arms: true` 또는 `out_fork_require_sign: false`.  
자세한 인지 계약: [lane-occlusion-fork-strategy.md §0](./lane-occlusion-fork-strategy.md).

## NORMAL 추종기 (`tracker` / `mask_pursuit`)

| `tracker.normal` | 의미 |
|------------------|------|
| `mask_p` | **보드 SSOT.** hard white corridor COM (mask_hard_wide, cruise≈0.28) |
| `pp` / `hybrid` | 실험용 |

**OUT NORMAL:** 흰 hard corridor `mask_p` + 표지 후 fork는 **branch PP**.  
저속 동결: cruise **0.28** / curve **0.18** (`mask_policy` 랩 승자 + 감속).

갈림 L/R 확인:

```bash
PYTHONUNBUFFERED=1 python3 scripts/drive_test/fork_spawn_unit.py \
  --mode live --scenario all --duration 8 --viz control --drive
```

창: `Lane drive`, `Fork select`. 로그: `data/captures/fork_drive_logs/<stamp>/`.

## 코스 선택과 설정

기본값과 튜닝값은 `config/main_planner.yaml` 한곳에서 관리한다.

```yaml
route:
  mode: out          # out | in
  # prefer_yellow omitted: OUT=항상 흰 전용(강제). IN=기본 True.
  sign_confirm_frames: 3
  out_fork_require_sign: true   # OUT: 표지 후에만 갈림 인지
  out_fork_sign_hold_sec: 3.0   # 표지 소실 후 유지 시간
```

launch에서 이번 실행만 덮어쓸 수도 있다. 인자를 생략하면 YAML 값을 쓴다.

```bash
ros2 launch inference auto_driving.launch.py route_mode:=out   # 흰 차선
ros2 launch inference auto_driving.launch.py route_mode:=in    # 노란 차선
```

- **`route_mode:=out`:** 흰 센터라인·흰/`road_split` 갈래만. 노란으로 끌려가지 않음.  
- **`route_mode:=in`:** 노란이 안정적으로 보이면 노란 우선, 아니면 흰(진입·합류).
- **`forced_turn:=left|right`:** 표지 무시하고 방향·rank 고정 (Out·In 공통).

설정 영역:

- `route`: In/Out, 표지판 확인, **Out 갈림 표지 게이트**, 분기 경로 유지·재진입 방지
- `pure_pursuit`: look-ahead, wheelbase, 최대 조향각, 변화율 제한
- `speed`: 직선/곡선 throttle과 곡선 판정 기준
- `tracker` / `mask_pursuit`: NORMAL·원형 mask COM / fork→PP 가드
- `path`: 최소 점 개수, 색상 confidence, 경로 소실 조건
- `roundabout`: 최소 회전 시간, 탈출 branch, debounce, **원형 fork-PP 억제**, throttle
- `signals`: 시작 초록불 및 빨간불 정지 활성화
- `safety`: 카메라 프레임 watchdog
- `debug`: 상태·판단 변경 로그

설정은 노드 시작 시 한 번 로드하므로 변경 후 inference 노드를 재시작한다.

## 상태와 경로

| 상태 | 의미 | 제어 경로 |
|---|---|---|
| `WAIT_GREEN` | 초록불 대기(설정 시) | 정지 |
| `NORMAL` | 기본 주행 | `tracker.normal`: PP(색 센터) 또는 `mask_p`(코리도 가드) |
| `FORK_TURN` | **Out 갈림** (표지 잠금) | 선택 갈래 (`branches[rank]`) — 마스크 COM 미사용 |
| `ROUNDABOUT_CIRCLE` | In · 회전 **유지** | 색상 센터라인 / mask 설정 시 동일 가드 |
| `ROUNDABOUT_EXIT_READY` | 한 바퀴 후 · 탈출 갈래 대기 | 색상 센터라인 |
| `ROUNDABOUT_EXIT` | **In 탈출 분기** · 선택 갈래 추종 | 설정된 갈래 |

ArUco 정지는 상태를 바꾸지 않는 최우선 인터럽트다. 해제되면 기존 상태에서
다시 주행한다.

Out 코스는 같은 표지판을 `sign_confirm_frames` 동안 연속 확인한 뒤 방향을
래치한다. **갈림 인지(`enable_fork`)는 표지가 보인 뒤에만 켜진다**
(`route.out_fork_require_sign`, hold=`out_fork_sign_hold_sec`). 평소·갈림
완료 후(`FORK_TURN`→`NORMAL`)에는 다시 **흰 센터라인/마스크만** 본다 — 상시
branch 패널로 옆 코스에 끌리지 않게 한다. `fork_active`도 `branch_on_frames`만큼
연속 검출되어 rising event가 발생해야 `FORK_TURN`으로 진입한다. 진입 순간
방향과 branch rank를 잠그므로 회전 중 다른 표지판이 오검출되어도 선택이
바뀌지 않는다. **LEFT → rank 0, RIGHT → rank 1** (두 레이어). `UNKNOWN`이면
설정의 `default_out_branch_rank`를 즉시 사용한다. PP는 **선택한 레이어만** 본다.

분기 인지가 잠시 끊기면 마지막 선택 경로를 `fork_path_hold_frames` 동안
유지하고, 이후 색상 센터라인으로 fallback한다. `fork_active`가
`fork_exit_off_frames` 동안 사라지면 `NORMAL`로 돌아가 방향 래치를 지우며,
`fork_reentry_cooldown_sec` 동안 같은 분기의 즉시 재진입을 막는다.

ONNX 모델/runtime이 있으면 학습 모델을 우선 사용하고, 없으면 파란 원 내부
흰색 화살표의 OpenCV fallback으로 LEFT/RIGHT를 판별한다.

In 코스 **회전 유지**(`ROUNDABOUT_CIRCLE`)에서는 `circle_ignore_fork_for_control`이면
branch가 보여도 mask/센터라인으로 계속 추종하고(포크 PP로 깜빡이지 않음),
branch·노란 가로선은 debounce/rearm 카운터로 등장 이벤트만 센다.
탈출 ready/exit에서는 선택 갈래 PP를 사용한다.

```text
최소 회전 시간 충족
AND (branch 이벤트 2회 OR 가로선 이벤트 2회)
→ EXIT_READY → 유효한 branch 2개 확인 → 설정된 rank로 탈출
```

## Pure Pursuit

인지가 출력하는 Metric IPM 점열은 `(x 전방, y 왼쪽)` 미터 단위다. 현재
카메라 원점에서 rear axle까지의 `0.265 m`를 planner 경계에서 한 번만 더해
PP, heading, CTE가 모두 같은 rear-axle 좌표를 사용한다. 경로 점 하나를
고르는 대신 경로 선분과 차량 중심 LD 원의 교점을 계산하므로 센터라인 점
개수가 바뀌어도 `target_distance`가 현재 LD에 유지된다.

```text
curvature = 2*y / (x²+y²)
steer_angle = atan(wheelbase*curvature)
steering = -steer_angle/max_steer_angle
```

D-Racer 규약은 음수 조향이 왼쪽, 양수 조향이 오른쪽이다. 최종 조향은 PP에
경로 접선 heading 보정과 bounded Stanley 형태의 CTE 복귀항을 더한다.

```text
raw_steering = pp_steering + heading_steering + cte_steering
```

조향 포화와 초당 변화율 제한을 적용하므로 카메라 FPS가 바뀌어도 반전 속도가
유지된다. 경로가 잠깐 사라지면 직전 조향을 유지하고, 계속 유실되면 설정된
초당 복귀율로 중립까지 풀며 마지막에는 throttle을 0으로 만든다.

현재 LD와 속도는 앞쪽 3점 곡률로 함께 조절한다. 직선에서는 긴 LD와
`cruise_throttle`, 곡선에서는 짧은 LD와 `curve_throttle`을 사용하며 그
사이는 곡률 비율로 연속 보간한다. 코너 진입 시 LD는 빠르게 줄고 직선 복귀
시에는 천천히 늘어나도록 별도 변화율 제한을 둔다.

## 디버깅

```bash
ros2 topic echo /debug/planner --field data
ros2 topic echo /perception/lane --once
ros2 topic echo /debug/aruco
ros2 topic echo /control
```

`/debug/planner`는 아래 형식의 한 줄 문자열을 발행한다. 상태·판단·표지판·분기
선택이 바뀌면 즉시 발행하고, 수치 확인용 snapshot은 `debug.publish_hz`로
제한한다.

```text
sign_seen=left candidate=left/3 latched=left locked=left |
state=fork_turn fork=1/2 event=1 events=1 |
choice=sign_left rank=0 path=left_branch decision=out_fork_left |
steer=-0.420 throttle=+0.080
```

- `state`, `decision`, `route`
- `path_source`: `white_centerline`, `yellow_centerline`, `left_branch`,
  `right_branch`, `hold_previous`, `stop`
- 흰색/노란색 visibility와 confidence
- branch·가로선 검출, 이벤트 여부와 누적 횟수
- 신호등, 표지판, ArUco 상태
- 확인 중인 표지판과 연속 프레임 수, 래치/잠금 방향
- 선택 branch rank와 선택 이유, steering, throttle

상세 PP 수치는 `PlannerOutput.debug`에 유지하며 테스트와 추가 계측에서 사용할
수 있다. ROS 로그는 매 프레임이 아니라 state/path/decision/sign/fork 변경 시
출력한다.

## ROS2 Docker 시뮬레이션

호스트가 ROS1이어도 ROS2 명령은 `2026-smh-sim` 컨테이너에서 실행한다.

```bash
docker exec -it 2026-smh-sim bash
cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run inference inference_node --ros-args \
  -p use_sim_time:=true \
  -p planner_config_file:=/workspace/config/main_planner.yaml \
  -p route_mode:=out
```

custom message를 echo하려면 새 컨테이너 셸마다 overlay를 source해야 한다.

실차 launch는 `use_vehicle_steer_trim=true`로 `vehicle_config.yaml`의 servo
중립 보정값을 사용한다. 시뮬 launch는 `use_vehicle_steer_trim=false`와
`steer_trim=0.0`을 전달하므로 실차 trim이 Gazebo 조향에 섞이지 않는다.

## 현재 검증 상태와 주의사항

- ROS2 Docker 빌드 및 Planner import 검증 통과
- planner 20개와 방향 표지판 fallback 3개 단위 테스트 통과
- rear-axle 좌표 변환, LD 원 교점 보간, 곡선 근거리 외삽, heading/CTE,
  경로 유실 복귀, 회전교차로 상태, OUT 분기 잠금·캐시를 테스트로 검증
- 카메라 QoS는 newest-frame 우선(`BEST_EFFORT`, depth 1)이며 새 인지 결과의
  제어 명령을 즉시 발행한다. 타이머 발행은 watchdog/heartbeat로 유지한다.
- 일반 흰색 차선의 직선·연속 곡선 주행은 시뮬에서 확인했다.

아직 남은 한계:

- 회전교차로 탈출구와 Out 갈림길의 `fork_active`/branch 분리는 추가 인지
  튜닝이 필요하다.
- branch는 실제 차선 중심이 아니라 현재 카메라에 보이는 주행 가능 영역의
  중심이다. 분기점에 가까워져 근거리 영역이 가려지면 branch 시작점이 앞쪽으로
  이동해 제어 경로가 흔들릴 수 있다.
- 현재 `FORK_TURN`은 선택 branch를 직접 추종한다. 근거리 차선과 원거리
  branch를 연결한 hybrid path 및 odometry 기반 temporal path 보정은 아직
  구현하지 않았다.
- 표지판이 확정되지 않은 채 분기 event가 먼저 발생하면 감속 대기 없이
  `default_out_branch_rank`를 사용한다.
- 단안 카메라의 근거리 사각지대는 제한된 다항 외삽으로 완화할 뿐 완전히
  복원할 수 없다. 실제 트랙에서 분기·가로선 임계값과 속도를 재조정해야 한다.
