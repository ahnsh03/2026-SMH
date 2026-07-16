# 주행 제어 (LaneController + 미션 모드)

인지(`LaneResult`)를 받아 조향/스로틀(`Control`)을 만드는 제어 계층. 대회
In/Out 코스를 모드로 구분한다.

## 인지→제어는 토픽이 아니라 import (in-process)

`lane_drive_node`(driving) 하나가 카메라만 구독하고, **한 프로세스 안에서**
인지·제어를 함수 호출로 연결한다. `/perception/lane`(LaneDetections) 토픽
왕복이 없어 직렬화/전송 지연이 없다.

```
/camera/image/compressed
        │  (구독)
        ▼
  LaneDetector.detect(frame) ──▶ LaneResult   ┐  같은 프로세스,
        │                                       │  import 로 직접 전달
        ▼                                       │  (토픽 아님)
  MissionController.plan(lane, dt) ──▶ Control ┘
        │  (발행)
        ▼
     /control  ──▶ D-Racer control_node (서보)
```

`driving` 이 `inference` 를 import 한다(빌드순서 lane_msgs→inference→driving).
LaneDetections 발행이 따로 필요하면(모니터링) `inference_node` 를 별도 실행한다.

## 파일

| 파일 | 역할 |
|------|------|
| [lane_drive_node.py](../src/driving/driving/lane_drive_node.py) | 통합 노드(카메라→인지→제어→/control) |
| [planner/lane_controller.py](../src/driving/driving/planner/lane_controller.py) | Pure Pursuit + 곡률 FF (`LaneController`) |
| [planner/mission.py](../src/driving/driving/planner/mission.py) | In/Out 모드 상태기계 (`MissionController`) |
| [config/lane_control.yaml](../config/lane_control.yaml) | 게인·모드·회전교차로 파라미터 |
| [launch/lane_drive.launch.py](../src/driving/launch/lane_drive.launch.py) | camera + lane_drive_node + control_node |
| test/[test_lane_controller.py](../src/driving/test/test_lane_controller.py) · [test_mission.py](../src/driving/test/test_mission.py) | 부호/거동·모드전환·랩 |

## LaneController (Pure Pursuit)

- 입력 centerline `[x전방+, y우측+]`(카메라 프레임) → 후륜축 이동(x += 0.20).
- 전방 3점(0.45/0.70/0.95 m)으로 부호있는 곡률 κ 추정.
- 적응형 look-ahead: `Ld = base(1-r) + curve·r`, `r=clip(|κ|/full,0,1)` — 급커브일수록 짧게.
- Pure Pursuit `δ = atan(L·2y/d²)` + 곡률 FF `δ_ff = k_ff·atan(L·κ)`.
- EMA + rate-limit 평활 → 정규화 조향(δ/δ_max) + STEER_TRIM.
- 스로틀 `cruise(1-r) + curve·r` (커브 감속), 경로소실 시 0(정지).

**부호 규약(안전 직결)**: 인지 centerline y 우측+ → 내부 PP 는 우측=+ 로 계산.
**이 D-Racer 차량은 오른쪽=음수(−), 왼쪽=양수(+)** 이므로 출력에 `steer_sign`(기본
**−1**)을 곱해 변환한다. 우회전 → 음수, 좌회전 → 양수.
> ⚠ 백파일 검증: 사람 /control 은 좌회전 시 +1.0. steer_sign=+1 로 두면 사람 조향과
> **역상관(반대조향)** → 실차 즉시 이탈. steer_sign=−1 이 정상. test 가 고정.

차량 기하: L=0.175, δ_max=0.3054 rad(17.5°), 카메라→후륜축 0.20 m, 최소선회 0.535 m.

## 미션 모드

### Out 코스 (출발→S자→갈림길→동적장애물→도착)
거의 흰선 → `white_centerline` 를 LaneController 로 추종. 갈림길 표지/신호등은 후속.

### In 코스 (출발→회전교차로→동적장애물→도착)
노란선으로 진입 → 회전교차로 1바퀴 → 탈출. 상태기계:

| 상태 | 추종 | 전이 조건 |
|------|------|-----------|
| APPROACH | 흰선 | 노란 경로 `entry_confirm_frames` 연속 감지 → ENTERING |
| ENTERING | 노란선 | 누적 heading ≥ `enter_commit_deg` → ROUNDABOUT |
| ROUNDABOUT | 노란선 | 1바퀴(누적 heading≥`roundabout_exit_deg` 또는 `lap_time_s`) → EXITING |
| EXITING | 흰선 | 흰선 `exit_confirm_frames` 안정 → DONE |
| DONE | 흰선 | (Out 과 동일) |

**노란선 진입 판단**: 인지가 `yellow_centerline`(노랑 전용 경로, 흰과 독립 추적)
을 제공. 그 경로가 충분히 길게(≥`entry_min_yellow_pts`) 연속 감지되면 진입으로
본다. (표지 인식 없이 노랑 마킹 자체로 판단 — 후속에서 방향표지와 결합 가능.)

**회전교차로 1바퀴 판정(오도메트리 없음 → 자전거모델 heading 적분)**:
`Δψ = (v/L)·tan(δ)·dt`, `v≈roundabout_speed_mps`(커브 정속 가정). 누적 heading
이 `roundabout_exit_deg`(기본 300°) 이상이면 탈출. `roundabout_lap_time_s>0` 이면
시간 기반으로 대체. `min/max_time` 가드로 조기·무한 회전 방지(규정: 반드시 1회↑
회전 후 탈출).
> ⚠ 이 lap 판정은 **실차 튜닝 필수**: `roundabout_speed_mps`(또는 실측 `lap_time_s`)
> 를 실차에서 맞춰야 한다. 한 바퀴가 짧/길면 두 값부터 조정.

## 실행

```bash
# Out 코스 (흰선 추종)
ros2 launch driving lane_drive.launch.py course_mode:=out

# In 코스 (노란선 진입 + 회전교차로)
ros2 launch driving lane_drive.launch.py course_mode:=in

# 게인 오버라이드
ros2 run driving lane_drive_node --ros-args -p course_mode:=in \
     -p cruise_throttle:=0.25 -p roundabout_speed_mps:=0.4
```

## 검증 (bag 리플레이 vs 사람 /control)

`scripts/vision_tune/drive_replay.py` 로 인지+제어(in-process)를 재생해 Out 모드
제어 조향을 bag 의 사람 수동조향(/control)과 비교(bag_20260715_204143, 986 프레임).

| 지표 | 값 | 해석 |
|------|-----|------|
| 조향 상관(주행중) | **+0.401** | 사람과 같은 방향(부호 수정 전 −0.401 역상관) |
| my/human 조향 std | 0.352 / 0.327 | 크기대 비슷(사람은 수동 과조향·지터) |
| In 모드 상태전이 | approach→entering→roundabout→exiting→done | 노랑 진입·회전교차로 1바퀴·탈출 전 구간 동작 |

상관 +0.401 은 참조 bag(+0.664)보다 낮다 — 이 코스에 게인 미튜닝 + 사람 반응지연 +
수동 지터 때문. **핵심(부호 안전성·모드 전이)은 검증 완료**, 게인은 실차 튜닝.

```bash
python3 scripts/vision_tune/drive_replay.py --bag <bag> --stride 4
```

## 다음 단계
- 회전교차로 lap 파라미터 실차 튜닝(속도/시간).
- 신호등(출발 초록/도착 빨강)·좌우 표지·ArUco·동적장애물(빨강 구간 정지) 인지·판단
  결합 — 현재 미포함. 상위 판단은 이 노드 안에 상태로 추가하거나 planner 계층 신설.
