# Main Planner · Pure Pursuit 통합 가이드

## 목적과 데이터 흐름

`pipeline.py`가 차선·신호등/표지판·ArUco 인지 결과를 같은 카메라
프레임에서 직접 받아 최종 조향과 속도를 결정한다. 내부 연산은 ROS 토픽을
재구독하지 않으며, `/perception/lane`과 `/debug/*`는 검증·기록용이다.

```
/camera/image/compressed
        ▼
MainPlanner.step(frame)
  ├─ lane_detection.detect()   → LaneDetections
  ├─ traffic_sign.detect()     → TrafficResult
  └─ aruco_detection.detect()  → ArucoResult
        ▼
Pure Pursuit + mission state → ControlCommand → /control
```

기존 `modules/roundabout.py` override는 제거했다. 일반 주행, Out 코스
갈림길, In 코스 회전교차로의 상태와 최종 제어권은 `MainPlanner`가 소유한다.

## 코스 선택과 설정

기본값과 튜닝값은 `config/main_planner.yaml` 한곳에서 관리한다.

```yaml
route:
  mode: out          # out | in
  prefer_yellow: false
```

launch에서 이번 실행만 덮어쓸 수도 있다. 인자를 생략하면 YAML 값을 쓴다.

```bash
ros2 launch inference auto_driving.launch.py route_mode:=out
ros2 launch inference auto_driving.launch.py route_mode:=in
```

In 코스에서 노란 센터라인을 우선하려면 `prefer_yellow: true`로 설정한다.
노란 경로가 confidence·점 개수 기준을 만족하지 못하면 흰색으로 fallback한다.

설정 영역:

- `pure_pursuit`: look-ahead, wheelbase, 최대 조향각, 변화율 제한
- `speed`: 직선/곡선 throttle과 곡선 판정 기준
- `path`: 최소 점 개수, 색상 confidence, 경로 소실 조건
- `roundabout`: 최소 회전 시간, 탈출 branch, 이벤트 debounce/rearm
- `signals`: 시작 초록불 및 빨간불 정지 활성화
- `safety`: 카메라 프레임 watchdog
- `debug`: 상태·판단 변경 로그

설정은 노드 시작 시 한 번 로드하므로 변경 후 inference 노드를 재시작한다.

## 상태와 경로

| 상태 | 의미 | 제어 경로 |
|---|---|---|
| `WAIT_GREEN` | 초록불 대기(설정 시) | 정지 |
| `NORMAL` | 기본 주행 | 흰색/노란색 센터라인 |
| `FORK_TURN` | Out 코스 표지판 갈림길 | 선택한 좌/우 branch |
| `ROUNDABOUT_CIRCLE` | In 코스 회전 중 | 색상 센터라인 |
| `ROUNDABOUT_EXIT_READY` | 한 바퀴 완료, branch 대기 | 색상 센터라인 |
| `ROUNDABOUT_EXIT` | 회전교차로 탈출 | 설정된 branch |

ArUco 정지는 상태를 바꾸지 않는 최우선 인터럽트다. 해제되면 기존 상태에서
다시 주행한다.

Out 코스는 표지판 결과를 기억한 뒤 `fork_active`에서 LEFT는 가장 왼쪽,
RIGHT는 가장 오른쪽 branch를 선택한다. `UNKNOWN`이면 설정의 기본 rank를
사용한다. 방향 모델/runtime이 없으면 facade는 안전하게 `UNKNOWN`을 반환한다.

In 코스 회전 중에는 branch가 보여도 색상 센터라인을 계속 사용한다. branch와
노란 가로선은 독립적인 debounce/rearm 카운터로 등장 이벤트만 센다.

```text
최소 회전 시간 충족
AND (branch 이벤트 2회 OR 가로선 이벤트 2회)
→ EXIT_READY → 유효한 branch 2개 확인 → 설정된 rank로 탈출
```

## Pure Pursuit

입력은 `base_link` 기준 `(x 전방, y 왼쪽)` 미터 점열이다.

```text
curvature = 2*y / (x²+y²)
steer_angle = atan(wheelbase*curvature)
steering = -steer_angle/max_steer_angle
```

D-Racer 규약은 음수 조향이 왼쪽, 양수 조향이 오른쪽이다. 조향 포화와
프레임당 변화율 제한을 적용하며 큰 조향에서는 `curve_throttle`을 사용한다.

## 디버깅

```bash
ros2 topic echo /debug/planner --field data
ros2 topic echo /perception/lane --once
ros2 topic echo /debug/aruco
ros2 topic echo /control
```

`/debug/planner`는 매 프레임 JSON을 발행한다.

- `state`, `decision`, `route`
- `path_source`: `white_centerline`, `yellow_centerline`, `left_branch`,
  `right_branch`, `hold_previous`, `stop`
- 흰색/노란색 visibility와 confidence
- branch·가로선 검출, 이벤트 여부와 누적 횟수
- 신호등, 표지판, ArUco 상태
- PP 목표점, 경로 점 개수, steering, throttle

ROS 로그는 매 프레임이 아니라 state/path/decision 변경 시 출력한다.

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

## 현재 검증 상태와 주의사항

- ROS2 Docker 빌드 및 Planner import 검증 통과
- PP 조향 부호, 짧은 경로 거부, 이벤트 debounce/rearm 테스트 통과
- 두 번째 branch와 최소 시간에 의한 In 코스 탈출 상태 테스트 통과
- `/control` 약 10 Hz, `sim_control_bridge` 1:1 연결 확인
- 시뮬에서 직선 이후 곡선에 진입하며 조향이 포화되고 흰색 센터라인을 잃는
  현상이 관찰됐다. `target_y`, `steering`, `white_confidence`, `path_points`의
  시간 순서를 비교해 PP와 곡선 인지를 추가 튜닝해야 한다.

