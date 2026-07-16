# 자율주행 실행 & 튜닝

인지→제어를 한 프로세스(import)로 돌리는 통합 노드 `lane_drive_node` 로 주행한다.
코스는 `course_mode`(out/in)로 고른다.

## 0. 사전 (최초 1회)

D-Racer-Kit(카메라/서보/제어 메시지)는 **언더레이**로 먼저 빌드·소싱한다.

```bash
# ROS 소싱 (셸에 없으면)
source /opt/ros/humble/setup.bash

# 1) 언더레이: D-Racer-Kit
cd ~/D-Racer-Kit
colcon build --symlink-install
source install/setup.bash

# 2) 오버레이: team-new (lane_msgs → inference → driving 순서로 자동 해석)
cd ~/2026-SMH-team-new
colcon build --symlink-install
source install/setup.bash
```

> 새 터미널마다 `source ~/D-Racer-Kit/install/setup.bash` **다음** `source
> ~/2026-SMH-team-new/install/setup.bash` 순으로 소싱한다.

## 1. 자율주행 실행

```bash
# Out 코스 (흰선 추종: 출발→S자→갈림길→동적장애물→도착)
ros2 launch driving lane_drive.launch.py course_mode:=out

# In 코스 (노란선 진입→회전교차로 1바퀴→탈출)
ros2 launch driving lane_drive.launch.py course_mode:=in
```

이 launch 는 `camera_node`(D-Racer) + `lane_drive_node`(인지+제어) + `control_node`
(D-Racer 서보)를 함께 띄운다. `/camera/image/compressed` → (in-process) → `/control`.

정지: `Ctrl-C`. 인지가 `lane_timeout`(0.5s) 이상 끊기면 자동 안전정지(throttle 0).

**동작 확인**
```bash
ros2 topic hz /control            # 명령 발행 주기(=command_hz, 기본 20Hz)
ros2 topic echo /control          # steering(우=−/좌=+), throttle 값
```

## 2. 튜닝

### A. 오프라인 (백파일 — 실차 없이, 권장 1차)

실차를 굴리기 전에 **백파일 리플레이로** 인지·제어를 맞춘다. 빠르고 안전하다.
(스크립트는 `config/` 소스를 직접 읽으므로 **재빌드 불필요**.)

```bash
cd ~/2026-SMH-team-new

# (1) HSV 재tap — 조명/카메라 설정 바뀌면
python3 scripts/vision_tune/lane_replay.py --bag <bag> --hsv-stats --stride 20

# (2) 검출 확인 — BEV 오버레이(흰=빨강/노랑=초록/중심선=마젠타) 저장
python3 scripts/vision_tune/lane_replay.py --bag <bag> --out /tmp/lane --stride 6
#   전체 검출률/근거리 y 통계도 출력. --video /tmp/lane.mp4 로 영상.

# (3) 제어 확인 — 제어 조향을 사람 수동조향(/control)과 비교
python3 scripts/vision_tune/drive_replay.py --bag <bag> --stride 4
#   steer corr(주행중) 이 +면 방향 일치. In 상태전이도 출력.
```

- HSV 는 `config/lane_vision.yaml` 의 `hsv.white / hsv.yellow` 를 편집 → (2) 재확인.
- 제어 게인은 `config/lane_control.yaml` 편집 → (3) 재확인.

### B. 실차 파라미터

오프라인으로 맞춘 뒤 실차에서 미세조정. 주요 값(`config/lane_control.yaml`):

| 파라미터 | 기본 | 언제 조정 |
|----------|------|-----------|
| `cruise_throttle` | 0.22 | 직진 속도. 느리면↑ (올리면 lookahead 도 같이↑) |
| `curve_throttle` | 0.13 | 커브 속도. 커브서 밀리면↓ |
| `base_lookahead_m` | 0.85 | 직진 안정성. 흔들리면↑, 굼뜨면↓ |
| `curve_lookahead_m` | 0.45 | 커브 반응. 커브 못 물면↓ |
| `curvature_ff_gain` | 0.5 | 커브 진입 예측. 진동나면 0.3, 늦으면↑ |
| `steer_sign` | **−1** | 차량 조향 부호(우=−/좌=+). **바꾸면 반대조향** |
| `steer_trim` | 0.10 | 직진이 한쪽으로 쏠리면 보정 |

값 반영 방법:
```bash
# ★ 방법1(권장): 주행 중 라이브 변경 — 재빌드/재실행 없이 즉시 반영
#   (다른 터미널에서 소싱 후) lane_drive_node 가 param 콜백으로 바로 적용
ros2 param set /lane_drive_node steer_trim 0.15
ros2 param set /lane_drive_node cruise_throttle 0.26
ros2 param get /lane_drive_node steer_trim          # 현재값 확인

# 방법2: 확정값을 config/lane_control.yaml 에 저장 후 재빌드(다음 실행 기본값)
colcon build --packages-select driving --symlink-install && source install/setup.bash
```

### 직진 중앙정렬 (한쪽 쏠림)
이 차량은 **오른쪽=음수(−), 왼쪽=양수(+)**. 직진에서 한쪽으로 쏠리면 `steer_trim` 으로 보정:
- **오른쪽으로 쏠림 → steer_trim 을 더 크게(+)** (왼쪽으로 보정). 예 0.10 → 0.13 → 0.16 …
- 왼쪽으로 쏠림 → steer_trim 을 더 작게(−방향).
- **직선을 실제로 주행시키며** `ros2 param set /lane_drive_node steer_trim <값>` 으로 곧아질 때까지 조금씩. 찾으면 config 에 저장.

### C. 회전교차로 1바퀴 (In 코스 — **실차 튜닝 필수**)

오도메트리가 없어 자전거모델로 heading 을 적분해 1바퀴를 센다. 실차 속도에 맞춰야 한다.

| 파라미터 | 기본 | 조정 |
|----------|------|------|
| `roundabout_speed_mps` | 0.33 | heading 적분용 정속 추정. 한 바퀴가 **짧게 끝나면↓, 길면↑** |
| `roundabout_lap_time_s` | 0.0 | >0 이면 **시간 기반**(실측 1바퀴 초). 가장 확실: 스톱워치로 재서 넣기 |
| `roundabout_exit_deg` | 300 | lap 판정 각도(≈1바퀴). |
| `roundabout_min/max_time_s` | 2 / 20 | 조기·무한 회전 방지 가드 |
| `enter_commit_deg` | 45 | 노란선 진입 후 이 각도 돌면 회전상태로 |
| `entry_confirm_frames` | 3 | 노란 경로 몇 프레임 연속 감지 시 진입 |

가장 안전한 방법: **실측 1바퀴 시간**을 `roundabout_lap_time_s` 에 넣기(예: 6초면 6.0).

### 권장 튜닝 순서
1. (오프라인) HSV → 검출률 확인(Out 99%+ 목표).
2. (오프라인) `drive_replay` 조향 상관 + 부호(+) 확인.
3. (실차) `steer_trim` 으로 직진 정렬 → `cruise_throttle` 속도 → lookahead 안정.
4. (실차, In) 회전교차로 `roundabout_lap_time_s` 실측값.

## 참고
- 인지 상세: [lane-perception.md](lane-perception.md) · 제어 상세: [lane-driving.md](lane-driving.md)
- 인지만 단독(모니터링): `ros2 launch inference lane_perception.launch.py`
