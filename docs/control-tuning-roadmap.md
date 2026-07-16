# 제어·차선 튜닝 로드맵 (향후 과제)

2026-07-16 보드 세션에서 **즉시 튜닝은 보류**. 아래 순서로 마무리한다.

관련 bag 목록·카메라 SSOT: `external/D-Racer-Kit/bagfile/bagfile.md`  
조향/스로틀 YAML: `config/main_planner.yaml`, `config/vehicle_config.yaml`

---

## 목표 흐름

```
1. 차선 피팅 로직 (bag 바탕 구현)
        ↓
2. bag 입력 → 실차 스로틀·조향각 확인
        ↓
3. 튜닝 → 실제 트랙 구동 → 재튜닝
        ↓
4. 개별 미션 검증
        ↓
5. 통합 (풀 랩)
```

---

## 1. 차선 피팅 로직 — bag 바탕 구현

- OUT/IN 조이스틱 bag으로 BEV·주행가능영역·centerline 피팅을 오프라인 검증.
- 기준 bag (재녹화 23:01~):
  - OUT: `bag_20260715_230145`
  - IN: `bag_20260715_230316`
- 인지/마스크 품질이 안 되면 조향 튜닝 전에 HSV·IPM·blob을 먼저 고정 (`config/lane_vision.yaml`).

## 2. bag 재생 + 실차 `/control` 확인

카메라만 bag, 나머지(인지→플래너→`control_node`)는 실차와 동일.

- `camera_node` **기동 금지** (토픽 충돌).
- bag에서 **`/camera/image/compressed`만** play (`/control`·`/joystick` 재생 금지).
- `route_mode:=out|in`, `traffic_pass:=true`, `publish_bev_debug:=true`.
- 관찰: 모니터 `:5000`, `ros2 topic echo /control`, `/debug/planner`, `scripts/board_monitor_term.py`.

예시:

```bash
# 터미널 1 — control + inference + monitor (카메라 제외)
cd ~/2026-SMH && source install/setup.bash
VC=$HOME/2026-SMH/config/vehicle_config.yaml
PC=$HOME/2026-SMH/config/main_planner.yaml
ros2 run control control_node --ros-args \
  -p use_joystick_control:=false -p vehicle_config_file:=$VC &
ros2 run inference inference_node --ros-args \
  -p vehicle_config_file:=$VC -p planner_config_file:=$PC \
  -p route_mode:=out -p traffic_pass:=true -p publish_bev_debug:=true \
  -p drive_debug_log:=true -p bringup_crawl_throttle:=0.18 &
ros2 run monitor monitor_node --ros-args \
  -p vehicle_config_file:=$VC -p debug_image:=true &

# 터미널 2 — 카메라만
ros2 bag play external/D-Racer-Kit/bagfile/bag_20260715_230145 \
  --topics /camera/image/compressed
# 느리게: --rate 0.5   /  반복: --loop
```

튜닝 키 (요약):

| 파일 | 항목 |
|------|------|
| `config/main_planner.yaml` | `pure_pursuit.*`, `speed.cruise_throttle` / `curve_throttle`, `mask_pursuit.*` |
| `config/vehicle_config.yaml` | `STEER_TRIM`, `STEER_INVERT` |
| launch / node | `bringup_crawl_throttle` |

YAML 변경 후 `inference_node` 재시작.

## 3. 트랙 구동 ↔ 튜닝

- 라이브 카메라로 `debug_monitor.launch.py` / `auto_driving.launch.py`.
- bag에서 맞춘 게인이 실차에서 흔들리면 Ld·rate limit·trim부터 재조정.
- 스탠드에서 좌우 극성 확인: `/control` `steering=+1` → 바퀴 **우**.

## 4. 개별 미션 검증

| 미션 | 확인 |
|------|------|
| OUT 직진·S자 | white PP / hybrid 안정 |
| OUT 갈림 | 표지 OR capture arm, 기본 RIGHT |
| IN 회전교차로 | yellow / roundabout throttle·Ld |
| ArUco 정지·재출발 | `stop_on_aruco` |
| (선택) 신호 | 현재 OFF·15s 가정 또는 `traffic_pass` |

단위: `./scripts/board_test.sh fork|planner|blob|signs|aruco`

## 5. 통합

- `route_mode`별 풀 랩, 대회 전 `publish_bev_debug:=false`로 CPU 절약.
- 최종 게인·카메라 SSOT를 yaml/문서에 고정 후 freeze.

---

## 상태

| 단계 | 상태 |
|------|------|
| 1 차선 피팅 (bag) | 향후 |
| 2 bag→실차 control 확인 | 향후 (절차는 위 §2) |
| 3 트랙 튜닝 | 향후 |
| 4 개별 미션 | 향후 |
| 5 통합 | 향후 |
