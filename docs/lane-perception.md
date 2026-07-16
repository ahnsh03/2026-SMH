# 차선 인지 (Metric BEV + HSV)

실차 백파일 `bag_20260715_204143`(대회 트랙, 시계방향 주행)로 개발·검증한
차선 인지. 카메라 프레임 → 등거리(metric) BEV → HSV 차선 마스크 → **주행 차로
중심선**(base_link 미터)을 만들어 `LaneResult.white_centerline` 으로 낸다.

## 파일

| 파일 | 역할 |
|------|------|
| [metric_bev.py](../src/inference/inference/modules/metric_bev.py) | IPM warp + 좌표변환 (`MetricBev`) |
| [lane_detection.py](../src/inference/inference/modules/lane_detection.py) | HSV·행별 좌우선 추적·중심선 (`LaneDetector`) |
| [config/lane_vision.yaml](../config/lane_vision.yaml) | metric_ipm / hsv / lane_detect (SSOT) |
| [lane_replay.py](../scripts/vision_tune/lane_replay.py) | 백파일 재생·검증·HSV 재tap 도구 |
| [lane_perception.launch.py](../src/inference/launch/lane_perception.launch.py) | 인지 단독 실행 |
| [test_lane_detection.py](../src/inference/test/test_lane_detection.py) | 좌표부호·단측선 좌/우 식별 테스트 |

## 좌표 규약 (안전 직결)

`white_centerline` 은 `[x 전방+, y 우측+]` 미터. **y 우측이 양수**(일반 ROS y-left
와 반대). 후속 LaneController 가 `δ = +atan(L·2y/d²)`(부호 반전 없이) 동작하도록
맞췄다. 우회전 → `y>0`. SSOT 는 `MetricBev.bev_uv_to_xy` / `xy_to_bev_uv`.

## Metric BEV 파라미터 (config: metric_ipm)

팀 참조본(`/home/topst/2026-SMH/config/lane_vision.yaml`, 2026-07-12 lock) 기반.

- 카메라: hfov 70.42°, height 0.13 m, **pitch_down 10.0°**
- 범위: x [0.22, 1.30] m, y ±0.77 m, mpp 0.004 → BEV 386×271
- BEV: row0=원거리(x_max), 마지막 row=근거리(x_min); col0=좌, 마지막 col=우

**보정 검증**: 트랙 스펙(폭 350 mm, 경계선 30 mm)과 대조. BEV 실측 두 흰선 간격
~0.33 m 이고 **거리에 무관하게 거의 일정**(0.328→0.344) → 평행선이 BEV 에서 평행
유지 = IPM 보정 정확. (pitch 13.8°는 거리 따라 0.284→0.228 수렴 = 과보정이라 기각.)

## HSV (config: hsv) — 카메라 v4l2 고정(어둡게) 후 tap

차선=흰(+회전교차로만 노랑), 노면=검정/빨강/파랑. **빨강 노면 구간도 흰 차선**.
카메라를 v4l2 로 색온도·설정 고정(전반 어둡게)해 재tap(bags 230145 Out / 230316 In).
어두운 조명 + 고정 WB 로 흰선이 **웜화이트**(약간 노란기)라 S·V 여유를 크게 준다.

| 색 | H | S | V | 근거 |
|----|---|---|---|------|
| white | 0–179 | 0–**72** | **165**–255 | 웜화이트: 흰선 S≤73/어두운·블러 흰선 V≥170. 빨강노면 S≥90·노랑 S≥80 과 분리 |
| yellow | **13**–30 | **80**–220 | **100**–255 | 어두워져 노랑 H가 red(H≤10)와 근접 → H≥13 + S상한 + V≥100 으로 red/어두운노면 배제 |

`lane_detect.color: white` — Out·직진·**빨강 구간**은 흰선 추종. 노랑은 회전교차로용으로
`yellow_centerline`(흰과 독립 추적) 별도 계산. 조명 바뀌면 재tap: `lane_replay.py --hsv-stats`.

## 검출 로직

1. `MetricBev.warp` → 등거리 BEV (상단 crop 이 관중/배경 제거).
2. HSV → white∪yellow 경계 마스크 → morph(open3/close7).
3. **근거리 시드**: 차는 항상 두 선 사이 → 가장 가까운 행들에서 ego-중심 좌/우 분할로
   좌·우 선 identity 확정.
4. **행별 근접 추적**(근거리→원거리): 이전 위치 근처의 런만 좇음. ego-중심 재판정을
   안 하므로 **급커브에서 선이 중심을 넘어도 좌/우가 안 뒤집힘**.
5. 중심선: 좌·우 모두 → 중점 / 한쪽만 → `(track_width-line_width)/2 = 0.16 m` 오프셋.
6. `bev_uv_to_xy`(y 우측+) → 이동평균 평활.

### 단측선 좌/우 오배정 방지

한쪽 선만 보일 때 우선을 좌선으로(또는 반대로) 오인하면 오프셋이 반대로 걸려
치명적이다. 두 겹으로 막는다:
- **근거리 앵커**: 차가 두 선 사이인 근거리에서 identity 확정 (신뢰 가능).
- **프레임 간 연속성**(`_reconcile_seed`): 단측선의 좌/우를 직전 프레임 트랙으로 식별.
  차가 선을 살짝 넘어 단선이 중심 반대편에 나타나도 뒤집히지 않는다.

`test_lane_detection.py` 가 이 규약을 고정한다. `reset()` 으로 시간 상태 초기화.

## 검증 결과 (v4l2 고정 카메라, 두 bag)

`lane_replay.py --stride 6`:

| bag | 센터라인 검출률 | 비고 |
|-----|------|------|
| Out (230145) | **99.8%** (흰선) | 흰 차선 가시율 100% |
| In (230316) | **99.7%** (흰∪노랑 합집합) | 흰 66% (본선) + 노랑 44% (회전교차로) |

Out=흰선 정중앙 추종. In=흰 본선 + 회전교차로 노랑 경로(yellow_centerline). 근거리 y
mean≈0, std~0.06. 어두운 웜화이트 흰선도 S≤72/V≥165 완화로 견고(완화 전 Out 80.7%).

## 실행

```bash
# 오프라인 검증 (백파일 재생 → 검출 오버레이/통계)
python3 scripts/vision_tune/lane_replay.py \
  --bag /home/topst/2026-SMH/external/D-Racer-Kit/bagfile/bag_20260715_204143 \
  --out /tmp/lane_out --video /tmp/lane.mp4 --stride 3

# HSV 재tap (조명 변화 시)
python3 scripts/vision_tune/lane_replay.py --bag <bag> --hsv-stats --stride 40

# 실차 인지 단독 (카메라 + inference_node)
ros2 launch inference lane_perception.launch.py
```

## 다음 단계 (제어/미션)

- 차선추종 제어기(LaneController, Pure Pursuit)는 `white_centerline` 을 그대로 소비.
  y 우측+ 규약 유지 필수. 대회 차선이탈 페널티(+30초) → 견고한 추종이 최우선.
- 갈림길(좌/우 표지판)·회전교차로·동적 장애물(빨강 구간)·신호등은 상위 planner /
  별도 인지(traffic_sign/direction_sign)에서. `LaneResult.fork_active/branches` 미구현.
- 참고: 트랙 물리규격·대회 규정은 세션 메모리(track-geometry / race-rules) 참조.
