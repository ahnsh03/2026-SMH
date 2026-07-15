# 기본 주행 루프 — 개발 구조·전략

> **2026-07-13 현재 런타임:** 아래 문서는 Metric IPM과 인지 인터페이스의
> 설계 배경을 보존한다. 실제 자율주행 제어는 더 이상
> `lane_control_node`가 아니라 `inference_node` 내부
> `pipeline.MainPlanner`가 담당한다. 현재 구조와 튜닝값은
> [main-planner.md](./main-planner.md)를 SSOT로 사용한다.

> 런타임 제어 SSOT: [main-planner.md](./main-planner.md) ·  
> **실차 하이브리드·T0–T7:** [control-hybrid-strategy.md](./control-hybrid-strategy.md) ·  
> 보드 적용: [board-freeze-control.md](./board-freeze-control.md).

관련: [roles.md](./roles.md) · [main-planner.md](./main-planner.md) · [lane-perception-topic.md](./lane-perception-topic.md) · [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) · [meetings/2026-07-10.md](./meetings/2026-07-10.md) · [hardware-camera.md](./hardware-camera.md) · [vehicle-geometry.md](./vehicle-geometry.md) · [collaboration.md](./collaboration.md)

---

## 0. 개정 요약 (2026-07-13)

| 항목 | 상태 |
|------|------|
| 인지/제어 | `inference_node` → `MainPlanner` → `/control`; `/perception/lane`은 검증용 |
| BEV 런타임 | **Metric IPM** (`lane_vision.yaml`) — 사다리꼴은 참고 툴만 |
| `VISUALIZE` | 기본 **off** · `LANE_VISUALIZE=control`→`Lane drive` 1창 · `on`→+`HSV masks` |
| 주행 조향 | `MainPlanner` PP **또는** `mask_p`(코리도+갈림 가드) — [main-planner.md](./main-planner.md) |
| OUT 갈림 인지 | **표지 시에만** `enable_fork` — 평소 흰 추종 |
| 차선 인지 담당 | **안승현(임시)** / 장원태 — 갈림길·곡선·한쪽선 L/R |
| 레거시 | `lane_control_node` · `lane_planner.py` — **실행 금지** |

인지 `detect()`에는 조향을 넣지 않는다.  
**팀원 필독:** [main-planner.md](./main-planner.md) · [lane-perception-topic.md](./lane-perception-topic.md) (시각화 검증 §).

---

## 1. 목표와 범위

### 목표

Gazebo / D3-G에서 **차선 기반 기본 주행 루프**를 안정적으로 돌린다.

- 노드 실행 → 출발, 종료 → 정지 (시뮬에서는 신호등 게이트 생략 가능)
- 조향은 **부드럽고** 추종이 안정적일 것 (기존 ROS1 LIMO 체감 수준)
- 갈림길·회전교차로에서는 **두 갈래 경로를 각각 분리 감지**하고, 모드에 따라 하나만 선택
- ROI / IPM / HSV 파라미터는 **트랙바 툴로 확정 → YAML 저장 → 런타임·실차 재사용**

### 비목표 (이 문서 범위 밖)

- 신호등 실차 검증 (장원정 / 실차)
- ArUco 로직 재구현 (완료)
- 회전교차로 Pure Pursuit 전부 (양서준·박성준과 인터페이스만 맞춤) — 원태 polyline을 입력으로 쓸 수 있게만 맞춤
- 학습 기반 트랙 검출 (wego CNN) — 1차 미채택

---

## 2. 트랙·센서 전제

| 항목 | 값 |
|------|-----|
| 카메라 | C920e, 팀 출력 **320×180** JPEG `/camera/image/compressed` |
| 제어 | `Control(steering, throttle ∈ [-1,1])` → (시뮬) `control_bridge` → `/cmd_vel` |
| 일반 구간 | 배경 **파랑**, 차로 **검정**, 차선 **흰색** |
| 회전교차로 | 차선 **노란색** |
| 동적 장애물 | 차로 **빨간색** (신호등 빨강과 **색·위치·컨텍스트 분리**) |

**코스 ↔ 색 (SSOT):** **Out = 흰만**. **In = 노란이 있으면 노란 우선**, 없으면 흰  
(미션 흐름: 흰 진입 → 노란 회전 → 흰 합류).  
장면: **Out 갈림** · **In 탈출 분기**. → [lane-occlusion-fork-strategy.md §0](./lane-occlusion-fork-strategy.md)

조향 부호: D-Racer `+steering = right`. LIMO `Twist.angular.z`와 의미·스케일이 다름 → external 제어값을 그대로 쓰지 않음.  
휠베이스·트레드 등 기하 차이(시뮬↔실차 게인 이전): [vehicle-geometry.md](./vehicle-geometry.md).

---

## 3. 권장 아키텍처 (인지 / 판단 분리)

**현재 런타임 (단일 판단·제어 소유자):**

```
/camera → inference_node
       → lane_detection + traffic_sign + aruco_detection
       → pipeline.MainPlanner
       → /control
       └→ /perception/lane, /debug/* (검증·기록용)
```

`lane_control_node`와 `modules/lane_planner.py`는 이전 토픽 분리 실험 코드이며
현재 auto-driving launch에는 연결하지 않는다. 동시에 실행하면 `/control`에
publisher가 둘 생기므로 실행하지 않는다.

상세: [lane-perception-topic.md](./lane-perception-topic.md).

### 계층별 선택

| 계층 | 1차 채택 | 담당 | 비고 |
|------|----------|------|------|
| ROI/IPM 튜닝 | **Metric IPM** YAML (`y_half=0.77`) · 사다리꼴은 참고 | 인지 + 승현 툴 | §4.3 · **런타임도 Metric IPM** |
| 런타임 인지 | `LaneDetections` (조향 없음) | 안승현(임시) / 장원태 | `LANE_VISUALIZE` 기본 off · 갈림길·곡선 L/R |
| 경로·조향 | `pipeline.MainPlanner` + **Pure Pursuit** | 양서준 | In/Out FSM · 최종 `/control` |
| 다중 경로 | `fork_active` / `branches` | 인지 → MainPlanner | 셀 기반 분기 |
| 종방향 | MainPlanner cruise/curve throttle | 양서준 | `main_planner.yaml` |

**minyong:** hold/decay·슬라이딩 윈도우 폴백.  
**v2:** 스무딩·감속 손맛.  
**wego DL:** 1차 제외.

---

## 4. BEV / IPM 규약 (팀 확정 방향)

> **SSOT:** Metric IPM (§4.3, `y_half=0.77`). §4.1–4.2는 사다리꼴 참고 배경.

### 4.1 사다리꼴 참고 — 문제 인식

고전 사다리꼴은 **윗변(먼 쪽)을 좁히고** 아랫변을 이미지 하단 폭 100%에 맞춘다.  
이렇게 하면:

- 먼 바닥의 **좌우 가장자리**가 좁은 윗변 밖으로 잘려 BEV에 안 들어감
- 아랫변을 이미지 폭에 고정하면, warp가 시야를 “꽉 채우도록” 잡아 **더 넓은 지면 범위**를 담기 어렵다

### 4.2 사다리꼴 참고 — “상단만 제외, 나머지 픽셀 최대한 사용”

**의도 (한 줄):** 카메라 이미지에서 **상단 N%만 제외**하고, 그 아래 사각형 안의 **모든 픽셀을 빠짐없이** BEV에 넣는다.  
사다리꼴 `src`는 **이미지 테두리 안에만 그리는 고전 ROI가 아니다** — 아랫변 꼭짓점은 프레임 **밖(가상)** 에 둘 수 있음.

1. **제외:** 윗부분 `crop_top_ratio`(상단 N%)만 사용하지 않음.
2. **포함:** `y = crop_top … H-1`, `x = 0 … W-1`의 **전 픽셀**이 warp 입력에 기여해야 함.
3. **사다리꼴 윗변:** 제외선 높이에서 **가로 풀폭**.
4. **사다리꼴 아랫변:** 이미지 하단 폭보다 **넓게** (프레임 밖 가상 꼭짓점 OK). BEV 가장자리 검정 부채꼴 허용.
5. 확정값은 `config/lane_vision.yaml`에 저장 → 시뮬·실차 동일 스키마.

```
카메라 프레임 (W × H)                 BEV (직사각 캔버스)
───────────────────────────           ─────────────────────
|///// 상단 N% 제외 /////|            |                     |
|←— 윗변 = 이미지 풀폭 —→|            |  crop 아래 전 픽셀   |
|  ★ 이 영역 픽셀 전부 ★  |    →     |  이 안으로 투영      |
|← 아랫변 > W (프레임 밖) →|           | /   부채꼴  \        |
```

### 4.3 Metric IPM — 팀 잠정 SSOT (2026-07-12)

**잠정 확정:** 런타임·플래너 BEV는 **Metric IPM** (원태 `build_ipm_maps` 계열).  
기본 툴: `tune_bev.py`. 사다리꼴 참고: `tune_bev_roi.py`.

카메라 모델: 높이 **0.13 m**, pitch **10°** down, HFoV **70.42°** (C920e / `sim_interface.yaml`).

사다리꼴에서 **상단 39%를 자른 이유**는 그 행이 전방 **≈1.5 m**이기 때문이었다.  
동일 카메라 모델로 투영하면 `x=1.50 m → v/H≈0.389`로 일치한다. 이미지 하단 유효 범위는 `x≈0.22 m`.

| 항목 | 사다리꼴 (참고) | Metric IPM (**SSOT**) |
|------|-----------------|------------------------|
| 먼 쪽 | `crop_top=0.39` | `x_max_m=1.5` |
| 가까운 쪽 | 이미지 하단 | `x_min_m=0.22` |
| 횡 | guide 44 ↔ 0.35 m | **`y_half_width_m=0.77`** (`y_half_cm=77`), BEV≈386×321 |
| 스케일 | 종≠횡 | **등방** m/px |
| 툴 | `tune_bev_roi.py` (사다리꼴만) | **`tune_bev.py`** (`--compare`로 병치) |


YAML: `config/lane_vision.yaml` → `metric_ipm:`.  
`full_image_width: false` — 풀폭(~±1.05 m)이 필요하면 트랙바/`f`로만 임시 확장.

### 4.4 권장 전방 거리 (대회·로봇 기준)

카메라 ~0.13 m / pitch ~10° / 차로 0.35 m / 저속 스케일카 기준으로:

| 항목 | 권장 |
|------|------|
| 전방 BEV | 약 **0.25 ~ 1.5 m** (제어 본체 1.2 m + 갈림길 여유) |
| 좌우 | **±0.77 m** (팀 잠정 `y_half`) · 풀폭 필요 시 ~±1.05 m |
| 2 m 이상 | 픽셀 압축·이득 적음 → 비권장 |

표지판·ArUco·신호등은 BEV 거리가 아니라 원본 프레임에서 처리.

### 4.5 Phase 0 시뮬 튜닝 현황 (2026-07-12) — Metric IPM SSOT

Gazebo에서 Metric IPM으로 **전방 1.5 m · 횡 ±0.77 m** 잠정 확정.  
SSOT: [`config/lane_vision.yaml`](../config/lane_vision.yaml) → **`metric_ipm:`**.

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| `crop_top_ratio` | **0.39** | ≈ `x_max` 행 (기하 일치) |
| `x_min_m` / `x_max_m` | **0.22 / 1.5** | 이미지 하단 ~ 전방 1.5 m |
| `y_half_width_m` | **0.77** | `y_half_cm=77`, `full_image_width: false` |
| `meters_per_pixel` | **0.004** (4 mm/px) | 종·횡 **등방** 격자 해상도. look-ahead 거리(`x_max`)와 별개 |
| BEV | **≈ 386 × 321** | `(2·y_half)/mpp+1` × `(x_max−x_min)/mpp+1` |
| `track_width_m` / guide | **0.35** / **≈44 px** | 차로 마크 정렬 |
| 카메라 | 0.13 m · 10° · HFoV 70.42° | C920e / sim_interface |

사다리꼴 참고값(`bev_roi:`): crop 0.39, bottom 6.35, 500×370 — 시각 비교용.

**캘리브 매트:** 월드 `(2.5, -6.5)` `bev_calib_mat` (0.1 / 0.5 m).

**툴 (기본=IPM):**

```bash
python3 scripts/vision_tune/tune_bev.py              # Metric IPM (SSOT)
python3 scripts/vision_tune/tune_bev.py --compare    # 사다리꼴 병치
python3 scripts/vision_tune/tune_bev_roi.py          # 사다리꼴만 (참고)
python3 scripts/vision_tune/capture_camera.py
```

상세: [`scripts/vision_tune/README.md`](../scripts/vision_tune/README.md).

### 4.6 트랙바 파라미터 (Metric IPM)

| 파라미터 | 의미 |
|----------|------|
| `crop_top_%` | 상단 제외 (≈ 전방 `x_max`) |
| `x_min_cm` / `x_max_cm` | 근·원거리 |
| `y_half_cm` | 횡반폭 (**기본 77**) |
| `mpp_mm` | 등방 mm/px |
| `pitch_x10` / `height_cm` | 카메라 미세 조정 |

**UI:** 원본(crop 선) · Metric BEV(미터 보조선). `s` 저장, `f` 풀폭 스냅, `q` 종료.  
`--compare` 시 사다리꼴 창 추가.
---

## 5. 색·모드·다중 경로

### 필수 — 분기에서 **두 차로 분리 감지**

차선이 둘로 갈라지면 평균 중심선으로 합치지 말고 **경로 후보를 각각** 유지한다. 선택은 모드/FSM.  
원태 현재는 색당 L/R **한 쌍** → 동일색 이중 분기는 Phase 3 보강.

### 마스크 / 모드

흰·노란 차선, 검·빨 차로. `LANE_FOLLOW` / `FORK_*` / `ROUNDABOUT_*` / `OBSTACLE_ZONE`.

```
lane_detection → LaneDetections
lane_planner   → path_candidates[] → mode 선택 → LaneResult
```

---

## 6. 제어·스무딩

현재 구현은 Metric IPM 점열을 rear-axle 좌표로 옮긴 뒤 PP + 경로 heading +
bounded CTE 보정을 사용한다. LD와 throttle은 경로 곡률로 연속 조절하며,
조향 변화율과 경로 유실 복귀율은 초 단위다. 상세 수식·파라미터·알려진
분기 경로 한계는 [main-planner.md](./main-planner.md)를 따른다.

---

## 7. 구현 단계

| Phase | 내용 | 담당 | 산출물 | 상태 |
|-------|------|------|--------|------|
| **0** | Metric IPM 튜닝 + 캡처 + 캘리브 매트 | 안승현 | `tune_bev.py`, `metric_ipm`, `bev_calib_mat` | **시뮬 잠정** |
| **1** | **HSV 공용 튜너** (시뮬·실차) · 값 정밀화는 원태 협업 | **안승현**(툴) / 원태(최종값) | `tune_hsv.py`, `hsv:` yaml | **진행** |
| **2** | 흰 차선 `lane_planner` + 제어 게인 튜너 | **안승현** | `lane_planner.py`, `tune_lane_control.py`, Gazebo 단경로 | **진행** |
| **3** | 노란·분기 이중 경로 + 모드 | 안승현 (+원태) | 갈림길·교차로 | 대기 |
| **4** | (선택) PP / Stanley | 안승현 | 제어 교체 | 대기 |

> Phase 1: 승현이 **시뮬·실차 공용 HSV 트랙바 툴**을 소유하고 yaml에 저장.  
> 대회용 **정밀 최종값**은 원태가 인지 파이프라인과 맞춰 다듬는다.  
> 원태 인지 merge 전엔 `lane_detection`에 **흰 차선 임시 스텁**을 두고, merge 후 detection만 교체한다.

---

## 8. 레포 배치

| 경로 | 역할 |
|------|------|
| `scripts/vision_tune/` | IPM·캡처·**HSV**·제어 게인 트랙바 |
| `config/lane_vision.yaml` | ROI/IPM/HSV |
| `config/lane_control.yaml` | planner P/EMA/rate/look-ahead |
| `modules/lane_detection.py` | 인지 (장원태; 당분간 흰 스텁) |
| `modules/lane_planner.py` | 경로·조향 (안승현) |

---

## 9. 튜닝 툴

| 툴 | 시점 | 기능 |
|----|------|------|
| **`tune_bev.py`** | Phase 0 | **기본 진입** → Metric IPM |
| `tune_metric_ipm.py` | Phase 0 | IPM UI (tune_bev가 호출) |
| `tune_bev_roi.py` | Phase 0 | 사다리꼴만 (참고) |
| `capture_camera.py` | Phase 0 | 토픽 캡처 |
| **`tune_hsv.py`** | Phase 1 | **시뮬·실차 공용** HSV (흰/노란/검/빨) · 클릭 샘플 |
| **`tune_lane_control.py`** | Phase 2 | **시뮬·실차 공용** 제어 게인 (P/EMA/rate/cruise/look-ahead) |

---

## 10. External·팀 자산 참고점

### 10.1 `feature/wontae-lane` (`0cbcdbe`)

인지 전용 `LaneDetections`. **런타임 BEV = Metric IPM.** 최종 조향은 `MainPlanner`. `LANE_VISUALIZE` 기본 off. 갈림길 셀 분기·한쪽선 L/R은 승현(임시)이 이어서 검증.

### 10.2 `limo_sim_code_v2`

EMA·rate limit·감속 → planner 손맛.

### 10.3 `limo_minyong`

슬라이딩 윈도우·hold/decay 폴백.

### 10.4–10.6

wego DL 보류. D-Racer-Kit은 I/O만. F1TENTH/Stanley 관행은 Phase 2–4.

---

## 11. 성공 기준

| Phase | Done when |
|-------|-----------|
| 0 | Metric IPM 잠정 SSOT (`y_half=0.77`, 전방 1.5 m) · 사다리꼴 참고 · 캘리브 매트 |
| 1 | `tune_hsv`로 `hsv.white`(및 필요 시 yellow) yaml 저장 · 시뮬 마스크 사용 가능 |
| 2 | 단일색 추종 + `--drive` 튜너 + Ctrl+C 정지 · **L/R 편측 오할당은 미해결 (§11.6)** |
| 3 | 분기 후보 2개 + 모드 전환 |

---

## 11.5 Gazebo 차선 추종 테스트 (MainPlanner · Out 흰 / In 노란우선)

**런타임 SSOT:** `inference_node` + `MainPlanner` (`config/main_planner.yaml`).  
레거시 `lane_control_node` / `lane_follow_color` / ROS `-p cruise_throttle:=` 는 **auto 경로에서 사용하지 않음**.

색은 **`route_mode`**: Out=`out`(흰만) · In=`in`(노란 우선). 속도는 YAML `speed.cruise_throttle`(시뮬 잠정 **0.33**).

### 준비

```bash
# 터미널1 — 시뮬 (기본 view:=both · fork만 끌 때 view:=none)
./scripts/dev_container.sh sim-bringup

# HSV가 비어 있으면 (컨테이너 셸)
docker exec -it 2026-smh-sim bash -lc \
  'source /opt/ros/humble/setup.bash && source install/setup.bash && \
   python3 scripts/vision_tune/tune_hsv.py --channel white'
```

### Out 흰 차로 (기본)

```bash
./scripts/dev_container.sh sim-auto route_mode:=out viz:=lane
```

### In 노란 우선 (회전교차로)

```bash
./scripts/dev_container.sh sim-bringup spawn_pose:=in_roundabout_entry
./scripts/dev_container.sh sim-auto route_mode:=in viz:=lane
```

게인·cruise 변경: `config/main_planner.yaml` 편집 후 **sim-auto만** 재시작.  
벤치: `scripts/drive_test/` (`mask_policy_bench.py`, `out_lap_bench.py` — **sim-auto OFF**).

### 레거시 (실행 금지 · 참고만)

`tune_lane_control.py` / `lane_control.yaml` / `lane_control_node` 는 예전 PP 실험용.  
MainPlanner와 `/control`이 **충돌**하므로 auto와 동시 실행하지 말 것.

### 정지

- **Ctrl+C** (sim-auto) → `/control` throttle 0 발행 후 종료.
- 조이스틱 E-Stop.

### 확인

```bash
ros2 topic echo /control --once
ros2 topic echo /debug/planner   # state · fork_on · tracker
```

---

## 11.6 시뮬 주행 메모 · 핸드오프 (개정 2026-07-15)

### 현재 main (요약)

- **NORMAL tracker:** `mask_p` + `mask_pursuit` 코리도 (`corridor_mode: hard`)
- **Out 갈림:** 표지 게이트 (`out_fork_require_sign`) · `forced_turn`은 rank만
- **In 탈출:** `yellow_alt` / tip_mode 분리 — [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md)
- 제어·미션: [main-planner.md](./main-planner.md)

### 알려진 이슈 (역사 · Phase 2 시절)

우회전 코너에서 편측 L/R 오할당이 관측됐음 → parallel-rail / active_lane / mask 코리도로 완화 중.  
완전 해결 여부는 spawn 단위·랩 벤치로 재확인.

### 다음에 할 일 (팀)

| 우선 | 담당 | 내용 |
|------|------|------|
| **P0** | 승현 / 원태 | Out 표지 hold·In tip 안정화 · [fork-test-pipeline.md](./fork-test-pipeline.md) |
| P1 | 서준 | `main_planner.yaml` 게인 · 실차 `planner_profile: real` |
| P1 | 원정 | 표지·신호 입력과 Out fork 게이트 정합 |
| P2 | 전원 | 실차 latency — [board-latency-and-sim2real.md](./board-latency-and-sim2real.md) |

---

## 12. 결정 요약

1. **인지(원태) / 판단·제어(안승현)** — `LaneDetections` → planner → `LaneResult`  
2. **BEV SSOT:** Metric IPM (`y_half=0.77`, `x_max=1.5`) · 기본 툴 `tune_bev.py`  
3. **단일색 추종:** 흰 기본 · 노란 단독 · Ctrl+C 즉시 정지 · `--drive` 게인 튜너  
4. **열린 이슈:** 코너 편측 시 **L/R 오할당** (§11.6) — 인지 수정이 우선  
5. **순서:** HSV · closed-loop → **L/R 안정화** → 원태 merge → 다중경로  
