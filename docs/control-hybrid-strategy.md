# 하이브리드 제어 전략 — mask ↔ paint 상호보완

> 개정: 2026-07-16 (안승현) — ego-blob SSOT · fork capture/judgment 반영 후 **보드 실차 검증용**  
> 런타임 코드 SSOT: [`main-planner.md`](./main-planner.md) · YAML: `config/main_planner.yaml`  
> 인지·갈림: [`hsv-profiles.md`](./hsv-profiles.md) · [`lane-occlusion-fork-strategy.md`](./lane-occlusion-fork-strategy.md) §5.1.4 · [`out-ego-fork-shape.md`](./out-ego-fork-shape.md)  
> 보드 동결·적용: [`board-freeze-control.md`](./board-freeze-control.md)

**전제:** 지금까지 넣은 제어기(`mask_p` / Stanley / hybrid / soft corridor)는 **실차에서 아직 검증되지 않음.**  
아래는 인지 분석(좁은 차로·낮은 카메라·ego blob·fork 포착)을 바탕으로 한 **우선순위 제안**이다.

---

## 1. 환경 제약 (왜 하이브리드인가)

| 제약 | 제어 함의 |
|------|-----------|
| 차로폭 ~0.35 m (`half_width` 0.175 m) | CTE 여유 거의 없음 → 큰 게인·지연은 즉시 가드 침범 |
| 카메라 낮음 + Metric IPM short horizon | 커브에서 **far paint 소실**, near만 남음 |
| 곡선에서 한쪽/양쪽 차선 소실 | paint-only PP/Stanley는 heading 손실 |
| 갈림에서 area COM | L/R 면적 평균 → **잘못된 갈래** — fork 시 COM 금지 |
| 원형에서 road COM만 | 링 이탈 → CIRCLE은 **yellow paint PP** |
| soft / far_blend corridor | 과거 코너 **lag**로 기각 |

**결론:** “차선만”도 “마스크만”도 부족. **역할 분리한 상호보완**이 맞다.

---

## 2. 신호 역할 분리

```
● 주행가능영역 (ego blob / DT-strip)     ← hsv-profiles · near-band SSOT
  - “어디에 서 있어도 안전한가” → 횡방향 오차 (COM / near mid)
  - paint가 깨져도 near에 면적이 있으면 유지

● 차선 paint (white OUT / yellow IN)
  - “길이 어디로 꺾이는가” → heading · 곡률 · lookahead
  - fork L/R geometry · CIRCLE 링 추종

● 미션 게이트 (judgment — §5.1.4)
  - OUT: 표지 ∧ out_fork_capture → branch PP
  - IN:  in_circle_fork_moment pass1 우(유지) / pass2 좌(탈출) → branch PP
```

| 구간 | 주 신호 | 보조 |
|------|---------|------|
| NORMAL (직선·완만 커브) | **mask COM (ego blob)** | paint mid 약 pull (검증 후) · `track_state` |
| paint 소실 커브 | mask + occlusion hold | 짧은 paint fallback |
| `FORK_TURN` / `ROUNDABOUT_EXIT` | **선택 branch PP** | mask 끄기 |
| `ROUNDABOUT_CIRCLE` | **yellow PP** | moment 후 rank면 해당 갈래 PP; COM 금지 |

---

## 3. 권장 레이어 (코드 매핑)

### Layer A — NORMAL ★실차 1순위

| 항목 | 설정·동작 |
|------|-----------|
| tracker | `tracker.normal: mask_p` |
| 입력 | `LaneDetections.drivable_area` (= ego blob SSOT) |
| 법칙 | `mask_pursuit.steer_law: sim_v2`, `center_mode: area` |
| 안정화 | `track_state` EMA + jump reject + `half_width_m: 0.175` hold |
| corridor | **T1–T2: `off`**. soft/hard는 paint 양쪽 안정 시에만 A/B |
| fork | `fork_force_pp: true` — forkish면 COM 대신 색/갈래 PP |
| 소실 | `occlusion_hold_frames` → 직전 조향 유지 → 짧게 paint PP |

### Layer B — 특수 구간 (FSM)

| 상태 | 조향 |
|------|------|
| `FORK_TURN` | `branches[rank]` PP (`_selected_layer_path` / ego follow) |
| `ROUNDABOUT_CIRCLE` | `circle_tracker: pp` (yellow). moment가 rank 고르면 그 갈래 |
| `ROUNDABOUT_EXIT` | 좌 탈출 branch PP (`exit_branch_rank: 0`) |
| blob+paint 동시 약함 | throttle↓ + last steer hold |

Arm 정책 코드: `perception/fork/judgment.py` · 플래너 `_fork_perception_allowed` / `_apply_in_moment_pass`.

### Layer C — 후순위 (검증 후에만)

- Stanley / hybrid 전체 flip (`board-freeze-control`·`out_lap_bench` 전 **금지**)
- OUT soft S-curve corridor 부활
- 갈림 진입 outer±half_w vs legacy pairs (포착·judgment는 완료, **진입 기하만** A/B)

---

## 4. 의사코드 (운영 규칙)

```
if FORK or EXIT or (CIRCLE and paint_ok):
    steer = PP(selected_paint_or_branch)
elif CIRCLE and paint_weak:
    steer = hold + slow          # COM으로 링 추정 금지
elif NORMAL:
    e = mask_COM(ego_blob)       # 주
    if both_rails_visible:       # 보조 — T3 이후
        e = blend(e, paint_mid, α_small)
    steer = P(e) + track_state
    if blob_occlusion:
        hold / brief paint fallback
```

---

## 5. 실차 테스트 순서 (보드 피드백용)

각 단계 **성공 기준 1개**. 실패 시 다음으로 넘기지 말 것.

| # | 무엇 | 조건 | 성공 | 실패 시 |
|---|------|------|------|---------|
| **T0** | HSV·ego blob 육안 | 정차 BEV | near blob이 차로만 | HSV/morph |
| **T1** | **mask_p만** 저속 직진 | OUT 5–10 m | 중앙±~5 cm, hunting↓ | `steer_k` / `err_alpha` / deadband |
| **T2** | mask_p 완만 커브 | OUT S 입구 | 이탈·반대편 흡인 없음 | occlusion hold · curve throttle |
| **T3** | paint soft pull ON/OFF | 같은 T2 | 도움이면 채택 | 흔들리면 OFF |
| **T4** | CIRCLE yellow PP | IN ½바퀴 | 링 유지 | COM 혼입 여부 |
| **T5** | moment→우 유지 1회 | 탈출 갈림 1st | 원 재진입 | `in_fork_pass_count` / rank |
| **T6** | 좌 탈출 | 2nd 갈림 | 합류 | branch PP |
| **T7** | 표지∧capture→갈림 | out_fork | 표지 방향 진입 | `fork_arm_reason` · capture |

### T1–T2 보드 시작 YAML 요지

```yaml
tracker:
  normal: mask_p
mask_pursuit:
  corridor_mode: off          # T1–T2
  fork_force_pp: true
  steer_law: sim_v2
track_state:
  enable_path_hold: true
  half_width_m: 0.175
speed:
  cruise_throttle: 0.18~0.22  # 먼저 느리게; real_car overlay 병행
roundabout:
  circle_tracker: pp
  in_exit_use_moment: true
route:
  out_fork_require_sign: true
  out_fork_require_capture: true
```

실차 기하: [`main_planner.real_car.yaml`](../config/main_planner.real_car.yaml) (`wheelbase` 0.175 등) + [`board-freeze-control.md`](./board-freeze-control.md).

### 피드백에 남기면 좋은 것 (짧게)

1. 직진: 좌/우 치우침 · hunting 주기  
2. 커브: paint 소실 구간에서 **안쪽 말림 / 바깥 이탈**  
3. BEV 한 장: near ego blob이 차로 폭을 얼마나 채우는지  

디버그: `/debug/planner` → `fork_arm_reason`, `out_fork_capture`, `in_circle_fork_moment`, `in_fork_pass_count`, `path_source`.

---

## 6. 다른 브랜치에서 이어갈 때

| 갈래 | 이 브랜치에서 가져갈 것 | 보드에서 할 일 |
|------|-------------------------|----------------|
| 제어 튜닝 | Layer A YAML · T1–T3 | mask_p 실차 루프 |
| 갈림 진입 | capture + judgment | T5–T7 · L/R 기하 A/B |
| HSV | `real_car` lock · morph 3/13 | T0만 확인 후 건드리지 말 것 |

관련 코드:

- `perception/fork/{ego_shape,capture,judgment,moment}.py`
- `pipeline.MainPlanner` (`_fork_perception_allowed`, `_apply_in_moment_pass`, `_mask_com_pursuit`)
- 오프라인: `scripts/vision_tune/score_out_fork_capture.py`
