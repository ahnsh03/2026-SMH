# 보드(실차) 동결 — mask_p + 갈림 게이트 + 실차 T1–T7

> 개정: 2026-07-16 — 하이브리드 제어 전략·fork judgment 반영.  
> **제어 설계 SSOT:** [`control-hybrid-strategy.md`](./control-hybrid-strategy.md)  
> YAML: `config/main_planner.yaml` (+ 실차 [`main_planner.real_car.yaml`](../config/main_planner.real_car.yaml))

---

## 현재 보드 권장 (T1–T2 시작점)

| 항목 | 값 |
|------|-----|
| NORMAL | `tracker.normal: mask_p` · **corridor_mode: off** (soft/hard는 T3+) |
| 입력 | ego blob `drivable_area` (near-band HSV SSOT) |
| mask | `steer_law: sim_v2`, `steer_k: 2.0`, `steer_alpha: 0.40`, `fork_force_pp: true` |
| track_state | EMA + jump reject + `half_width_m: 0.175` hold |
| 속도 | 시뮬 cruise보다 낮게 시작 (`0.18`–`0.22`); real_car overlay |
| CIRCLE | `circle_tracker: pp` (yellow) — COM 금지 |
| OUT 갈림 arm | **표지 ∧ `out_fork_capture`** (`out_fork_require_capture: true`) |
| IN 탈출 | moment pass1 **우 유지** / pass2 **좌 탈출** (`in_exit_use_moment`) |
| Fork 조향 | 선택 branch PP만 (LEFT=0, RIGHT=1). mask COM 미사용 |

Stanley / hybrid / soft corridor 전체 flip은 **T1–T2 통과 전 금지**.

---

## 실차 검증 순서 (요약)

자세한 성공 기준: [`control-hybrid-strategy.md` §5](./control-hybrid-strategy.md).

| # | 초점 |
|---|------|
| T0 | HSV·ego blob 육안 |
| **T1** | mask_p 저속 직진 |
| **T2** | mask_p 완만 커브 |
| T3 | paint soft pull A/B |
| T4–T6 | CIRCLE PP · IN keep/exit |
| T7 | OUT 표지∧capture → 갈림 PP |

---

## 보드 적용 (Phase C)

1. 이 브랜치(`feature/seunghyun-recover-pre-pdc` 또는 merge 후)를 보드에 sync.
2. `main_planner.yaml` + `main_planner.real_car.yaml` 기하·속도 반영:
   - `wheelbase_m: 0.175`, `max_steer_angle_rad: 0.4266`
   - cruise를 시뮬보다 낮게
3. `STEER_TRIM` → `config/vehicle_config.yaml`.
4. hunting만 남으면 `track_state.delay_pred_sec` `0.06~0.12`.
5. **HSV · morph · near-band select**는 잠금본 유지 (T0 이상이면 제어만 건드림).
6. 다른 feature 브랜치에서 제어/갈림 진입만 파생 개발·검증.

### 디버그 필드 (`/debug/planner`)

`fork_arm_reason` · `out_fork_capture` · `in_circle_fork_moment` · `in_fork_pass_count` · `path_source` · `fork_perception`

---

## 갈림 확인 (시뮬 재현)

```bash
# bringup ON, sim-auto OFF
PYTHONUNBUFFERED=1 python3 scripts/drive_test/fork_spawn_unit.py \
  --mode live --scenario all --duration 8 --viz control --drive
```

| 시나리오 | 스폰 | 기대 |
|----------|------|------|
| out_left / out_right | `out_fork` | left/right_branch |
| in_exit_left / in_exit_right | `in_roundabout_exit` | left/right_branch |

오프라인 capture:  
`PYTHONPATH=scripts/vision_tune:src/inference python3 scripts/vision_tune/score_out_fork_capture.py --from-bag out --stride 5`

### tracker A/B (시뮬 — T1 이후)

```bash
python3 scripts/drive_test/mask_steer_bench.py \
  --variants mask_p_hard_wide,stanley_soft --segments start,out_in_merge
python3 scripts/drive_test/out_lap_bench.py --families mask_p,stanley
```

`out_lap_bench` 확정 전 Stanley YAML flip 금지.
