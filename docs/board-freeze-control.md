# 보드(실차) 동결 — 저속 mask_hard_wide + 갈림 L/R

시뮬에서 검증한 제어를 보드에 올릴 때 **이 YAML이 SSOT**입니다.

- 파일: `config/main_planner.yaml`
- NORMAL: `tracker.normal: mask_p` + hard corridor 0.38
- 속도: `cruise_throttle: 0.28`, `curve_throttle: 0.18`
- mask: `steer_k: 2.0`, `alpha: 0.40`, `near_band: 0.85`, `fork_force_pp: true`
- Phase A: `track_state` 횡오프셋 EMA + jump reject + 센터라인 half-width hold
- Phase B A/B: `tracker.normal: stanley` (κ FF 포함) — **YAML SSOT는 mask_p 유지**
  (오프라인 synthetic은 stanley 쪽으로 기울 수 있음 → `out_lap_bench` 확정 전 flip 금지)
- Fork: 표지 후 / `forced_turn` 시 선택 branch PP만 추종 (LEFT=0, RIGHT=1)

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

모니터링: OpenCV `Lane drive`, `Fork select` · 스냅 `data/captures/fork_drive_logs/<stamp>/r00_*/snap_*.png`

## 보드 적용 (Phase C)

1. `main_planner.yaml`을 보드에 동기화 (갈림·mask corridor 유지).
2. 실차 기하·속도는 [`config/main_planner.real_car.yaml`](../config/main_planner.real_car.yaml) 값을
   `pure_pursuit` / `speed` / `track_state`에 반영:
   - `wheelbase_m: 0.175`, `max_steer_angle_rad: 0.4266`
   - cruise를 시뮬보다 낮게 시작 (`0.22`)
3. `STEER_TRIM`은 `config/vehicle_config.yaml` (웹/조이스틱 트림).
4. 직진 hunting이 남으면만 `track_state.delay_pred_sec`를 `0.06~0.12`로 켬.
5. 실차에서는 속도·트림·delay만 만지고, 갈림 분리 로직은 시뮬 동결본을 유지.

### tracker A/B (시뮬)

```bash
# 짧은 구간 steer_rms
python3 scripts/drive_test/mask_steer_bench.py \
  --variants mask_p_hard_wide,stanley_soft --segments start,out_in_merge

# OUT 랩 패밀리
python3 scripts/drive_test/out_lap_bench.py --families mask_p,stanley

# 오프라인 synthetic A/B (Gazebo 없이 steer_rms)
python3 scripts/drive_test/tracker_ab_offline.py
```

오프라인 스모크: `fork_spawn_unit.py --mode offline --scenario all` (프레임 기반 L/R layer 확인).
