# 보드(실차) 동결 — 저속 mask_hard_wide + 갈림 L/R

시뮬에서 검증한 제어를 보드에 올릴 때 **이 YAML이 SSOT**입니다.

- 파일: `config/main_planner.yaml`
- NORMAL: `tracker.normal: mask_p` + hard corridor 0.38
- 속도: `cruise_throttle: 0.28`, `curve_throttle: 0.18`
- mask: `steer_k: 2.0`, `alpha: 0.40`, `near_band: 0.85`, `fork_force_pp: true`
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

## 보드 적용

1. 위 `main_planner.yaml`을 보드 워크스페이스에 동기화 (기존처럼 repo/`config` 배포).
2. 실차 기하만 보드에서 보정: `pure_pursuit.wheelbase_m` / `max_steer_angle_rad`  
   (D-Racer L=0.175 → δ≈0.4266 — `docs/vehicle-geometry.md`).
3. `STEER_TRIM`은 `config/vehicle_config.yaml` (웹/조이스틱 트림).
4. 실차에서는 속도·트림만 만지고, 갈림 분리 로직은 시뮬 동결본을 유지.

오프라인 스모크: `--mode offline --scenario all` (프레임 기반 L/R layer 확인).
