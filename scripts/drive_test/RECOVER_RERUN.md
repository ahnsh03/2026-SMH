# OUT/IN 보드 동결 — 재실행

제어 SSOT: **저속 `mask_p` hard_wide** (`docs/board-freeze-control.md`).

```bash
# 갈림 L/R 라이브 (텔레포트 + 주행 + 시각화)
PYTHONUNBUFFERED=1 python3 scripts/drive_test/fork_spawn_unit.py \
  --mode live --scenario all --duration 8 --viz control --drive
```

| 시나리오 | 스폰 |
|----------|------|
| `out_left` / `out_right` | `out_fork` |
| `in_exit_left` / `in_exit_right` | `in_roundabout_exit` |

창: `Lane drive`, `Fork select` · 로그: `data/captures/fork_drive_logs/`
