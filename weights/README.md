# Sign / signal weights (board/race-control)

| File | Role | When loaded |
|------|------|-------------|
| `sign_best.onnx` | Left/Right **direction signs** (2-class) | **Always** (race) |
| `sign_light_best_v5b.onnx` | Red/Green light YOLO | **Unused** (lights off) |

Synced from `main` @ `7cfb063` (wonjung58, 2026-07-16) — SHA256 `3f66a0b6…15b21657`.

**Traffic lights (OpenCV + YOLO) are disabled** in runtime. Direction-sign YOLO
still runs once per frame.

## Start gate

`require_green_to_start: true` waits, then after `green_wait_timeout_sec`
(default **15 s**) assumes green and starts. Mid-track: `traffic_pass:=true`.
