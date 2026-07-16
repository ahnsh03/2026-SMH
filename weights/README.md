# Sign / signal weights (board/race-control)

| File | Role | When loaded |
|------|------|-------------|
| `sign_best.onnx` | Left/Right **direction signs** (2-class) | **Always** (race) |
| `sign_light_best_v5b.onnx` | 4-class; runtime uses **Red/Green only** (classes 2/3) | **Always** (YOLO lights) |

Synced from `main` @ `7cfb063` (wonjung58, 2026-07-16) — SHA256 `3f66a0b6…15b21657`.

Light model matches **team-new** `SignLightYolo` (`sign_light_best_v5b.onnx`, conf 0.35).

**OpenCV HSV traffic-light detection is disabled** (false positives on nearby colors).

## Start gate

`require_green_to_start: true` waits for YOLO green. If none within
`green_wait_timeout_sec` (default **30 s**), the planner assumes green and starts.
