# Sign / signal weights

| File | Role |
|------|------|
| `sign_best.onnx` | Left/Right fork **direction signs** (YOLO26n, imgsz 416) — used by `direction_sign` |

Traffic **lights** (red/green) use HSV in `modules/trafficsign/color_detector.py` (no neural weights).

Place `sign_best.onnx` here (repo root `weights/`), or set `SIGN_MODEL_PATH`.

```bash
# quick check on board
SIGN_MODEL_PATH=$PWD/weights/sign_best.onnx python3 scripts/check_sign_webcam.py
```
