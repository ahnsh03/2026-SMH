# Sign / signal weights (board/race-control)

| File | Role | Source |
|------|------|--------|
| `sign_best.onnx` | Left/Right **direction signs only** (2-class) | `origin/board` |
| `sign_light_best_v5b.onnx` | 4-class YOLO; runtime uses **Red/Green Light only** (cls 2/3) | `feature/seongjun-sign-traffic` |

Sungjun의 표지판 클래스(0/1)는 쓰지 않습니다. 표지판은 항상 `sign_best.onnx`.

## Traffic light backend

```bash
export TRAFFIC_LIGHT_BACKEND=yolo_then_opencv   # default
# opencv | yolo | yolo_then_opencv | opencv_then_yolo

PYTHONPATH=src/inference python3 scripts/check_traffic_light_ab.py --webcam 0
```

트랙에서 A/B 후, 잘 되는 쪽으로 `TRAFFIC_LIGHT_BACKEND`만 고정하면 됩니다.
