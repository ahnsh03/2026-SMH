# Sign / signal weights (board/race-control)

| File | Role | When loaded |
|------|------|-------------|
| `sign_best.onnx` | Left/Right **direction signs** (2-class) | **Always** (race) |
| `sign_light_best_v5b.onnx` | 4-class; runtime uses **Red/Green only** | Only if `TRAFFIC_LIGHT_BACKEND` ≠ `opencv` |

## CPU policy (D3-G)

**프레임당 YOLO는 최대 1개.** 표지판이 필수이므로 기본은:

* 표지판 = `sign_best.onnx`
* 신호등 = OpenCV HSV (`TRAFFIC_LIGHT_BACKEND=opencv`)

신호등 YOLO는 정지 상태 A/B (`check_traffic_light_ab.py`)용으로만 켠다.  
레이스에서 light YOLO를 켜면 표지판 YOLO와 **둘이 같이** 돌아가 지연이 커질 수 있다.

```bash
# race default (env optional — code default is already opencv)
export TRAFFIC_LIGHT_BACKEND=opencv

# A/B only (intentionally runs a second ONNX)
export TRAFFIC_LIGHT_BACKEND=yolo
PYTHONPATH=src/inference python3 scripts/check_traffic_light_ab.py --webcam 0
```
