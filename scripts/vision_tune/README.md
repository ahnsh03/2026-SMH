# Vision tuning tools (Phase 0+)

시뮬·실차 공통. 설계: [docs/lane-drive-strategy.md](../../docs/lane-drive-strategy.md).

## 요구

- Python3, OpenCV, PyYAML
- 토픽 모드: ROS2 Humble + `sensor_msgs` (컨테이너 `2026-smh-sim` 또는 보드)

## Phase 0

### BEV ROI 트랙바

```bash
# 정지 영상
python3 scripts/vision_tune/tune_bev_roi.py --image /path/to.png

# 캡처 폴더
python3 scripts/vision_tune/tune_bev_roi.py --folder data/captures/sim

# 라이브 카메라 (bringup / camera_node 실행 중)
python3 scripts/vision_tune/tune_bev_roi.py --topic /camera/image/compressed
```

- 창: `bev_tune_origin` · `bev_tune_roi` · `bev_tune_bev` · 트랙바
- `crop_top_%`: 상단 제외
- `bottom_half_%`: 아랫변 반폭 비율 (100=이미지 폭, 135=프레임 밖 확장)
- `s` → `config/lane_vision.yaml` 저장

### 카메라 캡처

```bash
# 자동 저장 (10프레임마다)
python3 scripts/vision_tune/capture_camera.py --out data/captures/sim --every 10 --count 20

# 미리보기 + c 키로 저장
python3 scripts/vision_tune/capture_camera.py --out data/captures/sim --preview
```

보드에서는 `--out ~/captures/board` 등으로 저장한 뒤 PC로 가져와 BEV를 튜닝하면 된다.
