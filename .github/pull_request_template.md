## Summary

<!-- 무엇을 변경했는지 한 줄 요약 -->

## 담당 모듈

- [ ] `lane_detection.py` (장원태)
- [ ] `traffic_sign.py` (장원정)
- [ ] `aruco/detector.py` (안승현)
- [ ] `aruco/stop_logic.py` (박성준)
- [ ] `roundabout.py` (양서준)
- [ ] 통합 (`pipeline.py`, `types.py`) — 팀장 only

## 변경 범위 확인

- [ ] **담당 파일만** 수정했습니다 (`docs/collaboration.md` 참고)
- [ ] `pipeline.py` / `inference_node.py` 는 수정하지 않았습니다 (통합 PR이 아닌 경우)

## 테스트

- [ ] PC 또는 보드에서 `colcon build --symlink-install --packages-select inference` 성공
- [ ] (보드) `./scripts/board_sync.sh --no-pull` 성공
- [ ] (보드, 가능 시) `ros2 launch inference auto_driving.launch.py` 실행 확인

## 스크린샷 / 로그 (선택)

<!-- 주행 테스트 결과, rqt/graph 스크린샷 등 -->
