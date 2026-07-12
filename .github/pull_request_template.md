## Summary

<!-- 무엇을 변경했는지 한 줄 요약 -->

## Git 규약 확인

- [ ] `main`에서 **feature 브랜치**를 생성해 작업했습니다 (`main` 직접 push 아님)
- [ ] PR 전 `git pull origin main` 또는 rebase로 main과 맞췄습니다
- [ ] [docs/lane-perception-topic.md](docs/lane-perception-topic.md) 구조를 읽고 담당 위치를 확인했습니다

## 담당 모듈

- [ ] `lane_detection.py` (장원태) — 인지 only, Metric IPM, 조향 없음
- [ ] `traffic_sign.py` (장원정)
- [ ] `aruco/detector.py` (안승현)
- [ ] `aruco/stop_logic.py` (박성준)
- [ ] `roundabout.py` / 경로 추종 (양서준) — `/perception/lane` → types/adapters
- [ ] `lane_planner.py` (안승현)
- [ ] 통합 (`types.py`, `lane_adapters.py`, `pipeline.py`, 노드) — 팀장 only

## 변경 범위 확인

- [ ] **담당 파일만** 수정했습니다 (`docs/collaboration.md` 참고)
- [ ] `pipeline.py` / `types.py` / `lane_adapters.py` / 노드를 수정하지 않았습니다 (통합 PR이 아닌 경우)
- [ ] `/control`을 인지 모듈에서 발행하지 않습니다
- [ ] (원태) 런타임 BEV를 사다리꼴(`warp_bev`)로 되돌리지 않았습니다

## 테스트

- [ ] PC 또는 보드에서 `colcon build --symlink-install --packages-up-to inference` 성공
- [ ] (시뮬) `sim_auto_driving` 또는 `inference_node`+`lane_control_node`로 `/control` 확인
- [ ] (보드) `./scripts/board_sync.sh --no-pull` 성공
- [ ] (보드, 가능 시) `ros2 launch inference auto_driving.launch.py` 실행 확인

## 스크린샷 / 로그 (선택)

<!-- topic echo, rqt/graph, 주행 로그 등 -->
