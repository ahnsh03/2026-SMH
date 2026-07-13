## Summary

<!-- 무엇을 변경했는지 한 줄 요약 -->

## Git 규약 확인

- [ ] `main`에서 **feature 브랜치**를 생성해 작업했습니다 (`main` 직접 push 아님)
- [ ] PR 전 `git pull origin main` 또는 rebase로 main과 맞췄습니다
- [ ] [docs/lane-perception-topic.md](docs/lane-perception-topic.md) · [docs/main-planner.md](docs/main-planner.md) · [docs/roles.md](docs/roles.md)를 확인했습니다

## 담당 모듈

- [ ] `lane_detection.py` (안승현 임시 / 장원태) — 인지 only, Metric IPM, 조향 없음
- [ ] `traffic_sign.py` (장원정)
- [ ] `aruco/detector.py` (안승현)
- [ ] `aruco/stop_logic.py` (박성준)
- [ ] `pipeline.py` / MainPlanner / `main_planner.yaml` (양서준)
- [ ] 통합 (`types.py`, `lane_adapters.py`, `inference_node.py`) — 팀장 only

## 변경 범위 확인

- [ ] **담당 파일만** 수정했습니다 (`docs/collaboration.md` 참고)
- [ ] 통합 파일이 아니면 `types.py` / `lane_adapters.py` / `inference_node.py`를 수정하지 않았습니다
- [ ] 인지 모듈은 결과만 반환하고 최종 `/control`은 `inference_node`의 **MainPlanner 하나만** 발행합니다
- [ ] `lane_control_node`를 MainPlanner와 동시에 실행하지 않습니다
- [ ] 런타임 BEV를 사다리꼴(`warp_bev`)로 되돌리지 않았습니다

## 테스트

- [ ] PC 또는 보드에서 `colcon build --symlink-install --packages-up-to inference` 성공
- [ ] (시뮬) `sim_auto_driving.launch.py`로 `/control` 확인
- [ ] (인지) `LANE_VISUALIZE=control` 또는 `on`으로 창 검증 (해당 시)
- [ ] (보드) `./scripts/board_sync.sh --no-pull` 성공
- [ ] (보드, 가능 시) `ros2 launch inference auto_driving.launch.py` 실행 확인

## 스크린샷 / 로그 (선택)

<!-- topic echo, LANE_VISUALIZE 창, 주행 로그 등 -->
