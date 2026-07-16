# Board-race tests

기능별 단위테스트 + 트랙사이드 모니터/터미널 디버그.

## 폴더

```
test/
├── conftest.py
├── unit/                         ← pytest (기능별)
│   ├── perception/
│   │   ├── blob/                 compose_road · rail · detect smoke
│   │   └── fork/                 judgment · moment · lane_pairs
│   ├── planner/                  MainPlanner · PP · active_lane
│   ├── signs/                    traffic · direction
│   └── aruco/
├── integration/                  ROS msg adapters
└── manual/                       라이브 점검 (pytest 수집 안 함)
```

## 단위테스트 실행

```bash
# 보드 / PC worktree
./scripts/board_test.sh
./scripts/board_test.sh fork
./scripts/board_test.sh planner
./scripts/board_test.sh -k out_arm

# 또는
cd src/inference && PYTHONPATH=. pytest
```

Docker(개발 이미지) 예:

```bash
docker run --rm -v "$PWD":/board -w /board/src/inference --entrypoint bash \
  2026-smh-dev:latest -lc 'PYTHONPATH=. pytest'
```

## 모니터 · 터미널 디버그

| 경로 | 용도 |
|------|------|
| **Web** `http://<보드IP>:5000` | 카메라 + White / IN ego / OUT ego |
| **Launch** `debug_monitor.launch.py` | 디버그 기본값(`traffic_pass`, BEV on) |
| **Terminal** `scripts/board_monitor_term.py` | `/debug/planner` · `/control` · `/debug/aruco` |
| **Sign probe** `scripts/check_sign_topic.py` | 표지판만 터미널 |

```bash
# A: 주행 + 웹 모니터
ros2 launch inference debug_monitor.launch.py route_mode:=out

# B: 터미널 대시보드 (+ BEV hz)
python3 scripts/board_monitor_term.py --hz
```

대회 시 BEV 끄기: `publish_bev_debug:=false`  
또는 `auto_driving.launch.py` + 동일 터미널 스크립트.

## manual 라이브 스크립트

```bash
python3 src/inference/test/manual/check_traffic_sign_topic.py
python3 src/inference/test/manual/check_traffic_sign_webcam.py --show
```
