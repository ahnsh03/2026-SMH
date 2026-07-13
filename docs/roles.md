# 팀 역할 분담

> 개정: 2026-07-13 (MainPlanner 통합 + 원태 공석 동안 차선 인지 합류)  
> 최근 회의: [meetings/2026-07-10.md](./meetings/2026-07-10.md)  
> **런타임 SSOT:** [main-planner.md](./main-planner.md) · [lane-perception-topic.md](./lane-perception-topic.md) ★  
> 협업 규칙: [collaboration.md](./collaboration.md)

## 공통

- 대회 미션 숙지, 작년 영상으로 트랙 형태 파악
- **Git: `main` 직접 push 금지** — `feature/이름-기능` 브랜치 → PR → merge
- **담당 파일만** 수정 ([충돌 방지](./collaboration.md) §2)
- **PC 검증**: [simulation-setup.md](./simulation-setup.md) — `sim_auto_driving` = `inference_node`(MainPlanner 포함)
- PR 전 [lane-perception-topic.md](./lane-perception-topic.md) 체크리스트 확인

## 현재 과제 (2026-07-13)

| 담당 | 모듈·영역 | 과제 |
|------|-----------|------|
| **안승현** (임시) | `modules/lane_detection.py` | 갈림길·곡선·한쪽선 L/R 인지 안정화 · `LANE_VISUALIZE` 순차 검증 (원태 공석) |
| **장원태** | `lane_detection` (복귀 후) | 알고리즘 핸드오프·공동 소유 |
| **양서준** | `pipeline.MainPlanner` · `config/main_planner.yaml` | PP·In/Out 미션 FSM · 게인 튜닝 |
| **장원정** | `modules/traffic_sign.py` | 신호등·표지판 (MainPlanner 입력) |
| **박성준** | ArUco stop | 정지 로직 · MainPlanner 최우선 인터럽트와 정합 |
| **안승현** (팀장) | 시뮬·통합·Metric IPM YAML | launch·노드·문서 SSOT |

## 인지 (Perception)

| 담당 | 모듈 | 파일 | 반환 |
|------|------|------|------|
| **안승현** (임시) / **장원태** | 차선 인지 | `modules/lane_detection.py` | 모듈 `LaneDetections` (조향 없음) |
| **장원정** | 신호등·표지판 | `modules/traffic_sign.py` | `TrafficResult` |
| **안승현** | ArUco 검출 | `modules/aruco/detector.py` | `list[int]` |
| **박성준** | ArUco 정지 | `modules/aruco/stop_logic.py` | `(bool, int \| None)` |

## 판단·제어 (Planning / Control)

| 담당 | 모듈 | 파일 | 비고 |
|------|------|------|------|
| **양서준** | MainPlanner | `pipeline.py` + `config/main_planner.yaml` | 최종 `/control` · [main-planner.md](./main-planner.md) |
| — | (레거시) | `lane_control_node.py`, `modules/lane_planner.py` | **auto launch 미사용 · MainPlanner와 동시 실행 금지** |

`modules/roundabout.py`는 제거됨. 회전교차로는 `MainPlanner` In 코스 상태로 통합.

## 통합 (팀장)

| 파일 | 역할 |
|------|------|
| `types.py` | SSOT dataclass |
| `lane_adapters.py` | module/msg → types (검증·외부용) |
| `pipeline.py` | **MainPlanner** (런타임 판제) |
| `inference_node.py` | 카메라 → MainPlanner → `/control` + 검증 토픽 |
| `config/lane_vision.yaml` | Metric IPM + HSV |
| `config/main_planner.yaml` | PP·미션 게인 (양서준 튜닝, 팀장 잠금 협의) |

## 데이터 흐름 (현재 main)

```
/camera/image/compressed
        │
        ▼
  inference_node
        │
        ├─ MainPlanner.step(frame)
        │    ├─ lane_detection.detect()
        │    ├─ traffic_sign.detect()
        │    └─ aruco_detection.detect()
        ├─ /perception/lane, /debug/*  (검증·기록)
        └─ /control
                │
     ┌──────────┴──────────┐
     ▼                     ▼
sim_control_bridge    control_node (실차)
```

상세: [main-planner.md](./main-planner.md) · 노드·시각화: [lane-perception-topic.md](./lane-perception-topic.md)

## 검증 환경

| 환경 | Launch | 문서 |
|------|---------|------|
| PC Gazebo | `sim_auto_driving.launch.py` (`inference_node`만으로 자율) | [simulation-setup.md](./simulation-setup.md) |
| D3-G 실차 | `auto_driving.launch.py` | [board-workflow.md](./board-workflow.md) |
