# 팀 역할 분담

> 최근 회의: [meetings/2026-07-10.md](./meetings/2026-07-10.md)  
> **런타임 구조 SSOT:** [lane-perception-topic.md](./lane-perception-topic.md) ★  
> 협업 규칙: [collaboration.md](./collaboration.md)

## 공통

- 대회 미션 숙지, 작년 영상으로 트랙 형태 파악
- **Git: `main` 직접 push 금지** — `feature/이름-기능` 브랜치 → PR → merge
- **담당 `modules/` 파일만** 수정 ([충돌 방지](./collaboration.md) §2)
- **PC 모듈 검증**: [simulation-setup.md](./simulation-setup.md) — 자율주행은 **인지+제어 둘 다**
- PR 전 [lane-perception-topic.md](./lane-perception-topic.md) §8 체크리스트 확인

## 현재 과제

| 담당 | 모듈·영역 | 과제 |
|------|-----------|------|
| **장원태** | `modules/lane_detection.py` | Metric IPM BEV 위 차선 인지 실증 (시뮬·교내) |
| **장원정** | `modules/traffic_sign.py` | 신호등·표지판 — 모듈 API 유지, ROS 합류는 통합 시 |
| **안승현** | 통합·시뮬·`lane_control_node` | 토픽 분리·Metric IPM SSOT·게인 튜닝 |
| **박성준** | `modules/roundabout.py` (합류) · ArUco stop | 회전교차로·정지 로직 |
| **양서준** | 경로 추종 / roundabout | `/perception/lane` 기반 PP 등 — **임시 P/EMA와 교체 합류** |

## 인지 (Perception)

| 담당 | 모듈 | 파일 | 반환 |
|------|------|------|------|
| **장원태** | 차선 인지 | `modules/lane_detection.py` | 모듈 `LaneDetections` → `/perception/lane` |
| **장원정** | 신호등·표지판 | `modules/traffic_sign.py` | `TrafficResult` |
| **안승현** | ArUco 검출 | `modules/aruco/detector.py` | `list[int]` |
| **박성준** | ArUco 정지 | `modules/aruco/stop_logic.py` | `(bool, int \| None)` |

## 판단·제어 (Planning / Control)

| 담당 | 모듈 | 파일 | 비고 |
|------|------|------|------|
| **안승현** (임시) | 흰차선 Pure Pursuit | `modules/lane_planner.py` + `lane_control_node.py` | `/perception/lane` → `/control` |
| **양서준** | PP / 미션 planner | (브랜치) | types/adapters 사용, control 노드 교체 시 합의 |
| **양서준·박성준** | 회전 교차로 | `modules/roundabout.py` | |

## 통합 (팀장)

| 파일 | 역할 |
|------|------|
| `types.py` | SSOT dataclass (`LaneDetections` 등) |
| `lane_adapters.py` | module/msg → types |
| `pipeline.py` | 단프로세스·테스트 fusion (런타임 기본 경로 아님) |
| `inference_node.py` | 인지 ROS 노드 |
| `lane_control_node.py` | 임시 제어 ROS 노드 |
| `config/lane_*.yaml` | vision/control SSOT |

## 데이터 흐름 (현재 main)

```
/camera/image/compressed
        │
        ▼
  inference_node          ← 인지만 (lane + aruco)
        │
        ├─ /perception/lane  (lane_msgs)
        └─ /debug/aruco
                │
                ▼
        lane_control_node   ← detections_from_msg + lane_planner
                │
                ▼
            /control
                │
     ┌──────────┴──────────┐
     ▼                     ▼
sim_control_bridge    control_node (실차)
```

단프로세스 테스트: `pipeline.run_perception` → adapter → planner → `fuse_control`  
상세·PR 체크리스트: [lane-perception-topic.md](./lane-perception-topic.md)

## 검증 환경

| 환경 | Launch | 문서 |
|------|---------|------|
| PC Gazebo | `sim_auto_driving.launch.py` (인지+제어) | [simulation-setup.md](./simulation-setup.md) |
| D3-G 실차 | `auto_driving.launch.py` | [board-workflow.md](./board-workflow.md) |
