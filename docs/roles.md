# 팀 역할 분담

> 최근 회의: [meetings/2026-07-10.md](./meetings/2026-07-10.md) (2026-07-10 역할 재분배)  
> 정기 회의: **매주 월요일 15시**  
> 협업 규칙: [collaboration.md](./collaboration.md)

## 공통

- 대회 미션 숙지, 작년 영상으로 트랙 형태 파악
- **Git: `main` 직접 push 금지** — `feature/이름-기능` 브랜치 → PR → merge ([협업 가이드](./collaboration.md) §1)
- **담당 `modules/` 파일만** 수정 ([충돌 방지](./collaboration.md) §2)
- **PC 모듈 검증**: [simulation-setup.md](./simulation-setup.md) (`./scripts/dev_container.sh sim`) 또는 교내 트랙 — 교내 트랙은 대회 트랙과 다름 ([회의록](./meetings/2026-07-10.md))

## 현재 과제 (2026-07-10 기준)

| 담당 | 모듈·영역 | 과제 |
|------|-----------|------|
| **장원태** | `modules/lane_detection.py` | BEV 차선 인식 실증 (교내 트랙 또는 Gazebo) |
| **장원정** | `modules/traffic_sign.py` | 신호등 불 인식 코드 · 표지판 YOLO 인식 |
| **안승현** | `src/dracer_sim/` (시뮬 월드) | 갈림길 좌/우 표지판 · 장애물 ArUco 배치 → 신호등·표지판 주행 **합류** |
| **박성준** | `modules/roundabout.py` (합류) | 회전 교차로 코드 개발 |
| **양서준** | `modules/roundabout.py` | 회전 교차로 (Pure Pursuit + 차선 선택) 설계·개발·실증 |

### 완료

| 담당 | 내용 |
|------|------|
| 안승현, 박성준 | ArUco 검출·정지 (`detector.py`, `stop_logic.py`) |
| 안승현 | Gazebo 시뮬 기본 스택 (`dracer_sim`, D-Racer 토픽·트랙) |

## 인지 (Perception) — 모듈 매핑

| 담당 | 모듈 | 파일 | 반환 타입 |
|------|------|------|-----------|
| **장원태** | 차선 인지 | `modules/lane_detection.py` | `LaneResult` |
| **장원정** | 신호등·표지판 | `modules/traffic_sign.py` | `TrafficResult` |
| **안승현** | ArUco 검출 | `modules/aruco/detector.py` | `list[int]` |
| **박성준** | ArUco 정지 판단 | `modules/aruco/stop_logic.py` | `(bool, int \| None)` |

## 판단 (Planning)

| 담당 | 모듈 | 파일 | 반환 타입 |
|------|------|------|-----------|
| **양서준** | 회전 교차로 | `modules/roundabout.py` | `RoundaboutResult` |
| **박성준** | (합류) | `modules/roundabout.py` | — |

## 통합 (팀장)

| 파일 | 역할 |
|------|------|
| `types.py` | 모듈 간 공통 데이터 타입 |
| `pipeline.py` | 모듈 호출 + 우선순위 fusion · **모드 전환** 설계 |
| `inference_node.py` | ROS2 노드 (camera 구독 → /control 발행) |
| `modules/aruco_detection.py` | ArUco facade (detector + stop_logic) |

### 아키텍처 방향 (회의 합의)

- 미션별 **모드 변경** + launch/파라미터 기반 전환 검토
- 신호등 **빨간불** vs 장애물 미션 **빨간 차로** 혼동 방지 (색·위치·컨텍스트 분리)

## 데이터 흐름

```
/camera/image/compressed
        │
        ▼
  inference_node
        │
        ▼
  pipeline.run_perception()
    ├── lane_detection.detect()      → LaneResult
    ├── traffic_sign.detect()        → TrafficResult
    ├── aruco_detection.detect()     → ArucoResult
    └── roundabout.plan()            → RoundaboutResult
        │
        ▼
  pipeline.fuse_control()  → steering, throttle
        │
        ▼
      /control  →  control_node
```

## 검증 환경

| 환경 | 문서 | 비고 |
|------|------|------|
| PC Gazebo | [simulation-setup.md](./simulation-setup.md) | CW 트랙, 320×180, D-Racer 토픽 |
| D3-G 실차 | [board-workflow.md](./board-workflow.md) | 최종 주행 확인 |
| 교내 트랙 | — | 흰 차선·검은 바닥, 대회와 경로·폭 다름 |

## 하드웨어

| 항목 | 담당 | 설명 |
|------|------|------|
| 로봇 조립 | 장원태 | 진행 중 |
| 카메라 배치 | 미정 | 라인 트레이싱 화각 + 전방 객체 인식 균형 |
| 외관 디자인 | 미정 | 차량 외관 |
