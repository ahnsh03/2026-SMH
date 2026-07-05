# 팀 역할 분담

> 출처: [Notion 회의록 26.06.30](https://app.notion.com/p/7581b0cdce9b831b8cb781a472c1621a)  
> 정기 회의: **매주 월요일 15시**  
> 협업 규칙: [collaboration.md](./collaboration.md)

## 공통

- 대회 미션 숙지, 작년 영상으로 트랙 형태 파악
- **Git: `main` 직접 push 금지** — `feature/이름-기능` 브랜치 → PR → merge ([협업 가이드](./collaboration.md) §1)
- **담당 `modules/` 파일만** 수정 ([충돌 방지](./collaboration.md) §2)

## 인지 (Perception)

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
| 안승현, 박성준 | (합류 예정) | ArUco 완료 후 `roundabout.py` 지원 | — |

## 통합 (팀장)

| 파일 | 역할 |
|------|------|
| `types.py` | 모듈 간 공통 데이터 타입 |
| `pipeline.py` | 모듈 호출 + 우선순위 fusion |
| `inference_node.py` | ROS2 노드 (camera 구독 → /control 발행) |
| `modules/aruco_detection.py` | ArUco facade (detector + stop_logic) |

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

## 하드웨어

| 항목 | 담당 | 설명 |
|------|------|------|
| 카메라 배치 | 미정 | 라인 트레이싱 화각 + 전방 객체 인식 균형 |
| 외관 디자인 | 미정 | 차량 외관 |
