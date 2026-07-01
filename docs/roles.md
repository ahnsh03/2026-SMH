# 팀 역할 분담

> 출처: [Notion 회의록 26.06.30](https://app.notion.com/p/7581b0cdce9b831b8cb781a472c1621a)  
> 정기 회의: **매주 월요일 15시**

## 공통

- 대회 미션 숙지, 작년 영상으로 트랙 형태 파악
- 팀 GitHub 활용 (`feature/이름-기능` 브랜치 권장)

## 인지 (Perception)

| 담당 | 모듈 | ROS2 패키지 (예정) | 설명 |
|------|------|-------------------|------|
| **장원태** | 차선 인지 | `lane_detection` | 라인 트레이싱, 차선 이탈 방지 |
| **장원정** | 신호등·표지판 | `traffic_sign` | 초록/빨강 신호등, 좌/우 갈림길 표지판 |
| **안승현, 박성준** | ArUco 마커 | `aruco_detection` | 동적 장애물 마커 인식 및 정지 |

## 판단 (Planning)

| 담당 | 모듈 | 설명 |
|------|------|------|
| **양서준** | 회전 교차로 | In 코스 선택 시 교차로 진입·회전·탈출 판단 |
| 안승현, 박성준 | (합류 예정) | ArUco 작업 완료 후 회전 교차로 지원 |

## 하드웨어

| 항목 | 담당 | 설명 |
|------|------|------|
| 카메라 배치 | 미정 | 라인 트레이싱 화각 + 전방 객체 인식 균형 |
| 외관 디자인 | 미정 | 차량 외관 |

## inference_node 통합

모든 인지·판단 모듈의 결과는 `inference_node`에서 취합해 `/control` 토픽으로 출력합니다.

```
camera ──► [lane / traffic / aruco / roundabout] ──► inference_node ──► /control ──► control_node
```

각 담당자는 담당 모듈을 `src/inference/inference/` 하위 또는 별도 패키지로 개발한 뒤 `inference_node`에 연동합니다.
