# 문서 목차 (2026-SMH)

> 팀 레포 문서 진입점. **처음 오는 팀원**은 [../README.md](../README.md) 빠른 시작부터 보세요.

## 누가 무엇을 읽나

| 대상 | 1순위 | 2순위 |
|------|--------|--------|
| **PC 시뮬 개발** | [simulation-setup.md](./simulation-setup.md) ★ | [dev-environment.md](./dev-environment.md) |
| **D3-G 보드 주행** | [board-workflow.md](./board-workflow.md) ★ | [setup.md](./setup.md) |
| **역할·과제** | [roles.md](./roles.md) | [meetings/2026-07-10.md](./meetings/2026-07-10.md) |
| **차선·기본 주행** | [main-planner.md](./main-planner.md) ★ | [control-hybrid-strategy.md](./control-hybrid-strategy.md) · [lane-perception-topic.md](./lane-perception-topic.md) |
| **보드 실차 제어** | [board-freeze-control.md](./board-freeze-control.md) ★ | [control-hybrid-strategy.md](./control-hybrid-strategy.md) T0–T7 |
| **Git·PR** | [collaboration.md](./collaboration.md) ★ (`gh` §1.7) | — |

## 전체 목록

### 협업·운영

| 문서 | 내용 |
|------|------|
| [collaboration.md](./collaboration.md) | 브랜치·PR·충돌 방지 · **`gh` 설치/PR** ★ |
| [lane-perception-topic.md](./lane-perception-topic.md) | 노드 인벤토리 · **인지 시각화 검증** · 레거시 주의 ★ |
| [main-planner.md](./main-planner.md) | MainPlanner·PP·In/Out·디버깅 ★ |
| [control-hybrid-strategy.md](./control-hybrid-strategy.md) | **mask↔paint 하이브리드 제어 · 실차 T0–T7** ★ |
| [board-freeze-control.md](./board-freeze-control.md) | 보드 동결 YAML · 실차 적용 체크리스트 ★ |
| [roles.md](./roles.md) | 역할 분담 (인지 임시 합류 포함) |
| [lane-drive-strategy.md](./lane-drive-strategy.md) | Metric IPM·인지 설계 배경 (제어 SSOT는 main-planner · hybrid) |
| [out-route-reference.md](./out-route-reference.md) | OUT 맵 **직선·코너·S자** 로봇 보조지표 / CTE 합격선 |
| [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) | **소실·Out 갈림·In 탈출** 인지 전략 + **용어·판단 SSOT** ★ |
| [fork-moment-detection.md](./fork-moment-detection.md) | **IN/OUT 직전 시점 게이트** + 차로쌍·**데이터 라벨** ★ |
| [out-ego-fork-shape.md](./out-ego-fork-shape.md) | OUT ego Y-stretch + tip fuse · bag 검증 ★ |
| [hsv-profiles.md](./hsv-profiles.md) | **시뮬·실차 HSV + 주행가능(road\|시안→ego blob) SSOT** ★ |
| [../scripts/vision_tune/README.md](../scripts/vision_tune/README.md) | **`tune_bev`** · **`tune_hsv`** · **`tune_lane_control`** · 캡처 |
| [../config/lane_vision.yaml](../config/lane_vision.yaml) | `metric_ipm:` + `hsv.profiles` (sim / real_car) |
| [../config/lane_control.yaml](../config/lane_control.yaml) | planner P/EMA/rate/look-ahead (시뮬·실차 공용) |
| [vehicle-geometry.md](./vehicle-geometry.md) | D-Racer ↔ LIMO 휠베이스·트레드·조향 스펙 (제어 튜닝) ★ |
| [meetings/2026-07-10.md](./meetings/2026-07-10.md) | 최근 회의록 (역할 재분배) |
| [ANNOUNCEMENT.md](./ANNOUNCEMENT.md) | 대회 공지 요약 |

### 환경·실행

| 문서 | 내용 |
|------|------|
| [setup.md](./setup.md) | 보드·PC 셋업 (Case A/B) |
| [board-workflow.md](./board-workflow.md) | D3-G 개발·주행 ★ |
| [dev-environment.md](./dev-environment.md) | Docker·CI ★ |
| [simulation-setup.md](./simulation-setup.md) | Gazebo 시뮬 재현 · 터미널 2개 · **§4.8 직접 명령** ★ |
| [simulation.md](./simulation.md) | 시뮬 GPU·트러블슈팅 |

### 하드웨어·대회

| 문서 | 내용 |
|------|------|
| [hardware-board.md](./hardware-board.md) | D3-G / D-Racer 플랫폼 |
| [hardware-camera.md](./hardware-camera.md) | C920e · 320×180 |
| [vehicle-geometry.md](./vehicle-geometry.md) | 실차↔시뮬 조향·기하 차이 (튜닝) |
| [competition.md](./competition.md) | 대회 정보 통합 |
| [references.md](./references.md) | 외부 링크 |

### 패키지

| 문서 | 내용 |
|------|------|
| [../src/dracer_sim/README.md](../src/dracer_sim/README.md) | Gazebo `dracer_sim` (트랙·표지판·카메라) |
| [../src/inference/](../src/inference/) | 팀 `inference` 소스 |

## 레포 구조 (요약)

```
2026-SMH/
├── config/vehicle_config.yaml   # 팀 카메라 320×180
├── docs/                        # ← 이 폴더
├── scripts/dev_container.sh     # PC Docker·시뮬
├── scripts/board_sync.sh        # 보드 동기화
├── src/inference/               # 팀 자율주행 (Git)
├── src/dracer_sim/              # Gazebo 시뮬 + 미션 표지판 3종 (Git)
├── vendor/limo_car/             # LIMO 모델 (Git)
└── external/D-Racer-Kit/        # init 시 clone (Git 제외)
```
