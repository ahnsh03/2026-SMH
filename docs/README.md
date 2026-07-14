# 문서 목차 (2026-SMH)

> 팀 레포 문서 진입점. **처음 오는 팀원**은 [../README.md](../README.md) 빠른 시작부터 보세요.

## 누가 무엇을 읽나

| 대상 | 1순위 | 2순위 |
|------|--------|--------|
| **PC 시뮬 개발** | [simulation-setup.md](./simulation-setup.md) ★ | [dev-environment.md](./dev-environment.md) |
| **D3-G 보드 주행** | [board-workflow.md](./board-workflow.md) ★ | [setup.md](./setup.md) · [board-latency-and-sim2real.md](./board-latency-and-sim2real.md) |
| **역할·과제** | [roles.md](./roles.md) | [meetings/2026-07-10.md](./meetings/2026-07-10.md) |
| **차선·기본 주행** | [main-planner.md](./main-planner.md) ★ | [lane-perception-topic.md](./lane-perception-topic.md) |
| **갈림·합류 테스트** | [fork-test-pipeline.md](./fork-test-pipeline.md) ★ | [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) |
| **Git·PR** | [collaboration.md](./collaboration.md) ★ (`gh` §1.7) | — |

## 전체 목록

### 협업·운영

| 문서 | 내용 |
|------|------|
| [collaboration.md](./collaboration.md) | 브랜치·PR·충돌 방지 · **`gh` 설치/PR** ★ |
| [lane-perception-topic.md](./lane-perception-topic.md) | 노드 인벤토리 · **인지 시각화 검증** · 레거시 주의 ★ |
| [main-planner.md](./main-planner.md) | MainPlanner·mask_p/PP·In/Out·디버깅 ★ |
| [roles.md](./roles.md) | 역할 분담 (인지 임시 합류 포함) |
| [lane-drive-strategy.md](./lane-drive-strategy.md) | Metric IPM·인지 설계 배경 (제어 SSOT는 main-planner) |
| [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) | **소실·Out 갈림·In 탈출** 인지 전략 + **용어 SSOT** ★ |
| [fork-test-pipeline.md](./fork-test-pipeline.md) | 구간별 spawn·viz·fork_on 합격 기준 ★ |
| [../scripts/vision_tune/README.md](../scripts/vision_tune/README.md) | **`tune_bev`** · **`tune_hsv`** · **`tune_lane_control`**(레거시) · 캡처 |
| [../scripts/drive_test/README.md](../scripts/drive_test/README.md) | mask/랩/코스 벤치 · fork spawn (sim-auto OFF) |
| [../config/lane_vision.yaml](../config/lane_vision.yaml) | `metric_ipm:` 잠정 파라미터 (`y_half=0.77`) |
| [../config/main_planner.yaml](../config/main_planner.yaml) | tracker·speed·route·profiles (시뮬·실차 SSOT) |
| [../config/lane_control.yaml](../config/lane_control.yaml) | 레거시 PP 게인 (`tune_lane_control` 전용 · auto 미사용) |
| [vehicle-geometry.md](./vehicle-geometry.md) | D-Racer ↔ LIMO 휠베이스·트레드·조향 스펙 ★ |
| [meetings/2026-07-10.md](./meetings/2026-07-10.md) | 최근 회의록 (역할 재분배) |
| [ANNOUNCEMENT.md](./ANNOUNCEMENT.md) | 대회 공지 요약 |

### 환경·실행

| 문서 | 내용 |
|------|------|
| [setup.md](./setup.md) | 보드·PC 셋업 (Case A/B) |
| [board-workflow.md](./board-workflow.md) | D3-G 개발·주행 ★ |
| [board-latency-and-sim2real.md](./board-latency-and-sim2real.md) | 보드 실측 지연 · 연산 병목 · sim2real ★ |
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
├── config/
│   ├── vehicle_config.yaml    # 팀 카메라 320×180
│   ├── main_planner.yaml      # MainPlanner SSOT
│   └── lane_vision.yaml       # Metric IPM + HSV
├── docs/                        # ← 이 폴더
├── scripts/dev_container.sh     # PC Docker·시뮬
├── scripts/board_sync.sh        # 보드 동기화
├── src/inference/               # 팀 자율주행 (Git)
├── src/dracer_sim/              # Gazebo 시뮬 + 미션 표지판 3종 (Git)
├── vendor/limo_car/             # LIMO 모델 (Git)
└── external/D-Racer-Kit/        # init 시 clone (Git 제외)
```
