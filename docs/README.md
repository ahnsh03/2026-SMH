# 문서 목차 (2026-SMH)

> 팀 레포 문서 진입점. **처음 오는 팀원**은 [../README.md](../README.md) 빠른 시작부터 보세요.

## 누가 무엇을 읽나

| 대상 | 1순위 | 2순위 |
|------|--------|--------|
| **PC 시뮬 개발** | [simulation-setup.md](./simulation-setup.md) ★ | [dev-environment.md](./dev-environment.md) |
| **D3-G 보드 주행** | [board-workflow.md](./board-workflow.md) ★ | [setup.md](./setup.md) |
| **역할·과제** | [roles.md](./roles.md) | [meetings/2026-07-10.md](./meetings/2026-07-10.md) |
| **차선·기본 주행** | [lane-drive-strategy.md](./lane-drive-strategy.md) ★ | [vehicle-geometry.md](./vehicle-geometry.md) |
| **Git·PR** | [collaboration.md](./collaboration.md) ★ (`gh` §1.7) | — |

## 전체 목록

### 협업·운영

| 문서 | 내용 |
|------|------|
| [collaboration.md](./collaboration.md) | 브랜치·PR·충돌 방지 · **`gh` 설치/PR** ★ |
| [roles.md](./roles.md) | 역할 분담·데이터 흐름 |
| [lane-drive-strategy.md](./lane-drive-strategy.md) | 기본 주행 루프·**Metric IPM SSOT**·external 참고 ★ |
| [../scripts/vision_tune/README.md](../scripts/vision_tune/README.md) | Phase 0 **`tune_bev.py`(IPM)** · 캡처 · 사다리꼴 레거시 |
| [../config/lane_vision.yaml](../config/lane_vision.yaml) | `metric_ipm:` 잠정 파라미터 (`y_half=0.77`) |
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
