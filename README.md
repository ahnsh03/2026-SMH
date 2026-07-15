# 2026-SMH

**2026 SEA:ME 해커톤** — AIM 학술동아리 자율주행 팀 저장소

> **이 체크아웃이 `board/race-control` 이면:** 실차·대회용 골격입니다.  
> → 먼저 [**BOARD.md**](BOARD.md) · `./scripts/board_race_sync.sh` 를 사용하세요.  
> PC feature 작업은 별도 폴더(`2026-SMH` worktree)에서 합니다.

> Repo 이름은 대회 약자(SMH)를 사용합니다. 로컬 경로·셸 변수: `$SMH`

> **PC(WSL) 상위 프로젝트**: [../README.md](../README.md)  
> **D3-G 보드** — `board/race-control` 권장 경로 `~/2026-SMH-board`

| | |
|---|---|
| **대회** | 2026.7.14 ~ 7.16 / 호텔 파크하비오 |
| **주제** | AI 네이티브 스케일카 자율주행 챌린지 |
| **정기 회의** | 매주 월요일 15시 |
| **Notion** | [팀 대시보드](https://app.notion.com/p/55e1b0cdce9b8292a19d81c5b1605983) |

---

## 빠른 시작 (D3-G 보드)

상세: [docs/board-workflow.md](docs/board-workflow.md)

**`~/D-Racer-Kit`이 이미 있는 경우 (Case A)**

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
mkdir -p external
rm -rf external/D-Racer-Kit
ln -sfn ~/D-Racer-Kit external/D-Racer-Kit
./scripts/board_sync.sh --no-pull
```

**처음 시작하는 경우 (Case B)**

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull
```

이후 코드 받을 때:

```bash
./scripts/board_sync.sh             # pull + init + build
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

자세한 셋업: [docs/setup.md](docs/setup.md)  
**보드 개발·주행 (팀원 필독)**: [docs/board-workflow.md](docs/board-workflow.md) ★  
**PC Docker 환경 (팀 표준)**: [docs/dev-environment.md](docs/dev-environment.md)  
**PC Gazebo 시뮬**: [docs/simulation-setup.md](docs/simulation-setup.md) ★  
협업 규칙: [docs/collaboration.md](docs/collaboration.md)

---

## 빠른 시작 (PC 시뮬 — 레포만 clone)

> **팀원 필독**: [docs/simulation-setup.md](docs/simulation-setup.md) §4 — **터미널 1·2 개발 방법**

상위 monorepo·`external/limo_ros2` 없이 **이 레포만** clone하면 됩니다.  
`vendor/limo_car`(mesh 포함, ~100MB)는 레포에 포함되어 있으며, D-Racer-Kit은 `init` 시 자동 clone됩니다.

### 최초 1회 (이미지·Gazebo)

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh

./scripts/dev_container.sh build
./scripts/dev_container.sh install-gazebo   # Gazebo 최초 1회 (~5–10분)
./scripts/dev_container.sh init
./scripts/dev_container.sh check-gpu        # 선택: GPU 렌더링 확인
```

### 매일 개발 (컨테이너 1개 + 터미널 2개)

> **빌드 구분**: `build` = Docker **이미지** (`Dockerfile`) · `build-sim` / `colcon` = **ROS 코드** (`src/`). 시뮬 매일 켤 때는 후자만.

| 터미널 | 명령 | 역할 |
|--------|------|------|
| — | `./scripts/dev_container.sh sim-up` | `2026-smh-sim` 생성 (없을 때만) |
| **1** | `./scripts/dev_container.sh sim-bringup` | Gazebo + 브리지 + 카메라 프리뷰 |
| **2** | `docker exec -it 2026-smh-sim bash` | inference 빌드·실행 |
| — | `./scripts/dev_container.sh sim-down` | 하루 작업 끝 |

```bash
# 터미널 1
./scripts/dev_container.sh sim-bringup

# 터미널 2
docker exec -it 2026-smh-sim bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run inference inference_node --ros-args -p use_sim_time:=true

# 검증 (호스트, bringup 실행 중)
./scripts/dev_container.sh verify-sim
```

| 명령 | 설명 |
|------|------|
| `sim-up` | 시뮬 컨테이너 `2026-smh-sim` 생성·시작 |
| `sim-bringup` | 터미널1: build-sim + Gazebo launch (Ctrl+C → launch만 종료) |
| `sim-down` | 시뮬 컨테이너 삭제 |
| `build-sim` | **ROS 워크스페이스** colcon (`src/` 코드 변경 후) |
| `build` | Docker **이미지** (`Dockerfile`, 최초·드묾) |
| `sim` | bringup + inference 자율주행 (한 터미널 통합 테스트) |
| `verify-sim` | 토픽·카메라 검증 (bringup 실행 중) |

시뮬 기본: 카메라 **320×180** (16:9). 터미널2는 **`docker exec`만** 쓰면 됩니다.  
**직접 docker/ros2 명령**: [simulation-setup.md §4.8](docs/simulation-setup.md#48-직접-명령어-치트시트-스크립트-없이)

---

## 저장소 구조

```
2026-SMH/
├── docs/
│   ├── README.md          # 문서 목차
│   ├── collaboration.md   # ★ 브랜치·PR·충돌 방지 (팀원 필독)
│   ├── roles.md           # 역할 분담
│   ├── meetings/          # 회의록
│   ├── simulation-setup.md # ★ PC 시뮬 재현 가이드 (팀원 필독)
│   ├── simulation.md       # 트러블슈팅·GPU
│   └── competition.md     # 대회 정보
├── Dockerfile             # PC 개발용 (22.04 + Humble)
├── docker-compose.yml
├── config/
│   └── vehicle_config.yaml # 팀 카메라 320×180 (init → src/config 링크)
├── scripts/
│   ├── init_workspace.sh  # D-Racer-Kit clone + src/ 링크
│   ├── dev_container.sh   # ★ PC: Docker 빌드·시뮬·검증
│   ├── verify_sim.sh      # 시뮬 토픽 검증
│   ├── check_sim_gpu.sh   # GPU 렌더링 확인
│   └── board_sync.sh      # ★ 보드: pull + init + build
├── external/              # D-Racer-Kit (Git 제외, init 시 자동 clone)
├── vendor/
│   └── limo_car/          # LIMO Gazebo 모델 (레포 포함, 시뮬용)
└── src/
    ├── inference/         # ★ 팀 자율주행 패키지 (Git 추적)
    │   ├── inference/
    │   │   ├── types.py
    │   │   ├── pipeline.py
    │   │   ├── inference_node.py
    │   │   └── modules/
    │   └── launch/
    └── dracer_sim/        # ★ Gazebo 시뮬 (D-Racer 토픽 호환)
        ├── launch/
        ├── urdf/
        └── models/
```

주최측 패키지(camera, control 등)는 `init_workspace.sh`가 `src/`에 심볼릭 링크합니다.

---

## 역할 분담

| 담당 | 모듈 | 파일 |
|------|------|------|
| **장원태** | 차선 인지 | `modules/lane_detection.py` |
| **장원정** | 신호등·표지판 | `modules/traffic_sign.py` |
| **안승현** | ArUco 검출 | `modules/aruco/detector.py` |
| **박성준** | ArUco 정지 | `modules/aruco/stop_logic.py` |
| **양서준** | 통합 판단·Pure Pursuit·회전교차로 | `pipeline.py` |

상세: [docs/roles.md](docs/roles.md)

---

## 데이터 흐름

```
/camera/image/compressed
        │
        ▼
  inference_node → pipeline.MainPlanner.step()
        │            (lane / traffic / aruco 직접 결과 전달)
        ▼
  Pure Pursuit + mission state  →  /control  →  control_node
        └──────── 검증용 /perception/lane, /debug/*
```

코스 선택: `ros2 launch inference auto_driving.launch.py route_mode:=in` (기본 `out`).
주행·미션 파라미터는 `config/main_planner.yaml`에서 한 번에 조정합니다.

---

## 실행 (Docker 권장)

PC에서 시뮬로 inference를 검증할 때:

```bash
./scripts/dev_container.sh sim          # Gazebo + inference
```

실기(D3-G)에서:

```bash
source install/setup.bash

# 수동 주행 (camera + monitor)
ros2 launch inference manual_driving.launch.py

# 자율주행 (camera + monitor + inference)
ros2 launch inference auto_driving.launch.py
```

웹 모니터: `http://<WEB_HOST>:5000` (`src/config/vehicle_config.yaml`)  
ArUco 보드 확인: `ros2 topic echo /debug/aruco` — 상세는 [docs/board-workflow.md](docs/board-workflow.md) §3.3

> `~/D-Racer-Kit`에서 camera/monitor를 따로 실행하지 마세요. 장치 충돌로 영상이 멈춥니다.

---

## 브랜치 규칙 (필수)

> 상세: [docs/collaboration.md](docs/collaboration.md) §1

1. **`main` 직접 push 금지** — PR merge로만 반영
2. **`feature/이름-기능` 브랜치**에서만 개발 → commit → push → **PR**
3. **한 PR = 담당 `modules/` 한 모듈** (작은 PR)
4. merge 후 보드에서 `./scripts/board_sync.sh`로 `main` 테스트

```
main → feature/wontae-lane → PR → merge → board_sync.sh
```

---

## 문서

- [문서 목차](docs/README.md)
- [협업 가이드](docs/collaboration.md) ★
- [역할 분담](docs/roles.md) · [회의록 2026-07-10](docs/meetings/2026-07-10.md)
- [보드 개발·주행 가이드](docs/board-workflow.md) ★
- [셋업 가이드](docs/setup.md)
- [개발 환경 규약 · Docker](docs/dev-environment.md) ★
- [시뮬레이터 재현 가이드](docs/simulation-setup.md) ★
- [시뮬레이터 트러블슈팅](docs/simulation.md)
- [대회 정보](docs/competition.md)
- [플랫폼·보드 스펙 (D3-G)](docs/hardware-board.md)
- [카메라 스펙 (C920e)](docs/hardware-camera.md)
- [참고 링크](docs/references.md)

---

## 주최측 제공 자료

- 공식 ROS2: https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0
- 참고 영상: https://drive.google.com/file/d/1QpnQdkiiYtEs1k2Ll4sRCjBB_1pBNbmG/view
