# 2026-SMH

**2026 SEA:ME 해커톤** — AIM 학술동아리 자율주행 팀 저장소

> Repo 이름은 대회 약자(SMH)를 사용합니다. 로컬 경로·셸 변수: `$SMH`

> **PC(WSL) 상위 프로젝트**: [../README.md](../README.md)  
> **D3-G 보드 단독 clone** (`~/2026-SMH`) — 아래 빠른 시작만으로 충분합니다.

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

> **팀원 필독**: [docs/simulation-setup.md](docs/simulation-setup.md) — 단계별 스크린샷급 가이드

상위 monorepo·`external/limo_ros2` 없이 **이 레포만** clone하면 됩니다.  
`vendor/limo_car`(mesh 포함, ~100MB)는 레포에 포함되어 있으며, D-Racer-Kit은 `init` 시 자동 clone됩니다.

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh

./scripts/dev_container.sh build
./scripts/dev_container.sh install-gazebo   # Gazebo 최초 1회 (~5–10분)
./scripts/dev_container.sh init
./scripts/dev_container.sh build-sim
./scripts/dev_container.sh check-gpu        # 선택: GPU 렌더링 확인
./scripts/dev_container.sh sim-bringup      # Gazebo + RViz
# inference 테스트: ./scripts/dev_container.sh sim
# 검증 (다른 터미널): ./scripts/dev_container.sh verify-sim
```

| 명령 | 설명 |
|------|------|
| `sim-bringup` | Gazebo + 트랙 + LIMO + 카메라 브리지 + **RViz** |
| `sim` | bringup + **inference** 자율주행 |
| `sim-manual` | bringup + 조이스틱 수동주행 |
| `verify-sim` | 토픽·카메라 동작 검증 (sim 실행 중) |

시뮬 기본: 카메라 **320×180** (16:9), 웹 모니터 **OFF** (RViz로 확인).

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
| **양서준** | 회전 교차로 | `modules/roundabout.py` |

상세: [docs/roles.md](docs/roles.md)

---

## 데이터 흐름

```
/camera/image/compressed
        │
        ▼
  inference_node → pipeline.run_perception()
        │            (lane / traffic / aruco / roundabout)
        ▼
  pipeline.fuse_control()  →  /control  →  control_node
```

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
