# 팀 개발 환경 규약

> **목표**: 팀원 PC(WSL) 버전이 달라도 **동일한 Ubuntu 22.04 + ROS2 Humble** 환경에서 빌드·검증  
> **기준 환경**: D3-G 보드 (Ubuntu 22.04, ROS2 Humble) — 실제 주행 테스트는 보드에서만 수행

---

## 1. 환경 역할 분담

| 환경 | OS / ROS | 용도 | 필수 여부 |
|------|----------|------|-----------|
| **D3-G 보드** | Ubuntu 22.04 + Humble | `colcon build`, launch, 조이스틱·카메라·모터 테스트 | **필수** (주행 검증) |
| **PC Docker** | Ubuntu 22.04 + Humble (컨테이너) | 빌드, import 검증, **Gazebo 시뮬** | **권장** (WSL 버전 무관) |
| **PC WSL 네이티브** | Ubuntu **22.04** + Humble 직접 설치 | Docker 없이 동일 작업 | 22.04만 해당 |
| **PC WSL (24.04/26.04 등)** | 호스트 ROS **설치 불가** | Git 편집 + **Docker 필수** | **Docker 사용** |

### Docker로 할 수 있는 것 / 없는 것

| 가능 | 불가능 (D3-G 전용) |
|------|-------------------|
| `init_workspace.sh` | USB 조이스틱 (WSL USB 패스스루 별도) |
| `colcon build` (inference, dracer_sim) | 실기 USB 카메라 스트리밍 |
| `inference.pipeline` import 검증 | PCA9685 모터/조향 I2C 제어 |
| **Gazebo 시뮬 + 카메라 토픽 검증** (`./scripts/dev_container.sh sim`) | 실차 주행 최종 튜닝 |

---

## 2. 팀 표준 규약

### 2.1 기준 버전 (Single Source of Truth)

| 항목 | 팀 표준 | 비고 |
|------|---------|------|
| OS | **Ubuntu 22.04 (Jammy)** | D3-G 공식 이미지와 동일 |
| ROS2 | **Humble** | Jazzy(24.04) 사용 금지 |
| D-Racer-Kit | `release/v1.0.0` | `init_workspace.sh`가 자동 clone |
| Python | 3.10 (22.04 기본) | inference 모듈 |
| 빌드 도구 | colcon + `--symlink-install` | 보드·CI·Docker 동일 |
| CI 검증 패키지 | `inference` (+ 의존 `control_msgs`) | `.github/workflows/ci.yml` |

### 2.2 개발 워크플로 (권장)

```
1. feature 브랜치 생성
2. modules/ 코드 수정
3. PC Docker에서 build-inference / check  ← PR 전 로컬 검증
4. commit → push → PR
5. GitHub CI 통과 확인
6. merge 후 D3-G에서 board_sync.sh → launch 주행 테스트
```

### 2.3 ROS 도메인

| 변수 | 팀 기본값 | 설명 |
|------|-----------|------|
| `ROS_DOMAIN_ID` | `0` | 같은 WiFi에서 여러 팀 로봇이 있으면 팀별로 분리 (예: 42) |

PC Docker와 D3-G가 같은 네트워크에서 ROS 통신을 시험할 때만 `ROS_DOMAIN_ID`를 맞춥니다. 일반적인 PC 빌드 검증에는 필요 없습니다.

### 2.4 파일·빌드 산출물

| 경로 | Git 추적 | 설명 |
|------|----------|------|
| `src/inference/` | O | 팀 자율주행 코드 |
| `src/dracer_sim/` | O | Gazebo 시뮬 패키지 |
| `vendor/limo_car/` | O | LIMO URDF·mesh (시뮬용) |
| `config/vehicle_config.yaml` | O | 팀 카메라 320×180 |
| `src/camera`, `src/control` 등 | X | `init_workspace.sh` 심볼릭 링크 |
| `external/D-Racer-Kit/` | X | clone 시 자동 생성 |
| `build/`, `install/`, `log/` | X | colcon 산출물 |

---

## 3. 사전 준비 (PC / WSL)

### 3.1 Windows + WSL2

1. **Docker Desktop** 설치 (WSL2 백엔드 사용)
2. Docker Desktop → Settings → Resources → WSL Integration → 사용 중인 배포판 활성화
3. WSL 터미널에서 확인:

```bash
docker --version
docker compose version
```

> WSL 배포판이 22.04·24.04·26.04 중 무엇이든 **상관없습니다**. 컨테이너 안은 항상 22.04 + Humble입니다.

### 3.2 저장소 clone

```bash
# 상위 프로젝트 구조 (PC)
cd ~/projects/2026-seame-hackathon
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
```

---

## 4. Docker 컨테이너 사용법

### 4.1 이미지 빌드 (최초 1회 + Dockerfile 변경 시)

```bash
cd ~/projects/2026-seame-hackathon/2026-SMH

# 방법 A — 헬퍼 스크립트 (권장)
./scripts/dev_container.sh build

# 방법 B — docker compose 직접
docker compose build
```

빌드 완료 후 이미지 이름: `2026-smh-dev:latest`

예상 소요: 최초 3~8분 (네트워크 속도에 따라 다름)

### 4.2 워크스페이스 초기화 (D-Racer-Kit 링크)

```bash
./scripts/dev_container.sh init
```

`external/D-Racer-Kit` clone + `src/` 심볼릭 링크가 생성됩니다. **인터넷 연결 필요.**

### 4.3 inference 빌드·검증

`2026-smh-sim`이 **실행 중**이면 `init` / `build-sim` / `build-inference` / `check`도 **같은 컨테이너**에서 실행됩니다.  
시뮬 없이 inference만 검증할 때는 sim 컨테이너가 없으면 일회성 `dev` 컨테이너로 폴백합니다.

```bash
./scripts/dev_container.sh sim-up          # 권장: 시뮬 개발 시 먼저 생성
./scripts/dev_container.sh build-inference
./scripts/dev_container.sh check
```

컨테이너 셸이 필요하면:

```bash
docker exec -it 2026-smh-sim bash
```

### 4.4 CI와 동일한 빌드·검증 (PR 전 권장)

```bash
# inference + 의존 패키지 빌드
./scripts/dev_container.sh build-inference

# 빌드 + import 검증 (GitHub Actions와 동일)
./scripts/dev_container.sh check
```

`check`가 `ok`를 출력하면 CI `build-inference` job과 같은 검증을 통과한 것입니다.

### 4.6 Gazebo 시뮬 (WSL 24.04 / 26.04 포함 — Docker 필수)

> **팀원 재현 가이드**: [simulation-setup.md](./simulation-setup.md) §4 ★  
> **핵심**: 컨테이너 `2026-smh-sim` 1개 + 터미널 2개 (시뮬 / 코드)

호스트 WSL에 `ros-humble-desktop`을 설치할 수 **없습니다** (Humble은 Ubuntu 22.04 전용).  
시뮬도 **같은 Docker 이미지** 안에서 실행합니다.

#### 최초 1회

```bash
./scripts/dev_container.sh build
./scripts/dev_container.sh install-gazebo
./scripts/dev_container.sh init
./scripts/dev_container.sh check-gpu      # 선택
```

#### 매일 개발 — 터미널 2개

| 터미널 | 할 일 |
|--------|--------|
| **1** | `./scripts/dev_container.sh sim-bringup` — Gazebo·브리지·카메라 (Ctrl+C로 launch만 중지) |
| **2** | `docker exec -it 2026-smh-sim bash` — `modules/` 수정 후 inference 빌드·실행 |
| 호스트 | `./scripts/dev_container.sh verify-sim` — 토픽 검증 (bringup 실행 중) |

```bash
./scripts/dev_container.sh sim-up          # 컨테이너 없을 때만
./scripts/dev_container.sh sim-bringup     # 터미널1

docker exec -it 2026-smh-sim bash          # 터미널2
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run inference inference_node --ros-args -p use_sim_time:=true

./scripts/dev_container.sh sim-down        # 작업 끝
```

코드 수정 루프: 호스트에서 편집 → 터미널2에서 `colcon build --packages-select inference` → inference 재실행.  
상세: [simulation-setup.md §4.4](./simulation-setup.md#44-일상-개발-루프-코드-수정--빌드--재실행) · **직접 명령**: [§4.8](./simulation-setup.md#48-직접-명령어-치트시트-스크립트-없이)

**사전 조건**

1. Docker Desktop 실행 + WSL Integration 활성화
2. Windows **11** + WSLg (Gazebo·카메라 프리뷰 GUI)
3. Windows 10: VcXsrv 등 X 서버 + `export DISPLAY=...` 필요할 수 있음

**시뮬 기본 설정**

- 카메라: **320×180** JPEG (`config/vehicle_config.yaml`)
- 시각화: **OpenCV 카메라 프리뷰** (`sim_camera_preview`, 기본 640×360)
- 웹 모니터: **OFF** (`use_monitor:=true`로 선택 가능)

상세 트러블슈팅: [simulation.md](./simulation.md)

### 4.7 임의 명령 실행

`2026-smh-sim`이 실행 중이면 **같은 컨테이너**에서 실행됩니다 (`sim-up` 없으면 일회성 dev).

```bash
./scripts/dev_container.sh "colcon test --packages-select inference"
```

---

## 5. Cursor / VS Code Dev Container

1. Cursor에서 `2026-SMH` 폴더 열기
2. Command Palette → **Dev Containers: Reopen in Container**
3. `.devcontainer/devcontainer.json` 기준으로 컨테이너가 열리고, `postCreateCommand`로 `init_workspace.sh`가 자동 실행됩니다.

Dev Container를 쓰면 터미널·Python 확장이 모두 컨테이너 환경에서 동작합니다.

---

## 6. D3-G 보드 (주행 테스트)

Docker는 PC 빌드 검증용입니다. **merge된 코드의 실제 주행 확인은 보드에서** 합니다.

```bash
cd ~/2026-SMH
./scripts/board_sync.sh
source install/setup.bash
ros2 launch inference manual_driving.launch.py   # 수동
ros2 launch inference auto_driving.launch.py     # 자율
```

상세: [setup.md](./setup.md) · [board-workflow.md](./board-workflow.md)

---

## 7. 트러블슈팅

### `docker: command not found`

- Docker Desktop이 실행 중인지 확인
- WSL Integration이 해당 배포판에 켜져 있는지 확인

### `permission denied` (build/install 파일)

컨테이너가 root로 `build/`, `install/`을 생성한 경우 WSL 호스트에서 권한 문제가 날 수 있습니다.

```bash
# 호스트(WSL)에서
sudo rm -rf build install log
./scripts/dev_container.sh build-inference
```

`build/`, `install/`, `log/`는 `.gitignore` 대상이므로 삭제해도 소스에는 영향 없습니다.

### `control_msgs` 빌드 오류

`inference`만 단독 빌드하면 의존 패키지가 없어 실패합니다. 항상:

```bash
colcon build --symlink-install --packages-up-to inference
```

### D-Racer-Kit clone 실패

- 네트워크·방화벽 확인
- 수동 clone:

```bash
git clone --branch release/v1.0.0 --depth 1 \
  https://github.com/topst-development/D-Racer-Kit.git \
  external/D-Racer-Kit
```

### CI는 통과하는데 보드에서만 실패

- 보드에서 `./scripts/board_sync.sh` 재실행
- 하드웨어(I2C, USB, 전원) 확인 — [D-Racer 조립 가이드](https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0/docs)

---

## 8. 관련 문서

| 문서 | 내용 |
|------|------|
| [setup.md](./setup.md) | D3-G 보드 셋업, launch 실행 |
| [collaboration.md](./collaboration.md) | 브랜치·PR 규약 |
| [Dockerfile](../Dockerfile) | 팀 dev 이미지 정의 |
| [docker-compose.yml](../docker-compose.yml) | 컨테이너 실행 설정 |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | GitHub Actions CI |

---

## 9. 요약 치트시트

```bash
# 최초 셋업 (WSL 22.04 / 24.04 / 26.04 동일)
chmod +x scripts/*.sh
./scripts/dev_container.sh build
./scripts/dev_container.sh init

# 매 작업 (sim-up 중이면 init/check/build도 같은 컨테이너)
./scripts/dev_container.sh sim-up
./scripts/dev_container.sh check          # PR 전 inference 검증

# Gazebo 시뮬 (Docker 안에서 — 호스트에 ros-humble 불필요)
./scripts/dev_container.sh sim-up
./scripts/dev_container.sh sim-bringup      # 터미널1
docker exec -it 2026-smh-sim bash           # 터미널2

# merge 후 (D3-G 실차)
./scripts/board_sync.sh
ros2 launch inference auto_driving.launch.py
```
