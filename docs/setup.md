# 개발 환경 셋업

## 사전 요구사항

- **보드**: D3-G (Ubuntu 22.04 + ROS2 Humble 공식 이미지) — 스펙: [hardware-board.md](./hardware-board.md)
- **PC**: Windows 10/11 + WSL2 (선택) 또는 D3-G 직접 작업
- 공식 가이드: [D-Racer-Kit docs](https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0/docs)

---

## D3-G 보드 (권장 — 단독 clone)

### Case A — `~/D-Racer-Kit`이 이미 있는 경우

공식 가이드로 D-Racer-Kit 세팅을 마쳤다면 **다시 clone하지 않고** 링크만 연결합니다.

```bash
cd ~
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh

# external/D-Racer-Kit이 이미 clone 디렉터리면 먼저 제거
rm -rf external/D-Racer-Kit
mkdir -p external
ln -sfn ~/D-Racer-Kit external/D-Racer-Kit

./scripts/board_sync.sh --no-pull
```

### Case B — 처음부터 시작하는 경우

```bash
cd ~
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull
```

D-Racer-Kit은 `<repo>/external/D-Racer-Kit`에 clone됩니다 (Git 추적 안 함).

> 보드에서 개발·주행 전체 흐름: **[board-workflow.md](./board-workflow.md)** ★

### 코드 업데이트 (매번)

```bash
cd ~/2026-SMH
./scripts/board_sync.sh
```

`board_sync.sh` = `git pull` + `init_workspace.sh` + `colcon build`

---

## PC (WSL) — Docker 개발 환경 (권장)

팀원 WSL 버전(22.04 / 24.04 / 26.04 등)이 달라도 **동일한 Ubuntu 22.04 + ROS2 Humble** 컨테이너를 사용합니다.

**사전 요구**: [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL2 통합 활성화)

```bash
cd ~/projects/2026-seame-hackathon/2026-SMH
chmod +x scripts/*.sh

# 최초 1회
./scripts/dev_container.sh build
./scripts/dev_container.sh init

# PR 전 검증 (GitHub CI와 동일)
./scripts/dev_container.sh check

# 개발 셸 진입
./scripts/dev_container.sh shell
```

상세 규약·트러블슈팅: **[dev-environment.md](./dev-environment.md)**  
Git 규약: [collaboration.md](./collaboration.md)

### PC (WSL) — 네이티브 22.04 (Docker 미사용 시)

Ubuntu 22.04 WSL에 ROS2 Humble이 설치된 경우에만 해당합니다.

```bash
cd ~/projects/2026-seame-hackathon/2026-SMH
chmod +x scripts/*.sh
./scripts/init_workspace.sh

source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

D-Racer-Kit이 `../external/D-Racer-Kit`에 있으면 자동으로 사용합니다.

> **WSL 24.04 / 26.04 등**: `/opt/ros/humble`이 없을 수 있습니다.  
> **Docker 사용을 권장**합니다. 주행 테스트는 **D3-G 보드**에서 하세요.

---

## init_workspace.sh 동작

| 항목 | 내용 |
|------|------|
| D-Racer-Kit 위치 1 | `<repo>/external/D-Racer-Kit` (보드 단독 clone) |
| D-Racer-Kit 위치 2 | `<repo>/../external/D-Racer-Kit` (PC 상위 프로젝트) |
| `src/` 링크 | camera, control, joystick 등 공식 패키지 심볼릭 링크 |
| 재실행 | 안전 — 링크를 최신 경로로 갱신 |

### 주의

- `src/camera`, `src/control` 등 **공식 패키지 링크는 Git에 없습니다**
- **매 clone·새 보드마다** `init_workspace.sh` (또는 `board_sync.sh`) 실행 필요
- D-Racer-Kit 최초 clone에 **네트워크** 필요 (`release/v1.0.0`)

---

## ROS2 환경

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 실행

```bash
source install/setup.bash

# 수동 주행 (조이스틱) — camera + monitor 포함
ros2 launch inference manual_driving.launch.py

# 자율주행 (팀 inference 파이프라인) — camera + monitor + inference 포함
ros2 launch inference auto_driving.launch.py
```

> 주최측 launch (`ros2 launch control auto_driving.launch.py`) 대신  
> **팀 `inference` 패키지 launch**를 사용하세요.

팀 launch에 `monitor_node`가 포함되어 있습니다.  
웹 모니터 주소는 `src/config/vehicle_config.yaml`의 `WEB_HOST` / `WEB_PORT`입니다  
(보드 예: `http://10.0.0.23:5000`).

> `~/D-Racer-Kit`에서 `camera_node` / `monitor_node`를 **따로 켜지 마세요.**  
> `/dev/video1`을 점유하면 카메라·모니터·`/debug/aruco`가 모두 멈춥니다.

---

## 유용한 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | 카메라 영상 |
| `/control` | `control_msgs/Control` | steering / throttle |
| `/debug/aruco` | `std_msgs/String` | ArUco 디버그 (`detected` / `should_stop` / `marker_id`) |
| `/joystick` | `joystick_msgs/Joystick` | 조이스틱 (E-Stop) |
| `/battery_status` | `battery_msgs/Battery` | 배터리 |

### ArUco 인쇄물 보드 확인 (실차 검증됨)

- Dictionary: **`DICT_6X6_50`**
- Stop ID: **`3`만** 정지 (한 변 10 cm / 15 cm 모두 OK — 크기는 실물만, 코드에 cm 파라미터 없음)

```bash
cd ~/2026-SMH
source install/setup.bash
ros2 launch inference auto_driving.launch.py

# 다른 터미널들
ros2 topic hz /camera/image/compressed    # ~30 Hz여야 함
ros2 topic echo /debug/aruco
# 브라우저: http://10.0.0.23:5000
```

타이밍 (혼동 주의):

- **ENTER 0.15초** — 마커가 보이기 시작한 뒤 `should_stop=1`까지 (빨리 정지)
- **EXIT 1.5초** — 마커가 사라진 뒤 `should_stop=0`까지 (재출발)

기대:

```text
detected=1 should_stop=0 marker_id=3   # 보이기 시작
detected=1 should_stop=1 marker_id=3   # ~0.15s 후 정지
detected=0 should_stop=0 marker_id=None # 치운 뒤 ~1.5s
```

카메라·모니터가 비면: [board-workflow.md §8](./board-workflow.md) 트러블슈팅 참고.

---

## 협업 (Git 규약)

**모든 코드 수정은 feature 브랜치 → PR → merge 방식입니다.** `main` 직접 push 금지.

| 단계 | 명령·행동 |
|------|-----------|
| 1 | `git checkout main && git pull` |
| 2 | `git checkout -b feature/이름-기능` |
| 3 | 담당 `modules/` 수정 · commit · push |
| 4 | GitHub PR 생성 → 팀장 merge |
| 5 | 보드: `./scripts/board_sync.sh` |

상세: [collaboration.md](./collaboration.md)
