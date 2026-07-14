# 보드 개발·주행 가이드

> **대상**: D3-G 보드에서 코드를 받아 빌드하고 주행 테스트하는 팀원  
> **작업 루트**: `~/2026-SMH` (기존 `~/D-Racer-Kit`은 공식 패키지 공급원으로만 사용)

---

## 한 줄 요약

보드 공유 브랜치는 **`board`** 입니다 (`snapshot/wonjung-board-*` 에서 개명).  
팀원이 보드에서 push/pull·주행 테스트할 때 이 브랜치를 씁니다.

```bash
cd ~/2026-SMH
./scripts/board_sync.sh          # origin/board checkout + pull + build
source install/setup.bash
ros2 launch inference auto_driving.launch.py
# = camera + control_node + inference_node(MainPlanner → /control)
#   + battery + joystick + monitor(웹 관측)
# launch는 planner_profile:=real → PDC fork 판단 + 실차 프로파일
```

### 실차 launch에 올라가는 노드

| 노드 | 필수? | 설명 |
|------|-------|------|
| `camera_node` | 필수 | USB 카메라 |
| `control_node` | 필수 | `/control` → 모터 |
| `inference_node` | 필수 | MainPlanner → `/control` (+ `/perception/lane` 검증용) |
| `joystick_node` | 권장 | E-Stop |
| `battery_node` | 권장 | 배터리 |
| `monitor_node` | 선택 | PC 브라우저로 보드·영상 확인 (`http://<보드IP>:5000`). **없어도 주행 가능** |

**실행하지 말 것:** `lane_control_node` (레거시 — MainPlanner와 `/control` 충돌).

시뮬 전용(`sim_*`, Gazebo)은 보드에서 실행하지 않음.  
전체 비교: [lane-perception-topic.md §2](./lane-perception-topic.md) · [main-planner.md](./main-planner.md)

---

## 1. 디렉터리 관계

```
~/2026-SMH/                 ← ★ 작업 루트
├── config/vehicle_config.yaml
├── external/D-Racer-Kit → ~/D-Racer-Kit  (심볼릭 링크, 또는 init 시 clone)
├── src/
│   ├── inference/          ← ★ 팀 자율주행 (Git)
│   ├── dracer_sim/         ← ★ Gazebo 시뮬 (Git)
│   ├── limo_car/  → vendor/limo_car
│   ├── camera/    → external/D-Racer-Kit/src/camera
│   └── ...
├── vendor/limo_car/        ← LIMO URDF·mesh (Git)
└── install/                ← colcon 빌드 결과
```

| 경로 | 역할 |
|------|------|
| `~/D-Racer-Kit` | 공식 ROS2 패키지 원본 (수정하지 않음) |
| `~/2026-SMH` | 팀 워크스페이스 — 개발·주행 |
| `~/2026-SMH/src/inference/` | 팀 자율주행 코드 |
| `~/2026-SMH/src/dracer_sim/` | Gazebo 시뮬 (D-Racer 토픽 호환) |
| `~/2026-SMH/install/` | 빌드 결과 |

**PC에서 모듈 검증** (보드 전): [simulation-setup.md](./simulation-setup.md) §4 — 터미널1 `sim-bringup`, 터미널2 `sim-auto route_mode:=out viz:=lane`

실차 자율: `ros2 launch inference auto_driving.launch.py` (`planner_profile: real`).  
코스: `route_mode:=in|out`. 디버그: `/debug/planner`, `/debug/aruco`.

---

## 2. 보드 최초 셋업 (1회)

### Case A — `~/D-Racer-Kit`이 이미 있는 경우 (권장)

공식 가이드로 D-Racer-Kit 세팅을 마쳤다면, **다시 clone하지 않고** 링크만 연결합니다.

```bash
cd ~

# 1) 팀 레포 clone
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh

# 2) 기존 D-Racer-Kit 재사용
# external/D-Racer-Kit이 이미 clone 디렉터리면 먼저 제거
rm -rf external/D-Racer-Kit
mkdir -p external
ln -sfn ~/D-Racer-Kit external/D-Racer-Kit

# 3) 워크스페이스 구성 + 빌드
./scripts/board_sync.sh --no-pull
```

### Case B — 처음부터 시작하는 경우

`external/D-Racer-Kit`이 없으면 `init_workspace.sh`가 자동으로 clone합니다.

```bash
cd ~
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull
```

### 셋업 확인

```bash
cd ~/2026-SMH
source install/setup.bash

# 수동 주행 (조이스틱) — camera + monitor 포함
ros2 launch inference manual_driving.launch.py

# 자율주행 (팀 파이프라인) — camera + monitor + inference 포함
ros2 launch inference auto_driving.launch.py
```

> **주의**: `ros2 launch control auto_driving.launch.py`는 주최측 launch입니다.  
> 팀은 **`ros2 launch inference auto_driving.launch.py`** 를 사용합니다.

팀 launch에는 `monitor_node`가 포함되어 있습니다.  
웹 모니터: `http://<WEB_HOST>:<WEB_PORT>` (`src/config/vehicle_config.yaml`, 예: `http://10.0.0.23:5000`)

**절대 하지 말 것**: `~/D-Racer-Kit`에서 `camera_node` / `monitor_node`를 따로 실행하는 것.  
장치가 점유되면 카메라 프레임이 안 나오고, `/debug/aruco`도 비어 보입니다.

---

## 3. 팀원 개발 흐름 (표준)

팀 표준은 **PC에서 PR → `main` merge → 보드 `board`에 반영 후 테스트**입니다.

```
PC (Docker)                    GitHub                         D3-G 보드
───────────                    ──────                         ─────────
feature 브랜치 개발    →    PR → merge → main    →    board에 merge/반영
modules/ 수정                                          board_sync.sh → launch
```

보드에서 바로 고칠 때도 **`board` 브랜치에서만** push/pull 합니다 (`main` 직접 push 금지).

### 3.1 PC에서 개발 (권장)

```bash
cd ~/projects/2026-seame-hackathon/2026-SMH   # 또는 개인 clone 경로

git checkout main && git pull origin main
git checkout -b feature/wontae-lane           # 담당자별 브랜치

# modules/ 수정 후 PR 전 검증
./scripts/dev_container.sh check

git add src/inference/inference/modules/lane_detection.py
git commit -m "feat(lane): HSV 기반 차선 중심 추정"
git push -u origin feature/wontae-lane

# GitHub에서 PR 생성 → 팀장 merge
```

상세 Git 규약: [collaboration.md](./collaboration.md)

### 3.2 merge 후 보드에서 테스트

```bash
cd ~/2026-SMH
./scripts/board_sync.sh          # pull + init + build
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

보드에서는 **개인 feature 브랜치로 주행 테스트하지 않습니다.** 공유 브랜치 `board`만 사용합니다.

### 3.3 ArUco 보드 테스트 (검증됨)

인쇄 크기(10 cm / 15 cm)는 **거리 계산에 쓰지 않습니다.**  
반드시 **`DICT_6X6_50` / ID `3`** 마커여야 정지합니다.

```bash
cd ~/2026-SMH
./scripts/board_sync.sh
source install/setup.bash

# 터미널 1 — 팀 launch만 (D-Racer-Kit 노드 별도 실행 금지)
ros2 launch inference auto_driving.launch.py

# 터미널 2 — 카메라 발행 확인 (~30 Hz)
ros2 topic hz /camera/image/compressed

# 터미널 3 — ArUco 디버그
ros2 topic echo /debug/aruco
```

| 확인 | 기대 |
|------|------|
| 모니터 | `http://10.0.0.23:5000` (또는 `WEB_HOST`)에서 카메라 영상 |
| `/debug/aruco` | 마커를 들면 `detected=1 marker_id=3` → ~0.15 s 후 `should_stop=1` |
| 마커 제거 | ~1.5 s 후 `should_stop=0` |

처음에는 정차 상태에서 마커만 비추는 편이 안전합니다.

---

## 4. 담당 모듈 (수정 가능 파일)

| 담당 | 파일 |
|------|------|
| **안승현(임시)** / 장원태 | `src/inference/inference/modules/lane_detection.py` |
| 장원정 | `src/inference/inference/modules/traffic_sign.py` |
| 안승현 | `src/inference/inference/modules/aruco/detector.py` |
| 박성준 | `src/inference/inference/modules/aruco/stop_logic.py` |
| **양서준** | `src/inference/inference/pipeline.py` (MainPlanner) · `config/main_planner.yaml` |

**건드리지 말 것** (팀장 통합 / 양서준 플래너 영역 혼동 방지):

- `inference_node.py`, `types.py`, `lane_adapters.py` — 팀장
- MainPlanner FSM·PP 게인 — 양서준 (인지 작업 시 건드리지 않음)

### 모듈만 수정 후 빠른 빌드

```bash
cd ~/2026-SMH
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select inference
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

---

## 5. 보드에서 직접 개발할 때 (비권장, 가능)

긴급 수정이 필요하면 보드에서도 개발할 수 있지만, **`main` 직접 push는 금지**입니다.

```bash
cd ~/2026-SMH

git checkout main
git pull origin main
git checkout -b feature/seunghyun-aruco-detect

# modules/ 수정 후
git add src/inference/inference/modules/aruco/detector.py
git commit -m "feat(aruco): add marker detection"
git push -u origin feature/seunghyun-aruco-detect

# GitHub에서 PR 생성 → 팀장 merge
```

### Cursor / Claude Code / Codex 사용 시

작업 디렉터리를 **`~/2026-SMH`** 로 지정하세요. `~/D-Racer-Kit`에서 실행하지 않습니다.

```bash
cd ~/2026-SMH
claude
# 또는 Cursor / Codex 채팅에서 동일 경로로 작업
```

Claude·Cursor·Codex에게 지시할 때는 **담당 `modules/` 파일만** 수정하도록 명시하세요.

보드·PC에서 **PR까지** 맡기려면 **`gh` 설치 + `gh auth login`** 이 필요합니다.  
설치·프롬프트 예시·본문 템플릿: **[collaboration.md §1.7](./collaboration.md#17-github-cli-gh-설치--인증--pr)**.

---

## 6. `board_sync.sh`가 하는 일

| 단계 | 내용 |
|------|------|
| `git fetch` + `checkout board` + `pull --ff-only` | 최신 공유 브랜치 `board` 반영 (`--no-pull`이면 생략) |
| `init_workspace.sh` | `external/D-Racer-Kit/src/*`에서 `src/*`로 심볼릭 링크 |
| `colcon build` | 팀 `inference` + 공식 패키지 한 번에 빌드 |

```bash
./scripts/board_sync.sh          # pull + init + build (일반)
./scripts/board_sync.sh --no-pull  # init + build only (최초 셋업·로컬 변경 테스트)
```

---

## 7. 자주 하는 실수

| 실수 | 올바른 방법 |
|------|-------------|
| `~/D-Racer-Kit`에서 Claude 실행 | `~/2026-SMH`에서 실행 |
| `control` launch 사용 | `inference` launch 사용 |
| `~/D-Racer-Kit`에서 camera/monitor 따로 실행 | 팀 launch만 사용 (`monitor` 포함됨) |
| 공식 패키지(`src/control` 등) 직접 수정 | 수정하지 않음 — `src/inference/`만 |
| clone 후 `init_workspace.sh` 안 함 | `board_sync.sh` 반드시 실행 |
| `main`에 직접 push | `feature/이름-기능` → PR |
| `~/D-Racer-Kit/install/` 사용 | `~/2026-SMH/install/` 사용 |
| 잘못된 ArUco 딕셔너리/ID 인쇄 | `DICT_6X6_50` ID `3`만 정지 |

---

## 8. 트러블슈팅

### `board_sync.sh` 실패

```bash
cd ~/2026-SMH
./scripts/init_workspace.sh
set +u && source /opt/ros/humble/setup.bash && set -u
colcon build --symlink-install
```

### 빌드만 다시

```bash
cd ~/2026-SMH
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select inference
source install/setup.bash
```

### `external/D-Racer-Kit` 링크 확인

```bash
ls -la ~/2026-SMH/external/D-Racer-Kit
# → ~/D-Racer-Kit 을 가리켜야 함 (Case A)
```

### 카메라 / 모니터 / ArUco 가 전부 비어 있을 때

증상: 모니터 영상 없음, `ros2 topic hz /camera/image/compressed`에 publish 없음, `/debug/aruco` 무응답.

1. **구 D-Racer-Kit 노드가 장치를 잡고 있는지** 확인 후 종료:

```bash
ps aux | grep -E 'camera_node|monitor_node' | grep -v grep
# D-Racer-Kit/install/... 경로면 종료
pkill -f 'D-Racer-Kit/.*/camera_node|D-Racer-Kit/.*/monitor_node' || true
fuser /dev/video1   # 비어 있어야 함
```

2. **팀 launch만** 다시 실행:

```bash
cd ~/2026-SMH
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

3. 확인:

```bash
ros2 topic hz /camera/image/compressed   # ~30 Hz
curl -s -o /dev/null -w '%{http_code}\n' http://10.0.0.23:5000/   # 200
ros2 topic echo /debug/aruco --once
```

---

## 관련 문서

- [collaboration.md](./collaboration.md) — 브랜치·PR 규약
- [dev-environment.md](./dev-environment.md) — PC Docker 개발 환경
- [setup.md](./setup.md) — 전체 셋업
- [roles.md](./roles.md) — 역할 분담
