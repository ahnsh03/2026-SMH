# 대회·실차 보드 브랜치 (`board/race-control`)

> PC의 `feature/*` 작업과 **분리된 worktree**에서만 다룹니다.  
> 경로 예: `~/projects/2026-seame-hackathon/2026-SMH-board`

## 한 줄

공식 D-Racer-Kit는 **수정하지 않고** 링크만 하고, 대회 코드는 **`src/inference/` + `config/`** 만 올린다.

```bash
cd /path/to/2026-SMH-board   # 이 worktree / 보드 clone
./scripts/board_race_sync.sh
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

## PC에서 이 폴더를 쓰는 이유

| 경로 | 브랜치 | 용도 |
|------|--------|------|
| `…/2026-SMH` | `feature/…` | 인지 튜닝·실험 (기존 작업 유지) |
| `…/2026-SMH-board` | `board/race-control` | 실차·대회 실행 골격 |

`git worktree`로 묶여 있어 **체크아웃이 서로 덮어쓰지 않습니다.**

```bash
# 현재 레포에서 worktree 목록
cd …/2026-SMH && git worktree list
```

## 레이아웃 (Kit 지침)

```
2026-SMH-board/                 ← 이 브랜치 작업 루트
├── BOARD.md                    ← 지금 문서
├── config/                     ← lane_vision, main_planner(+real_car), vehicle
├── scripts/
│   ├── board_init_workspace.sh ← Kit 링크만 (limo/시뮬 불필요)
│   └── board_race_sync.sh      ← pull + init + 보드 패키지 빌드
├── external/D-Racer-Kit/       ← 공식 clone/링크 (Git 커밋 안 함)
└── src/
    └── inference/              ← ★ 팀 자율주행 (인지·제어·launch)
```

공식 패키지(`camera`, `control`, `joystick`, …)는 `board_init_workspace.sh`가  
`external/D-Racer-Kit/src/*` → `src/*` 심볼릭 링크로 붙입니다.

**실행하지 말 것:** `lane_control_node`, Gazebo/`dracer_sim` (보드 비대상).

## 보드 최초 1회

```bash
cd ~
git clone -b board/race-control https://github.com/ahnsh03/2026-SMH.git 2026-SMH-board
cd 2026-SMH-board
chmod +x scripts/*.sh

# 이미 ~/D-Racer-Kit 이 있으면
rm -rf external/D-Racer-Kit
mkdir -p external
ln -sfn ~/D-Racer-Kit external/D-Racer-Kit

./scripts/board_race_sync.sh --no-pull
```

Kit가 없으면 `board_init_workspace.sh`가 `release/v1.0.0`을 clone합니다.  
보드에서 공식 **최신**을 쓰려면 `~/D-Racer-Kit`에서 `git fetch && git checkout` 후 위 링크를 유지하세요.

## 일상 동기화

```bash
cd ~/2026-SMH-board   # 또는 PC worktree
./scripts/board_race_sync.sh
source install/setup.bash
ros2 launch inference auto_driving.launch.py route_mode:=in   # or out
```

## 이 브랜치에 넣을 코드

| 포함 | 제외(보드에서 안 씀) |
|------|----------------------|
| `src/inference/` | `src/dracer_sim/` 실행 |
| `config/*.yaml` (실차 프로필) | `data/captures/`, bags |
| `scripts/board_*.sh` | vision_tune 대량 산출물 |
| 짧은 보드 문서 (`BOARD.md`) | Docker/시뮬 전용 워크플로 |

제어 로직은 `src/inference/` 안에서만 확장합니다.  
공식 `src/control` 등은 **읽기 전용(링크)** 입니다.

## 설정 SSOT

| 파일 | 역할 |
|------|------|
| `config/lane_vision.yaml` | HSV·BEV (`hsv.active: real_car`) |
| `config/main_planner.yaml` | 추종·미션 |
| `config/main_planner.real_car.yaml` | 실차 L/δ 오버레이 |
| `config/vehicle_config.yaml` | 실차 캘리브·트림 |

## feature 브랜치에서 가져오기

PC에서 인지/제어가 안정되면:

```bash
cd …/2026-SMH-board
git fetch origin
git cherry-pick <commit>          # 또는
git checkout feature/… -- path/to/file
```

반대 방향(보드 튜닝 → feature)도 동일하게 path checkout / cherry-pick.

## 관련 문서

- 상세 보드 워크플로(레거시 포함): [docs/board-workflow.md](docs/board-workflow.md)
- HSV SSOT: [docs/hsv-profiles.md](docs/hsv-profiles.md)
