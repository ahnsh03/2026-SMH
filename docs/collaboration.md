# 협업 가이드 (브랜치 · PR · 충돌 방지)

> **목표**: PC에서 개발 → GitHub PR → 보드에서 `git pull` 후 **바로 주행 테스트**

---

## 1. 핵심 원칙

| 규칙 | 설명 |
|------|------|
| **모듈은 각자, 통합은 팀장** | 담당자는 `modules/` 아래 **자기 파일만** 수정 |
| **pipeline은 공유 자원** | `pipeline.py`, `types.py` 변경은 팀장 PR만 |
| **보드는 main만** | D3-G에서는 `main` 브랜치 + `board_sync.sh` 사용 |
| **기능 브랜치에서 PR** | `feature/이름-기능` → `main` PR → merge |

---

## 2. 디렉터리 ↔ 담당자

```
src/inference/inference/
├── types.py              ← 팀장 (공통 타입 — 함부로 수정 X)
├── pipeline.py           ← 팀장 (모듈 결과 합치기 — 함부로 수정 X)
├── inference_node.py     ← 팀장 (ROS2 노드 — 함부로 수정 X)
└── modules/
    ├── lane_detection.py       ← 장원태
    ├── traffic_sign.py         ← 장원정
    ├── roundabout.py           ← 양서준
    ├── aruco_detection.py      ← facade (수정 불필요)
    └── aruco/
        ├── detector.py         ← 안승현
        └── stop_logic.py       ← 박성준
```

### 수정 가능 / 불가 파일

| 파일 | 담당 | PR 가능? |
|------|------|----------|
| `modules/lane_detection.py` | 장원태 | ✅ |
| `modules/traffic_sign.py` | 장원정 | ✅ |
| `modules/roundabout.py` | 양서준 | ✅ |
| `modules/aruco/detector.py` | 안승현 | ✅ |
| `modules/aruco/stop_logic.py` | 박성준 | ✅ |
| `modules/aruco_detection.py` | facade | ❌ (팀장만) |
| `pipeline.py`, `types.py` | 팀장 | ❌ (팀원 PR 금지) |

---

## 3. 모듈 입출력 규격

모든 모듈은 `types.py`에 정의된 dataclass를 **반환**합니다. dict/raw tuple 사용 금지.

### lane_detection.detect(frame) → `LaneResult`

```python
LaneResult(steering_offset=-0.2, confidence=0.9)
# steering_offset: -1.0(좌) ~ +1.0(우)
# confidence: 0.0 ~ 1.0
```

### traffic_sign.detect(frame) → `TrafficResult`

```python
TrafficResult(signal=TrafficSignal.GREEN, turn=TurnSign.LEFT)
```

### aruco (내부 2단계)

1. `detector.detect_markers(frame)` → `list[int]` (마커 ID 목록)
2. `stop_logic.should_stop_for_markers(ids)` → `(bool, int | None)`

facade `aruco_detection.detect()` 가 `ArucoResult`로 합칩니다.

### roundabout.plan(frame) → `RoundaboutResult`

```python
RoundaboutResult(active=True, steering=0.3, throttle=0.2)
# active=False 이면 차선 추종(lane)으로 fallback
```

### 통합 우선순위 (`pipeline.fuse_control`)

1. ArUco 정지 (`should_stop`)
2. 빨간 신호등 정지
3. 회전교차로 override (`roundabout.active`)
4. 차선 추종 (기본)

우선순위 변경이 필요하면 **팀장에게 이슈/카톡** → 팀장이 `pipeline.py` PR.

---

## 4. 브랜치 워크플로

```bash
# 1. main 최신화
git checkout main
git pull origin main

# 2. 기능 브랜치 (예: 장원태)
git checkout -b feature/wontae-lane

# 3. 담당 파일만 수정
#    src/inference/inference/modules/lane_detection.py

# 4. 빌드 확인 (PC 또는 보드)
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select inference

# 5. push & PR
git add src/inference/inference/modules/lane_detection.py
git commit -m "feat(lane): add HSV line detection"
git push -u origin feature/wontae-lane
# GitHub에서 PR 생성 → 팀장 merge
```

### 브랜치 이름 규칙

```
feature/<영문이름>-<기능>
```

예: `feature/wontae-lane`, `feature/wonjung-traffic`, `feature/seunghyun-aruco-detect`

---

## 5. D3-G 보드 — pull 후 바로 실행

### 최초 1회

```bash
cd ~
git clone https://github.com/ahnsh03/SEA-Me-Hackathon.git
cd SEA-Me-Hackathon
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull   # clone 직후엔 pull 생략
```

`board_sync.sh`가 자동으로:

1. `git pull` (옵션)
2. `init_workspace.sh` — D-Racer-Kit clone + `src/` 링크
3. `colcon build --symlink-install`

### 이후 매번 (코드 받을 때)

```bash
cd ~/SEA-Me-Hackathon
./scripts/board_sync.sh
```

### 주행 실행 (팀 launch 사용)

```bash
source install/setup.bash

# 수동 주행 (하드웨어 확인)
ros2 launch inference manual_driving.launch.py

# 자율주행
ros2 launch inference auto_driving.launch.py
```

> **Note**: `ros2 launch control auto_driving.launch.py`는 주최측 D-Racer-Kit launch입니다.  
> 팀 inference 파이프라인은 **`inference` 패키지 launch** 를 사용하세요.

---

## 6. PC (WSL) 개발 환경

상위 프로젝트(`2026-seame-hackathon`)를 쓰는 경우 D-Racer-Kit은  
`../external/D-Racer-Kit`에 있어도 `init_workspace.sh`가 자동 인식합니다.

```bash
cd ~/projects/2026-seame-hackathon/SEA-Me-Hackathon
./scripts/init_workspace.sh
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

보드 단독 clone은 `<repo>/external/D-Racer-Kit`에 clone됩니다 (Git 제외).

---

## 7. 충돌이 날 때

| 상황 | 해결 |
|------|------|
| 같은 파일을 두 명이 수정 | **파일 분리 규칙** 준수 (ArUco처럼) |
| `pipeline.py` merge conflict | 팀장이 rebase 후 통합 PR |
| 보드에서 build 실패 | `./scripts/init_workspace.sh` 재실행 후 build |
| `src/camera` 등 없음 | `init_workspace.sh` 미실행 — 스크립트 실행 |

```bash
# feature 브랜치를 main에 맞추기
git checkout feature/my-branch
git fetch origin
git rebase origin/main
# conflict → 담당 파일만 직접 해결 → git rebase --continue
```

---

## 8. PR 체크리스트 (요약)

- [ ] 담당 `modules/` 파일만 변경
- [ ] `colcon build --packages-select inference` 성공
- [ ] PR template 체크
- [ ] (가능 시) 보드에서 `board_sync.sh` 후 launch 테스트

---

## 9. CODEOWNERS

`.github/CODEOWNERS`에 모듈별 리뷰어가 지정되어 있습니다.  
GitHub username 확인 후 팀장이 각 담당자 handle로 업데이트하세요.
