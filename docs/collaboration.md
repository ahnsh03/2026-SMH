# 협업 가이드 (브랜치 · PR · 충돌 방지)

> **목표**: PC에서 개발 → GitHub PR → 보드에서 `git pull` 후 **바로 주행 테스트**  
> **필독**: 모든 팀원은 코드 수정 전 본 문서의 **§1 Git 규약**을 따릅니다.

---

## 1. 팀 Git 규약 (필수)

### 1.1 기본 원칙

| 규칙 | 설명 |
|------|------|
| **`main` 직접 push 금지** | 안정 브랜치는 **Pull Request merge로만** 반영 |
| **브랜치에서만 개발** | `main`에서 바로 코딩하지 않음 — 반드시 feature 브랜치 생성 |
| **작은 PR** | 한 PR = 한 기능 또는 한 버그 수정 (담당 모듈 단위) |
| **merge 후 정리** | merge 완료 후 로컬 feature 브랜치 삭제, `main` pull |
| **보드는 `main`만** | D3-G 배포·주행 테스트는 merge된 `main`만 사용 |

```
main (안정 — 보드 deploy)
  └── feature/<이름>-<기능>   ← 여기서만 개발
         ├── 코드 수정 · 커밋
         ├── push
         ├── Pull Request 생성
         ├── (팀) 리뷰 · CI
         └── merge → main  →  보드에서 board_sync.sh
```

> **요약**: `브랜치 생성 → 작업 → commit → push → PR → (리뷰) → merge`  
> **`main`에 직접 push하는 방식은 사용하지 않습니다.**

### 1.2 표준 개발 절차 (매 작업마다)

```bash
# 0. 저장소 이동
cd ~/projects/2026-seame-hackathon/2026-SMH
# 또는: cd ~/2026-SMH  (보드·단독 clone)

# 1. main 최신화
git checkout main
git pull origin main

# 2. 작업 브랜치 생성 (main에서 분기)
git checkout -b feature/seunghyun-aruco-detect

# 3. 담당 파일만 수정
#    예: src/inference/inference/modules/aruco/detector.py

# 4. 커밋 (담당 파일만 stage)
git add src/inference/inference/modules/aruco/detector.py
git commit -m "feat(aruco): add marker detection with cv2.aruco"

# 5. push
git push -u origin feature/seunghyun-aruco-detect

# 6. Pull Request 생성 (GitHub 웹 또는 gh CLI)
gh pr create \
  --title "feat(aruco): marker detection 초기 구현" \
  --body "## 변경\n- detector.py: ArUco 검출 추가\n\n## 테스트\n- [ ] colcon build --packages-select inference"

# 7. merge 후 로컬 정리
git checkout main
git pull origin main
git branch -d feature/seunghyun-aruco-detect
```

PR은 GitHub 웹에서 생성해도 됩니다. merge 권한은 **팀장**이 수행합니다.

### 1.3 브랜치 이름 규칙

| 패턴 | 예 | 사용 시점 |
|------|-----|-----------|
| `feature/<이름>-<기능>` | `feature/wontae-lane` | **기능 개발 (기본)** |
| `fix/<이름>-<이슈>` | `fix/wonjung-red-signal` | 버그 수정 |
| `docs/<이름>-<주제>` | `docs/seunghyun-setup` | 문서만 변경 |

**팀원별 예시**

| 담당 | 브랜치 예 |
|------|-----------|
| 장원태 | `feature/wontae-lane` |
| 장원정 | `feature/wonjung-traffic` |
| 안승현 | `feature/seunghyun-aruco-detect` |
| 박성준 | `feature/sungjun-aruco-stop` |
| 양서준 | `feature/seojun-roundabout` |

**비권장**

- `main`에서 직접 commit & push
- 개인 이름 브랜치 하나(`seunghyun` 등)에 모든 작업을 계속 쌓기 → PR이 커지고 충돌·리뷰가 어려워짐
- 한 PR에 여러 담당자의 모듈을 동시에 수정

### 1.4 커밋 메시지

```
feat(scope): 한 줄 요약     # 기능 추가
fix(scope): 버그 수정
docs: 문서만 변경
chore: 빌드·설정 등
```

| scope 예 | 담당 모듈 |
|----------|-----------|
| `lane` | 차선 인지 |
| `traffic` | 신호등·표지판 |
| `aruco` | ArUco |
| `roundabout` | 회전 교차로 |

예: `feat(lane): HSV 기반 차선 중심 추정 추가`

### 1.5 PR 타이밍

| 상황 | 행동 |
|------|------|
| 동작하는 최소 단위 완성 | PR 생성 |
| 방향 피드백 필요하지만 미완 | **Draft PR** |
| 리뷰·merge 가능 | Ready for review → 팀장 merge |

PR 생성 시 `.github/pull_request_template.md` 체크리스트를 채웁니다.

### 1.6 도구 역할 (참고)

| 도구 | 역할 |
|------|------|
| **Cursor / 터미널** | 브랜치 생성, commit, push, `gh pr create` |
| **GitHub 웹** | PR 생성·리뷰·merge |
| **Git GUI** (Sublime Merge 등) | 히스토리·diff·merge conflict 해결 |

---

## 2. 코드 수정 규칙 (충돌 방지)

| 규칙 | 설명 |
|------|------|
| **모듈은 각자, 통합은 팀장** | 담당자는 `modules/` 아래 **자기 파일만** 수정 |
| **pipeline은 공유 자원** | `pipeline.py`, `types.py` 변경은 팀장 PR만 |

### 디렉터리 ↔ 담당자

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

| 파일 | 담당 | feature 브랜치 PR |
|------|------|-------------------|
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

## 4. 개발 환경별 역할

| 환경 | 역할 | 빌드 |
|------|------|------|
| **PC (WSL)** | 코드 편집, Git 브랜치·PR | Ubuntu 26.04 등 — ROS2 Humble **미설치** 가능 |
| **D3-G 보드** | `colcon build`, 주행 테스트 | Ubuntu 22.04 + ROS2 Humble |

WSL에서 ROS Humble이 없으면 **코드 편집 + PR**까지 진행하고, 빌드·주행 확인은 **D3-G**에서 `board_sync.sh` 후 테스트합니다.

### PC (WSL) — clone·브랜치 작업

```bash
cd ~/projects/2026-seame-hackathon/2026-SMH
git checkout main && git pull

# feature 브랜치에서 modules/ 수정 → commit → push → PR
git checkout -b feature/wontae-lane
# ... 개발 ...
git push -u origin feature/wontae-lane
gh pr create   # 또는 GitHub 웹
```

로컬 경로·환경 변수 예시는 팀원 개인 `DEV-ENVIRONMENT.md` 참고 (선택).

### D3-G 보드 — merge된 main 테스트

```bash
cd ~/2026-SMH
./scripts/board_sync.sh          # pull + init + build
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

보드에서는 **feature 브랜치로 주행 테스트하지 않습니다.** PR merge 후 `main`을 pull해서 확인합니다.

---

## 5. D3-G 보드 셋업

### 최초 1회

```bash
cd ~
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull   # clone 직후엔 pull 생략
```

### 이후 매번 (main merge 반영)

```bash
cd ~/2026-SMH
./scripts/board_sync.sh
```

> **Note**: `ros2 launch control auto_driving.launch.py`는 주최측 D-Racer-Kit launch입니다.  
> 팀 inference 파이프라인은 **`ros2 launch inference auto_driving.launch.py`** 를 사용하세요.

---

## 6. 충돌이 날 때

| 상황 | 해결 |
|------|------|
| 같은 파일을 두 명이 수정 | **파일 분리 규칙** 준수 (ArUco처럼) |
| `pipeline.py` merge conflict | 팀장이 rebase 후 통합 PR |
| feature 브랜치가 main보다 뒤처짐 | rebase 후 push |
| 보드에서 build 실패 | `./scripts/init_workspace.sh` 재실행 후 build |

```bash
git checkout feature/my-branch
git fetch origin
git rebase origin/main
# conflict → 담당 파일만 직접 해결 → git rebase --continue
git push --force-with-lease
```

---

## 7. PR 체크리스트 (요약)

- [ ] `main`에서 feature 브랜치를 생성했는가
- [ ] **`main`에 직접 push하지 않았는가**
- [ ] 담당 `modules/` 파일만 변경했는가
- [ ] `colcon build --packages-select inference` 성공 (보드 또는 Humble 환경)
- [ ] PR template 체크
- [ ] merge 후 로컬 feature 브랜치 삭제

---

## 8. CODEOWNERS

`.github/CODEOWNERS`에 모듈별 리뷰어가 지정되어 있습니다.  
GitHub username 확인 후 팀장이 각 담당자 handle로 업데이트하세요.

---

## 관련 문서

- [setup.md](./setup.md) — D3-G·WSL 셋업
- [roles.md](./roles.md) — 역할 분담
- [README.md](../README.md) — 저장소 개요
