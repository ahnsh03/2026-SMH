# 구간별 테스트 파이프라인 (Out 갈림 / In 탈출 / 합류)

**SSOT:** 갈림·색·용어는 [lane-occlusion-fork-strategy.md §0](./lane-occlusion-fork-strategy.md) · spawn 좌표는 [simulation-setup.md §5.0](./simulation-setup.md) · `spawn_poses.yaml`.

---

## 1. 공통 기본값 (모든 라이브 테스트)

한 번 정해 두고 구간마다 **spawn·route·forced_turn** 만 바꾼다.

| 항목 | 기본값 | 어디서 | 이유 |
|------|--------|--------|------|
| Gazebo 창 | **`view:=both`** (주행 확인) / fork만 `none` | T1 `sim-bringup` | 기본은 카메라+BEV; fork 오버레이 가릴 때만 none |
| 인지 창 | **`viz:=lane`** | T2 `sim-auto` | `Lane / Fork Perception` **1창** (표지-armed fork만 강조) |
| 인지 창 끔 | `viz:=off` | T2 | 헤드리스·로그만 |
| 튜닝용 | `viz:=debug` | T2 | lane + inference `Lane drive` (`LANE_VISUALIZE=control`) |
| HSV까지 | `viz:=all` | T2 | + `HSV masks` |
| 방향 강제 | **`forced_turn:=left`** (rank 0) / **`right`** (rank 1) | T2 | 표지 **방향만** 고정. OUT 갈림 인지 상시 ON 아님 (`out_fork_forced_turn_arms`) |
| route | spawn 구간과 **일치** | T2 | Out=`route_mode:=out`, In=`route_mode:=in` |
| 텔레포트 | bringup 유지 + `./scripts/dev_container.sh teleport <pose>` | T3 | Gazebo 재기동 없이 구간 이동 |

**OUT 갈림 인지:** 평소 `enable_fork=false`(흰 추종). 방향 표지 검출 후 hold 동안만 fork 발행 → `FORK_TURN` → 완료 후 다시 흰.  
전체 랩 A/B: `scripts/drive_test/course_mode_bench.py` · 정책 스윕: `mask_policy_bench.py`.

**터미널 역할**

| 터미널 | 명령 |
|--------|------|
| **T1** | `./scripts/dev_container.sh sim-bringup spawn_pose:=<구간> view:=none` |
| **T2** | `./scripts/dev_container.sh sim-auto route_mode:=<in\|out> [forced_turn:=…] viz:=lane` |
| **T3** (선택) | `python3 scripts/drive_test/fork_spawn_unit.py --mode live …` 또는 `teleport` |

**코드 수정 후**

| 변경 | 재시작 |
|------|--------|
| `src/inference/**/*.py` | T2 `sim-auto`만 |
| `*.launch.py`, bridge | `build-sim` + **T1 bringup** |

---

## 2. 시각화 창 · 키 · 확인 필드

### 2.1 창 매트릭스

| 목적 | bringup `view` | sim-auto `viz` | 창 이름 |
|------|----------------|----------------|---------|
| **갈림 SSOT (권장)** | `none` | `lane` | `Lane / Fork Perception` |
| **주행 확인** | `both` | `debug` | cam + BEV + `Lane drive` |
| 로그만 | `none` | `off` | 없음 |
| 마스크/HSV 튜닝 | `none` | `all` | lane + `Lane drive` + `HSV masks` |

### 2.2 `Lane / Fork Perception` 키

| 키 | 동작 |
|----|------|
| `0` | 전체 갈래 (`focus=all`) |
| `1` | rank **0** (LEFT) |
| `2` | rank **1** (RIGHT) |
| `a` | 플래너 잠금 rank에 **자동 동기** |
| `q` / Esc | 창 닫기 |

### 2.3 패널에서 볼 값 (합격 체크)

| 필드 | Out 갈림 | In 탈출 | 합류(무시) |
|------|----------|---------|------------|
| `fork` | `1` (탐색 중) | `1` | **`0` 유지** |
| `branches` | `2` → 잠금 후 `1` | `2` → 잠금 후 `1` | **`1`** |
| `src` | `road_split_marks` / `white_*` | `yellow_alt_marks` 우선 | fork 소스 **없음** |
| `policy` | `explore` → `locked` | 동일 | `explore` 또는 `ego_only` |
| `white_c` / `yellow_c` | Out: white 우세 | In: yellow 우세(가능 시) | ego 색만 우세 |

---

## 3. 구간별 파이프라인

### 3.0 요약 표

| ID | 구간 | spawn_pose | route | forced_turn | viz | fork 기대 | 오프라인 |
|----|------|------------|-------|-------------|-----|-----------|----------|
| **S0** | 출발·직선 | `start` | `out` | — | `off` 또는 `lane` | fork=0 | — |
| **S1** | In/Out 코스 선택 | `inout_fork` | `in`/`out` | — | `lane` | fork=0 | — |
| **O1** | Out 갈림 · 강제/표지 창 | `out_fork` | `out` | `left` 또는 실표지 | `lane` | **fork_on=1**, fork=1, br=2 | — |
| **O1b** | Out 평상 게이트 | 흰 직선·합류 전 | `out` | — | `lane` | **fork_on=0** | — |
| **O2** | Out 갈림 LEFT | `out_fork` | `out` | `left` | `lane` | rank **0** 잠금 | `out_left` |
| **O3** | Out 갈림 RIGHT | `out_fork` | `out` | `right` | `lane` | rank **1** 잠금 | `out_right` |
| **O4** | Out 합류 무시 | `out_fork_merge_left` | `out` | — | `lane` | **fork=0** | (수동) |
| **O5** | Out 합류 무시 | `out_fork_merge_right` | `out` | — | `lane` | **fork=0** | (수동) |
| **I1** | In 진입(흰) | `in_roundabout_entry` | `in` | — | `lane` | fork=0, white path | — |
| **I2** | In 회전(노란) | `in_roundabout_entry` → 주행 | `in` | — | `lane` | yellow 우선 | — |
| **I3** | **In 탈출 탐색** | `in_roundabout_exit` | `in` | — | `lane` | fork=1, br=2 | — |
| **I4** | In 탈출 LEFT | `in_roundabout_exit` | `in` | `left` | `lane` | rank **0**, src=yellow_alt | `in_exit_left` |
| **I5** | In 탈출 RIGHT | `in_roundabout_exit` | `in` | `right` | `lane` | rank **1** (유지) | `in_exit_right` |
| **M1** | In→Out 합류 | `in_out_merge` | `in` | — | `lane` | fork=0 | (수동) |
| **M2** | Out→In 합류 | `out_in_merge` | `out` | — | `lane` | fork=0 | (수동) |

---

### 3.1 S0 · S1 — 갈림 아님 (회귀)

**목적:** 단일 차로·색 계약만 확인.

```bash
# S0
./scripts/dev_container.sh sim-bringup spawn_pose:=start view:=none
./scripts/dev_container.sh sim-auto route_mode:=out viz:=off

# S1
./scripts/dev_container.sh teleport inout_fork
./scripts/dev_container.sh sim-auto route_mode:=in viz:=lane   # 또는 out
```

**합격:** `fork=0`, `branches=1`, Out이면 흰 중심만.

---

### 3.2 O1–O3 — Out 갈림 (진짜 분기)

**고정:** `view:=none`, `viz:=lane`, `route_mode:=out`.

```bash
./scripts/dev_container.sh sim-bringup spawn_pose:=out_fork view:=none

# O1b 평상 게이트 (forced 없음 → fork_on=0 기대)
./scripts/dev_container.sh sim-auto route_mode:=out viz:=lane

# O1/O2 LEFT (forced → fork_on=1, rank 0)
./scripts/dev_container.sh sim-auto route_mode:=out forced_turn:=left viz:=lane

# O3 RIGHT
./scripts/dev_container.sh sim-auto route_mode:=out forced_turn:=right viz:=lane
```

**합격**

- O1b: `fork_on=0` (실표지 없으면 갈림 패널 억제)
- O1/O2/O3 (`forced_turn`): `fork_on=1`, 로그 `*** FORCED_TURN=… ***`, `sign_ignored`, 잠금 후 `active=0|1`, PP가 선택 측만
- 노란 갈래 채택 없음 (`route_mode:=out`)

**오프라인 (Gazebo 불필요)**

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all
# 또는 out_left / out_right
```

로그: `data/captures/fork_drive_logs/<stamp>/` · 기대: **`rank_ok`**

---

### 3.3 O4–O5 · M1–M2 — 합류 (fork 로 취급하면 안 됨)

**정책:** [§0.0.1](./lane-occlusion-fork-strategy.md) — 다른 차로 페인트는 far-only spur → `suppress_merge_spur_branches`.

```bash
./scripts/dev_container.sh sim-bringup spawn_pose:=out_fork_merge_left view:=none
./scripts/dev_container.sh sim-auto route_mode:=out viz:=lane
# right / in_out_merge / out_in_merge 는 teleport 로 pose 만 변경
./scripts/dev_container.sh teleport out_fork_merge_right
./scripts/dev_container.sh teleport in_out_merge
./scripts/dev_container.sh teleport out_in_merge
```

**합격:** `fork=0` (또는 spur 제거 후 `branches=1`). `policy=ego_only` 이면 merge 필터 동작 중.

**코드:** `active_lane.py` `MERGE_SPUR_NEAR_X_M=0.55`, `MERGE_SPUR_MIN_NEAR_POINTS=3`.  
**한계:** near 노이즈로 spur가 3점 이상이면 오탐 가능 — 이 구간은 **수동 spawn 검증** + 필요 시 상수 튜닝. 오프라인 harness는 아직 fork 4시나리오만 (`fork_spawn_unit.py`).

---

### 3.4 I1–I2 — In 접근·회전 (갈림 전)

```bash
./scripts/dev_container.sh sim-bringup spawn_pose:=in_roundabout_entry view:=none
./scripts/dev_container.sh sim-auto route_mode:=in viz:=lane
```

**I1 합격:** 진입 직전 **흰** 경로, `fork=0`.  
**I2:** entry에서 주행하며 회전교차로 내부 — **노란이 보이면 노란 우선** (`yellow_c` 우세). 전용 spawn 없음 → 주행 또는 수동 `teleport custom`.

---

### 3.5 I3–I5 — In 탈출 분기

**고정:** `view:=none`, `viz:=lane`, `route_mode:=in`.

```bash
./scripts/dev_container.sh sim-bringup spawn_pose:=in_roundabout_exit view:=none

# I3 탐색
./scripts/dev_container.sh sim-auto route_mode:=in viz:=lane

# I4 탈출
./scripts/dev_container.sh sim-auto route_mode:=in forced_turn:=left viz:=lane

# I5 유지(순환)
./scripts/dev_container.sh sim-auto route_mode:=in forced_turn:=right viz:=lane
```

**합격:** `src=yellow_alt_marks` (폴백 `yellow_marks`), rank 0/1, 잠금 후 단일 갈래.

**오프라인:** `in_exit_left` / `in_exit_right` / `--scenario all`

---

## 4. 오프라인 · pytest · 캡처

| 단계 | 명령 | 기대 |
|------|------|------|
| 단위 | `pytest src/inference/test/test_active_lane.py src/inference/test/test_fork*.py -q` | green |
| fork 계약 | `python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all` | `rank_ok` ×4 |
| 비전 튜닝 | `python3 scripts/vision_tune/tune_lane_detect.py --mode fork` + `c` | `data/captures/lane_tune_logs/` |
| 라이브 로그 | `fork_spawn_unit.py --mode live --scenario out_left --duration 8` | CSV under `fork_drive_logs/` |

**오프라인 고정 프레임** (`fork_spawn_unit.py` 기본):

| 키 | 경로 |
|----|------|
| Out | `data/captures/lane_tune_logs/auto_fork/out_fork/runs/.../source_frame.png` |
| In exit | `data/captures/lane_tune_logs/auto_fork/in_roundabout_exit/runs/.../source_frame.png` |

---

## 5. 빠른 복붙 (일일 루틴)

```bash
# Out 갈림 LEFT (가장 자주 쓰는 조합)
./scripts/dev_container.sh sim-bringup spawn_pose:=out_fork view:=none
./scripts/dev_container.sh sim-auto route_mode:=out forced_turn:=left viz:=lane

# In 탈출 LEFT
./scripts/dev_container.sh sim-bringup spawn_pose:=in_roundabout_exit view:=none
./scripts/dev_container.sh sim-auto route_mode:=in forced_turn:=left viz:=lane

# 오프라인 스모크
python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all
```

---

## 6. 관련 문서

- [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) — 갈림/합류·색·WonJung 이중 DP
- [dev-environment.md §4.6](./dev-environment.md) — 터미널 2개·viz 표
- [scripts/drive_test/README.md](../scripts/drive_test/README.md) — `fork_spawn_unit` 사용법
