# Fork moment detection + lane-based branch geometry

> **시점(moment)** 플래그와 **갈래 geometry**는 계층이 다름.  
> 임계·락 이력 SSOT는 [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) **§5.1.2–5.1.3**.  
> 본 문서는 **코드 위치 · 데이터 · 런타임/오프라인 사용법** 보관용.

---

## 1. 계층 구분

| 계층 | 역할 | 코드 | IN / OUT |
|------|------|------|----------|
| **Moment** | 갈림·탈출 **직전** 프레임 플래그 (approach only) | `perception/fork/moment.py` | IN=`in_circle_fork_moment` · OUT=`out_fork_moment` |
| **Lane pairs** | L/R 갈래 **차로 쌍 geometry** | `legacy/lane_detection.py` + `fork/adapter.py` | IN=`yellow_alt`/`yellow_*` · OUT=`road_split`/`white_*` |
| **Planner** | 표지 게이트 · L/R 잠금 · PP | MainPlanner / FSM | moment rising + `enable_fork` |

Moment는 **조향을 직접 만들지 않는다.** 이후 판단·debounce·FSM arm에 쓰일 입력으로 보관한다.

```
BEV HSV masks
    ├─ moment.score_*  →  hard/soft flags (시점)
    └─ legacy fork_*   →  ForkLanePair / RoadBranch (갈래 기하)
              └─ adapter.merge_fork_from_legacy  (표지 게이트 시 overlay)
```

---

## 2. 코드 맵

### 2.1 Moment (신·정리본)

| 경로 | 내용 |
|------|------|
| [`src/inference/inference/modules/perception/fork/moment.py`](../src/inference/inference/modules/perception/fork/moment.py) | `score_in_circle_fork_moment`, `score_out_fork_moment`, 밴드/임계 상수, `combine_road_masks` |
| [`perception/fork/adapter.py`](../src/inference/inference/modules/perception/fork/adapter.py) | 기존: 표지 게이트 시 legacy fork 필드 병합 |
| [`perception/fork/__init__.py`](../src/inference/inference/modules/perception/fork/__init__.py) | `merge_fork_from_legacy` + moment API export |
| [`perception/fork/README.md`](../src/inference/inference/modules/perception/fork/README.md) | 패키지 요약 |

### 2.2 Offline CLI (검증용 thin wrapper)

| 스크립트 | 기본 폴더 | 기대 PASS |
|----------|-----------|-----------|
| [`scripts/vision_tune/score_in_fork_moment.py`](../scripts/vision_tune/score_in_fork_moment.py) | `data/captures/from_bag/in` | POS 0008/09/19 hard · labeled NEG reject |
| [`scripts/vision_tune/score_out_fork_moment.py`](../scripts/vision_tune/score_out_fork_moment.py) | `data/captures/from_bag/out` | POS 0011/12 hard · nontarget FP=0 |

```bash
# 2026-smh-sim 권장
cd /workspace   # or repo root
PYTHONPATH=scripts/vision_tune:src/inference python3 \
  scripts/vision_tune/score_in_fork_moment.py \
  --folder data/captures/from_bag/in \
  --csv data/captures/raw_hsv_masks/in_fork_moment_scores.csv

PYTHONPATH=scripts/vision_tune:src/inference python3 \
  scripts/vision_tune/score_out_fork_moment.py \
  --folder data/captures/from_bag/out \
  --csv data/captures/raw_hsv_masks/out_fork_moment_scores.csv
```

### 2.3 Lane-based fork separation (기존)

| API (legacy) | 용도 |
|--------------|------|
| `fork_lane_pairs_from_dual_courses` | WonJung primary+alt → 좌/우 `ForkLanePair` |
| `extract_marking_fork_lane_pairs` | 차선 마스크 트랙 → 갈래 쌍 |
| `extract_road_split_fork_lane_pairs` | 도로 split / white Out tip |
| tip modes `in_curve` / `out_forward` | IN 곡선 tip · OUT 전방 tip |
| `merge_fork_from_legacy` | blob SSOT 유지 + fork 필드 overlay |

단위 테스트: [`src/inference/test/test_fork_lane_pairs.py`](../src/inference/test/test_fork_lane_pairs.py)  
Moment 단위 테스트: [`src/inference/test/test_fork_moment.py`](../src/inference/test/test_fork_moment.py)

전략·용어 SSOT: [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) (§0 용어, §5 갈림, §5.2 우선순위).

---

## 3. 사용 데이터 기록 (2026-07-15)

캡처는 보통 gitignore (`data/captures/`). bag에서 뽑은 **프레임 id / stem**을 재현 기준으로 삼는다.

### 3.1 소스 bag · 폴더

| 코스 | bag / 캡처 루트 | Mosaic·CSV (선택) |
|------|-----------------|-------------------|
| **IN** | `data/captures/from_bag/in/` ← bag `bags/in_course` 계열 | `data/captures/raw_hsv_masks/in/` · `in_fork_moment_scores.csv` |
| **OUT** | `data/captures/from_bag/out/` | `raw_hsv_masks/out/` · `out_fork_moment_scores.csv` |
| **OUT glare** (시안 LED) | `from_bag/out_glare/` · `raw_hsv_masks/cyan_ab/` | HSV `black_cyan` — road SSOT용, moment 게이트와 별개 |

공통 전처리: `config/lane_vision.yaml` Metric IPM BEV + HSV (`yellow` / `white` / `black_road` / `red_road` / `black_cyan`).  
도로 합성: `road_raw = black | red | cyan` (`combine_road_masks`).

### 3.2 IN moment 라벨

| 역할 | id | stem |
|------|----|------|
| **hard POS** | 0008 | `frame_20260715_045830_994784_0714` |
| **hard POS** | 0009 | `frame_20260715_045837_397668_0734` |
| **hard POS** | 0019 | `frame_20260715_045902_029012_1174` |
| early arm (`hard_base` only) | 0007 | `frame_20260715_045828_850604_0694` |
| hard NEG (평행 레일 등) | 0002, 0004, 0010, 0015, 0017, 0018, 0020 | (스크립트 `EXPECTED_HARD_NEG`) |
| 허용 extra | 0021 | 재접근 — hard 허용 |

입력 마스크: **yellow** + **free = road & ~dilate(yellow)**.  
주의: ego road blob CC 수 단독으로는 불가 (0019는 단일 CC).

### 3.3 OUT moment 라벨

| 역할 | id | stem |
|------|----|------|
| **hard POS** | 0011 | `frame_20260715_045046_248624_1758` |
| **hard POS** | 0012 | `frame_20260715_045053_939644_1784` |
| near / post-apex (직전 게이트면 miss 정상) | 0013 | `frame_20260715_045104_991720_1976` |

검증 (n≈37): hard hit **0011·0012만**, nontarget FP **0**.  
입력: **white** + **road**. road dual만 쓰면 오탐 다수 → **sepW ≥ 150** 필수.

### 3.4 Lane-pair / fork geometry에 쓰이던 신호

| 모드 | 마스크·코스 | tip / source |
|------|-------------|--------------|
| Out | white, road split | `out_forward`, `road_split_marks` / `white_*` |
| In | yellow, yellow_alt (WonJung dual) | `in_curve`, `yellow_alt_marks` |
| 표지 게이트 | planner `enable_fork` | adapter가 legacy fork만 merge |

오프라인 파이프라인 참고: [fork-test-pipeline.md](./fork-test-pipeline.md).

### 3.5 OUT fork *capture* = tip(moment) + stretch(ego) (bag 검증)

| 항목 | 내용 |
|------|------|
| 목적 | bag **갈림 미션**을 white **tip** + ego_blob **Y-stretch**로 통합 포착 |
| Fuse | `capture = in_stretch ∨ tip_in_context` — [`capture.py`](../src/inference/inference/modules/perception/fork/capture.py) |
| CLI | [`score_out_fork_capture.py`](../scripts/vision_tune/score_out_fork_capture.py) (`--from-bag` / `--from-bev` / `--folder`) |
| 문서 | [out-ego-fork-shape.md](./out-ego-fork-shape.md) |
| tip SSOT | §3.3 · [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) §5.1.3 |
| stretch | Gate C `ego.hard` — far dual sep·wide + near throat |
| **판단** | OUT: **표지 ∧ capture** · IN: moment만 + 1회우유지/2회좌탈출 — §5.1.4 · [`judgment.py`](../src/inference/inference/modules/perception/fork/judgment.py) |

`moment.hard`와 `ego.hard`를 플래너에 **날것으로 OR하지 말 것** — `score_out_fork_capture` + `decide_out_fork_arm`만 사용.

---

## 4. Moment API (요약)

```python
from inference.modules.perception.fork.moment import (
    combine_road_masks,
    score_in_circle_fork_moment,
    score_out_fork_moment,
)

road = combine_road_masks(black, red, cyan)  # cyan optional
in_m = score_in_circle_fork_moment(yellow_bev, road)   # .hard / .hard_base / .boosted
out_m = score_out_fork_moment(white_bev, road)         # .hard
```

밴드 (BEV, v=0 = far): far 5–45% · mid 40–70% · near 70–95% · top20 0–20%.

임계 상수는 `moment.py` 상단 (`IN_*`, `OUT_*`) — 수치 변경 시 §5.1.2/§5.1.3와 이 문서의 데이터표를 함께 갱신.

---

## 5. 제어·주행에 붙일 때 (권장)

1. **IN:** `prefer_yellow` / `route_mode=in`일 때만 `in_m.hard` 사용. K프레임 연속 rising → arm. 조향은 이후 `yellow_alt` pairs + FSM.
2. **OUT:** `prefer_yellow=False`일 때만 `out_m.hard`. 표지/`enable_fork`와 AND 권장. L/R은 `road_split`/`white_*` only.
3. Moment와 ego-blob CC count를 **OR로 합치지 말 것** (IN 0019 반례).
4. OUT에 IN yellow 규칙 / IN에 OUT white 규칙 **교차 적용 금지**.

---

## 6. 관련 문서

| 문서 | 링크 |
|------|------|
| 갈림·소실 전략 SSOT | [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) |
| HSV · road SSOT | [hsv-profiles.md](./hsv-profiles.md) |
| Planner | [main-planner.md](./main-planner.md) |
| Fork 테스트 파이프라인 | [fork-test-pipeline.md](./fork-test-pipeline.md) |
| Vision tune | [../scripts/vision_tune/README.md](../scripts/vision_tune/README.md) |
