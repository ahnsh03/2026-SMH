# OUT ego-blob fork stretch + white tip 통합 (bag 검증)

> **목적:** OUT bag 갈림 미션을 **두 신호의 합**으로 포착.  
> **비목적:** 표지 L/R · 반폭 진입 · legacy 차선 추종 (추후).

---

## 1. 두 계층 (기존 + 신규)

| 계층 | API | 입력 | 역할 | bag 상 위치 |
|------|-----|------|------|-------------|
| **Tip / moment** (기존 §5.1.3) | `score_out_fork_moment` | **white** + road | 흰 유도선/고어가 벌어지는 **직전·선단** | out_cam tip near **1280–1313** |
| **Stretch** (ego Y) | `score_out_ego_fork_shape` | **ego_blob** | far 이중 lobe + near 단일 throat **구간** | out_cam label **1280–1313** (접근 1268–) |

```
bag JPEG → Metric IPM BEV
    ├─ white, road ──► moment.hard  = tip
    └─ ego_blob   ──► ego.hard      = in_stretch
              └─ score_out_fork_capture ──► capture
```

**OR로 raw moment와 ego를 플래너에 그냥 합치지 말고**, 아래 fuse만 쓴다.

---

## 2. Fuse (`perception/fork/capture.py`)

| 플래그 | 정의 |
|--------|------|
| `in_stretch` | `ego.hard` (Gate C) |
| `tip` | `moment.hard` |
| `tip_in_context` | `tip ∧ (ego.hard ∨ ego.soft)` — tip이 Y 문맥에서만 |
| **`capture`** | `in_stretch ∨ tip_in_context` |

런타임 권장: `capture`/`in_stretch`에 **K프레임 debounce** → rising arm.  
표지/`enable_fork` AND는 추후.

왜 tip만으로 부족한가: tip은 bag still 2장(고어 apex)에만 hard — **미션 구간 전체를 안 덮음**.  
왜 stretch만으로도 tip이 필요한가: stretch는 구간 존재, tip은 **진짜 흰 갈림 선단 확정** (진입 arm 정밀화).

---

## 3. Gate 요약

### Stretch Gate C (`ego.hard`) — [out-ego-fork-shape.md](./out-ego-fork-shape.md)

`sep_far≥130` · `w_far≥220` · `wr_fn≥2.2` · `dual_far≥50` · `dual_near≤5` · `max_run≥45` · `cov≥30` · `solid∈[0.68,0.82]`

### Tip (`moment.hard`) — [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) §5.1.3

white far/mid dual · `sepW≥150` · wa2 · road dual/span · `road_pct≥28`

---

## 4. bag 기반 검증 명령

```bash
cd /workspace   # 2026-smh-sim
export PYTHONPATH=scripts/vision_tune:src/inference

# A) rosbag2 직접 (bags/out_course)
python3 scripts/vision_tune/score_out_fork_capture.py --from-bag out --stride 5

# B) bag에서보낸 BEV mp4 (반복 실험)
python3 scripts/vision_tune/score_out_fork_capture.py \
  --from-bev data/captures/bev_videos/out.mp4 --stride 5 --recall-dense

# C) 라벨 stills (from_bag/out) — tip POS 회귀
python3 scripts/vision_tune/score_out_fork_capture.py \
  --folder data/captures/from_bag/out
```

코드: [`capture.py`](../src/inference/inference/modules/perception/fork/capture.py) ·  
CLI [`score_out_fork_capture.py`](../scripts/vision_tune/score_out_fork_capture.py) ·  
단위: tip [`moment.py`](../src/inference/inference/modules/perception/fork/moment.py) · stretch [`ego_shape.py`](../src/inference/inference/modules/perception/fork/ego_shape.py).

### 합격 기준

| 소스 | 기준 |
|------|------|
| **folder** | tip POS `1758`/`1784` hard · nontarget tip FP=0 · POS ⊂ `capture` (구 bag) |
| **BEV / bag** | `out_cam` **1280–1313** capture 전 구간 · distant FP≈0 (접근 1268–1279만) |
| **recall** | label 1280–1313에서 `capture` = 34/34 (stride 1) |

---

## 6. 플래너 판단 (요약)

| 코스 | Arm | 선택 |
|------|-----|------|
| OUT | `sign ∧ capture` | 표지 → rank |
| IN | `in_circle_fork_moment` rising | pass1 우 유지 → pass2 좌 탈출 |

[`judgment.py`](../src/inference/inference/modules/perception/fork/judgment.py) · §5.1.4.

---

## 5. 플래너 판단 (채택)

| 코스 | Arm | L/R |
|------|-----|-----|
| **OUT** | **표지 ∧ `out_fork_capture`** | 표지 → rank |
| **IN** | **`in_circle_fork_moment`만** | 1회→우(유지) · 2회→좌(탈출) |

코드: [`judgment.py`](../src/inference/inference/modules/perception/fork/judgment.py) · §5.1.4.  
IN에는 OUT ego capture를 쓰지 않는다 (노란 moment SSOT).

---

## 6. 열린 이슈

- 표지 AND 후 L/R 진입(반폭 vs legacy)
- IN moment↔yellow_alt 추종 실차/시뮬 랩 검증
- 임계 SSOT 잠금은 다른 bag 재검증 후
