# HSV 프로필 (시뮬 · 실차)

> OpenCV HSV (H: 0–179, S/V: 0–255). 런타임·튜너 SSOT = [`config/lane_vision.yaml`](../config/lane_vision.yaml) `hsv:`  
> 관련: [vision_tune/README.md](../scripts/vision_tune/README.md) · [hardware-camera.md](./hardware-camera.md) · [board-workflow.md](./board-workflow.md)

---

## 0. 한 줄 요약

| 환경 | 프로필 | 용도 |
|------|--------|------|
| **Gazebo 시뮬** | `hsv.profiles.sim` | 밝은 Gazebo 텍스처·조명 (원태 시드) |
| **D3-G 실차** | `hsv.profiles.real_car` | 실차 bag 재생 캡처로 튜닝 (보드 적용용) |

런타임은 `hsv.active`가 가리키는 프로필을 **평탄화(flatten)** 한 채널(`hsv.white` …)로 읽는다.  
보드에는 **`active: real_car`** 상태의 `hsv:` 블록만 복사하면 된다.

```bash
# 시뮬 개발 전 프로필 전환 (2026-smh-sim 안)
python3 scripts/vision_tune/hsv.py --apply-profile sim
python3 scripts/vision_tune/hsv.py --apply-profile real_car   # 보드와 동일
```

---

## 1. 채널 역할

| 채널 | 인지 용도 | 코스 |
|------|-----------|------|
| `white` | 흰 차선 마스크 | OUT |
| `yellow` | 노란 차선 마스크 | IN |
| `black_road` | 검정 아스팔트 (도로) | 공통 |
| `red_road` | 빨강 아스팔트 (도로) | 공통 |
| `black_cyan` | 전광판·LED **시안 반사**가 덮인 아스팔트 | OUT 글레어 · IN 일부 |
| `black_cyan_2` | 2차 시안/틸 아스팔트 패치 | IN bag (~930) |

**도로 원시 마스크** `road_raw` = `black_near | red_road | cyan_near`  
`black_near` / `cyan_near` = 각 채널의 **BEV 하단 ego CC만** (open/close **전**, trial #1).  
시안은 white 차선과 OR하지 않음.

차선은 코스별로 white **또는** yellow만 쓴다 (서로 OR 하지 않음).

`red_road`는 hue wrap: `h_min`이 낮을 때 `detect_tune.red_h_low_wrap`으로 저-H 대역을 OR한다.

---

## 1.1 주행가능 영역 (drivable) 생성 — SSOT

**LOCKED 2026-07-16 (trial #1 near + HSV final):**  
black/cyan = morph 전 **near-밴드 질량** CC · morph **open3/close13/1**.  
`real_car` HSV (확정):

| 채널 | H | S | V |
|------|---|---|---|
| white | 0–179 | 0–20 | 210–255 |
| yellow | 15–50 | 50–150 | 160–255 |
| black_road | 17–70 | 0–255 | 15–140 |
| red_road | 0–9 | **110–255** | 120–255 |
| black_cyan | 90–100 | **200–215** | **190–238** |
| black_cyan_2 | 97–105 | 240–255 | 105–180 |

흑 H14/V180 retune은 기각. red S·cyan S/V만 반영. 영상: `ssot_near_morph3_13/`.

캡처 A/B(`viz_raw_hsv_masks.py` · `viz_cyan_ab.py`)로 확정한 순서:

```
BEV(Metric IPM)
  → HSV: white, yellow, black, red, black_cyan, black_cyan_2   (6채널)
  → black_near / cyan_near = near-band mass CC   ← morph 전 (trial #1)
  → road_raw = black_near|red|cyan_near
  → course paint (never OR white∧yellow)
  → morph open 3 / close 13 / 1 iter + 소구멍
  → BEV 하단 ego CC (= 최종 ego blob; 밴드 질량 점수)  ← morph 후
  → drivable_area
```

| 단계 | 역할 |
|------|------|
| black/cyan pre-ego | morph 전에 로봇 근처(near 밴드 질량) 덩어리만 — 트랙 밖·글레어 bridge 차단 |
| morph | open **3** / close **13** / 1회 · 채널 마스크는 open만 |
| course paint | `resolve_course_lane_mask` — IN 노란 우선, OUT 흰만 |
| ego blob (후) | morph 뒤 하단 밴드 질량 최대 CC 1개 |
| 런타임 | `masks.extract_bev_masks` 동일 black_near / cyan_near → morph / near-ego |

튜너/검증 (이전 bag 캡처 기준, **6채널**):

```bash
# 전체 캡처 세트에서 1–6 채널 순회 튜닝
python3 scripts/vision_tune/tune_hsv.py --from-bag all

python3 scripts/vision_tune/tune_hsv.py --from-bag out --channel white
python3 scripts/vision_tune/tune_hsv.py --from-bag in_yellow --channel yellow
python3 scripts/vision_tune/tune_hsv.py --from-bag black_road --channel black_road
python3 scripts/vision_tune/tune_hsv.py --from-bag out --channel red_road
python3 scripts/vision_tune/tune_hsv.py --from-bag out_glare --channel black_cyan
python3 scripts/vision_tune/tune_hsv.py --from-bag in --channel black_cyan_2

python3 scripts/vision_tune/viz_raw_hsv_masks.py --from-bag out_glare --all
python3 scripts/vision_tune/score_near_floor_select.py \
  --from-bev data/captures/bev_videos/out.mp4 --start 1412 --end 1491
```

IN 회전교차로 **유지/탈출 직전** 시점 감지(yellow far dual)는 차로 마스크 SSOT와 별도 — [lane-occlusion-fork-strategy.md](./lane-occlusion-fork-strategy.md) **§5.1.2**, 코드·데이터는 [fork-moment-detection.md](./fork-moment-detection.md).

---

## 2. 프로필 비교표

### white (OUT 차선)

| | H | S | V |
|--|---|---|---|
| **sim** | 0 – 179 | 0 – 29 | 174 – 255 |
| **real_car** | 0 – 179 | 0 – 20 | 210 – 255 |

실차는 채도·명도를 더 타이트하게 잡아 Gazebo보다 밝은 흰 테이프 노이즈를 줄인다.

### yellow (IN 차선)

| | H | S | V |
|--|---|---|---|
| **sim** | 0 – 55 | 32 – 255 | 79 – 255 |
| **real_car** | 15 – 50 | 50 – 150 | 160 – 255 |

실차는 노란 점선이 어두운 아스팔트 위에서 채도·명도 범위가 좁다.

### black_road

| | H | S | V |
|--|---|---|---|
| **sim** | 0 – 179 | 0 – 255 | 0 – 30 |
| **real_car** | 17 – 70 | 0 – 255 | **15 – 140** |

실차 검정 차로: V 15–140 유지 (H14/V180 retune은 기각 — 과확장).  
**LOCKED 순서 (trial #1):** near-band **질량** CC 선택 → morph(open3/close13) → 하단 ego.  
선택은 **near 밴드 내부 면적** 최대 (전체 area 아님 — OUT 커브 1412–1491 오선택 수정).  
near 밴드 = BEV **하단 높이 비율** (`keep_near_floor` 35%, morph 후 `keep_bottom` 18%).  
trial #2(top_drop)는 기각 — 트랙 밖 바닥이 남는 경우 있음 (`black_trials_*` 비교 영상).  
검증: `score_near_floor_select.py --from-bev …/out.mp4 --start 1412 --end 1491`

### black_cyan (OUT LED 바닥 반사)

| | H | S | V |
|--|---|---|---|
| **sim** | 90 – 100 | 190 – 220 | 200 – 230 |
| **real_car** | 90 – 100 | **200 – 215** | **190 – 238** |

OUT 글레어 retune 적용: S 좁힘 + V 약간 확장. `black_cyan_2`와 OR 후 **near-ego CC만**.

### black_cyan_2 (2차 시안/틸 패치)

| | H | S | V |
|--|---|---|---|
| **sim / real_car** | 97 – 105 | 240 – 255 | 105 – 180 |

IN bag ~930 대역. 튜너 키 **`6`**.

### red_road

| | H | S | V | 비고 |
|--|---|---|---|-----|
| **sim** | 170 – 179 | 125 – 192 | 161 – 229 | 고-H 래핑 대역 |
| **real_car** | 0 – 9 | **110 – 255** | 120 – 255 | 저-H + `red_h_low_wrap: 15` |

실차 빨강 도로: S 하한을 110으로 완화(2026-07-16 retune).

---

## 3. 출처·튜닝 이력

### sim (`profiles.sim`)

| 항목 | 값 |
|------|-----|
| 환경 | Gazebo LIMO, C920e 320×180 (시뮬 카메라) |
| 시드 | `feature/wontae-lane` 상수 (`hsv.py` `_SIM_DEFAULTS`) |
| 튜너 | `tune_hsv.py` 키 **`d`** (Won Tae seed reset) |
| 검증 | `tune_hsv.py --topic /camera/image/compressed` 또는 `data/captures/sim/` |

### real_car (`profiles.real_car`)

| 항목 | 값 |
|------|-----|
| 환경 | D3-G 보드 + C920e, Metric IPM BEV |
| bag | IN `bag_20260711_150234`, OUT `bag_20260711_144948` |
| 캡처 | `capture_from_bag.py` → `data/captures/from_bag/{in,out}/` · 글레어 `out_glare/` |
| 튜너 | `tune_hsv.py --from-bag all\|in\|out\|…`, 저장 키 **`s`** |
| Git | `0191811` (전 채널), `35ba99e` (`red_road`), `2e37ae8` (`black_cyan`), **2026-07-16** near SSOT + red/cyan retune (black H/V 유지) |
| 확정 메모 | black **미확대단** 기각 · red Smin110 · cyan S200–215 V190–238 · morph **3/13/1** |

### 참고: origin/board 1차 실차값 (레거시)

`origin/board` 브랜치 초기 필드 튜닝. `tune_hsv.py` 키 **`b`** 로 튜너에서만 불러온다.  
현재 보드 SSOT는 위 **real_car** 프로필이며, 이 값과 일부 채널이 다르다.

---

## 4. 보드 적용 절차

1. PC에서 `board_sync.sh` 또는 `git pull`로 최신 `feature/seunghyun-recover-pre-pdc` (또는 머지된 main) 받기
2. `config/lane_vision.yaml` 확인:
   - `hsv.active: real_car`
   - `hsv.white` … `hsv.black_cyan`이 real_car 프로필과 일치
3. **HSV만** 옮길 때: `hsv:` 블록 전체 복사 (`metric_ipm:` 은 별도 협의)
4. 런타임 재시작 (`auto_driving.launch.py`)

보드에서 HSV만 수동 편집할 경우 `profiles.real_car`와 평탄화 채널을 **둘 다** 맞춘다.

---

## 5. 시뮬 개발 절차

1. Gazebo bringup 후 컨테이너에서:
   ```bash
   python3 scripts/vision_tune/hsv.py --apply-profile sim
   ```
2. 마스크 확인:
   ```bash
   python3 scripts/vision_tune/tune_hsv.py --topic /camera/image/compressed
   ```
3. 조정 후 `s` 저장 → `profiles.sim` + `active` 갱신 (시뮬 전용이면 `active: sim` 유지)
4. 실차 bag으로 다시 맞출 때는 `--apply-profile real_car` 후 `tune_hsv.py --from-bag …`

---

## 6. YAML 스키마

```yaml
hsv:
  active: real_car          # sim | real_car
  profiles:
    sim:
      meta: { environment, source, tuned, tool }
      white: { h_min, h_max, s_min, s_max, v_min, v_max }
      # yellow, black_road, red_road …
    real_car:
      meta: { … }
      white: { … }
  # 아래 = profiles[active] 미러 (런타임이 읽는 SSOT)
  white: { … }
  yellow: { … }
  black_road: { … }
  red_road: { … }
  black_cyan: { … }   # OUT LED floor wash; OR into road_raw
```

도구: [`scripts/vision_tune/hsv.py`](../scripts/vision_tune/hsv.py) · [`tune_hsv.py`](../scripts/vision_tune/tune_hsv.py)

| 튜너 키 | 동작 |
|---------|------|
| `d` | 활성 채널 → **sim** 시드 |
| `b` | 활성 채널 → origin/board 1차 실차 (레거시) |
| `s` | 현재 값 → `profiles[active]` + 평탄화 저장 |

---

## 7. 담당

| 항목 | 담당 |
|------|------|
| 튜너·yaml·문서 | 안승현 |
| 대회 최종 정밀값 | 장원태 (sim 시드 제공) |
| 보드 배포 | 보드 담당 (`board-workflow.md`) |
