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
| `black_cyan` | 전광판·LED **시안 반사**가 덮인 아스팔트 | OUT 글레어 구간 |

**도로 원시 마스크** `road_raw` = `black_road | red_road | black_cyan`  
(2026-07-15 OUT LED 바닥 반사 bag 검증 후 **SSOT 확정**. 시안은 white 차선과 OR하지 않음.)

차선은 코스별로 white **또는** yellow만 쓴다 (서로 OR 하지 않음).

`red_road`는 hue wrap: `h_min`이 낮을 때 `detect_tune.red_h_low_wrap`으로 저-H 대역을 OR한다.

---

## 1.1 주행가능 영역 (drivable) 생성 — SSOT

캡처 A/B(`viz_raw_hsv_masks.py` · `viz_cyan_ab.py`)로 확정한 순서:

```
BEV(Metric IPM)
  → HSV: white | yellow | (black|red|black_cyan)
  → morph open/close + 소구멍 채움
  → BEV 하단(로봇)에 닿는 최대 CC만 유지  (= ego blob)
  → drivable_area (제어·mask_p 입력)
```

| 단계 | 역할 |
|------|------|
| `black_cyan` | OUT 전광판 시안 반사로 `black_road`가 뚫리는 구간 보완 |
| ego blob | 바닥과 떨어진 노이즈 덩어리 제거 |
| 런타임 | `perception.backend: blob` — `masks.extract_bev_masks`의 `road_raw`에 시안 OR 후 morph / near-ego CC (`corridor` · `morph_blob`) |

튜너/검증:

```bash
python3 scripts/vision_tune/tune_hsv.py --from-bag out_glare --channel black_cyan
python3 scripts/vision_tune/viz_raw_hsv_masks.py --from-bag out_glare --all
python3 scripts/vision_tune/viz_cyan_ab.py --from-bag out_glare --clean
```

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
| **real_car** | 17 – 70 | 0 – 255 | 50 – 140 |

실차 아스팔트는 Gazebo보다 밝고 색조가 있다 → V 하한·H 밴드 조정.

### black_cyan (OUT LED 바닥 반사)

| | H | S | V |
|--|---|---|---|
| **sim** | 90 – 100 | 190 – 220 | 200 – 230 |
| **real_car** | 90 – 100 | 190 – 220 | 200 – 230 |

OUT 코스 전광판 시안 반사 bag(`from_bag/out_glare`, 2026-07-15)에서 튜닝.  
좁은 고채도·고명도 cyan만 잡아 흰 카펫·흰 테이프 bleed를 피한다. `black_road`와 **OR**.

### red_road

| | H | S | V | 비고 |
|--|---|---|---|-----|
| **sim** | 170 – 179 | 125 – 192 | 161 – 229 | 고-H 래핑 대역 |
| **real_car** | 0 – 9 | 155 – 255 | 120 – 255 | 저-H + `red_h_low_wrap: 15` |

실차 빨강 도로는 조명·카메라에서 저-H 쪽이 더 안정적이다 (커밋 `35ba99e`에서 S/V 완화).

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
| 튜너 | `tune_hsv.py --from-bag in\|out\|out_glare`, 저장 키 **`s`** |
| Git | `0191811` (전 채널), `35ba99e` (`red_road`), `2e37ae8` (`black_cyan`) |

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
