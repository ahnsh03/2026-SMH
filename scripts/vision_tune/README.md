# Vision tuning tools (Phase 0+)

시뮬·실차 공통. 설계: [docs/lane-drive-strategy.md](../../docs/lane-drive-strategy.md) §4.

**팀 잠정 SSOT (2026-07-12):** 런타임·플래너 BEV = **Metric IPM**  
(`y_half=0.77 m`, `x_max=1.5 m`). 사다리꼴은 참고용만.

## 어디서 실행하나

| 환경 | 가능? | 비고 |
|------|--------|------|
| 호스트 WSL | ❌ | `cv2` / `rclpy` 없음 |
| **`2026-smh-sim`** | ✅ | **이미지 재빌드 불필요** — ROS source만 |
| D3-G 보드 | ✅ | |

```bash
./scripts/dev_container.sh sim-bringup view:=none   # fork 튜닝용 (기본은 both)
# 또는 일상: ./scripts/dev_container.sh sim-bringup
docker exec -it 2026-smh-sim bash          # 터미널2 (튜너·수동 셸)
source /opt/ros/humble/setup.bash
source install/setup.bash
```

**인지 검증은 Gazebo를 다시 띄우지 마세요.** `sim_auto_driving` / `LANE_VISUALIZE=…` 올인원 launch는 bringup이 이미 있으면 Gazebo가 하나 더 뜹니다.  
자율 주행 검증은 **`sim-auto`** (MainPlanner). `tune_lane_control.py`는 **레거시 `lane_control.yaml`** 전용이며 auto와 `/control` 충돌.  
튜너는 카메라 토픽만·`prefer_yellow`는 장면별 (Out=`--label out_fork` → False, In=`exit` → True). 계약: [lane-occlusion-fork-strategy.md §0](../../docs/lane-occlusion-fork-strategy.md).

---

## 인지 모드 튜너 (주 검증 경로 · Gazebo-free)

**단계형 튜닝 (권장):**

| Phase | 키 | 목표 |
|-------|----|------|
| **A** | `2` yellow → `3` dash | **In 탈출**용 노란 점선 4가닥 |
| **A′** | `1` white → `6` fork | **Out 갈림**용 흰/`road_split` 차로 쌍 |
| **B** | `4` / `5` | (선택) 한쪽 고어 점선만 |
| **C** | `6` / `7` / `8` | L/R 차로 쌍·중앙 (장면별 `src` 확인) |

기본 시작 모드 = **`dash` (Phase A)**.

```bash
python3 scripts/vision_tune/tune_lane_detect.py              # dash부터 (legacy backend)
python3 scripts/vision_tune/tune_lane_detect.py --mode fork
python3 scripts/vision_tune/tune_lane_detect.py --folder data/captures/lane_tune_logs
```

> **레거시 전용.** 이 튜너는 polyfit/11자 rail·fork 스윕용. 시작 시  
> `lane_detection.set_perception_backend('legacy')` 를 쓰거나 yaml을  
> `perception.backend: legacy` 로 둔다. 일반 추종 검증은  
> [`preview_out_drivable.py`](preview_out_drivable.py) (blob) 를 쓴다.

| 키 | 동작 |
|----|------|
| `c` / `SPACE` | **리뷰 번들 저장** → `data/captures/lane_tune_logs/<stamp>_<mode>/` (`LATEST.txt` 갱신) |
| `s` | `hsv:` + `detect_tune:` → `config/lane_vision.yaml` |
| `1`–`9` / `0` | 모드 전환 |
| `n` / `p` | 폴더 다음/이전 |
| `q` | 종료 |

캡처 폴더 내용: `frame.png`, `preview.png`, `yellow_dash_points.png`, `yellow_connected.png`, `meta.yaml` 등.  
에이전트 확인용: `data/captures/lane_tune_logs/LATEST.txt` · `INDEX.md`.

상세: [lane-perception-topic.md §6.2](../../docs/lane-perception-topic.md) · [lane-occlusion-fork-strategy.md](../../docs/lane-occlusion-fork-strategy.md).

---

## 기본 BEV 튜너 = Metric IPM

```bash
python3 scripts/vision_tune/tune_bev.py
# 동일 구현: tune_metric_ipm.py (내부 모듈용 이름)
```

```bash
# 사다리꼴과 나란히 비교
python3 scripts/vision_tune/tune_bev.py --compare

# 토픽 / 오프라인
python3 scripts/vision_tune/tune_bev.py --topic /camera/image/compressed
python3 scripts/vision_tune/tune_bev.py --folder data/captures/sim
```

| 창 | 역할 |
|----|------|
| `ipm_tune (origin \| BEV)` | **단일 창** — 좌: 원본+crop / 우: 변환 BEV + 트랙바 (WSLg 다중창 누락 방지) |

캡처 폴더 예: `python3 scripts/vision_tune/tune_bev.py --folder data/captures/from_bag/in`  
트랙바(`pitch`/`height`/`crop` …)를 움직이면 **오른쪽 BEV가 바로** 다시 워프됩니다.

### 트랙바 (IPM)

| 이름 | 역할 | 잠정 기본 |
|------|------|-----------|
| `crop_top_%` | 상단 제외 (≈ `x_max` 행) | **39** |
| `x_min_cm` / `x_max_cm` | 근·원거리 (cm) | **22 / 150** |
| `y_half_cm` | 횡반폭 (cm) | **77** |
| `mpp_mm` | mm/px (등방) | **4** |
| `pitch_x10` / `height_cm` | 카메라 미세 조정 | 100 / 13 |

### 단축키

| 키 | 동작 |
|----|------|
| `s` | `config/lane_vision.yaml` → `metric_ipm:` 저장 |
| `f` | `y_half`를 원거리 이미지 풀폭(~±1.05 m)으로 스냅 |
| `q` / ESC | 종료 |
| `n` / `p` | 폴더 모드 다음/이전 |

### 잠정 SSOT (`config/lane_vision.yaml` → `metric_ipm:`)

| 항목 | 값 |
|------|-----|
| 카메라 | h=0.13 m, pitch=10°, HFoV=70.42° |
| `x_min` / `x_max` | **0.22 / 1.5 m** |
| `y_half_width_m` | **0.77** (`full_image_width: false`) |
| `meters_per_pixel` | **0.004** |
| `crop_top_ratio` | **0.39** |
| BEV | **≈ 386 × 321** |
| 차로 가이드 | half ≈ **44 px** ↔ 0.35 m |

기하: crop 39% ↔ 전방 1.5 m (320×180).

---

## 카메라 캡처 (단축키만 저장)

```bash
python3 scripts/vision_tune/capture_camera.py --out data/captures/sim
```

- 창 `capture_hotkey` (640×360)에 포커스
- **`c` 또는 Space** → PNG 1장 · **`q`** → 종료
- 저장 후 uid 1000으로 chown (호스트 삭제 가능)

---

## 조이스틱 bag 재생 + 캡처 (IN / OUT)

실차 수동 주행 bag을 **Gazebo 없이** 재생하며 PNG를 뽑는다.

| alias | bag 폴더 | 의미 |
|-------|----------|------|
| `out` / `in` | `bags/out_course` · `in_course` | 구 카메라 (07-11) |
| **`out_cam`** | `bags/out_cam_20260715` ← `data/bag_20260715_230145` | **신 카메라 OUT** |
| **`in_cam`** | `bags/in_cam_20260715` ← `…230316` | 신 카메라 IN |
| `sign_right` / `sign_left` | `…230515` / `…230601` | 우·좌회전 표지 |

```bash
# 2026-smh-sim 안에서
source /opt/ros/humble/setup.bash && source install/setup.bash

# --- 카메라 재설정 후 HSV 재튜닝 (OUT부터) ---
python3 scripts/vision_tune/capture_from_bag.py out_cam --dump-stride 15
PYTHONPATH=scripts/vision_tune python3 scripts/vision_tune/tune_hsv.py \
  --from-bag out_cam --channel white
# s 로 lane_vision.yaml 저장 → 채널 1·3·4·5 (white / black_road / red_road / cyan)

# interactive scrub
python3 scripts/vision_tune/capture_from_bag.py out_cam
python3 scripts/vision_tune/capture_from_bag.py out         # 구 bag
```

| 키 | 동작 |
|----|------|
| `c` | 현재 프레임 PNG 저장 → `data/captures/from_bag/<in\|out>/` |
| `SPACE` | 재생 / 일시정지 (**시작은 PAUSE**) |
| `←` `→` / `,` `.` | 프레임 단위 이동 |
| `[` `]` | 재생 속도 ↓ / ↑ |
| `r` | 처음으로 |
| `q` / ESC | 종료 |

창이 안 보이면: 이전 프로세스가 남아 있는지 확인(`Ctrl+C`) 후 다시 실행. 작업표시줄에서 `bag_capture_*` 창을 찾거나, 창은 (48,48)에 뜹니다.

### bag 실시간 주행가능 마스크 (BEV | 6_ego_blob)

**주의:** 초판 `play_bag_drivable`는 런타임 blob(카메라 HSV→워프·morph 3/5)을 써서
사진 mosaic의 **`6_ego_blob`과 달랐다.** 지금은 `viz_raw_hsv_masks.extract_five`와 동일.

권장: bag → **BEV mp4 저장** → 그 영상에서 ego blob 재생 (IPM 1회만).

```bash
# 2026-smh-sim
source /opt/ros/humble/setup.bash && source install/setup.bash

# 1) bag → BEV 영상 저장
python3 scripts/vision_tune/play_bag_drivable.py in --export-bev \
  data/captures/bev_videos/in.mp4

# 2) 저장한 BEV에서 6_ego_blob (좌:BEV / 우:흰 마스크)
python3 scripts/vision_tune/play_bag_drivable.py --from-bev \
  data/captures/bev_videos/in.mp4

# 한 번에: export 후 바로 재생
python3 scripts/vision_tune/play_bag_drivable.py out --export-bev \
  data/captures/bev_videos/out.mp4 --play-after-export

# live (저장 없이) — extract_five/6_ego, IN paint=Y|road or W|road
python3 scripts/vision_tune/play_bag_drivable.py out --rate 0.4
python3 scripts/vision_tune/play_bag_drivable.py --from-bev data/captures/bev_videos/in.mp4 \
  --paint-course in

# black near-CC: score=near-band area (OUT curve 1412–1491 검증)
python3 scripts/vision_tune/score_near_floor_select.py \
  --from-bev data/captures/bev_videos/out.mp4 --start 1412 --end 1491
python3 scripts/vision_tune/play_bag_drivable.py out \
  --from-bev data/captures/bev_videos/out.mp4 --start 1412

# black trial #1 vs #2 × IN/OUT → data/captures/bev_videos/black_trials/
python3 scripts/vision_tune/export_black_trial_videos.py
# SSOT near + morph3/13 + red/cyan HSV → IN/OUT BEV|ego
python3 scripts/vision_tune/export_black_trial_videos.py --modes near --courses in,out \
  --outdir data/captures/bev_videos/ssot_near_morph3_13
# 캡처 mosaic (extract_five SSOT)
python3 scripts/vision_tune/viz_raw_hsv_masks.py --from-bag in --all --clean
python3 scripts/vision_tune/viz_raw_hsv_masks.py --from-bag out --all --clean
python3 scripts/vision_tune/play_bag_drivable.py out \
  --from-bev data/captures/bev_videos/out.mp4
```

IN 페인트 규칙: **노란 차선 있으면 `yellow|road`**, 없으면 **`white|road`** (`resolve_course_lane_mask`, 흰∧노란 OR 금지).  
OUT: 항상 `white|road`.

| 키 | 동작 |
|----|------|
| `SPACE` | 재생 / 일시정지 (**시작은 PAUSE**) |
| `←` `→` / `,` `.` | 프레임 단위 이동 |
| `[` `]` | 재생 속도 ↓ / ↑ |
| `o` | 우측: 흰 이진 ↔ BEV 위 흰색 overlay |
| `r` | 처음으로 |
| `q` / ESC | 종료 |

캡처 후:

```bash
python3 scripts/vision_tune/tune_hsv.py --from-bag in
python3 scripts/vision_tune/tune_hsv.py --from-bag out
```

(기준 HSV는 `origin/board` 실차 튜닝값이 `lane_vision.yaml`에 로드됨)

---

## 제어 게인 튜너 — Pure Pursuit (시뮬·실차 공용)

스무딩: **`PP(δ) → EMA → rate-limit → out`** (창 하단 cyan / magenta / orange 바).  
시뮬 기하 기본 = **LIMO Gazebo** (`wheelbase=0.24`, `δ_max=30°`).

```bash
source /opt/ros/humble/setup.bash
# 주행 중 트랙바 (lane_control_node 끄고 sim-bringup만)
python3 scripts/vision_tune/tune_lane_control.py --drive
```

| 증상 | 조절 |
|------|------|
| 커브 **너무 일찍** 꺾임 | `lookahead_cm` ↑ |
| 연속 커브에서 **늦게/덜** 꺾여 이탈 | `rate_x100` ↑, `max_steer_%`=100, `cruise_%` ↓ |
| 조향이 약함/강함 (물리) | `wheelbase_cm` / `max_steer_deg` (실차는 실측 후) |
| 속도 | **`cruise_%`** |

| 트랙바 | 의미 |
|--------|------|
| `cruise_%` | 차 속도 (throttle) |
| `lookahead_cm` | PP look-ahead \(L_d\) |
| `wheelbase_cm` | 휠베이스 \(L\) (sim 24) |
| `max_steer_deg` | \(\delta_{\max}\) (sim 30) |
| `ema_%` / `rate_x100` / `max_steer_%` | 출력 스무딩·클립 |
| `slow_scale_%` / `half_w_cm` / `hold_%` / `color_0w1y` | 감속·반폭·홀드·색 |

키: `s` 저장 · `w` 창 재배치 · `space` pause · `q` 정지 종료

**D-Racer:** 시뮬 게인 그대로 쓰지 말 것. 실측 항목 → [vehicle-geometry.md §4.1](../../docs/vehicle-geometry.md).

---

## HSV 마스크 튜너 (Phase 1 · 시뮬·실차 분리)

흰/노란 차선 · 검정/빨강 차로 마스크를 Metric IPM BEV에서 맞춘다.  
**프로필 SSOT:** [`docs/hsv-profiles.md`](../../docs/hsv-profiles.md) · `config/lane_vision.yaml` → `hsv.profiles.{sim,real_car}`

| 프로필 | 환경 | 보드 적용 |
|--------|------|-----------|
| `sim` | Gazebo | ❌ (시뮬 전용) |
| `real_car` | D3-G bag 캡처 튜닝 | ✅ **`hsv.active: real_car`** |

```bash
# 프로필 전환 (런타임 평탄화 채널도 같이 갱신)
python3 scripts/vision_tune/hsv.py --apply-profile sim
python3 scripts/vision_tune/hsv.py --apply-profile real_car

# 실차 bag 캡처로 튜닝 (6채널: 흰/노란/검/빨/시안/시안2)
python3 scripts/vision_tune/tune_hsv.py --from-bag all
python3 scripts/vision_tune/tune_hsv.py --from-bag in
python3 scripts/vision_tune/tune_hsv.py --folder data/captures/from_bag/out
python3 scripts/vision_tune/tune_hsv.py --from-bag out_glare --channel black_cyan
python3 scripts/vision_tune/tune_hsv.py --from-bag black_road --channel black_road
python3 scripts/vision_tune/tune_hsv.py --from-bag in_yellow --channel yellow
```

기준값: `real_car` = bag `from_bag` 캡처. `s` 저장 · `d`=sim 시드 · `b`=board 레거시.

| `--from-bag` | 용도 / 추천 채널 |
|--------------|------------------|
| `all` | in+out+out_glare+black_road+in_yellow — **6채널 전체** |
| `in` / `out` | 흰·노란·검·빨 일반 |
| `out_glare` | **black_cyan** |
| `black_road` | **black_road** (커브/V 튜닝 프레임) |
| `in_yellow` | **yellow** / **black_cyan_2** (~870–955) |
| `both` | in+out 병합 |

**주행가능 영역 SSOT (2026-07-16 LOCKED trial #1):**  
`road_raw` = `compose_road_raw` (red→red only; cyan2 IN-only; cyan1 off if yellow visible)  
→ morph open **3** / close **13** / 1회 → ego (하단 밴드 질량).  
HSV retune: black 유지(H17 V15–140); **red S≥110**, **cyan S200–215 V190–238**.

| 창 | 역할 |
|----|------|
| `hsv_tune (origin \| BEV \| mask)` | **단일 창** — 좌: 원본 / 중: BEV+마스크 / 우: 이진 마스크 + 트랙바 |

| 키 | 동작 |
|----|------|
| `1`–`6` | white / yellow / black_road / red_road / **black_cyan** / **black_cyan_2** |
| `d` | 활성 채널 → **sim** (Gazebo / 원태 시드) |
| `b` | 활성 채널 → origin/board 1차 실차 (레거시) |
| 클릭 | ORIGIN/BEV 패널에서 해당 픽셀 HSV로 범위 **확장** |
| `s` | `config/lane_vision.yaml` → `profiles[active]` + 평탄화 저장 |
| `n` / `p` | 폴더 모드 다음/이전 |
| `q` / ESC | 종료 |

---

## OUT/IN 주행가능 영역 프리뷰 (blob corridor)

기본 인지 백엔드 = **`perception.backend: blob`**  
지금은 **차선 mid/리본 보정 없음** — `black|red` 도로 마스크 + open/close 노이즈 제거 + ego-near 최대 blob 1개.  
S·곡선·원형은 마스크 추종(`mask_pursuit`)에 맡김.

road = **검정|빨강**, 코리도 = **`road_clean` blob**.  
갈림만 `--fork` / 표지 게이트 (legacy).

레거시 폴리핏: `perception.backend: legacy` 또는  
`lane_detection.set_perception_backend('legacy')` / `tune_lane_detect.py`.

```bash
python3 scripts/vision_tune/preview_out_drivable.py --from-bag out --course out
python3 scripts/vision_tune/preview_out_drivable.py --from-bag out --start 11 --fork
python3 scripts/vision_tune/preview_out_drivable.py --from-bag in --course in
python3 scripts/vision_tune/preview_out_drivable.py --from-bag out --start 5 --once
```

| 키 | 합성 모드 |
|----|----------|
| `1` | `road` — black\|red raw |
| `2` | `between` — selected **blob** (**기본** 추종) |
| `3` | `road_in` — road ∩ blob |
| `4` | `union` — blob ∪ road |
| `f` | `enable_fork` 토글 (**기본 OFF** — 11–12만) |
| `0` | fork 갈래 필터 both → L → R |

코리도 생성: `black|red` morph denoise → ego-near largest blob. 차선 기하 보정 없음.

`n`/`p` · `SPACE` · `s` → `data/captures/out_drivable_preview/`  
오버레이: road=회색 · blob=초록 · 차선 HSV · **mid 중심선**=시안

런타임 경로: [`modules/perception/blob/`](../../src/inference/inference/modules/perception/blob/)  
레거시 격리: [`modules/perception/legacy/`](../../src/inference/inference/modules/perception/legacy/)

---

## 사다리꼴 ROI (참고 / 레거시)

시각 검증용. **런타임 SSOT 아님.** `tune_bev.py`와는 **다른 코드**(호모그래피 ROI).

```bash
python3 scripts/vision_tune/tune_bev_roi.py
python3 scripts/vision_tune/tune_bev_roi.py --folder data/captures/sim
```

| 트랙바 | 역할 |
|--------|------|
| `crop_top_%` | 상단 제외 |
| `bottom_half_%` | 아랫변 확장 (최대 1500) |
| `bev_w` / `bev_h` | 픽셀 해상도 (미터 아님) |
| `guide_half_px` | 차로 0.35 m 가이드 (기본 44) |

참고값: `crop=0.39`, `bottom_half=6.35`, `bev=500×370`. 종·횡 m/px는 **다름**.

---

## Gazebo 캘리브 매트

```bash
python3 scripts/prepare_bev_calib_mat.py
# 월드 반영: build-sim 또는 bringup 재실행
```

위치 `(2.5, -6.5)` — `src/dracer_sim/config/bev_calib_mat.yaml`.  
Metric IPM이면 종·횡이 이미 등방이라 검증용; 사다리꼴 쓸 때만 종방향 필수.

---

## 파일

| 파일 | 역할 |
|------|------|
| `tune_lane_detect.py` | **인지 모드 검증·튜너** (Gazebo 미기동, topic/image) |
| `tune_bev.py` | **기본 BEV 진입** → Metric IPM |
| `tune_metric_ipm.py` | Metric IPM UI·로직 (`tune_bev.py`가 호출) |
| `metric_ipm.py` | remapping · `(u,v)→(x,y) m` |
| `tune_hsv.py` | **HSV 마스크** (흰/노란/검/빨/**시안**) |
| `hsv.py` | HSV load/save/mask |
| `viz_raw_hsv_masks.py` | BEV · morph · **ego blob** (road=`black\|red\|cyan`) |
| `viz_cyan_ab.py` | 시안 전후 ego A/B 모자이크 (`out_glare`) |
| `score_in_fork_moment.py` | IN keep/exit 직전 게이트 오프라인 (`fork.moment` thin CLI) |
| `score_out_fork_moment.py` | OUT 갈림 직전 게이트 오프라인 (`fork.moment` thin CLI) |
| `score_out_ego_fork_shape.py` | OUT **ego_blob Y-stretch** 단독 스코어 |
| `score_out_fork_capture.py` | OUT **tip+stretch 통합** (`fork.capture`, bag/BEV/folder) |
| → 코드·데이터 SSOT | [`docs/fork-moment-detection.md`](../../docs/fork-moment-detection.md) · [`docs/out-ego-fork-shape.md`](../../docs/out-ego-fork-shape.md) · [`perception/fork/moment.py`](../../src/inference/inference/modules/perception/fork/moment.py) · [`ego_shape.py`](../../src/inference/inference/modules/perception/fork/ego_shape.py) · [`capture.py`](../../src/inference/inference/modules/perception/fork/capture.py) |
| `tune_lane_control.py` | **Pure Pursuit** 게인 튜너 (레거시 경로) |
| `window_layout.py` | OpenCV 창 화면 안 배치 (`w`) |
| `tune_bev_roi.py` | 사다리꼴만 (참고) |
| `bev_roi.py` | 사다리꼴 기하 |
| `capture_camera.py` | 핫키 캡처 (라이브 토픽) |
| `capture_from_bag.py` | **IN/OUT bag 재생 + 핫키 캡처** |
| `play_bag_drivable.py` | **bag→BEV 저장 / BEV\|6_ego_blob 재생** (`extract_five` SSOT) |
| `preview_out_drivable.py` | OUT/IN: road + 피팅 레일 between 프리뷰 |
| `out_drivable.py` | road / fill_between_fitted_rails 헬퍼 |
| `../../config/lane_vision.yaml` | `metric_ipm:` + `hsv:` + `detect_tune:` SSOT |
| `../../config/lane_control.yaml` | planner 게인 |
