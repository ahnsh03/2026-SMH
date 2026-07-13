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
./scripts/dev_container.sh sim-bringup     # 터미널1 — Gazebo만
docker exec -it 2026-smh-sim bash          # 터미널2
source /opt/ros/humble/setup.bash
source install/setup.bash
```

**인지 검증은 Gazebo를 다시 띄우지 마세요.** `sim_auto_driving` / `LANE_VISUALIZE=… launch`는 bringup이 이미 있으면 Gazebo가 하나 더 뜹니다. 카메라 토픽만 구독하는 튜너를 쓰세요.

---

## 인지 모드 튜너 (주 검증 경로 · Gazebo-free)

**단계형 튜닝 (권장):**

| Phase | 키 | 목표 |
|-------|----|------|
| **A** | `2` yellow → `3` dash | 노란 점선이 이어져 **4가닥**이 명확히 보이게 |
| **B** | `4` / `5` | (선택) 한쪽 고어 점선만 남는지 |
| **C** | `6` / `7` / `8` | A 통과 후 L/R 차로 쌍·중앙 분리 |

기본 시작 모드 = **`dash` (Phase A)**.

```bash
python3 scripts/vision_tune/tune_lane_detect.py              # dash부터
python3 scripts/vision_tune/tune_lane_detect.py --mode fork
python3 scripts/vision_tune/tune_lane_detect.py --folder data/captures/lane_tune_logs
```

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
| `ipm_tune_origin` | 원본 + crop 선 (`x_max`) |
| `ipm_tune_bev` | Metric BEV (등방 m/px) |
| `ipm_tune_trapezoid` | `--compare` 일 때만 (참고) |
| `ipm_tune_controls` | 트랙바 |

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

## HSV 마스크 튜너 (Phase 1 · 시뮬·실차 공용)

흰/노란 차선 · 검정/빨강 차로 마스크를 Metric IPM BEV에서 맞춘다.  
**툴·yaml 저장 = 승현**, 대회용 **최종 정밀값 = 원태**와 맞춤. 시드는 원태 브랜치 기본값.

```bash
python3 scripts/vision_tune/tune_hsv.py
python3 scripts/vision_tune/tune_hsv.py --channel white
python3 scripts/vision_tune/tune_hsv.py --folder data/captures/sim
```

| 창 | 역할 |
|----|------|
| `hsv_tune_origin` | 원본 + crop |
| `hsv_tune_bev` | BEV + 마스크 오버레이 (클릭 샘플) |
| `hsv_tune_mask` | 이진 마스크 |
| `hsv_tune_controls` | channel + H/S/V min/max |

| 키 | 동작 |
|----|------|
| `1`–`4` | white / yellow / black_road / red_road |
| 클릭 | 해당 픽셀 HSV로 범위 **확장** |
| `d` | 활성 채널을 원태 시드 기본값으로 |
| `s` | `config/lane_vision.yaml` → `hsv:` 저장 |
| `n` / `p` | 폴더 모드 다음/이전 |
| `q` / ESC | 종료 |

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
| `tune_hsv.py` | **HSV 마스크** (흰/노란/검/빨) |
| `hsv.py` | HSV load/save/mask |
| `tune_lane_control.py` | **Pure Pursuit** 게인 튜너 (레거시 경로) |
| `window_layout.py` | OpenCV 창 화면 안 배치 (`w`) |
| `tune_bev_roi.py` | 사다리꼴만 (참고) |
| `bev_roi.py` | 사다리꼴 기하 |
| `capture_camera.py` | 핫키 캡처 |
| `../../config/lane_vision.yaml` | `metric_ipm:` + `hsv:` + `detect_tune:` SSOT |
| `../../config/lane_control.yaml` | planner 게인 |
