# 보드 지연 · 연산 과부하 · 시뮬↔실차 괴리 (2026-07-11 실측)

D3-G 보드(aarch64 4코어 / 7GB / ROS 2 Humble)에서 노드 수신 지연을 계측하고,
연산 과부하 지점과 시뮬레이션 전제가 실차에서 깨지는 지점을 정리한다.
**측정값은 전부 이 보드에서 실제로 잰 것이고, 추정치는 추정이라고 표시했다.**

---

## 1. 측정된 현실

| 조건 | `/camera/image/compressed` | 지연 | `/perception/lane` |
|---|---|---|---|
| camera_node 단독 | 29.93 Hz (std 2 ms) | 9 ms | — |
| + inference_node | 29.94 Hz | 13 ms | **2.27 Hz** |
| + monitor_node | 29.88 Hz | 17 ms (**최대 214 ms**) | 2.11 Hz |
| 위 + 수정 적용 후 | 29.60 Hz | 15 ms (최대 31 ms) | 2.25 Hz |

**카메라는 정상이다.** 문제는 인식 파이프라인이 프레임당 ~455 ms를 써서
30 Hz 카메라를 2.2 Hz로 소비하는 것이다. 즉 **제어가 보는 차선 정보는 항상
0.45초 묵어 있다.** 이것이 체감 지연의 정체다.

프레임당 비용 (320×180 실제 프레임, 유휴 CPU):

| 단계 | 비용 | 비중 |
|---|---|---|
| `lane_detection.detect` | **255~390 ms** | 74% |
| `traffic_sign.detect` (ONNX YOLO 416×416) | 130 ms | 25% |
| `aruco_detection.detect` | 6 ms | 1% |

`lane_detection`의 비용은 OpenCV가 아니라 **파이썬 루프**다. 프로파일 결과
`connect_dashed_components` → `track_boundary_path`가 207 ms를 쓰고, 그 안에서
`enumerate_boundary_candidates`가 프레임당 529회, `append_if_valid`가 2,622회 호출된다.

### 구조적 배경

모든 노드가 `rclpy.spin()` = **SingleThreadedExecutor**이고, 인식 파이프라인 전체가
`image_callback` 안에서 **동기 실행**된다 (`inference_node.py:142` → `pipeline.py:807-811`).
따라서 10 Hz로 설정된 `publish_control` 하트비트 타이머는 **콜백을 선점하지 못한다.**
보드에서 "10 Hz 제어 하트비트"는 존재하지 않는다 — 455 ms 콜백이 끝난 직후 몰아서 발화한다.

---

## 2. 이미 적용한 수정 (브랜치 `perf/wonjung-latency-fixes`)

| 항목 | 내용 |
|---|---|
| `lane_detection.py:65` | `VISUALIZE_MODE` `"on"` → `"off"`. headless 보드에서 `cv2.imshow` GTK 에러로 **inference_node가 즉사**했다. 이는 시각화 경로(창 13개 + `waitKey`)가 프레임마다 ROS 콜백 스레드에서 실제 실행됐다는 증거다. 필요 시 `LANE_VISUALIZE=on`. |
| `vehicle_config.yaml:28` | `OPENCV_DEBUG_MODE` `true` → `false`. `monitor_node.get_yaml_or_param_bool_multi`가 YAML을 ROS 파라미터보다 우선하므로 런치의 `debug_image=False`가 **조용히 무시**되어, 모니터가 실행되지도 않는 `/opencv/image/{blur,edge,grayscale}` 3개를 계속 구독하고 있었다. |
| 런치 파일 | `camera_node debug_log=False`. 초당 30줄 로깅 제거. |
| `lane_vision.yaml` | `meters_per_pixel`이 유일한 BEV 노브임을 명시. |
| `~/.bashrc` | `export ROS_DOMAIN_ID=4` 3중 중복 → 1회. (도메인 4는 주최측 배정값이므로 유지) |

결과: 카메라 지연 최대치 **214 ms → 31 ms**, CPU 85% → 80%.
`/perception/lane`은 2.2 Hz 그대로 — 아래 항목들을 손대야 움직인다.

---

## 3. 연산 과부하 — 아직 남은 것

### 3.1 `connect_dashed_components`가 프레임마다 **두 번** 돈다 — 최대어

`lane_detection.py:3520-3523`

```python
white_dash_points_bev = extract_dash_point_mask(white_cut_bev)
white_dash_connected_bev = connect_dashed_components(   # <- 게이팅 없음
    white_cut_bev,
    road_clean,
)
```

바로 위 yellow 버전은 게이팅돼 있다 (`:3503-3507`):

```python
yellow_dash_points_bev = (
    extract_dash_point_mask(yellow_boundary_raw_bev)
    if window_enabled("yellow_dash_points")
    else None
)
```

`white_dash_connected_bev`의 소비처는 `LaneDebugFrame`(`:3802`) → `make_dash_preview`(`:4801`)
뿐이고, 런타임 경로인 `detect()`(`:3422`)는 디버그 프레임을 **버린다**.
`white_cut_bev`는 실선 흰 마킹 + 스펙클이라 노란 점선 마스크보다 연결 성분이 많아
**둘 중 더 비싼 쪽일 가능성이 높다.**

**yellow와 똑같이 `window_enabled()`로 감싸면 된다. 한 줄이고, 런타임 동작은 바뀌지 않는다.**
추정 절감 50~100+ ms. 이것 하나로 lane_detection이 거의 반토막 날 수 있다.
**최우선 검증 대상.**

### 3.2 구독자 0인 토픽에 매 프레임 대용량 메시지를 만든다

`/perception/lane`의 유일한 소비자 `lane_control_node`는 **어떤 런치 파일에도 없다** (구독자 0).
그럼에도 `inference_node.publish_lane_detections`가 매 프레임:

- `_to_point32_list`(`inference_node.py:38`)로 rclpy `Point32` 객체를 **1,000~2,500개** 생성
  (경계당 최대 `BEV_HEIGHT`=321점 × 최대 4개 + 중심선 + 갈래). rosidl 생성 메시지의
  `__init__`은 필드마다 `assert isinstance`를 돈다. 추정 10~30 ms.
- 386×321 mono8 `drivable_area` **124 KB**를 3번 복사해 메시지에 싣는다
  (`road_clean.copy()` → `np.ascontiguousarray` → `tobytes()`).

**`if self.lane_pub.get_subscription_count() == 0: return` 한 줄.** 추정 절감 10~30 ms.

### 3.3 IPM remap 테이블을 매 프레임 재생성 (시각화용인데 시각화는 꺼져 있다)

`lane_detection.py:3690`

```python
bev_color = warp_metric_ipm(frame, METRIC_IPM_PARAMS)
```

`metric_ipm.py:305`의 `warp_metric_ipm`은 **캐시가 없어** 매 호출마다 `build_ipm_maps`로
386×321 `meshgrid` + sin/cos + 유효성 마스크를 다시 만든다 (프레임당 ~5 MB 할당).
정작 `lane_detection`은 같은 맵을 `_ensure_ipm_maps`(`:285-300`)에 **이미 캐시해 두고** 있는데
이 호출만 캐시를 우회한다. `bev_color`는 `if VISUALIZE:`(꺼짐)와 `LaneDebugFrame`에만 쓰인다.

추정 절감 3~10 ms. **단, `LaneDebugFrame.bev`는 `tune_lane_detect.py` 등 튜너가 쓰므로
`detect()` 런타임 경로에서만 건너뛰어야 한다** — 튜닝 워크플로를 깨지 않도록 주의.

### 3.4 라벨 하나마다 전체 이미지를 스캔하는 O(라벨×픽셀) 루프 3곳

`lane_detection.py:762` (`remove_far_specks`), `:2847` (`extract_dash_point_mask`),
`:2871` (`connect_dashed_components`) 모두 같은 패턴:

```python
for label in range(1, count):
    ...
    cleaned[labels == label] = 0        # 라벨마다 124k 픽셀 전체 비교
```

124k 픽셀 × 60개 성분 ≈ 740만 회 비교. 라벨 LUT(`lut[labels]`) 또는
`CC_STAT_*` 바운딩 박스 슬라이싱으로 바꾸면 된다. 추정 절감 10~25 ms. 기계적 수정.

### 3.5 35×35 커널 MORPH_CLOSE 5회

`lane_detection.py:4226`, `LANE_CUT_CLOSE_ANGLES_DEG = (50, 70, 90, 110, 130)` (`:4090`),
커널 길이 `0.14 / 0.004` = **35 px**. 124k 픽셀 × 35×35 탭 × 5각도.
추정 15~40 ms. **이건 실제로 쓰이는 연산이므로(갈림길 셀 커터) 각도를 3개(70/90/110)로
줄이는 건 동작 변화다 — bag으로 검증 후 결정.**

주목: `meters_per_pixel`을 키우면 이 커널의 px 길이도 같이 줄어 **BEV 노브가 여기도 함께 싸게 만든다.**

### 3.6 쓰이지 않는데 매 프레임 도는 검출기들

- **`detect_signal`** (신호등): `main_planner.yaml:91-92`가 `require_green_to_start: false`,
  `stop_on_red: false`라 소비처가 없는데 `pipeline.py:810`이 무조건 호출한다.
  전체 HSV 변환 + `inRange` 3회 + 5×5 morphology 4회 + `findContours` 2회. 추정 2~5 ms.
- **`detect_turn_rule_based`** (`detector.py:250-258`): ONNX가 아무것도 못 찾으면
  **추가로** 룰 기반 검출을 돌린다. 표지판은 대부분의 프레임에 없으므로 사실상
  **거의 모든 프레임에서 130 ms ONNX + 룰 기반이 둘 다 돈다.** 같은 프레임의
  **세 번째** 전체 HSV 변환이며, 컨투어마다 전체 크기 배열과 구조 요소를 새로 만든다.
  추정 3~10 ms. **가중치가 실제로 없을 때(`_session is None`)만 폴백하도록 게이팅해야 한다.**

### 3.7 우선순위

| # | 항목 | 추정 절감 | 위험 |
|---|---|---|---|
| 1 | `white_dash_*` 게이팅 (`:3520`) | **50~100+ ms** | 없음 (디버그 전용) |
| 2 | `/perception/lane` 구독자 가드 | 10~30 ms | 없음 |
| 3 | 라벨 스캔 → LUT/ROI | 10~25 ms | 낮음 |
| 4 | `bev_color` 런타임 경로에서 제거 | 3~10 ms | 없음 (튜너 경로 보존 필요) |
| 5 | 룰 기반 폴백 게이팅 | 3~10 ms | 없음 |
| 6 | `detect_signal` 스킵 | 2~5 ms | 없음 |
| 7 | `LANE_CUT_CLOSE_ANGLES_DEG` 3개로 | 10~25 ms | **동작 변화 — bag 검증 필수** |

1~6은 동작 변화 없는 죽은 코드 제거이며 합치면 **80~180 ms** 회수 가능 —
알려진 두 핫루프를 손대기 전에 **인식 속도 2배**가 가능하다는 뜻이다.

메모리 누수는 **없다.** 프레임 경로에 무한 증가하는 리스트/딕트/데크가 없음을 확인했다.

---

## 4. BEV 해상도 노브

`config/lane_vision.yaml`의 `metric_ipm.meters_per_pixel`이 **유일한 노브**다.
바로 아래 `bev_width` / `bev_height` / `guide_half_width_px`는 `metric_ipm.py:116-128`에서
이 값으로부터 **계산되며 YAML 값은 읽지 않는다.** 손대지 말 것.

보드 실측 (320×180 실제 프레임):

| `meters_per_pixel` | BEV | 픽셀 수 | `lane_detection.detect` |
|---|---|---|---|
| **0.004 (현재)** | 386×321 | 123,906 | **255 ms** |
| 0.005 | 309×257 | 79,413 | 160 ms (−37%) |
| 0.006 | 258×214 | 55,212 | 108 ms (−58%) |
| 0.008 | 193×161 | 31,073 | 72 ms (−72%) |

비용은 BEV 픽셀 수에 거의 선형이다. **CPU 이득만 검증됐고 검출 품질은 미검증이다.**

검증 도구: `scripts/vision_tune/bench_bev_resolution.py` — 녹화된 bag으로
해상도별 **비용과 검출률을 함께** 잰다. bag의 `.db3`를 직접 읽어 ROS 그래프 없이
보드에서 headless로 돈다.

```bash
python3 scripts/vision_tune/bench_bev_resolution.py <bag_dir> --frames 40
```

`both` 열(좌우 경계를 모두 찾은 프레임 비율)이 0.004와 같게 유지되는
**가장 거친 값**을 고르면 된다.

> ⚠️ **2026-07-08에 녹화한 bag 4개는 프레임이 320×240이라 쓸 수 없다.**
> 현 파이프라인은 320×180 기준(`crop_top_ratio: 0.39`)이므로 IPM 기하가 맞지 않아
> 어떤 해상도에서도 검출률이 0%로 나온다. **튜닝용 bag은 새로 녹화해야 한다.**

---

## 5. 시뮬 → 실차 괴리 (이쪽이 더 위험하다)

핵심 산수: **시뮬 인식 주기 33 ms(30 Hz) vs 실차 455 ms(2.2 Hz) = 13.6배.**
프레임 수나 고정 dt에 기대는 모든 로직이 이 비율만큼 어긋난다.

### 5.1 🔴 안전 워치독이 실제 프레임 주기 **바로 위**에 있다

`config/main_planner.yaml:95` → `command_watchdog_sec: 0.5`
실측 인식 주기 **0.455초**. 여유 **45 ms (10%)**.

`inference_node.py:240-252`:

```python
stale = (... now_sec - self._last_frame_time_sec > self.planner.config.command_watchdog_sec)
if stale:
    self.planner.neutralize_steering()
    command = pipeline.ControlCommand(steering=0.0, throttle=0.0)
```

GC 일시정지, 느린 ONNX 추론, `monitor_node`와의 CPU 경합 — **한 프레임만 느려도**
0.5초를 넘긴다. 그러면 **주행 중 바퀴가 갑자기 직진으로 튀고 스로틀이 끊긴다.**
게다가 `neutralize_steering()`은 레이트 리미터 상태를 0으로 밀어버려
다음 정상 프레임에서 조향을 처음부터 다시 램프업해야 한다.

시뮬(33 ms)에서는 **이 분기에 도달할 수 없다.** 실차에서는 10% 여유로 상시 대기 중이다.

**조치: `command_watchdog_sec` ≥ 1.0~1.5로 올리고, 한 번 늦었다고 조향 적분기를 0으로 밀지 말 것.**

### 5.2 🔴 `dt`가 0.25초로 클립되어 시간 기반 제한이 45% 깎인다

`pipeline.py:325-332`:

```python
return float(np.clip(dt, 0.001, 0.25))
```

실차 dt = 0.455초 → **항상 0.25로 클립.** 시뮬 dt = 0.033초 → 클립 안 됨.

따라서 `_pure_pursuit`(`:758-759`)의 `steering_rate_limit_per_sec: 6.0`은
실제로는 `6.0 × 0.25 / 0.455` = **초당 3.3**, 설정값의 **55%**로 동작한다.
`path_lost_steering_return_rate_per_sec`도 같은 비율로 깎인다.

이는 `main_planner.yaml:35-36`의 주석
`# Time-based limits: behavior no longer changes with camera FPS.`를 **정면으로 반증한다.**
프레임 주기가 0.25초를 넘는 순간부터 FPS에 다시 종속된다 — 실차가 정확히 그 상태다.

**조치: 클립 상한을 0.6초 이상으로.**

### 5.3 🔴 모든 미션 디바운스가 **프레임 수**라서 실차에서 13.6배 길어진다

`RisingEventCounter`(`pipeline.py:231-261`)와 카운터들에 **시간 항이 전혀 없다.**

| 설정 | 값 | 시뮬(30 Hz) | **실차(2.2 Hz)** |
|---|---|---|---|
| `sign_confirm_frames` | 3 | 0.10초 | **1.36초** |
| `branch_on_frames` | 3 | 0.10초 | **1.36초** |
| `branch_off_frames` | 8 | 0.27초 | **3.64초** |
| `crossing_off_frames` | 5 | 0.17초 | **2.27초** |
| `path_lost_stop_frames` | 10 | 0.33초 | **4.55초** |
| `fork_exit_off_frames` | 8 | 0.27초 | **3.64초** |

구체적 결과:

- **표지판이 영영 안 걸릴 수 있다.** `_update_desired_turn`(`:369-383`)은 **같은** 표지판이
  3프레임 연속 = **1.36초 끊김 없는 YOLO 검출**을 요구하고, UNKNOWN 한 프레임이면
  카운터가 0으로 리셋된다. 3프레임 중 1개만 놓쳐도 확정되지 않고,
  `_lock_fork_selection`이 `default_out_branch_rank: 0`으로 떨어져 **표지판과 무관하게 항상 좌회전**한다.
- **갈림길 인지가 1.36초 늦는다.** 차가 이미 갈림길에 진입한 뒤에 `FORK_TURN`으로 들어간다.
- **`path_lost_stop_frames: 10` = 4.55초.** 시뮬에서는 0.33초짜리 글리치 필터지만,
  실차에서는 **경로를 잃고도 4.5초를 순항 속도로 계속 달린다.**

**조치: 전부 초 단위로 바꾸고 dt로 나눌 것.** `aruco/stop_logic.py`는 이미 그렇게 하고 있다.

### 5.4 🔴 실차 실측 기하가 측정만 되고 **쓰이지 않는다**

`config/main_planner.yaml:32-33, 63`:

```yaml
  wheelbase_m: 0.24              # 시뮬 LIMO 값
  max_steer_angle_rad: 0.5236    # 30°
  perception_to_rear_axle_x_m: 0.265   # = 0.24(시뮬 축거) + 0.025
```

그런데 `config/lane_control.yaml:9-17`에 7/12에 **실측한 값이 주석으로만** 있다:

```
#   wheelbase_m: 0.175
#   max_steer_angle_rad: 0.3054   # δ_max = 17.5°
#   Do NOT copy sim L/δ_max blindly.
```

**어떤 코드도 이 주석을 읽지 않는다.** 런치에도 오버라이드가 없다.

결과 (`pipeline.py:744-746`):

```python
steer_angle = math.atan(self.config.wheelbase_m * curvature)
pp_steering = -steer_angle / self.config.max_steer_angle_rad
```

(0.24, 0.5236) 대신 (0.175, 0.3054)를 써야 하므로 정규화 조향 명령이
**있어야 할 값의 약 0.8배** → **상시 20% 언더스티어**, 그것도 하필 급코너와
갈림길 탈출에서 가장 크게. CTE·헤딩 보정도 같은 분모로 나뉘어 물리 각도 기준
1.7배 작다. `perception_to_rear_axle_x_m: 0.265`도 실차 기준 ≈0.20이어야 하므로
**모든 인식 점이 6.5 cm 앞으로 밀린 채** 플래너에 들어간다.

시뮬 기본값은 `pipeline.py:63-64`, `lane_planner.py:22-24`, `control_mapping.py:22`에도
하드코딩돼 있다.

**조치: `main_planner.yaml`에 실측값을 넣을 것. 가장 값싸고 효과 큰 수정.**

### 5.5 🟠 C920e 자동 노출·화이트밸런스가 그대로인데 HSV는 그림자 없는 렌더로 튜닝됐다

두 저장소 어디에도 **노출·화이트밸런스·게인 제어가 없다.**
`camera_node.py:146-152`의 GStreamer 파이프라인에 `extra-controls`가 없어
C920e는 **완전 자동 노출 + 자동 AWB**로 돌아간다. 차가 조명 쪽으로 돌 때마다
영상이 계속 재정규화된다.

반면 튜닝 기준이었던 `src/dracer_sim/worlds/track_cw.world:11-17`은
단일 방향광 + **`<cast_shadows>false</cast_shadows>`** + 균일 0.9 확산광 —
그림자도 노출 변화도 없는 세계다.

그 위에서 잡힌 임계값들:

- `white`: `s_max: 29` — 채도 상한 29/255는 거의 완벽한 무채색을 요구한다.
  실제 AWB가 흰 선에 색조를 입히면 **그대로 뚫고 나간다.**
- `red_road`: `s_min:125, s_max:192, v_min:161, v_max:229` — S와 V **양쪽 다 상한이 있는 좁은 상자.**
  고정 노출에서만 성립하며, 노출이 조금만 움직여도 **빨간 도로 마스크가 통째로 사라진다.**
- `yellow`: **`h_min: 0`** — Hue 0은 **순수 빨강**이다. 트랙에 빨간 도로 영역이 실재하므로
  (그래서 `red_road` 임계값이 따로 있다) 실제 카메라에서 **노란 마스크가 빨간 도로에 얼라이어싱**된다.
  노랑은 로터리 진입을 게이팅하므로(`entry_on_yellow: true`) → **빨간 도로 구간에서 오진입.**

**조치: `v4l2src`의 `extra-controls`로 노출·WB를 고정한 뒤 실차 영상으로 HSV 재튜닝.
`yellow.h_min`을 0 → 15 부근으로. 노출이 고정되지 않으면 이 파일의 어떤 HSV 값도 재현되지 않는다.**

### 5.6 🟠 카메라 소스 캡스에 해상도가 없다 — 320×240 bag의 원인

Gazebo 카메라 모델은 실차와 **일치한다**
(`limo_dracer_sim.xacro`: 높이 0.13 m, 피치 10°, HFOV 70.42°, 640×360 16:9 → 320×180). 이건 잘 돼 있다.

문제는 실차 쪽이다. `camera_node.py:148`의 소스 캡스가 `image/jpeg,framerate=30/1` —
**해상도 제약이 없다.** `v4l2src`가 C920e가 주는 아무 MJPG 모드나 협상한 뒤
`videoscale`이 320×180으로 **강제 스케일**한다. 협상된 모드가 4:3이면
**종횡비를 무시한 찌그러뜨리기**가 일어나 fy ≠ fx가 된다.

이건 IPM에 치명적이다. `metric_ipm.py:199-212`가 정사각 픽셀을 가정한다:

```python
fx = img_w / (2.0 * np.tan(np.deg2rad(p.hfov_deg) / 2.0))
fy = fx
```

4:3 → 16:9 찌그러뜨리기는 행↔전방거리 매핑을 조용히 망가뜨리고,
`crop_top_ratio: 0.39`와 `METERS_PER_PIXEL`로 환산되는 모든 미터 임계값이 같이 틀어진다.
**7/8 bag이 320×240으로 찍힌 것이 바로 이 증상이다.**

**조치: 소스 캡스를 `image/jpeg,width=1920,height=1080,framerate=30/1`로 고정하고
`v4l2-ctl --list-formats-ext`로 30 fps 지원을 확인할 것.**

또한 렌즈 왜곡 모델이 **어디에도 없다.** Gazebo 카메라에 `<distortion>`이 없고
`metric_ipm.py`도 순수 핀홀이라 undistort 단계가 없다. 실제 C920e는 70° HFOV에서
배럴 왜곡이 있고, 그건 화면 가장자리에서 직선을 가장 크게 휘게 만든다 —
하필 갈래 분리 로직(`min_branch_separation_m: 0.15`)이 일하는 곳이다.

### 5.7 🟡 시뮬은 완벽한 속도 서보, 실차는 개루프 ESC 펄스

시뮬(`control_mapping.py:35`)은 `linear = clip(throttle) * max_linear_speed`이고
Gazebo ackermann 플러그인이 이를 **완벽한 제어기가 붙은 속도 지령**으로 처리한다.
`cruise_throttle: 0.17` → 정확히 **0.204 m/s**, 즉시, 모터 데드밴드도 관성도 슬립도 배터리 새그도 없이.

실차(`d3racer.py:61-73`, `EscCalib(neutral_us=1500, fwd_us=2000)`)는
throttle 0.17 → **1585 µs**, 중립보다 **겨우 85 µs 위**다. **대부분의 ESC 데드밴드 안이다.**
차가 **아예 안 움직일 수도 있다.** 움직인다 해도 속도는 미지수이고 배터리 의존적이며
0.204 m/s와 아무 관계가 없다.

모든 PP 게인과 `lookahead_m: 0.80` / `curve_lookahead_m: 0.45` 스케줄이
그 "정확한 속도" 플랜트 위에서 튜닝됐다.
조향 역시 `ServoCalib(center_us=1500, span_us=500)`의 ±500 µs가 **실제 타이어 각으로
얼마인지 이 저장소에서 한 번도 캘리브레이션된 적이 없다.**

**조치: ESC 데드밴드를 벤치에서 측정해 차가 실제로 움직이기 시작하는 최소 throttle을 찾고,
`cruise_throttle: 0.17`이 그 위인지 확인할 것.**

### 5.8 🟡 그 밖

- **ArUco 정지 로직만 시간 기반**(`stop_logic.py:18-19`, `time.monotonic()`)인데,
  시뮬은 `use_sim_time: true`라 플래너는 `/clock`을, 이 모듈은 **벽시계**를 본다.
  Gazebo RTF ≠ 1이면 서로 다른 시간축이라 시뮬에서 본 것이 실차와 다르다.
  실차에서는 `_ENTER_STOP_SECONDS = 0.15`가 0.455초 주기 때문에 사실상 2프레임 = **0.91초**가 된다.
- **`lane_control_node`는 어떤 런치에도 없지만 entry point로 등록돼 있다**(`setup.py:30`).
  실행되면 본 플래너보다 훨씬 나쁘다 — `steer_rate_limit: 0.15`가 **프레임당** 적용되어
  2.2 Hz에서는 초당 0.33만 조향할 수 있다(30 Hz에서는 4.5). **코너를 물리적으로 돌 수 없다.**
  대회 중 실수로 켜지지 않게 entry point를 막는 것을 권한다.
- **신호등 검출에 시간 필터가 전혀 없다**(`color_detector.py:88-91`, 단일 프레임 판정).
  지금은 `require_green_to_start: false` / `stop_on_red: false`라 무해하지만,
  대회에서 켜면 오검출 한 프레임에 최소 0.45초 정지한다. 게다가 red 범위가
  **빨간 도로 표면**에도 반응한다(5.5와 같은 얼라이어싱).
- **`STEER_TRIM: 0.1`이 플래너 클립 뒤에 더해진다**(`pipeline.py:804`).
  실차 명령 범위가 **[-0.90, +1.00]**이 되어 좌조향 권한이 10% 적다. 시뮬은 트림 0이라 대칭.
- **`vehicle_config.yaml:10-11`의 `ROI_TOP`/`ROI_LEFT`는 아무도 읽지 않는다.**
  프레임이 이미 크롭된 줄 오해하게 만든다.

---

## 6. 대회장 네트워크

주행 노드는 전부 보드 안에서 도므로 **DDS가 무선을 탈 이유가 없다.**
그런데 현재 `ROS_LOCALHOST_ONLY=0`이고 카메라 토픽이 **RELIABLE**이라,
팀원 노트북이 같은 도메인에서 `rqt`나 `ros2 topic echo`로 붙는 순간 프레임이 WiFi로 나가고,
**패킷이 깨질 때마다 재전송이 걸려 퍼블리셔가 밀린다.** 대회장의 혼잡한 전파가
그대로 제어 루프로 새어 들어온다.

**주행 중** — 보드에서 `export ROS_LOCALHOST_ONLY=1`.
DDS가 무선을 아예 타지 않아 전파 상태와 무관해진다.
모니터 대시보드는 Flask HTTP(포트 5000)라 이 설정과 무관하게 계속 접속된다. 관제는 이걸로.

**피트 디버깅 시에만** — `ROS_LOCALHOST_ONLY=0`으로 풀고 **전용 AP(5 GHz, 한산한 채널 고정,
클라이언트 격리 끄기)** 로 붙는다.

**폰 핫스팟 테더링은 쓰지 말 것.** 클라이언트 격리로 기기 간 통신이 막히고
**DDS 디스커버리가 쓰는 UDP 멀티캐스트를 대부분 드롭**한다. 2.4 GHz인 경우가 많고
폰이 절전으로 스로틀링한다. ROS 2에 최악의 조합이다.

`ROS_DOMAIN_ID=4`는 주최측 배정값이므로 유지한다 (`~/.bashrc`, 중복 제거 완료).

---

## 7. 대회 전 권장 순서

1. **`command_watchdog_sec` 0.5 → 1.5** — 한 줄. 주행 중 무작위 스로틀 컷 방지. (5.1)
2. **`main_planner.yaml`에 실측 기하 반영** — `wheelbase_m: 0.175`,
   `max_steer_angle_rad: 0.3054`, `perception_to_rear_axle_x_m ≈ 0.20`. (5.4)
3. **`white_dash_*` 게이팅** — 한 줄, 동작 변화 없음, lane_detection 거의 반토막. (3.1)
4. **프레임 카운트 → 초 단위 변환** — 표지판이 안 걸리고 갈림길이 늦는 근본 원인. (5.3)
5. **`_step_dt` 클립 0.25 → 0.6+** (5.2)
6. **노출·WB 고정 후 실차 영상으로 HSV 재튜닝**, `yellow.h_min` 0 → 15. (5.5)
7. **카메라 소스 캡스 1920×1080 고정** — 4:3 찌그러짐 재발 방지. (5.6)
8. **ESC 데드밴드 측정** — `cruise_throttle: 0.17`로 차가 실제로 움직이는지. (5.7)
9. **BEV 해상도 확정** — 새 bag 녹화 후 `bench_bev_resolution.py`로 결정. (4)

1·3·5는 동작 변화가 거의 없는 값싼 수정이고, 2·4는 실차 주행 품질을 좌우한다.
BEV 해상도를 낮추면 인식 주기가 짧아져 **5.1·5.2·5.3의 여유가 동시에 넓어진다.**
