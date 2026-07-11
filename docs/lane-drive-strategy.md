# 기본 주행 루프 — 개발 구조·전략

> 작성: 2026-07-11 (안승현)  
> 브랜치: `feature/seunghyun-lane-drive`  
> 목적: 시뮬·실차 공통의 **차선 인식 + 기본 주행**을 어떤 구조로 만들지 문서화. 구현 전 설계 SSOT.

관련: [roles.md](./roles.md) · [meetings/2026-07-10.md](./meetings/2026-07-10.md) · [hardware-camera.md](./hardware-camera.md) · [collaboration.md](./collaboration.md)

---

## 1. 목표와 범위

### 목표

Gazebo / D3-G에서 **차선 기반 기본 주행 루프**를 안정적으로 돌린다.

- 노드 실행 → 출발, 종료 → 정지 (시뮬에서는 신호등 게이트 생략 가능)
- 조향은 **부드럽고** 추종이 안정적일 것 (기존 ROS1 LIMO 체감 수준)
- 갈림길·회전교차로에서는 **두 갈래 경로를 각각 분리 감지**하고, 모드에 따라 하나만 선택
- ROI / IPM / HSV 파라미터는 **트랙바 툴로 확정 → YAML 저장 → 런타임·실차 재사용**

### 비목표 (이 문서 범위 밖)

- 신호등 실차 검증 (장원정 / 실차)
- ArUco 로직 재구현 (완료)
- 회전교차로 Pure Pursuit 전부 (양서준·박성준과 인터페이스만 맞춤)
- 학습 기반 트랙 검출 (wego CNN) — 1차 미채택

---

## 2. 트랙·센서 전제

| 항목 | 값 |
|------|-----|
| 카메라 | C920e, 팀 출력 **320×180** JPEG `/camera/image/compressed` |
| 제어 | `Control(steering, throttle ∈ [-1,1])` → (시뮬) `control_bridge` → `/cmd_vel` |
| 일반 구간 | 배경 **파랑**, 차로 **검정**, 차선 **흰색** |
| 회전교차로 | 차선 **노란색** |
| 동적 장애물 | 차로 **빨간색** (신호등 빨강과 **색·위치·컨텍스트 분리**) |

조향 부호: D-Racer `+steering = right`. LIMO `Twist.angular.z`와 의미·스케일이 다름 → external 제어값을 그대로 쓰지 않음.

---

## 3. 권장 아키텍처

스케일카(F1TENTH·BFMC·국내 대회)·팀 보유 코드·논문을 종합하면, **BEV 중심선 + 기하/비례 조향 + 스무딩**이 완성도·디버그·보드 부하 균형이 가장 좋다.

```
/camera/image/compressed
        │
        ▼
┌─────────────────────── lane vision ───────────────────────┐
│  1. ROI (상단 crop + 확장 사다리꼴 src)                      │
│  2. IPM / BEV warp                                         │
│  3. HSV 마스크 (white / yellow / black_road / red_road)    │
│  4. 차선·차로 → 중심선 후보 (단차로 1개 / 분기 시 좌·우 분리 2개+) │
│  5. mode에 따라 path 하나 선택 → LaneResult                       │
└────────────────────────────┬──────────────────────────────┘
                             │ steering_offset, confidence
                             │ (+ 선택: path_candidates)
                             ▼
┌────────────────────── control polish ─────────────────────┐
│  EMA + rate limit + |steer| 감속 + 미검출 hold/decay         │
│  (여유 시 Pure Pursuit / Stanley로 교체 가능한 인터페이스)   │
└────────────────────────────┬──────────────────────────────┘
                             ▼
                    pipeline.fuse_control()
                             ▼
                          /control
```

### 계층별 선택

| 계층 | 1차 채택 | 근거 | 여유 시 |
|------|----------|------|---------|
| ROI/IPM | **확장 사다리꼴 warp** (아래 §4) + YAML | 트랙바 튜닝·실차 이식 용이 | 카메라 높이·pitch 기하 IPM으로 검증 |
| 인식 | BEV + 색마스크 + 중심선(poly / 경계) | 원태·minyong·대회 표준 | 슬라이딩 윈도우 폴백 |
| 다중 경로 | 후보 N개 + **모드 선택** | 갈림길·교차로 | outer/inner FSM (원태) |
| 횡방향 제어 | **오프셋 P + EMA/rate-limit** | limo_sim_code_v2 안정성 | Stanley 또는 Pure Pursuit |
| 종방향 | 상수 cruise + \|steer\| 감속 | 단순·효과 | 곡률·모드별 속도표 |

**원태 코드:** Metric IPM·흰/노란 경계·planning centerline은 **인식 골격 후보**. `detect()` 조향 미연결·imshow 18창은 제거/게이트 후 흡수.  
**minyong:** 슬라이딩 윈도우·한쪽 미러·hold/decay → 폴백·실패 대응.  
**본인 v2:** 제어 스무딩·커브 감속 → **손맛의 기준**.  
**wego DL / LiDAR 미션:** 1차 제외.

---

## 4. BEV / ROI 규약 (팀 확정 방향)

### 4.1 문제

고전 사다리꼴은 **윗변(먼 쪽)을 좁히고** 아랫변을 이미지 하단 폭 100%에 맞춘다.  
이렇게 하면:

- 먼 바닥의 **좌우 가장자리**가 좁은 윗변 밖으로 잘려 BEV에 안 들어감
- 아랫변을 이미지 폭에 고정하면, warp가 시야를 “꽉 채우도록” 잡아 **더 넓은 지면 범위**를 담기 어렵다

### 4.2 팀 규약 — “상단만 제외, 나머지 픽셀 전부 사용”

**의도 (한 줄):** 카메라 이미지에서 **상단 N%만 제외**하고, 그 아래 사각형 안의 **모든 픽셀을 빠짐없이** BEV에 넣는다.  
사다리꼴 `src`는 **이미지 테두리 안에만 그리는 고전 ROI가 아니다** — 아랫변 꼭짓점은 프레임 **밖(가상)** 에 둘 수 있다.

1. **제외:** 윗부분 `crop_top_ratio`(상단 N%)만 사용하지 않음.  
2. **포함:** `y = crop_top … H-1`, `x = 0 … W-1`의 **전 픽셀**이 warp 입력에 기여해야 함  
   - 좁은 윗변 사다리꼴로 좌·우를 잘라내지 않음  
   - 하단 행도 crop하지 않음  
3. **사다리꼴 윗변:** 제외선 높이에서 **가로 풀폭** (`(0, top_y)`–`(W-1, top_y)`)  
4. **사다리꼴 아랫변:** 이미지 하단 폭보다 **넓게** — 꼭짓점이 `x < 0`, `x ≥ W` 등 **프레임 밖**이어도 됨  
   - 목적: 남는 픽셀을 버리지 않고, 넓은 지면 범위로 투영  
   - 카메라에 없는 영역은 BEV에 **부채꼴·삼각 검정(무정보)** 으로 남아도 허용  
5. 확정값은 `config/lane_vision.yaml` (가칭)에 저장 → 시뮬·실차 동일 스키마

```
카메라 프레임 (W × H)                 BEV (직사각 캔버스)
───────────────────────────           ─────────────────────
|///// 상단 N% 제외 /////|            |                     |
|←— 윗변 = 이미지 풀폭 —→|            |  crop 아래 전 픽셀   |
|  ★ 이 영역 픽셀 전부 ★  |    →     |  이 안으로 투영      |
|  (한 픽셀도 버리지 않음) |   warp    |  /  유효  \         |
|← 아랫변 > W (프레임 밖) →|           | /   부채꼴  \        |
   사다리꼴은 이미지 밖까지              |/___검정__검정_\|
```

### 4.3 트랙바로 조절할 파라미터 (Phase 0 툴)

| 파라미터 | 의미 |
|----------|------|
| `top_y` / `crop_top_ratio` | 윗변 세로 위치 (이 위는 미사용) |
| `bottom_half_width_ratio` | 아랫변 반폭 / (W/2). **1.0 = 이미지 하단 풀폭**, **>1.0 = 바깥으로 확장** |
| (선택) `top_inset_ratio` | 기본 0. 예외적으로 윗변만 살짝 줄일 때 |
| `bev_width`, `bev_height` | BEV 해상도 |
| (선택) `dst_top_margin`, `dst_side_margin` | BEV 캔버스 여백 |

**튜닝 UI:** 한 화면에 **원본 + ROI 오버레이(사다리꼴) + BEV** 동시 표시.  
검정 부채꼴이 보여도, **차선·차로가 BEV 중앙 유효 영역에 왜곡 없이** 들어오면 성공.

### 4.4 기하 IPM과의 관계

확장 사다리꼴은 **빠른 시각 캘리브**용이다.  
카메라 높이·pitch·HFoV(`sim_interface.yaml` / 실측)로 Metric IPM을 쓰면 동일한 “지면 범위”를 물리 단위로 검증할 수 있다.  
**순서:** 사다리꼴로 확정 → (선택) 기하 파라미터와 cross-check → YAML 동결.

---

## 5. 색·모드·다중 경로

### 필수 요구 — 분기·갈림길에서 **두 차로 분리 감지**

대회 트랙에는 **차로가 둘로 갈라지는 구간**이 있다 (좌우 갈림길, 회전교차로 진입·탈출 등).

| 요구 | 설명 |
|------|------|
| **분리 감지** | 차선(또는 차로)이 두 갈래로 보이면 **하나의 평균 중심선으로 합치지 말고**, 좌·우(또는 outer/inner) **경로를 각각** 검출한다 |
| **동시 유지** | 분기 구간에서는 `path_candidates`에 **최소 2개**가 동시에 유효할 수 있어야 한다 |
| **선택만 모드** | “어느 갈래를 따를지”는 인식이 아니라 **모드/표지판/FSM**이 고른다. 인식은 후보를 모두 제공한다 |
| **단차로 복귀** | 다시 한 차로로 합쳐지면 후보 1개(또는 단일 중심선)로 돌아가면 된다 |

잘못된 예: 두 갈래 차선의 mid-point만 취해 분기 한가운데로 조향.  
올바른 예: `left_branch` / `right_branch` 중심선을 각각 추정 → `FORK_LEFT`면 왼쪽만 `LaneResult`에 반영.

### 마스크 레이어

| 레이어 | 용도 |
|--------|------|
| `white_lane` | 일반 Out 코스 차선 |
| `yellow_lane` | 회전교차로 차선 |
| `black_road` | 주행 가능 차로 (일반) |
| `red_road` | 장애물 구간 차로 힌트 (**빨간불과 분리**) |

### 경로 후보와 모드

```
lane_detection
  → path_candidates[]   # 분기 시 반드시 2개+ (left/right 또는 outer/inner)
  → 각 후보: centerline, confidence, (선택) lane_width

pipeline / FSM
  → selected_path       # 표지판·미션 모드로 하나 선택
  → LaneResult(steering_offset, confidence)
```

| 모드 예 | 선택 |
|---------|------|
| `LANE_FOLLOW` | 단일(또는 기본) 중심선 |
| `FORK_LEFT` / `FORK_RIGHT` | 갈림길 **해당 갈래만** 추종 (다른 후보는 디버그·전환용으로 유지) |
| `ROUNDABOUT_*` | 노란 차선 + outer/inner (서준 모듈과 계약) |
| `OBSTACLE_ZONE` | 빨간 차로 컨텍스트 + ArUco 정지 우선 |

시뮬 기본: 신호등 무시 가능. ArUco `should_stop`은 `fuse_control` 최우선 유지.

---

## 6. 제어·스무딩

`limo_sim_code_v2`에서 검증된 패턴을 D-Racer `[-1,1]`에 맞게 이식한다.

| 단계 | 내용 |
|------|------|
| 오차 | BEV 중심선 vs 차량 중심 (픽셀 또는 m) |
| raw steer | P gain → `steering_offset` |
| EMA | `α`로 저역 통과 |
| rate limit | 프레임당 Δsteer 상한 |
| 속도 | `cruise_throttle`; `|steer|` 크면 감속 |
| 실패 | confidence 낮으면 **hold + decay** (minyong `test_lkas`) |

여유 시: 동일 중심선을 Pure Pursuit / Stanley 입력으로 교체. 인식 출력 인터페이스는 유지.

---

## 7. 구현 단계 (코딩 순서)

| Phase | 내용 | 산출물 |
|-------|------|--------|
| **0** | ROI/IPM 튜닝 툴 + 캡처 툴 | `tools/tune_bev_roi`, `tools/capture_camera`, YAML 초안 |
| **1** | HSV/마스크 트랙바 + 클릭 피커 | `tools/tune_hsv`, `lane_vision.yaml` 색 구간 |
| **2** | 단경로 주행 루프 + 스무딩 | `lane_detection` → `/control`, Gazebo 직선·완만 곡선 |
| **3** | 다중 경로 + 모드 | 갈림길·교차로에서 **두 차로 분리 감지** + 모드 선택 |
| **4** | (선택) PP / Stanley | 제어만 교체 |

**원칙:** ROI/IPM 확정 **전**에 HSV·주행을 깊게 튜닝하지 않는다.  
원태 PR은 Phase 2에서 **필요한 인식 부분만** 흡수하고, imshow 게이트·조향 어댑터를 전제로 한다.

---

## 8. 레포 배치 (예정)

| 경로 | 역할 |
|------|------|
| `src/inference/tools/` 또는 `scripts/vision_tune/` | 트랙바·캡처·피커 (시뮬·실차 공용) |
| `config/lane_vision.yaml` | ROI/IPM/HSV 확정값 |
| `src/inference/inference/modules/lane_detection.py` (+ `lane/` 서브패키지) | 런타임 인식 |
| `pipeline.py` / `inference_node.py` | fusion·스무딩·모드 (담당 협의) |

보드: `image_topic`만 바꾸면 동일 툴 사용. 캡처는 monorepo `data/` 또는 보드 로컬 경로로 저장.

---

## 9. 튜닝 툴 목록

| 툴 | 시점 | 기능 |
|----|------|------|
| `tune_bev_roi` | Phase 0 | 원본 · ROI 사다리꼴 · BEV, 확장 아랫변 트랙바, YAML 저장 |
| `capture_camera` | Phase 0 | 토픽 → `data/captures/...` |
| `tune_hsv` / 클릭 피커 | Phase 1 | 픽셀 HSV, 레이어별 inRange, BEV 위 마스크 오버레이 |
| (선택) 히스토그램 뷰 | Phase 1–3 | 갈림길 2-peak·차선 강도 확인 |

---

## 10. External·팀 코드 자산 — 참고점

상위 monorepo `external/` 및 팀 브랜치. **통째 이식 금지**, 아래만 cherry-pick.

### 10.1 `feature/wontae-lane` (장원태 PR #16)

| 항목 | 내용 |
|------|------|
| 경로 | `2026-SMH` `origin/feature/wontae-lane` → `modules/lane_detection.py` |
| 강점 | Metric IPM(카메라 HFoV·높이·pitch가 시뮬과 정합), 흰/노란 경계, planning centerline, outer/inner FSM |
| 가져올 것 | IPM·중심선·교차로/점선에 강한 경계 추적 아이디어 |
| 버릴·고칠 것 | `detect()`가 항상 `(0,0)`, `cv2.imshow` 18창 상시, ~3500줄 일체 merge, `sim_bringup` timeout 혼입 |
| 우리 구조에서의 위치 | **인식 본체 후보** (Phase 2 흡수). 조향·스무딩·debug 게이트는 별도 |

### 10.2 `external/limo_sim_code_v2` (안승현 ROS1 LIMO)

| 항목 | 내용 |
|------|------|
| 핵심 파일 | `nodes/lane_follower_bev.py` |
| 강점 | BEV+도로 마스크 단순성, **EMA + rate limit**, \|steer\| 감속, 안전 stop, 디버그 이미지 토픽 |
| 가져올 것 | **제어 스무딩·속도 스케줄·실패 시 정지/홀드 철학** |
| 버릴 것 | HSV “어두운 도로” 가정(대회는 흰/노란 차선), 하드코딩 BEV, `/cmd_vel` Twist 직접, LiDAR M3/M4 |
| 우리 구조에서의 위치 | Phase 2 **손맛 기준**. 인식 색모델은 쓰지 않음 |

### 10.3 `external/limo_minyong` (학과 친구 ROS1)

| 항목 | 내용 |
|------|------|
| 핵심 파일 | `src/limo_drive/scripts/robust_lkas_node.py`, `test_lkas_node.py` |
| 강점 | BEV+HSV+**슬라이딩 윈도우**+polyfit, 한쪽 차선 미러 폴백, `test_lkas`의 **hold/decay·파라미터화** |
| 가져올 것 | 윈도우/폴백/실패 유지, 중앙 ROI 노이즈 컷·세로 모폴로지 아이디어 |
| 버릴 것 | LiDAR FSM·시간 오픈루프 미션, Melodic/`Twist`, LIMO 고정 warp/HSV 수치 |
| 우리 구조에서의 위치 | 인식 **폴백**, 제어 실패 대응. 주 인식이 원태/확장BEV면 전체를 다시 짜지 않음 |

### 10.4 `external/wego-wego_deep_learning`

| 항목 | 내용 |
|------|------|
| 핵심 파일 | `dl_ros2_application/.../tf_track_detect.py`, `dl_control.py` |
| 강점 | ROS2, CNN 트랙점, gain·prev/current 블렌딩, 표지판 연동 속도 |
| 가져올 것 | (후순위) 조향 블렌딩 아이디어만 |
| 버릴·보류 | 모델·학습 데이터 의존, 보드 부담 → **1차 미채택** |

### 10.5 `external/D-Racer-Kit`

| 항목 | 내용 |
|------|------|
| 참고 | `control_node`(`/control`→PCA9685), 카메라 토픽·`vehicle_config`, launch 골격 |
| 자율 차선 로직 | **없음** (`opencv_node`는 Canny 데모) |
| 우리 구조에서의 위치 | 인터페이스·실차 배선만. 알고리즘 SSOT는 팀 `inference` |

### 10.6 기타 (`limo_ros2`, `ahns_limo_sim`, `ugv_gazebo_sim`, `wego-LimoIsaacSIM`)

Gazebo/Isaac **모델·브리지** 위주. 비전 차선추종 알고리즘 자산 없음. 시뮬 월드·Ackermann 플러그인 참고만.

### 10.7 외부 관행 (검색·대회 코드 요약)

| 출처 | 참고점 |
|------|--------|
| F1TENTH lane assist | BEV ROI 튜닝 스크립트 → YAML, 중심선 추종 |
| 국내 스케일카 (sliding window + PP) | 검출·경로·Pure Pursuit 분리 |
| 모델카 Stanley lane-keeping | crosstrack+heading, 고속에서 P만보다 안정 보고 |
| Small-scale AD survey | 카메라+색/외형 기반이 소형 플랫폼의 주류 |

→ 우리 1차는 **BEV+마스크+중심선+스무딩 P**, 인터페이스만 PP/Stanley 교체 가능하게.

---

## 11. 성공 기준

| Phase | Done when |
|-------|-----------|
| 0 | 확장 아랫변 BEV에서 직진 구간 차선이 대략 세로·평행, YAML 저장·재로드 가능 |
| 1 | 흰/노란/검/빨 마스크가 시뮬 캡처에서 안정 |
| 2 | Gazebo에서 기본 차선 추종, 조향 떨림 없이 S자 전 구간 수준 유지 |
| 3 | 분기에서 **좌·우(또는 outer/inner) 중심선이 동시에** 디버그에 보이고, 모드 전환 시 추종 갈래만 바뀜 |

---

## 12. 결정 요약

1. **구조:** ROI(확장 사다리꼴) → BEV → 다중 색마스크 → **분기 시 차로별 중심선 분리 감지** → 모드 선택 → 스무딩 → `/control`  
2. **BEV:** 상단 N%만 제외 · 그 아래 **전 픽셀 사용** · 사다리꼴 `src`는 프레임 **밖 확장 허용**(무정보 검정 OK) · 하단 crop 없음  
3. **분기:** 차선이 둘로 갈라지면 **합치지 않고** 두 경로를 모두 유지; 선택은 FSM/표지판  
4. **자산:** 인식≈원태(+minyong 폴백), 손맛≈v2, 실차 I/O≈D-Racer-Kit, DL·LiDAR 미션은 후순위  
5. **순서:** 툴·ROI/IPM → HSV → 단경로 주행 → 다중경로/모드 → (선택) PP/Stanley  

구현은 이 문서를 기준으로 Phase 0부터 진행한다.
