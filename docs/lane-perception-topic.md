# 차선 인지 출력 · 토픽 구조 (원태)

`inference_node`가 카메라 프레임마다 **단일 토픽 `/perception/lane`** 으로 차선 인지 결과 전체를 발행한다. 판단제어(판제)는 이 토픽 하나만 구독하면 된다.

- 좌표계: `base_link` (x 전방, y 왼쪽, z=0), 단위 m
- 이 노드는 **인지 전용** — 조향/제어(`/control`)는 계산하지 않는다(외부 판제 노드가 소유).

---

## 1. 노드 구조 (perception-only)

```
/camera/image/compressed (CompressedImage)
        │ image_callback
        ▼
   run_perception(frame)                    # pipeline.py
        └ lane_detection.detect() → LaneDetections(dataclass)
        │
        ├─► /perception/lane   (lane_msgs/LaneDetections)   ← 판제로 송신
        └─► /debug/aruco       (std_msgs/String)
```

| 방향 | 토픽 | 타입 |
|------|------|------|
| 구독 | `/camera/image/compressed` | `sensor_msgs/CompressedImage` |
| 발행 | `/perception/lane` | `lane_msgs/LaneDetections` |
| 발행 | `/debug/aruco` | `std_msgs/String` |

> `/control`은 이 노드가 발행하지 않는다. 조향은 외부 판제가 `/perception/lane`을 구독해 계산·발행한다.

---

## 2. 토픽 메시지 (`lane_msgs/LaneDetections`)

새 인터페이스 패키지 **`src/lane_msgs`** (ament_cmake). `inference`가 ament_python이라 msg를 직접 정의하지 못해 분리했다.

```
std_msgs/Header header               # frame_id="base_link"

# 색상별 차선 마킹 후보
LaneMarking[] lanes
  uint16 id                          # 프레임 내 후보 번호
  uint8  color                       # 0=UNKNOWN 1=WHITE 2=YELLOW
  uint8  side_hint                   # 0=UNK 1=LEFT 2=RIGHT 3=CENTER
  float32 confidence                 # 0~1
  float32 length                     # m
  float32 heading                    # rad
  float32 curvature
  geometry_msgs/Point32[] points     # base_link polyline

bool white_visible / yellow_visible / left_visible / right_visible
float32 white_confidence / yellow_confidence / left_confidence / right_confidence

# 흰/노란 차선 센터라인(좌우 경계 중점)
geometry_msgs/Point32[] white_centerline
geometry_msgs/Point32[] yellow_centerline

# 노란 가로 실선(정지선/원형교차로 진입선) 등장 여부
bool yellow_crossing_line

# 갈림길
bool fork_active                     # 분기 발생 여부
RoadBranch[] branches                # 분기(없으면 단일 경로 1개)
  uint16 branch_id                   # 분기 번호 0=가장 왼쪽
  float32 confidence
  float32 width                      # m
  geometry_msgs/Point32[] centerline # base_link 센터라인

# 주행가능영역 (BEV mono8 그리드 0/255)
sensor_msgs/Image drivable_area
float32 meters_per_pixel             # 그리드→base_link 스케일
float32 x_forward_max
```

### drivable_area 해석 (그리드 → base_link)
```
x = x_forward_max - row * meters_per_pixel
y = ((width - 1) / 2 - col) * meters_per_pixel
```

---

## 3. 인지 파이프라인 요약 (`modules/lane_detection.py`)

- **BEV**: `config/lane_vision.yaml`의 `bev_roi`를 `scripts/vision_tune/bev_roi.py`의 `warp_bev`(trapezoid)로 변환. 500×370 px, 가로 m/px≈0.003977.
- **주행가능영역 보정**: 도로에 둘러싸인 점선 구멍(flood-fill) + 도로 가로지르는 실선(등방 close·line-support)을 메움. 분기 섬은 보존.
- **갈림길 분기(half-split)**: 공통 stem을 midline으로 좌/우 반 갈라 각 반쪽 중심을 갈래로. Gaussian 스무딩(코너/교차 없음).
- **같은 색 갈래만**: 흰/노랑 마스크로 현재 도로 색 판정 → 다른 색(별개 도로) 갈래는 분기로 보지 않음.
- **센터라인**: 흰/노란 좌우 경계 중점.
- **노란 가로 실선 플래그**: `find_crossing_lines`(도로 위 노란선 중 차선과 직교하는 성분). ⚠️ 현재 실데이터에서 신뢰도 낮음 — 강화 필요.

---

## 4. 빌드 · 확인

```bash
colcon build --packages-up-to inference   # lane_msgs → inference
source install/setup.bash
ros2 run inference inference_node

ros2 interface show lane_msgs/msg/LaneDetections
ros2 topic echo /perception/lane --once
```
