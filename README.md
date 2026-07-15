# 2026-SMH — D-Racer 자율주행 (인지/제어 분리 구조)

D-Racer-Kit 환경 위에서 동작하는 자율주행 스택. **인지(perception)** 와
**제어(control)** 코드를 별도 패키지로 분리한 구조다.
([ahnsh03/2026-SMH `board`](https://github.com/ahnsh03/2026-SMH/tree/board)
브랜치의 구조/이름을 참고, D-Racer 서보구동 `control` 과 충돌하는 제어 패키지만 `driving` 으로 명명.)

## 주행 데이터 흐름

```
camera_node          inference_node            lane_control_node          control_node
 (D-Racer)   ──▶      (inference/인지)   ──▶     (driving/제어)     ──▶      (D-Racer/서보)
        /camera/image/compressed   /perception/lane          /control
        sensor_msgs/CompressedImage  lane_msgs/LaneDetections  control_msgs/Control
```

- **인지**: 카메라 영상 → 차선/표지/갈림길 인지 → `LaneDetections` 발행
- **제어**: `LaneDetections` → 경로선택 + Pure Pursuit → `Control`(steering/throttle) 발행
- **액추에이터**: D-Racer `control_node` 가 `/control` 을 받아 PCA9685 서보로 구동

## 워크스페이스 구조

```
2026-SMH/
├── config/                       # 파라미터 (board 파일명 동일)
│   ├── lane_vision.yaml          #   인지 파라미터
│   ├── lane_control.yaml         #   제어 파라미터
│   └── main_planner.yaml         #   미션/상위 플래너
└── src/
    ├── inference/                # ★ 인지 패키지 (board 동일 이름/구조)
    │   ├── inference/
    │   │   ├── types.py          #   ROS 비의존 자료형
    │   │   ├── pipeline.py       #   비전 모듈 오케스트레이션
    │   │   ├── inference_node.py #   node: 영상 → LaneDetections
    │   │   └── modules/          #   lane/aruco/traffic_sign/direction_sign
    │   └── launch/               #   auto_driving / manual_driving (통합)
    ├── driving/                  # ★ 제어 패키지 (board의 control 로직 분리)
    │   ├── driving/
    │   │   ├── types.py
    │   │   ├── control_node.py   #   node: LaneDetections → Control
    │   │   └── planner/          #   lane_planner(Pure Pursuit) / path_select
    │   └── (launch 은 inference/launch 통합 사용)
    └── lane_msgs/                # 인지→제어 인터페이스 (board 동일)
        └── msg/                  #   LaneDetections / LaneMarking / RoadBranch
```

> D-Racer 패키지(camera/control/joystick/battery/monitor, `control_msgs`)는
> 이 워크스페이스에 포함하지 않고 **언더레이**로 사용한다.

## 빌드 & 실행 (오버레이 방식)

```bash
# 1) 언더레이: D-Racer-Kit 먼저 빌드 & 소싱
cd ~/D-Racer-Kit
colcon build --symlink-install
source install/setup.bash

# 2) 오버레이: 이 워크스페이스 빌드 (lane_msgs → inference/driving 순으로 자동 해석)
cd ~/2026-SMH
colcon build --symlink-install
source install/setup.bash

# 3) 자율주행 실행
ros2 launch inference auto_driving.launch.py
#   수동주행(teleop):  ros2 launch inference manual_driving.launch.py
```

## 인지↔제어 계약 (Interface Contract)

| 토픽 | 타입 | 발행 | 구독 |
|------|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | camera_node(D-Racer) | inference_node |
| `/perception/lane` | `lane_msgs/LaneDetections` | inference_node | lane_control_node |
| `/control` | `control_msgs/Control` | lane_control_node | control_node(D-Racer) |

이 계약만 지키면 인지팀/제어팀이 독립적으로 개발·교체할 수 있다.
현재 각 모듈은 **스켈레톤**(TODO 표시)이며, 구조/토픽 배선은 완성되어 있다.
