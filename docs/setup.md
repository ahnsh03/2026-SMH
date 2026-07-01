# 개발 환경 셋업

## 사전 요구사항

- **보드**: D3-G (Ubuntu 22.04 + ROS2 Humble 공식 이미지)
- **PC**: Windows 10/11 (SSH 원격 개발) 또는 D3-G 직접 작업
- 공식 가이드: [D-Racer-Kit docs](https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0/docs)

## 1. 워크스페이스 구성

팀 레포와 주최측 D-Racer-Kit을 합쳐 하나의 ROS2 워크스페이스로 사용합니다.

```bash
# D3-G 보드에서 실행
cd ~
git clone https://github.com/ahnsh03/SEA-Me-Hackathon.git
cd SEA-Me-Hackathon
./scripts/init_workspace.sh
```

`init_workspace.sh`는 공식 D-Racer-Kit(`release/v1.0.0`)을 `external/`에 받고,  
`src/`에 공식 패키지 + 팀 `inference` 패키지를 링크합니다.

## 2. ROS2 환경

```bash
echo "source /opt/ros/humble/local_setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 3. 빌드

```bash
cd ~/SEA-Me-Hackathon
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 4. 수동 주행 테스트 (하드웨어 확인)

```bash
ros2 launch control manual_driving.launch.py
```

## 5. 자율주행 실행 (개발 중)

```bash
ros2 launch control auto_driving.launch.py
```

## 6. 유용한 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | 카메라 영상 |
| `/control` | `control_msgs/Control` | steering / throttle |
| `/joystick` | `joystick_msgs/Joystick` | 조이스틱 (E-Stop) |
| `/battery_status` | `battery_msgs/Battery` | 배터리 |

## 7. 브랜치 규칙

- `main` — 안정 버전 (팀장 merge)
- `feature/이름-기능` — 개인 개발 브랜치
- PR 전: 해당 노드 단독 실행 또는 launch 테스트
