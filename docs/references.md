# 참고 링크

## 주최측 제공

| 자료 | 링크 |
|------|------|
| D-Racer-Kit (공식 ROS2 패키지) | https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0 |
| 참고 영상 (Google Drive) | https://drive.google.com/file/d/1QpnQdkiiYtEs1k2Ll4sRCjBB_1pBNbmG/view |
| D3-G Ubuntu 이미지 | [다운로드](https://topst-downloads.s3.ap-northeast-2.amazonaws.com/Ubuntu/22.04/D-Racer-ubuntu-22.04-v1.0.1.zip) |

## 팀 내부

| 자료 | 링크 |
|------|------|
| 팀 GitHub | https://github.com/ahnsh03/2026-SMH |
| PC 로컬 영상·트랙 자료 (`data/`, Git 밖) | WSL: `~/projects/2026-seame-hackathon/data/` |
| PC Docker 개발 환경 | [dev-environment.md](./dev-environment.md) |
| Notion 대시보드 | https://app.notion.com/p/55e1b0cdce9b8292a19d81c5b1605983 |
| Notion 대회 정보 | https://app.notion.com/p/3901b0cdce9b81eaa6eff92ecd0f026b |
| 정기 회의 | 매주 **월요일 15시** |
| 공지 | 카카오톡 오픈채팅방 |

## 실행 (팀 launch)

```bash
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

## 하드웨어 · 시뮬

| 자료 | 링크 |
|------|------|
| **D3-G / D-Racer 플랫폼 스펙** | [hardware-board.md](./hardware-board.md) |
| C920e 카메라 스펙 | [hardware-camera.md](./hardware-camera.md) |
| **D-Racer ↔ LIMO 조향·기하 (튜닝)** | [vehicle-geometry.md](./vehicle-geometry.md) |
| LIMO 시뮬 (팀 `dracer_sim`) | [simulation-setup.md](./simulation-setup.md) · [dracer_sim README](../src/dracer_sim/README.md) |
| limo_ros2 upstream (참고) | https://github.com/agilexrobotics/limo_ros2.git — **팀은 `vendor/limo_car` 사용** |
| ugv_gazebo_sim (ROS1) | https://github.com/agilexrobotics/ugv_gazebo_sim.git |
| limo_sim_code_v2 (작년 ROS1 앱) | https://github.com/ahnsh03/limo_sim_code_v2.git |
| Logitech C920e Sync Hub 스펙 | https://hub.sync.logitech.com/c920e/post/specifications---c920e-business-webcam-TKnike7FetCzuAt |

PC monorepo(`2026-seame-hackathon`)를 쓰는 경우 시뮬 레포 평가 메모: 상위 `docs/sim/limo-simulator-assessment.md`.  
**레포만 clone** 시에는 [simulation-setup.md](./simulation-setup.md)가 SSOT입니다.

## D-Racer-Kit 주요 패키지

| 패키지 | 역할 |
|--------|------|
| `camera` | C920e USB → `/camera/image/compressed` (팀: 320×180) |
| `control` | `/control` → 모터/서보 |
| `joystick` | 조이스틱 + E-Stop |
| `opencv` | OpenCV 데모 (차선 추종 참고) |
| `monitor` | 웹 대시보드 |
| `battery` | 배터리 모니터링 |
| **`inference`** | **팀 자율주행 패키지 (본 레포)** |

## PC 로컬 `data/` 폴더 (Git 미포함)

WSL 상위 프로젝트 `2026-seame-hackathon/data/`에 신호등·표지판 **참고 영상** 등을 둡니다.  
Gazebo 시뮬에 쓰는 표지판 PNG는 **`src/dracer_sim/assets/signs/`** (팀 레포 Git 포함) — clone만으로 시뮬 가능.

| 폴더명 | 설명 |
|--------|------|
| `offboard_handheld_videos/` | 로봇 시점 없이 손으로 직접 촬영한 참고 영상 |
| `onboard_perspective_videos/` | 로봇 카메라 높이·각도를 맞춰 촬영한 영상 |
| `organizer_provided_videos/` | 주최측 제공 영상 |

## ROS2 토픽

```
/camera/image/compressed  →  inference_node  →  /control  →  control_node
/joystick (E-Stop)         →  control_node
```
