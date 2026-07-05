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
| 팀 GitHub | https://github.com/ahnsh03/SEA-Me-Hackathon |
| Notion 대시보드 | https://app.notion.com/p/55e1b0cdce9b8292a19d81c5b1605983 |
| Notion 대회 정보 | https://app.notion.com/p/3901b0cdce9b81eaa6eff92ecd0f026b |
| 정기 회의 | 매주 **월요일 15시** |
| 공지 | 카카오톡 오픈채팅방 |

## 실행 (팀 launch)

```bash
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

## D-Racer-Kit 주요 패키지

| 패키지 | 역할 |
|--------|------|
| `camera` | 카메라 → `/camera/image/compressed` |
| `control` | `/control` → 모터/서보 |
| `joystick` | 조이스틱 + E-Stop |
| `opencv` | OpenCV 데모 (차선 추종 참고) |
| `monitor` | 웹 대시보드 |
| `battery` | 배터리 모니터링 |
| **`inference`** | **팀 자율주행 패키지 (본 레포)** |

## ROS2 토픽

```
/camera/image/compressed  →  inference_node  →  /control  →  control_node
/joystick (E-Stop)         →  control_node
```
