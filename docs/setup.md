# 개발 환경 셋업

## 사전 요구사항

- **보드**: D3-G (Ubuntu 22.04 + ROS2 Humble 공식 이미지)
- **PC**: Windows 10/11 + WSL2 (선택) 또는 D3-G 직접 작업
- 공식 가이드: [D-Racer-Kit docs](https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0/docs)

---

## D3-G 보드 (권장 — 단독 clone)

```bash
cd ~
git clone https://github.com/ahnsh03/SEA-Me-Hackathon.git
cd SEA-Me-Hackathon
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull
```

D-Racer-Kit은 `<repo>/external/D-Racer-Kit`에 clone됩니다 (Git 추적 안 함).

### 코드 업데이트 (매번)

```bash
cd ~/SEA-Me-Hackathon
./scripts/board_sync.sh
```

`board_sync.sh` = `git pull` + `init_workspace.sh` + `colcon build`

---

## PC (WSL) — 상위 프로젝트 사용

```bash
cd ~/projects/2026-seame-hackathon/SEA-Me-Hackathon
chmod +x scripts/*.sh
./scripts/init_workspace.sh

source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

D-Racer-Kit이 `../external/D-Racer-Kit`에 있으면 자동으로 사용합니다.

---

## init_workspace.sh 동작

| 항목 | 내용 |
|------|------|
| D-Racer-Kit 위치 1 | `<repo>/external/D-Racer-Kit` (보드 단독 clone) |
| D-Racer-Kit 위치 2 | `<repo>/../external/D-Racer-Kit` (PC 상위 프로젝트) |
| `src/` 링크 | camera, control, joystick 등 공식 패키지 심볼릭 링크 |
| 재실행 | 안전 — 링크를 최신 경로로 갱신 |

### 주의

- `src/camera`, `src/control` 등 **공식 패키지 링크는 Git에 없습니다**
- **매 clone·새 보드마다** `init_workspace.sh` (또는 `board_sync.sh`) 실행 필요
- D-Racer-Kit 최초 clone에 **네트워크** 필요 (`release/v1.0.0`)

---

## ROS2 환경

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 실행

```bash
source install/setup.bash

# 수동 주행 (조이스틱)
ros2 launch inference manual_driving.launch.py

# 자율주행 (팀 inference 파이프라인)
ros2 launch inference auto_driving.launch.py
```

> 주최측 launch (`ros2 launch control auto_driving.launch.py`) 대신  
> **팀 `inference` 패키지 launch**를 사용하세요.

---

## 유용한 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | 카메라 영상 |
| `/control` | `control_msgs/Control` | steering / throttle |
| `/joystick` | `joystick_msgs/Joystick` | 조이스틱 (E-Stop) |
| `/battery_status` | `battery_msgs/Battery` | 배터리 |

---

## 협업

브랜치·PR·충돌 방지: [collaboration.md](./collaboration.md)
