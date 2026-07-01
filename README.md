# SEA-Me-Hackathon

**2026 SEA:ME 해커톤** — AIM 학술동아리 자율주행 팀 저장소

| | |
|---|---|
| **대회** | 2026.7.14 ~ 7.16 / 호텔 파크하비오 |
| **주제** | AI 네이티브 스케일카 자율주행 챌린지 |
| **정기 회의** | 매주 월요일 15시 |
| **Notion** | [팀 대시보드](https://app.notion.com/p/55e1b0cdce9b8292a19d81c5b1605983) |

---

## 빠른 시작 (D3-G 보드)

```bash
git clone https://github.com/ahnsh03/SEA-Me-Hackathon.git
cd SEA-Me-Hackathon
chmod +x scripts/init_workspace.sh
./scripts/init_workspace.sh

source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

자세한 셋업: [docs/setup.md](docs/setup.md)

---

## 저장소 구조

```
SEA-Me-Hackathon/
├── docs/
│   ├── competition.md   # 대회 정보·규정·주최측 문답
│   ├── roles.md         # 역할 분담
│   ├── setup.md         # 개발 환경 셋업
│   └── references.md    # 링크 모음
├── scripts/
│   └── init_workspace.sh   # D-Racer-Kit + 팀 패키지 연동
└── src/
    └── inference/          # 팀 자율주행 패키지
        └── inference/
            ├── inference_node.py
            └── modules/
                ├── lane_detection.py    # 장원태
                ├── traffic_sign.py      # 장원정
                ├── aruco_detection.py   # 안승현, 박성준
                └── roundabout.py        # 양서준
```

주최측 공식 코드([D-Racer-Kit](https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0))는 `init_workspace.sh`로 `external/`에 받아 `src/`에 링크됩니다.

---

## 역할 분담

| 담당 | 모듈 | 파일 |
|------|------|------|
| **장원태** | 차선 인지 | `modules/lane_detection.py` |
| **장원정** | 신호등·표지판 | `modules/traffic_sign.py` |
| **안승현, 박성준** | ArUco 마커 | `modules/aruco_detection.py` |
| **양서준** | 회전 교차로 | `modules/roundabout.py` |

상세: [docs/roles.md](docs/roles.md)

---

## 데이터 흐름

```
/camera/image/compressed
        │
        ▼
  inference_node  ◄── modules/ (lane, traffic, aruco, roundabout)
        │
        ▼
    /control  ──►  control_node  ──►  모터/서보
```

---

## 실행

```bash
# 수동 주행 (하드웨어 확인)
ros2 launch control manual_driving.launch.py

# 자율주행 (inference 포함)
ros2 launch control auto_driving.launch.py
```

---

## 브랜치 규칙

1. `main` — 안정 버전 (팀장 merge)
2. `feature/이름-기능` — 개인 개발
3. 담당 `modules/*.py` 수정 후 PR

---

## 문서

- [대회 정보](docs/competition.md)
- [역할 분담](docs/roles.md)
- [셋업 가이드](docs/setup.md)
- [참고 링크](docs/references.md)

---

## 주최측 제공 자료

- 공식 ROS2: https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0
- 참고 영상: https://drive.google.com/file/d/1QpnQdkiiYtEs1k2Ll4sRCjBB_1pBNbmG/view
