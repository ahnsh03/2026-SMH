# 2026-SMH

**2026 SEA:ME 해커톤** — AIM 학술동아리 자율주행 팀 저장소

> Repo 이름은 대회 약자(SMH)를 사용합니다. 로컬 경로·셸 변수: `$SMH`

> **PC(WSL) 상위 프로젝트**: [../README.md](../README.md)  
> **D3-G 보드 단독 clone** (`~/2026-SMH`) — 아래 빠른 시작만으로 충분합니다.

| | |
|---|---|
| **대회** | 2026.7.14 ~ 7.16 / 호텔 파크하비오 |
| **주제** | AI 네이티브 스케일카 자율주행 챌린지 |
| **정기 회의** | 매주 월요일 15시 |
| **Notion** | [팀 대시보드](https://app.notion.com/p/55e1b0cdce9b8292a19d81c5b1605983) |

---

## 빠른 시작 (D3-G 보드)

```bash
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull   # 최초 1회
```

이후 코드 받을 때:

```bash
./scripts/board_sync.sh             # pull + init + build
source install/setup.bash
ros2 launch inference auto_driving.launch.py
```

자세한 셋업: [docs/setup.md](docs/setup.md)  
협업 규칙: [docs/collaboration.md](docs/collaboration.md)

---

## 저장소 구조

```
2026-SMH/
├── docs/
│   ├── collaboration.md   # ★ 브랜치·PR·충돌 방지 (팀원 필독)
│   ├── roles.md           # 역할 분담
│   ├── setup.md           # 셋업
│   └── competition.md     # 대회 정보
├── scripts/
│   ├── init_workspace.sh  # D-Racer-Kit clone + src/ 링크
│   └── board_sync.sh      # ★ 보드: pull + init + build
├── external/              # D-Racer-Kit (Git 제외, init 시 생성)
└── src/
    └── inference/         # ★ 팀 자율주행 패키지 (Git 추적)
        ├── inference/
        │   ├── types.py           # 공통 타입
        │   ├── pipeline.py        # 모듈 통합 (팀장)
        │   ├── inference_node.py  # ROS2 노드
        │   └── modules/           # ★ 담당자별 개발
        └── launch/
            ├── auto_driving.launch.py
            └── manual_driving.launch.py
```

주최측 패키지(camera, control 등)는 `init_workspace.sh`가 `src/`에 심볼릭 링크합니다.

---

## 역할 분담

| 담당 | 모듈 | 파일 |
|------|------|------|
| **장원태** | 차선 인지 | `modules/lane_detection.py` |
| **장원정** | 신호등·표지판 | `modules/traffic_sign.py` |
| **안승현** | ArUco 검출 | `modules/aruco/detector.py` |
| **박성준** | ArUco 정지 | `modules/aruco/stop_logic.py` |
| **양서준** | 회전 교차로 | `modules/roundabout.py` |

상세: [docs/roles.md](docs/roles.md)

---

## 데이터 흐름

```
/camera/image/compressed
        │
        ▼
  inference_node → pipeline.run_perception()
        │            (lane / traffic / aruco / roundabout)
        ▼
  pipeline.fuse_control()  →  /control  →  control_node
```

---

## 실행

```bash
source install/setup.bash

# 수동 주행
ros2 launch inference manual_driving.launch.py

# 자율주행
ros2 launch inference auto_driving.launch.py
```

---

## 브랜치 규칙 (필수)

> 상세: [docs/collaboration.md](docs/collaboration.md) §1

1. **`main` 직접 push 금지** — PR merge로만 반영
2. **`feature/이름-기능` 브랜치**에서만 개발 → commit → push → **PR**
3. **한 PR = 담당 `modules/` 한 모듈** (작은 PR)
4. merge 후 보드에서 `./scripts/board_sync.sh`로 `main` 테스트

```
main → feature/wontae-lane → PR → merge → board_sync.sh
```

---

## 문서

- [협업 가이드](docs/collaboration.md) ★
- [역할 분담](docs/roles.md)
- [셋업 가이드](docs/setup.md)
- [대회 정보](docs/competition.md)
- [참고 링크](docs/references.md)

---

## 주최측 제공 자료

- 공식 ROS2: https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0
- 참고 영상: https://drive.google.com/file/d/1QpnQdkiiYtEs1k2Ll4sRCjBB_1pBNbmG/view
