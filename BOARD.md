# 대회·실차 보드 (`board/race-control`)

PC feature 작업과 **분리된 worktree** (`…/2026-SMH-board`).  
시뮬·Gazebo 내용은 이 브랜치 SSOT가 아닙니다.

## 한 줄

```bash
cd /path/to/2026-SMH-board
./scripts/board_race_sync.sh
source install/setup.bash
ros2 launch inference auto_driving.launch.py route_mode:=in   # or out
# 모니터: http://<보드IP>:5000
```

## 미션 FSM (적용)

1. 실행 시 `route_mode:=in|out` (또는 yaml `route.mode`)으로 코스 고정  
2. **초록불** → 출발 (`require_green_to_start`)  
3. 첫 갈림/분기에서 선택한 길로 진입  
   - **OUT:** S자 · 표지+capture로 `out_fork` L/R 추종  
   - **IN:** 회전교차로 · moment로 keep(우)→exit(좌)  
4. **ArUco** 보이면 정지, 사라지면 재출발  
5. **빨간불** → 정지  

## 실차 확정 SSOT (이 브랜치)

| 영역 | 내용 |
|------|------|
| 카메라 | **320×180**, Kit 패치로 native caps + V4L2 auto off |
| Metric IPM | height **0.13 m**, pitch **13.8°**, x 0.22–**1.3 m**, mpp **0.004** |
| HSV + drivable | `real_car` + near-band black/cyan + morph **open3/close13** + ego blob |
| Fork 판단 | `fork/judgment.py` — OUT sign∧capture · IN moment keep/exit (**실차 검증 대기**) |
| 기하 | L **0.175**, R_min **0.385**, δ_max **0.4266**, d_rc **0.200** (시뮬값 금지) |
| `/control` | 소프트웨어 계약 **+steering = 우**. 이 보드 서보 반대 배선 → `STEER_INVERT: true` |
| 조이스틱 | launch에 포함 (E-Stop / 수동) |
| 신호등 | OpenCV HSV(**board**) **또는** 성준 YOLO light-only(`sign_light_best_v5b`) — `TRAFFIC_LIGHT_BACKEND`로 A/B |
| 방향 표지판 | YOLO `weights/sign_best.onnx` (**origin/board** 2-class). 성준 통합모델의 표지판 클래스는 **미사용** |
| 신호·ArUco | green start / red stop / ArUco stop **ON** |

상세 수치: `config/lane_vision.yaml`, `config/main_planner.yaml`, `config/vehicle_config.yaml`, `docs/hsv-profiles.md`, `docs/vehicle-geometry.md`.

## 레이아웃

```
2026-SMH-board/
├── BOARD.md
├── config/                 ← 실차 yaml만 신뢰
├── patches/                ← 공식 Kit에 적용 (서보 invert·카메라)
├── scripts/board_*.sh
├── external/D-Racer-Kit/   ← 공식 (커밋 안 함, 패치만)
└── src/inference/          ← 팀 인지·플래너·제어
```

## 보드 동기화

```bash
# Kit가 이미 ~/D-Racer-Kit 이면
rm -rf external/D-Racer-Kit && mkdir -p external
ln -sfn ~/D-Racer-Kit external/D-Racer-Kit
./scripts/board_race_sync.sh --no-pull
```

`board_init_workspace.sh`가 Kit 링크 후 `patches/*.patch`를 적용합니다.

## 트랙에서 다음으로 할 일

1. **마스크/BEV 확인** — `/perception/lane`의 `drivable_area`, 모니터 대시보드, 필요 시 OpenCV debug  
2. **조향 스탠드 체크** — `/control` `steering=+1` → 바퀴 **오른쪽** (invert 후). 조이스틱으로 L/δ·trim 재확인  
3. **제어 추종** — NORMAL 추종은 아직 실차 튜닝 대상. fork/exit 판단도 라이브로 검증  
4. 조명·노출이 bag과 다르면 HSV만 보드에서 미세 조정 (`hsv.active: real_car`)

## worktree

| 폴더 | 브랜치 | 용도 |
|------|--------|------|
| `2026-SMH` | `feature/…` | 실험·bag 튜닝 |
| `2026-SMH-board` | `board/race-control` | 실차·대회 실행 |
