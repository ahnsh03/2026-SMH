# 대회·실차 보드 (`feature/seunghyun-board-race`)

PC feature 작업과 **분리된** 슬림 트리. 시뮬·vendor·Docker 없음.

## 한 줄

```bash
cd ~/2026-SMH-board   # 또는 PC worktree 2026-SMH-board
./scripts/board_race_sync.sh
source install/setup.bash
ros2 launch inference auto_driving.launch.py route_mode:=in   # or out
# 신호등 없이 트랙 중간 테스트: traffic_pass:=true  (초록 대기·빨간 정지 스킵, ArUco는 유지)
# 모니터 패널: Lane (HSV paint) + Road (drivable). 조이스틱 노드 미기동.
# 패치 적용(모니터 라벨·카메라 로그): ./scripts/board_init_workspace.sh 또는
#   python3 patches/apply_monitor_bev_labels.py external/D-Racer-Kit
#   python3 patches/apply_camera_quiet_logs.py external/D-Racer-Kit
# 모니터: http://<보드IP>:5000
```

## 이 브랜치에 있는 것

```
├── BOARD.md / README.md
├── config/                 ← lane_vision, main_planner, vehicle
├── patches/                ← Kit 카메라·서보 invert
├── scripts/board_*.sh      ← init / sync
├── scripts/check_sign_*    ← 표지판·신호등 점검
├── weights/                ← sign_best.onnx (+ light A/B용 v5b)
├── docs/                   ← 실차 SSOT 문서만
├── src/inference/          ← 팀 자율주행
├── src/lane_msgs/          ← /perception/lane 메시지
└── external/D-Racer-Kit/   ← 링크/clone (Git 밖)
```

## 미션 FSM

1. `route_mode:=in|out`  
2. (테스트) `traffic_pass:=true` → 초록 대기/빨간 정지 스킵  
3. 초록불 → 출발  
4. OUT: S자·갈림 / IN: 회전교차로  
5. ArUco 보이면 정지, 사라지면 재출발  
6. 빨간불 → 정지  

## 실차 SSOT 요약

| 영역 | 내용 |
|------|------|
| 카메라 | 320×180 + Kit 패치 |
| Metric IPM | `scripts/vision_tune/metric_ipm.py` + yaml (`pitch_down_deg=10`, `x_max_m=1.5`) |
| HSV+drivable | real_car + near + morph 3/13 |
| 표지판 | YOLO `sign_best.onnx` (프레임당 YOLO 1개) |
| 신호등 | 기본 OpenCV HSV |
| 기하 | L 0.175 / δ 0.4266 / d_rc 0.200 |
| 조향 | 소프트웨어 + = 우, `STEER_INVERT: true` |

## Kit 연결

```bash
rm -rf external/D-Racer-Kit && mkdir -p external
ln -sfn ~/D-Racer-Kit external/D-Racer-Kit
./scripts/board_race_sync.sh --no-pull
```

`board_init_workspace.sh`가 Kit → `src/camera|control|joystick|…` 심볼릭 링크 + patches 적용.
