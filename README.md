# 2026-SMH — 실차·대회 보드 브랜치

원격: `feature/seunghyun-board-race`  
로컬 worktree 이름: `board/race-control`

> 이 브랜치는 **D3-G 실차 실행에 필요한 것만** 담습니다.  
> Gazebo / `vendor/limo_car` / Docker / vision_tune **없음**.

자세한 내용은 **[BOARD.md](BOARD.md)**.

```bash
git clone -b feature/seunghyun-board-race https://github.com/ahnsh03/2026-SMH.git ~/2026-SMH-board
cd ~/2026-SMH-board
# Kit가 있으면
mkdir -p external && ln -sfn ~/D-Racer-Kit external/D-Racer-Kit
./scripts/board_race_sync.sh --no-pull
source install/setup.bash
ros2 launch inference auto_driving.launch.py route_mode:=in
# 신호등 없이 중간 배치 테스트:
# ros2 launch inference auto_driving.launch.py route_mode:=in traffic_pass:=true
# 모니터: 카메라 + Lane(HSV) + Road(drivable) BEV 마스크 (조이스틱 노드 없음)
```
