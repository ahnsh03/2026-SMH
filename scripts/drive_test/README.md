# Fork spawn unit tests

`fork_spawn_unit.py` — Out 갈림 / In 탈출 스폰 구간 단위 테스트 + 로깅.

## 계약

| | |
|--|--|
| LEFT | rank **0** |
| RIGHT | rank **1** |
| IN | 노란 우선 + yellow fork layers |
| OUT | 흰 중앙 + white/`road_split` layers |
| 선택 후 | PP는 선택 레이어만 |

## Offline (시뮬 불필요)

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all
```

로그: `data/captures/fork_drive_logs/<stamp>/`

## Live (Gazebo bringup 필요)

```bash
# 터미널 A: bringup (예)
ros2 launch dracer_sim sim_auto_driving.launch.py route_mode:=out spawn_pose:=out_fork

# 터미널 B: 단위 시나리오
python3 scripts/drive_test/fork_spawn_unit.py --mode live \
  --scenario out_left --duration 8 --repeat 2

# In 탈출
python3 scripts/drive_test/fork_spawn_unit.py --mode live \
  --scenario in_exit_left --duration 8
# (스크립트가 teleport + force_fork_choice 수행)
```

수동 텔레포트만:

```bash
./scripts/dev_container.sh teleport out_fork
./scripts/dev_container.sh teleport in_roundabout_exit
```
