# Fork spawn unit tests

`fork_spawn_unit.py` — Out 갈림 / In 탈출 스폰 구간 단위 테스트 + 로깅.

# Mask steer bench (short A/B) — see also `out_lap_bench.py` for full OUT lap.

`mask_steer_bench.py` — limo_sim식 **mask COM + P + EMA** vs **PP** 짧은 구간 A/B.

## Course mode full-lap IN/OUT (`course_mode_bench.py`)

출발점(`start`)에서 **OUT / IN** 각각 풀랩을 반복해 강건성·속도·차선 가시성을 비교.

```bash
PYTHONUNBUFFERED=1 python3 scripts/drive_test/course_mode_bench.py \
  --modes out,in --repeats 2 \
  --max-lap-sec-out 180 --max-lap-sec-in 240
```

로그: `data/captures/course_mode_logs/<stamp>/` (`all_laps.csv`, `REPORT.md`).

## Mask course-color / fork-guard A/B (`mask_policy_bench.py`)

갈림·합류에서 **코스색 코리도 / fork 강제 PP** 조합을 텔레포트 스윕·상위 랩으로 평가.

```bash
# bringup만 (sim-auto OFF)
PYTHONUNBUFFERED=1 python3 scripts/drive_test/mask_policy_bench.py \
  --phase all --duration 8 --repeat 1 --top-n 3 --max-lap-sec 150
```

로그: `data/captures/mask_policy_logs/<stamp>/` (`meta.json`, `segment_ranking.json`, `REPORT.md`).

> **데이터 신뢰도:** Gazebo 조향 joint `velocity=0.5` 시절 OUT 랩 로그
> (`out_lap_logs/20260714_111751`, `…_113332`)는 **UNTRUSTED** —
> [out_lap_logs/README.md](../../data/captures/out_lap_logs/README.md),
> [vehicle-geometry.md §2.4](../../docs/vehicle-geometry.md).

## OUT full-lap bench (continuous)

`out_lap_bench.py` — **흰 차선 OUT 코스 한 바퀴** 단위 실험.

1. 한 제어 패밀리(`pp` / `mask_p`)로 start부터 연속 주행
2. 차선 유실이 `lost_hold_sec` 이상 유지되면 **최근접 OUT spawn**으로 텔레포트 후 재시도
3. 같은 구간에서 `max_retries` 초과 시 **다음 체크포인트로 스킵**
   (`obstacle` 다음은 **`start`** — 장애물 구간 무한 텔레포트 방지)
4. ArUco/빨간차로 **미션 스탑은 유실로 취급하지 않음**
5. start 복귀 시 랩 완료 → 파라미터 mutate → 다음 trial → 패밀리 전환

로그: `data/captures/out_lap_logs/<stamp>/`

```bash
# bringup (기본 view:=both = 카메라+BEV)
./scripts/dev_container.sh sim-bringup spawn_pose:=start

# 컨테이너 — sim-auto OFF
PYTHONUNBUFFERED=1 python3 scripts/drive_test/out_lap_bench.py \
  --families mask_p,pp --max-param-trials 2 --max-lap-sec 180
```

체크포인트 (CW): `start → inout_fork → out_fork → out_fork_merge_left → out_in_merge → obstacle → (start)`

**구간별 라이브·합류·viz 기본값:** [docs/fork-test-pipeline.md](../../docs/fork-test-pipeline.md)

## 계약

| | |
|--|--|
| LEFT | rank **0** |
| RIGHT | rank **1** |
| IN | 노란이 있으면 노란 우선, 없으면 흰 · yellow fork layers |
| OUT | 흰 중앙 + white/`road_split` layers |
| 선택 후 | PP는 선택 레이어만 |
| **합류** | far-only spur는 fork 아님 → `suppress_merge_spur_branches` ([§0.0.1](../../docs/lane-occlusion-fork-strategy.md)) |

## Offline (시뮬 불필요)

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all
```

로그: `data/captures/fork_drive_logs/<stamp>/`

## Live (Gazebo bringup 필요)

Gazebo와 자율을 **분리**해서 자율만 껐다 켜는 것을 권장합니다.

```bash
# 터미널 A: Gazebo만 (카메라/BEV 창 OFF 권장)
./scripts/dev_container.sh sim-bringup spawn_pose:=out_fork view:=none

# 터미널 B: 자율 + 갈림 프리뷰 1창, 표지 무시
./scripts/dev_container.sh sim-auto route_mode:=out forced_turn:=left viz:=lane
# viz:=off|lane|debug|all  ·  forced_turn 시 로그에 sign_ignored(forced=…)

# In
./scripts/dev_container.sh sim-bringup spawn_pose:=in_roundabout_exit view:=none
./scripts/dev_container.sh sim-auto route_mode:=in forced_turn:=left viz:=lane   # 탈출
./scripts/dev_container.sh sim-auto route_mode:=in forced_turn:=right viz:=lane  # 원 유지
# Ctrl+C 시 /control=0 발행 + bringup bridge watchdog → 로봇 정지
```

```bash
# 터미널 C: 단위 시나리오
python3 scripts/drive_test/fork_spawn_unit.py --mode live \
  --scenario out_left --duration 8 --repeat 2

python3 scripts/drive_test/fork_spawn_unit.py --mode live \
  --scenario in_exit_left --duration 8
# (스크립트가 teleport + force_fork_choice 수행)
```

올인원(bringup+자율, 끄면 Gazebo도 같이 종료):

```bash
ros2 launch dracer_sim sim_auto_driving.launch.py route_mode:=out spawn_pose:=out_fork
```

수동 텔레포트만:

```bash
./scripts/dev_container.sh teleport out_fork
./scripts/dev_container.sh teleport in_roundabout_exit
./scripts/dev_container.sh teleport out_fork_merge_left   # 합류 무시 검증
```

## 시나리오 ↔ 파이프라인 ID

| scenario | 파이프라인 | spawn |
|----------|------------|-------|
| `out_left` | O2 | `out_fork` |
| `out_right` | O3 | `out_fork` |
| `in_exit_left` | I4 | `in_roundabout_exit` |
| `in_exit_right` | I5 | `in_roundabout_exit` |

합류(O4–O5, M1–M2)는 아직 오프라인 시나리오 없음 — [fork-test-pipeline.md](../../docs/fork-test-pipeline.md) §3.3 라이브 spawn.
