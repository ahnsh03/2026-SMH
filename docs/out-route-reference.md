# OUT 코스 로봇 보조지표 (직선 · 코너 · S자)

> **용도:** 시뮬/실차에서 **직선·코너·S자** 추종 품질을 같은 기준으로 평가할 때 쓰는 맵 레퍼런스.  
> **위치:** 런타임 캡처 `data/captures/out_best_route/` (및 monorepo `data/out_best_route/`).  
> **SSOT 요약:** 그 폴더의 [`REFERENCE.md`](../../data/out_best_route/REFERENCE.md) — 좌표는 `reference_segments.json`.

추출 소스: `track_cw_real.png` · 스케일 SSOT [`track_plane.yaml`](../src/dracer_sim/config/track_plane.yaml) (12.0 × 8.9975 m, 원점=plane 중심).

---

## 1. 언제 쓰나

| 질문 | 존 |
|------|-----|
| 직진이 흔들리지 않나? | `eval_straight_*` |
| 단일 코너에서 중앙을 지키나? | `eval_corner_*` |
| S자에서 좌우 전환이 안정적인가? | `eval_s_curve` |

전 랩 웨이포인트(`out_best_waypoints.json`)는 **참고만**. 채점은 아래 존만.

---

## 2. 평가 존

| zone_id | 기동 | wp | ≈길이 | 비고 |
|---------|------|-----|-------|------|
| `eval_s_curve` | S자 | 24–72 | 9.5 m | 좌측 웨이브 |
| `eval_straight_top` | 직선 | 124–132 | 1.6 m | 상단 동진 |
| `eval_corner_ne` | 코너 | 133–144 | 2.2 m | 동→남 |
| `eval_straight_east` | 직선 | 145–167 | 4.4 m | 우측 남진 · 빨간 구간 포함 |
| `eval_corner_se` | 코너 | 168–179 | 2.2 m | 남→서 |
| `eval_straight_bottom` | 직선 | 180…16 wrap | 3.9 m | start 라인 · **wp16까지만** |

JSON 경로: `evaluation.zones[]` · CSV: `reference_eval_zones.csv`.

```text
maneuver_summary:
  straight → eval_straight_top | eval_straight_east | eval_straight_bottom
  corner   → eval_corner_ne | eval_corner_se
  s_curve  → eval_s_curve
```

---

## 3. 채점

1. zone polyline `(x_m, y_m)`에 로봇 `(x,y)` 투영 → CTE, heading error.  
2. 구간 통계 vs `pass_criteria`:

| 기동 | CTE RMS | \|CTE\| max | heading RMS |
|------|---------|-------------|-------------|
| 직선 | ≤ 0.04 m | ≤ 0.08 m | ≤ 0.12 rad |
| 코너 | ≤ 0.06 m | ≤ 0.12 m | ≤ 0.30 rad |
| S자 | ≤ 0.06 m | ≤ 0.12 m | ≤ 0.25 rad |

S자는 추가로 **곡률 부호 전환을 따라가며** 내측 컷/차로 이탈이 없어야 함.

스폰 힌트:

```bash
./scripts/dev_container.sh teleport start      # eval_straight_bottom
./scripts/dev_container.sh teleport obstacle   # eval_straight_east 인근
# 其餘: spawn_pose:=custom + zone entry_xy (REFERENCE.md / JSON spawn_hint)
```

---

## 4. 제외 구간

| ID | wp | 이유 |
|----|-----|------|
| dash_merge_jump | ≈18–22 | 점선 merge 튀김 |
| floor_text_pull | 76, 80 | 바닥 글자 끌림 |
| fork_collapsed | ≈88–112 | 갈림 병합 실패 |

시각 확인: `reference_segments_overlay.png` (녹색=trusted, 주황 X=bad).

---

## 5. 관련

| 자료 | 링크 |
|------|------|
| 상세 README | [`data/out_best_route/REFERENCE.md`](../../data/out_best_route/REFERENCE.md) (또는 captures 동일 파일명) |
| IN 회전교차로 (노란) | `data/captures/in_roundabout_route/` · [`extract_in_roundabout_waypoints.py`](../scripts/extract_in_roundabout_waypoints.py) |
| track 스케일 | [track_plane.yaml](../src/dracer_sim/config/track_plane.yaml) |
| spawn | [spawn_poses.yaml](../src/dracer_sim/config/spawn_poses.yaml) |
| 추출/내보내기 | [`scripts/export_out_reference_segments.py`](../scripts/export_out_reference_segments.py) |
| 플래너 | [main-planner.md](./main-planner.md) |
