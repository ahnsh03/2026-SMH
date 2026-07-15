#!/usr/bin/env python3
"""Export trusted OUT centerline segments + robot eval zones (straight/corner/S)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path("/workspace/data/captures/out_best_route")
IMG = Path(
    "/workspace/src/dracer_sim/models/track_plane/materials/textures/track_cw_real.png"
)
WIDTH_M = 12.0
HEIGHT_M = 8.9975

TRUSTED = [
    {
        "id": "west_scurve",
        "label": "좌측 S자 (연속 코너)",
        "wp_range": [24, 72],
        "wrap": False,
        "role": "s_curve",
        "eval_zones": ["eval_s_curve"],
        "note": "S자 평가 레퍼런스. eval_s_curve 사용.",
    },
    {
        "id": "east_loop_close",
        "label": "상단→우직진→하단 복귀(직선+코너 묶음)",
        "wp_range": [124, 16],
        "wrap": True,
        "role": "straight_and_corner_bundle",
        "eval_zones": [
            "eval_straight_top",
            "eval_corner_ne",
            "eval_straight_east",
            "eval_corner_se",
            "eval_straight_bottom",
        ],
        "note": "직선·단일코너 평가용으로 evaluation.zones로 쪼개 사용.",
    },
]

EVAL_ZONES = [
    {
        "id": "eval_s_curve",
        "maneuver": "s_curve",
        "label": "좌측 S자 (연속 좌우 코너)",
        "wp_range": [24, 72],
        "wrap": False,
        "parent_segment": "west_scurve",
        "spawn_hint": "near (-2.08, -3.91) heading north after bottom-left",
        "pass_criteria": {
            "cte_rms_m_max": 0.06,
            "cte_abs_max_m": 0.12,
            "heading_err_rms_rad_max": 0.25,
            "notes": "곡률 부호 전환을 따라가며 차선 중앙 유지. 내측 컷/외측 이탈 없이 S 파형 유지.",
        },
    },
    {
        "id": "eval_straight_top",
        "maneuver": "straight",
        "label": "상단 직선 (동진)",
        "wp_range": [124, 132],
        "wrap": False,
        "parent_segment": "east_loop_close",
        "spawn_hint": "near (1.73, 3.72) heading +X",
        "pass_criteria": {
            "cte_rms_m_max": 0.04,
            "cte_abs_max_m": 0.08,
            "heading_err_rms_rad_max": 0.12,
            "notes": "요동 없이 직선 추종. 조향 거의 0에 수렴.",
        },
    },
    {
        "id": "eval_corner_ne",
        "maneuver": "corner",
        "label": "우상단 코너 (동→남)",
        "wp_range": [133, 144],
        "wrap": False,
        "parent_segment": "east_loop_close",
        "spawn_hint": "near (3.53, 3.72) heading +X into corner",
        "pass_criteria": {
            "cte_rms_m_max": 0.06,
            "cte_abs_max_m": 0.12,
            "heading_err_rms_rad_max": 0.30,
            "notes": "단일 코너. 진입·정점·탈출 CTE 급증 없이 원호 추종.",
        },
    },
    {
        "id": "eval_straight_east",
        "maneuver": "straight",
        "label": "우측 직선 (남진, 빨간 구간 포함)",
        "wp_range": [145, 167],
        "wrap": False,
        "parent_segment": "east_loop_close",
        "spawn_hint": "teleport obstacle 또는 (4.86, 2.06) heading -Y",
        "pass_criteria": {
            "cte_rms_m_max": 0.04,
            "cte_abs_max_m": 0.08,
            "heading_err_rms_rad_max": 0.12,
            "notes": "긴 직선. 빨간 차로 페인트에 치우치지 말고 중앙 유지.",
        },
    },
    {
        "id": "eval_corner_se",
        "maneuver": "corner",
        "label": "우하단 코너 (남→서)",
        "wp_range": [168, 179],
        "wrap": False,
        "parent_segment": "east_loop_close",
        "spawn_hint": "near (4.86, -2.54) heading -Y into corner",
        "pass_criteria": {
            "cte_rms_m_max": 0.06,
            "cte_abs_max_m": 0.12,
            "heading_err_rms_rad_max": 0.30,
            "notes": "단일 코너 후 하단 직선으로 안정 진입.",
        },
    },
    {
        "id": "eval_straight_bottom",
        "maneuver": "straight",
        "label": "하단 직선 (서진, start 라인 통과)",
        "wp_range": [180, 16],
        "wrap": True,
        "parent_segment": "east_loop_close",
        "spawn_hint": "teleport start · heading -X (yaw≈-π)",
        "pass_criteria": {
            "cte_rms_m_max": 0.04,
            "cte_abs_max_m": 0.08,
            "heading_err_rms_rad_max": 0.12,
            "notes": "wp16까지만. wp18–22 점선/갈림 오염 구간은 평가 제외.",
        },
    },
]

FAILED = [
    {
        "id": "dash_merge_jump",
        "wp_approx": [18, 22],
        "symptom": "점선(merge dash) 미인식으로 경로 튀김",
        "example_wp": 20,
    },
    {
        "id": "fork_collapsed",
        "wp_approx": [88, 112],
        "symptom": "out_fork 두갈래가 ribbon 병합으로 한 줄로 뭉침",
        "example_wp": 107,
    },
    {
        "id": "floor_text_pull",
        "wp_approx": [76, 80],
        "symptom": "트랙 바닥 글자/로고(VOLKSWAGEN GROUP KOREA 등)에 끌림",
        "example_wp": [76, 80],
    },
]


def expand_range(lo: int, hi: int, wrap: bool, n: int) -> list[int]:
    if not wrap:
        return list(range(lo, hi + 1))
    return list(range(lo, n)) + list(range(0, hi + 1))


def poly_length_m(wps: list, idxs: list[int]) -> float:
    length = 0.0
    for a, b in zip(idxs, idxs[1:]):
        length += math.hypot(
            wps[b]["x_m"] - wps[a]["x_m"], wps[b]["y_m"] - wps[a]["y_m"]
        )
    return round(length, 3)


def main() -> int:
    d = json.loads((OUT / "out_best_waypoints.json").read_text(encoding="utf-8"))
    wps = d["waypoints"]
    n = len(wps)
    img = cv2.imread(str(IMG))
    if img is None:
        raise SystemExit(f"failed to read {IMG}")

    segments = []
    for seg in TRUSTED:
        lo, hi = seg["wp_range"]
        idxs = expand_range(lo, hi, seg["wrap"], n)
        pts = [
            {k: wps[i][k] for k in ("i", "s_m", "x_m", "y_m", "u_px", "v_px")}
            for i in idxs
        ]
        segments.append(
            {
                **seg,
                "num_waypoints": len(pts),
                "length_m": poly_length_m(wps, idxs),
                "s_start_m": pts[0]["s_m"],
                "s_end_m": pts[-1]["s_m"],
                "waypoints": pts,
            }
        )

    zones_out = []
    for z in EVAL_ZONES:
        lo, hi = z["wp_range"]
        idxs = expand_range(lo, hi, z["wrap"], n)
        pts = [
            {k: wps[i][k] for k in ("i", "s_m", "x_m", "y_m", "u_px", "v_px")}
            for i in idxs
        ]
        zones_out.append(
            {
                **{k: z[k] for k in z if k != "pass_criteria"},
                "num_waypoints": len(pts),
                "length_m": poly_length_m(wps, idxs),
                "entry_xy": [pts[0]["x_m"], pts[0]["y_m"]],
                "exit_xy": [pts[-1]["x_m"], pts[-1]["y_m"]],
                "pass_criteria": z["pass_criteria"],
                "waypoints": pts,
            }
        )

    failures = []
    for fail in FAILED:
        ex = fail["example_wp"]
        ids = ex if isinstance(ex, list) else [ex]
        failures.append(
            {
                **fail,
                "examples": [
                    {
                        "i": i,
                        "x_m": wps[i]["x_m"],
                        "y_m": wps[i]["y_m"],
                        "u_px": wps[i]["u_px"],
                        "v_px": wps[i]["v_px"],
                    }
                    for i in ids
                ],
            }
        )

    ref = {
        "source": "out_best_waypoints.json",
        "status": "partial_reference",
        "purpose": (
            "로봇 주행 QA 보조지표: 직선·코너·S자별로 신뢰 센터라인과 합격 기준을 제공. "
            "전 랩 맵이 아니라 trusted 구간만 사용."
        ),
        "quality_policy": {
            "trusted": "evaluation.zones 에 명시된 wp 구간만 보조지표로 사용",
            "untrusted": "그 외 전 구간 — 점선/갈림 병합/글자 오염 가능성",
        },
        "trusted_segments": segments,
        "known_failures": failures,
        "evaluation": {
            "frame": "gazebo_world / track_plane centered",
            "reference_polyline": "eval_zones[*].waypoints (x_m,y_m)",
            "how_to_score": [
                "주행 궤적 (x,y)를 해당 zone polyline에 투영해 CTE(횡오차) 계산",
                "투영점 접선 대비 heading error 계산",
                "zone 구간 통과 중 CTE RMS / |CTE|max / heading RMS를 pass_criteria와 비교",
                "S자는 추가로 내측 컷·차로 이탈 없이 곡률 부호 전환을 따라가는지 확인",
            ],
            "maneuver_summary": {
                "straight": [
                    "eval_straight_top",
                    "eval_straight_east",
                    "eval_straight_bottom",
                ],
                "corner": ["eval_corner_ne", "eval_corner_se"],
                "s_curve": ["eval_s_curve"],
            },
            "zones": zones_out,
        },
        "gazebo_frame": d["meta"].get("frame"),
        "gazebo_track_plane_m": d["meta"]["gazebo_track_plane_m"],
        "meters_per_pixel": d["meta"]["meters_per_pixel"],
        "spacing_m": d["meta"]["spacing_m"],
    }
    (OUT / "reference_segments.json").write_text(
        json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (OUT / "reference_segments.csv").open("w", encoding="utf-8") as f:
        f.write("segment_id,role,i,s_m,x_m,y_m,u_px,v_px\n")
        for seg in segments:
            for wp in seg["waypoints"]:
                f.write(
                    f"{seg['id']},{seg['role']},{wp['i']},{wp['s_m']},"
                    f"{wp['x_m']},{wp['y_m']},{wp['u_px']},{wp['v_px']}\n"
                )

    with (OUT / "reference_eval_zones.csv").open("w", encoding="utf-8") as f:
        f.write("zone_id,maneuver,i,s_m,x_m,y_m\n")
        for z in zones_out:
            for wp in z["waypoints"]:
                f.write(
                    f"{z['id']},{z['maneuver']},{wp['i']},"
                    f"{wp['s_m']},{wp['x_m']},{wp['y_m']}\n"
                )

    def pix(u, v):
        return int(round(u)), int(round(v))

    ov = img.copy()
    for i in range(n):
        a, b = wps[i], wps[(i + 1) % n]
        cv2.line(
            ov,
            pix(a["u_px"], a["v_px"]),
            pix(b["u_px"], b["v_px"]),
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    # color by maneuver
    colors = {
        "s_curve": (0, 220, 80),
        "straight": (220, 180, 0),
        "corner": (0, 160, 255),
    }
    for z in zones_out:
        col = colors[z["maneuver"]]
        pts = z["waypoints"]
        for a, b in zip(pts, pts[1:]):
            cv2.line(
                ov,
                pix(a["u_px"], a["v_px"]),
                pix(b["u_px"], b["v_px"]),
                col,
                3,
                cv2.LINE_AA,
            )
        mid = pts[len(pts) // 2]
        p = pix(mid["u_px"], mid["v_px"])
        cv2.putText(
            ov,
            z["maneuver"],
            (p[0] + 4, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            col,
            1,
            cv2.LINE_AA,
        )

    fail_color = (0, 140, 255)
    for fail in failures:
        for ex in fail["examples"]:
            p = pix(ex["u_px"], ex["v_px"])
            cv2.drawMarker(ov, p, fail_color, cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(
                ov,
                f"BAD {ex['i']}",
                (p[0] + 8, p[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                fail_color,
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        ov,
        "EVAL: green=S-curve | cyan=corner | yellow=straight | orange X=do-not-use",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        ov,
        "BAD: ~20 dash | fork 88-112 | 76/80 floor text",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        fail_color,
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OUT / "reference_segments_overlay.png"), ov)

    plot_w, plot_h = 1200, 920
    plot = np.full((plot_h, plot_w, 3), 28, dtype=np.uint8)
    margin = 55

    def w2plot(x, y):
        px_ = margin + (x + WIDTH_M / 2) / WIDTH_M * (plot_w - 2 * margin)
        py_ = margin + (HEIGHT_M / 2 - y) / HEIGHT_M * (plot_h - 2 * margin)
        return int(round(px_)), int(round(py_))

    for gx in range(-6, 7):
        cv2.line(plot, w2plot(gx, -HEIGHT_M / 2), w2plot(gx, HEIGHT_M / 2), (55, 55, 55), 1)
    for gy in range(-4, 5):
        cv2.line(plot, w2plot(-WIDTH_M / 2, gy), w2plot(WIDTH_M / 2, gy), (55, 55, 55), 1)
    for i in range(n):
        a, b = wps[i], wps[(i + 1) % n]
        cv2.line(
            plot,
            w2plot(a["x_m"], a["y_m"]),
            w2plot(b["x_m"], b["y_m"]),
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
    for z in zones_out:
        col = colors[z["maneuver"]]
        pts = z["waypoints"]
        for a, b in zip(pts, pts[1:]):
            cv2.line(
                plot,
                w2plot(a["x_m"], a["y_m"]),
                w2plot(b["x_m"], b["y_m"]),
                col,
                3,
                cv2.LINE_AA,
            )
        p0 = w2plot(pts[0]["x_m"], pts[0]["y_m"])
        cv2.putText(
            plot,
            z["id"].replace("eval_", ""),
            (p0[0] + 4, p0[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            col,
            1,
        )
    for fail in failures:
        for ex in fail["examples"]:
            cv2.drawMarker(
                plot,
                w2plot(ex["x_m"], ex["y_m"]),
                fail_color,
                cv2.MARKER_TILTED_CROSS,
                14,
                2,
            )
    cv2.putText(
        plot,
        "Robot eval refs — green S | cyan corner | yellow straight",
        (margin, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OUT / "reference_segments_gazebo_m.png"), plot)

    # Keep REFERENCE.md concise pointer; full Korean SSOT may live in docs + data copy.
    md = Path("/workspace/docs/out-route-reference.md")
    summary = [
        "# OUT 코스 로봇 보조지표 — 직선 · 코너 · S자",
        "",
        "자세한 채점·합격선: 레포 [`docs/out-route-reference.md`](../../../docs/out-route-reference.md).",
        "",
        "## Eval zones",
        "",
        "| zone_id | 기동 | wp | 길이 |",
        "|---------|------|-----|------|",
    ]
    for z in zones_out:
        wr = " wrap" if z["wrap"] else ""
        summary.append(
            f"| `{z['id']}` | {z['maneuver']} | {z['wp_range'][0]}–{z['wp_range'][1]}{wr} | {z['length_m']} m |"
        )
    summary += [
        "",
        "## Pass criteria (default)",
        "",
        "| 기동 | CTE RMS | |CTE| max | heading RMS |",
        "|------|---------|-----------|-------------|",
        "| 직선 | ≤ 0.04 m | ≤ 0.08 m | ≤ 0.12 rad |",
        "| 코너 | ≤ 0.06 m | ≤ 0.12 m | ≤ 0.30 rad |",
        "| S자 | ≤ 0.06 m | ≤ 0.12 m | ≤ 0.25 rad |",
        "",
        "## Do not use",
        "",
    ]
    for fail in failures:
        summary.append(f"- **{fail['id']}** wp≈{fail['wp_approx']}: {fail['symptom']}")
    summary += [
        "",
        "## Files",
        "",
        "- `reference_segments.json` — `evaluation.zones` + pass_criteria",
        "- `reference_eval_zones.csv`",
        "- `reference_segments_overlay.png` / `reference_segments_gazebo_m.png`",
        "",
    ]
    (OUT / "REFERENCE.md").write_text("\n".join(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "zones": [
                    {
                        "id": z["id"],
                        "maneuver": z["maneuver"],
                        "wp_range": z["wp_range"],
                        "length_m": z["length_m"],
                    }
                    for z in zones_out
                ]
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"wrote {OUT}")
    # sync note for host data/ mirror if bind-mounted sibling is unavailable
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
