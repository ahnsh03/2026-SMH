#!/usr/bin/env python3
"""Board entrypoint for traffic-light bag viz (delegates to main vision_tune).

Uses this tree's ``color_detector`` thresholds via PYTHONPATH, and the monorepo
``2026-SMH`` bag helpers / bags/ when present.

  # from 2026-SMH-board (ROS sourced)
  python3 scripts/viz_traffic_light.py --from-bag out_cam
  python3 scripts/viz_traffic_light.py --from-bag out --hits-only --no-gui \\
      --export-dir data/captures/traffic_light_viz/out
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_BOARD = Path(__file__).resolve().parents[1]
_MAIN = _BOARD.parent / '2026-SMH'
_MAIN_TUNE = _MAIN / 'scripts' / 'vision_tune'
_MAIN_INF = _MAIN / 'src' / 'inference'
_BOARD_INF = _BOARD / 'src' / 'inference'

# Prefer board inference (race thresholds); keep main vision_tune helpers on path.
for p in (_BOARD_INF, _MAIN_TUNE, _MAIN_INF):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)

target = _MAIN_TUNE / 'viz_traffic_light.py'
if not target.is_file():
    raise SystemExit(
        f'Missing {target}. Clone/checkout 2026-SMH next to this board tree, '
        'or run scripts/vision_tune/viz_traffic_light.py from 2026-SMH.'
    )

# Ensure capture_from_bag resolves bags under 2026-SMH (its parents[2]).
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name='__main__')
