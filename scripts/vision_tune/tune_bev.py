#!/usr/bin/env python3
"""Default BEV tuner — Metric IPM (team SSOT).

Thin entry for tune_metric_ipm.py. Prefer this name in docs.

  python3 scripts/vision_tune/tune_bev.py
  python3 scripts/vision_tune/tune_bev.py --compare

Trapezoid (reference only): tune_bev_roi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tune_metric_ipm import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
