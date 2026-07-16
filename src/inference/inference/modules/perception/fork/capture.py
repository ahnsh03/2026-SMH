"""OUT fork *capture* — integrates white tip moment + ego-blob stretch.

Hierarchy (neither replaces the other):

* ``moment`` (``score_out_fork_moment``) — white+road **tip / approach**
* ``ego`` (``score_out_ego_fork_shape``) — ego_blob **Y-stretch** (out_cam
  label window **1280–1313** after camera retune)

Combined flags (per frame, no temporal debounce here):

* ``in_stretch``  = ego.hard
* ``tip``         = moment.hard
* ``tip_in_context`` = tip AND (ego.hard OR ego.soft)
* ``capture``     = in_stretch OR tip_in_context

Runtime should add K-frame debounce on ``capture`` / ``in_stretch``.
Sign / L-R entry is *not* here.

Docs: ``docs/out-ego-fork-shape.md`` · ``docs/fork-moment-detection.md`` §3.5
· ``docs/lane-occlusion-fork-strategy.md`` §5.1.3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference.modules.perception.fork.ego_shape import (
    OutEgoForkShape,
    score_out_ego_fork_shape,
)
from inference.modules.perception.fork.moment import (
    OutForkMoment,
    score_out_fork_moment,
)


@dataclass(frozen=True)
class OutForkCapture:
    """Integrated OUT fork presence for one BEV frame."""

    moment: OutForkMoment
    ego: OutEgoForkShape
    in_stretch: bool
    tip: bool
    tip_in_context: bool
    capture: bool


def score_out_fork_capture(
    white: np.ndarray,
    road: np.ndarray,
    ego_blob: np.ndarray,
) -> OutForkCapture:
    """Score tip (white+road) and stretch (ego_blob); fuse to ``capture``."""

    moment = score_out_fork_moment(white, road)
    ego = score_out_ego_fork_shape(ego_blob)
    in_stretch = bool(ego.hard)
    tip = bool(moment.hard)
    tip_in_context = tip and (bool(ego.hard) or bool(ego.soft))
    capture = in_stretch or tip_in_context
    return OutForkCapture(
        moment=moment,
        ego=ego,
        in_stretch=in_stretch,
        tip=tip,
        tip_in_context=tip_in_context,
        capture=capture,
    )
