"""Fork *judgment* — when to arm perception and which L/R pass to take.

Course split (SSOT):

* **OUT ``out_fork``:** arm when **turn-sign window OR** ``out_fork_capture``
  (tip+stretch). Confirmed camera sign picks L/R; on sign miss competition
  uses ``default_out_branch_rank`` (RIGHT=1). Strict AND mode available via
  ``require_sign=true`` + ``require_capture=true``.
* **IN ``in_exit``:** ``in_circle_fork_moment`` alone (no sign) arms keep/exit
  selection. Pass policy: **1st rising → right (rank 1) = circle keep**;
  **2nd rising → left (rank 0) = exit**. Enables legacy fork follow ON for
  that choice (geometry still from ``yellow_alt`` / pairs).

Debounce (K frames) lives in the planner ``RisingEventCounter``; this module
is pure per-event / per-frame policy.

Docs: ``docs/lane-occlusion-fork-strategy.md`` §5.1.4 ·
``docs/out-ego-fork-shape.md``
"""

from __future__ import annotations

from dataclasses import dataclass

# lateral_rank: 0 = left, 1 = right (same as MainPlanner._lock_fork_selection)
RANK_LEFT_EXIT = 0
RANK_RIGHT_KEEP = 1


@dataclass(frozen=True)
class OutForkArm:
    """Whether OUT detect() should publish fork geometry this frame."""

    sign_window: bool
    capture: bool
    require_sign: bool
    require_capture: bool
    arm: bool
    reason: str


def decide_out_fork_arm(
    *,
    sign_window: bool,
    capture: bool,
    require_sign: bool = True,
    require_capture: bool = True,
    force_mid_manoeuvre: bool = False,
) -> OutForkArm:
    """OUT arm policy.

    * Strict (both required): sign_window AND capture.
    * Race default (``require_sign=False``, ``require_capture=True``):
      arm on **sign_window OR capture** so a confirmed turn sign can open
      fork perception before tip/stretch, and capture still arms on sign miss.
    * Sign-only / capture-only: that gate alone.
    """

    if force_mid_manoeuvre:
        return OutForkArm(
            sign_window=sign_window,
            capture=capture,
            require_sign=require_sign,
            require_capture=require_capture,
            arm=True,
            reason='mid_manoeuvre',
        )

    if require_sign and require_capture:
        arm = bool(sign_window and capture)
        reason = (
            'sign_and_capture'
            if arm
            else (
                'wait_sign_and_capture'
                if (not sign_window and not capture)
                else ('wait_sign' if not sign_window else 'wait_capture')
            )
        )
    elif require_sign:
        arm = bool(sign_window)
        reason = 'sign' if arm else 'wait_sign'
    elif require_capture:
        # Race: sign alone OR capture (sign miss → capture / default rank).
        arm = bool(sign_window or capture)
        if arm and sign_window and capture:
            reason = 'sign_or_capture'
        elif arm and sign_window:
            reason = 'sign'
        elif arm:
            reason = 'capture'
        else:
            reason = 'wait_sign_or_capture'
    else:
        arm = True
        reason = 'open'

    return OutForkArm(
        sign_window=sign_window,
        capture=capture,
        require_sign=require_sign,
        require_capture=require_capture,
        arm=arm,
        reason=reason,
    )


@dataclass(frozen=True)
class InExitPass:
    """IN keep/exit choice after one debounced moment rising edge."""

    pass_index: int
    select_rank: int
    wants_exit: bool
    reason: str


def decide_in_exit_pass(
    previous_pass_count: int,
    *,
    keep_passes: int = 1,
    keep_rank: int = RANK_RIGHT_KEEP,
    exit_rank: int = RANK_LEFT_EXIT,
) -> InExitPass:
    """Map N-th moment rising → keep (right) or exit (left).

    ``previous_pass_count`` = completed risings before this one (0 at circle entry).
    After this call the planner should store ``pass_index`` as the new count.
    """

    keep_n = max(0, int(keep_passes))
    pass_index = int(previous_pass_count) + 1
    if pass_index <= keep_n:
        return InExitPass(
            pass_index=pass_index,
            select_rank=int(keep_rank),
            wants_exit=False,
            reason=f'in_keep_pass{pass_index}_rank{int(keep_rank)}',
        )
    return InExitPass(
        pass_index=pass_index,
        select_rank=int(exit_rank),
        wants_exit=True,
        reason=f'in_exit_pass{pass_index}_rank{int(exit_rank)}',
    )


def in_wants_exit_from_passes(
    pass_count: int,
    *,
    keep_passes: int = 1,
) -> bool:
    """True once keep passes are done (ready to take exit on next arm)."""

    return int(pass_count) > max(0, int(keep_passes))
