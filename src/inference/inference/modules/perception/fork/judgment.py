"""Fork *judgment* — when to arm perception and which L/R pass to take.

Course split (SSOT):

* **OUT ``out_fork``:** camera turn **sign** AND ``out_fork_capture`` (tip+stretch)
  together arm fork perception / ``FORK_TURN``. Sign picks L/R rank.
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
    """OUT arm = (sign if required) AND (capture if required), unless mid-turn."""

    if force_mid_manoeuvre:
        return OutForkArm(
            sign_window=sign_window,
            capture=capture,
            require_sign=require_sign,
            require_capture=require_capture,
            arm=True,
            reason='mid_manoeuvre',
        )
    ok_sign = (not require_sign) or sign_window
    ok_cap = (not require_capture) or capture
    arm = bool(ok_sign and ok_cap)
    if arm:
        reason = 'sign_and_capture' if (require_sign and require_capture) else (
            'sign' if require_sign else ('capture' if require_capture else 'open')
        )
    elif not ok_sign and not ok_cap:
        reason = 'wait_sign_and_capture'
    elif not ok_sign:
        reason = 'wait_sign'
    else:
        reason = 'wait_capture'
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
