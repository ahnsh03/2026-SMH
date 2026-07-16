"""Unit tests for OUT arm / IN exit-pass judgment."""

from __future__ import annotations

from inference.modules.perception.fork.judgment import (
    RANK_LEFT_EXIT,
    RANK_RIGHT_KEEP,
    decide_in_exit_pass,
    decide_out_fork_arm,
    in_wants_exit_from_passes,
)


def test_out_arm_requires_sign_and_capture():
    d = decide_out_fork_arm(
        sign_window=True, capture=True, require_sign=True, require_capture=True
    )
    assert d.arm is True
    assert d.reason == 'sign_and_capture'


def test_out_arm_blocks_sign_only_or_capture_only():
    assert (
        decide_out_fork_arm(
            sign_window=True, capture=False, require_sign=True, require_capture=True
        ).arm
        is False
    )
    assert (
        decide_out_fork_arm(
            sign_window=False, capture=True, require_sign=True, require_capture=True
        ).arm
        is False
    )


def test_out_arm_mid_manoeuvre_forced():
    d = decide_out_fork_arm(
        sign_window=False,
        capture=False,
        require_sign=True,
        require_capture=True,
        force_mid_manoeuvre=True,
    )
    assert d.arm is True
    assert d.reason == 'mid_manoeuvre'


def test_out_arm_race_sign_or_capture():
    """Race YAML: require_sign=False, require_capture=True → OR."""
    assert (
        decide_out_fork_arm(
            sign_window=True, capture=False, require_sign=False, require_capture=True
        ).arm
        is True
    )
    assert (
        decide_out_fork_arm(
            sign_window=False, capture=True, require_sign=False, require_capture=True
        ).arm
        is True
    )
    assert (
        decide_out_fork_arm(
            sign_window=False, capture=False, require_sign=False, require_capture=True
        ).arm
        is False
    )
    both = decide_out_fork_arm(
        sign_window=True, capture=True, require_sign=False, require_capture=True
    )
    assert both.arm is True
    assert both.reason == 'sign_or_capture'


def test_in_exit_pass_keep_then_exit():
    p1 = decide_in_exit_pass(0, keep_passes=1)
    assert p1.pass_index == 1
    assert p1.select_rank == RANK_RIGHT_KEEP
    assert p1.wants_exit is False

    p2 = decide_in_exit_pass(1, keep_passes=1)
    assert p2.pass_index == 2
    assert p2.select_rank == RANK_LEFT_EXIT
    assert p2.wants_exit is True

    assert in_wants_exit_from_passes(1, keep_passes=1) is False
    assert in_wants_exit_from_passes(2, keep_passes=1) is True
