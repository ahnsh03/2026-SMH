"""Shared pytest fixtures for board-race feature unit tests.

Adds ``src/inference`` to ``sys.path`` so tests run without an ament overlay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INFERENCE_SRC = Path(__file__).resolve().parents[1]
if str(_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_SRC))


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / 'config' / 'main_planner.yaml').is_file():
            return cand
    return here.parents[3]


@pytest.fixture(scope='session')
def inference_src() -> Path:
    return _INFERENCE_SRC


@pytest.fixture(scope='session')
def repo_root() -> Path:
    return _find_repo_root()


@pytest.fixture(scope='session')
def planner_config_path(repo_root: Path) -> Path:
    return repo_root / 'config' / 'main_planner.yaml'
