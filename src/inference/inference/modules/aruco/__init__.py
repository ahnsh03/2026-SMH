"""ArUco submodules — split to avoid merge conflicts between two assignees."""

from inference.modules.aruco.detector import detect_markers
from inference.modules.aruco.stop_logic import should_stop_for_markers

__all__ = ['detect_markers', 'should_stop_for_markers']
