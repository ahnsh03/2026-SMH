"""Traffic light submodule — OpenCV HSV and optional Sungjun YOLO (lights only)."""

from inference.modules.trafficsign.color_detector import detect_signal as detect_signal_opencv
from inference.modules.trafficsign.yolo_light import detect_signal_yolo

# Backward-compatible alias (unit tests / old imports = OpenCV path).
detect_signal = detect_signal_opencv

__all__ = ['detect_signal', 'detect_signal_opencv', 'detect_signal_yolo']
