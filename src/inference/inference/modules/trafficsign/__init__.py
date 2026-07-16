"""Traffic light color submodule — split out from traffic_sign.py facade."""

from inference.modules.trafficsign.color_detector import detect_signal
from inference.modules.trafficsign.debounce import debounce_signal

__all__ = ['detect_signal', 'debounce_signal']
