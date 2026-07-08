"""Traffic light color submodule — split out from traffic_sign.py facade."""

from inference.modules.trafficsign.color_detector import detect_signal

__all__ = ['detect_signal']
