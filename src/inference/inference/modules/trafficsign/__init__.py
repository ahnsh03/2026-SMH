"""Traffic light color submodule — split out from traffic_sign.py facade."""

from inference.modules.trafficsign.color_detector import (
    BlobInfo,
    SignalInspect,
    detect_signal,
    inspect_signal,
)

__all__ = ['BlobInfo', 'SignalInspect', 'detect_signal', 'inspect_signal']
