"""Shared OpenCV viz toggle for drive_test benches."""

from __future__ import annotations


def apply_lane_viz(mode: str | None) -> str:
    """Set ``lane_detection`` visualize mode.

    Modes
    -----
    off           — no OpenCV perception windows
    control/drive — ``Lane drive`` only (what control follows)
    on/all/debug  — ``Lane drive`` + ``HSV masks``
    """
    from inference.modules import lane_detection as ld

    raw = str(mode or 'control').strip().lower()
    if raw in ('off', 'none', '0', 'false'):
        ld.VISUALIZE_MODE = ld.VISUALIZE_OFF
        ld.VISUALIZE = False
        chosen = 'off'
    elif raw in ('on', 'all', 'full', '1', 'true'):
        ld.VISUALIZE_MODE = ld.VISUALIZE_ON
        ld.VISUALIZE = True
        chosen = 'on'
    else:
        # control | ctrl | drive | lane | debug → single driving canvas
        ld.VISUALIZE_MODE = ld.VISUALIZE_CONTROL
        ld.VISUALIZE = True
        chosen = 'control'

    for name in ('Lane drive', 'HSV masks', 'lane_control', 'road_branches'):
        try:
            import cv2

            # highgui logs ERROR if the window was never created; swallow quietly.
            if hasattr(cv2, 'getWindowProperty'):
                try:
                    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 0:
                        continue
                except Exception:
                    continue
            cv2.destroyWindow(name)
        except Exception:
            pass
    return chosen
