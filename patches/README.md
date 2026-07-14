# patches/

Fixes to **D-Racer-Kit** (the organizers' repo at `topst-development/D-Racer-Kit`).

We cannot push to that repo, and `scripts/init_workspace.sh` clones it fresh —
so any change we make to the kit would be silently lost. Patches here are
applied by `init_workspace.sh` on top of the untouched checkout, so the fix
survives a re-clone and is reviewable in our own history.

Applying is idempotent: an already-applied patch is skipped.

## camera-native-caps.patch

`camera_node.build_candidate_pipelines` left the GStreamer **source** caps as
`image/jpeg,framerate=30/1` — no width or height. `v4l2src` therefore negotiates
the camera's largest MJPG mode: measured **1920x1080** on our C920e. The board
then decodes a 1080p JPEG and CPU-downscales it to 320x180 on every frame.

Two problems:

1. **Wasted work.** The C920e offers `image/jpeg 320x180 @30fps` natively.
   Measured capture-pipeline CPU: **6.2% -> 4.4%** of one core. Modest, but free.
2. **Silent aspect-ratio corruption.** If the negotiated mode is 4:3 and the
   target is 16:9, `videoscale` squashes the image anamorphically, so
   `fy != fx`. `scripts/vision_tune/metric_ipm.py` assumes square pixels
   (`fy = fx`), so the IPM row-to-forward-distance mapping breaks and every
   metric threshold derived from `METERS_PER_PIXEL` goes with it. **This already
   bit us**: the 2026-07-08 bags were recorded at 320x240.

The patch pins the source caps to the configured `IMAGE_WIDTH`/`IMAGE_HEIGHT`
and keeps the old unpinned pipeline as a fallback, so a camera with no native
mode at that size still works.
