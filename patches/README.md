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

## camera-v4l2-controls.patch

The C920e ran in **full auto-exposure + auto-AWB + autofocus**, so scene
brightness drifted frame to frame (measured V-mean **82–132** across one bag),
and `exposure_auto_priority=1` let the camera **drop below 30fps** in dim light —
straight latency into the control loop. No fixed threshold survives that.

Setting these via `cap.set(cv2.CAP_PROP_*)` is a **silent no-op**: the node opens
the camera through `cv2.CAP_GSTREAMER`, not the V4L2 backend. The controls have to
ride on the `v4l2src` element itself, so the patch adds a `build_v4l2_source()`
helper and a `v4l2_extra_controls` ROS param (default disables all four autos).
With them applied, V-mean held at **113 ± 0.1** on this board.

**`exposure_absolute` is intentionally not set.** This board's C920e rejects every
value (EILSEQ, verified across 3–2047), so exposure can only be *frozen* where the
camera's own AE leaves it (156), not chosen. To trim brightness, add `gain=<0..255>`
to the param (`gain` *is* writable here); do not add `exposure_absolute`.

## control-steer-invert.patch

The SSOT steering contract is **`+steering = right`** (perception emits `+y = left`,
`pipeline._pure_pursuit` negates so a left target → negative steering → left; see
`docs/vehicle-geometry.md`). The whole software chain is internally consistent with
that. But **this board's steering servo is wired the opposite way** — a `+` command
turns the wheels **left** — so every auto command drives the car the wrong way on
curves. It is invisible on straights (target ≈ 0), which is why it hid behind the
centerline work.

`D3Racer.set_steering_percent` is a plain linear map (`pulse = center + p*span`) with
no direction flag, so nothing in code corrects this. The patch adds a
`load_steer_invert()` read of **`STEER_INVERT`** (from `vehicle_config.yaml`, default
`false`) and, on the **AUTO `/control` path only**, reflects the command about the
mechanical centre: `steering = 2*STEER_TRIM - steering`. This flips left/right while
**preserving the straight-ahead trim** (`STEER_TRIM = 0.1` here), and leaves the
manual joystick path untouched. Sim and correctly-wired boards omit the key → no change.

Set `STEER_INVERT: true` in `config/vehicle_config.yaml` on this board. Re-verify on a
stand after applying: `steering=+1.0` on `/control` must now point the wheels **right**,
and `steering=0` must stay centred.
