# Legacy polyfit / boundary-DP perception

**Reference only.** Do not add new driving logic here.

This is the former `modules/lane_detection.py` monolith (~7.7k lines):
HSV → Metric IPM → boundary DP + polyfit rails → fork lane pairs.

Runtime default is now **`perception.backend: blob`** (see `config/lane_vision.yaml`).
Keep `backend: legacy` only for A/B comparison and fork/rail tuners
(`scripts/vision_tune/tune_lane_detect.py`, fork sweeps, etc.).

Fork branch discrimination at signed forks is invoked from
`perception.fork.adapter`, which may call into this module when
`enable_fork=True`. Normal-lane driving must not use polyfit rails.
