# `perception.fork`

갈림 관련 인지 모듈. **시점(moment)** 과 **갈래 geometry(legacy merge)** 를 함께 둔다.

| 모듈 | 역할 |
|------|------|
| [`moment.py`](./moment.py) | IN/OUT approach flags (`score_in_circle_fork_moment`, `score_out_fork_moment`) |
| [`ego_shape.py`](./ego_shape.py) | OUT ego-blob Y-stretch |
| [`capture.py`](./capture.py) | OUT tip+stretch fuse |
| [`judgment.py`](./judgment.py) | OUT sign∧capture arm · IN keep/exit pass policy |
| [`adapter.py`](./adapter.py) | 표지 게이트 시 legacy `fork_lane_pairs` / branches 를 blob 결과에 merge |

문서·데이터 라벨·임계 SSOT:

- [`docs/fork-moment-detection.md`](../../../../../../docs/fork-moment-detection.md) — 코드·데이터 보관
- [`docs/lane-occlusion-fork-strategy.md`](../../../../../../docs/lane-occlusion-fork-strategy.md) §5.1.2–5.1.3

오프라인:

```bash
PYTHONPATH=scripts/vision_tune:src/inference python3 scripts/vision_tune/score_in_fork_moment.py
PYTHONPATH=scripts/vision_tune:src/inference python3 scripts/vision_tune/score_out_fork_moment.py
```
