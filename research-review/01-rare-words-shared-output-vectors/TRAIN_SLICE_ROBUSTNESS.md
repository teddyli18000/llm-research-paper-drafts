# Paper 01 — independent robustness across four FineWeb training slices

Date 2026-11-16. Verified [Actions #43851693792, success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43851693792).

## Controlled setup

- Reconstructed GPT2-tokenized FineWeb first 100M-token public shard; `counts.npy` and validation held fixed, training slice shifted across four disjoint 819,200-token windows.
- Same Transformer, V6000, width32, 2 layers, 400 steps, batch32, head-only Adam epsilon 1e-5.
- Same two model initialization seeds for baseline and dynamic grouping **within each data window**. Online clusters recomputed with KMeans every 50 steps; no source project code edited.
- Dataset offsets are 0, 10,000,000, 40,000,000, 70,000,000 tokens in the train shard.

## Independent results

| Training offset | Baseline CE seed0/seed1 | Dynamic CE seed0/seed1 | Paired gain seed0/seed1 | Mean gain |
|---|---|---|---|---:|
| 0 | 5.0139 / 5.0118 | 4.9932 / 4.9965 | 0.0207 / 0.0153 | **0.01800** |
| 10M | 4.9685 / 4.9759 | 4.9569 / 4.9605 | 0.0116 / 0.0154 | **0.01350** |
| 40M | 4.9966 / 4.9970 | 4.9808 / 4.9807 | 0.0158 / 0.0163 | **0.01605** |
| 70M | 4.9877 / 4.9803 | 4.9737 / 4.9637 | 0.0140 / 0.0166 | **0.01530** |

**All 8 seed/slice pairs improved.** Cross-slice variation exists (0.0116 to 0.0207), but it does not reverse the effect. This makes the result less likely to be an artifact of the particular first 819k training tokens or one lucky parameter initialization. Different seeds share the same held-out validation shard; this is robustness across train sampling, not across validation domains.

These results are narrow but strengthen the empirical existence of the effect. Questions still open: large/global batches; full tokenizer; strong modern baselines; 100M+ parameter pretraining.
