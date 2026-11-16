# Paper 01 — large batch at identical optimization-step count

Date: 2026-11-16. [GitHub Actions #43836943981 completed successfully](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43836943981).

## Motivation

The author's initial batch ablation held **training tokens constant**, so increasing batch size reduced gradient steps and reduced the number of periodic re-clustering operations. Our [token-aligned re-clustering study](BATCH_SCHEDULE_STUDY.md) showed that keeping re-clustering cadence constant by **tokens**, not steps, partly restores the benefit, but leaves an ambiguity: are larger batches worse per se, or only because training performs fewer parameter updates?

We therefore fixed **400 optimizer updates** while increasing batch to 128 sequences × 64 tokens, equivalent to 3,276,800 processed tokens. The original default batch32 × 400-step run processed 819,200 tokens. Same model, data, tokenization, epsilon, learning-rate schedule and two model-initialization seeds.

## Observations

| Seed | Epsilon-only baseline CE | Dynamic K-means every 50 steps | Gain | K-means every 12 steps | Gain |
|---|---:|---:|---:|---:|---:|
| 0 | 4.7535 | 4.7431 | +0.0104 | 4.7432 | +0.0103 |
| 1 | 4.7536 | 4.7468 | +0.0068 | 4.7449 | +0.0087 |
| **Mean** | **4.75355** | **4.74495** | **+0.00860** | **4.74405** | **+0.00950** |

Default batch32 × 400 steps yielded mean gain +0.01730 across seeds 0,1,2; the batch128 matched-400-step mean gain was +0.00860–0.00950 across two seeds. Thus even when holding optimizer update count fixed and feeding *four times as many examples*, the difference is smaller, though consistently positive.

Re-clustering 33 times instead of 7 times offered only a marginal difference at batch128. **Clustering cadence alone cannot explain the remaining batch effect**.

## Interpretation and next falsifier

The observed effect is a plausible property of sparse rare-token positive-target gradient updates and Adam normalization: large batches average many token events before each step and reduce the temporal coupling benefit from borrowed cluster gradient. However, these comparisons hold LR fixed; optimal head/shared LR could vary with batch, and the large-batch run consumed 4× tokens. There is no proof that the trend survives optimized learning rates or much longer schedules.

For publication, test at V32k+, real training global batches and at least one modern wider language model. If gains flatten under realistic batches, the method may still be useful in low-batch fine-tuning or online adaptation; that would be a narrower but important target regime. The mathematical mechanism and small-batch reproducibility remain intact.
