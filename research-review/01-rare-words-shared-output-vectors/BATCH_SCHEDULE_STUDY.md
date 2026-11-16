# Independent batch-size and re-clustering schedule experiment (paper 01)

Date: 2026-11-16. [Verified successful Actions #43834707881](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43834707881). Original `grid.py` was executed via a research-only instrumented copy. Same public FineWeb train/val, reconstructed vocabulary counts, model configuration, head-only epsilon=1e-5, and **819,200 total read tokens** across all batch conditions. Two independent model-init seeds 0 and 1.

## Results: gain in validation CE relative to epsilon-only baseline (higher is better)

| Batch (sequences × 64 tokens) | Train steps | Every 50 **steps** (seed0 / seed1, mean) | Every ~102400 **tokens** (seed0 / seed1, mean) | Recluster count for token-aligned |
|---:|---:|---:|---:|---:|
| 16 | 800 | 0.0178 / 0.0220, **0.01990** | 0.0227 / 0.0242, **0.02345** | 7 |
| 32 | 400 | 0.0208 / 0.0149, **0.01785** | same schedule | 7 |
| 64 | 200 | 0.0115 / 0.0150, **0.01325** | 0.0133 / 0.0163, **0.01480** | 7 |
| 128 | 100 | 0.0047 / 0.0024, **0.00355** | 0.0079 / 0.0090, **0.00845** | 8 |

The standard constant-50-step schedule produces 15, 7, 3, 1 re-cluster events for batch 16, 32, 64, 128. Token-aligned scheduling changes the event counts to roughly 7–8 for all batches.

**Finding:** A **fixed token-based reclustering interval mitigates** the fall-off at large batches and, interestingly, helps batch16 by avoiding too frequent reassignment. At batch128 the mean improvement rises from **0.00355 → 0.00845** when clustering is tied to tokens rather than steps (about 2.38× relative improvement). This isolates reclustering cadence as a contributing factor.

**Not all degradation disappears:** even after equalizing group-refreshes by *tokens*, gain declines from 0.02345 (batch16) to 0.00845 (batch128), while total optimization steps fall 800 → 100. A **matched-step** large-batch experiment has therefore been launched separately to disentangle batch size vs number of optimizer updates. The result will determine whether large batch genuinely hurts group sharing or whether the earlier fixed-token comparison only examines severely undertrained runs.

Data and run artifacts are at the linked CI run; numbers above are direct outputs, not predicted gains.
