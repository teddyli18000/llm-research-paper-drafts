# Independent 3-seed confirmation — paper 01

Date: 2026-11-16. [Verified Actions run #43834412315](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43834412315), completed **success**.

## Setup

Original `grid.py` is copied and instrumented at runtime; upstream source untouched. V=6000, width=32, 2 Transformer blocks, 400 training steps, batch=32, context=64, seed ∈ {0,1,2}. Same train data, validation data, selected LR 0.1, head-only Adam epsilon `1e-5`, all other settings equal within each seed. Recluster interval=50 steps. Three seeds modify torch model initialization, not the dataset order. Run data was reconstructed from publicly available `kjj0/fineweb10B-gpt2`: `counts.npy` based on the first 100M-token shard because the author's original corpus-wide `counts.npy` is absent from the fork; hence the numerical losses differ from the author PDF, while relative comparisons here are matched.

## Raw validation cross-entropies (lower is better)

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean paired gain vs baseline |
|---|---:|---:|---:|---:|
| Tuned-epsilon baseline | 5.0139 | 5.0118 | 5.0091 | — |
| Dynamic clusters / keep Adam moments | **4.9937** | **4.9967** | **4.9925** | **0.01730**, sample SD 0.00262 |
| Dynamic clusters / reset cluster moments | **4.9938** | **4.9960** | **4.9907** | **0.01810**, SD 0.00217 |
| Dynamic / reset cluster + private moments | 5.0008 | 5.0014 | 4.9944 | 0.01273, SD 0.00217 |
| One-time cluster at step 50 | 5.0056 | 5.0097 | 5.0043 | 0.00507, SD 0.00311 |
| Fixed random group assignment | 5.0125 | 5.0142 | 5.0110 | -0.00097, SD 0.00206 |
| Fixed frequency group assignment | 5.0148 | 5.0243 | 5.0109 | -0.00507, SD 0.00645 |

All 21 arms ran to completion. Loss difference `baseline - method` is computed **within the same seed**. Standard deviations are sample SD of **three paired gains**, not confidence intervals.

## Mechanistic conclusions grounded in this experiment

1. Online learned grouping beats the fixed controls across all three model-initialization seeds, with per-seed gains **0.0202 / 0.0151 / 0.0166**. This supports a reproducible training effect, rather than a single lucky initialization.
2. Resetting the cluster Adam moments gives paired gains **0.0201 / 0.0158 / 0.0184**, essentially equal to or larger than retaining stale moments. The PDF–code disagreement about `reset_adam_state(C)` is real, but **the benefit is not explained solely by retaining outdated cluster moments**.
3. Resetting *both* cluster and private-word Adam moments lowers gains (mean **0.01273**), indicating private optimizer history contributes to training dynamics. Parameter-space reassignments are function-preserving but not optimizer-state invariant.
4. Fixed random/frequency grouping does not recover the gain, and a single matched learned recluster recovers only ~0.005. This supports **repeated, content-informed reassignment** as an important part of the mechanism.
5. All runs passed the exact `R_before=R_after` numerical check at each reclustering. The analytical gradient-coupling identity `ΔR = -(η_U I+η_C A Aᵀ)G` for SGD was also checked independently.

## Next scientific question

Does the effect survive a realistic large-batch optimizer regime, or is it mainly a small-step/rare-frequency training regime? This is being tested with (1) matched total tokens and cluster-update frequency, (2) **AMSGrad** as a modern rare-token optimizer comparison. Both probes have separately logged GitHub Actions workflows. No publication-quality extrapolation should be based on the three-seed 400-step setting alone.
