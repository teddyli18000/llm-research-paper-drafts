# Paper 01 — α2 class-projected Ember at V16k: three fresh seeds completed

**Completed three V16k jobs of [Actions #44017422802](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017422802), all successful.** Model seeds 137,149,163 were not part of the V6k projector/α selection grid or the earlier V16k modern-optimizer experiments. Frozen [protocol](PROJECTOR_SCALE_CONFIRMATION_PROTOCOL.md): original public FineWeb train/val, effective V≈16000, width128, two layers/four heads, batch32, L64, 400 updates, head Ember LR.05/β₂=.95, normalized projector α2, K64 online groups every50 steps. All results from finished raw job logs.

## Four matched methods (CE, lower better)

| Method | Seed137 | Seed149 | Seed163 | Mean CE |
|---|---:|---:|---:|---:|
| Official Ember head-only, plain | 5.8246 | 5.8288 | 5.8329 | 5.82877 |
| Upstream online shared centroids + Ember | 5.7996 | 5.8127 | 5.8085 | 5.80693 |
| Original online centroids + Adam | 5.7809 | 5.7883 | 5.7743 | 5.78117 |
| **Our single-head normalized projector α2 + Ember** | **5.7534** | **5.7593** | **5.7615** | **5.75807** |

All three independent seeds favor the projector against **every** registered comparator:
- Gain vs standalone official Ember: **+0.0712,+0.0695,+0.0714 CE**; mean **+0.07070**.
- Gain vs original author's **centroid+Ember**: **+0.0462,+0.0534,+0.0470**; mean **+0.04887**.
- Gain vs original author's stronger **centroid+Adam**: **+0.0275,+0.0290,+0.0128**; mean **+0.02310**.

This is substantially larger than earlier small-run host variation and than the V6k pilot's near-tie. On these specific V16k hyperparameters and data, the *different* normalized one-matrix projected gradient method is genuinely stronger than reusing the original author's split-parameter method.

**Interpretation pending remaining studies:** still a small width128 / two-layer model, with short L64 train sequences and no modern large-batch guarantee. The [batch128 equal-token study](PROJECTOR_BATCH_SCALING_RESULTS.md) has already shown the α2 dynamic projector fails at batch128/V6k, so this is not blanket scaling evidence. Moreover, a separate V6k [causal ablation](PROJECTOR_CAUSAL_ABLATION_RESULTS.md) found static train-context groups outperform dynamic KMeans; whether that persists at V16k is now being tested in [contextual transfer](CONTEXTUAL_PROJECTOR_TRANSFER_PROTOCOL.md).

**Next check:** the three V50k jobs in the same pre-registered Action, plus external shard002 heldout and group-context alternatives. Do not extrapolate a top-conference methods result until those agree.
