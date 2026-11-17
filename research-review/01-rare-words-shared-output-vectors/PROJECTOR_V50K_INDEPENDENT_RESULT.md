# Paper 01 — Independent V≈50k projected-Ember result (all three seeds completed)

**Verified:** [GitHub Actions #44017422802](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017422802) **success 6/6** (both V16k and V50k); this note reports all **V50k** jobs. Source logs, model seeds 137/149/163, preregistered [protocol](PROJECTOR_SCALE_CONFIRMATION_PROTOCOL.md).

Setup: original 2-layer width64/4-head tiny Transformer, almost full GPT-2 vocabulary (~50,257 mapped classes), FineWeb train prefix 1,638,400 tokens; batch32/sequence64/gradient accum4, 800 updates; online KMeans K64 refresh50, direct normalized gradient projector α2; official output-head Ember LR.05 β₂=.999; all other head/body/centroid controls fixed before training. Each arm is freshly initialized on the same model seed and training shard.

## Final validation CE; lower is better

| Head method | Seed137 | Seed149 | Seed163 | Three-seed mean |
|---|---:|---:|---:|---:|
| Official Ember plain | 6.2790 | 6.2775 | 6.2825 | **6.27967** |
| Upstream split-centroid online + Ember | **6.2518** | 6.2592 | 6.2584 | **6.25647** |
| Our flat-head normalized projector α2 + Ember | 6.2590 | **6.2494** | **6.2582** | **6.25553** |
| Upstream split-centroid online + Adam | 6.2581 | 6.2516 | 6.2577 | **6.25580** |

Within-seed projector vs standalone Ember gains: **+.0200,+.0281,+.0243 CE**, mean **+.02413**, all positive. Against original centroid+Ember, the paired gains are **−.0072,+.0098,+.0002**, mean **+.00093**. Against original centroid+Adam they are **−.0009,+.0022,−.0005**, mean **+.00027**.

**Important conclusion:** The new parameter-free method **reproduces a positive gain over modern ungrouped Ember at near-full vocabulary**, but its *accuracy advantage over the original shared-centroid optimizer is not established*. The three grouped methods are practically tied at V50k, with a substantial sign reversal by seed. Do not report “new SOTA” from the +.00093 mean.

## Observed training-wall time, same CPU job and sequential method order

| Method | Mean reported train seconds |
|---|---:|
| Ember plain | 477.97 s |
| Upstream centroid+Ember | 549.60 s |
| Normalized projector+Ember | **507.40 s** |
| Upstream centroid+Adam | 564.67 s |

In these GitHub-hosted CPU trials, our projector was ~**7.7% faster** than original centroid+Ember under matched preprocessing/training scope: 507.4 vs 549.6 s, yet ~6.2% slower than plain Ember. The timing comparison is useful descriptive evidence, **not an equal-wall-clock convergence comparison**: speed varied between GitHub hosts and methods execute in a fixed order, and token-group KMeans overhead is not separately profiled. The practical advantage may be simpler optimizer-state/assignment handling and moderate compute rather than a material parameter-memory saving. At V50k, only Kd~4k trainable centroid values are removed relative to the >3M-row-element head.

## Cross-scale lesson

[Separate V16k data](PROJECTOR_V16K_INDEPENDENT_RESULT.md): the new projector gained **~0.049 CE** vs original centroid+Ember and **~0.023** vs original centroid+Adam over three seeds, much clearer than at V50k. The improvement is **not monotonic in vocabulary**: V50k's benefit over the original method collapses to near zero, though the benefit over plain Ember remains positive.

This is consistent with the [rank fraction](PROJECTOR_RANK_AND_COHERENCE_LEMMA.md): with K64, the normalized group subspace accounts for only (K−1)/(V−1)≈0.125% of random zero-sum gradient energy at ~50k, compared with ~0.394% at 16k. **That calculation is a null-model prediction, not an explanation proven by measurement at 50k**; actual learned gradient coherence is measured only at V6k so far.

## Remaining confirmation

1. [V50k static semantic/context-late comparison](CONTEXTUAL_PROJECTOR_TRANSFER_PROTOCOL.md), currently executing separate seeds 181/191/197; may reveal better fixed group priors.
2. [External shard002 test](LOCKED_FINAL_HELDOUT_5ARM_PROTOCOL.md), new seeds 809/811/821, to check whether the small 50k validation advantage carries out-of-shard.
3. [Tuned standalone Ember challenge at V16k](EMBER_HEAD_LR_TUNING_CHALLENGE_V16K.md) to remove weak-optimizer LR as alternative explanation.
4. Higher batch/large backbone remain open; original dynamic projector **failed batch128** [with same training tokens](PROJECTOR_BATCH_SCALING_RESULTS.md), whereas static context groups [rescued](HIGH_BATCH_CONTEXT_RESCUE_RESULTS.md) that condition at V6k.

No novelty claim about the original author's dynamic split-centroid method; ours is a distinct normalized single-head update rule whose main value remains to be established relative to well-tuned and semantically grouped baselines.
