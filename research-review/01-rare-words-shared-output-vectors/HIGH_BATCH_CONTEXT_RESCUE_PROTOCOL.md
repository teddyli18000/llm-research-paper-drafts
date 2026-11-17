# Paper 01 — Can context-first grouping or more frequent refresh rescue batch128?

**New exploratory test preregistered after the failed fixed-α2 batch128 result** ([evidence](PROJECTOR_BATCH_SCALING_RESULTS.md)). We do not modify the original negative result. Use three fresh model seeds **281,283,293**.

## Controlled setting

- V6000, width32/2 transformer layers/2 heads, same FineWeb 819,200-token prefix and fixed600-window validation, batch128/L64/100 updates, accumulation4, official Ember head LR.05/beta2=.95/eps1e−8, all body/embedding settings fixed.
- The new projector remains `G'=(I+2P)G`, K64. No retuning α.
- All semantic priors built from the **same 819,200 training tokens**, not from validation/test. Adjacent-token symmetric co-occurrence, TF-IDF, SVD48, KMeans64.
- Seven locked arms, each seed trained from scratch (21 runs):

| Arm | Projection/grouping |
|---|---|
| `ember_plain` | No projector or groups |
| `ember_centroid` | Original split-centroid dynamic online KMeans, refresh50 (baseline) |
| `proj_dynamic_50` | Our original failed group projector, initial random partition, update at step50 |
| `proj_dynamic_10` | Same projector, initial random partition, learned KMeans refresh every10 steps |
| `proj_dynamic_0_10` | KMeans on initial rows at step0 and every10 thereafter |
| `proj_context_static` | Train-context fixed groups from step0, never regroup |
| `proj_context_late` | Train-context groups from step0; online KMeans refresh at steps20,30,…90 |

## Decision

Primary: `CE(ember_plain)−CE(proj_context_static)` and `CE(ember_centroid)−CE(proj_context_static)`. Secondary: are shorter regrouping cycles sufficient on their own to prevent the failure, and is context→online strictly better than frozen context?

All three seeds and all arms are reported. Strong batch128 gains would motivate a separate precommitted high-batch evaluation at larger vocabulary and on untouched test data. A null result establishes that the normalized projector has a serious batch dependence under the tested regime; it is not rescued by these simple controls.
