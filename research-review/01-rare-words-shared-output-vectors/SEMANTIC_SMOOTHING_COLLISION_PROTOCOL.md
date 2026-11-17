# Paper 01 — Is class-projected gradient really different from semantic label smoothing?

**Preregistered before runs** based on the exact decomposition in [GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md](GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md). This is an explicit strong-prior control prompted by [Syntactic Smoothing (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.344/); the tested simple within-class soft target is **not** claimed to exactly replicate that paper's syntax/POS construction.

## Fixed model and data

Original Paper01 2-layer Transformer; V6000,width32,heads2,batch32,sequence64,400 updates, original FineWeb train prefix and fixed 600 validation windows. Official pinned **head-only Ember** LR=.05, β₂=.95, eps1e−8; unchanged Muon body / embedding Adam. All projected methods α=2, K64, learned KMeans refresh50 or fixed training-prefix co-occurrence TF-IDF/SVD/KMeans semantic classes. **New three seeds 353/359/367**, not used in optimization/smoothing grids. Group priors are derived solely from training prefix.

## Twelve locked arms per seed (36 full trainings)

1. `plain`: ungrouped ordinary CE with Ember head.
2. `upstream_centroids`: original author's online `R=U+AC` group-centroid method; head Ember and class-centroid Adam.
3. `projector_dynamic`: proposed flat-head normalized class residual `G'=(I+2P)G` with online KMeans.
4. `projector_context`: same projector with fixed train-context classes.
5–7. `smooth_head_context_eps025/050/075`: **head-only** conventional within-class target smoothing `y'=(1−ε)y+εPy` applied directly as a correction to the output-head gradient, leaving all backbone and input-embedding gradients from the original CE unchanged; `ε∈{.25,.5,.75}`.
8–10. `smooth_full_context_eps025/050/075`: the **actual class-smoothed CE loss** with target y′ above, changing the backbone and head gradients together. Standard NLL evaluated at end, not the smoothed training objective.
11. `smooth_head_dynamic_eps050`: head-only ε=.5 target smoothing, moving KMeans classes.
12. `smooth_full_dynamic_eps050`: full smoothed CE, moving KMeans classes.

Each arm from scratch with same initialization seed and training prefix. No post-hoc changing ε or discarding arms.

## Interpretation

- Primary: paired CE `projector_context` vs best context-smoothed **full** objective, plus online projector vs dynamic smoothed CE. If the projected-head advantage disappears against equally well-chosen within-class soft targets, the substantive novelty is weaker (akin to semantic learning-signal smoothing rather than new optimization).
- Secondary: head-only smoothing isolates the output-head gradient; full loss permits useful backbone feedback and is a stronger standard baseline. Same-group full-smoothed objective should not be mislabeled an equal-backbone-gradient comparison.
- Every arm's loss/bins, group-size summary and train wall-clock time must be archived; report target frequency-bin change, not just mean CE.
- Since ε is swept on original validation, a winning smoothed configuration is a **selection result**, and a clean independent-seed/independent-shard followup must be registered before claiming a new method superior.
- Standard full smoothed loss changes objective; projected gradient is a **preconditioned CE direction** whose group correction vanishes when class probability residual is zero. Neither is guaranteed equivalent under Ember's nonlinear update.

EMNLP 2024 provides established prior art for distributing rare-token learning signal to syntactically related tokens; its published method and original Paper01 both must be acknowledged even if direct projector still wins.
