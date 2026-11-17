# Paper 01 — Paired uncertainty of the new projector results

**Descriptive 95% paired Student-t intervals across three independently initialized models**. Calculate per-seed difference at identical data, architecture and training hyperparameters, then `mean ± t_(.975,2) × sample_std/√3` with `t_(.975,2)=4.30265`. These intervals treat the three model-initialization outcomes as the sampling units and assume their differences are approximately normally distributed; n=3 gives very little basis to test this assumption. They do **not** incorporate data/corpus shift, optimizer search selection, or substantial future training-scale differences.

| Paired contrast (positive = new projector better) | Individual seed CE differences | Mean improvement | 95% t interval |
|---|---|---:|---:|
| V16k dynamic projector − plain official Ember (plain−proj) | +.0712,+.0695,+.0714 | **+.07070** | **[+.06811,+.07329]** |
| V16k dynamic projector vs original centroid+Ember (centroid−proj) | +.0462,+.0534,+.0470 | **+.04887** | **[+.03906,+.05867]** |
| V16k dynamic projector vs original centroid+Adam | +.0275,+.0290,+.0128 | **+.02310** | **[+.00086,+.04534]** |
| V6k batch128 **contextual** projector vs plain Ember | +.0174,+.0176,+.0242 | **+.01973** | **[+.01012,+.02935]** |
| V6k batch128 contextual projector vs original centroids+Ember | +.0128,+.0152,+.0135 | **+.01383** | **[+.01077,+.01690]** |
| V6k batch32 contextual projector vs dynamic projector | +.0042,+.0083,+.0009 | **+.00447** | **[−.00474,+.01368]** |

The V6k contextual-vs-dynamic difference, though positive on all three observed seeds, has **an interval that crosses zero**; do not call the 0.00447 contrast a confirmed meaningful advantage. The V16k projector vs original centroid+Adam contrast has an interval whose lower bound is just +0.00086, so the superiority there has sizable uncertainty about *magnitude*. The larger V16k projector vs original centroid+Ember effect is much more consistent in these three model seeds.

Separately, the V6k batch128 **dynamic** projector worsened all three seeds vs plain Ember; that negative result motivated the independently preregistered contextual-group rescue test rather than being hidden.

This statistical audit improves how the experiment is reported; it does not convert small-model paired metrics into a general LLM result. The most important next checks remain the preregistered V50k and untouched external-shard tests.
