# Paper 01 — Ember head-only optimizer challenge (pre-registered)

Status: **protocol before experiments**. Official implementation: [katop1234/ember](https://github.com/katop1234/ember), pinned at `76f30117b2d948874cd1c965c4f6813fa9de8caa` (MIT). Paper: [Ember: A Kilobyte-State Optimizer for Token Interfaces](https://arxiv.org/abs/2607.01455), v4 2026-10-11.

## Why this is a direct challenge

Ember replaces Adam's token-table first and second moments by a factored row × column squared-gradient estimator, with **no first moment**. The official 2026 work reports gains over tuned AdamW and memory savings at much larger scale. Its claimed novelty already covers exploiting special LM-head gradient geometry; Paper01 cannot claim this concept.

For a gradient `G` of output rows `R∈R^(V×d)`, official Ember estimates:

```
r_t[i] = b2*r_(t-1)[i] + (1-b2)*mean_d G[i,d]^2
c_t[d] = b2*c_(t-1)[d] + (1-b2)*mean_i G[i,d]^2
s_t = b2*s_(t-1) + (1-b2)*mean_(i,d) G[i,d]^2
vhat[i,d] = r_t[i]*c_t[d]/(s_t*(1-b2^t))
R -= lr*G/(sqrt(vhat)+eps)
```

The test imports **official pinned Ember**, not a guessed reimplementation. Apply it only to the output head (private `U` or plain `R`); retain unchanged input embedding and Muon/body Adam to isolate a head-only optimizer mechanism. This is a strong *head-only* challenger, not a reproduction of official joint input/head Ember pretraining.

## Fixed matched setting

- Same public GPT-2-token FineWeb shard, V6000 width32 layers2 heads2 batch32 L64 400steps, seeds0/1/2, CPU one thread, standard backbone/embedding optimizers and shared data.
- Per-seed controls: ordinary Adam output head (`heps=1e-5, hlr=0.1`), online 64-group KMeans output head with Adam (`clr=0.02`).
- Per-seed official Ember heads: head `LR∈{0.01,0.02,0.05,0.1}` × `beta2∈{0.95,0.999}` × **plain/online grouped** (16 settings).
- Ember epsilon fixed at `1e-8` (official default). A later epsilon sweep must be explicitly distinguished from this registered grid.
- Online KMeans refresh every 50 steps, shared-group `C` remains on ordinary Adam as in the original method; Ember only optimizes the private `U` rows.
- Total **54 full trainings**, 3 seeds × 18 methods. Raw per-token frequency-bin loss preserved.

## Decision rule

Report each LR × beta2 result and the 8 paired improvements when switching on adaptive grouping at identical Ember parameters. Independently tuned Ember alone is compared with the original dynamic-group Adam. If Ember-only matches the method, the scientific claim must narrow to a choice of optimizer geometry rather than a uniquely useful dynamic grouping procedure. If grouping improves both coupled and factored optimizers at matched settings, this is evidence for a distinct adaptive cluster-gradient mechanism.

Selection on these V6k losses is exploratory. Any winning combination must pass a preselected transfer test at V16k and near-full GPT-2 vocabulary on independent seeds before making scale/novelty claims.
