# Paper 01 — Coupled Adam optimizer challenge (pre-registered)

**Status: design fixed before running; no outcomes selected.** This is the next direct challenger to the original dynamic output-row sharing hypothesis. Work branch: `research/01-modern-optimizer-baselines`.

## Why this matters

The upstream `grid.py` already contains an implementation of *Coupled Adam* for the output head (`--coupled 1`). It maintains a token-specific first moment `m[i,d]` and a **vocabulary-averaged, feature-wise** second moment `v[d]`; it does **not** pool gradient directions from related words. Prior comparisons tested Adam versus AMSGrad but did **not** evaluate this native Coupled Adam baseline.

This creates a decisive separation of mechanisms:
- **Optimizer-only:** ordinary `R=U` trained by Coupled Adam, which shares adaptive second-moment normalization across all token rows.
- **Structure + optimizer:** original `R=U+AC` with online K-means, Coupled Adam on `U`, and original Adam on shared group vectors `C`.
- **Original control:** normal Adam with head-specific `eps=1e-5`, with and without the same online K-means groups.

The overall head and softmax outputs are still the same functional family. What differs is the update geometry.

## Exact Coupled Adam update used here

For the per-token gradient matrix `G` of the ordinary output head:

```
m_t = b1*m_(t-1) + (1-b1)*G_t
v_t[d] = b2*v_(t-1)[d] + (1-b2)*(1/V)*sum_i G_t[i,d]^2
U_t[i,d] = U_(t-1)[i,d] - lr * mhat_t[i,d] / (sqrt(vhat_t[d]) + eps_head)
```

The original `grid.py` used global `a.eps` for its Coupled Adam denominator even though `--heps` only changed the ordinary head Adam parameter group. To avoid changing embedding/backbone optimizer epsilon when tuning Coupled Adam, the instrumented *study copy* adds a separate `--cheps` argument to this denominator only. Default other-parameter epsilon remains `1e-8` in every arm. The native coupled moment rule is not otherwise changed.

## Fixed training protocol

- Public FineWeb GPT-2 train shard and val shard, reconstructed token-frequency map; same paths as the prior independent replication.
- Vocabulary `V=6000`, width32, 2 layers, batch32, seq64, 400 optimizer updates.
- Model seeds **0, 1, 2**, each from scratch and paired across conditions.
- CPU with **one thread** for deterministic repeated runs; every arm uses the same first 400 steps' training data, embedding/backbone optimizer and learning rate schedule.
- Adam controls (per seed): ordinary head vs dynamic 64 groups, both `--heps 1e-5 --hlr 0.1 --clr 0.02`.
- Coupled Adam grid (per seed): head LR `{0.02, 0.05, 0.1, 0.2}` × head epsilon `{1e-6, 1e-5}` × **plain or online-grouped head**.
- Dynamic assignments: 64 online K-means groups, refresh at steps 50, 100, ..., 350. Same base parameter initialization and deterministic assignment as original.
- Total **54 full trainings**: `3 seeds × (2 Adam controls + 16 Coupled Adam variants)`.
- Coupled Adam's `beta1=0.9, beta2=0.95` and all other hyperparameters are inherited unchanged from the upstream study.

## Primary questions and analysis

1. **Optimizer alternative:** Does the *best* tuned Coupled Adam-only head match or beat the original dynamic-group Adam result, without learned word groups?
2. **Complementarity:** At matched head LR and epsilon, does dynamic grouping *still* improve a Coupled Adam head? Report all 8 matched three-seed paired contrasts, not only the best.
3. **Stability:** Which head LR × epsilon settings diverge or underperform? Preserve every result, including poor runs.
4. **Long-tail:** Retain original validation frequency bucket metrics. Check whether rare-row gains survive a shared-second-moment optimizer and whether any common-token regression remains.
5. **Training-vocabulary dependence:** Treat this V6k grid as a **selection study only**. The single best fixed Coupled Adam configuration selected by mean V6k CE must subsequently be evaluated at V16k and near-full V50k with fresh seeds/data; this is the confirmatory test, not another round of V6k cherry-picking.

## Collision/attribution boundary

[Better Embeddings with Coupled Adam (2025)](https://arxiv.org/abs/2502.08441) already established shared vocabulary second moments, anisotropy concerns, and related training benefits. The specific claim under test is *additional dynamic inter-token gradient transfer* beyond the independently tuned shared-second-moment optimizer, not invention of Coupled Adam or generic rare-token bias. See [PAPER_THESIS.md](../../research-lab/01-rare-words-shared-output-vectors/PAPER_THESIS.md) and [PRIOR_ART_EXPANDED.md](PRIOR_ART_EXPANDED.md).
