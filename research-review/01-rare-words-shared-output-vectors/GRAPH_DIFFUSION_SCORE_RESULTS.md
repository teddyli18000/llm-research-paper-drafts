# Paper 01 — graph proper-score collision: all preregistered jobs completed

**Run:** [GitHub Actions #44090597865](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44090597865) — six of six jobs successful, 30/30 matched full trainings validated. Original log line `GRAPH_SCORE` audited for all six jobs (30 unique seed/batch/arm tuples); artifacts contain per-arm JSONL. **Preregistered design:** [graph-score protocol](GRAPH_DIFFUSION_SCORE_PROTOCOL.md).

## Setup

V6k, width32/two-layer Transformer, sequence64, equal **819,200** tokens, batches 32 (400 steps) and 128 (100 steps). Fresh paired seeds 683,691,701. K64 fixed groups derived from training text. Head-only official Ember LR .05, beta2 .95. Ordinary token validation CE for all arms (lower is better). Direct `G+2PG`; three head-only auxiliary-score losses with coefficient λ=1: class-projection P-limit score, clique-graph diffusion score at τ=.02, and graph-resistance score.

## Complete results: held-out CE

| Arm | Batch32 mean | Batch128 mean |
|---|---:|---:|
| Plain Ember | 5.00017 | 5.07983 |
| **Direct projector + Ember** | **4.96403** | **5.06163** |
| CE + P-limit quadratic score | 5.00023 | 5.07983 |
| CE + graph-diffusion score | 5.00113 | 5.08033 |
| CE + graph-resistance score | 5.00020 | 5.07983 |

### Per-seed scores

| Batch | Seed | Plain | Direct | P-limit score | Diffusion score | Resistance score |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 683 | 4.9952 | **4.9611** | 4.9951 | 4.9963 | 4.9952 |
| 32 | 691 | 5.0042 | **4.9654** | 5.0043 | 5.0054 | 5.0042 |
| 32 | 701 | 5.0011 | **4.9656** | 5.0013 | 5.0017 | 5.0012 |
| 128 | 683 | 5.0708 | **5.0566** | 5.0708 | 5.0711 | 5.0708 |
| 128 | 691 | 5.0827 | **5.0642** | 5.0827 | 5.0836 | 5.0827 |
| 128 | 701 | 5.0860 | **5.0641** | 5.0860 | 5.0863 | 5.0860 |

Direct projector beats plain Ember by **0.03613 CE** at batch32 (paired gains 0.0341–0.0388) and **0.01820 CE** at batch128 (paired gains 0.0142–0.0219), all six paired seeds. The tested graph-score additions perform essentially like plain Ember, sometimes slightly worse.

## Why this does not make a sweeping mathematical novelty claim

For `r=p−y`, the direct update adds `2Pr` to the logit-space gradient. The matching quadratic class-probability score is `S_P=rᵀPr`. **Ordinary differentiation through softmax** yields `∇_z S_P=2J_softmax Pr`, *not* `2Pr`; the softmax Jacobian can strongly attenuate the effect. Thus the score-only arms here with **fixed λ=1** are weak controls in update magnitude; their near-null effects do not establish that a **tuned coefficient, different auxiliary loss, natural-gradient score, or preconditioned scalar objective** could not replicate or improve direct projection.

In fact there is an exact per-example **logit Fisher-natural-gradient connection**:

`J_softmax⁺ ∇_z S_P = 2Pr`

because `Pr` has zero vocabulary sum and `J_softmax` has nullspace spanned by the all-ones vector. See [full derivation and numerical check](GRAPH_SCORE_FISHER_GEOMETRY.md). Consequently, although the direct operation is generally **not the ordinary Euclidean logit-gradient of a scalar loss**, its class residual has an existing probabilistic scoring-rule / Fisher-metric interpretation. This narrows broad theoretical priority claims.

Original paper: [Hilfiker, Cincera Brack & Sterchi, *Strictly proper scoring rules for node selection tasks on graphs*, November 2, 2026, arXiv:2611.03274](https://arxiv.org/abs/2611.03274). Its resistance and diffusion scores are proper quadratic probability scores, but it does not describe this normalized class projector as an LM-head optimizer. In the disconnected union-of-cliques `P` limit, the quadratic score is proper but not strictly proper.

## Research decision

- **Accepted empirical result:** the direct flat-head class gradient projector wins all six matched seed/batch experiments over plain Ember and these fixed-coefficient auxiliary losses.
- **Narrow claim only:** comparing to graph auxiliary CE at λ=1 is **not** a comprehensive optimizer- or loss-level superiority result. The exact natural-gradient equivalence is additional reason to describe the independent contribution as an implementation / adaptive-optimizer coordinate interaction rather than a new mathematical principle.
- This experiment does **not** change the higher-level [independent external V50k finding](INDEPENDENT_HELDOUT_V50K_RESULTS.md): the new projected Ember head is nearly tied with the original author's stronger online centroid + Adam on untouched heldout text.

All raw rows are in the Actions artifacts; the six jobs passed `GRAPH_DIFFUSION_SCORE_IDENTITY` numerical assertions before training and verified equal budgets afterwards.
