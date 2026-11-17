# Paper 01 — Measured projected gradient energy, not just loss

**[Actions #44019195126](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44019195126), 3/3 jobs successful.** Replayed the matched V6k/width32/400step, official Ember LR.05/β₂=.95, α2 projector with **identical seeds 223/227/229** for dynamic KMeans, same-group-size shuffled KMeans and static train-only contextual groups. This is a causal-mechanism **diagnostic replay**, not new independent efficacy validation.

## Quantity

Before editing the dense untied LM-head gradient and applying Ember, compute

`c_t=||P_t G_t||_F² / ||G_t||_F²`, `P_t=A_t(A_tᵀA_t)^−1 A_tᵀ`.

Under the deliberately idealized **isotropic zero-sum gradient null**, `E[c]=(K−1)/(V−1)=63/5999≈0.01050` at K64/V6000. Real LM gradients are not isotropic and groupings are learned, so the real control is the same-size shuffled group assignment, not that null expectation.

## All-three-seed mean fraction of squared gradient energy in group-constant subspace

| Step | Dynamic KMeans | Same-size shuffled | Fixed train-only contextual |
|---:|---:|---:|---:|
| 0 | 0.01013 | 0.01013 | **0.03386** |
| 25 | 0.01362 | 0.01362 | **0.01738** |
| 50 | **0.02118** | 0.01016 | **0.03207** |
| 100 | **0.05019** | 0.01113 | 0.00726 |
| 200 | **0.05495** | 0.00807 | 0.01342 |
| 399 | **0.01622** | 0.01014 | 0.01054 |
| Reproduced mean final CE (lower better) | 4.97300 | 4.99467 | **4.96843** |

Raw per-seed coherence traces are available in the job artifact for each seed, not only these means. One replay (seed229) differs by ~0.0003 CE from the original causal geometry experiment, consistent with small cross-run numerical variation.

## Scientific interpretation

1. **Learned groups actually capture meaningful gradient energy, not mere group size:** after regrouping, dynamic KMeans projected energy is roughly 5% at steps100–200 while size-preserving shuffled groups stay around **~1%**. This substantiates the geometric mechanism independently of validation CE: learned row clusters align with the directions of subsequent output-head training gradients.
2. **Static context groups are semantically useful immediately:** at step0 they capture 3.39% energy versus 1.01% from random initial groups; at step50 3.21% versus 2.12% dynamic. This explains a *possible* advantage early in a short run.
3. **Total projected energy is not sufficient to predict the best final CE:** static context has lower projected-energy fraction than dynamic at steps100/200/399, yet achieves better overall CE. Coherence can correlate with useful transfer but it is not itself a performance objective; early grouping and bias in which class directions are amplified may matter more than norm alone.
4. The last step's dynamic ratio falls back to 1.62%, so it is invalid to claim learned coherence remains high throughout pretraining.

The original author’s learned centroid groups already transfer gradients; our independent contribution is a **normalized single-head preconditioner** plus this controlled measurement against shuffled and semantic partitions. A possible future method would adapt projection strength using measured, train-only class-gradient reliability, but that would be a *new hypothesis requiring preregistered evaluation*, not an established result of these traces.
