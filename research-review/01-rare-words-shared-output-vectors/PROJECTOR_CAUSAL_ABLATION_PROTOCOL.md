# Paper 01 — Causal test of learned group geometry in parameter-free projected Ember

**Pre-registered before trial.** The exploratory [projector α sweep](INDEPENDENT_SCALE_AND_PROJECTOR_RESULTS.md) shows that α=2 on V6k matches the original dynamic shared-centroid model on three small seeds. This experiment determines whether that benefit actually requires **learned adaptive group geometry**, rather than any partition, a global learning-rate effect, or first-50-step transient behavior.

## Fixed setting

- The same instrumented upstream `grid.py`, public FineWeb shard, V6000/width32/2 layers/2 heads/batch32/L64/400 optimizer steps, head-only official Ember LR.05/beta2=.95/eps1e−8; other optimizers and data untouched.
- New independent model initialization seeds **223,227,229**.
- One single fixed α=**2**, K=64 groups in every projected intervention. KMeans refresh at steps 50,100,...,350 where applicable.
- 9 arms per seed, **27 full trainings**, no retrospective hyperparameter retuning or arm deletion:

| Arm | Group assignment or control |
|---|---|
| `plain` | Ember only, no projection |
| `plain_lr2` | Ember only with doubled head LR.10 |
| `plain_lr3` | Ember only with tripled head LR.15 |
| `projected_dynamic` | Online KMeans on head rows every50 steps |
| `projected_shuffled` | Same online KMeans every50 steps, but permute token-to-cluster assignments after each refresh to preserve exact group sizes and compute cost while eliminating learned identity |
| `projected_random` | Static random groups (same 64 clusters, initial assignment RNG fixed) |
| `projected_frequency` | Static equal-unigram-mass frequency groups from same training counts |
| `projected_context` | Static adjacent-token co-occurrence TF-IDF/SVD/KMeans groups from *training data only* (same-data strong semantic comparator) |
| `projected_one_time` | Use the initial random partition until step50; one KMeans refresh then freeze; tests whether repeated reassignment matters |

## Mechanistic questions

1. Are **dynamic KMeans** outcomes lower CE than both **shuffled size-matched** groups and random/frequency controls on *all three paired seeds*? If not, the value of learned group identity is not identified.
2. Does repeated online regrouping beat one-time grouping? If not, the temporal adaptation claim weakens.
3. Does static train-only semantic grouping do equally well or better? Then stronger group semantics, not dynamism, could explain improvements.
4. Do LR×2/×3 controls close the gap? They are imperfect because Ember's gradient normalization is nonlinear, but distinguish an ordinary step-size improvement from an essential group operator.

## Reporting

- For each seed and arm: final CE, target-frequency bucket CE, wall training seconds and group churn/size.
- Report paired dynamic-minus-control CE **and** zero-exposure/10–99-count effects, not only the winning arm.
- Clustering partition *identity* must be altered, not only cluster label numbers; the shuffled arm permutes membership across word rows after computing the same online KMeans labels.
- Fixed-context preprocessing time must be included for realistic cost comparisons; original head gradient projection is O(Vd) plus KMeans assignments every50 steps, and inference parameter count is unchanged.

## Outcome threshold

Matching the original shared-centroid V6k method only establishes a useful **new implementation** of its known mechanism. A distinct method paper would require a measurable advantage over Ember alone, strong group baselines and the original algorithm on new seeds and larger vocab/parameter scales, or a significant efficiency advantage proved under fair compute accounting. Retain null outcomes.
