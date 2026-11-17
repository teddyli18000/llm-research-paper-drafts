# Paper 01 — No: simply retuning official Ember does NOT remove the V16k projector gain

**GitHub Actions [#44020024307](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44020024307) succeeded 3/3**, all **39 preregistered trainings** (new seeds331/337/347). V16k width128, 2 layers, batch32/L64/400steps; exact original training code instrumented. Official pinned head-only Ember optimizer across 10 (LR,beta2) combinations, two projected-Ember α2 controls and original author's centroid+Adam control. The grid was precommitted in [protocol](EMBER_HEAD_LR_TUNING_CHALLENGE_V16K.md).

## Relevant full-grid means (lower CE better)

| Method | Mean val CE |
|---|---:|
| **Projected Ember α2, beta2=.95, head LR=.03** | **5.74967** |
| Projected Ember α2, beta2=.95, head LR=.05 | 5.75490 |
| **Best standalone Ember: beta2=.999, head LR=.01** | **5.78270** |
| Upstream original online shared-centroid + Adam | 5.78377 |
| Standalone Ember beta2=.999, head LR=.02 | 5.78980 |
| Standalone Ember beta2=.999, head LR=.03 | 5.80317 |
| Best standalone Ember beta2=.95, head LR=.02 | 5.82637 |
| Other six standalone arms (mean CE range) | 5.82897–5.92977 |

**Complete three seed matched values, primary comparison:**

| Seed | Tuned plain Ember (β₂=.999 LR=.01) | Projected Ember (β₂=.95 LR=.03) | Paired benefit |
|---|---:|---:|---:|
| 331 | 5.7776 | **5.7501** | **+.0275** |
| 337 | 5.7815 | **5.7501** | **+.0314** |
| 347 | 5.7890 | **5.7488** | **+.0402** |
| Mean | **5.78270** | **5.74967** | **+.03303** |

All ten standalone Ember combinations were trained and saved; they did not close the gap. The best model's choice of **beta2 and LR differ** from the selected projector: this is a **method-specific optimizer parameter comparison**, not a fixed-identical-hyperparameter causal contrast. Both sides were given an explicit sweep, but the projector's beta2=.999 branch was not swept here; this is a *strong modern optimizer challenge*, not a proof of absolute global hyperparameter optimality.

The original author + Adam at mean5.78377 is essentially tied with the best standalone Ember mean5.78270, and significantly worse than projected Ember on these three seeds. This replicates that the new normalized class projector is more than just rescuing a poorly tuned bare Ember head **at V16k small-model scale**.

**Important limits:** original val600 repeatedly used to choose new ideas, new seed counts n=3, true backbone is only ~0.39M transformer weights with ~4.1M token-table weights, and optimizer-wide comparisons depend on compute budgets. Independent heldout evidence so far is V50k projector gain vs plain Ember with a near-tie versus the author's centroid+Adam, documented in [INDEPENDENT_HELDOUT_V50K_RESULTS.md](INDEPENDENT_HELDOUT_V50K_RESULTS.md). No top-conference-scale claim from the V16k tuning grid alone.
