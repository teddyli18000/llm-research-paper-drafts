# Paper 01 — Modern-optimizer transfer to V16k and V50k (pre-registered)

Registered after reading the complete **V6k Ember grid** and before training any of these new seeds. Source [official Ember code](https://github.com/katop1234/ember/tree/76f30117b2d948874cd1c965c4f6813fa9de8caa), pinned commit `76f30117b2d948874cd1c965c4f6813fa9de8caa`. Exploratory V6k measurements: [MODERN_OPTIMIZER_AND_TIED_RESULTS.md](MODERN_OPTIMIZER_AND_TIED_RESULTS.md).

## Hypothesis

Does direct dynamic output-row group-gradient transfer add a **positive marginal benefit over the modern official Ember token-table optimizer**, once vocabulary size and model width increase? A second question: does an informative same-training-data fixed context partition remain stronger than online grouping at V16k, as it was at early V6k and in the tied-head study?

## Locked transfer settings

| Property | V16k test | Near-full GPT-2 V50k test |
|---|---|---|
| `--V` argument | 16000 | 50258 |
| width / heads / layers | 128 / 4 / 2 | 64 / 4 / 2 |
| steps / batch / sequence | 400 / 32 / 64 | 800 / 32 / 64 |
| gradient accumulation | 1 | 4 |
| Plain/grouped Adam head LR | .03 | .05 |
| Shared centroid Adam LR | .005 | .01 |
| Ember head LR (all) | **.05** | **.05** |
| All head Adam epsilon | 1e−5 | 1e−5 |
| Official Ember epsilon | 1e−8 | 1e−8 |
| Ember beta2 (fixed candidates) | **.95 and .999** | **.95 and .999** |
| Grouping K / regroup schedule | 64, every50 steps | 64, every50 steps |
| Seeds (never used in previous V6k/V16k/V50k studies) | **41, 73, 109** | **41, 73, 109** |
| Validation | Original fixed 600 FineWeb validation windows | Same 600 windows |

The runner is the original study's instrumented `grid.py`, not a new LM implementation. Fixed train data prefix per scale is used across all arms. `--threads 2` mirrors earlier V50k feasibility runs; outcomes close to 0.001–0.005 CE require the same-run paired comparisons and should not be overinterpreted.

## Seven preregistered V16k arms / six V50k arms

1. `adam_plain`: head Adam with head-specific epsilon.
2. `adam_dynamic`: original online group-sharing method; centroid Adam.
3. `ember_plain_b999`: strongest V6k **standalone** official Ember selection (LR.05, beta2=.999).
4. `ember_plain_b095`: same Ember hyperparameters as the strongest V6k combined model; matched no-group comparator.
5. `ember_dynamic_b999`: groups + Ember with beta2=.999 (matched to arm3).
6. `ember_dynamic_b095`: groups + Ember with beta2=.95 (matched to arm4).
7. `context_static`: **V16k only** same-training-data semantic KMeans labels from `context_groups_16000.py` (precomputed from first 819,200 train tokens, no validation access).

No post-hoc LR/epsilon sweep at the transfer scale. Result count: (7 + 6) × 3 = **39 full trainings**. The existing output group model retains its original `R=U+AC` parameterization, even with Ember on `U`; shared centroids `C` remain on ordinary Adam. We are not claiming this is an official fully Ember-optimized model.

## Fixed analysis

- All six configurations common across scales, three paired seeds each, report per-arm mean CE and paired differences within each seed.
- **Primary:** `ember_plain_b095 − ember_dynamic_b095` and `ember_plain_b999 − ember_dynamic_b999` at both scales. Both must be reported, including any sign reversals.
- **Strong standalone comparison:** best frozen V6k standalone Ember (beta2=.999) versus grouped Ember (.95 and .999); also compare grouped Adam.
- **Static prior:** on V16k compare context_static with every online mechanism; if static wins, report it without weakening the baseline definition.
- **Frequency:** report target-frequency contributions, specifically zero-exposure, 1–9, 10–99, >=100 target counts and corpus-frequency buckets. Do not claim that all rare tokens improve: the V50k original experiment showed strong negative transfer to zero-exposure rows.
- No changes to arms, thresholds or selected hyperparameters after results arrive. Save all result JSONL/logs as Actions artifacts, include runtime and group churn records.

## Outcome interpretation

If grouped Ember improves both matched Ember settings at both scales on independent seeds, the case for *additional* cluster-gradient coupling is materially stronger than the original V6k pilot. If Ember-only matches grouped methods at V50k or the benefit is limited to rare undertrained rows and vanishes for older seeds, narrow the claim accordingly. The next independent gate for a major conference remains >100M-parameter pretraining/strong same-compute semantic grouping and realistic throughput; full vocabulary alone is not the same thing as a large LM.
