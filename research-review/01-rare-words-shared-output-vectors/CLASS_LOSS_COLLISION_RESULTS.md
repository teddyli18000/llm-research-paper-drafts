# Paper 01 — matched semantic smoothing and concept-class loss collision tests

Date: 2026-11-17. **Independent audit of original complete GitHub Actions job logs**, not hypothetical trials. These are comparisons against alternative objectives on the same group partitions, not direct replications of the published Syntactic Smoothing or Hypernym Class Prediction systems.

## 1. Semantic target smoothing

**Run:** [#44029810293](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44029810293); 3/3 successful jobs, seeds 353/359/367; 12 arms per seed. Full preregistration: [semantic smoothing protocol](SEMANTIC_SMOOTHING_COLLISION_PROTOCOL.md). V6k, width32, 2-layer model, batch32, 400 optimizer steps, common training data. Ordinary token-level validation cross-entropy (lower is better):

| Arm | Three-seed mean CE |
|---|---:|
| Plain Ember | 5.00573 |
| Original online centroid + Ember | 4.98383 |
| Dynamic direct class projector + Ember | 4.97870 |
| **Train-context fixed direct class projector + Ember** | **4.97140** |
| Head-only train-context target smoothing, ε=0.25 | 5.11287 |
| Head-only train-context target smoothing, ε=0.50 | 5.34597 |
| Head-only train-context target smoothing, ε=0.75 | 5.70630 |
| Full-objective train-context target smoothing, ε=0.25 | 5.08153 |
| Full-objective train-context target smoothing, ε=0.50 | 5.28690 |
| Full-objective train-context target smoothing, ε=0.75 | 5.66020 |
| Head-only dynamic-group target smoothing, ε=0.50 | 5.42983 |
| Full-objective dynamic-group target smoothing, ε=0.50 | 5.33980 |

All three seeds favor projected Ember over every tested smoothing configuration. **Scope:** the tested ε values start at 0.25; this does **not** rule out a more carefully tuned small-ε smoothing objective or the original published POS-based method.

Exact difference: target smoothing changes only target distribution, `G_smooth=(p-y)+ε(y-Py)`; direct projected CE gradient is `G'=(p-y)+α(Pp-Py)`, including a *prediction-side* `Pp` residual. See [derivation](GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md) and the published [Syntactic Smoothing (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.344/).

## 2. Concept/class-probability auxiliary objective

**Run:** [#44030216366](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44030216366); 3/3 successful jobs, seeds 397/409/419; 7 arms per seed. Full preregistration: [concept-class loss protocol](CONCEPT_CLASS_LOSS_COLLISION_PROTOCOL.md). Same V6k architecture, batch32, 400 steps, original CE as evaluation metric.

| Arm | Mean validation CE |
|---|---:|
| Plain Ember | 5.00777 |
| **Fixed train-context direct projector + Ember** | **4.96863** |
| Dynamic KMeans direct projector + Ember | 4.97670 |
| Fixed-context concept-class CE λ=0.1 | 5.00330 |
| Fixed-context concept-class CE λ=0.3 | 4.99400 |
| **Fixed-context concept-class CE λ=0.5** | **4.99037** |
| Dynamic-group concept-class CE λ=0.3 | 4.99370 |

The full class-probability auxiliary objective was `(1−λ) CE_token + λ(-log(sum_{i∈group(y)} p_i))`. Its exact logit derivative is `p_i − 1[i∈group(y)] p_i / p_group(y)` for the class portion. This is not the normalized class residual `P(p−y)`.

The direct fixed-context projector beats the best tested fixed-context class-loss coefficient by **0.02174 CE** averaged over three matched seeds. The direct dynamic projector beats the matched λ=.3 dynamic concept loss by **0.01700 CE**. All three seeds agree in both comparisons.

**Scope:** this is a finite λ-grid on the reused validation set, so the best chosen λ is exploratory. Unlike these matched loss experiments, [Hypernym Class Prediction (ACL 2022)](https://aclanthology.org/2022.acl-long.96/) and [Concept-Level Training Objectives (EACL 2026)](https://aclanthology.org/2026.eacl-short.34/) additionally specify distinct semantic structures/curricula; the experiments do not establish superiority over those full published systems.

## Interpretation for originality

A direct normalized mean-projected output-head gradient is **not simply a rebranded target-smoothed CE or class-probability auxiliary CE**: its equations differ, and it performs better in the tested matched small-model comparisons. This is strong evidence that the *specific gradient intervention* matters.

It **does not establish a new class-sharing principle**. The original source author's `R=U+AC` shares class residual gradients already; Tree Loss (AISTATS 2022) is another major shared-class-parameter antecedent. The decisive remaining comparison is a properly tuned direct projector against its equivalent-in-SGD normalized-centroid parameterization under actual adaptive optimizers, plus real model-scale evidence. See [normalized-centroid LR-sweep results](NORMALIZED_CENTROID_LR_SWEEP_RESULTS.md).

**Source provenance:** mean scores were calculated from the `concept_...: val ...` and `smooth_...: val ...` per-arm lines in all six original Actions job logs. No arms are omitted from the tables.
