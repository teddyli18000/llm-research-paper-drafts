# Paper 01 — decoupled second-moment routing for class-projected LM-head updates

**Status: preregistered; fresh seeds 521/523/541; outcomes not inspected.**

## Question

Does the normalized class-gradient projector lose part of its intended effect because the modified gradient also drives the adaptive optimizer's second-moment estimator?

For a token-row gradient \(G\) and a fixed partition projector

\[
P=A(A^\top A)^{-1}A^\top,
\]

the proposed parameter update uses

\[
\widetilde G=(I+\alpha P)G,\qquad \alpha=2.
\]

In the standard implementation, Ember receives \(\widetilde G\) for both its parameter update numerator and its factored second-moment statistics. The new variant routes these separately:

- **Update direction:** \(\widetilde G\).
- **Second-moment statistics:** raw \(G\).

This preserves the original gradient's row, column and global squared-magnitude statistics while allowing the class-projected direction to control the actual head update.

## Why this is worth testing

The pinned official Ember implementation has no first moment for token tables. It maintains factored second moments:

\[
r_i\leftarrow \beta_2 r_i+(1-\beta_2)\frac1d\sum_j G_{ij}^2,
\]
\[
c_j\leftarrow \beta_2 c_j+(1-\beta_2)\frac1V\sum_i G_{ij}^2,
\]
\[
s\leftarrow \beta_2 s+(1-\beta_2)\frac1{Vd}\sum_{ij}G_{ij}^2,
\qquad
\widehat v_{ij}=\frac{r_i c_j}{s(1-\beta_2^t)}.
\]

The normal update is \(-\eta G_{ij}/(\sqrt{\widehat v_{ij}}+\epsilon)\). Sending \(\widetilde G\) to both paths makes the projector alter its own adaptive denominator. The decoupled variant instead uses \(\widehat v(G)\) and applies \(\widetilde G\) in the numerator.

This is an LM-head/factored-optimizer adaptation of the **modified-first-moment/raw-second-moment** routing studied by Hu et al. (2026) for gradient modification under Adam. Hu et al. do not test class-gradient projectors or Ember. Ember itself has no first moment, so the experiment tests only the corresponding factored-second-moment pathway.

## Locked setup

- Data: same original FineWeb train prefix, GPT-2 vocabulary 6,000, sequence length 64.
- Model: width 32, two layers, two heads.
- Training budget: exactly 819,200 tokens per run.
- Effective batch sizes: 32 and 128, both accumulated from two microbatches.
- Steps: 400 at batch 32; 100 at batch 128.
- Fresh model seeds: 521, 523, 541.
- Output-head optimizer: official Ember pinned to commit 76f30117b2d948874cd1c965c4f6813fa9de8caa, LR 0.05, \(\beta_2=0.95\), \(\epsilon=10^{-8}\).
- Projection strength: \(\alpha=2\), \(K=64\).
- Body and embedding optimizers, learning-rate schedule, token order, validation windows and all model hyperparameters are identical across matched arms.

## Group schedules

1. **Dynamic:** initialize K-means on the initial output rows before the first update; refresh every 50 steps.
2. **Context:** fixed train-only co-occurrence context groups for the full run.
3. **Context-to-dynamic:** start from the same fixed context groups; switch to output-row K-means after 20% of the run, then refresh every 50 steps. This is step 80 for batch 32 and step 20 for batch 128.
4. **Upstream centroid:** original split-centroid parameterization, K-means initialization and refresh every 50 steps.

The dynamic projector is initialized from actual output rows at step 0, not from random group IDs. Context groups are computed from the training prefix only.

## Arms

| Arm | Group structure | Ember second-moment source |
|---|---|---|
| Plain Ember | None | Raw \(G\) |
| Upstream centroid + Ember | Original shared-centroid parameters | Its native head and centroid gradients |
| Dynamic projector, shared moments | Dynamic K-means | \(\widetilde G\) |
| Dynamic projector, raw moments | Dynamic K-means | \(G\) |
| Context projector, shared moments | Fixed context groups | \(\widetilde G\) |
| Context projector, raw moments | Fixed context groups | \(G\) |
| Context-to-dynamic, shared moments | Context then online K-means | \(\widetilde G\) |
| Context-to-dynamic, raw moments | Context then online K-means | \(G\) |

All projector arms apply \(\widetilde G=G+2PG\) to the head update. The only difference between each shared/raw pair is which gradient supplies Ember's factored second-moment statistics.

## Measurements

- Overall held-out CE and all original target-exposure buckets.
- Mean \(\|\widetilde G\|_F/\|G\|_F\).
- Mean actual LM-head parameter-update norm.
- Paired same-seed CE differences for every routing pair and group schedule.
- Runtime per training arm.

## Prior-art boundary

- Hu et al. (2026), *Hidden Failure Modes of Gradient Modification under Adam, and Adaptive Decoupled Moment Routing as a Repair*, directly establishes the general idea of sending modified gradients to the first moment and raw gradients to the second moment. It does not test an LM-head group projector or factored Ember.
- Shah (2026), *Ember: A Kilobyte-State Optimizer for Token Interfaces*, supplies the official factored second-moment optimizer used here. The implementation has no first moment for token tables.
- Han et al. (2026), *A Deafening Silence*, establishes output-only Adam epsilon adjustment as a strong under-exposed-token/continual-pretraining baseline. It does not use group-gradient sharing.
- Somasundaram et al. (2026), *Toward a First-Principles Update Geometry for the Language-Model Head*, and Lau & Su (2026), *Symmetry-Compatible Principle for Optimizer Design*, use global vocabulary centering and row-normalized/spectral LM-head updates. Neither uses within-token-group gradient averaging.
- Tree Loss (Wang & Izbicki, AISTATS 2022) establishes class-structured parameterization; the fixed-projector SGD reparameterization collision and near-tie under Ember/Adam are documented separately.
- ACL 2022 AGG and EMNLP 2024 Syntactic Smoothing remain important rare-token and semantic-learning-signal comparators.

The experiment does **not** claim the general moment-routing principle as new. It tests whether raw-stat routing makes this particular normalized class-projected LM-head update more effective.

## Decision rule

A useful result requires raw-stat routing to beat shared-stat routing for the same group schedule on at least two of three seeds at **both** batch sizes, and to improve the strongest same-regime comparator rather than merely the plain Ember baseline. Bucket results must identify where the gain occurs.

A batch-32-only gain is a batch-regime effect. If raw-stat routing fails this test, stop tuning second-moment routing and move to the exact prediction-residual versus target-residual decomposition.

## Execution

The workflow is registered on the repository default branch and runs against research/01-modern-optimizer-baselines. The protocol and analytic identity test are committed before the fresh-seed matrix. Raw logs and JSONL frequency-bucket results are archived as Actions artifacts.


## Motivation from the completed stable-capture study

The prior fresh-seed matrix (Actions #44037476138) did not meet its preregistered decision rule: mean CE for the energy gate was 4.96290 at batch 32 and 5.06120 at batch 128; eta selection was 4.96533 and 5.06033, respectively. Eta therefore loses to the energy gate at batch 32 and only differs by 0.00087 at batch 128. The next study changes the optimizer's moment pathway rather than continuing to tune the reliability score.

The registered Actions launcher is the existing research workflow path `.github/workflows/research-01-stable-capture.yml`, now dedicated to this moment-routing matrix.
