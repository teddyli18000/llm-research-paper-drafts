# Paper 01 — 2026 LM-head optimizer and rare-token prior-art audit

**Literature cutoff: 17 November 2026.** This note distinguishes full-text findings from the experiment results already produced in this fork. It is a novelty-boundary document, not a priority claim.

## Candidate update under review

For an untied vocabulary head \(W\in\mathbb R^{V\times d}\), one token's standard head gradient is

\[
G=(p-y)h^\top.
\]

For a vocabulary partition with indicator matrix \(A\in\{0,1\}^{V\times K}\), group-size matrix \(D=A^\top A\), and normalized class-mean projector

\[
P=AD^{-1}A^\top,
\]

the candidate flat-head intervention is

\[
G'=(I+\alpha P)G
=\left[(p-y)+\alpha(Pp-Py)\right]h^\top.
\]

The distinctive algebraic term is the prediction-side class residual \(Pp\), paired with the target-side class residual \(-Py\). The same projector can be built from train-only token-context groups or from K-means over output rows. The forward logits and parameter matrix remain unchanged.

## Closest prior art

| Work | What it establishes | Exact overlap | What it does not do |
|---|---|---|---|
| **The Tree Loss** (Wang & Izbicki, AISTATS 2022) | Class-structured parameterization and class-level gradient sharing | A shared-centroid parameterization \(R=U+AC\) induces class-coupled output-row updates under SGD | It does not use the normalized class-mean projector \(AD^{-1}A^\top\) as a flat-head preconditioner |
| **Syntactic Smoothing** (Diehl Martinez et al., EMNLP 2024) | Syntactically informed distribution of target learning signal | Uses a token-similarity operator to move target mass to related tokens | It changes the target distribution; its logit gradient has no prediction-side \(Pp\) term |
| **AGG** (Yu et al., ACL 2022) | Frequency-aware gating of harmful rare-output-embedding gradient components | Directly modifies output-embedding gradients for rare/very-rare tokens | Groups by recent token exposure and attenuates selected gradient components; it does not share semantic class-mean gradients |
| **Hypernym Class Prediction** (Bai et al., ACL 2022) | Class prediction can aggregate context for related rare words | Uses semantic classes and a token/class training curriculum | It changes the prediction objective/factorization rather than applying \(P(p-y)\) to the output-head gradient |
| **Gradient Grouping (SGG)** (Li et al., ACL 2025) | Groupwise adaptive learning-rate calibration from gradient statistics | Dynamic gradient groups and collective scaling | Groups parameter-gradient statistics inside layers; not vocabulary-row residuals or token-class projectors |

### The Tree Loss: closest structural collision

The Tree Loss is the most direct prior art for the original shared-output-vector idea. Its parameterization makes related class rows share a trainable component; under SGD, the induced update contains the unnormalized class-sum coupling \(AA^\top G\). For a fixed projector \(P\), the update \(G+\alpha PG\) can itself be represented by a linear change of coordinates under SGD. The fork's separate direct-versus-reparameterized experiment found near-tied results under both Ember and Adam across three seeds.

The distinction between \(AA^\top\) and \(AD^{-1}A^\top\) is real: the former has group-size-dependent eigenvalues, while the latter is an orthogonal projector with eigenvalues 0/1. That is a bounded-spectrum normalization distinction, not a wholly new concept of class sharing or gradient coupling.

### Syntactic Smoothing: target-side, not prediction-side

The full-text Syntactic Smoothing objective uses a target distribution of the form

\[
t=(1-\epsilon)y+\epsilon Py,
\]

so its logit gradient is

\[
p-t=(p-y)+\epsilon(y-Py).
\]

It redistributes target mass. It does **not** apply \(P\) to the prediction vector \(p\). The candidate update instead contains

\[
(p-y)+\alpha(Pp-Py).
\]

This difference is important: the candidate changes the prediction-side residual as well as the target-side residual. The paper's printed Equation (3) has an indexing ambiguity in its similarity normalization; a faithful reproduction should preserve and resolve that issue explicitly rather than assuming an implementation.

## Newer LM-head optimizer geometry

### Somasundaram et al. (2026): projected RowNorm

*Toward a First-Principles Update Geometry for the Language-Model Head*, arXiv:2608.22253.

The paper treats the LM head and softmax as one module and derives a functional update metric from Hilbert projective distance. Its update-size measure is the maximum pairwise token-row separation, \(D(S)=\max_{i<j}\|s_i-s_j\|_2\). Its tractable projected RowNorm update globally centers vocabulary rows,

\[
P_{\mathrm{global}}=I-\frac{\mathbf1\mathbf1^\top}{V},
\]

then row-normalizes and re-centers the direction. It reports 190M, 380M and 640M experiments, with large reductions in row-diameter and Hilbert-RMS perturbations and small validation-loss increases relative to its AdamW recipe.

**Collision boundary:** it is direct prior art for softmax-aware LM-head update geometry and projected head optimization. It does not partition vocabulary rows into token groups or average within-group gradients. Its projector removes only the single global shared-logit-shift direction.

The geometric distinction is exact. For a true softmax head gradient, each feature column sums to zero across vocabulary rows: \(\mathbf1^\top G=0\). Therefore the global-centering projector \(P_{global}=I-\mathbf1\mathbf1^\top/V\) leaves that gradient unchanged: \(P_{global}G=G\). By contrast, \(P_A=AD^{-1}A^\top\) replaces each row by its within-group mean and removes within-group zero-sum variation while preserving each group's total gradient. The two projectors act on different subspaces.

### Lau & Su (2026): symmetry-compatible LM-head updates

*Symmetry-Compatible Principle for Optimizer Design*, arXiv:2605.18106.

The paper derives optimizer rules from the symmetries of each parameter block. For LM heads it uses vocabulary-row permutation/right-orthogonal equivariance and the global shared-row-shift symmetry of softmax. Its centered row-norm, centered right-spectral and hybrid updates use the same global vocabulary-centering projector \(P_{\mathrm{global}}\), with a final projection after row-dependent nonlinear scaling to preserve zero row sum.

**Collision boundary:** strong prior art for global centering and row-aware LM-head optimization. It does not use semantic or K-means token groups, nor the class-mean projector \(AD^{-1}A^\top\).

### Ma et al. (2026): MuonIO

*MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads*, arXiv:2610.02705.

MuonIO uses rowwise momentum normalization for the LM head. Each token row gets its own normalized momentum direction; a shared scalar aggregates row norms. It does not average directions across token groups. Its C4 pretraining comparison includes 60M, 130M and 1B models, and Muon Tuned is the strongest standard-Muon comparison.

**Collision boundary:** a strong, modern LM-head optimizer baseline. It is not an algorithmic anticipation of class-mean gradient projection.

### Shah (2026): Token Geometry / Ember

*Token Geometry*, arXiv:2607.01455 (current v4; the optimizer introduced is Ember).

The paper frames Ember as a square-root empirical-Fisher preconditioner for token interfaces. It argues that rowwise gradient energy tracks token activation frequency and that the LM-head gradient admits the same factorized treatment as an embedding table. The official implementation stores factored row/column second moments for token tables and no first-moment state. Its statistics are computed from the same gradient used in the parameter update. The fork's experiments use the official pinned implementation.

**Collision boundary:** a strong baseline and the adaptive denominator with which the candidate projector interacts. A projected update numerator with raw-gradient factored statistics is a new variant of Ember; it is not the published Ember algorithm.

## Gradient modification and rare-output-token dynamics

### Hu et al. (2026): decoupled moment routing

*Hidden Failure Modes of Gradient Modification under Adam in Continual Learning*, arXiv:2604.22407.

The paper shows that sending a modified gradient to both Adam moments can undermine the intended gradient modification: reducing a protected direction also reduces its second-moment denominator and can inflate its effective learning rate. Its repair sends the modified gradient to the first moment and raw gradient magnitudes to the second moment, with adaptive strength.

**Collision boundary:** the general principle of modified-update/raw-second-moment routing is established. It does not test token-class projectors or Ember's factored second-moment estimator. The current experiment tests this specific adaptation to a class-projected LM head.

### Han et al. (2026): absent-token output drift

*A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks*, arXiv:2610.09835.

The paper identifies one-sided softmax gradients on absent output-token rows as a source of forgetting in vocabulary-deficient continual training. It raises Adam epsilon only for the output projection (\(10^{-4}\) instead of \(10^{-8}\)); across seven stable settings from 160M to 12B it reports 39.4–67.9% forgetting reduction. It also compares row freezing and update gating.

**Collision boundary:** strong baseline for rare/absent output rows when the new corpus does not cover the old vocabulary. It does not use token groups or cross-row gradient sharing. It is especially relevant to the zero-exposure bucket and should be added as an optimizer baseline before claiming rare-token-specific benefit.

### Godey & Artzi (2026): LM-head backward bottleneck

*Lost in Backpropagation: The LM Head is a Gradient Bottleneck*, COLM 2026, arXiv:2603.10145.

The paper studies the rank-\(d\) mapping from hidden states to \(V\)-dimensional logits. It reports that 95–99% of logit-gradient norm is suppressed when the signal is backpropagated through the LM head into the preceding network.

**Collision boundary:** this is a different bottleneck from token-row gradient sharing. A class-mean projector applied to output-head parameter gradients does not, by itself, restore logit-gradient components in the null space of the head's backward map. Do not claim the projector solves the LM-head backward bottleneck.

## What the fork's evidence currently supports

- The flat-head projector improves over plain Ember at V≈50k on the independent seeds tested, but is essentially tied with the upstream shared-centroid method and centroid+Adam.
- On V≈50k, fixed train-context groups lose to online output-row groups; context-first then online regrouping is near-tied with online grouping.
- On V6k at batch128, fixed train-only context groups give a strong short-horizon gain over plain Ember and upstream centroids. The same fixed-context result does not transfer to V≈50k.
- The original dynamic projector is not batch-robust without context initialization.
- The cross-microbatch rho/eta selection experiments do not show a stable advantage over the simpler projected-energy gate.
- Standard target-side semantic smoothing and concept-class CE do not reproduce the projector's validation results.
- The Tree-Loss reparameterization control is near-tied with the direct projector under Ember and Adam.

These results support a **narrow normalized class-mean preconditioner hypothesis**, not a broad claim that dynamic semantic grouping is new or that the method is already a stronger LM-head optimizer.

## Required next comparisons

1. Finish the preregistered raw-second-moment routing study. The general routing principle is Hu et al.'s; only an empirical LM-head/Ember adaptation can be claimed.
2. Add output-only epsilon \(10^{-4}\) from Han et al. as a rare/absent-row baseline, with per-exposure buckets.
3. Add a faithful projected RowNorm baseline from Somasundaram et al. and a MuonIO baseline at V≈50k.
4. Isolate prediction-side \(Pp\) and target-side \(-Py\) contributions. Standard target smoothing is not an exact comparator for the full residual transform.
5. Test on an untouched shard and at a substantially larger model before writing a general LM-pretraining claim.

## Bibliographic links

- Tree Loss: https://doi.org/10.48550/arXiv.2204.07727
- AGG: https://doi.org/10.18653/v1/2022.acl-long.3
- Hypernym Class Prediction: https://doi.org/10.48550/arXiv.2203.10692
- Syntactic Smoothing: https://doi.org/10.48550/arXiv.2410.11462
- Gradient Grouping: https://doi.org/10.18653/v1/2025.acl-long.164
- Som26 RowNorm: https://doi.org/10.48550/arXiv.2608.22253
- Symmetry-Compatible Optimizers: https://doi.org/10.48550/arXiv.2605.18106
- Hu26 Decoupled Routing: https://doi.org/10.48550/arXiv.2604.22407
- Token Geometry / Ember: https://doi.org/10.48550/arXiv.2607.01455
- MuonIO: https://arxiv.org/abs/2610.02705
- Han26: https://arxiv.org/abs/2610.09835
- LM-head Gradient Bottleneck: https://doi.org/10.48550/arXiv.2603.10145
