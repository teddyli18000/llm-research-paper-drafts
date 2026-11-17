# Paper 01 — cross-microbatch reliability-gated class-gradient projection

**Status: preregistered experiment; no outcome inspected.** This tests a new extension of the original shared-output-vector method. It does not claim that gradient agreement itself is new.

## Question

Does cross-microbatch agreement identify which token-group projector transfers useful output-head gradient signal, and does reliability-weighted projection outperform a fixed-strength projector or a current-batch energy gate?

## Estimator

For a candidate partition A, let P_A be the orthogonal projection that replaces each output row's gradient with the mean gradient of its group. At a single frozen parameter state, split the effective training batch into two disjoint microbatches and compute the ordinary, unprojected LM-head gradients G_1 and G_2. Define

\[
\rho_A=\frac{\langle P_A G_1,P_A G_2\rangle_F}{\tfrac12(\|P_A G_1\|_F^2+\|P_A G_2\|_F^2)+\epsilon}.
\]

By Cauchy–Schwarz, \(\rho_A\in[-1,1]\). If \(G_b=\mu+\xi_b\), with independent zero-mean microbatch noise conditional on the current model and partition, then the expected numerator is \(\|P_A\mu\|_F^2\); the expected denominator is signal plus projected noise energy. Thus the ratio estimates the reliable component of the group-projected gradient.

## Locked training rule

- Use the same original FineWeb token stream, GPT-2 vocabulary 6,000, width 32, 2 layers, 2 heads, 64-token sequences, 819,200 tokens per run, official pinned Ember, output LR 0.05, alpha maximum 2, K=64.
- Seeds: 443, 449, 457. Effective batch sizes: 32 and 128. Both are split into exactly two equal microbatches; all arms use the same deterministic within-batch sequence permutation and gradient accumulation.
- Candidate partitions: fixed train-only context groups and current-head K-means groups. K-means is refreshed every 50 optimizer steps, before either current microbatch is seen. Both gate EMAs are reset whenever K-means refreshes the dynamic partition, then initialized from the same new microbatch pair. No score is carried across a partition change.
- The two microbatch gradients are computed at the same parameter state. Their sum is the ordinary effective-batch gradient. No optimizer step occurs between them.
- Both gates use lagged EMAs with beta 0.9; at each partition refresh, both EMAs restart from the same current-step observation. At step t, it selects the partition with the larger EMA from earlier steps; the current pair updates the EMA only after the current projection has been applied. Ties select context groups. Step 0 and the first step after each K-means refresh warm-start with context groups and alpha=2; the current pair initializes both scores for the new comparison window.
- The selected update is \(G' = G + 2\max(\rho_{selected},0)P_{selected}G\), where the coefficient uses the lagged EMA reliability, clipped to [0,1].

## Arms

1. Plain Ember.
2. Fixed context-group projector, alpha 2.
3. Dynamic K-means projector, alpha 2.
4. Corrected lagged projected-gradient-energy gate (the score is \(\|PG\|^2/\|G\|^2\), not cross-microbatch coherence), alpha 2.
5. Cross-microbatch partition selection with fixed alpha 2 (selection-only ablation).
6. New cross-microbatch reliability gate above (selection plus reliability-scaled alpha).

## Measurements

- Overall held-out CE and all original target-frequency bins.
- Per-step \(\rho_{context}\), \(\rho_{dynamic}\), selected partition, selected reliability, alpha, and selection fraction.
- Runtime overhead relative to the same two-microbatch accumulation in the other arms.
- Same-seed paired differences at both batch sizes.

## Prior art that defines the novelty boundary

- The Tree Loss (Wang & Izbicki, AISTATS 2022) establishes class-structured parameter sharing and the SGD reparameterization connection.
- Softmax Gradient Tampering (Das et al., ICLR 2022; arXiv:2111.12495) establishes custom softmax backward-gradient transformations without changing the forward pass.
- Gradient Agreement Filtering (Chaubard et al., 2024; arXiv:2412.18052) establishes using agreement between microbatch gradients to filter updates.
- Syntactic Smoothing (Diehl Martinez et al., EMNLP 2024) and Beyond Tokens (Iyer et al., EACL 2026) establish related token-group and concept-level training signals.
- MuonIO (Ma et al., 2026; arXiv:2610.02705) is a strong modern optimizer for LM-head rows; it uses momentum followed by row normalization.
- Why Adaptive Optimizers Underestimate Rare Tokens (Kar, 2026; arXiv:2609.37535) analyzes rare-token underprediction and shows why vocabulary-coupled second moments, AMSGrad, and linear/orthogonalized updates behave differently from coordinate-wise Adam.

The candidate contribution is therefore not generic gradient tampering, class sharing, or gradient agreement by itself. It is the reliability estimate for a *token-group output-head subspace* and its use to choose and scale the group-sharing update.

## Interpretation rules

- Compare against context and dynamic projectors, not only plain Ember.
- A gain on batch 32 but not batch 128 is a batch-regime result, not a general optimizer improvement.
- A higher rho without better held-out CE does not establish useful transfer.
- Cross-microbatch agreement filtering is established prior art (e.g. Gradient Agreement Filtering, 2024). The candidate contribution is the group-subspace reliability estimator and its use to select/scale token-row sharing; this is a narrow extension that must beat the energy gate and fixed groups.
- No hyperparameter selection uses the held-out validation set. A positive result requires a new independent test shard and a separate scale replication.

## Execution

The launcher is registered on the repository's default branch and runs against research/01-modern-optimizer-baselines. Changes to this protocol or the experiment scripts on that branch start the six matched seed/batch jobs. Raw logs and JSONL results are archived as Actions artifacts.
