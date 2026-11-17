# Paper 01 — stable projected-gradient capture gate

**Status: preregistered follow-up; fresh seeds; no outcomes inspected.** This tests a new estimator motivated by the completed microbatch-reliability run. The previous rho gate did not improve held-out CE consistently.

## Hypothesis

The projected-gradient reliability score

\[
\rho_A=\frac{\langle P_A G_1,P_A G_2\rangle_F}{\tfrac12(\|P_A G_1\|_F^2+\|P_A G_2\|_F^2)+\epsilon}
\]

measures purity within a group subspace but ignores how much total gradient energy that subspace captures. A small, clean component can therefore outrank a larger useful component.

Test the stable-capture score

\[
\eta_A=\frac{\langle P_A G_1,P_A G_2\rangle_F}{\tfrac12(\|G_1\|_F^2+\|G_2\|_F^2)+\epsilon}.
\]

Because \(P_A\) is an orthogonal projection, \(\eta_A\in[-1,1]\). It is the product of within-subspace reliability and projected-energy capture:

\[
\eta_A=\rho_A\cdot\frac{\tfrac12(\|P_A G_1\|_F^2+\|P_A G_2\|_F^2)}{\tfrac12(\|G_1\|_F^2+\|G_2\|_F^2)}.
\]

Under \(G_b=\mu+\xi_b\), with independent zero-mean microbatch noise conditional on the current model and partition, the expected numerator is \(\|P_A\mu\|_F^2\), while the denominator is total gradient signal plus noise energy. Thus eta estimates the stable gradient energy captured by the group subspace as a fraction of total gradient energy.

## Locked experiment

- Fresh seeds: 461, 467, 479.
- Effective batch sizes: 32 and 128; exactly two equal microbatches per effective batch.
- Same original FineWeb stream, GPT-2 vocabulary 6,000, width 32, 2 layers, 2 heads, sequence length 64, 819,200 training tokens per run, pinned Ember head optimizer, output LR 0.05, K=64.
- All arms use the same deterministic within-batch sequence permutation and two-microbatch gradient accumulation.
- Candidate partitions are fixed train-only context groups and current-head K-means groups. K-means refreshes every 50 optimizer steps before the current microbatch pair is observed. Both candidate-score EMAs reset at each refresh and initialize from the same current pair. Ties select context groups.
- EMA beta is 0.9. The current pair updates the score only after the current projection update is applied. At each step, the partition with the higher lagged EMA is selected; ties select context groups.
- Fixed projection strength is alpha=2. The eta-positive gate applies alpha=2 only when the selected lagged eta is positive; otherwise it applies no group projection. Eta is not used as a continuous multiplier because its magnitude is a capture fraction, not an optimal step-size estimate.

## Arms

1. Plain Ember.
2. Fixed context-group projector, alpha 2.
3. Dynamic K-means projector, alpha 2.
4. Corrected projected-gradient-energy gate: \(\|PG\|^2/\|G\|^2\), alpha 2.
5. Rho selection: choose by \(\rho_A\), alpha 2.
6. Cosine selection: choose by cosine similarity of \(P_A G_1\) and \(P_A G_2\), alpha 2.
7. Eta selection: choose by \(\eta_A\), alpha 2.
8. Eta-positive gate: choose by \(\eta_A\), alpha 2 if the selected lagged eta is positive, otherwise alpha 0.

## Measurements

- Overall held-out CE and every original target-frequency bucket.
- Mean rho, cosine and eta by candidate partition; selected-partition fraction; alpha; gate traces.
- Same-seed paired differences for batch32 and batch128.
- Training wall time and overhead from the extra microbatch projections.

## Prior-art boundary

- Gradient Agreement Filtering (arXiv:2412.18052) uses whole-model microbatch-gradient cosine agreement to filter updates. The cosine arm is an explicit local comparator.
- Softmax Gradient Tampering (ICLR 2022; arXiv:2111.12495) establishes custom softmax backward transformations.
- ACL 2022 AGG establishes recent-frequency-based grouping and gating of rare output-embedding gradient components. It is a close adjacent method, not semantic cross-token gradient transfer.
- The Tree Loss (AISTATS 2022) establishes class-structured parameter sharing and the SGD reparameterization connection.
- MuonIO (arXiv:2610.02705) and the 2026 rare-token optimizer analysis are strong optimizer-level competitors.

The candidate contribution is the stable projected-gradient capture score for choosing a semantic token-group subspace. It is not generic gradient agreement or rare-token gradient gating.

## Decision rule

Eta must beat the corrected energy gate and both rho/cosine controls on matched seeds at both batch sizes, with improvements visible in held-out frequency buckets. If it does not, stop the global partition-selection line and move to a different mechanism rather than tuning this score on the same validation set.


## Execution

The launcher is registered on the default branch and runs against research/01-modern-optimizer-baselines. This protocol and the estimator tests were committed before the fresh-seed run. Raw logs and JSONL frequency-bucket results are archived as Actions artifacts.
