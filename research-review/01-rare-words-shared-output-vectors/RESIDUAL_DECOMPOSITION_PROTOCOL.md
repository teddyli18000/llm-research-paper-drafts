# Paper 01 — prediction-side versus target-side class residual decomposition

**Status: preregistered; fresh seeds 547/557/563; outcomes not inspected.**

## Question

The class-projected output-head gradient has an exact decomposition:

\[
G=(p-y)h^\top,\qquad
G'=(I+\alpha P)G
=\left[(p-y)+\alpha Pp-\alpha Py\right]h^\top.
\]

Which term carries the validation gain?

- **Prediction-side residual:** \(Pp\,h^\top\).
- **Target-side residual:** \(-Py\,h^\top\).
- **Full class residual:** \((Pp-Py)h^\top=PG\).

Standard within-class label smoothing redistributes target mass and has no prediction-side \(Pp\) term. The current experiment therefore separates the two terms rather than treating label smoothing as an exact collision.

## Exact decomposition used in the runtime

For each microbatch with hidden states \(H\), softmax probabilities \(Q\), target indices \(t\), and group assignment \(a\):

\[
C^{(p)}_g=\sum_s H_s\left(\sum_{i:a_i=g}Q_{si}\right),
\]
\[
C^{(y)}_g=\sum_{s:a_{t_s}=g}H_s.
\]

The row corrections for token \(i\) in group \(g=a_i\) are

\[
(Pp\,h^\top)_i=C^{(p)}_g/(N|g|),\qquad
(Py\,h^\top)_i=C^{(y)}_g/(N|g|),
\]

where \(N\) is the effective batch's number of prediction positions. Corrections are summed across both accumulated microbatches. The full correction is checked against a direct matrix projection \(PG\) at every optimizer step.

## Locked setup

- FineWeb training prefix, GPT-2 vocabulary 6,000, sequence length 64.
- Model width 32, two layers, two heads.
- Exactly 819,200 training tokens per run.
- Effective batch sizes 32 and 128, each accumulated from two microbatches.
- 400 steps at batch 32; 100 steps at batch 128.
- Fresh seeds 547, 557, 563.
- Official pinned Token Geometry/Ember implementation, LR 0.05, beta2 0.95, epsilon 1e-8.
- Projector alpha 2, K=64.
- All arms share model initialization, data order, learning-rate schedule, body optimizer and validation windows.

## Group schedules

1. **Fixed context:** train-only co-occurrence groups throughout.
2. **Dynamic:** K-means initialized from the initial output rows before the first update, refreshed every 50 steps.
3. **Context-to-dynamic:** context groups until absolute optimizer step 200, then K-means refreshed every 50 steps. At batch 128 the run has only 100 steps and remains on context groups.

The absolute-step schedule tests the observed batch-regime interaction without switching the high-batch run before it has accumulated enough optimizer updates.

## Arms

1. Plain Ember.
2. Original shared-centroid parameterization + Ember.
3. Fixed-context full projector: \(G+2P(p-y)h^\top\).
4. Fixed-context prediction-only: \(G+2Pp\,h^\top\).
5. Fixed-context target-only: \(G-2Py\,h^\top\).
6. Dynamic full projector.
7. Context-to-dynamic full projector, switching at step 200.
8. Full context-group soft-target cross-entropy with \(\epsilon=2/3\).

The epsilon value matches the target-mass redistribution coefficient \(\alpha/(1+\alpha)\) for \(\alpha=2\). It is a target-side comparator, not an algebraic equivalent of the full projector.

## Measurements

- Overall held-out CE and every original target-exposure bucket.
- Per-step norms of \(Pp\,h^\top\), \(Py\,h^\top\), and their difference.
- Cosine between the prediction-side and target-side corrections.
- Maximum error between the direct projection \(PG\) and the explicit residual decomposition.
- Paired seed differences for prediction-only, target-only and full projector.

## Decision rules

- If prediction-only retains the full projector's gain while target-only does not, the prediction-side class residual is the main mechanism and is absent from standard target smoothing.
- If target-only retains the gain, the overlap with semantic target redistribution is closer than currently believed.
- If only the full combination works, the interaction between prediction and target residuals is essential.
- The full projector must reproduce the direct matrix projection within \(2\times10^{-5}\) maximum absolute error at every step.
- Context-to-dynamic is a schedule control; it must beat fixed context at batch 32 without degrading batch 128 to be retained as a general schedule.

## Prior-art boundary

- Syntactic Smoothing (EMNLP 2024) redistributes target mass through syntactic similarity; its gradient has no \(Pp\) term.
- AGG (ACL 2022) gates frequency-defined rare/very-rare output-gradient components; it does not apply the semantic class residual \(P(p-y)\).
- Tree Loss (AISTATS 2022) establishes class-shared parameterization and the SGD reparameterization collision.
- The 2026 LM-head optimizer and rare-token prior-art audit is in 2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md.

The experiment tests the specific role of the prediction-side class residual. It does not claim generic class sharing, target smoothing, or adaptive moment routing as new.

## Execution

The workflow is registered on the repository default branch and runs against research/01-modern-optimizer-baselines. The protocol and analytic identity test are committed before the fresh-seed matrix. Raw logs and frequency-bucket JSONL files are archived as Actions artifacts.

The registered Actions launcher uses the existing research workflow path `.github/workflows/research-01-stable-capture.yml` and runs the matrix on this branch.
