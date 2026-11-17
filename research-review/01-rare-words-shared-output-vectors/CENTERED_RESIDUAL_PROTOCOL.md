# Paper 01 — centered prediction/target residual ablation

**Status: preregistered; fresh seeds 617/619/631; outcomes not inspected.**

## Why this follow-up is necessary

The previous residual decomposition study applied \(Pp\) and \(-Py\) separately. Each has a vocabulary-uniform component \(u=\mathbf1/V\), while the full difference cancels it:

\[
Pp=(Pp-u)+u,\qquad -Py=-(Py-u)-u.
\]

A uniform update adds the same vector to every output row. For a given hidden state this shifts every logit by the same scalar, leaving the current softmax probabilities unchanged, but it can still alter parameter values and adaptive-optimizer state. The previous uncentered one-sided outcomes therefore confound the class residual with this softmax-invariant mode.

This study separates them explicitly.

## Exact decomposition

For each effective batch, let \(\bar h\) be the mean hidden state across prediction positions and define the vocabulary-uniform row correction

\[
U=\frac1V\mathbf1\bar h^\top.
\]

The projected terms are

\[
C_p=Pp\,h^\top,\qquad C_y=Py\,h^\top.
\]

Their centered forms are

\[
\bar C_p=C_p-U,\qquad \bar C_y=C_y-U.
\]

Both centered corrections sum to zero across vocabulary rows, and

\[
\bar C_p-\bar C_y=C_p-C_y=P(p-y)h^\top.
\]

The uniform component \(U\) is softmax-invariant for the current batch: adding it to every output row adds the same scalar to all logits for each hidden state.

## Locked setup

- FineWeb train prefix and validation windows used in the previous residual study.
- GPT-2 vocabulary 6,000, sequence length 64, model width 32, two layers, two heads.
- Exactly 819,200 tokens per run.
- Effective batches 32 and 128, accumulated from two microbatches.
- 400 steps at batch32; 100 steps at batch128.
- Fresh seeds 617, 619, 631.
- Fixed train-only context groups, K=64.
- Official pinned Ember, head LR 0.05, beta2 0.95, epsilon 1e-8.
- Projector strength alpha=2.
- Same initialization, token order, body optimizer and validation windows within each matched seed/batch.

## Arms

1. Plain Ember.
2. Full projector: \(G+2(Pp-Py)h^\top\).
3. Raw prediction-only: \(G+2Pp\,h^\top\).
4. Raw target-only: \(G-2Py\,h^\top\).
5. Centered prediction-only: \(G+2(Pp-U)\,h^\top\).
6. Centered target-only: \(G-2(Py-U)\,h^\top\).
7. Uniform-only plus: \(G+2U\).
8. Uniform-only minus: \(G-2U\).

The uniform-only arms isolate the effect of adding a softmax-invariant common row shift to the adaptive optimizer.

## Measurements

- Overall held-out CE and target-exposure buckets.
- Norms of raw \(Pp\), raw \(Py\), centered \(Pp-U\), centered \(Py-U\), full difference, and uniform component.
- Cosine between raw terms and centered terms.
- Per-step maximum error between direct matrix projection \(PG\) and the explicit centered decomposition.
- Sum across vocabulary rows of each centered correction.
- Paired same-seed CE differences.

## Decision rules

- If centered one-sided arms still lose while the full difference wins, the interaction interpretation is supported.
- If a centered one-sided arm retains the gain, revise the mechanism claim: the previous raw ablation was confounded by the common mode.
- If uniform-only arms alter later training despite leaving the current softmax unchanged, that quantifies optimizer-state sensitivity to a function-invariant direction.
- The full correction must match \(PG\) within \(2\times10^{-5}\) maximum absolute error at every step; both centered terms must sum to zero across vocabulary rows.

## Execution

The workflow runs six jobs, one per seed/batch pair. Each job executes all eight arms and archives raw logs and per-arm JSONL frequency-bucket results.
