# Paper 01 — direct projector versus normalized shared-centroid parameterization

**Status: preregistered; fresh seeds 601/607/613; outcomes not inspected.**

## Question

For fixed token groups, the direct projector update

\[
G'=(I+\alpha P)G,\qquad P=AD^{-1}A^\top,\quad D=A^\top A
\]

is exactly equivalent under SGD to the parameterization

\[
W=U+\sqrt{\alpha}\,AD^{-1/2}C,
\]

provided \(U\) and \(C\) use the same learning rate. This study tests whether that equivalence remains close under adaptive optimizers, especially Ember.

## Locked setup

- Same FineWeb train prefix and validation windows as the residual-decomposition study.
- GPT-2 vocabulary 6,000, sequence length 64.
- Model width 32, two layers, two heads.
- Exactly 819,200 training tokens per run.
- Effective batches 32 and 128, accumulated from two microbatches.
- 400 steps at batch32; 100 steps at batch128.
- Fresh seeds 601, 607, 613.
- Fixed train-only context groups, K=64.
- Projector strength alpha=2.
- Head learning rate 0.05; Ember beta2=0.95, epsilon=1e-8; Adam beta1=0.9, beta2=0.95, epsilon=1e-8.
- All arms share initialization, token order, body optimizer, learning-rate schedule, and validation windows.

## Arms

1. Plain Ember.
2. Direct projector + Ember: one flat head \(W\); apply \(G+2PG\) before the Ember update.
3. Normalized-centroid parameterization + Ember: \(W=U+\sqrt{2}AD^{-1/2}C\); initialize \(U=W_0,C=0\); optimize \(U\) and \(C\) with Ember using the same hyperparameters.
4. Plain Adam.
5. Direct projector + Adam.
6. Normalized-centroid parameterization + Adam, with the same \(U,C\) parameterization and optimizer settings.

The direct and factorized arms use exactly the same fixed assignment matrix \(A\). No regrouping occurs in this study.

## Measurements

- Overall held-out CE and target-exposure buckets.
- Mean actual output-matrix update norm.
- Paired same-seed CE differences between direct and normalized-centroid arms for Ember and Adam.
- Optimizer state size and runtime per arm.

## Decision rule

- First verify the analytic SGD identity test in CI.
- If direct and normalized-centroid arms are close under both adaptive optimizers, treat the flat-head projector as a practical implementation of normalized shared-centroid coupling rather than a distinct optimizer geometry.
- If they differ, inspect update norms and frequency buckets, then run a matched head-LR sweep before attributing the difference to optimizer-state geometry.
- Do not infer a general adaptive-optimizer advantage from one fixed learning rate.

## Execution

The experiment uses one fixed train-only assignment per matched seed/batch. Raw logs and JSONL results are archived as Actions artifacts.
