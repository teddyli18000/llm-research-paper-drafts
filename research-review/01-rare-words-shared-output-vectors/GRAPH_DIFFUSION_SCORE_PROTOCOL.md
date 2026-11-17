# Paper 01 — graph-diffusion proper-score collision

**Status: preregistered; fresh seeds 683/691/701; outcomes not inspected.**

## Question

Hilfiker et al. (2026) show that a graph-diffusion proper score has probability-space gradient \(2H_\tau(p-y)\). For a disjoint union of complete within-class cliques, \(H_\tau\to P=AD^{-1}A^\top\) as \(\tau\to\infty\). This is the closest loss-level mathematical neighbor to the class projector.

The key distinction is the softmax chain rule:

- Direct optimizer update: \(G'=(I+2P)G\), corresponding to logit residual \(r+2Pr\).
- Graph-score loss: \(CE+\lambda r^\top H_\tau r\), whose logit gradient is \(r+2\lambda J_{\rm softmax}H_\tau r\).

This experiment tests whether the proper-score alternatives reproduce the projector's training effect.

## Locked setup

- FineWeb train prefix, GPT-2 vocabulary 6,000, sequence length 64.
- Model width 32, two layers, two heads.
- Exactly 819,200 tokens per run.
- Effective batches 32 and 128, accumulated from two microbatches.
- 400 steps at batch32; 100 steps at batch128.
- Fresh seeds 683, 691, 701.
- Fixed train-only context groups, K=64.
- Official pinned Ember, head LR 0.05, beta2 0.95, epsilon 1e-8.
- Projector alpha=2; graph-score coefficient lambda=1; diffusion tau=0.02.
- Graph score is head-only: its auxiliary logits use detached hidden states, matching the projector's output-head-only scope. The ordinary CE term still trains the full model.

## Score definitions

Let \(r=p-y\), and \(P=AD^{-1}A^\top\).

1. **P-limit score:** \(S_P=r^\top P r\). This is the \(\tau\to\infty\) diffusion-score limit on disjoint clique components. It is proper but not strictly proper because it ignores within-group redistributions.
2. **Finite diffusion score:** \(S_{\rm diff}=r^\top H_\tau r\), with
   \[
   H_\tau=P+\sum_g e^{-\tau n_g/2}(I-P_g),
   \]
   where \(P_g\) is the mean projector for group g and \(n_g\) its size. This is the exact heat kernel for a disjoint union of complete cliques and is strictly proper for finite tau.
3. **Resistance score:** \(S_{\rm res}=r^\top L^\dagger r\), with
   \[
   L^\dagger=\sum_g\frac1{n_g}(I-P_g).
   \]
   This is the within-group complement control.

Each graph-score arm optimizes \(CE+\lambda S\), with lambda=1.

## Arms

1. Plain Ember.
2. Direct class projector + Ember: apply \(G+2PG\) to the output-head gradient.
3. P-limit proper score + Ember, head-only.
4. Finite graph-diffusion score, tau=0.02 + Ember, head-only.
5. Resistance score + Ember, head-only.

## Measurements

- Overall held-out CE and target-exposure buckets.
- Mean output-head update norm.
- Paired same-seed CE differences against direct projection and plain Ember.
- Per-exposure behavior, especially zero/low exposure.

## Decision rule

The P-limit score is the closest scalar-loss collision. If it matches the direct projector, the observed gain may be reproduced by a proper loss once the softmax Jacobian is included. If it does not, the logit-space preconditioning distinction remains empirically relevant.

The finite diffusion and resistance scores diagnose whether within-group contrast directions contribute. This is a fixed-coefficient collision test, not a broad loss hyperparameter search.

## Execution

Six jobs cover three fresh seeds and two batch sizes. Each job runs all five matched arms and archives per-arm JSONL scores and frequency buckets.
