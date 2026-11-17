# Paper 01 — head learning-rate sweep for direct versus normalized-centroid updates

**Status: preregistered; fresh seeds 641/643/647; outcomes not inspected.**

## Question

The matched adaptive comparison found a direct-projector advantage over the exact normalized-centroid parameterization at batch32, but near-tie behavior at batch128. This study tests whether that difference survives a small matched head-learning-rate sweep.

## Locked setup

- Same FineWeb train prefix, validation windows, GPT-2 vocabulary 6,000, sequence length 64.
- Model width 32, two layers, two heads.
- Exactly 819,200 tokens per run.
- Effective batch sizes 32 and 128, accumulated from two microbatches.
- 400 steps at batch32; 100 steps at batch128.
- Fresh seeds 641, 643, 647.
- Fixed train-only context groups, K=64; no regrouping.
- Projector alpha=2.
- Head learning rates: 0.025, 0.05, 0.10.
- Ember beta2=0.95, epsilon=1e-8; Adam beta1=0.9, beta2=0.95, epsilon=1e-8.
- Body and embedding optimizers, token order, and all other hyperparameters remain fixed.

## Arms

For each batch, seed, optimizer, and head LR:

1. Direct flat-head projector + Ember.
2. Exact normalized-centroid parameterization \(W=U+\sqrt{2}AD^{-1/2}C\) + Ember.
3. Direct flat-head projector + Adam.
4. Exact normalized-centroid parameterization + Adam.

This is 12 matched arms per seed/batch: two parameterizations × two optimizers × three learning rates.

## Measurements

- Overall held-out CE and target-exposure buckets.
- Output-matrix update norm.
- Paired same-seed direct-minus-centroid CE at each optimizer/LR/batch.
- Full CE-versus-LR curves; no winner selected from a single setting.

## Decision rule

A direct-projector advantage is robust to LR choice if it beats the normalized-centroid parameterization on at least two of three seeds at two or more LR values, and its best mean across the preregistered LR grid is lower. If the advantage appears only at one LR, classify it as LR-sensitive. If batch128 remains near-tied across the grid, retain the batch-regime boundary.

The LR grid is a sensitivity study, not a large-scale optimizer tuning benchmark.

## Execution

Six jobs cover three fresh seeds and two batch sizes. Each job runs all 12 matched arms and archives per-arm JSONL scores and frequency buckets.
