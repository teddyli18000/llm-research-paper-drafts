# Paper 01 — absolute-step context-to-dynamic regrouping schedule

**Status: preregistered; fresh seeds 569/571/577; outcomes not inspected.**

## Question

The residual-decomposition study found that fixed train-only context groups beat online K-means at both batch sizes. A context-to-dynamic switch at absolute step 200 improved batch32, while batch128 (100 steps total) remained on context groups and matched the fixed-context arm. The earlier 20%-of-run switch was too early at batch128.

This experiment tests whether the batch32 gain depends on the absolute switch point.

## Locked setup

- Same FineWeb training prefix and validation windows as the residual-decomposition study.
- GPT-2 vocabulary 6,000, sequence length 64.
- Model width 32, two layers, two heads.
- Exactly 819,200 training tokens per run.
- Effective batch 32, accumulated from two microbatches.
- Exactly 400 optimizer steps.
- Fresh model seeds 569, 571, 577.
- Official pinned Token Geometry/Ember implementation, head LR 0.05, beta2 0.95, epsilon 1e-8.
- Projector strength alpha 2, K=64.
- All arms share initialization, token order, schedule, body optimizer, and validation windows.

## Arms

1. Fixed train-only context groups for all 400 steps.
2. Context groups until absolute step 150, then output-row K-means, refreshed every 50 steps.
3. Context groups until absolute step 200, then output-row K-means, refreshed every 50 steps.
4. Context groups until absolute step 250, then output-row K-means, refreshed every 50 steps.

At the switch, K-means is initialized from the current output rows. Group changes affect only the gradient projector; the model's forward logits and output matrix are unchanged.

## Measurements

- Overall held-out CE and target-exposure buckets.
- Mean projected-to-raw gradient norm ratio.
- Mean actual LM-head update norm.
- Per-seed paired CE differences against fixed context.

## Decision rule

Retain a switch point only if it beats fixed context on at least two of three seeds and has a lower three-seed mean CE. If multiple thresholds are within 0.002 CE, prefer the later threshold because it preserves the context prior longer. If no switch threshold meets the rule, keep fixed context.

The batch128 schedule remains fixed context: its 100-step horizon never reaches the tested absolute thresholds.

## Execution

The workflow runs three fresh-seed jobs. Each job executes all four matched arms and archives the per-arm JSONL scores, frequency buckets, and update diagnostics.
