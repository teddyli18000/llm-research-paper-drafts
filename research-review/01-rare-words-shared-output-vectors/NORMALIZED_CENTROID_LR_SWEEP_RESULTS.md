# Paper 01 — matched head-LR sweep of direct class projection versus normalized centroid reparameterization

Date: 2026-11-17. **Newly audited, completed result**, not a proposed experiment.

- GitHub Actions: [#44055310959](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44055310959), **6/6 successful jobs**.
- [Preregistered design](NORMALIZED_CENTROID_LR_SWEEP_PROTOCOL.md) and [exact-SGD equivalence test](../../research-lab/01-rare-words-shared-output-vectors/normalized_centroid_equivalence_test.py); identity also passed in [#44053650775](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44053650775).
- This report independently parsed the `LR_SWEEP` lines in the **six original job logs**, yielding 72 unique arm records, rather than assuming a green workflow meant the claimed performance held.

## Design

Each of three fresh seeds (641, 643, 647) × effective batch (32 with 400 optimizer steps; 128 with 100 steps) trained the same 819,200 GPT-2-token budget on the same FineWeb train prefix. V=6000, width=32, 2 Transformer layers, sequence=64, fixed train-text context groups K=64, alpha=2, untied softmax head. Two head optimizers (official pinned Ember, Adam); three head LR values (0.025, 0.05, 0.10); two parameterizations.

1. **Direct single-head projector:** apply `G'=(I+2P)G` to the head gradient before the optimizer, with `P=A(AᵀA)⁻¹Aᵀ`.
2. **Normalized-centroid reparameterization:** `W=U+sqrt(2) A(AᵀA)⁻¹/² C`, optimize U and C as separate parameters. Under vanilla SGD this gives *exactly the same effective W updates* at fixed groups; the nonlinear adaptive optimizer states generally break equivalence.

The identical LR is applied to the two coordinates U,C in the reparameterized arm, and to W in the direct arm. Both parameterizations receive the same per-head LR grid. These are **adaptive-optimizer implementation comparisons**, not a claim of different SGD geometry.

## Complete mean validation cross-entropy across three paired seeds (lower is better)

| Effective batch | Head optimizer | LR | Direct | Reparameterized | Direct − reparam | Direct wins / 3 |
|---|---|---:|---:|---:|---:|---:|
| 32 | Ember | 0.025 | 4.97993 | 4.99650 | **−0.01657** | 3 |
| 32 | Ember | 0.050 | 4.97063 | 4.98667 | **−0.01603** | 3 |
| 32 | Ember | 0.100 | 4.96817 | 4.98063 | **−0.01247** | 3 |
| 32 | Adam | 0.025 | 4.99617 | 5.01970 | **−0.02353** | 3 |
| 32 | Adam | 0.050 | 4.98970 | 5.01290 | **−0.02320** | 3 |
| 32 | Adam | 0.100 | 4.99303 | 5.01460 | **−0.02157** | 3 |
| 128 | Ember | 0.025 | 5.08600 | 5.08633 | −0.00033 | 1 |
| 128 | Ember | 0.050 | 5.06127 | **5.06017** | +0.00110 | 1 |
| 128 | Ember | 0.100 | **5.05943** | 5.06780 | −0.00837 | 3 |
| 128 | Adam | 0.025 | 5.12563 | 5.13017 | −0.00453 | 2 |
| 128 | Adam | 0.050 | 5.10333 | 5.10503 | −0.00170 | 2 |
| 128 | Adam | 0.100 | **5.09763** | 5.10200 | −0.00437 | 3 |

These numbers are means of the exact 5-decimal `CE` output printed by each job, not a re-aggregation of rounded intermediate summaries. Artifacts and raw JSONL: six `normalized-centroid-lr-sweep-b*-seed-*` uploads on run #44055310959.

## Outcome against the preregistered decision rule

- **Batch 32: positive.** Direct is better in **all 18/18 paired comparisons** (2 optimizers × 3 LRs × 3 seeds). Its best mean CE is better for both optimizers (Ember 4.96817 vs 4.98063; Adam 4.98970 vs 5.01290). The previously observed direct advantage is **not just a one-LR accident**.
- **Batch 128: qualified.** The methods are almost tied at Ember LR 0.025 and 0.05, and direct only decisively wins the three seeds at LR 0.10. Direct also wins 3/3 at Adam LR 0.10, but differences are modest. It does not support a learning-rate-insensitive advantage at high batch.
- Both methods share the same forward softmax, grouping, and **exact fixed-group SGD functional update**. Their differences under Ember/Adam establish *optimizer-state coordinate dependence*. These runs do **not** isolate whether the observed advantage is caused by particular moment statistics, effective step norms, or numerical trajectories.
- This grid uses the same repeatedly examined V6k validation split. It is a preregistered **LR sensitivity check**, not a locked independent heldout test and not proof of large-model transfer.

## Scientific interpretation

A direct, normalized blockwise gradient transform is a **real incremental algorithmic variant** of class sharing. It is not equivalent to a normalized shared-centroid optimizer under Adam/Ember, and batch-32 results show a repeatable advantage across these head LRs. Under SGD it is equivalent, so the work's independent method contribution cannot rest on SGD preconditioning alone.

The decisive independent test already completed at V≈50k found direct projector + Ember **6.29370 CE** on the untouched external shard, versus plain Ember **6.31510**, but near-tied with the original author's centroid + Adam **6.29480** ([full result](INDEPENDENT_HELDOUT_V50K_RESULTS.md)). That near-tie, and the 6.5M-parameter toy contextual backbone, still determine the current publication ceiling. The V6k learning-rate robustness makes the optimizer-coordinate claim stronger; it **does not supersede** the independent V50k near-tie.

**Decision:** accept the batch-32 adaptive-optimizer effect as experimentally supported; reject a blanket claim that direct class projection is superior to shared centroids at all batch regimes or at modern LLM scale. The original author's core adaptive centroid method must receive explicit credit.
