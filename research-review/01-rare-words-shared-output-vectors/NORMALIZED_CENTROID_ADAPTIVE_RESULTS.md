# Paper 01 — direct projector versus normalized-centroid adaptive optimizer results

**Run:** [GitHub Actions #44051580714](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44051580714), six jobs successful.  
**Seeds:** 601, 607, 613. **Batches:** 32 and 128. **Budget:** 819,200 tokens per arm. Six arms per seed/batch, 36 runs total.

## Overall held-out cross-entropy

Lower is better. Means are across three paired seeds.

| Head optimizer | Parameterization | Batch 32 | Batch 128 |
|---|---|---:|---:|
| Ember | Plain flat head | 5.00767 | 5.08553 |
| Ember | Direct projector | **4.97100** | **5.06773** |
| Ember | Exact normalized centroid | 4.98760 | 5.06900 |
| Adam | Plain flat head | 5.03373 | 5.12610 |
| Adam | Direct projector | **4.98993** | **5.10557** |
| Adam | Exact normalized centroid | 5.01470 | 5.10753 |

The direct projector beats plain optimization on all 12 matched optimizer/batch/seed comparisons. The exact normalized-centroid parameterization also beats plain optimization on all 12.

## Direct projector versus exact normalized-centroid parameterization

The matched paired difference is direct minus scaled-centroid CE:

| Optimizer | Batch | Mean difference | Per-seed differences |
|---|---:|---:|---|
| Ember | 32 | −0.01660 | −0.0161, −0.0171, −0.0166 |
| Adam | 32 | −0.02477 | −0.0216, −0.0264, −0.0263 |
| Ember | 128 | −0.00127 | +0.0034, −0.0031, −0.0041 |
| Adam | 128 | −0.00197 | +0.0032, −0.0003, −0.0088 |

At batch32, direct projection wins over the exact parameterization for all three seeds under both Ember and Adam. At batch128, the difference is small and seed-inconsistent. Thus the practical advantage is batch-regime dependent, not a general adaptive-optimizer result.

## Frequency-resolved difference

The largest direct-versus-centroid gap is concentrated in zero and low-exposure rows at batch32.

| Exposure | Batch32 Ember direct | Batch32 Ember centroid | Batch32 Adam direct | Batch32 Adam centroid |
|---|---:|---:|---:|---:|
| 0 | **13.0650** | 16.2801 | **12.8056** | 16.6216 |
| 1–9 | **10.8120** | 11.6382 | **10.7955** | 11.6995 |
| 10–31 | **9.7184** | 9.8627 | **9.7372** | 9.9193 |
| 32–99 | 8.4957 | 8.4995 | **8.5290** | 8.5365 |
| 100–315 | 7.1338 | **7.1264** | 7.1548 | **7.1583** |
| 316–999 | 5.6704 | **5.6616** | 5.6620 | **5.6627** |
| 1k–3k | 4.6725 | **4.6581** | 4.7025 | **4.7004** |
| 3k+ | **2.8471** | 2.8519 | 2.8717 | **2.8765** |
| Unknown | **1.6922** | 1.6936 | **1.7005** | 1.7059 |

At batch128, direct-versus-centroid bucket differences are much smaller. This aligns with the overall CE pattern: the direct flat-head implementation's extra benefit over the exact shared-centroid parameterization is primarily a small-batch, rare-row effect.

## Update diagnostics

| Optimizer | Batch | Direct head update norm | Scaled-centroid output-matrix update norm |
|---|---:|---:|---:|
| Ember | 32 | 15.0172 | 14.2603 |
| Adam | 32 | 3.9635 | 4.0281 |
| Ember | 128 | 15.6723 | 16.5186 |
| Adam | 128 | 4.4082 | 4.6342 |

The CE advantage is not explained by a uniformly larger direct update: direct updates are larger under batch32 Ember, slightly smaller under batch32 Adam, and smaller under batch128 for both optimizers.

## Interpretation

1. The fixed-group SGD equivalence is exact and separately verified by a 25-step nonlinear test at <2.3e-15 maximum error.
2. The direct projector and normalized-centroid parameterization are not equivalent under Adam/Ember because their adaptive states live in different coordinates.
3. Direct projection has a reproducible advantage over the exact parameterization at batch32, mainly in zero/low-exposure buckets. At batch128, the two are near-tied.
4. This is evidence for a **practical adaptive-optimizer implementation effect**, not a new SGD geometry. A matched head-LR sweep is required before attributing the batch32 gap to state geometry rather than LR sensitivity.

## Next test

Run a matched head-LR sweep for direct and normalized-centroid parameterizations under Ember and Adam, using LR values 0.025, 0.05, 0.10, with fresh seeds and both batch sizes. Keep the same fixed context groups and token budget. Report all LR curves; do not select a winner on one setting only.
