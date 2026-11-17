# Paper 01 — cross-microbatch reliability experiment results

**Run:** [GitHub Actions #44034386895](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44034386895), all six jobs successful.  
**Design:** seeds 443/449/457 × effective batch 32/128; 819,200 training tokens per arm; six matched arms; two microbatches at the same parameter state.

## Overall held-out cross-entropy

Lower is better.

| Arm | Batch 32 mean CE | Batch 128 mean CE |
|---|---:|---:|
| Plain Ember | 5.00183 | 5.08377 |
| Fixed context-group projector | 4.96767 | 5.06203 |
| Dynamic K-means projector | 4.97640 | 5.08907 |
| Lagged projected-energy gate | **4.96393** | 5.06203 |
| Cross-microbatch rho selection, fixed alpha 2 | 4.96533 | 5.06243 |
| Cross-microbatch rho selection, alpha scaled by rho | 4.96870 | **5.06157** |

The rho-scaled method does not produce a stable gain. At batch 32 it is worse than the fixed context projector by 0.00103 CE. At batch 128 it is better by only 0.00047. The rho-selection-only arm improves batch 32 by 0.00233 but is 0.00040 worse at batch 128. The energy gate improves batch 32 by 0.00373 and exactly matches context at batch 128 because it never selects dynamic groups.

## What the reliability score measured

Mean cross-microbatch rho:

| Batch | Context groups | Dynamic K-means groups |
|---|---:|---:|
| 32 | 0.57604 | 0.56227 |
| 128 | 0.71266 | 0.72341 |

The context groups are more internally reliable at batch 32; dynamic groups are marginally more reliable at batch 128. This ranking does not predict held-out CE by itself. The rho-scaled gate also reduces the effective projector coefficient (mean alpha 1.27 at batch 32 and 1.70 at batch 128), which weakens a fixed-alpha update that was already useful.

## Frequency-resolved result

Bucket CE differences are weighted by the validation target mass to show their contribution to the overall CE change. Positive contribution means the method improves on the fixed context projector.

### Dynamic K-means projector versus fixed context projector

| Training target exposures | Batch 32 CE difference | Batch 32 contribution | Batch 128 CE difference | Batch 128 contribution |
|---|---:|---:|---:|---:|
| 0 | -1.434 | +0.00014 | -1.069 | +0.00011 |
| 1–9 | -0.246 | +0.00170 | -0.035 | +0.00024 |
| 10–31 | -0.050 | +0.00399 | +0.098 | -0.00780 |
| 32–99 | +0.051 | -0.00648 | +0.095 | -0.01198 |
| 100–315 | +0.042 | -0.00523 | +0.041 | -0.00507 |
| 316–999 | +0.003 | -0.00030 | +0.007 | -0.00078 |
| 1k–3k | +0.025 | -0.00229 | +0.005 | -0.00048 |
| 3k+ | -0.002 | +0.00059 | 0.000 | -0.00008 |

The dynamic projector sharply improves the zero- and 1–9-exposure bucket, but those buckets carry very little validation mass in this V6k setup. It harms the much larger 32–99 and 100–315 buckets. At batch 128, the 10–315 exposure penalties dominate the aggregate loss gap.

The energy gate's batch-32 improvement comes mainly from the 10–31 bucket (+0.00538 weighted contribution) and 3k+ (+0.00169); at batch 128 it chooses context for all steps and therefore reproduces the context result exactly.

## Interpretation

1. **A single grouping is not uniformly best across exposure ranges.** Dynamic K-means is useful for the extreme low-exposure tail but causes negative transfer in the 32–315 range. Context groups are better for the middle buckets in this run.
2. **The rho score measures purity, not useful signal amount.** It asks how reliable the projected gradient is within a subspace, but a small, clean component can outrank a larger component that captures more stable gradient energy.
3. **The current cross-microbatch rho gate is not a viable standalone contribution.** It has no robust overall advantage over context or the energy gate.
4. **The next falsifiable estimator is stable gradient capture:**

\[
\eta_A=\frac{\langle P_A G_1,P_A G_2\rangle_F}
{\tfrac12(\|G_1\|_F^2+\|G_2\|_F^2)+\epsilon}.
\]

Under independent zero-mean microbatch noise, the expected numerator is the stable gradient energy captured by the group subspace, while the denominator is total gradient energy. Unlike rho, eta combines within-subspace agreement with the amount of gradient energy captured. A fresh-seed experiment will compare eta with rho and cosine agreement before deciding whether this line has value.

## Artifacts

The six Actions artifacts contain the raw JSONL result rows, all frequency buckets, gate traces, and logs. Artifact IDs: 14307510830, 14307455963, 14307286014, 14306926953, 14306439047, 14306298744.
