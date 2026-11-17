# Paper 01 — centered prediction/target residual results

**Run:** [GitHub Actions #44052591527](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44052591527), six jobs successful.  
**Seeds:** 617, 619, 631. **Batches:** 32 and 128. **Budget:** 819,200 tokens per arm. Eight arms per seed/batch, 48 runs total.

## Overall held-out cross-entropy

Lower is better. Means are across three paired seeds.

| Arm | Batch32 | Batch128 |
|---|---:|---:|
| Plain Ember | 5.00410 | 5.08187 |
| Full projector \(G+2(Pp-Py)h^\top\) | **4.96977** | **5.06223** |
| Raw prediction-only \(G+2Pp\,h^\top\) | 5.63463 | 5.62820 |
| Raw target-only \(G-2Py\,h^\top\) | 6.05157 | 5.98537 |
| Centered prediction-only \(G+2(Pp-U)h^\top\) | 6.24463 | 6.26007 |
| Centered target-only \(G-2(Py-U)h^\top\) | 6.04097 | 6.08003 |
| Uniform-only plus \(G+2U\) | 5.10037 | 5.17973 |
| Uniform-only minus \(G-2U\) | 5.93007 | 6.01323 |

The full projector beats plain Ember on all three seeds at both batches. Both centered one-sided terms lose badly on all seeds at both batches. The centered ablation therefore supports the conclusion that **the useful update is the difference between prediction-side and target-side class residuals, not either term independently**.

## What the uniform-mode control shows

Let \(U=(1/V)\mathbf1\bar h^\top\). A gradient update in this row-uniform direction shifts all logits by the same scalar for each hidden state, so it is invisible to the current softmax function under a plain SGD step. Ember's factored preconditioner does not preserve that row-uniform direction in its actual parameter update.

The uniform-only controls confirm this matters:

- \(G+2U\) adds +0.09627 CE at batch32 and +0.09787 at batch128.
- \(G-2U\) adds +0.92597 CE at batch32 and +0.93137 at batch128.

The common mode is therefore not harmless once it passes through adaptive optimizer state. It explains why the original uncentered one-sided ablation was confounded, but it is not the whole explanation: after removing U, both centered one-sided terms still perform much worse than the full difference.

## Geometry diagnostics

| Diagnostic | Batch32 | Batch128 |
|---|---:|---:|
| Mean \(\|Pp-U\|\) | 0.06289 | 0.10281 |
| Mean \(\|Py-U\|\) | 0.06601 | 0.10702 |
| Mean \(\|U\|\) | 0.02018 | 0.03374 |
| Raw prediction/target cosine | 0.96863 | 0.95504 |
| Centered prediction/target cosine | 0.96947 | 0.95734 |
| Maximum decomposition error | \(3.26\times10^{-9}\) | \(6.06\times10^{-9}\) |

The centered terms remain highly aligned. Their difference is substantially smaller than either term. The full projector extracts the class-level prediction-versus-target discrepancy after cancellation of the shared component.

## Frequency-resolved result

At batch32, the full projector reduces zero-exposure validation CE from 16.7884 to 13.1867 and improves the 1–9 exposure bucket from 11.7420 to 10.8522. The centered one-sided arms can improve the zero-exposure bucket, but severely damage the rest of the exposure distribution; for example, their 316–999 bucket CE rises to 8.5635/6.1328 versus 5.6631 for plain Ember. The uniform-only arms also damage the tail, especially the negative uniform direction.

At batch128, the same pattern holds: the full projector improves zero and low-exposure buckets, while centered one-sided and uniform-only controls degrade broad portions of the distribution.

## Decision

1. **Retain the full projector result.** Its overall gain survives the centered-control study at both batches.
2. **The cancellation mechanism is supported.** Prediction-only and target-only class residuals fail independently even after the vocabulary-uniform mode is removed.
3. **Adaptive optimizer state is sensitive to function-invariant gradient directions.** The uniform-only controls quantify this effect.
4. Do not use the first uncentered one-sided results as standalone causal evidence; use this centered study instead.
5. Continue with the matched head-LR sweep and graph-diffusion proper-score collision before making a novelty claim.
