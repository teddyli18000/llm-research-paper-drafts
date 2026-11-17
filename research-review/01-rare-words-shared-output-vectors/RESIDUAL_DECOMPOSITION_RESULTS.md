# Paper 01 — prediction-side versus target-side residual decomposition results

**Run:** [GitHub Actions #44046155032](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44046155032), six jobs successful.  
**Seeds:** 547, 557, 563. **Batches:** 32 and 128. **Budget:** 819,200 tokens per arm. Eight arms per seed/batch, 48 runs total.

## Main result

Lower held-out cross-entropy is better. Each entry is the mean of three paired seeds.

| Arm | Batch 32 | Batch 128 |
|---|---:|---:|
| Plain Ember | 5.00317 | 5.08033 |
| Upstream centroid + Ember | 4.97940 | 5.07497 |
| Fixed-context full projector \(G+2PG\) | **4.96840** | **5.06013** |
| Prediction-only \(G+2Pp\,h^\top\) | 5.62083 | 5.61930 |
| Target-only \(G-2Py\,h^\top\) | 6.05603 | 6.01383 |
| Context-group soft-target CE, \(\epsilon=2/3\) | 5.51073 | 5.58250 |
| Dynamic K-means full projector | 4.97363 | 5.08167 |
| Context-to-dynamic at absolute step 200 | **4.96340** | 5.06013 |

The full projector beats plain Ember on all three seeds at both batch sizes (mean differences −0.03477 and −0.02020). It also beats the upstream centroid baseline on all six matched runs (−0.01100 and −0.01483).

Neither component reproduces the result:
- Prediction-only is worse than the full projector by +0.65243 CE at batch 32 and +0.55917 at batch 128; all three paired seeds agree.
- Target-only is worse by +1.08763 and +0.95370; all three paired seeds agree.
- The matched group-label smoothing arm is worse by +0.54233 and +0.52237.

The result supports an **interaction between prediction-side and target-side class residuals**. It does not support either term as an independently useful update.

## Geometry of the two terms

For one position, with token-row gradient \(G=(p-y)h^\top\),

\[
PG=Pp\,h^\top-Py\,h^\top.
\]

The measured mean norms and cosine between the two corrections are:

| Statistic | Batch 32 | Batch 128 |
|---|---:|---:|
| \(\|Pp\,h^\top\|_F\), mean per-step norm | 0.06660 | 0.11060 |
| \(\|Py\,h^\top\|_F\), mean per-step norm | 0.06933 | 0.11425 |
| Cosine, mean per-step | 0.96948 | 0.95703 |

The terms have similar magnitudes and are strongly aligned. Applying either one alone introduces a large coherent shift. Their difference is much smaller and retains the class-level prediction-versus-target discrepancy. The full projector is therefore not “prediction smoothing” or “target smoothing”; its useful signal is the **difference between the two projected masses**.

The direct matrix projection and the explicit decomposition agree to maximum absolute error below \(1.6\times10^{-8}\) in the analytic identity test and below \(2\times10^{-5}\) at every training step. All six CI jobs passed this assertion.

## Context and grouping schedule

- Fixed train-only context groups beat online K-means in both regimes: by 0.00523 CE at batch 32 and 0.02153 at batch 128.
- Switching context groups to online K-means at absolute step 200 improves batch 32 from 4.96840 to 4.96340; all three seeds improve by 0.0048–0.0053.
- Batch 128 has only 100 optimizer steps, so the step-200 schedule remains on context groups and exactly matches the fixed-context result.
- This is a more coherent schedule than switching after a fixed fraction of training, which switched too early in the 100-step regime.

The next schedule test should compare step 150, 200, 250 and never-switch at batch 32, while keeping batch 128 on context groups. The current three-seed result is promising but does not establish the optimal switch point.

## Frequency buckets

Validation target mass is heavily concentrated outside the zero-exposure bucket: zero-exposure tokens account for only 0.01% of validation targets. The full context projector improves the zero-exposure bucket substantially, but its overall gain is not limited to that bucket.

| Exposure in training | Batch 32 plain | Batch 32 full projector | Batch 128 plain | Batch 128 full projector |
|---|---:|---:|---:|---:|
| 0 | 16.5955 | 12.9175 | 13.7974 | 12.9761 |
| 1–9 | 11.7462 | 10.8429 | 11.0768 | 10.9422 |
| 10–31 | 9.9299 | 9.7270 | 9.8385 | 9.7647 |
| 32–99 | 8.5405 | 8.4930 | 8.6166 | 8.5669 |
| 100–315 | 7.1455 | 7.1301 | 7.2699 | 7.2342 |
| 316–999 | 5.6722 | 5.6653 | 5.8349 | 5.8115 |
| 1k–3k | 4.6723 | 4.6666 | 4.8133 | 4.7984 |
| 3k+ | 2.8488 | 2.8447 | 2.9542 | 2.9558 |
| Unknown | 1.6960 | 1.6867 | 1.7095 | 1.7140 |

The prediction-only arm improves zero-exposure tokens but degrades most other buckets. The target-only arm also improves zero-exposure tokens while severely worsening the common/unknown tail, especially the unknown bucket. The full difference avoids these one-sided distortions.

## Interpretation and novelty boundary

1. **Mechanism:** the useful object is \(P(p-y)\), not \(Pp\) or \(-Py\) in isolation. The experiment isolates this algebraically and empirically.
2. **Target smoothing:** a class-uniform target distribution changes only the target vector. It cannot reproduce the prediction-side projection, and the tested matched smoothing arm is much worse.
3. **Prior art:** Tree Loss induces a path-incidence Gram preconditioner \(SS^\top G\) through parameter sharing; it is not generally the normalized orthogonal projector \(P G\). Syntactic smoothing changes target mass. AGG gates selected rare-token negative-gradient components. Softmax Gradient Tampering applies a nonlinear power transform to probabilities. None of these full texts describes the exact flat-head operation \(G+\alpha A(A^\top A)^{-1}A^\top G\).
4. **Claim scope:** this is evidence for a normalized blockwise class-residual preconditioner on a flat untied output head. It is not evidence that class sharing, gradient modification, target smoothing, or LM-head preconditioning generally are new.
5. **Remaining work:** test the switch point; compare with output-only epsilon, projected RowNorm and MuonIO; validate on an untouched data shard and a larger model; finish the full-text audit of cluster-soft-target and metric-augmentation methods whose PDFs were unavailable in this pass.

## Raw evidence

The six run artifacts contain the per-arm JSONL scores, exposure buckets, correction norms, cosine and decomposition error. See Actions run #44046155032.
