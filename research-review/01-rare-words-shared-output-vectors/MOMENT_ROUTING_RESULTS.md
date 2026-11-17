# Paper 01 — decoupled second-moment routing results

**Run:** [GitHub Actions #44043530225](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44043530225), six jobs successful.  
**Seeds:** 521, 523, 541. **Effective batches:** 32 and 128. **Budget:** 819,200 tokens per arm. Eight matched arms per job.

## Overall held-out cross-entropy

Lower is better. Each entry is the mean of three seeds.

| Arm | Batch 32 | Batch 128 |
|---|---:|---:|
| Plain Ember | 5.00530 | 5.09003 |
| Upstream centroid + Ember | 4.98257 | 5.08373 |
| Dynamic projector, shared moments | 4.97620 | 5.09447 |
| Dynamic projector, raw second moments | 5.01990 | 5.13720 |
| Fixed context projector, shared moments | 4.96917 | **5.06757** |
| Fixed context projector, raw second moments | 5.02997 | 5.13660 |
| Context-to-dynamic, shared moments | **4.96597** | 5.07120 |
| Context-to-dynamic, raw second moments | 5.04203 | 5.11333 |

The raw-stat route fails the preregistered decision rule. It loses to the matched shared-stat implementation for every context and context-to-dynamic seed at both batch sizes. Its mean paired CE penalty is:

| Group schedule | Batch 32 raw − shared | Batch 128 raw − shared |
|---|---:|---:|
| Dynamic K-means | +0.04370 | +0.04273 |
| Fixed context | +0.06080 | +0.06903 |
| Context-to-dynamic | +0.07607 | +0.04213 |

For the fixed-context pair, raw-stat routing is worse on all three seeds at both batch sizes. The paired 95% t intervals are [0.0251, 0.0965] at batch 32 and [0.0138, 0.1243] at batch 128.

## Why raw-stat routing failed

The projected gradient itself is only about 1.06–1.11 times the raw gradient norm. But the actual Ember head update becomes much larger when the second-moment statistics remain based on the raw gradient:

| Group schedule | Batch 32 shared update norm | Batch 32 raw-stat update norm | Batch 128 shared update norm | Batch 128 raw-stat update norm |
|---|---:|---:|---:|---:|
| Dynamic K-means | 14.590 | 20.505 | 15.401 | 29.549 |
| Fixed context | 14.744 | 24.424 | 15.422 | 24.688 |
| Context-to-dynamic | 14.594 | 22.882 | 15.268 | 21.712 |

The raw-stat route therefore removes a stabilizing effect of Ember's factored denominator. At the preregistered \(\alpha=2\), it amplifies the actual parameter update by roughly 1.4–1.9× and oversteps.

This is a useful negative result: for this projector and this token-table optimizer, feeding the modified gradient into the second-moment estimator is not merely an unwanted cancellation. It is part of the effective step-size control.

## Frequency-resolved result

The fixed-context comparison shows a systematic tradeoff. Raw-stat routing improves the zero-training-exposure validation bucket, but worsens every nonzero-exposure bucket.

| Target exposure in training | Batch 32 shared | Batch 32 raw stats | Batch 128 shared | Batch 128 raw stats |
|---|---:|---:|---:|---:|
| 0 | 13.1402 | **12.4591** | 13.0287 | **12.2612** |
| 1–9 | **10.8610** | 11.6222 | **10.9582** | 11.3157 |
| 10–31 | **9.7248** | 9.9210 | **9.7734** | 9.9217 |
| 32–99 | **8.4900** | 8.5655 | **8.5723** | 8.6758 |
| 100–315 | **7.1264** | 7.1909 | **7.2365** | 7.3307 |
| 316–999 | **5.6662** | 5.7341 | **5.8217** | 5.8888 |
| 1k–3k | **4.6685** | 4.7214 | **4.8149** | 4.8925 |
| 3k+ | **2.8498** | 2.8776 | **2.9661** | 3.0163 |

The zero-exposure bucket has only 0.01% validation target mass in this setup. Its large CE improvement does not compensate for the degradation across the much larger nonzero buckets. This is a specific frequency tradeoff, not an overall optimizer improvement.

The likely mechanism is that raw statistics make group-transferred updates unusually large on rows whose own raw gradients are small. That helps completely unseen rows, but it also over-amplifies low-exposure and common rows once their group receives a nonzero gradient. A row-selective moment route could isolate this effect, but it would be a separate hypothesis and is not justified as a general method by this result.

## Context-to-dynamic schedule

The shared-stat context-to-dynamic arm is best at batch 32 (4.96597), improving over fixed context (4.96917) by 0.00320; it wins on all three seeds. At batch 128 it is worse than fixed context (5.07120 vs. 5.06757), also on all three seeds.

The switch occurred after 20% of the run: step 80 at batch 32 and step 20 at batch 128. This suggests that a fractional switch is too early for the 100-step high-batch run. The next schedule test should use a fixed optimizer-step threshold (e.g. step 200), leaving the 100-step run on context groups and allowing the 400-step run to transition later.

## Decision

1. **Reject raw-stat routing at \(\alpha=2\).** It does not improve the same projector under either batch regime and has a clear update-norm overshoot.
2. **Retain shared-stat Ember.** Its factored second moment is an important part of the method's effective scaling.
3. **Retain context-to-dynamic as a batch-32 schedule candidate only.** The 20%-of-run schedule is not batch invariant.
4. **Do not continue tuning the global raw-stat route.** The next higher-value experiment is the exact residual decomposition into prediction-side \(Pp\) and target-side \(-Py\), followed by a fixed-step context-to-dynamic schedule if the decomposition supports it.

## Artifacts

Actions artifacts: 14311130667, 14311071168, 14310897305, 14310763196, 14310692465, 14309994453.
