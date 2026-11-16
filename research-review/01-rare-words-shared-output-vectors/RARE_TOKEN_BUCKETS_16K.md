# 16k-token frequency decomposition: where dynamic grouping actually helps

Date 2026-11-16; verified [Actions #43851810419](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43851810419), all six experiments + explicit per-bin parsing **completed successfully**.

V≈16000 GPT2 tokens, width128, batch32×64, 400 train steps, two model-initialization seeds, reconstructed same-public-shard token counts, held-out 600 validation sequences. Identical model architecture and learning rates. Original method = online KMeans shared-parameter output head; strong control = static co-occurrence-derived groups from **the same train tokens**, with the same output head parameterization.

## Overall results

| Seed | Epsilon-only baseline | Online dynamic KMeans | Static train-context groups |
|---|---:|---:|---:|
| 0 | 5.7998 | 5.7838 (gain **0.0160**) | 5.7775 (gain **0.0223**) |
| 1 | 5.7984 | 5.7789 (gain **0.0195**) | 5.7737 (gain **0.0247**) |

The identical headline experiment [#43841944336](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43841944336) had seed0 dynamic 5.7781 vs this run 5.7838; identical seed/config nevertheless differs in the last 0.0057 CE. We should explicitly audit nondeterminism across GitHub CPU hosts/torch kernels and repeat before claiming effects at the 0.003–0.006 resolution. Direction remains positive.

## Target-frequency buckets: contribution to global CE reduction

`Contribution = validation target mass × (baseline bucket CE − method bucket CE)`. Negative contribution means harmful regression. Numbers are the two-seed mean of **observed** contributions, calculated from the actual GitHub output.

| Token corpus frequency | Validation mass | Online dynamic contribution | Static context contribution |
|---|---:|---:|---:|
| 3×10⁻⁶ to 1×10⁻⁵ | 4.64% | **+0.00604** | +0.00141 |
| 1×10⁻⁵ to 3×10⁻⁵ | 10.11% | **+0.01056** | +0.00770 |
| 3×10⁻⁵ to 1×10⁻⁴ | 13.20% | +0.00570 | **+0.01106** |
| 1×10⁻⁴ to 3×10⁻⁴ | 12.58% | +0.00054 | **+0.00584** |
| 3×10⁻⁴ to 1×10⁻³ | 10.44% | +0.00064 | +0.00071 |
| >1×10⁻³ | 42.18% | **−0.00447** | −0.00247 |
| Unk bucket | 6.84% | −0.00125 | −0.00075 |

## Key mechanism implication

**The two methods help different frequency regimes.** Dynamic self-learned sharing helps extremely rare but learnable tokens (roughly 3×10⁻⁶ to 3×10⁻⁵) more than a static distributional grouping extracted from 819k tokens. The static context grouping transfers much more effectively to moderate-frequency tokens (roughly 3×10⁻⁵ to 3×10⁻⁴). Both methods trade some performance away from very frequent tokens (>10⁻³). This explains why static context has a lower *overall* CE even while dynamic is better on the extreme rare tail.

**This is a concrete scientific opportunity, not a vague semantic claim:** share gradients selectively toward rare rows or let dynamic and static group predictors specialize by exposure regime. One might define an exposure-conditioned sparse coupling `G' = (I + W_r A D⁻¹ Aᵀ)G` with `W_r` per-row weights based only on *training* frequencies, minimizing negative transfer to high-frequency outputs. Such a variant requires a fresh training-domain holdout to avoid tuning thresholds on the same evaluation sample.

This source-based finding is more informative than just comparing the final mean loss. Follow-up: a controlled rare-row-targeted gradient intervention, paired across model seeds and evaluated on a fresh validation slice.
