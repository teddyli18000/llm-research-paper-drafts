# Paper 01 — V50k and 1600-step results (2026-11-16)

All figures below are from completed GitHub Actions logs.

## Near-full GPT-2 vocabulary

[Actions #43854953562](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854953562) completed all four conditions. V≈50k, width64, 2 layers, 4 heads, batch32 × sequence64, gradient accumulation4, 800 optimizer steps (~1.64M training tokens), head epsilon1e-5, head LR0.05, centroid LR0.01, 64 clusters with K-means every50 steps.

| Seed | Plain epsilon-head CE | Dynamic K-means CE | Plain minus dynamic |
|---:|---:|---:|---:|
| 0 | 6.2965 | 6.2659 | +0.0306 |
| 1 | 6.2881 | 6.2588 | +0.0293 |
| Mean | 6.29230 | 6.26235 | **+0.02995** |

The original `grid.py` maps up to `a.V-2` most frequent GPT-2 IDs plus EOT and creates a final `UNK` class, so `--V 50258` is **near-full**, not a no-remapping canonical GPT-2 head. `rare<100=0.292` is the validation target mass of tokens with under100 occurrences in this particular small training prefix, not a corpus-wide intrinsic-rarity measure.

## 1600-step V6k

[Actions #43852180766](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852180766). V6000,width32,batch32,L64,1600 steps (~3.28M tokens); two model seeds.

| Seed | Plain | Online groups | Fixed same-data context groups | Context groups→online at step200 |
|---:|---:|---:|---:|---:|
| 0 | 4.7401 | 4.7217 | 4.7267 | **4.7203** |
| 1 | 4.7394 | 4.7175 | 4.7243 | **4.7165** |
| Mean | 4.73975 | 4.71960 | 4.72550 | **4.71840** |

Dynamic minus baseline mean gain **0.02015 CE** survives 4× longer training. The online-grouping method now beats fixed same-data semantic groups, contrary to the early 400-step ranking. The late hybrid is modestly best in both seeds but improves over online by only **0.0012 CE**, not established as a significant new method.

## Same-seed CPU repeatability

[Actions #43854706295](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854706295): V16k,width128, three repeats at each thread setting. `threads=1`: **5.7841 / 5.7841 / 5.7841** (display precision); `threads=2`: **5.7837 / 5.7828 / 5.7837**, range **0.0009**. Earlier cross-job 0.0057 CE discrepancy remains relevant for interpreting small differences.

## Consequence

Positive adaptive grouping gains persist across data slices, longer runs, V16k and V50k, and are not limited to the original tiny pilot. This confirms *a real effect under tested settings*, not a top-conference novelty claim. Strong independently tuned modern optimizers and static semantic groups at V50k remain essential challengers. The new [Coupled Adam preregistration](COUPLED_ADAM_CHALLENGE_PROTOCOL.md) tests one such competitor.
