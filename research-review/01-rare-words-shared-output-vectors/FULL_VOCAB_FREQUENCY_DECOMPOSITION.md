# Paper 01 — Near-full GPT-2 vocabulary, frequency-resolved effect

**Data:** all four accepted raw JSONL artifacts in [Actions #43854953562](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854953562): seed0/1, epsilon-only vs dynamic KMeans. This is **a separately parsed analysis**, not an inference from rounded overall validation loss. V≈50k, width64, 800 steps, 1,638,400 training tokens, 600 fixed validation windows. Positive contribution is a reduction in aggregate validation cross-entropy.

## Matched headline

| Seed | Epsilon-only CE | Dynamic CE | CE reduction |
|---:|---:|---:|---:|
| 0 | 6.2965 | 6.2659 | +0.0306 |
| 1 | 6.2881 | 6.2588 | +0.0293 |
| Mean | 6.29230 | 6.26235 | **+0.02995** |

Every bucket below was retrieved from the run's stored `bins` and `fbins` objects. `bucket contribution = target mass × (baseline CE − dynamic CE)`. Mass is exactly shared between paired arms here, so contributions sum to total mean CE gain up to rounding.

## Buckets by actual training target occurrences

| Target-token count in this 1.64M-token prefix | Val target mass | Plain bucket CE | Dynamic bucket CE | Dynamic − plain CE reduction | Contribution to overall CE |
|---|---:|---:|---:|---:|---:|
| **0** | 0.81% | 12.8883 | 13.3300 | **−0.4416** | **−0.00358** |
| **1–9** | 7.59% | 11.9750 | 11.9916 | −0.0165 | −0.00125 |
| **10–31** | 9.14% | 10.7157 | 10.5063 | **+0.2095** | **+0.01914** |
| **32–99** | 11.68% | 9.1557 | 9.0704 | **+0.0853** | **+0.00996** |
| 100–315 | 13.73% | 7.5187 | 7.5186 | +0.0001 | +0.00001 |
| 316–999 | 10.74% | 6.2253 | 6.2306 | −0.0053 | −0.00056 |
| 1k–3k | 10.15% | 4.9604 | 4.9707 | −0.0103 | −0.00105 |
| 3k+ | 36.16% | 2.8373 | 2.8171 | +0.0201 | +0.00727 |
| **Sum** | **100%** | | | | **+0.02995** |

**The headline gain does NOT mean the most extreme rare tokens benefit.** Dynamic groups degrade zero-exposure target rows quite substantially and also slightly degrade the 1–9-exposure bucket. The net gain is dominated by tokens observed **10–99** times in this training prefix; their combined contribution is **+0.02910 CE**, about **97%** of the overall +0.02995. Frequent tokens (>3k) contribute a smaller +0.00727; the negative ultra-rare and intermediate bins offset part of that gain.

A companion view by *full-shard corpus frequency* confirms this is not solely an artifact of the actual-count bucket cutpoints:

| Corpus token probability | Val mass | Plain CE | Dynamic CE | Contribution to overall CE |
|---|---:|---:|---:|---:|
| <1e−6 | 0.50% | 12.6128 | 12.9268 | −0.00157 |
| 1e−6–3e−6 | 3.44% | 12.3108 | 12.4588 | −0.00509 |
| 3e−6–1e−5 | 7.54% | 11.4794 | 11.3728 | +0.00803 |
| 1e−5–3e−5 | 10.11% | 10.1772 | 9.9939 | **+0.01853** |
| 3e−5–1e−4 | 13.20% | 8.5061 | 8.4706 | +0.00469 |
| 1e−4–3e−4 | 12.58% | 7.0243 | 7.0332 | −0.00112 |
| 3e−4–1e−3 | 10.44% | 5.6451 | 5.6505 | −0.00056 |
| >1e−3 | 42.18% | 3.1164 | 3.0997 | +0.00704 |
| **Sum** | **100%** | | | **+0.02995** |

## Implication for the method's academic claim

This strengthens the *mechanism* but revises the original broad formulation. A more accurate finding is **intermediately underexposed token gradient transfer**: there appears to be an exposure sweet spot where there is enough evidence for a meaningful cluster and too little direct row-specific learning for a plain output head to be efficient. For unseen tokens, self-induced cluster assignments can be wrong and negative transfer can dominate.

It also reveals a **change from the earlier V16k-width128** bucket study, where online grouping's largest positive contributions were reported at corpus probabilities 3e−6–3e−5 while the strongest fixed-context groups helped middle frequencies. At V50k, the most extreme <3e−6 tail is negative, while 1e−5–3e−5 is strongly positive. Differences in model width, vocabulary, training horizon, and token exposure are coupled across these experiments; a follow-up 2×2 controlled test would be needed to attribute the shift to vocabulary size alone.

The current scientific question is: **Can a group-gradient coupling rule recognize when a row is reliably groupable, instead of transfering indiscriminately to zero-exposure and high-frequency rows?** The naive frequency-only attenuation already failed [an earlier fixed study](RARE_SELECTIVE_GRADIENT_RESULT.md), so a new method needs an explicit reliability/interference estimator rather than further threshold tuning.
