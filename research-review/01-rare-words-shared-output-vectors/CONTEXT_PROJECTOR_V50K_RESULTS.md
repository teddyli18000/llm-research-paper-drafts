# Paper 01 — V≈50k contextual vs learned class projection: online learning wins

**[GitHub Actions #44018640211](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018640211) completed successfully, 6/6 jobs.** This reports the three V≈50k outcomes under the precommitted [contextual transfer protocol](CONTEXTUAL_PROJECTOR_TRANSFER_PROTOCOL.md), new independent seeds 181,191,197. Original 2-layer width64 transformer, near-full GPT2 vocabulary, batch32/L64/acc4, 800 steps, official Ember head β₂=.999 and LR=.05, group-projection α2 and K64.

| Arm | Seed181 | Seed191 | Seed197 | Mean CE |
|---|---:|---:|---:|---:|
| Ember plain | 6.2935 | 6.2880 | 6.2794 | 6.28697 |
| Original online centroid + Ember | 6.2659 | **6.2593** | **6.2522** | 6.25913 |
| **New online KMeans projector + Ember** | **6.2597** | 6.2598 | 6.2577 | **6.25907** |
| Static train-context projector + Ember | 6.2836 | 6.2818 | 6.2861 | **6.28383** |
| Context-first, switch to online projector at step200 | 6.2617 | 6.2569 | 6.2623 | 6.26030 |

All three seeds favor dynamic projector versus static contextual projector by +.0239,+.0220,+.0284 CE (mean +.02477). **The two dynamic-group methods are tied:** online normalized projector 6.25907 vs original centroid+Ember 6.25913, a difference of ~0.00007 CE. Context-first/late switch is also near-tied at 6.26030 and never convincingly better.

Together with the previous [V16k contextual test](CONTEXT_PROJECTOR_V16K_RESULTS.md), this disproves a single simple claim that semantic fixed classes always improve projection. V6k at short horizon and batch128 favors semantic fixed classes; V16k and V50k with batch32 favor online classes. Token-table width, effective exposure, optimizer steps and class gradient alignment vary simultaneously, so this **observational crossover does not prove one scalar threshold**.

The exact head-gradient identity from [GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md](GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md) shows the operator amplifies **class-level probability minus target residual**. A strong class-smoothed-label baseline remains required. No special claim of a novel effect for upstream KMeans regrouping.
