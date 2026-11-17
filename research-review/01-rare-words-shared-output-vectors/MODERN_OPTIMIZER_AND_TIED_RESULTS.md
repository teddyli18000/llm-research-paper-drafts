# Paper 01 — Modern optimizer competition and tied-token-table transfer

Date: 2026-11-16. **All three pre-registered matrices completed successfully.** Sources: [Coupled Adam #43927461457](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43927461457), [official Ember #43928661510](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43928661510), [tied embeddings #43929175351](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43929175351).

## 1. Official Ember: direct challenge that **strengthens**, rather than erases, group-gradient contribution

All experiments use V6000,width32,2 layers,batch32,L64,400 steps,three identical initialization seeds (0/1/2),one CPU thread. Official [Ember](https://github.com/katop1234/ember) was installed from pinned source commit `76f30117b2d948874cd1c965c4f6813fa9de8caa`, applied **to output head only**. The input embedding and backbone retain original optimizers, so these numbers are not a complete reproduction of original paper's end-to-end Ember training.

| Head optimization | Mean val CE (3 seeds) |
|---|---:|
| Standard head Adam + epsilon 1e−5 | **5.01160** |
| Same Adam + dynamic 64 clusters | **4.99443** |
| Best standalone Ember in registered grid: LR0.05, beta2=.999 | **4.99390** |
| Same standalone-Ember config + dynamic clusters | **4.97960** |
| Best grouped-Ember config in registered grid: LR0.05, beta2=.95 | **4.97780** |
| Standalone Ember at LR0.05, beta2=.95 | **4.99943** |

At Ember LR0.05,beta2=.95 the **paired same-optimizer** improvements are `+0.0186,+0.0193,+0.0270` CE, average **+0.02163**. At Ember LR0.05,beta2=.999, paired gains are `+0.0164,+0.0096,+0.0169`, average **+0.01430**.

Important distinction: standalone Ember LR0.05/beta2=.999 matches/slightly exceeds the earlier dynamic-cluster Adam score (4.99390 vs 4.99443). **Thus the original method is not inherently superior to modern token-table optimization.** However, adding dynamic group-gradient communication *on top of* Ember improves every one of those three seeds, and the best combined result 4.97780 is materially below both standalone alternatives. The gain is not explained away merely by replacing Adam's rare-token second moments.

The original Ember code estimates row×column factored squared gradients and has no token-specific first moment; it does not pool directional gradients across token groups. The group-sharing mechanism is plausibly complementary. This is evidence for a distinct direction of improvement, not yet evidence of strong scaling or causal identification of precisely which term explains the gain.

## 2. Coupled Adam: not competitive on this unscaled toy LM

Original `grid.py`'s Coupled Adam shares a vocabulary-averaged per-feature second moment; the instrumented independent copy adds a head-only `--cheps` so its denominator can be tuned without changing the embedding/body Adam epsilon. Registered head LR `0.02/0.05/0.1/0.2`, `cheps` `1e−6/1e−5`; three model seeds; dynamic/no-group paired for all eight combinations.

| Setting | Mean val CE |
|---|---:|
| Standard head Adam (shared control across studies) | 5.01160 |
| Grouped standard Adam | **4.99443** |
| Best standalone Coupled Adam (LR.05,eps1e−5) | 5.09880 |
| Best grouped Coupled Adam (LR.05,eps1e−6) | 5.09960 |

Grouping helps Coupled Adam at LR.1/eps1e−6 (paired gain **+0.02063** across all three seeds), but hurts at LR.02/eps1e−6 (paired **−0.01510**). The best Coupled variants remain worse than Adam in these settings. **Do not claim Coupled Adam is generally worse:** this challenge retains the original toy model's backbone hyperparameters and searched a finite LR/epsilon grid, not the optimizer's best modern-scale recipe.

## 3. Tied embeddings: adaptive grouping persists, but static semantic groups are better

The upstream `--tie 1` uses the equivalent token table `R=U+AC` for both input lookup and output classification. Every regroup is function-preserving in both uses. Data: V6000,width32,2 layers,batch32,L64; three independent model seeds, same-data co-occurrence static clusters; two full training horizons, each with its own fixed schedule.

| Steps | Tied plain CE | Tied online KMeans CE | Tied static context-group CE |
|---:|---:|---:|---:|
| 400 | 5.10013 | 5.08567 | **5.06007** |
| 1600 | 4.79720 | 4.77350 | **4.77063** |

At 400 steps, online minus plain improvement **+0.01447**, but static context groups give **+0.04007**. At 1600, online minus plain **+0.02370**, static minus plain **+0.02657**. The static advantage shrinks strongly with training horizon (online−static CE: 0.02560 at 400, 0.00287 at 1600), and by 1600 online slightly beats static in one of three seeds. But the **best mean tied result is static** in both horizons.

This confirms **architecture transfer of the sharing effect** to tied token tables; it does not establish that online regrouping is better than an informative *fixed* grouping, nor that the method's only mechanism is the untied softmax zero-sum gradient projector. Input-embedding gradients break that zero-sum property when weights are tied.

## 4. Updated decision

- **Positive:** meaningful gradient-sharing from content-informed groups adds to official Ember under identical LR/beta2, at fixed small-model compute; the effect is not equivalent to simply using a rare-row second-moment adaptive optimizer.
- **Qualified:** strong static semantic groups are hard competitors, particularly with tied input/output weights or few updates. The scientific gain may be in *adapting group identity late during pretraining*, rather than in online KMeans per se.
- **A genuinely distinctive paper must establish:** adaptive group-gradient transfer *on top of modern head optimizers* with larger vocabularies and models, frequency-resolved generalization, strong fixed-context groups, and realistic compute overhead. The [V50k baseline results](FULL_VOCAB_LONG_HORIZON_EVIDENCE.md) and [V50k target-frequency analysis](FULL_VOCAB_FREQUENCY_DECOMPOSITION.md) support this next test.

The **next confirmatory protocol** must use newly declared initialization seeds and fixed Ember hyperparameters transferred from the V6k grid, rather than reselecting the winner on the same tiny validation set.
