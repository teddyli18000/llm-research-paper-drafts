# Working paper thesis — Paper 01 (not submission draft)

Date: 2026-11-16. Internal research planning only. Main dataset and evidence: [WORKLOG](WORKLOG.md), [expanded literature](../../research-review/01-rare-words-shared-output-vectors/PRIOR_ART_EXPANDED.md).

**Originality correction after rereading the complete upstream paper:** The core `R=U+AC` algorithm, KMeans regrouping while retaining logits, gradient pooling of a group, rare-target exposure threshold, and basic length/width/tie/batch effects are **already in the original author’s [01/PAPER.md](../../01-rare-words-shared-output-vectors/PAPER.md)**. They cannot be presented as our newly invented method or observations. Our independent extension must credit that work and establish a genuinely new mechanism/optimizer variant or substantive comparative/scaling findings. Full [originality ledger](../../research-review/01-rare-words-shared-output-vectors/ORIGINAL_IDEA_VS_NEW_CONTRIBUTION.md).

## Research question

Can a language model **learn which output token rows should share their training signal** and adapt this sharing online, without sacrificing the exact standard softmax decoder or increasing inference complexity? How does this compare to using a **fixed but informative semantic partition** derived from the same training corpus?

This question is more precise than "improve rare words". The actual token-frequency decomposition at V16k reveals a complementary bias: learned online groups help the *extreme low-frequency learnable tail*, whereas fixed context-derived groups help the *middle-frequency tail*. Strong pretrained GPT2 static semantic groups are even more competitive, indicating group quality can dominate update frequency.

## Candidate algorithmic contribution

For ordinary full softmax output rows `R∈R^(V×d)`, keep a decomposition `R=U+AC`, with one-hot adaptive token assignment `A∈{0,1}^(V×K)`. Every τ train updates:
1. Determine `A_new` by K-means on **current equivalent output rows** `R`.
2. Choose cluster means `C_new`.
3. Set `U_new=R−A_new C_new` exactly; no discontinuity in logits or token probabilities.
4. Retain private-row Adam state; reset the shared-cluster moments and bias-correction step count where cluster identities changed.

The output-gradient identity `∇_C L=AᵀG`, `G=∇_R L`, implies **a learned class-level gradient** `[P(g|h)−1(g_y=g)]h`. For ordinary SGD, the effective update is `−(η_U I+η_C A Aᵀ)G`. The time-varying `A Aᵀ` is a learned blockwise preconditioner for output-token logits. This mathematically separates *what output rows predict* from *how training gradients spread among them*.

A parameter-free direct optimizer version applies `G ← (I+αAAᵀ)G` before Adam; it recovers much of the original small-model effect but is not exactly equivalent to split-parameter Adam.

A size-normalized variant uses `P=A(AᵀA)⁻¹Aᵀ`: it preserves zero-sum softmax gradients and avoids eigenvalues scaling with group size. In small-model tests it helps consistently, but does not yet surpass the original method.

## Collision boundary that must appear explicitly in any manuscript

- Botha & Blunsom 2014 / Syntactic Smoothing 2024: lexical/morphological/POS sharing is established prior art.
- LightRNN 2016 / GroupReduce 2018: learned/reassigned shared word groups are established prior art.
- LENS ACL 2025: KMeans on a pretrained LM output head is established prior art.
- SGG ACL 2025: gradient statistics grouped for LR adaptation, not direct group-aware LM output-row mixing.
- Coupled Adam 2025, Kar 2026, Han 2026, Somasundaram 2026: modern token-interface optimizer geometry/rare-row gradient effects are established competing explanations or baselines.

The **research hypothesis**, not yet protected by broad modern-scale evidence, is the *joint procedure*: online adaptive grouping of the actual output rows + function-preserving repartition + train-time group-gradient transfer, yielding rare-token gains in regimes where static external semantic group labels are unavailable.

## Empirical case in current independent results

- 3-seed 400-step tiny LM: original online KMeans group sharing yields +0.0173 mean CE gain versus head-epsilon-only baseline.
- Four non-overlapping train slices × two seeds: all eight matched pairs improve, gains 0.0116–0.0207.
- Random reassignments, periodic recentering without learned groups, and one-time self-generated groups are weaker: *meaningful group alignment* matters.
- Full reset of shared-group Adam state retains gain; resetting private token-row history nearly removes it. The issue is parameter-history transport, not simply stale cluster moments.
- Strong fixed **semantic** partitions improve more than random/frequency groups; some are stronger than online grouping. Therefore static-vs-dynamic comparisons must use genuinely informative static classes.
- Same-data distributional static groups help middle-frequency tokens; online groups help the rarest learnable tokens. This motivates **frequency-aware reliability**, but naïve rare-target gradient attenuation did not improve overall CE.
- A simple auxiliary class-CE loss is weaker than actual shared-parameter or direct pooled gradient optimization; group-level targets alone don't explain the full effect.
- Existing 16k-width128 runs show positive gains; consistency across CPU hosts near the 0.005 CE level is under evaluation.

## Critical experiment gates

**Gate A: true long-horizon and full vocabulary**. Run 1600-step V6k and near-full GPT2 50k-embedding vocab. Current GitHub Actions [long horizon](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852180766) and [full vocabulary](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854953562).

**Gate B: negative-transfer structure**. Reproduce frequency-bucket gains across fresh validation data and corpus, separating extremely rare, moderately rare, common and unused token groups.

**Gate C: strong method comparisons**. At matched training FLOPs and input token budget, compare tuned head-only Adam epsilon, AMSGrad, Coupled Adam, RowNorm LM Head, train-context static lexical groups, pretrained output-group teacher, and learnable group softmax. Report global CE, token-bucket CE, efficiency and nuisance sensitivity.

**Gate D: realistic language-model scaling**. At least a 100M+ parameter LM, standard 32–50k vocabulary, realistic global batch and warmup/decay. This cannot be replaced by extrapolation from width128 models.

**Gate E: causal mechanism**. Show whether gains arise from selected group-gradient communication, optimizer moment-history policy, or effective hierarchical class learning, using direct-gradient and parameterization ablations.

## Publication pathway — independent extension, not rebranding

The base clustering method belongs to the original author. The independent paper's strongest plausible direction is **what directional inter-token group-gradient transfer adds to modern token-table optimizers** and **how to prevent low-exposure negative transfer**. Candidate own contributions are:

1. A **different mathematically specified operator**, such as the bounded projector `(I+αA(AᵀA)⁻¹Aᵀ)G` before Ember, without extra learned centroid parameters, or a reliability-aware gradient-transfer rule. This must be evaluated against both the original shared-centroid model and full-strength head optimizers.
2. Fresh multi-seed, exact-implementation comparisons with [Ember v4](https://arxiv.org/html/2607.01455v4), strong fixed semantic groups, and the original algorithm at V16k/V50k; preserve frequency-resolved negative effects and realistic compute overhead.
3. A large-parameter LM and realistically batched training result establishing whether the method survives beyond toy training. The original paper itself already predicted/explained the 10+ exposure threshold, so reproducing it is validation rather than a new mechanism claim.

The V6k [modern-optimizer results](../../research-review/01-rare-words-shared-output-vectors/MODERN_OPTIMIZER_AND_TIED_RESULTS.md) indicate complementarity, while the [new class-projected Ember pilot](../../research-review/01-rare-words-shared-output-vectors/PROJECTED_EMBER_PROTOCOL.md) and [precommitted larger-vocabulary test](../../research-review/01-rare-words-shared-output-vectors/EMBER_SCALE_TRANSFER_PROTOCOL.md) test whether anything genuinely distinctive remains. Do not promise a top-conference method on the basis of standalone replications.
