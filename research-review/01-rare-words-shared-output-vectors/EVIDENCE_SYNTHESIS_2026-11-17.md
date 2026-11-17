# Paper 01 — evidence synthesis and paper decision (November 17, 2026)

Independent follow-up on Vuk Rosić's public [Paper 01](../../01-rare-words-shared-output-vectors/PAPER.md). This document distinguishes the upstream inventor's method, prior art, and our separately tested optimizer extension. **Scope is research status, not a submitted manuscript.**

## Attribution and exact mathematical claim

**Upstream author's contribution, not ours:** trainable LM output rows split as `R=U+AC`, dynamic KMeans on their current values, regroup without changing forward logits, group-gradient summation in trainable centroids, and a finding that learnable rare targets with ≥10 exposures benefit more than unseen words. The original draft also tests 1600-step runs, batch-size changes, tied embeddings, and vocabulary scaling.

**Established relevant prior art:**

- Botha & Blunsom (2014) share morphological word components; LightRNN (NeurIPS 2016) and GroupReduce (NeurIPS 2018) assign and reorganize vocabulary groups.
- [Tree Loss (AISTATS 2022)](https://proceedings.mlr.press/v151/wang22d.html) uses class/ancestor parameter sharing under CE, inducing Gram-matrix class-gradient coupling.
- [AGG (ACL 2022)](https://aclanthology.org/2022.acl-long.3/) modifies rare output-row gradients based on recent exposure frequency, with much stronger-scale rare PPL results; it does not use semantic class-mean gradient transfer.
- [Syntactic Smoothing (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.344/) redistributes target mass; it does not average prediction-side class residuals.
- 2025–2026 LM-head optimizer work, especially token-interface Ember, projected RowNorm, MuonIO, and SGG, establishes optimizer normalization and grouped statistics as prior art, but no reviewed source uses the exact direct normalized class projector on a flat untied LM head.

**Our candidate extension:**

[
G=(p-y)h^\top,\quad P=A(A^\top A)^{-1}A^\top,\quad G'=(I+\alpha P)G.
]

It uses one ordinary output parameter matrix, lets each token-row receive its group-*mean* gradient, preserves the exact forward softmax function and zero-sum gradient, and has group-size-independent operator spectrum `{1,1+α}`. Group assignments can be static train-text context groups or refreshed KMeans groups.

The group gradient is **`(Pp-Py)hᵀ`**, including a prediction-side probability error term absent from ordinary target label smoothing.

### Equivalence that limits theory novelty

For fixed A and SGD, the normalized shared-centroid parameterization `W=U+sqrt(α) A(AᵀA)^{-1/2} C` reproduces **exactly** the same effective W updates as direct `I+αP`. A 25-step nonlinear test reached 2.22e−15 maximum difference. Thus **not a new principle of class-gradient sharing and not a new SGD geometry**.

For adaptive Ember or Adam, gradient preconditioning in the flat W coordinates and optimization in split U/C coordinates generally produce different trajectories because the adaptive moment statistics are parameterization dependent. This is the independent implementation/mechanism hypothesis.

## Empirical milestones from independently verified logs

1. **Actual independent test, V≈50k**, Actions [#44017710007](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017710007), untouched FineWeb shard002, three new seeds. Plain Ember: **6.31510 CE**; new direct projector + Ember: **6.29370** (positive +0.02140 in all three seeds); original author's online centroids + Adam: **6.29480** (effectively tied, difference only 0.00110). Validation and independent-test rank between these last two methods reverses; no demonstrated new-method advantage over the original.
2. **Tuned V16k Ember**, Actions [#44020024307](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44020024307). Best plain head-only Ember **5.78270**, projected Ember **5.74967**, mean **+0.03303** nats improvement on previously used validation, three paired seeds. Real effect beyond the tested bare modern optimizer tune; original author's centroid+Adam at **5.78377** in this setting.
3. **Causes are group-dependent**, Actions [#44017278316](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017278316). Shuffled KMeans group memberships and random/frequency grouping underperform meaningful group structure; simply multiplying Ember's head LR by 2–3 does not replicate class projection. At V6k fixed train-text contextual groups can beat online KMeans; at V50k online wins. No universal static-vs-dynamic hierarchy.
4. **Batch regime differs**, Actions [#44018218531](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018218531). With equal training tokens, dynamic projector at effective batch128/100 optimizer steps yields **5.09497**, worse than plain Ember **5.08927** and original centroid+Ember **5.08183**; batch32/400 steps yields **4.97493** vs plain **5.00440**. Fixed contextual groups rescue the batch128 regime, while context→dynamic scheduling improves V6k batch32.
5. **Exact SGD-equivalent normalized-centroid vs direct under adaptive optimizers**, Actions [#44051580714](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44051580714) and [new head-LR sweep #44055310959](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44055310959). The new 72-arm sweep: **batch32 direct wins all 18/18 paired seeds × LR × optimizer** against normalized centroids; direct Ember best 4.96817 vs normalized-centroid Ember best 4.98063. **Batch128 direct vs scaled centroid is often near-tied**, winning reliably only at LR .10 across both optimizers. See [full table](NORMALIZED_CENTROID_LR_SWEEP_RESULTS.md).
6. **Compared alternative class objectives**, Actions [#44029810293](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44029810293) and [#44030216366](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44030216366). At matched V6k batch32, direct context projector **4.96863** vs best tested class-probability objective **4.99037**; direct projector **4.97140** vs best tested target-smoothing control **5.08153** (full-objective ε=.25). These finite coefficient grids do **not** reproduce entire published smoothing or concept-prediction systems. Details: [loss collision results](CLASS_LOSS_COLLISION_RESULTS.md).
7. **Mechanism decomposed**, Actions [#44052591527](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44052591527). The whole class error residual works; adding only prediction-side or only target-side centered terms at the same α breaks performance. The two large aligned terms largely cancel in the full update. These ablations are not a scale-matched proof of both terms' independent necessity.
8. **Independent model scale remains tiny.** V6k width32 ~0.41M weights, V16k width128 ~4.49M, V50k width64 ~6.54M (98% of the last in input/output token tables; ~0.1M Transformer backbone). Two transformer layers and 64-token sequence length in all these tests. See [scale audit](ACTUAL_MODEL_SCALE_AUDIT.md). This is *vocabulary* scaling, **not modern LLM scaling**.

## Current research verdict and paper positioning

- **Method distinctiveness: real but narrow.** The flat-head mean projector before Ember/Adam is a concrete tested implementation that is not identical to target smoothing/class-probability CE, or to split-centroid Adam under some batch regimes. Its generic class-gradient sharing and SGD geometry are *not* new.
- **Empirical benefit: real but not a strong win over original author's best comparator.** On a locked V50k independent test, the projector beats plain Ember by +0.02140 CE but is practically tied with original centroid+Adam by 0.00110.
- **Publishability:** sufficiently reproducible for a focused technical note / workshop contribution on optimizer-coordinate effects and rare-token output learning. **Not yet strong enough for an ICML/NeurIPS/ICLR main-track algorithm paper**: near-tie with the upstream invention on independently held-out tokens, no realistic large-backbone test, limited batch transfer, and modern strong optimizer controls incompletely explored.
- **Efficient next decision:** do not keep accumulating more V6k ablations of the same projection. Only a real >100M contextual backbone pretraining test, comprehensive tuned head-optimizer comparison (including input/output tying and AGG) and statistically meaningful extra advantage over original centroids could change the publication tier. Otherwise position this as a rigorous independent empirical extension with attribution rather than a foundational optimizer innovation.

## Last decisive collision study — completed

Preregistered [graph-diffusion score collision, Actions #44090597865](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44090597865) completed **6/6 success, 30 unique trainings, 3 seeds per batch**. Direct projector + Ember obtained mean **4.96403/5.06163 CE** at batch32/128 versus plain Ember **5.00017/5.07983**; fixed-λ=1 P-limit, diffusion and resistance auxiliary losses all remained within approximately 0.0013 CE of plain Ember. Full [original-log analysis](GRAPH_DIFFUSION_SCORE_RESULTS.md).

**Interpretation boundary:** the scalar graph score ordinarily backpropagates `2J_softmax P(p−y)` while direct projection adds `2P(p−y)`. At λ=1, the score gradient can be attenuated by J, so the failure of the fixed-coefficient losses **is not a proof of a universal advantage over optimally scaled losses**. Moreover, we established the exact single-prediction Fisher identity `J_softmax⁺ ∇_z S_P=2P(p−y)`, putting the algorithm's class residual in the natural-gradient family. The precise independent contribution is still the practical flat-head application before a modern adaptive optimizer, not a novel mathematical scoring primitive. See [derivation](GRAPH_SCORE_FISHER_GEOMETRY.md).

**Research evidence collection and paper-feasibility judgment for this stage are closed.** There are no incomplete critical 01 experiments in this handoff; the evidence does not clear the standard for a major-conference methods claim.

## Science and source-use notes

- [Research worklog](../../research-lab/01-rare-words-shared-output-vectors/WORKLOG.md) carries detailed experimental history and reproducibility caveats.
- The original author's README explicitly invites extensions and says code is MIT licensed; credit upstream inventions and cite antecedents, do not claim originality over a publicly established method.
- Literature checked through November 17 2026, including full-text analyses of Tree Loss, AGG, Syntactic Smoothing, projected RowNorm and 2026 graph proper scores. Exact identical prior implementation was not identified; algebraic/conceptual equivalents and stronger baselines were.
