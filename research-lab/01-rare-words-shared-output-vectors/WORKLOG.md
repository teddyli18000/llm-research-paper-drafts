# Paper 01 research worklog (2026-11-16)

Scope: original `01-rare-words-shared-output-vectors` draft. Original paper and code unchanged. All new experiments and docs remain under `research-lab/01-rare-words-shared-output-vectors`, `research-review/01-rare-words-shared-output-vectors`, and scoped workflows. Last upstream main was rebased/rewritten independently; this file tracks by run ID and content rather than relying on old ancestry.

## Mathematical contribution under investigation

`R=U+AC`; group gradients `∂L/∂C=Aᵀ∂L/∂R`; for ordinary SGD `ΔR=-(η_U I+η_C A Aᵀ)∂L/∂R`. Dynamic KMeans modifies the coupling `AAᵀ` while preserving `R` at assignment changes. The empirical point is whether this improves actual rare-token learning beyond static semantic groups and strong optimizers, not whether output-row clustering was invented from scratch.

## Verified experiment index

| Run | Setup | What has been measured |
|---|---|---|
| [#43833736739](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43833736739) | V6k, 200 steps, seed0 | mathematical identity, first moment ablation |
| [#43834412315](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43834412315) | V6k, 400 steps, 3 seeds | strong (+0.0173 average CE) online cluster effect |
| [#43834707881](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43834707881) | batches16–128, same tokens | re-clustering by tokens mitigates but doesn't remove batch degradation |
| [#43835794954](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43835794954) | matched Adam & AMSGrad | dynamic clustering helps both at same LR |
| [#43836617555](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43836617555) | full Adam-state reset incl. step | reset shared-C optimizer state okay; reset private-U state largely destroys gain |
| [#43836943981](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43836943981) | B128 × 400 steps | large batch still shows smaller effect; cadence alone insufficient |
| [#43838718914](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43838718914) | random refresh, same membership, warmstart | meaningful KMeans grouping needed; arbitrary reparameterization ineffective |
| [#43838902626](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43838902626) | GPT2 pretrained static semantics, 3 seeds | strong teacher-assisted static groups beat from-scratch online clustering |
| [#43839718101](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43839718101) | direct gradient-kernel vs shared param | non-parametric group-gradient mixing captures much of the effect |
| [#43839941058](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43839941058) | pretrained static -> online regroup | early re-clustering hurts strong teacher prior, late regroup roughly neutral |
| [#43840936159](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43840936159) | static context groups from same training data | teacher-free static context groups outperform scratch online KMeans at V6k |
| [#43841653746](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43841653746) | context static -> online regroup at step 50/200 | late start at step200 improves static context mean by 0.0032; early start harms |
| [#43841944336](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43841944336) | V16k width128 B32 400step, 2 seeds | baseline 5.7998/5.7990; dynamic 5.7781/5.7793; context static 5.7776/5.7737 |
| [#43842243603](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43842243603) | auxiliary group-CE target, 3 seeds | best alpha0.2 gain about 0.005 (below direct cluster sharing ~0.017) |

All 'success' notes above refer to actual saved run logs. Numeric values are from independently reproduced experiments, using public GPT2-FineWeb training data and a reconstructed token frequency count. They are **not exact bytewise reproductions of the original author's paper**, whose corpus-wide counts file is not in the fork.

## Active research experiments

- [#43851693792](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43851693792): four 100M-token FineWeb shard offsets; two initializations, baseline vs dynamic group.
- [#43851810419](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43851810419): V16k width128 per-target frequency CE decomposition on two seeds.
- [#43852180766](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852180766): V6k 1600-step long training, online vs static/hybrid groups.
- [#43852372143](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852372143): size-normalized projector gradient sharing across three seeds and five strengths.

## Closest confirmed literature

- [GroupReduce, NeurIPS 2018](https://arxiv.org/abs/1806.06950): dynamic refinements of LM softmax word clustering **for post-hoc compression**, not training-time exact-logit-preserving reclustering.
- [Syntactic Smoothing, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.344/): soft target modification, not a learnable output-row group parameter.
- [LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/): KMeans on pretrained LM head tokens; posttraining lexical representation.
- [SGG, ACL 2025](https://aclanthology.org/2025.acl-long.164/): clusters gradient **statistics** for optimizer LR adaptation, not target-token group-gradient aggregation.
- [Coupled Adam, 2025](https://arxiv.org/abs/2502.08441): shares adaptive second-moment denominator across token rows, not actual gradients.
- [Toward a First-Principles Update Geometry for the Language-Model Head, 2026](https://arxiv.org/abs/2608.22253): softmax-invariant centered row normalization, large model data, no semantic gradient grouping; a strong modern optimizer baselining reference.
- [A Deafening Silence, 2026](https://arxiv.org/abs/2610.09835): rare output-token gradient drift; supports optimizer epsilon baseline, does not introduce clusters.
- [Fast LM Head via Coarse-Fine Decomposition and Tile-Wise Weight Clustering, IEEE 2026](https://doi.org/10.1109/TC.2026.3725429): posttraining output-head codebook/candidate-screening inference acceleration, not training-time gradient sharing.

## Remaining gates before final publishability decision

1. Analyze all active runs and update method/novelty claims based on outcomes.
2. Run token-frequency and model-size robust comparisons, preferably multiple training data slices.
3. Strong 2026 optimizer baseline(s) tuned independently. Direct α-preconditioner requires optimization and scale validation.
4. Full-vocabulary no-UNK setting / realistic batch size / longer training; one non-toy modern LM benchmark if possible.
5. Clarify *which exact scientific claim* is novel and useful beyond existing shared embeddings, class-based loss, and frequency-aware optimizer mechanisms.
