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

## Second continuation — verified after history rewrite

- [#43851693792](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43851693792) **success**: 4 non-overlapping FineWeb training slices × 2 model seeds. All eight matched-arm comparisons favor dynamic grouping. Gains across offsets averaged **0.0180 / 0.0135 / 0.01605 / 0.01530** CE respectively. Independent write-up: [TRAIN_SLICE_ROBUSTNESS.md](../../research-review/01-rare-words-shared-output-vectors/TRAIN_SLICE_ROBUSTNESS.md).
- [#43851810419](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43851810419) **success**: V16k width128 target-frequency analysis. Dynamic grouping helps the extreme learnable rare tail more than same-data static context groups, but static context helps middle-frequency tokens more; both lose slightly on the most common tokens. A repeated seed0 run differed in dynamic CE by about 0.0057 from a previous job; numerical repeatability is now explicitly under test. Data: [RARE_TOKEN_BUCKETS_16K.md](../../research-review/01-rare-words-shared-output-vectors/RARE_TOKEN_BUCKETS_16K.md).
- [#43852372143](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852372143) **success**: bounded-spectrum/zero-sum-preserving cluster projector gradient mixing at α=1.0 achieved mean gain 0.01383 without learned centroid parameters, somewhat below original shared-parameter method's 0.0173. [PROJECTOR_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_RESULTS.md); mathematical lemma + test in [PROJECTOR_LEMMA.md](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_LEMMA.md).
- [#43854482667](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854482667) **success**: naive frequency-targeted attenuation of group-gradient mixing **did not improve** aggregate CE compared to ungated gradient mixing; this negative result is saved in [RARE_SELECTIVE_GRADIENT_RESULT.md](../../research-review/01-rare-words-shared-output-vectors/RARE_SELECTIVE_GRADIENT_RESULT.md).
- [#43852180766](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852180766): 1600-step comparison across baseline/online/static-context/late-hybrid, seeds0/1. Retrieve logs after completion.
- [#43854706295](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854706295): six repeatability runs on same seed V16k width128, threads1 vs2. Necessary to assess numerical variability before interpreting subtle hybrid gains.
- [#43854953562](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854953562): near-full GPT2 50k vocabulary, width64, 800 steps, two paired model seeds, baseline vs online KMeans, gradient accumulation to fit CI CPU memory. This is the current highest-priority scale falsifier.

**Priority after active jobs**: confirm the effect persists at full 50k token coverage and longer horizon; evaluate whether static semantic grouping remains stronger at realistic training scales. Existing results establish a robust *small-model* optimization effect, while method competitiveness versus modern strong alternatives is still a separate question.


## Third continuation — modern optimizer challenge + independent-scale confirmations

The earlier "active" run labels in the historical sections above are archival; all three 2026-11-16 long-horizon, repeatability and V50k jobs finished successfully.

- [#43852180766](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852180766) **success**: 1600-step two-seed V6k, online mean **4.71960** vs plain **4.73975** vs static-context **4.72550**; late-context-to-online **4.71840**. [Evidence](../../research-review/01-rare-words-shared-output-vectors/FULL_VOCAB_LONG_HORIZON_EVIDENCE.md).
- [#43854706295](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854706295) **success**: three `threads=1` repetitions exactly match to four decimals, two-thread variation ~0.0009 CE.
- [#43854953562](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854953562) **success**: V≈50k, 800steps, two independent seeds, paired improvements **+0.0306/+0.0293** CE. The 10–99 target-exposure bucket accounts for ~97% of total mean improvement, zero-exposure tokens lose: [source-verified decomposition](../../research-review/01-rare-words-shared-output-vectors/FULL_VOCAB_FREQUENCY_DECOMPOSITION.md).
- [#43927461457](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43927461457) **success**: 3-seed×8-grid Coupled Adam head baseline underperforms original head Adam, but some matched LR combinations benefit from clustering; no broad comparative claim against Coupled Adam in its intended regime.
- [#43928661510](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43928661510) **success**: official pinned Ember head-only on V6k, 3 seeds. Standalone Ember best mean4.99390 (beta2=.999,LR.05), effectively matches clustered Adam mean4.99443. At same beta2/learning rate adding dynamic groups improves to4.97960 (paired gain0.01430). At beta2=.95,LR.05 Ember plain4.99943 vs grouped4.97780 (paired gain0.02163).
- [#43929175351](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43929175351) **success**: tied input/output table, V6k, seeds0/1/2. Online grouping helps at400 and1600steps vs plain tying; fixed same-data contextual groups have stronger mean results at both horizons. [Full matrices](../../research-review/01-rare-words-shared-output-vectors/MODERN_OPTIMIZER_AND_TIED_RESULTS.md).
- [#44002828070](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44002828070) **success**: pinned original Ember and October-2026 v4 paper formula identical on dense softmax head across six seed/beta2 combinations; expected difference on sparse input-only embedding rows. [Algebra/provenance](../../research-review/01-rare-words-shared-output-vectors/EMBER_V4_CODE_EQUIVALENCE.md).

**New fixed-source, independent-seed experiments** (status must be checked on each continuation):
- [#44002552718](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44002552718), 3 fresh seeds × V16k/V50k, original Adam vs official Ember +/− online groups, and V16k static contextual groups. Precommit: [EMBER_SCALE_TRANSFER_PROTOCOL.md](../../research-review/01-rare-words-shared-output-vectors/EMBER_SCALE_TRANSFER_PROTOCOL.md).
- [#44003042366](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44003042366), 3 fresh seeds, full-vocab V50k same-data static context grouping vs static unigram vs late-online contextual adaptation. Precommit: [FULL_VOCAB_SEMANTIC_CONTROL_PROTOCOL.md](../../research-review/01-rare-words-shared-output-vectors/FULL_VOCAB_SEMANTIC_CONTROL_PROTOCOL.md).

**Publication hinge:** joint online group-gradient coupling and exactly function-preserving reassignment are distinct from the checked prior works, but not proof of a top-tier paper by themselves. A strong claim requires matched modern optimizer transfer **and** semantic clustering baselines at V50k, then meaningful LM scaling and compute measurement. Existing small-model signal is now well established.
