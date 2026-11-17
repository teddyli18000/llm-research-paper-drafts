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


## Fourth continuation — independent normalized projector; successes and falsification

All notes below concern the **independent new single-head projector** `G'=(I+2A(AᵀA)⁻¹Aᵀ)G`, which is distinct from the original author's trainable `R=U+AC` parameterization. Attribution is controlled by [originality ledger](../../research-review/01-rare-words-shared-output-vectors/ORIGINAL_IDEA_VS_NEW_CONTRIBUTION.md).

- [#44002552718](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44002552718) **success 6/6**: independent seeded V16k/V50k baseline and modern head optimizer comparisons. At V16k the best existing baseline is fixed train-context Adam mean CE5.78023 vs online Adam5.78330; at V50k online Adam6.25773, nearly equal to grouped Ember beta2=.999 at6.25877. Within matched optimizer grouping is positive on every seed. [Full results](../../research-review/01-rare-words-shared-output-vectors/INDEPENDENT_SCALE_AND_PROJECTOR_RESULTS.md).
- [#44003042366](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44003042366) **success 3/3**: V50k train-only contextual group mean6.28010, frequency groups6.27827, context→late online6.27000, pure online6.25773. [Full results](../../research-review/01-rare-words-shared-output-vectors/INDEPENDENT_SCALE_AND_PROJECTOR_RESULTS.md).
- [#44004122511](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44004122511) **success 3/3**: V6k normalized projector α2 mean4.97640, original shared-centroid Ember4.97723, Ember plain5.00227 (near-tie to upstream on tiny model).
- [#44017278316](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017278316) **success 3/3**, 27 registered V6k causal controls: normalized projector with learned KMeans mean4.97290 vs size-matched shuffled4.99487, random4.99027, one-time4.99317, frequency4.99820; **strong train-only fixed context groups win at4.96843**. Doubling plain Ember LR only gets4.99773. [Full proof](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_CAUSAL_ABLATION_RESULTS.md).
- [#44017422802](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017422802): α2 projector V16k/50k registered on new seeds137/149/163. **All three V16k jobs successful:** projector mean5.75807 vs original centroid+Ember5.80693, centroid+Adam5.78117, Ember plain5.82877. Clear paired V16k win, [details](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_V16K_INDEPENDENT_RESULT.md). **V50k jobs still in progress**.
- [#44018218531](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018218531) **success 6/6**: at fixed819,200 training tokens batch32 projector benefit over Ember plain **+0.02947** and over original centroid **+0.00663**; at batch128 projector **hurts** plain by0.00570 and underperforms original centroid by0.01313. Preserve this negative result, [full data](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_BATCH_SCALING_RESULTS.md).
- [#44017710007](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017710007): fully locked 5-arm external heldout on untouched FineWeb shard002, three fresh seeds809/811/821, in progress; [protocol](../../research-review/01-rare-words-shared-output-vectors/LOCKED_FINAL_HELDOUT_5ARM_PROTOCOL.md).
- [#44018640211](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018640211): scale transfer of fixed contextual vs dynamically learned normalized projector at V16k/V50k, fresh181/191/197, in progress.
- [#44018942584](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018942584): exploratory batch128 rescue of original failed normalized projector by initializing fixed context groups and/or more frequent KMeans refresh; fresh281/283/293, in progress.
- [#44019195126](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44019195126): measure actual fraction of head gradient energy in learned vs shuffled/context-group subspace across steps; diagnostic replay, in progress.

**Core unresolved decision:** the new operator is genuinely different and surprisingly strong on three-seed V16k but is **not batch-robust in its original dynamic form**. The high-batch and V50k external studies will determine whether a semantic-context-first variant offers a real new paper, or whether this is a bounded toy-regime optimizer effect. None of the upstream author's original discoveries are claimed as new.

## Fifth continuation — heldout results and full-vocabulary convergence checked

- **#44017422802 fully green 6/6.** V50k three unseen seeds137,149,163 projector+Ember mean CE **6.25553**, original centroid+Ember **6.25647**, original centroid+Adam **6.25580**, ungrouped Ember **6.27967**. Near-tie **vs original shared-centroid methods**, stable +.0241 vs ungrouped modern optimizer. [Source-verified V50k results](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_V50K_INDEPENDENT_RESULT.md).
- **#44017710007 fully green 3/3, untouched external FineWeb shard002.** Projector test CE **6.29370**; plain Ember **6.31510**, original centroid+Ember **6.29703**, original centroid+Adam **6.29480**, original fixed-context centroid+Adam **6.31780**. Against strongest original centroid+Adam only **+.00110 CE** mean, so no demonstrated win. [Full heldout report](../../research-review/01-rare-words-shared-output-vectors/INDEPENDENT_HELDOUT_V50K_RESULTS.md).
- **#44018640211 fully green 6/6.** V50k three new seeds: dynamic projector **6.25907**, static contextual projector **6.28383**, context→online after step200 **6.26030**, original centroid+Ember **6.25913**, plain Ember **6.28697**. Dynamic beats fixed-context at V50k. [Contextual scale report](../../research-review/01-rare-words-shared-output-vectors/CONTEXT_PROJECTOR_V50K_RESULTS.md).
- **#44020024307 fully green 3/3**: 10-grid official Ember standalone V16k optimizer challenge on fresh seeds331/337/347, best bare Ember beta2=.999 LR.01 **5.78270** vs best projector α2 beta2=.95 LR.03 **5.74967**; original centroid+Adam **5.78377**. Projector gain **+.03303 CE**, so V16k improvement is not simply a fixed poor Ember LR. [Tuned baseline](../../research-review/01-rare-words-shared-output-vectors/TUNED_EMBER_V16K_CHALLENGE_RESULTS.md).
- Critical exact-method collision with semantic label smoothing: projector head row gradient `G'_i=[p_i-y_i+α(p_g-y_g)/n_g]h`, whereas standard within-group soft-target smoothing yields `G_LS_i=[p_i-(1−ε)y_i−ε(Py)_i]h` lacking prediction-side class probability term. Compare head-only and full-loss smoothing; documents [decomposition](../../research-review/01-rare-words-shared-output-vectors/GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md) and [precommitted 36-arm protocol](../../research-review/01-rare-words-shared-output-vectors/SEMANTIC_SMOOTHING_COLLISION_PROTOCOL.md).
- Smoothing suite #44022506092 failed at analytic unit test due wrong group-size broadcasting; #44028843648 failed at second occurrence of same indexing; both exact faults fixed and code committed, third run **#44029034123** currently running 3×12 full V6k trainings, gradient identities pass. Preserve failures in the research record; no false green.
- New Undermind deep literature prior-art search for exact class-mean prediction-error transfer in optimizer vs label smoothing launched, workspace `8ce665de-7826-4793-88b5-47888b642743`. Check for finished and read closest papers before final novelty verdict.

**Current judgement:** the *parameter-free normalized projector* is a genuine different update rule with validated marginal gain over ungrouped modern Ember, but at full GPT2 vocabulary the best upstream centroids remain essentially equal. Competitive method paper hinges on larger-model/large-batch results, strong semantic smoothing baselines, or a new reliability-aware variant that materially beats original method. The original user's public source remains untouched, and credit to its algorithm is mandatory.

## Sixth continuation — collision with existing class-parametrization and smoothing

- Confirmed fully completed three key prior registered matrices: V50k projected Ember #44017422802 success 6/6; independent FineWeb heldout #44017710007 success 3/3; V16k/V50k context vs dynamic #44018640211 success 6/6; tuned Ember V16k #44020024307 success 3/3. All exact raw per-seed values in [PROJECTOR_V50K_INDEPENDENT_RESULT.md](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_V50K_INDEPENDENT_RESULT.md), [INDEPENDENT_HELDOUT_V50K_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/INDEPENDENT_HELDOUT_V50K_RESULTS.md), [CONTEXT_PROJECTOR_V50K_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/CONTEXT_PROJECTOR_V50K_RESULTS.md), [TUNED_EMBER_V16K_CHALLENGE_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/TUNED_EMBER_V16K_CHALLENGE_RESULTS.md).
- **New strongest prior-art collision:** [Tree Loss AISTATS 2022](https://proceedings.mlr.press/v151/wang22d.html) trains class-shared ordinary-CE classifier parameterizations; source PDF pages2–3 read. Exact proof for our fixed-P projector: `B=I+(sqrt(1+α)−1)P`, `B²=I+αP`; reparam `R=BU` optimized by SGD induces the *same* update on R as our direct `G'=G+αPG`. CI #44029545346 success validating unequal group sizes with autograd. Strong novelty correction: class-based gradient sharing and shared-representation preconditioning existed; our candidate is the **direct fixed-head projection into a modern adaptive Ember/Adam optimizer plus normalized group subspace**, not new SGD mathematics. See [TREE_LOSS_PRECONDITIONER_COLLISION.md](../../research-review/01-rare-words-shared-output-vectors/TREE_LOSS_PRECONDITIONER_COLLISION.md).
- New exact-provenance method challenge #44029716200 (three unused seeds373/379/383 × 5 arms) compares direct normalized projector vs mathematically matched single-matrix linear reparam using Ember and Adam at equal initial effective head logits. Precommit [TREE_REPARAM_EMBER_COLLISION_PROTOCOL.md](../../research-review/01-rare-words-shared-output-vectors/TREE_REPARAM_EMBER_COLLISION_PROTOCOL.md). In progress last poll.
- New collision: semantic target-label smoothing `G_LS=p−[(1−ε)y+ε Py]` is **different** from projected residual `G'=e+α P(p−y)`, which includes class-level prediction-error `Pp`. EMNLP2024 Syntactic Smoothing published such learning-signal redistribution; 2026 EACL *Beyond Tokens* is further concept-level objective prior. Exact derived [GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md](../../research-review/01-rare-words-shared-output-vectors/GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md).
- Smoothing benchmark precommitted 3×12=36 arms new seeds353/359/367 including label-smoothed true full CE and head-only gradient controls. Original #44022506092 + rerun #44028843648 failed at **test broadcast dimensionality** in group indexing; third #44029034123 passed both independent gradient identity tests but 3 training jobs failed at same issue in runtime head-only smoothing expression. Patched group size indexing and launched #44029810293. Report only completed full outcome; do not claim success before Actions verifies.
- Undermind deep collision search completed, but full-PDF reads temporarily rate-limited. Independently used official AISTATS PDF screenshots/source and ACL Anthology, not assistant speculation; note stronger prior Tree Loss, SGG, Syntactic Smoothing, concept-level prediction. The Tree Loss equivalence materially reduces claim of *new optimizer mathematics*.
- **Current academic position:** full-vocab projector marginally beats plain Ember on independent heldout (by0.0214 CE) but ties upstream stronger grouped Adam within0.0011 CE. Tuned V16k gap is real but backbone is tiny. Major original research paper still requires strong improvement in meaningful large-model/batch regime, ideally with reliability-informed group structure and new mechanism not reducible to Tree Loss.


## Seventh continuation — full class-projection collision matrix

- **Tree-Loss reparameterization #44029716200 completed 3/3.** Direct fixed-P projection and the matched linear reparameterization are near-tied under both Ember and Adam. This confirms the SGD equivalence is not enough to explain away the adaptive-optimizer implementation, but the measured difference is tiny and seed-inconsistent. The result is in [TREE_REPARAM_EMBER_COLLISION_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/TREE_REPARAM_EMBER_COLLISION_RESULTS.md).
- **Semantic smoothing #44029810293 completed 3/3, 36 arms.** Head-only and full within-group soft-target smoothing at epsilon .25/.5/.75 are all substantially worse than the projector. This is not an exact residual-transform collision: standard target smoothing contains the target-side -Py term but no prediction-side Pp term. The exact gradient distinction is in [GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md](../../research-review/01-rare-words-shared-output-vectors/GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md).
- **Concept-class CE #44030216366 completed 3/3.** Auxiliary class-probability CE does not reproduce the projector.
- **Train-gradient coherence gate #44031012683 completed 3/3.** No stable gain over the simpler projector/energy-gate baseline.
- **Cross-microbatch reliability #44034386895 and #44034639792 completed.** Rho selection and rho-scaled strength do not beat the energy gate consistently.
- **Stable capture #44037476138 completed 6/6.** Eta selection and eta-positive gating fail the preregistered decision rule: eta select means 4.96533 at batch32 and 5.06033 at batch128, versus energy gate 4.96290 and 5.06120. The small batch128 difference does not compensate for batch32 degradation. Results and traces are in [MICROBATCH_RELIABILITY_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/MICROBATCH_RELIABILITY_RESULTS.md). Stop tuning global rho/eta selection.

## Eighth continuation — moment routing and 2026 LM-head prior art

- **Decoupled second-moment routing #44043530225 completed 6/6.** The analytic identity test passed. Raw-stat routing (projected numerator, raw Ember second moments) loses to shared-stat routing for fixed context and context-to-dynamic on all three seeds at both batch sizes. Mean paired CE penalties are +0.06080/+0.06903 for context and +0.07607/+0.04213 for context-to-dynamic at batches32/128. Dynamic is also worse on average (+0.04370/+0.04273). Raw-stat routing raises actual head update norm by roughly 1.4–1.9×; the modified numerator overshoots when Ember's factored denominator no longer adapts to it. Do not claim generic moment routing as new; Hu et al. (2026) already establish the principle in Adam continual learning. Full result: [MOMENT_ROUTING_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/MOMENT_ROUTING_RESULTS.md).
- Frequency analysis shows a specific tradeoff: raw-stat context routing improves the zero-training-exposure validation bucket by about 0.68 CE (batch32) and 0.77 CE (batch128), but worsens every nonzero-exposure bucket. The zero-exposure validation mass is only 0.01%, so overall CE becomes much worse. This is an exposure-selective effect, not a general optimizer gain.
- **Context-to-dynamic schedule is batch-regime dependent.** With a switch after 20% of training, it improves fixed context by 0.00320 CE at batch32 on all three seeds, but loses by 0.00363 at batch128 on all three seeds. A future schedule test should use an absolute optimizer-step threshold (e.g. switch at step200), leaving the 100-step batch128 run on context groups.
- Full-text prior-art audit completed for 2026 LM-head optimizers: [2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md](../../research-review/01-rare-words-shared-output-vectors/2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md). Important updates:
  - arXiv:2607.01455 current v4 title is *Token Geometry*; Ember is the optimizer introduced in it. It frames Ember as a square-root empirical-Fisher preconditioner and uses factored row/column second moments without first moment.
  - Somasundaram et al. (2026) projected RowNorm and Lau & Su (2026) symmetry-compatible optimizers use global softmax-shift centering and rowwise normalization/spectral updates, not token-group averaging. For an exact softmax head gradient, 1ᵀG=0, so global centering leaves G unchanged; the group projector P_A=AD⁻¹Aᵀ is a different nontrivial blockwise-averaging operator.
  - MuonIO (2026) normalizes each vocabulary row independently and aggregates row norms; it does not share row directions across token groups.
  - Han et al. (2026) output-only Adam epsilon is a strong absent-token forgetting baseline; add it before any rare-token-specific claim.
  - Godey & Artzi's COLM 2026 LM-head gradient bottleneck is a separate rank-D backward-compression problem; the row-group projector does not solve it.
- Current evidence supports a narrow normalized class-mean head-gradient preconditioner, not a new general class-sharing principle. It beats plain Ember in several matched regimes but is near-tied with the upstream grouped-centroid method at V≈50k and has batch-dependent grouping behavior.

## Next research step

Run an exact residual decomposition with fresh seeds: compare the full projector G + αP(p−y) against prediction-side-only G + αPp and target-side-only G − αPy, using the same fixed context and context-initialized online groups. This directly tests the distinctive Pp term that target smoothing lacks and can explain the zero-exposure bucket behavior. After that, test an absolute-step context→dynamic switch if the batch-regime pattern persists. Add projected RowNorm/MuonIO and output-only epsilon baselines before a final novelty verdict.


## Ninth continuation — exact residual decomposition and expanded prior-art search

- **Residual decomposition #44046155032 completed 6/6, 48 training runs.** The analytic identity test and per-step matrix-projection equivalence passed; maximum observed discrepancy was below 1.6e-8. Full results: [RESIDUAL_DECOMPOSITION_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/RESIDUAL_DECOMPOSITION_RESULTS.md).
- Fixed-context full projector mean CE: 4.96840 (batch32), 5.06013 (batch128). It beats plain Ember by 0.03477/0.02020 and upstream centroid+Ember by 0.01100/0.01483, with all three paired seeds agreeing in both regimes.
- In the uncentered residual ablation, prediction-only and target-only corrections are much worse than the full projector, while context-group label smoothing is also worse. However, the one-sided terms contain opposite vocabulary-uniform modes that the full difference cancels. The raw one-sided outcomes are therefore confounded and do not establish that both centered terms are necessary. A centered follow-up with uniform-only controls is running as Actions #44052591527; see CENTERED_RESIDUAL_PROTOCOL.md.
- Dynamic K-means is worse than fixed context by +0.00523/+0.02153. Context→dynamic at absolute step 200 improves batch32 by 0.00500 on all three seeds; batch128 has only 100 steps and therefore remains fixed-context, matching it exactly.
- Frequency buckets show that full context projection improves most exposure bands, not only zero-exposure rows. Zero-exposure targets are just 0.01% of validation mass. Prediction-only/target-only have large zero-exposure improvements but degrade the broader distribution, especially the unknown bucket.
- **New deep literature search completed:** /01-output-embeddings/Exact prediction-side class residual projector prior art. Its abstract/metadata pass found no retrieved paper with the exact flat untied head update (I+αP)G, but this is not yet a proof of absence. Full-text checks:
  - Tree Loss [Wan22]: path-incidence reparameterization W=SV induces SSᵀG, a path-overlap Gram preconditioner. It is not generally the normalized orthogonal projector A(AᵀA)⁻¹AᵀG; equivalence requires a special matrix identity. The paper does not claim the proposed optimizer-side transform. Its printed U-tree difference sign in Eq. 2 conflicts with the path-sum identity in Eq. 3; document this equation caveat.
  - Syntactic Smoothing [Mar24]: target-only redistribution. Its implied gradient is T p − t; there is no prediction-side Pp. Its printed target normalization is ambiguous: Eq. 3 normalizes over the wrong index for a per-target distribution, and row-wise temperature softmax does not resolve the column-sum issue.
  - Softmax Gradient Tampering [Das21]: nonlinear normalized-power transform p'_i=p_i^α/Σ_j p_j^α, then p'−y; not a fixed projector, and target is unchanged.
  - AGG [Yu21b]: dynamically frequency-gates negative-class contributions to rare output rows; no gradient transfer between rows or class-mean projection. WikiText-103 GPT-2-medium: rare-token PPL 438.67→75.39, total PPL 15.51 unchanged; this is a strong rare-token baseline.
  - Hypernym Class Prediction [Bai22]: mixed vocabulary softmax over hypernym class symbols and unreplaced tokens. Original mapped token rows are excluded from HCP output support; class-target gradients accrue to class rows, not distributed to member token rows. Not the flat-head projector.
  - Why Adaptive Optimizers Underestimate Rare Tokens [Kar26b]: RMSProp periodic-arrival fixed-point theory for unigram output biases, with empirical small-LM rare-token underprediction. It does not propose group transfer; its theorem is not a contextual output-matrix-row theorem. Relevant baseline controls include SGD, AMSGrad, Coupled Adam.
  - Full-text PDFs for Cluster-based Soft Targets for Output Regularization [Mei23] and Improved Learning through Augmenting the Loss [Ina16b] were unavailable in this pass; do not claim their exact equations have been ruled out.
- Deep search and its card: [Undermind search](https://app.undermind.ai/projects/8ce665de-7826-4793-88b5-47888b642743?path=%2F01-output-embeddings%2FExact%20prediction-side%20class%20residual%20projector%20prior%20art).

## Next research step

1. Test the context→dynamic switch point at absolute steps 150, 200, 250, and never-switch on fresh seeds at batch32; keep batch128 fixed-context.
2. Add output-only Adam epsilon 1e-4, projected RowNorm, and MuonIO as head-optimizer baselines, matching token budget and validation buckets.
3. Continue the prior-art pass on the unavailable cluster-soft-target and metric-augmentation papers; then search specifically for blockwise class-residual preconditioning and gradient-sharing formulations beyond the current retrieved set.
4. Only after those comparisons, make a final novelty and paper-viability decision.


## Tenth continuation — exact normalized-centroid equivalence

- A stronger algebraic collision has been derived and committed in [2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md](../../research-review/01-rare-words-shared-output-vectors/2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md):
  - Let \(D=A^\top A\), \(P=AD^{-1}A^\top\).
  - The parameterization \(W=U+\sqrt{\alpha}AD^{-1/2}C\) has gradients \(\nabla_U L=G\) and \(\nabla_C L=\sqrt{\alpha}D^{-1/2}A^\top G\).
  - Same-LR SGD on U and C induces exactly \(\Delta W=-\eta(I+\alpha P)G\).
  - Therefore the normalized projector is exactly a normalized shared-centroid reparameterization for fixed groups under SGD. The unnormalized \(U+AC\) baseline is not the exact match.
- Added normalized_centroid_equivalence_test.py, a 25-step nonlinear softmax test that verifies the two output-matrix trajectories agree to <2e-12 in float64.
- Preregistered a direct-projector versus normalized-centroid comparison under Ember and Adam: [NORMALIZED_CENTROID_ADAPTIVE_PROTOCOL.md](../../research-review/01-rare-words-shared-output-vectors/NORMALIZED_CENTROID_ADAPTIVE_PROTOCOL.md). Trial runner is normalized_centroid_adaptive_trial.py; workflow registration remains to be added after the current switch-schedule run finishes.
- This changes the novelty boundary: the fixed-group SGD update geometry itself is not new. The empirical question is whether a direct flat-head update has a useful practical/adaptive-optimizer advantage over the exact scaled-centroid parameterization, and whether evolving semantic groups add value.



## Eleventh continuation — absolute-step switch schedule completed

- **Context-switch schedule #44050503629 completed 3/3.** Four arms per seed, 12 runs, 819,200 tokens each. Full results: [CONTEXT_SWITCH_SCHEDULE_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/CONTEXT_SWITCH_SCHEDULE_RESULTS.md).
- Fixed context mean CE 4.97100. Step150 4.96650; step200 4.96627; step250 4.96667. All three switch thresholds beat fixed context on every seed.
- The preregistered rule says thresholds within 0.002 CE should prefer the later switch. All three are within 0.00040, so **step250 is selected**, even though step200 has the smallest raw mean. This preserves context groups longer.
- Step250's bucket profile improves zero-exposure and 1–31 exposure, 3k+ and unknown; the 32–999 bands regress slightly. The schedule gain is small but seed-consistent and is not uniform across frequency.
- Batch128 remains fixed context because its 100-step horizon never reaches step250.



## Twelfth continuation — center the one-sided residual controls

- Re-auditing CLASS_RESIDUAL_COMPONENT_IDENTIFICATION.md exposed a confound in the prior one-sided residual ablation. For each batch, both Pp and Py sum across vocabulary rows to the same mean hidden vector. Their raw one-sided updates therefore include +U or -U, where U=(1/V)1 hbarᵀ. The full difference cancels U.
- U shifts all current logits by the same scalar per hidden state, so it is softmax-invariant for the current forward pass; under Ember it can still change parameter values and adaptive moment state. This can explain part of the raw one-sided degradation.
- The full-projector results and full-projector frequency buckets remain valid. The one-sided causal interpretation is withdrawn pending centered controls.
- Added CENTERED_RESIDUAL_PROTOCOL.md and centered_residual_decomposition_trial.py, plus an analytic test that verifies both centered corrections sum to zero, their difference equals PG, and the uniform mode leaves current softmax probabilities unchanged.
- **Centered residual run #44052591527 is queued/running:** 3 fresh seeds × 2 batch sizes × 8 arms, including centered/uncentered prediction-only, centered/uncentered target-only, full projector, and uniform-only plus/minus controls.



## Thirteenth continuation — graph-proper scoring and concept-level LM objectives

- Full-text audit found a closer mathematical neighbor: **Hilfiker et al. 2026, Strictly Proper Scoring Rules for Node Selection Tasks on Graphs [Hil26]**. Its diffusion score has probability gradient \(2H_\tau(p-y)\). For a disjoint union of complete within-class cliques, \(H_\tau=P+e^{-\tau n/2}(I-P)\), so \(H_\tau\to P\) as \(\tau\to\infty\). Thus the class-mean projector already appears as a probability-space graph-diffusion residual operator.
- The distinction is the softmax chain rule: the graph score's logit gradient is \(2J_{\mathrm{softmax}}P(p-y)\), whereas our optimizer-side transformation adds \(P(p-y)\) directly to the CE logit gradient. This is a strong adjacent prior-art collision, not an exact algorithmic match. Add the P-projected Brier/diffusion score as a scalar-loss baseline.
- **Conceptizer [Zhe26c]** keeps full token softmax and derives concept probabilities through token-membership weights. Its concept-loss gradient is probability-weighted within each concept; it is not uniform class-mean broadcast.
- **Learning Concepts, Not Tokens [Zha26u]** uses contextual synonym-set loss \(-\log\sum_{j\in C}p_j\), with logit gradient \(p_j-\mathbf1[j\in C]p_j/P_C\). The existing concept-class CE collision already tests the single-membership set-likelihood form on our context groups, though not the papers' contextual synonym construction or multi-membership target.
- These findings and exact equations are in [2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md](../../research-review/01-rare-words-shared-output-vectors/2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md).



## Fourteenth continuation — normalized-centroid adaptive comparison completed

- **Actions #44051580714 completed 6/6, 36 runs.** The 25-step nonlinear SGD equivalence test passed to <2.3e-15 maximum error.
- Direct projector beats plain Ember and plain Adam at both batches on all seeds. The exact normalized-centroid parameterization also beats both plain baselines.
- Direct versus scaled-centroid under Ember: batch32 mean paired CE −0.01660, all three seeds; batch128 −0.00127, seed-inconsistent.
- Direct versus scaled-centroid under Adam: batch32 −0.02477, all three seeds; batch128 −0.00197, seed-inconsistent.
- The batch32 gap is concentrated in zero/low-exposure buckets; batch128 is near-tied. Output update norms do not explain the gap monotonically.
- This establishes an adaptive-coordinate effect candidate, not a general advantage. A matched head-LR sweep (0.025/0.05/0.10) is now the next falsification step. Full result: [NORMALIZED_CENTROID_ADAPTIVE_RESULTS.md](../../research-review/01-rare-words-shared-output-vectors/NORMALIZED_CENTROID_ADAPTIVE_RESULTS.md).



## Fifteenth continuation — graph-diffusion proper-score baseline registered

- Hilfiker et al. [Hil26] exact clique-graph heat-kernel limit has been verified from full text: for unequal clique sizes \(n_g\), \(H_{\tau,g}=P_g+e^{-\tau n_g/2}(I-P_g)\), and \(H_\tau\to P\). The limit score is proper but not strictly proper; finite tau is strictly proper. Resistance score uses \(L^\dagger_g=(1/n_g)(I-P_g)\).
- Registered a head-only scalar-loss collision: P-limit score \(r^\top P r\), finite diffusion score at tau .02, and resistance score, each added to ordinary CE. Their logit gradients include \(J_{\rm softmax}\), unlike the direct projector. Protocol: [GRAPH_DIFFUSION_SCORE_PROTOCOL.md](../../research-review/01-rare-words-shared-output-vectors/GRAPH_DIFFUSION_SCORE_PROTOCOL.md).
- Added an analytic test verifying score formulas and gradients for unequal clique sizes. Trial/workflow are registered; execution will begin after the current centered-residual and LR-sweep matrices.
