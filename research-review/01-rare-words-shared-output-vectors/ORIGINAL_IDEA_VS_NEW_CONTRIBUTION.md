# Originality ledger — distinguish upstream idea from our independent research

**Source re-audited:** the original upstream draft [01/PAPER.md](../../01-rare-words-shared-output-vectors/PAPER.md) in the user's fork, not secondary descriptions. The original document itself contains a complete method, pseudocode, corpus experiments, frequency-exposure law, edge cases, and scaling discussion. This distinction governs honest paper authorship/attribution.

## Already present in the upstream author's paper — NOT our discoveries

1. **The core method itself**: output rows `R[w]=own[w]+C[c(w)]`; KMeans on current output vectors; periodic regrouping; reallocate `own` and `C` to preserve every output logit; merge back to one output matrix for inference. Section 2 already specifies this precisely.
2. **Group gradient aggregation**: Section 2 explicitly explains that a shared centroid receives the **sum of its member-token gradients**, including the rare-token “borrow a frequent neighbor's training” mechanism. Our operator identity `ΔR=−(η_U I+η_C AAᵀ)G` is a useful formal derivation of that already-described mechanism, **not invention of a new gradient-sharing method**.
3. **The exposure threshold and negative transfer**: the original abstract and Section 6 already state that gain concentrates on tokens seen roughly **10+ times**, and zero/near-zero exposure tokens can lose 0.2–0.45 nats from unreliable group membership. Our V50k bucket analysis **independently reproduces and extends** this to nearly all GPT-2 tokens, not a new observation or new theoretical claim.
4. **Length/batch/width/tie experiments**: upstream already ran up to 1600 steps at V6k, varied vocab up to24k, tuned output LR/centroid LR with width, tested tied input/output embeddings and large batches. Our tests add independent seeds and stronger baselines, but do not originate those dimensions.
5. **Comparators**: upstream tested plain Adam, tuned head epsilon, Coupled Adam, class-based/adaptive softmax, fixed random and unigram-frequency groups. Those comparisons should be acknowledged as the author's prior empirical work; repeating them isn't independent algorithm invention.

## Independent contributions that our lab actually added

| Contribution | Independent evidence | Novelty scope |
|---|---|---|
| More robust reproducibility (3 seeds, four train shards, CPU repeatability) | [TRAIN_SLICE_ROBUSTNESS.md](TRAIN_SLICE_ROBUSTNESS.md), [FULL_VOCAB_LONG_HORIZON_EVIDENCE.md](FULL_VOCAB_LONG_HORIZON_EVIDENCE.md) | Replication and validation, not a new algorithm |
| **Same-data fixed contextual/semantic clusters** vs online | [CONTENT_CLUSTER_SPECIFICITY.md](CONTENT_CLUSTER_SPECIFICITY.md), [MODERN_OPTIMIZER_AND_TIED_RESULTS.md](MODERN_OPTIMIZER_AND_TIED_RESULTS.md) | Corrects upstream's broad “fixed groups give no gain” claim: *random/frequency* fixed groups were tested there; content-informed fixed groups are much stronger |
| **Official 2026 Ember optimizer head-only**, grouped vs ungrouped, matched LR/beta2 | [MODERN_OPTIMIZER_AND_TIED_RESULTS.md](MODERN_OPTIMIZER_AND_TIED_RESULTS.md), [EMBER_V4_CODE_EQUIVALENCE.md](EMBER_V4_CODE_EQUIVALENCE.md) | New scientific comparison: directional group-gradient communication appears complementary to a modern row/column second-moment optimizer at V6k |
| Group-gradient kernel `AAᵀG`, normalized projector `A(AᵀA)⁻¹AᵀG`, and zero-sum invariance proof | [FORMAL_MECHANISM.md](FORMAL_MECHANISM.md), [PROJECTOR_LEMMA.md](PROJECTOR_LEMMA.md), [PROJECTOR_RESULTS.md](PROJECTOR_RESULTS.md) | Mathematical reinterpretation plus **new parameter-free optimizer variants**, empirically tested; each operator is simple linear algebra, so mathematical originality rests on formulation/application and strong results |
| Pinned-source V≈50k output head, two seeds, full exposure decomposition | [FULL_VOCAB_FREQUENCY_DECOMPOSITION.md](FULL_VOCAB_FREQUENCY_DECOMPOSITION.md) | Wider vocabulary and independent confirmation, not large-parameter LLM scaling |
| Full optimizer-state reset ablations | [FULL_RESET_CONTROL.md](FULL_RESET_CONTROL.md) | Clarifies difference between pseudo-code's resetting C optimizer state and executable original code's retention |

## Live falsification gates (do not preempt the data)

- [Independent-seed V16k/V50k Ember transfer](EMBER_SCALE_TRANSFER_PROTOCOL.md) tests whether modern-optimizer complementarity survives broader head sizes; Actions #44002552718.
- [Strong near-full-vocab contextual static groups](FULL_VOCAB_SEMANTIC_CONTROL_PROTOCOL.md) tests whether adaptive regrouping beats a meaningful fixed prior; Actions #44003042366.
- A >100M parameter LM and realistically large global training batch must be tested before a strong ICML/NeurIPS language-model training method claim.

## Publication/credit decision

A paper based only on `R=own+C[group]` with periodic exact-preserving KMeans would substantially repeat the public original author's work. **The upstream author must be cited and the adopted algorithm clearly credited.** Our lab's publishable question is different: *why does content-informed group-gradient coupling help on top of modern output-head optimizers, where exactly does it cause negative transfer, and what principled new update/geometric/reliability rule beats both strong static semantic groups and the original algorithm at realistic scale?* A distinct new method, robust mechanism or substantive new scientific finding could support an independent extension paper; replication alone is not an original-method publication.

Do not claim the V50k rare-token exposure threshold is newly discovered, nor claim that "fixed groups are useless" without distinguishing random/frequency from semantic static groups.
