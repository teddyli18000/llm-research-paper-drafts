# Paper 01 — Near-full vocabulary strong semantic grouping comparator

**Pre-registered before runs, 2026-11-16.** Current V50k adaptive result is positive but the earlier strong fixed same-data context partition has not yet been tested on V50k. Avoid concluding dynamic clustering is essential without this control.

## Frozen setup

- Model: original instrumented `grid.py`, V argument50258 (near-full GPT-2), width64, two blocks, four heads; batch32, sequence64, accumulation4, 800 updates, head LR.05, centroid LR.01, head epsilon1e−5.
- Model initialization seeds: **41,73,109** (same new seeds already precommitted for Ember V50k transfer, but these semantic runs use a separate workflow and the same fixed 600 validation windows).
- Training data: same original FineWeb GPT-2 tokenized shard prefix of 1,638,400 tokens for **both semantics estimation and LM training**. No pretrained external embedding teacher or validation labels enter the static clusters.
- Construct 64 semantic groups once by symmetrized adjacent-token cooccurrence, TF-IDF, 48-dimensional truncated SVD, normalized rows, KMeans n_init4/max_iter50/random_state0. The mean consumed token span equals the training prefix. Time the preprocessing stage and report it as **extra method compute**.
- **Arms:** (1) fixed semantic groups; (2) initialize from these fixed semantic groups, then start online KMeans regrouping at step200, every50 steps; (3) fixed unigram-frequency mass groups (same 64 group parameters). All arms have equivalent `R=U+AC` parameter count and trainable shared centroid structure.
- Main comparisons are with already-running `adam_dynamic` and `adam_plain` at V50k with **the same independent seeds and training hyperparameters** in [EMBER_SCALE_TRANSFER_PROTOCOL.md](EMBER_SCALE_TRANSFER_PROTOCOL.md), and the historical seeds0/1 full-vocab replication.

## Interpretation / checks

- If same-data static context outperforms dynamic, then adaptive clustering is not the strongest mechanism at V50k and paper framing must emphasize **good learned sharing structure** rather than repeated assignment.
- If late dynamic improves over context static at matched seed, report paired differences and adaptation cost; the V6k 1600-step pilot suggested a modest gain but its ~0.0012 CE margin was small.
- Record cluster sizes and empty-data rows, token occurrence bins (0,1–9,10–99,100+), corpus frequency bins, whole-val CE and per-arm training duration.
- Preprocessing (SVD/KMeans) is external training-time compute and **must be counted** in realistic cost comparisons. Inference full-softmax evaluation remains unchanged when (R=U+AC) is folded.
- Three seeds only: preserve all outcomes; do not select best seed, adjust K or modify semantic features after results.
