# Paper 01 — Independent heldout FineWeb shard002 confirms a real projector gain

**Completed successfully: [GitHub Actions #44017710007](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017710007), 3/3 jobs, all 15 preregistered trainings and independent-test evaluations.** New model seeds **809,811,821**; independent test `fineweb_train_000002.bin` (upstream nomenclature; **not used for training or tuning in this lab**). The original selection-validation remains `fineweb_val_000000.bin`. Source [locked protocol](LOCKED_FINAL_HELDOUT_5ARM_PROTOCOL.md). Model is V≈50k, width64, 2 layers, 800 steps, sequence64/batch32/acc4, same public train prefix shard001 for every arm.

## Paired end-of-training CE (lower better)

### Original already-used validation

| Arm | Seed809 | Seed811 | Seed821 | Mean CE |
|---|---:|---:|---:|---:|
| Upstream online centroid + Adam | 6.2550 | **6.2510** | **6.2562** | **6.25407** |
| Official Ember plain | 6.2841 | 6.2785 | 6.2836 | 6.28207 |
| Upstream online centroid + Ember | 6.2665 | 6.2550 | 6.2654 | 6.26230 |
| **New single-matrix class projector + Ember** | **6.2588** | 6.2530 | 6.2564 | **6.25607** |
| Fixed train-only context centroid + Adam | 6.2783 | 6.2816 | 6.2806 | 6.28017 |

### Untouched independent external test shard002 — **primary endpoint**

| Arm | Seed809 | Seed811 | Seed821 | Mean test CE |
|---|---:|---:|---:|---:|
| Upstream online centroid + Adam | 6.2926 | **6.2919** | 6.2999 | **6.29480** |
| Official Ember plain | 6.3115 | 6.3133 | 6.3205 | 6.31510 |
| Upstream online centroid + Ember | 6.2975 | 6.2927 | 6.3009 | 6.29703 |
| **New single-head normalized group projector + Ember** | **6.2895** | 6.2924 | **6.2992** | **6.29370** |
| Static train-context original centroid + Adam | 6.3136 | 6.3206 | 6.3192 | 6.31780 |

The projector compared to ungrouped modern Ember has three paired **heldout test improvements +.0220,+.0209,+.0213**, mean **+.02140 CE**, robust to this independent shard.

Against the upstream author's online split-centroid+Ember, paired test gains are **+.0080,+.0003,+.0017**, mean **+.00333 CE**, but the magnitude is small. Against the **stronger upstream online centroid+Adam**, test gains are **+.0031,−.0005,+.0007**, mean **+.00110 CE**: a *near tie*, **not a demonstrated accuracy improvement** over the upstream method.

### Crucial validation-versus-test ranking change

Original selection-validation favors upstream centroid+Adam (mean6.25407) over new Ember projector (6.25607) by .0020 CE. The out-of-shard **test** reverses the ranking by .0011 CE. This is too small for a reliable method selection; record both and preserve the preregistration, not whichever looks more favorable.

## Conclusions that survive selection bias

1. **Real positive effect:** a pretrained-source-shard-independent positive marginal benefit for a single-output-head normalized learned class-gradient projector **relative to the tested plain Ember setting**. This is a mechanism extension, not a new discovery of token group gradient sharing.
2. **No external accuracy SOTA over original:** stronger upstream online centroid+Adam essentially matches projector on independent test; the new method may be useful for optimizer simplicity or speed, but full large-scale cost must be measured.
3. **Vocabulary scale is not LM model scale:** actual V50k model is ~6.54M weights, ~98% token tables and ~0.1M contextual backbone. This test is **not** pretraining a 50M or 100M model.
4. **The original dynamic projector does not have batch robustness:** the batch128 condition failed against plain Ember. [Context-first grouping fixes that failure at V6k](HIGH_BATCH_CONTEXT_RESCUE_RESULTS.md) but has not been independently tested on external shard002 or on a large-batch large model.
5. The heldout test includes the **dynamic** projector, **not** the newly proposed context-first projector; do not advertise independent heldout evidence for the latter.

Additional hard baselines are in progress: [fair standalone Ember V16k LR sweep](EMBER_HEAD_LR_TUNING_CHALLENGE_V16K.md) and [semantic label-smoothing collision](SEMANTIC_SMOOTHING_COLLISION_PROTOCOL.md). A credible independent original-method paper would also need better absolute gains vs the upstream original, a >100M-parameter LM test, and compute-matched training budgets.
