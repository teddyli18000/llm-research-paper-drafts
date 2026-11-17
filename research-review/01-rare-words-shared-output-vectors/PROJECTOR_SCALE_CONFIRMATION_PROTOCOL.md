# Paper 01 — α=2 parameter-free group projector transfer to larger vocabularies

**Pre-registered after pilot, before confirmatory runs.** Original [V6k pilot](INDEPENDENT_SCALE_AND_PROJECTOR_RESULTS.md) selected α=2 by mean loss; its ~0.00083 CE advantage over the upstream split-centroid method is *too small* to call a win. A scale test is necessary.

## Frozen settings

- Exact original small Transformer + public FineWeb pretokenized data, untied softmax, 2 layers, no new architectural modifications.
- V16k width128/heads4/steps400/batch32/L64, head Adam LR.03, centroid LR.005.
- Near-full V50k width64/heads4/steps800/batch32/L64/accum4, head Adam LR.05, centroid LR.01.
- **New initialization seeds 137,149,163**, not used in previous scale studies or α grid.
- 64 groups recomputed from equivalent output rows by the same KMeans and schedule steps50,100,...; in the projector arm, only the single head matrix exists (`R`, no shared trainable `C`) and its accumulated gradients are replaced by `G+2A(AᵀA)⁻¹AᵀG` just before official Ember.
- Output-table Ember LR **.05** both scales, beta2 fixed **.95 at V16k** and **.999 at V50k**, selected *before* running this test based on the winner of the earlier registered V16k/V50k Ember hyperparameter transfer. Ember epsilon1e−8, no optimizer changes to backbone/embedding.
- Matched arms per seed and scale:
  1. Official head-only Ember without any groups.
  2. Upstream `R=U+AC` online centroid-group method + Ember on `U`, centroid Adam on `C`.
  3. **Our** single-matrix normalized group-projector Ember (α2).
  4. Original tuned plain-head Adam + online centroid groups.
- **24 full trainings** = 2 scales × 3 seeds × 4 arms. Original-model initialization and all data match within seed/scale.

## Measurements

Paired CE for `Ember_plain − Projector`, `Ember_centroid − Projector`, and `Adam_centroid − Projector`; target-frequency contribution bins and method wall training time. Report separately at each scale. A projected-head advantage smaller than host-variation ~0.001–0.005 CE cannot be called a practical algorithmic improvement.

If the projector matches split centroids consistently while being measurably faster/lower state, this may support a new optimizer implementation tradeoff even without accuracy SOTA. If it loses significantly to the existing method or plain strong baseline at both scales, it is not a compelling new method. The large-batch and 100M-parameter gates remain.
