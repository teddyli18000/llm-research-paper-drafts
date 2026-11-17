# Paper 01 — Fair tuned standalone Ember challenge at V16k

**Frozen prior to runs.** The three-seed V16k normalized-projector test [#44017422802](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017422802) favored the proposed α2 group-mean gradient projector over official Ember LR.05, β₂=.95 and original centroid methods. A critic could argue that this is **only repairing a poorly tuned standalone Ember head**, since the strong no-group Adam variant uses lower LR=.03 at width128. Before claiming algorithm advantage, tune the ungrouped Ember optimizer fairly.

## Model and objective

- Same 2-layer small Transformer, public GPT2-token FineWeb train shard V16000, width128/4heads, batch32,L64,400steps, untied output table. Same original body/embedding optimizers and no model changes.
- New seeds **331,337,347**, never used by the previous V16k LR selection, α2 selection or strong semantic-group studies. Original 600-window validation reused for this *selection benchmark*, not external generalization.
- All Ember heads use pinned official [katop1234/ember](https://github.com/katop1234/ember/tree/76f30117b2d948874cd1c965c4f6813fa9de8caa), epsilon1e−8, body/schedule fixed.
- **Ten standalone Ember combinations**: output-head LR `{.01,.02,.03,.05,.08}` × β₂ `{.95,.999}`.
- **Two new projected Ember controls:** fixed α2 K64 online KMeans refresh50, LR=.03 and .05, β₂=.95; no hyperparameter tuning of α.
- **One original centroid+Adam control:** head LR=.03, centroid LR=.005, head epsilon1e−5.
- Total 13 arms×3 seeds=**39 full trainings**.

## Locked readout

For every seed report CE of every standalone Ember variant and the two projected choices. Select best standalone by **3-seed mean**, not by individual best seed; compare to the registered projector α2/.05/beta.95 on the *same* seeds, and to original centroid Adam. This directly tests whether original projectile win was just a poorly chosen Ember learning rate.

This is a *hyperparameter selection study*. A winning projector comparison here still requires the independent-shard/realistic-scale gates. Include all 39 outcomes, unsuccessful as well as successful, and don't change the grid after seeing results.
