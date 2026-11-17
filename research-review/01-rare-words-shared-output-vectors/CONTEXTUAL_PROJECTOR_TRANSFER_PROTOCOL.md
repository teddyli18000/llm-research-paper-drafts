# Paper 01 — Do fixed train-context groups improve the new normalized projector at V16k/V50k?

**Pre-registered independent test on unused seeds 181,191,197.** Motivated by the completed V6k [causal ablation](PROJECTOR_CAUSAL_ABLATION_RESULTS.md), where a static train-only contextual group projector beat dynamically learned KMeans in all three matched seeds. This is a *new question* chosen after viewing V6k outcomes; do not use its scale results to retrospectively re-label the V6k study as confirmatory.

## Frozen settings

- Original untied Transformer, public same-shard FineWeb train prefix, fixed original val600, output vocab argument16000 or50258, 2 layers, 4 heads, K64, 50-step online refresh where enabled.
- V16k: width128, batch32×L64, 400 optimizer steps, head Ember LR.05/β₂=.95, original centroid LR.005, head Adam LR.03.
- V50k: width64, batch32×L64×acc4, 800 steps, head Ember LR.05/β₂=.999, original centroid LR.01, head Adam LR.05.
- All projector arms have fixed α=2 (selected in earlier V6k exploratory sweep); no further α/gradient norm tuning.
- Generate **fixed contextual groups solely from the LM's training prefix**: symmetrized adjacent-token cooccurrence, TF-IDF, 48-D TruncatedSVD and KMeans64. V16k uses exactly819,200 tokens, V50k 1,638,400 tokens. No pretrained external representation or validation/test accesses. Save preprocessing runtime.
- Three **new** model seeds **181,191,197**, paired over all methods and training data.

## Five locked arms / 30 full runs

1. `ember_plain`: official Ember single-head without groups.
2. `ember_centroid`: original upstream online trainable centroids + Ember on token-specific head and Adam on centroids.
3. `proj_dynamic`: **single** head with normalized KMeans group-mean-gradient projection every50 steps.
4. `proj_context_static`: same projector using once-computed **train-only context** groups, no online KMeans.
5. `proj_context_late`: same fixed context grouping until step200, then online KMeans at steps200/250/etc.

## Primary comparisons

- `CE(proj_dynamic) − CE(proj_context_static)` for each seed and scale. Positive means static context wins; negative means dynamic KMeans wins.
- `CE(proj_context_static) − CE(proj_context_late)`: is there measurable additional benefit to learning group identity over the second half of training?
- All against both official Ember plain and the original upstream split-centroid technique. Report all original loss bins and training/runtime timing.
- A static semantic prior built from all available train tokens is permitted as a training-only preprocessing baseline, but the extra SVD/KMeans preprocessing time must be included in compute accounting.
- V50k results are exploratory *selection* after V6k discovery; any new “best” method requires confirmation on untouched data/new seeds beyond these.

The methodology is a **different training-time preconditioner** from upstream `R=U+AC`; the previous author's regrouping algorithm itself is not ours. No inference-time architecture change in the projector arms.
