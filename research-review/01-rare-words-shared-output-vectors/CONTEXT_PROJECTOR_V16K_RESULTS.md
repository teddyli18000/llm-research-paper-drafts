# Paper 01 — V16k contextual-vs-learned group projection: dynamic **wins**

**Three completed successful V16k jobs in [Actions #44018640211](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018640211)**. Independently preregistered model seeds 181,191,197, untied output V≈16k, width128, 2 layers, 400 steps, batch32,L64, official head Ember LR.05/β₂=.95, normalized projector α2. All five methods use identical model initialization and training prefix within each seed.

## Validation CE, three unseen seeds

| Arm | Seed181 | Seed191 | Seed197 | Mean CE |
|---|---:|---:|---:|---:|
| Official Ember plain | 5.8308 | 5.8248 | 5.8254 | 5.82700 |
| Original author's online centroids + Ember | 5.8077 | 5.8082 | 5.8011 | 5.80567 |
| **Our dynamic online KMeans projector** | 5.7575 | **5.7519** | **5.7525** | **5.75397** |
| Fixed train-only contextual projector | 5.7725 | 5.7684 | 5.7687 | 5.76987 |
| **Context-first, switch to online KMeans at step200** | **5.7559** | 5.7527 | 5.7534 | **5.75400** |

All three seeds favor **online dynamic projection** over fixed semantic context projection by +0.0150/+0.0165/+0.0162 CE (mean +0.01590). The context→late-online hybrid recovers nearly all of that difference, finishing essentially tied with dynamic (means 5.75400 vs 5.75397). The proposed projector beats original split-centroid+Ember by mean **0.05170 CE**.

**Important reversal vs V6k:** on V6k/width32/400step a strong static context projector was best (mean4.96843 vs learned dynamic4.97290, [causal ablation](PROJECTOR_CAUSAL_ABLATION_RESULTS.md)). On V16k/width128/400step the online geometry is markedly stronger. This changes vocabulary, model width, token-exposure distribution and group-members-per-vocabulary together; do **not** attribute the reversal solely to vocabulary.

**Implication:** neither “learned regrouping always wins” nor “static semantic groups always win” is supported. The newer mechanistic hypothesis is an **exposure- and feature-quality-dependent switch between contextual initialization and learned output-row group geometry**, with changing optimizer step budgets. The existing [batch128 study](HIGH_BATCH_CONTEXT_RESCUE_RESULTS.md) shows static context rescues the optimizer when online KMeans cannot learn reliable groups in 100 steps. A theoretically justified adaptive switch criterion could be a distinct method, but selecting a schedule from these same validation results does not constitute external validation.

The three registered V50k contextual-projection tests remain in progress, and the unseen FineWeb shard002 test is the more important generalization gate.
