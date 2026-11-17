# Paper 01 — Equal-token batch-size test: normalized dynamic projector **fails at batch128**

**[GitHub Actions #44018218531](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018218531): success, 6/6 matrix jobs.** All 18 preregistered trainings completed (seeds269,271,277 × batch32/128 × official Ember plain/original centroids/our dynamic normalized projector). Public FineWeb, V6k width32, two layers, α2, same **819,200 training tokens** with batch32×400 updates versus batch128×100 updates. All three methods share the same Ember LR=.05, beta2=.95 and body optimizer; original group-centroid LR=.02.

## Paired validation CE (lower better)

| Batch / updates | Method | Seed269 | Seed271 | Seed277 | Mean |
|---|---|---:|---:|---:|---:|
| **32 / 400** | Ember plain | 4.9999 | 5.0095 | 5.0038 | 5.00440 |
| | Original online centroids + Ember | 4.9829 | 4.9831 | 4.9787 | 4.98157 |
| | **Our dynamic normalized projector + Ember** | **4.9710** | **4.9788** | **4.9750** | **4.97493** |
| **128 / 100** | Ember plain | **5.0793** | **5.0920** | **5.0965** | **5.08927** |
| | Original online centroids + Ember | 5.0694 | 5.0869 | 5.0892 | **5.08183** |
| | Our dynamic normalized projector + Ember | 5.0831 | 5.0988 | 5.1030 | **5.09497** |

- At batch32, projector gains **+0.02947 CE** over plain and **+0.00663 CE** over the author's original centroid method. All three seeds favor projector.
- At batch128, projector **loses −0.00570 CE** versus plain and **−0.01313 CE** versus the author's original centroid method. All three seeds disfavor projector.
- The original centroid method retains a small positive gain **+0.00743 CE** at batch128, confirming the original author's reported qualitative trend that larger batch weakens but does not destroy its sharing benefit.

## Mechanistic interpretation to test next

A normalized group-mean gradient is not automatically robust when the number of optimizer updates shrinks. Here batch128 performs only 100 steps, with KMeans refresh at step50 only once (vs seven refreshes at batch32). The initial partition is *random*, so the projector may waste its small projected-gradient energy on uninformative groupings during the early half of training. Other possibilities: fewer rare-token exposure opportunities per update, less optimizer stochasticity, unfavorable learning-rate warmup/decay, or interaction with Ember's factored moments.

The experiment changes global batch and number of optimizer steps **together** while fixing token budget, as intended. It cannot isolate the cause to KMeans cadence alone.

**Consequence:** do not claim that the α2 learned projector scales robustly to realistic large-batch pretraining based on its batch32 result. A stronger *fixed train-context projector* might address the early-random-group problem; that will require a separately registered experiment on new seeds and independent test evidence. The original author's centroid method remains the stronger comparator at batch128.
