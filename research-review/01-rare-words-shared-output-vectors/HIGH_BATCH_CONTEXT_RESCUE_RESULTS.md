# Paper 01 — Context-first normalized projector **rescues** the batch128 failure

**[Actions #44018942584](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44018942584), success 3/3 jobs, all 21 preregistered trainings completed.** Independent model seeds 281,283,293, effective V6000/width32, 100 updates of batch128 × sequence64 (819,200 train tokens), official Ember head LR.05/β₂=.95, projector α2 and K64. Data sources used for grouping are from the **same training prefix only**; this is not an external-teacher experiment.

## Raw validation CE

| Arm | Seed281 | Seed283 | Seed293 | Mean CE | Gain vs plain Ember |
|---|---:|---:|---:|---:|---:|
| **Fixed train-only contextual projector** | **5.0654** | **5.0710** | **5.0642** | **5.06687** | **+0.01973** |
| Context groups→online after step20, refresh10 | 5.0663 | 5.0719 | 5.0669 | **5.06837** | +0.01823 |
| Original split-centroid + Ember (online50) | 5.0782 | 5.0862 | 5.0777 | **5.08070** | +0.00590 |
| Dynamic projector refresh10 | 5.0826 | 5.0844 | 5.0873 | 5.08477 | +0.00183 |
| KMeans at step0 then refresh10 | 5.0777 | 5.0890 | 5.0876 | 5.08477 | +0.00183 |
| Plain head-only Ember | 5.0828 | 5.0886 | 5.0884 | **5.08660** | — |
| Original dynamic projector refresh50 | 5.0888 | 5.0896 | 5.0917 | **5.09003** | −0.00343 |

All three model seeds favor **fixed train-only contextual projector** over both **plain Ember** (+0.0174,+0.0176,+0.0242) and **the original author's centroid+Ember** (+0.0128,+0.0152,+0.0135, mean +0.01383). The dynamic α2 projector *without* context information fails at this batch size in this independent experiment too.

## Causal mechanism

- Reducing refresh interval from every50 to every10 steps only improves mean CE from 5.09003 to 5.08477, essentially plain Ember; **more frequent KMeans does not rescue** the method.
- Computing KMeans at step0 on initial random vectors (and refreshing every10) does not rescue it either (5.08477), which is consistent with needing *semantic relevance*, not merely an early group assignment.
- Fixed contextual groups from the full same-training-data prefix yield a sizeable robust gain without online regrouping. Starting with these groups and then switching online after step20 **hurts slightly** (mean +0.00150 CE), showing no evidence that frequent later regrouping is needed at this short horizon.
- The result is compatible with the previous measured gradient coherence: contextual partition captures ~3.4% pre-Ember gradient energy at step0, whereas random partitions capture ~1.0%, and online KMeans learns coherent groups only later. Coherence is a mechanism clue, not a complete predictor of NLL.

## Scientific and paper implication

The original normalized **dynamic** group-projector algorithm was **not batch-robust**, so its across-batch claim must remain negative. But the **context-informed single-matrix gradient projector is batch128-robust in a controlled V6k same-token test**. This shifts the most promising independent scientific proposal from “periodically re-cluster learned output rows” (largely anticipated by the upstream author) toward **train-context-guided class-subspace preconditioning of a flat LM head** with a modern token-table optimizer.

The fixed grouping computes TF-IDF/SVD/KMeans from the *entire train-token prefix before training*, i.e. an extra training-data analysis pass. Count preprocessing cost and compare fairly with a strong static-centroid baseline; this result alone does not show net wall-clock savings or 100M+ parameter LLM scaling.

Next essential confirmation: [static vs dynamic projector at V16k/50k](CONTEXTUAL_PROJECTOR_TRANSFER_PROTOCOL.md) with independent unused model seeds, plus the [fresh-shard external test](LOCKED_FINAL_HELDOUT_5ARM_PROTOCOL.md). If fixed-context wins at V16k/50k, pre-register a separate untouched-shard **contextual projector** generalization comparison; the ongoing external test only has the *dynamic* projector.
