# Paper 01 — Fixed 5-arm out-of-shard evaluation (locked before running)

**Registration date 2026-11-16 (PT), before running new seeds 809/811/821 and reading the new shard**. Strengthens the earlier [heldout plan](INDEPENDENT_OUT_OF_SAMPLE_TEST_PLAN.md) by locking all method identities, rather than selecting on these models' test metrics.

## Held-out corpus

Train **only** on the original 1,638,400-token prefix of `fineweb_train_000001.bin`. Original validation `fineweb_val_000000.bin` has been used in method selection. Fresh independent test: `fineweb_train_000002.bin` from the same upstream FineWeb10B GPT-2 data release (public file verified to exist), **never previously used by this lab**. Its shard name contains "train" but it is reserved as our external test. Fix 600 uniformly spaced 64-token windows, exact same evaluation/UNK mapping derived from shard000001's count ranking; only report its results after final models finish. Do not train, cluster, tune or select checkpoints using shard000002.

## Architecture, reproducibility, and arms

All models V argument50258 (effective vocabulary 50257), width64/heads4/layers2, batch32×sequence64, accumulation4, 800 updates; same original transformer architecture and Muon/body settings, train-data prefix, optimizer warmup/decay and 600-window original val. New model seeds **809,811,821**, none used in tuning or earlier published results.

Five **fixed arms per seed**, always all reported:

1. `adam_dynamic`: upstream original online `R=U+AC` groups (K64, refresh50), Adam head LR.05 / shared LR.01 / head eps1e−5.
2. `ember_plain`: modern official head-only Ember LR.05, beta2=.999, eps1e−8, ordinary single output table.
3. `ember_centroid`: original online shared centroid method, Ember on private head (LR.05/beta2=.999), Adam on shared centroids (LR.01).
4. `ember_projector`: **new candidate** ordinary single output table, no learned centroid, group projector `G'=(I+2P_t)G` with `P_t=A_t(A_tᵀA_t)⁻¹A_tᵀ` and KMeans refresh50, then same official Ember LR.05/beta2=.999.
5. `adam_static_context`: original split-centroid Adam head with fixed 64-cluster train-only adjacent-token co-occurrence TF-IDF/SVD/KMeans groups, same LR.05/.01, no online adaptation.

Other optimizer rates are identical to those in the V50k predecessor runs; no LR tuning here. Count train-only contextual preprocessing time. Official Ember pinned GitHub version 76f30117b2d948874cd1c965c4f6813fa9de8caa; dense head matches 2026 v4 update (tested).

15 full independent model trainings; same fixed result per seed includes original-validation CE and external shard002 test CE, target-count bins on both, original full-corpus token frequency bins, elapsed training time and group reassignment diagnostics.

## Falsifiable decision

The predeclared primary question is whether `ember_projector` improves **external test** compared with both `ember_plain` and `ember_centroid`. For a new algorithm paper, it must also have a credible accuracy/complexity advantage over `adam_dynamic` and `adam_static_context`, with correct attribution to upstream. All per-seed differences (including negative differences) must be reported. No use of test shard002 for alpha/beta2/learning-rate selection.

This evaluates generalization **to a different pretokenized shard**, not to a new data distribution or a 100M-parameter fully-pretrained LLM. The latter requires a separate larger-scale gate.
