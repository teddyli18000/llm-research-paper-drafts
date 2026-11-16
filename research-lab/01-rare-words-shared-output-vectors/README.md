# Paper 01 — independent research lab

Owner: independent follow-up on the original `01-rare-words-shared-output-vectors` draft.
The original paper and original code remain unchanged.

## Current research progress (2026-11-16)

See [WORKLOG.md](WORKLOG.md) for a complete run ID/evidence index and [research review](../../research-review/01-rare-words-shared-output-vectors/).

- Confirmed the original dynamic KMeans group-sharing effect on **three model seeds**, **four separate FineWeb train slices**, width32/V6000, and independently width128/V16000. The gain is real in these controlled settings.
- Learned group membership is key: random group refresh and function-preserving recenter alone have no benefit. Retaining a private-word optimizer history matters; resetting cluster Adam history itself does not negate the effect.
- A stronger **fixed semantic grouping** extracted from the training data beats the naive online KMeans method on the small runs. Late online adaptation of that semantic prior can add an improvement. With external GPT-2 pretrained token vectors, static semantic groups are stronger still.
- Derived an exact class-gradient coupling identity for `R=U+AC` and confirmed a direct no-extra-parameter dynamic gradient-kernel mechanism, plus a centered normalized group projector (see [PROJECTOR_LEMMA.md](../../research-review/01-rare-words-shared-output-vectors/PROJECTOR_LEMMA.md)).
- Current questions: long-horizon persistence, which target-frequency buckets benefit, sensitivity to different training slices, and whether bounded-geometry group-gradient mixing scales.
- Ongoing Actions results are recorded by run ID in WORKLOG; the original paper/code remain unchanged.

## Research hypothesis

For vocabulary output matrix `R=U+AC` with a one-hot word–cluster assignment matrix `A`, shared cluster vectors are a **learned, changing gradient-coupling operator**. For SGD with fixed assignment,

`R(t+1) = R(t) - [eta_U I + eta_C A A^T] grad_R L`.

This reveals an optimization/implicit-bias interpretation beyond “rare words borrow representations”. Periodic reclustering `A` changes the preconditioner while exactly preserving `R`. With Adam, moments associated with `U` and `C` matter; the original paper's pseudocode resets Adam on shared vectors, but the executable `grid.py` does not.

## Controlled experiments

- `theory_check.py`: asserts the exact SGD equivalence and zero-logit-jump regrouping.
- `prepare_data.py`: downloads the public GPT-2 FineWeb10B shard `kjj0/fineweb10B-gpt2` (train 000001 and val 000000), computes `counts.npy` from this one 100M-token train shard.
- `build_runner.py`: constructs an *instrumented copy* of upstream `grid.py` with `--seed` and `--moment` (`keep`, `reset_c`, `reset_both`). It asserts regroup logits remain equal and records reclustering churn. No upstream source edit.
- `run_matrix.py`: a matched-seed pilot comparing baseline, keep/reset variants, fixed group baselines and one-time vs repeated cluster updates.

## Data comparability

The public GPT-2 FineWeb10B shards use the same data format as the original source. **The original private `llm-per-word-weights/data/counts.npy` is not included in this fork.** We therefore recompute the frequency ranking from one public 100M-token training shard. This is a matched *replication study on a reconstructed vocabulary*, not a bitwise rerun of the author's full corpus-frequency selection. Cross-arm comparisons within this study are fair.

## Key prior art for continuing novelty review

- [Botha & Blunsom, 2014](https://proceedings.mlr.press/v32/botha14.html): morphologically shared output rows.
- [LightRNN, NeurIPS 2016](https://arxiv.org/abs/1610.09893): reassignment of shared word parameters.
- [GroupReduce, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html): iterative grouping of LM output rows for compression.
- [Syntactic Smoothing, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.344/): distribute learning signal to syntactically related tokens.
- [LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/): cluster pretrained LM head output vectors.
- [Why Adaptive Optimizers Underestimate Rare Tokens, Sep 2026](https://arxiv.org/abs/2609.37535): predicts rare output-row Adam bias; identifies SGD/AMSGrad differences. This is especially relevant to the original author's optimizer interpretation.

Next decision: finish 1600-step and target-frequency tests, then decide on full-vocabulary and stronger modern optimizer benchmarks.
