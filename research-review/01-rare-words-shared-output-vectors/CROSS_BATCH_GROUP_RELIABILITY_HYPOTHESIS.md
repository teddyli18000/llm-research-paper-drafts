# New method hypothesis (not yet experimentally established): cross-batch reliability-gated class-gradient projection

**Status: mathematical proposal only.** Motivated by measured data:
- Fixed semantic grouping beats random/dynamic at early V6k and at batch128, where repeated online KMeans has too few steps to build reliable group structure.
- Dynamic KMeans beats fixed context at V16k/width128 and nearly matches context-initialized late adaptation.
- Group-subspace projected gradient energy `||PG||²/||G||²` is high for dynamic KMeans midrun but does **not** by itself predict final validation CE: static contextual groups sometimes win despite lower late-step energy.

## Proposal

For every candidate group partition `A`, define `P_A=A(AᵀA)⁻¹Aᵀ`. At occasional *training-only* probe steps, use two disjoint microbatches B1 and B2 drawn from the same training prefix, and obtain their ordinary pre-optimizer output-head gradients `G1` and `G2` **without applying group projection**.

Let `C_A=⟨P_A G1, P_A G2⟩_F` (projected **cross-batch agreement**) and `E_A = 0.5(||P_A G1||²_F+||P_A G2||²_F)`.

A possible bounded reliability measure is `ρ_A=C_A/(E_A+ε)`, clipped to [0,1], for a partition A. Compare: (1) precomputed train-context groups, (2) current KMeans groups, and (3) same-size shuffled control. Choose the group structure or a mixing coefficient using **positive cross-batch agreement**, not the gradient energy of one minibatch alone. Then apply `G'=(I+α ρ_A P_A)G` (or select between P_context/P_dynamic) before Ember.

## Why cross-batch agreement may matter

Write `G_b=μ+ξ_b`, with Eξ=0 and independent noise in two microbatches. With a fixed projector P independent of the two probe batches,

`E ⟨P G1,P G2⟩=||P μ||²`, while
`E ||P G1||²=||P μ||²+E||P ξ1||²`.

Thus the cross-batch inner product suppresses *independent gradient noise* and estimates transferable group-level signal. This is an exact identity **under the fixed-P, independent, unbiased-gradient assumptions**. It is not a proof that picking the highest ρ improves CE with Ember. Applying KMeans to the same two probe batches would violate that conditional independence; calculate the grouping from earlier parameter rows or precomputed training contexts.

The static/dynamic crossover and batch128 rescue suggest that **a group’s reliability, not merely a token’s frequency or projected-gradient norm, should govern how strongly the optimizer trusts it**.

## Prior art to cite before any publication claim

- [Gradient Agreement Filtering (GAF), arXiv 2412.18052](https://arxiv.org/abs/2412.18052) explicitly uses agreement between microbatch gradients for robust optimization, and is key mechanism prior art.
- [Gradient Agreement as an Optimization Objective for Meta-Learning, arXiv 1810.08178](https://arxiv.org/abs/1810.08178) uses gradient agreement to weight tasks. Thus **cross-batch gradient agreement is NOT a new invention**.
- [SGG ACL 2025](https://aclanthology.org/2025.acl-long.164/) dynamically groups scalar momentum/learning-rate statistics; [Leviathan 2026](https://arxiv.org/abs/2601.22040) improves rare input token representation, but neither uses this exact output-row projected-gradient agreement operator.
- The original public Paper01 already supplied KMeans regrouping for output token groups and basic pooled gradient transfer. Any extension must attribute it.

## Experiment roadmap (not launched)

1. On fixed already-approved data and fresh model seeds, replay V6k/width32 at batch32 and batch128, plus V16k/width128. Use precommitted α2 and K64; train-only context groups from same corpus, no validation access to choose ρ. Compare fixed contextual projector, dynamic KMeans projector, and a *precommitted* ρ-controlled group selection rule. Include shuffled-size-matched controls and frequency bins.
2. Compute and archive actual `C_A`, `E_A` and candidate selection frequency at each fixed probe step; predict class gradient coherence before switching.
3. Account for extra backward passes (or microbatch gradient buffering) and storage. If the reliability estimator costs more than its performance gains at matched wall-clock, this is not a good method.
4. Validate on a **new independent test shard** and new seeds. Do not decide ρ thresholds or candidate schedules from the external heldout dataset.

This is potentially a deeper algorithm than just picking static vs dynamic clusters by batch size, but it requires a rigorous study and a collision check for projected-gradient agreement in other optimizers before implementation.
