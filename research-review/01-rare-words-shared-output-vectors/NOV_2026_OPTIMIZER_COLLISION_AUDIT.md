# Nov 2026 focused collision audit — class-projected output-head optimization

**Current literature review through 2026-11-16.** Scope: the **new single-output-matrix, moving normalized class projector** `G'=(I+αA(AᵀA)⁻¹Aᵀ)G` before a modern output-head optimizer, not the upstream author's already-public `R=U+AC` method.

## Closest checked competitors

| Source | Actual mechanism | Collision with the proposed method |
|---|---|---|
| [Original public Paper 01](../../01-rare-words-shared-output-vectors/PAPER.md), local fork | Exact-logit-preserving online KMeans token-group reassignment, trainable per-class centroid + per-token residual. Gradients sum into centroids; under SGD equivalent coupling `AAᵀG`. | **Strong.** Original shared-gradient mechanism is the author's, not ours. The no-extra-centroid, group-**mean** normalized projection before Ember is an independent engineering/method extension, still must win against the original. |
| [LightRNN, NeurIPS 2016](https://arxiv.org/abs/1610.09893) | Row/column shared vocabulary parameterization; learned reassignment. | Antecedent for changing shared groups, not normalized gradient projection and exact full-head equivalence. |
| [GroupReduce, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html) | Groupwise low-rank LM softmax compression. | Antecedent for output-row grouping, but goal is compression rather than optimizing a flat head with no inference structural change. |
| [Syntactic Smoothing, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.344/) | Target/label smoothing over syntactically related tokens. | Sharing target mass/loss signal; no learned projected row-mean-gradient preconditioner. |
| [LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/) | Clustering *pretrained* LM token embedding vectors for lexical retrieval features. | Direct antecedent to KMeans on output embeddings, different objective/training stage. |
| [SGG, ACL 2025](https://aclanthology.org/2025.acl-long.164/) | Groups flattened **scalar momentum entries** with MiniBatchKMeans; computes group-median deviations then scales each entry's existing optimizer LR (clamp 0.1–10). | Closest *optimizer-grouping* terminology collision, but **does not average/sum directional gradients across token vectors** or implement `PG`. |
| [SAGE, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.923/) | Embedding optimizer using Lion-style sign/momentum with O(d) adaptive per-feature damping, shared over vocabulary rows and bounded by 1. | Competing modern token-interface optimizer. **No token-row group assignment** and no cross-row directional projection. Published models reach 1.3B scale, highlighting our scaling evidence gap. |
| [SCALE, 2025](https://arxiv.org/search/?query=%22Memory-Efficient+LLM+Pretraining+via+Minimalist+Optimizer+Design%22&searchtype=all) | Matrix-column gradient normalization and last-layer momentum, models 60M–1B. | Competing memory-efficient optimizer; no group-gradient communication. |
| [Ember v4, Oct 2026](https://arxiv.org/html/2607.01455v4) | Row×column factored second-moment optimization of embedding/head, modern 50k-step pretraining results. | Hard optimizer baseline. Our tested contribution must be the extra structured directional sharing **on top** of Ember, not token-head optimization itself. |
| [Why Adaptive Optimizers Underestimate Rare Tokens, Sep 2026](https://arxiv.org/abs/2609.37535) | Theoretical account of rare-token probability bias under Adam/RMSProp/sign; unbiased SGD/AMSGrad special cases. | Important alternative explanation and baseline for rare output rows. Does not specify learned token-group mean-gradient projector. |

## The actually distinct mathematical object

`A` is the indicator matrix of current KMeans token-row memberships; `P=A(AᵀA)⁻¹Aᵀ` is the orthogonal projector onto the K-dimensional group-constant subspace. The direct head gradient is `G'=G+αPG`.

- By construction this **does not** require the upstream method's extra learned group-centroid parameter matrix or its optimizer moments.
- The spectrum of `I+αP` is bounded by `1+α`, independent of group cardinality; the upstream method's SGD-equivalent `I+ηAAᵀ` amplifies by cluster population. This is an algebraic formulation difference, not inherently a new mathematics theorem.
- For an untied flat-softmax head, `1ᵀG=0`, and the group-mean projection retains that invariant. The original `AAᵀ` need not retain it if group sizes differ.
- Under isotropic zero-mean gradients independent of grouping, expected projected-energy ratio is `(K−1)/(V−1)`, **1.05% at V6k and 0.125% at V≈50k** with K64; thus real-scale success must be driven by learned gradient/group coherence, not mere random grouping. [Derivation](PROJECTOR_RANK_AND_COHERENCE_LEMMA.md).

## What is established vs to be decided

Established from [V6k pilot](INDEPENDENT_SCALE_AND_PROJECTOR_RESULTS.md): α2 parameter-free projected Ember mean CE **4.97640** vs original split-centroid Ember **4.97723** vs standalone Ember **5.00227** over three seeds. This is a **near-tie** with the upstream method, not demonstrated superiority.

Decisive outstanding tests were locked before outcomes:
- [Group-geometry ablation](PROJECTOR_CAUSAL_ABLATION_PROTOCOL.md): learned KMeans vs same-size shuffled, random, frequency, context, frozen groups, and LR-only controls.
- [Independent V16k/V50k projected optimizer transfer](PROJECTOR_SCALE_CONFIRMATION_PROTOCOL.md) using fresh model seeds and matched original method.
- [Independent FineWeb-shard external test](LOCKED_FINAL_HELDOUT_5ARM_PROTOCOL.md), with no method tuning using that shard.
- [Equal-token batch scaling](PROJECTOR_BATCH_SCALING_PROTOCOL.md).

**Publication boundary:** if a parameter-free normalized projector reproducibly improves a strong token optimizer while outperforming matched static/shuffled controls, preserves the signal at realistic vocabulary and batch sizes, and has a meaningful compute/state tradeoff against upstream dynamic shared centroids, this can be a legitimate new *extension method* study (with source credit). Without those wins, the result may be a useful rigorous replication/mechanism analysis rather than a new top-conference algorithm. Large-model experiments remain necessary for a competitive broad LLM-pretraining claim.
