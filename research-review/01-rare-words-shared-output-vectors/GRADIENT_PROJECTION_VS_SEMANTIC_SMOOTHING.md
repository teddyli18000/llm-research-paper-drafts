# Paper 01 — Exact projected-softmax-gradient decomposition and label-smoothing collision

**Theory derived for the new direct flat-head projector** `G'=(I+αP)G`, fixed assignment A during one optimizer update. Original data source: [upstream 01/PAPER.md](../../01-rare-words-shared-output-vectors/PAPER.md). Closest directly relevant published learning-signal smoothing prior: [Diehl Martinez et al., EMNLP 2024, *Mitigating Frequency Bias and Anisotropy in Language Model Pre-Training with Syntactic Smoothing*](https://aclanthology.org/2024.emnlp-main.344/). That work redistributes loss/learning signal to syntactically related words; it is **not** our invention and must be a strong comparator.

## Exact identity for a single token prediction

Let the untied output matrix have rows `R_i`, hidden vector `h`, logits `z_i=R_i·h`, model distribution `p_i=softmax(z)_i`, hard target `y_i=1[i=t]`. For a disjoint vocabulary partition, `n_g` is cluster size and `P=A(AᵀA)⁻¹Aᵀ` averages coordinates within each group. The standard per-sample head gradient is

`G_i=(p_i-y_i)hᵀ.`

Define group probability `p_g=Σ_{j∈g}p_j` and group target `y_g=1[t∈g]`. Projected gradient has **exact rowwise form**

`G'_i=[p_i-y_i + α (p_{g(i)}-y_{g(i)})/n_{g(i)}]hᵀ.`

Thus it redistributes **both** class probability residuals `p_g` **and** class target residuals `y_g`. This is not ordinary label smoothing.

Rearrangement under `ε=α/(1+α)`:

`G'_i=(1+α)[p'_i-y'_i]hᵀ,`

with `p'_i=(1-ε)p_i+ε p_{g(i)}/n_{g(i)}` and `y'_i=(1-ε)y_i+ε y_{g(i)}/n_{g(i)}`. Both p′ and y′ are normalized over the vocabulary. This is algebraic equivalence of the **instantaneous first-order update vector**, not equivalence of a globally integrable scalar loss for an arbitrary row-dependent P.

### Why standard within-group label smoothing differs

For fixed group semantic label smoothing with target `y^ε=(1-ε)y+ε P y`, the gradient of cross-entropy w.r.t. output head is

`G^{LS}=[p-(1-ε)y-εPy]hᵀ=G+ε(y-Py)hᵀ.`

There is **no** `Pp` term, and it changes the target representation even if `p=y`. With P fixed during the step, our proposed update instead is `G+αP(p-y)hᵀ`; its group correction goes to zero when the model's **group probability** matches the target group, even though token-level loss may remain large.

**Critical novelty question:** is the improvement of `G+αP(p-y)` actually due to this *prediction-minus-target group residual*, or can a conventional within-cluster soft target achieve equal/better performance? A good paper must compare both, not simply cite label smoothing.

### Additional connection to original shared centroids

If a class offset `C_g` is added to all token logits for i∈g as in the upstream `R=U+AC` model, then the class-offset gradient is exactly `∂L/∂C_g=Σ_{i∈g}(p_i-y_i)hᵀ=(p_g-y_g)hᵀ`. Under simple SGD with row-specific step ηU and class-offset step ηC, this induces `ΔR=−(η_U I+η_C AAᵀ)G`. Our direct normalized `P=AD⁻¹Aᵀ` is **a bounded class-size-normalized version of the same geometric principle**. Therefore the true novelty is limited to **normalization, direct one-matrix preconditioning, and interactions with advanced optimizer / semantic groups**, not discovering group residuals or class-gradient sharing.

## Strong experiment preregistration

See [SEMANTIC_SMOOTHING_COLLISION_PROTOCOL.md](SEMANTIC_SMOOTHING_COLLISION_PROTOCOL.md): same corpus / same token groups, target-only within-cluster label smoothing **on head only** (isolating optimizer mechanism) and **full CE smoothing** (a stronger true loss-function baseline) against standard Ember, original author centroids, and projected Ember. This is a method-level collision test, not an exact reproduction of the 2024 authors' syntactic model or their POS-class construction.

The published 2024 paper is [EMNLP 2024 DOI 10.18653/v1/2024.emnlp-main.344](https://doi.org/10.18653/v1/2024.emnlp-main.344). It must be credited even if our class-residual optimization has different equations.
