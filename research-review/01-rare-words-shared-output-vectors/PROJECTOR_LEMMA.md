# A new theoretical control: size-normalized group gradient projection

Date 2026-11-16. Work from the actual `R=U+AC` output head code, not a claim made by the draft author. Numerical identities verified locally using PyTorch on asymmetric group sizes [2,4,6,9]; test script `research-lab/01-rare-words-shared-output-vectors/projector_theory.py`.

## The original group-coupling operator may generate output-head mean drift

For an untied standard LM head trained with flat-softmax CE, `G=∇_R L` satisfies `1ᵀG=0` (summing over the vocabulary rows). The SGD equivalent for shared group-parameter training is

`R⁺=R-[η_U I + η_C A Aᵀ]G`.

If clusters have unequal sizes `n_g`, the operator `AAᵀ` **does not generally preserve** `1ᵀG=0`:

`1ᵀ AAᵀ G = nᵀ Aᵀ G`

which is nonzero in general. This is a vocabulary-wide common shift of the output weights. For an untied flat-softmax head such a common shift is a gauge direction: it has no effect on output probabilities for a given hidden state, but does affect parameter norms and optimization state. (Under weight tying, shared embedding shifts can also affect the model input, so the gauge interpretation is limited to the untied head.)

## A principled projector instead of an unnormalized group sum

Let `D=AᵀA=diag(n_g)`. Define

`P=A D⁻¹ Aᵀ`.

Then (for nonempty groups) `P²=P`, `P=Pᵀ`, `P1=1`; its eigenvalues are 0 or 1 and it projects token gradients onto the subspace constant within groups.

For `1ᵀG=0`, `1ᵀPG=(P1)ᵀG=0`. Thus the update

`ΔR=-η(I+αP)G`

preserves **exactly** the zero-sum gradient property and gives a bounded eigenvalue ratio `1+α` independently of group size. Compared with `I+α AAᵀ`, the spectrum no longer scales with the largest cluster `max n_g`.

The online version learns `A_t` by K-means on current LM-head output rows; no additional parameters at inference. This is a mathematically motivated optimizer variant, not the same as the original split-head Adam implementation, whose moments are nonlinear and parameterization dependent.

## Novelty and validation hypothesis

Centered softmax update geometry has a recent 2026 precedent: [Toward a First-Principles Update Geometry for the Language-Model Head](https://arxiv.org/abs/2608.22253) proves softmax-invariant update geometry and uses centered row normalization; its algorithm **does not** use learned word clusters or a token-cluster projector. The centering itself is not novel; the question is whether *dynamically learned semantic subspaces of token rows* improve sample efficiency while keeping the update compatible with that geometry.

Our [original direct AAᵀ gradient experiment](GRADIENT_KERNEL_AND_SEMANTIC_HYBRID.md) got ~0.015 CE improvement at α=0.01 on a small LM. The [normalized projector sweep](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852372143) is testing α=0.25,0.5,1,2,4 on three seeds. **The lemma is proven; usefulness of the new preconditioner must come from the experimental results.**
