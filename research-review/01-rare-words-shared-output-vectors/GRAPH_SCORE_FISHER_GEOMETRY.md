# Exact logit-Fisher natural-gradient connection for class residual projection

Date: November 17, 2026. This is an **algebraic identity**, not a claimed new optimizer or trained result. It qualifies the originality interpretation of [direct class projection](FORMAL_MECHANISM.md) and [graph-diffusion scoring rules](GRAPH_DIFFUSION_SCORE_PROTOCOL.md).

## Setup

Let `p=softmax(z)`, `y` be a one-hot target, `r=p−y`, and let `P=A(AᵀA)⁻¹Aᵀ` be the orthogonal projector onto the vocabulary's constant-within-class directions. Because `P1=1` and `1ᵀr=0`, we have `1ᵀPr=0`.

For the (nonstrictly proper) class-projected quadratic score

[
S_P(p,y)=(p-y)^\top P(p-y),
]

the probability- and logit-space gradients are

[
\nabla_p S_P = 2P r,\qquad \nabla_z S_P = 2J P r,
]

where `J=diag(p)−ppᵀ` is the softmax Jacobian and categorical Fisher information matrix in the logit parameterization.

For strictly positive softmax probabilities, `ker(J)=span{1}` and `range(J)=1⊥`. The Moore–Penrose inverse obeys `J⁺J=I−11ᵀ/V`. Therefore

[
J^+ \nabla_z S_P
=2J^+JPr
=2Pr.
]

**Consequently**, the direct proposed logit residual `r+αPr` equals the ordinary cross-entropy's **Euclidean logit gradient** `r` plus `α/2` times the **logit-Fisher natural gradient** of `S_P`. This is an exact per-example identity.

For the whole output-head gradient `G=r hᵀ`, the intervention `G+α PG` matches the outer product of that hybrid logit vector with `h`. However **it is not generally the natural gradient in the shared LM-head parameter space**: the weight matrix is reused across contexts, and its Fisher metric couples samples and hidden features. Nor does this identity yield a usual scalar loss whose Euclidean logit gradient is `r+αPr`; the asymmetric generic Jacobian of `Pr` still obstructs that.

## Numerical verification

A direct double-precision 11-token, four-unequal-group test with nonuniform logits and a one-hot target gave

`max_abs(J⁺ @ (2 J P r) − 2 P r) = 7.77e−16`.

This identity is independent of graph-diffusion training and does not depend on Ember/Adam.

## Relation to prior art and claims

- [Hilfiker, Cincera Brack & Sterchi (2026), *Strictly proper scoring rules for node selection tasks on graphs*, arXiv:2611.03274](https://arxiv.org/abs/2611.03274), equations 15/20, define graph-resistance and graph-diffusion quadratic proper scores in probability space. The disconnected-group-clique long-time diffusion limit provides `P` in the diffusion quadratic; it is proper but not strictly proper at the `P` limit.
- This relation identifies **graph-proper-scoring + logit Fisher preconditioning** as a mathematical explanation of the candidate normalized class residual, **not evidence that the 2026 paper itself applies it as an LM-head optimizer**.
- Therefore the projector algebra and the observation that it amplifies a proper-score class residual are **not enough on their own** for a foundational optimizer-theory novelty claim. What remains to test is practical parameter-space adaptive optimizer behavior and scaled generalization.
- The existing score-only `CE+S_P` comparison adds ordinary backpropagated `2JPr`, not the Fisher-preconditioned `2Pr`; with lambda=1 its effective norm can be much smaller. Thus failure of an untuned graph-score objective is not proof that no well-scaled scalar loss or second-order method can match the direct intervention.

Sources: [Tree Loss mathematical collision](TREE_LOSS_PRECONDITIONER_COLLISION.md), [normalized centroid adaptive results](NORMALIZED_CENTROID_ADAPTIVE_RESULTS.md), [2026 LM-head optimizer prior art](2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md).
