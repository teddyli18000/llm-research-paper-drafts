# A class-gradient projector is generally not the gradient of any soft-target or hierarchical softmax loss

For a fixed token grouping let P be its symmetric orthogonal group-mean projector and let z be the vector of vocabulary logits, p(z)=softmax(z), y a one-hot hard target. The projected logit update is the vector field

`F(z)=(I+αP)(p(z)−y).`

This is the exact per-hidden-state preconditioned gradient of the original CE, and at output head rows G'=F hᵀ. Does there exist a scalar loss L_new(z) such that ∇_z L_new = F?

## Jacobian symmetry test

Softmax Jacobian is `J_p=diag(p)−ppᵀ`, symmetric positive semidefinite. The derivative of the projected vector field is

`J_F=(I+αP)J_p.`

Any twice continuously differentiable scalar potential has a **symmetric Hessian**, thus a necessary local condition for this vector field to be a gradient is

`(I+αP)J_p = J_p(I+αP)`,

or, for α != 0,

`[P,J_p]=PJ_p−J_pP=0.`

For an arbitrary grouping and nonuniform softmax probabilities, the two matrices **do not commute**. Therefore the output-head class-projection update **generally cannot be represented as the gradient of any fixed scalar loss in the ordinary logit coordinates**, including a conventional smoothed CE objective. This is a mathematical distinction, not necessarily an improvement in statistical performance.

A sufficient special condition is **p constant within each group**. Then Pp=p, and P commutes with diag(p), hence also with ppᵀ; J_F is symmetric and the local obstruction disappears. But trained LM token probabilities are generally not exactly uniform within arbitrary learned clusters, and this is a very special case.

## Why this does not contradict Tree Loss equivalence under SGD

As [Tree Loss reparameterization collision](TREE_LOSS_PRECONDITIONER_COLLISION.md) shows, with `R=B U`, `B=I+(sqrt(1+α)−1)P`, ordinary CE on U induces this same update on effective R under SGD. **In U coordinates it is standard conservative CE gradient descent.** In R coordinates, the non-Euclidean constant preconditioner changes the gradient geometry and the induced Euclidean vector field need not be conservative. This is expected: gradients depend on the metric/coordinates. Thus *being a nonintegrable field in R coordinates is not a proof of an entirely novel optimization principle*.

## Direct empirical falsifier

A deterministic small tensor test should construct unequal groups and nonuniform probabilities, calculate `PJ_p−J_pP`, and verify a nonzero skew-symmetric Jacobian component while the projected vector sum remains zero. A second test with probabilities identical within groups should give commutator zero. Distinguishes projected-group gradient from full smoothed CE with a genuine scalar objective.

## Publication consequence

The interesting experiment is not whether the formula looks like label smoothing; the relevant tests are whether *class-aware nonconservative preconditioning* outperforms:
1. True class-smoothed full CE,
2. Semantic group-probability CE,
3. Tree Loss-style forward reparameterization with the same initial logits,
4. Original author's shared-centroid KMeans method,
under modern Ember/Adam and meaningful model scale. The latter three are in ongoing independently registered Action matrices.
