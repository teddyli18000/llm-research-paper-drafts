# Strongest mathematical prior-art collision: Tree Loss (AISTATS 2022)

**Verified full PDF** [Wang & Izbicki, *The Tree Loss: Improving Generalization with Many Classes*, AISTATS 2022](https://proceedings.mlr.press/v151/wang22d.html), especially pp.2–3, Eq.(2)–(4), §4.3–4.4. The paper parameterizes a dense classification head as a sum of per-class and hierarchical ancestor vectors, uses the **same CE objective**, and transfers gradient signal among related labels **through shared parameters**. It reports an approximately 2× training slowdown in its particular implementation. This is substantially more relevant than generic token-embedding or label-smoothing references.

## Direct algebraic comparison

Take our vocabulary class partition A (0/1 row assignments, K nonempty clusters), group count D=AᵀA and symmetric projector `P=AD⁻¹Aᵀ`, `P²=P`. Our proposed direct flat-head preconditioned update under **SGD** is

`R_next = R − η(I+αP)G,` where `G=∇_R CE(R)`.

Define an invertible, **same-size** linear reparameterization `R=B U`, where

`B=I+(sqrt(1+α)−1)P`.

Since P is idempotent,

`B²=I+(2c+c²)P=I+αP`, with `c=sqrt(1+α)−1`.

The reparameterized CE `CE(BU)` has gradient `∇_U CE = B G`, so a standard SGD step on U induces

`ΔR=B ΔU=−η B² G=−η(I+αP)G`.

**Therefore our fixed-P normalized class projector update is exactly equivalent under plain SGD to ordinary CE training in a linear shared-class parameterization.** Its new-looking mathematical operator alone is not a fundamentally new learning principle. Unlike the original upstream split-head `R=U+AC`, the `B` parameterization keeps one V×d trainable matrix and embeds normalized group means in its forward representation; but a Tree Loss-like reparameterization is conceptually strong antecedent.

The original author's split-centroid reparam yields `ΔR=−η(I+AAᵀ)G` if SGD rates match. Tree Loss replaces A with hierarchy path indicators H; its corresponding SGD-effective head update is of the form `−η HHᵀG`. Thus all three share the **class-coordinate Gram preconditioning mechanism** at first order, differing in class normalization, additional free per-token basis, adaptive grouping and optimizer handling.

## What remains distinctive

1. Our direct method **does not change the forward softmax head representation** and can apply class-gradient mixing immediately before a modern non-SGD optimizer (Ember); it need not store a separate shared-centroid table or transport an optimizer's centroid states at regrouping.
2. **With Ember/Adam, the reparameterization equivalence above no longer generally holds** because coordinatewise/factored adaptive updates are not invariant to a mixing transformation B. The experimental difference is specifically an **optimizer-coordinate interaction**, not invented semantics of class sharing.
3. `I+αP` has bounded eigenvalues1 and1+α independent of cluster size, while `I+AAᵀ` can have eigenvalues proportional to large cluster populations. This is a useful normalization/conditioning distinction, though a straightforward algebraic design choice.
4. Learned/output-vector group identity versus train-only fixed semantic groups still matters; but prior Tree Loss and the author's original method already establish **related-label parameter sharing** as an established method.

## Revised novelty statement

A standalone methods paper based only on the projector equation and 6.5M-parameter LM gains is **not a major new optimizer theory**. There is a real candidate extension: **a class-normalized, forward-unchanged, direct output-gradient preconditioner for modern head optimizers** with measured real gradient alignment, robust untied V16k gain and an independent V50k marginal gain over standalone Ember. However the V50k original centroid baselines essentially tie the new method, and its naive dynamic form fails at batch128. A genuine high-impact contribution now requires stronger accuracy/efficiency advantage under realistic LM pretraining, or a new reliability-aware operator not reducible to existing class reparameterization and its group choices.

## Test

Verify `B²=I+αP` and induced SGD step equality for unequal cluster sizes and zero-sum output gradients. The companion deterministic unit test in the independent research folder is a mathematical identity test, not training evidence.
