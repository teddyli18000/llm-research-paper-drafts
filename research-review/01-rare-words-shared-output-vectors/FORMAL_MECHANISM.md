# 01 — Mathematical mechanism (derived from actual implementation)

Date: 2026-11-16. Distinct from earlier qualitative rare-word story. This analysis is a derivation; identities were tested numerically on a synthetic output layer in `research-lab/01-rare-words-shared-output-vectors/theory_check.py` ([pilot CI #43833736739](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43833736739)).

## 1. Exact algebra

Let `U ∈ R^(V×d)` be the private output rows, `A ∈ {0,1}^(V×K)` a one-hot word-to-cluster membership matrix, and `C ∈ R^(K×d)` the shared cluster vectors. Define effective output rows `R=U+AC`.

For hidden `h`, logits are `z=Rh`. For target word `y`, cross-entropy `L=-log softmax(z)_y`, and the effective-row gradient is

`G_i = ∂L/∂R_i = (p_i - 1[i=y]) h`.

The chain rule gives **exactly**

`∂L/∂U=G,` and `∂L/∂C=AᵀG`.

Thus for group `g`:

`∂L/∂C_g = [ Σ_(i ∈ g) p_i - 1[y∈g] ] h = [P(g|h)-1[g(y)=g]]h`.

This is **precisely a group-classification gradient** from a flat softmax, despite not introducing a separate supervised group classifier. This is the central mechanism.

## 2. Exact hierarchical probability factorization

Let `Z_g(h)=Σ_(i∈g) exp(hᵀU_i)`. Then

`P(g|h)=exp(hᵀC_g) Z_g(h) / Σ_k exp(hᵀC_k)Z_k(h)`,

`P(i | g,h) = exp(hᵀU_i) / Z_g(h)` for `i∈g`, and

`P(i|h)=P(g(i)|h) P(i|g(i),h)` **exactly**.

Thus the method trains a *changeable implicit class hierarchy inside an ordinary full softmax*. It is related to hierarchical/class-based softmax and syntactic smoothing, but logits and full vocabulary are never approximated or replaced. The newly interesting bit is optimizer geometry when group hierarchy is learned from current output rows and changes without a jump.

## 3. SGD = positive-semidefinite gradient preconditioning

For fixed `A` and plain SGD on `U` and `C` (learning rates `η_U, η_C`):

`R^+ = R - (η_U I + η_C A Aᵀ) G`.

`A Aᵀ` is a blockwise all-ones kernel: every word i receives part of its group's aggregate gradient. For equal-sized clusters of n words, eigenvalue is `η_U+n η_C` in each group's shared direction and `η_U` for within-group contrast directions.

This establishes an important optimizer interpretation: **online clustering changes a data-dependent preconditioner**, not merely a semantic representation. K-means changes `A` while keeping `R` constant.

## 4. Exact function-preserving re-clustering

Given a new assignment matrix `A'`, choose any `C'` and set `U'=R-A'C'`. Then `U'+A'C'=R`. The actual code chooses group means as `C'`, which is just one gauge choice. Prediction/logits are identical **at that moment**. The optimizer update after it is not invariant, because the `U,C` parameterization and, with Adam, the optimizer moments change.

**Why this matters:** the original `PAPER.md` pseudocode resets Adam moments for `C`; the executable `grid.py` does not. The exact property being validated in the paper is function preservation, **not optimizer-trajectory preservation**.

## 5. What is not new and what could be new

Class-based output logits, morphologically shared vectors, and learned word clusters have substantial prior art: Botha & Blunsom 2014, LightRNN 2016, GroupReduce 2018, Syntactic Smoothing 2024. [SGG at ACL 2025](https://aclanthology.org/2025.acl-long.164/) also learns dynamic clusters of **gradient statistics** for group-specific learning-rate scaling, but does **not** introduce the same output-row tied residual and class-level gradient sharing.

The original online shared-centroid mechanism is already present in the upstream author's paper and is not our discovery. The normalized projector P=AD⁻¹Aᵀ is also not a new fixed-group SGD geometry: it is exactly induced by W=U+√α A D⁻¹/² C when U and C use the same SGD learning rate. A graph-diffusion proper score on disjoint class cliques also converges to P in probability coordinates, though its logit gradient retains the softmax Jacobian. The remaining research question is narrower: whether a direct flat-head implementation of this normalized coupling interacts with Ember/Adam differently from its exact shared-centroid parameterization, and whether that practical difference yields robust rare-token gains at realistic scale.

Two crucial competing explanations:
- Simple group-level optimization conditioning instead of semantic gradient sharing might generate much of the benefit. Compare matched artificial groups, one-shot groups, plus direct gradient-kernel updates.
- [Why Adaptive Optimizers Underestimate Rare Tokens (Kar 2026)](https://arxiv.org/abs/2609.37535) predicts rare-token Adam/RMSProp bias; the method may partly patch optimizer pathology. Compare `AMSGrad`, `Coupled Adam`, tuned Adam epsilon, and cluster decompositions with moment resets.

## 6. Current experimental verification

[Initial pilot run #43833736739](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43833736739): single seed, 200 steps, V=6000, reconstructed counts from first 100M-token public FineWeb10B GPT-2 shard; same training and validation for all arms.

| Arm | Validation CE (lower better) | Difference vs epsilon |
|---|---:|---:|
| tuned-epsilon arm | 5.2018 | 0 |
| dynamic / keep moments | 5.1866 | **0.0152** |
| dynamic / reset C moments | 5.1843 | **0.0175** |
| dynamic / reset both C and U moments | 5.1878 | **0.0140** |
| one re-cluster at step 50 | 5.1951 | 0.0067 |
| fixed random groups | 5.2006 | 0.0012 |
| fixed frequency groups | 5.2017 | 0.0001 |

A nontrivial preliminary finding: **resetting only C moments does not remove the gain** in this pilot; in fact it marginally improves loss on seed 0. Thus the original implementation's missing reset is not the obvious sole explanation. A three-seed 400-step confirmation is needed before interpreting this small difference.

This pilot used a reconstructed `counts.npy`, not the unavailable original corpus-wide counts, so numeric values are not expected to exactly match the author's PDF. Relative matched-arm differences are interpretable.

## 7. Remaining research work before an originality verdict

- Complete centered prediction/target residual controls; the first uncentered one-sided ablation contains a vocabulary-uniform mode and is not a clean causal decomposition.
- Complete the matched head-LR sweep between direct projection and exact normalized-centroid parameterization under Ember and Adam.
- Compare against the graph-diffusion proper-score limit, output-only Adam epsilon, projected RowNorm, and MuonIO.
- Preserve the selected batch32 schedule candidate: context groups through step250, then online K-means; batch128 remains fixed context at the tested 100-step horizon.
- Test on an untouched shard and a substantially larger model before any general LM-pretraining claim.


## November 17 update

The fixed-group SGD equivalence has now been verified with a 25-step nonlinear softmax test (max output-matrix error 2.22e-15). Direct projector vs exact normalized-centroid under adaptive optimizers shows a batch32 advantage (−0.0166 CE under Ember, −0.0248 under Adam) but only a near-tie at batch128. The head-LR sweep is preregistered. The first raw one-sided residual decomposition is confounded by the uniform softmax mode; centered controls are running. See WORKLOG.md and the corresponding result/protocol files for exact run IDs.
