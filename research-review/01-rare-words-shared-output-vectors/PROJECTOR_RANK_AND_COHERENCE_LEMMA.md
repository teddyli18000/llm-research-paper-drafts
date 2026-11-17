# Paper 01 — Why a normalized class projector may lose power as vocabulary grows

**Exact linear algebra for an untied flat-softmax head; statistical expectations only under stated isotropic null model.** This is mathematical analysis of the **new** normalized projection variant. It is not a claim that the original split-centroid method is our invention.

## Setup

For vocabulary V, K nonempty word groups, indicator matrix `A∈{0,1}^{V×K}`, group sizes `D=AᵀA=diag(n_1,...,n_K)`, define the orthogonal group-mean projector

`P=A D^{-1} Aᵀ`.

The same operator is applied independently to each output embedding dimension. Proposed gradient modifier:

`G̃=(I+αP)G`, where `G∈R^{V×d}` is the ordinary full-softmax output-head gradient.

## Exact identities

1. `P²=P=Pᵀ`; `rank(P)=K` with nonempty groups.
2. `P1=1`. For an untied full-softmax head, `1ᵀG=0` for every sample and minibatch. Thus `1ᵀPG=0` and `1ᵀG̃=0`: the **global row-mean gradient remains zero**.
3. The spectral values of `I+αP` are `1` and `1+α` for α≥0; they do not grow with cluster population. For ordinary SGD, `⟨G,G̃⟩_F = ||G||_F² + α||PG||_F²`, so the direction is a strict first-order descent direction when `G≠0`.
4. Define `H={z∈R^V:1ᵀz=0}`, dim(V−1). `P` restricts to an orthogonal projector on `H` of **rank(K−1)**, since its all-ones direction is orthogonal to `H`.
5. Let `g` be uniform on the unit sphere in `H`, independent of a fixed partition. Then
   `E[||Pg||² / ||g||²] = trace(P|_H)/(V−1) = (K−1)/(V−1)`.
   Similarly, for independent isotropic columns of `G`, `E||PG||_F²/E||G||_F²=(K−1)/(V−1)`.

For K64 the isotropic null energy fraction is:

| Vocabulary effective V | (K−1)/(V−1) |
|---|---:|
| 6,000 | **1.050%** |
| 16,000 | **0.394%** |
| 50,257 | **0.125%** |

The preceding expectation **does not apply** if KMeans groups are learned from the current output rows and correlated with future gradients; precisely such alignment is the hypothesis. A diagnostic `c_t=||P_tG_t||_F²/||G_t||_F²` across training steps, compared against size-matched shuffled groups, quantifies how much the learned partition amplifies coherent gradient directions beyond the isotropic null.

## More general grouped-gradient eigenvalues

Let `P_p=A D^{-p}Aᵀ`. Its nonzero eigenvalue on cluster k is `n_k^{1-p}`:

- `p=0`: **original split-centroid SGD geometry** `AAᵀ`, eigenvalue n_k, strongly group-size-dependent.
- `p=1`: **our normalized projector** `A D^{-1} Aᵀ`, eigenvalue 1, bounded.
- `p=1/2`: intermediate eigenvalue `sqrt(n_k)`, potentially a scale-sensitive tradeoff.

No claim of novelty for these spectral identities; they are consequences of group indicator algebra. The empirically important issue is whether **learned group/gradient alignment** survives growth in V and width under a modern adaptive optimizer.

## Implications and required experiment

The α2 V6k pilot nearly matches original centroids; however, applying α2 unchanged at V50k can lose effect because random group-mean energy is `~8.4×` smaller in expectation. The already-preregistered [α2 V16k/V50k confirmation](PROJECTOR_SCALE_CONFIRMATION_PROTOCOL.md) directly tests this falsification. Do not retune α in that registered test.

If α2 fails at V50k, the mechanism can be investigated with **measured gradient-coherence fractions** and a separately preregistered `p∈{0.5,1}` scaling study. Do not introduce a posthoc p after seeing test-shard results. If it succeeds, learned partitions must align with output gradients enough to overcome the isotropic rank bottleneck; verify this rather than attributing it to the projector solely from loss numbers.
