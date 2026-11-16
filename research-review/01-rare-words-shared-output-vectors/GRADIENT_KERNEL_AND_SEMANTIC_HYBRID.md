# Paper 01: direct gradient coupling and pretrained-group adaptation

Date: 2026-11-16.

## 1. Is a learnable shared cluster-vector parameter essential?

[Direct gradient-kernel experiment, Actions #43839718101, success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43839718101).

We implement a plain **unshared** output head `R` trained by Adam, but modify its per-row gradients using clusters learned from the model's own output vectors, refreshed every 50 steps:

`G_i ← G_i + α Σ_(j ∈ g(i)) G_j.`

This is **exactly** the same family of block gradient coupling as the shared vector mechanism under ordinary SGD, where `α=η_C/η_U`; here it is applied **before Adam**, so it is not mathematically the same as original split-parameter Adam. We sweep α rather than claiming equivalence.

V6000, 400 steps, batch32, epsilon 1e-5, three identical model-init seeds, same reconstructed FineWeb data.

| α | Seed0 val | Seed1 val | Seed2 val | Mean gain against epsilon-only baseline |
|---:|---:|---:|---:|---:|
| 0 (unshared baseline) | 5.0139 | 5.0118 | 5.0091 | 0 |
| 0.01 | **5.0001** | **4.9988** | **4.9909** | **+0.0150** |
| 0.05 | 5.0053 | 5.0096 | 5.0055 | **+0.0048** |
| 0.20 | 5.0545 | 5.0524 | 5.0463 | **-0.043** |
| Original shared-vector head, no explicit gradient kernel | 4.9937 | 4.9967 | 4.9925 | **+0.0173** |

**Interpretation:** an independent single-matrix output head with a tuned **dynamic group-gradient preconditioner** recovers most of the original small-model improvement. The research question is therefore broader than “must store a shared centroid vector.” Much of the effect may be **group-aware gradient mixing**, with the residual+centroid parameterization one way to implement it. The role of Adam is central: the theoretically matched SGD coefficient (0.2) does not work when directly placed inside Adam's input gradient.

Do not treat α=0.01 as a fully tuned optimum; this is three-point preliminary screening. It nevertheless shows the main method is not simply additional parameters or projection re-centering. A dedicated explicit-gradient optimizer could lead to a more general methods contribution.

## 2. Do online updates improve a stronger pretrained static semantic prior?

[Pretrained semantic hybrid, Actions #43839941058, success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43839941058).

All three arms start with the same static groups from GPT-2's tied pretrained output-head vectors, KMeans K64 (strong external-pretraining baseline). We compare no regrouping, regrouping every 50 steps from step 50, and from step 200.

| Variant | Seed0 val | Seed1 val | Seed2 val | Mean val |
|---|---:|---:|---:|---:|
| Pretrained **static** semantic groups | **4.9848** | **4.9836** | **4.9750** | **4.9811** |
| Pretrained groups then online re-clustering beginning step 50 | 4.9910 | 4.9964 | 4.9852 | 4.9909 |
| Pretrained groups then online re-clustering beginning step 200 | **4.9841** | 4.9838 | 4.9763 | **4.9814** |

**Interpretation:** premature self-re-clustering degrades a *good, externally pretrained* group structure; waiting until step 200 roughly recovers the static level but does not improve it. This is not evidence that online grouping never helps: starting from random groups, online grouping helps substantially. It **does** refute the oversimplified claim that re-clustering continuously is always better. The useful policy depends on whether good semantic groups are available and how reliable the current output embedding geometry is.

A more promising general algorithm could adapt **only when the model-generated groups become more predictive than the prior**, using a held-out local criterion, stability or group discrimination. That would require a separate genuine discovery, not merely renaming the original approach.

## 3. Intellectual novelty implications

The shared-vector decomposition has a clean exact SGD interpretation:

`R^+=R-(η_U I+η_C A Aᵀ)∇_R L`.

It follows that the algorithm is a **time-varying, data-dependent block-gradient preconditioner** that can be implemented in parameter space or directly in gradient space.

The new 3-seed study supports this as the stronger paper story:
1. Random regrouping and repeated re-centering without new groups do not help (see [CONTENT_CLUSTER_SPECIFICITY.md](CONTENT_CLUSTER_SPECIFICITY.md)).
2. Learned content-based grouping matters (online KMeans / warm-start).
3. Direct gradient coupling helps even without new parameter vectors, strongly supporting the optimization mechanism.
4. Strong pretrained semantic groups are even better as a static baseline under this small-model task, showing that **group quality** is more important than always changing membership.
5. Overly aggressive pooling (α=0.2 before Adam) hurts; the coefficient and optimizer interaction determine viability.

## 4. Next steps

- Formalize the *Adam* case beyond the exact SGD identity, including effects of shared-cluster versus private per-word moment histories.
- Compare this direct gradient-kernel against same-data static distributional groups, not only external GPT2 pretrained groups (job in progress).
- Determine the gain regime with full vocabulary and realistic batch sizes; evaluate learned-only semantic grouping where external pretrained teachers are unavailable.
- Write paper contribution as **learned gradient sharing with stable output predictions**, not as first discovery of rare-token embedding parameter sharing.
