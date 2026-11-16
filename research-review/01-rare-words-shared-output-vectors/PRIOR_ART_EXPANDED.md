# Paper 01 — deeper full-text novelty comparison (2014–2026)

Date: 2026-11-16. The nearest neighbors below were checked in the original papers, including equations for loss functions and optimizer rules. This is not merely abstract keyword similarity.

## Closest architectural works

| Paper | Exact overlapping mechanism | Important non-overlap |
|---|---|---|
| [Botha & Blunsom, ICML 2014](https://proceedings.mlr.press/v32/botha14.html) | Each word's output representation contains shared morphological constituents | Constituents are supplied by morphology, not online learned from current output vectors |
| [LightRNN, NeurIPS 2016](https://arxiv.org/abs/1610.09893) | Iteratively reassigns words to shared row/column parameters during training | Reassignment changes predictions; lacks a full per-word additive residual compensating exactly for group changes |
| [GroupReduce, NeurIPS 2018](https://arxiv.org/abs/1806.06950) | Iteratively refines word group assignments in the **language model softmax matrix**, sharing block-wise basis vectors | Applied in compression and approximation; not periodic cluster reallocation during gradient pretraining, not logit-preserving |
| [Smaller Text Classifiers with Discriminative Cluster Embeddings, NAACL 2018](https://aclanthology.org/N18-2116/) | Trains word cluster assignments jointly with embeddings; cluster-adjustment variant adds a word-specific **scalar** | Word cluster vector plus concatenated scalar, rather than output-row **full-dimensional additive residual**; no function-preserving re-clustering |
| [Block-wise Word Embedding Compression Revisited, 2021](https://arxiv.org/abs/2108.03281) | Uses K-means for embedding block selection and low-rank compression | Clusters importance weights; explicitly avoids GroupReduce's iterative reallocation; no training-time unchanged softmax |
| [Syntactic Smoothing, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.344/) | Shares **target learning signal** among tokens with similar part-of-speech distributions during LM pretraining | Uses a **fixed soft target distribution** `t_i`; gradient `(p_i-t_i)h`, not trainable shared output vectors whose gradient is `(P(group)-1[group=y])h` |
| [LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/) | K-means on pretrained LLM output-head token embeddings | Clustered lexical representation for retrieval after pretraining, not language-model training with exact-logit-preserving reassignments |

Among these there is no direct equivalence to **training-time learned per-word group-gradient coupling via (R=U+AC), with repeated reassignment preserving (R)**. Its novelty is the precise algorithmic **combination** and the empirical coupling mechanism, not any one primitive.

## Closest optimizer and frequency-bias works

1. [Better Embeddings with Coupled Adam, Stollenwerk 2025](https://arxiv.org/abs/2502.08441): shares **second-moment denominators** across vocabulary rows, not gradient directions or head parameters. This is a materially different preconditioner from `AAᵀG` and is particularly close as a rare-token optimizer alternative.
2. [Rare Tokens Degenerate All Tokens, ACL 2022](https://aclanthology.org/2022.acl-long.3/): rare-token logit gradients and their negative contribution are a known problem; adaptive gradient gating of individual output embeddings addresses it without similarity groups.
3. [Why Adaptive Optimizers Underestimate Rare Tokens, Kar 2026](https://arxiv.org/abs/2609.37535): formalizes rare-token underprediction due to Adam/RMSProp normalization and contrasts SGD/AMSGrad. Online group sharing may change the effective update, but **not claim novelty for identifying Adam's rare-token bias**.
4. [A Deafening Silence, 2026](https://arxiv.org/abs/2610.09835): shows absent/rare target tokens' output embeddings drift coherently under Adam because nearly all their softmax gradients are negative-target absent, and finds stabilizing output-specific Adam epsilon. It does not cluster output rows nor share their gradients, but **reinforces the relevance of a tuned epsilon head baseline**.
5. [Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training, 2025](https://arxiv.org/abs/2508.15390): evaluates modern 85M–450M models and shows larger vocabularies change frequent/rare losses asymmetrically. This provides stronger **frequency-bucket evaluation context**; it does not introduce dynamic groups.
6. [ASGO: Adaptive Structured Gradient Optimization, 2025](https://arxiv.org/abs/2503.20762): matrix Gram-based preconditioning and fixed attention-head grouping. Its matrix operator differs from sparse block-group output-token preconditioning.
7. [Prototype Language Models / PRISM, 2026](https://arxiv.org/abs/2607.00510): decomposes **hidden states** into learned prototypes plus residuals and reconstructs logits exactly, but the decomposition occurs on position-specific hidden states `z_t`, not on vocabulary output rows, and does not periodically reassign word clusters.
8. [Softmax Reparameterization for Output-Head Quantization, 2026](https://arxiv.org/abs/2609.31291): exact softmax invariance to a common row shift for post-training quantization; no group-specific learnable centroids or online training.

## Mathematical novelty that survives these comparisons

Let `A` be a V×K one-hot word-to-cluster matrix, `C` group embeddings and `U` private token embeddings. `R=U+AC` yields an unmodified full-vocabulary softmax. Its group-vector gradient is

`∂L/∂C_g=(P(g|h)-1[g(y)=g])h`

and, under ordinary SGD, the effective full-head update is

`R^+ = R - [η_U I + η_C A Aᵀ] ∇_R L`.

The algorithm is therefore a learned, **time-dependent blockwise gradient-sharing operator**. The reassignment `A→A'` is accompanied by `U' = R - A'C'`, preserving all logits even if the new groups differ. In Adam, the moment and bias-correction state of `U,C` makes the optimizer non-invariant; the independent experiments show the importance of retaining private-row moments while resetting or remapping cluster moments.

**Collision assessment:** each component has strong precedent, but no closest checked paper provides the full dynamic function-preserving group-gradient coupling algorithm. This is a **specific, defendable algorithmic idea**, with empirical value now independently shown on a small LM. Its full significance still depends on strong semantic/static baselines and model-scale transfer; a 3-seed toy result alone does not establish a leading-conference paper.

## New lesson from independently run controls

Our experiments found that fixed pretrained semantic group assignments outperform from-scratch online K-means on V6000 small models. Thus the evidence does **not** support the original draft's broad dichotomy “fixed groups give no benefit; dynamic groups are necessary.” The correct claim is **content-informed groups** matter, and **online grouping** learns them when an external semantic teacher is unavailable. Detailed numbers: [CONTENT_CLUSTER_SPECIFICITY.md](CONTENT_CLUSTER_SPECIFICITY.md).

Also, preconditioning the direct per-token gradients with a dynamic `AAᵀG` term without extra shared parameters recovers almost the original loss improvement at coefficient 0.01 ([GRADIENT_KERNEL_AND_SEMANTIC_HYBRID.md](GRADIENT_KERNEL_AND_SEMANTIC_HYBRID.md)). This clarifies the contribution and suggests testing a simpler optimizer form.
