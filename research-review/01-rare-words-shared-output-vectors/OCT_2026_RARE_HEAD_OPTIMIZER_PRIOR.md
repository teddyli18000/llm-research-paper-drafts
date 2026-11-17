# Additional late-2026 head-optimizer competitors

## MuonIO (October 2026): strong large-scale head-specific optimizer

[Ma et al., MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads](https://arxiv.org/abs/2610.02705), arXiv 2026-10-02.

It adapts Muon-style normalized descent specifically to input embeddings and output softmax classifiers, motivated by the respective 1→2 and 2→∞ operator geometries. It uses **column normalization for the input table and row normalization for the head** (depending on layout). It claims 50% optimizer-state saving and roughly 46% I/O-update FLOP reduction vs a Muon setup on 1B LLaMA/C4 while improving validation perplexity.

**Not a direct cluster-gradient collision:** output rows are normalized individually, rather than pooled into semantic/KMeans class means. **But it raises the modern optimizer bar:** 1B-scale experiments with official MuonIO and standard AdamW/Muon should eventually be compared, not only Ember in a 6.5M LM.

## Rare-token underestimation by adaptive optimizers (September 2026)

[Kar, Why Adaptive Optimizers Underestimate Rare Tokens](https://arxiv.org/abs/2609.37535), submitted Sep29 2026. It identifies directional asymmetry between rare-token target-arrival and non-target softmax gradients. For certain RMSProp/unigram/periodic-arrival settings, the optimizer fixed point systematically underestimates rare-token probability. SGD and AMSGrad preserve an unbiased fixed point under the paper's assumptions. This is **prior theory** for optimizer-dependent rare-token behavior; it is not our new mechanism. Our empirically observed gain concentrated in 10–99 target exposures is not proof of the same analytic bias.

## Unseen output-row drift in continued pretraining (October 2026)

[Han et al., A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks](https://arxiv.org/abs/2610.09835), submitted Oct7 2026. Its different objective is continued-pretraining/fine-tuning without access to previous data; unseen output rows undergo Adam-driven drift. It tests head-specific Adam epsilon interventions up to 12B parameters. This is a relevant *negative-transfer* and group-exposure baseline, not direct class-projector anticipation.

## Consequence

The independent normalized-gradient projector uses a bounded class-mean operator before a head optimizer, but its rare-token benefit and output-head optimization problem space are **crowded** by methods with stronger scale and optimization analysis. A novel paper should not imply that head-aware gradient control or rare-token correction was newly discovered here. Specific novelty must come from measured class-subspace directional transfer **beyond the original author's trainable-centroid sharing**, Tree Loss reparameterization and strong modern MuonIO/Ember baselines.
