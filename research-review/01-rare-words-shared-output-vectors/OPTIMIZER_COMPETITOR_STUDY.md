# Independent Adam vs AMSGrad experiment on rare output rows

Date: 2026-11-16. [Actions run #43835794954 — success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43835794954). The same reconstructed public dataset, 400 steps, batch32, V6000, head-only epsilon 1e-5 and **identical LR hyperparameters** were used for all arms. Three paired model-initialization seeds. AMSGrad is enabled for all Adam parameter groups (not just the head).

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean validation CE |
|---|---:|---:|---:|---:|
| Adam baseline | 5.0139 | 5.0118 | 5.0091 | 5.01160 |
| Adam + online K-means sharing | **4.9937** | **4.9965** | **4.9922** | **4.99413** |
| AMSGrad baseline | 5.0267 | 5.0284 | 5.0226 | 5.02590 |
| AMSGrad + online K-means sharing | **5.0129** | **5.0183** | **5.0122** | **5.01447** |

Pairwise gain from dynamic cluster sharing:
- Adam: **0.0202 / 0.0153 / 0.0169**, mean **0.01747**.
- AMSGrad: **0.0138 / 0.0101 / 0.0104**, mean **0.01143**.

**Result:** dynamic sharing improves training loss under **both** adaptive optimizers for all seeds. Thus its small-model benefit is *not restricted to a peculiarity of Adam's non-maximized second-moment state*. However AMSGrad baseline itself is **worse** than tuned-epsilon Adam in this direct same-LR comparison. **AMSGrad LR and epsilon were not independently swept**, so this does not establish an optimizer winner or show dynamic sharing beats a fully tuned AMSGrad. For academic credibility a proper sweep of AMSGrad, Adam epsilon, and schedule must precede any claim beyond the identical-LR mechanistic comparison.

Closest theoretical source: [Why Adaptive Optimizers Underestimate Rare Tokens (Kar, Sep 2026)](https://arxiv.org/abs/2609.37535), which demonstrates an unbiased *unigram* fixed point for AMSGrad/SGD, and shifted fixed points for Adam/RMSProp. These tiny-Transformer 400-step results probe short-horizon optimization rather than the same asymptotic fixed point; neither paper contradicts the other.

The next focus should be the *mechanism of class-level gradient coupling*: dynamic output-row sharing is not simply a universal remedy for Adam's biased fixed points.
