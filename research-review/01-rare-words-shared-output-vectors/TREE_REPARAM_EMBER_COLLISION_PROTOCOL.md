# Paper 01 — Exact Tree Loss-like reparameterization versus direct class-gradient projection under Ember

**Pre-registered before any outcomes.** AISTATS 2022 [Tree Loss](https://proceedings.mlr.press/v151/wang22d.html) uses related-class shared parameterizations of an ordinary CE classifier. Our new projector `G'=(I+αP)G` is **exactly equivalent under SGD with fixed P** to training a linear reparameterization `R=B U`, `B=I+(sqrt(1+α)−1)P`. This experiment tests if direct preconditioning before a modern adaptive optimizer has *actual* advantages beyond that old shared-class parameterization.

## Locked setting

Same original FineWeb V6000/width32/layers2/heads2/seq64/batch32/400 steps, original 600-window validation, official Ember **head-only** LR=.05/beta2=.95, class K64, α=2, three **new seeds 373,379,383**. Co-occurrence context groups from the entire fixed 819200-token train prefix, never validation. Model is always a single ordinary V×d trainable softmax head except the original centroid positive control.

Five arms per seed, total 15 trainings:

1. `ember_plain`: standard head Ember, unmodified.
2. `ember_direct_context`: apply `G'=(I+2P)G` to the single head **before** official Ember; fixed train-context groups.
3. `ember_reparam_context`: keep a single trainable U table, use `R=B U` during forward CE, where `B=I+(sqrt(3)−1)P`. Initialize U as `B⁻¹ R_init` with the *same effective output logits* as arm 1/2. Run official Ember on U gradient without any manual projected-gradient change.
4. `adam_direct_context`: arm2 with original Adam head LR=.05/eps1e−5.
5. `adam_reparam_context`: arm3 with original Adam head LR=.05/eps1e−5.

**No dynamic regrouping**: the fixed P ensures the precise SGD reparameterization equivalence applies and avoids optimizer moment remapping confounds. All arms share body/embedding optimizer, initialization and training prefix. Each method gets exactly 400 gradient updates; save per-frequency result, val CE, train seconds.

## Fixed interpretation

- If `ember_direct_context` and `ember_reparam_context` are indistinguishable, the new operator may be just a faster/simpler implementation of the same old class-coordinate idea, not a stronger optimizer method.
- If they differ, it demonstrates **optimizer coordinate dependence** (expected mathematically under nonlinear adaptive updates), but it is only a valuable extension if direct projector is consistently better than reparameterization on independent seeds and scales.
- Adam comparison determines whether the coordinate difference is unique to Ember or general adaptive head optimizers.
- Initial forward logits must be exactly matched to within 1e−5 for both methods before training; the head input/hidden body remains unchanged.
- Comparison to actual original author's centroid reparameterization remains reported in the other registered experiments; do not conflate Tree Loss-like normalized B with that original structure.
