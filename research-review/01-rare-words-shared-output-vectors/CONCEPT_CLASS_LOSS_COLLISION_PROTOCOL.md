# Paper 01 — Concept-class probability loss alternative

**Pre-registered before runs.** Two prior works must be compared: [Bai et al., ACL 2022, Better Language Model with Hypernym Class Prediction](https://aclanthology.org/2022.acl-long.96/) and [Iyer et al., EACL 2026, Beyond Tokens: Concept-Level Training Objectives for LLMs](https://aclanthology.org/2026.eacl-short.34/). Our *class-probability* auxiliary control is not a reproduction of their WordNet or semantic taxonomy construction; it isolates the alternative supervision mechanism on identical training-only groups.

Let probabilities be p_i and target class g(y). Define p_g=sum_(i in g) p_i. The class-probability loss is L_group= -log p_g. Its exact logit derivative is p_i - indicator[i in g(y)] p_i/p_g. This is distinct both from target-uniform class label smoothing and from the normalized projector's p_i-y_i+α(p_g-y_g)/n_g.

Original V6000 width32, two transformer layers/two attention heads, batch32/sequence64/400 updates, head-only official Ember LR .05/beta2 .95 and untouched original body/input optimizers. K64 groups from train-only co-occurrence context or online output-row KMeans at refresh50. Three unused model seeds **397,409,419**.

Seven locked arms per seed (21 full trainings): plain Ember; direct projected Ember α2 fixed-context; direct projected Ember α2 online KMeans; full-objective concept-group cross-entropy with static context classes and λ=.1,.3,.5; and full-objective concept-group cross-entropy with online KMeans groups λ=.3. Optimize (1-λ)L_token + λ L_group for concept variants, and evaluate the **ordinary token cross-entropy** for all conditions.

Predeclared primary comparisons: the strongest mean *static-context concept-loss* alternative versus static-context projector; and λ=.3 online concept loss versus dynamic projected Ember on the same three model seeds. All results and target-frequency bins archived; no hyperparameter changes after scores. Since λ is swept on the same original validation windows, selected winners here are exploratory and must not be described as independent heldout confirmation.

The original author's trainable centroid mechanism is prior art. A positive projector result would support that *its particular gradient transformation* beats a closely related known semantic class objective in this small-model setting, not a new invention of group-based word learning.
