# Paper 01 — 2026 LM-head optimizer and rare-token prior-art audit

**Literature cutoff: 17 November 2026.** This note distinguishes full-text findings from the experiment results already produced in this fork. It is a novelty-boundary document, not a priority claim.

## Candidate update under review

For an untied vocabulary head \(W\in\mathbb R^{V\times d}\), one token's standard head gradient is

\[
G=(p-y)h^\top.
\]

For a vocabulary partition with indicator matrix \(A\in\{0,1\}^{V\times K}\), group-size matrix \(D=A^\top A\), and normalized class-mean projector

\[
P=AD^{-1}A^\top,
\]

the candidate flat-head intervention is

\[
G'=(I+\alpha P)G
=\left[(p-y)+\alpha(Pp-Py)\right]h^\top.
\]

The distinctive algebraic term is the prediction-side class residual \(Pp\), paired with the target-side class residual \(-Py\). The same projector can be built from train-only token-context groups or from K-means over output rows. The forward logits and parameter matrix remain unchanged.

## Closest prior art

| Work | What it establishes | Exact overlap | What it does not do |
|---|---|---|---|
| **The Tree Loss** (Wang & Izbicki, AISTATS 2022) | Class-structured parameterization and class-level gradient sharing | A shared-centroid parameterization \(R=U+AC\) induces class-coupled output-row updates under SGD | It does not use the normalized class-mean projector \(AD^{-1}A^\top\) as a flat-head preconditioner |
| **Syntactic Smoothing** (Diehl Martinez et al., EMNLP 2024) | Syntactically informed distribution of target learning signal | Uses a token-similarity operator to move target mass to related tokens | It changes the target distribution; its logit gradient has no prediction-side \(Pp\) term |
| **AGG** (Yu et al., ACL 2022) | Frequency-aware gating of harmful rare-output-embedding gradient components | Directly modifies output-embedding gradients for rare/very-rare tokens | Groups by recent token exposure and attenuates selected gradient components; it does not share semantic class-mean gradients |
| **Hypernym Class Prediction** (Bai et al., ACL 2022) | Class prediction can aggregate context for related rare words | Uses semantic classes and a token/class training curriculum | It changes the prediction objective/factorization rather than applying \(P(p-y)\) to the output-head gradient |
| **Gradient Grouping (SGG)** (Li et al., ACL 2025) | Groupwise adaptive learning-rate calibration from gradient statistics | Dynamic gradient groups and collective scaling | Groups parameter-gradient statistics inside layers; not vocabulary-row residuals or token-class projectors |

### The Tree Loss: closest structural collision

The Tree Loss is the most direct prior art for the original shared-output-vector idea. Its parameterization makes related class rows share a trainable component; under SGD, the induced update contains the unnormalized class-sum coupling \(AA^\top G\). For a fixed projector \(P\), the update \(G+\alpha PG\) can itself be represented by a linear change of coordinates under SGD. The fork's separate direct-versus-reparameterized experiment found near-tied results under both Ember and Adam across three seeds.

The distinction between \(AA^\top\) and \(AD^{-1}A^\top\) is real: the former has group-size-dependent eigenvalues, while the latter is an orthogonal projector with eigenvalues 0/1. That is a bounded-spectrum normalization distinction, not a wholly new concept of class sharing or gradient coupling.

### Syntactic Smoothing: target-side, not prediction-side

The full-text Syntactic Smoothing objective uses a target distribution of the form

\[
t=(1-\epsilon)y+\epsilon Py,
\]

so its logit gradient is

\[
p-t=(p-y)+\epsilon(y-Py).
\]

It redistributes target mass. It does **not** apply \(P\) to the prediction vector \(p\). The candidate update instead contains

\[
(p-y)+\alpha(Pp-Py).
\]

This difference is important: the candidate changes the prediction-side residual as well as the target-side residual. The paper's printed Equation (3) has an indexing ambiguity in its similarity normalization; a faithful reproduction should preserve and resolve that issue explicitly rather than assuming an implementation.

## Newer LM-head optimizer geometry

### Somasundaram et al. (2026): projected RowNorm

*Toward a First-Principles Update Geometry for the Language-Model Head*, arXiv:2608.22253.

The paper treats the LM head and softmax as one module and derives a functional update metric from Hilbert projective distance. Its update-size measure is the maximum pairwise token-row separation, \(D(S)=\max_{i<j}\|s_i-s_j\|_2\). Its tractable projected RowNorm update globally centers vocabulary rows,

\[
P_{\mathrm{global}}=I-\frac{\mathbf1\mathbf1^\top}{V},
\]

then row-normalizes and re-centers the direction. It reports 190M, 380M and 640M experiments, with large reductions in row-diameter and Hilbert-RMS perturbations and small validation-loss increases relative to its AdamW recipe.

**Collision boundary:** it is direct prior art for softmax-aware LM-head update geometry and projected head optimization. It does not partition vocabulary rows into token groups or average within-group gradients. Its projector removes only the single global shared-logit-shift direction.

The geometric distinction is exact. For a true softmax head gradient, each feature column sums to zero across vocabulary rows: \(\mathbf1^\top G=0\). Therefore the global-centering projector \(P_{global}=I-\mathbf1\mathbf1^\top/V\) leaves that gradient unchanged: \(P_{global}G=G\). By contrast, \(P_A=AD^{-1}A^\top\) replaces each row by its within-group mean and removes within-group zero-sum variation while preserving each group's total gradient. The two projectors act on different subspaces.

### Lau & Su (2026): symmetry-compatible LM-head updates

*Symmetry-Compatible Principle for Optimizer Design*, arXiv:2605.18106.

The paper derives optimizer rules from the symmetries of each parameter block. For LM heads it uses vocabulary-row permutation/right-orthogonal equivariance and the global shared-row-shift symmetry of softmax. Its centered row-norm, centered right-spectral and hybrid updates use the same global vocabulary-centering projector \(P_{\mathrm{global}}\), with a final projection after row-dependent nonlinear scaling to preserve zero row sum.

**Collision boundary:** strong prior art for global centering and row-aware LM-head optimization. It does not use semantic or K-means token groups, nor the class-mean projector \(AD^{-1}A^\top\).

### Ma et al. (2026): MuonIO

*MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads*, arXiv:2610.02705.

MuonIO uses rowwise momentum normalization for the LM head. Each token row gets its own normalized momentum direction; a shared scalar aggregates row norms. It does not average directions across token groups. Its C4 pretraining comparison includes 60M, 130M and 1B models, and Muon Tuned is the strongest standard-Muon comparison.

**Collision boundary:** a strong, modern LM-head optimizer baseline. It is not an algorithmic anticipation of class-mean gradient projection.

### Shah (2026): Token Geometry / Ember

*Token Geometry*, arXiv:2607.01455 (current v4; the optimizer introduced is Ember).

The paper frames Ember as a square-root empirical-Fisher preconditioner for token interfaces. It argues that rowwise gradient energy tracks token activation frequency and that the LM-head gradient admits the same factorized treatment as an embedding table. The official implementation stores factored row/column second moments for token tables and no first-moment state. Its statistics are computed from the same gradient used in the parameter update. The fork's experiments use the official pinned implementation.

**Collision boundary:** a strong baseline and the adaptive denominator with which the candidate projector interacts. A projected update numerator with raw-gradient factored statistics is a new variant of Ember; it is not the published Ember algorithm.

## Gradient modification and rare-output-token dynamics

### Hu et al. (2026): decoupled moment routing

*Hidden Failure Modes of Gradient Modification under Adam in Continual Learning*, arXiv:2604.22407.

The paper shows that sending a modified gradient to both Adam moments can undermine the intended gradient modification: reducing a protected direction also reduces its second-moment denominator and can inflate its effective learning rate. Its repair sends the modified gradient to the first moment and raw gradient magnitudes to the second moment, with adaptive strength.

**Collision boundary:** the general principle of modified-update/raw-second-moment routing is established. It does not test token-class projectors or Ember's factored second-moment estimator. The current experiment tests this specific adaptation to a class-projected LM head.

### Han et al. (2026): absent-token output drift

*A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks*, arXiv:2610.09835.

The paper identifies one-sided softmax gradients on absent output-token rows as a source of forgetting in vocabulary-deficient continual training. It raises Adam epsilon only for the output projection (\(10^{-4}\) instead of \(10^{-8}\)); across seven stable settings from 160M to 12B it reports 39.4–67.9% forgetting reduction. It also compares row freezing and update gating.

**Collision boundary:** strong baseline for rare/absent output rows when the new corpus does not cover the old vocabulary. It does not use token groups or cross-row gradient sharing. It is especially relevant to the zero-exposure bucket and should be added as an optimizer baseline before claiming rare-token-specific benefit.

### Godey & Artzi (2026): LM-head backward bottleneck

*Lost in Backpropagation: The LM Head is a Gradient Bottleneck*, COLM 2026, arXiv:2603.10145.

The paper studies the rank-\(d\) mapping from hidden states to \(V\)-dimensional logits. It reports that 95–99% of logit-gradient norm is suppressed when the signal is backpropagated through the LM head into the preceding network.

**Collision boundary:** this is a different bottleneck from token-row gradient sharing. A class-mean projector applied to output-head parameter gradients does not, by itself, restore logit-gradient components in the null space of the head's backward map. Do not claim the projector solves the LM-head backward bottleneck.

## Additional full-text collision checks

### Hierarchical superclass loss: Urbani et al. (2024)

*Harnessing Superclasses for Learning from Hierarchical Databases* [Urb24] defines a proper scoring rule over the probability mass of each ancestor superclass. For leaf softmax \(p\), superclass membership vector \(a_j\), and \(s_j=a_j^\top p\), its loss is

\[
L_T(p,y)=-\sum_{j\in\mathrm{ancestors}(y)}w_j\log s_j.
\]

The logit gradient is

\[
\frac{\partial L_T}{\partial u_r}
=p_r\left(W_y-\sum_{\substack{j\in\mathrm{ancestors}(y)\\r\in v_j}}\frac{w_j}{s_j}\right).
\]

This is a genuine class-probability-mass objective, so it is closer than ordinary label smoothing. But it is not the candidate projector: it weights each member by its own \(p_r\), depends nonlinearly on target-path superclass masses \(s_j\), and does not broadcast the same class-mean residual to every member row. The candidate remains a fixed optimizer-side map \(G\mapsto(I+\alpha P)G\), not this hierarchy-aware loss.

### Concept-level training: Iyer et al. (2026)

*Beyond Tokens: Concept-Level Training Objectives for LLMs* [Iye26] presents two routes: data augmentation with alternative synonym/hypernym completions, and a set-likelihood objective over acceptable completions. Under the natural one-token set-probability interpretation \(P_C=\sum_{i\in C}p_i\), the derived logit gradient is

\[
\frac{\partial[-\log P_C]}{\partial z_j}
=p_j-\mathbf1[j\in C]\frac{p_j}{P_C}.
\]

This is a probability-weighted conditional target over an accepted set, not a uniform class-mean residual. The paper does not derive or claim the projector operation. Its gradient equation is a derivation from the stated objective, not an equation printed in the paper.

### Embedding-similarity supervision: Inan & Khosravi (2016)

*Improved Learning through Augmenting the Loss* [Ina16b] is an important language-model neighbor. It ties the output projection to the input embedding matrix, \(z=L^\top h\), constructs an informative target by

\[
\widetilde y=\operatorname{softmax}(L^\top L y),
\]

and trains with

\[
\mathcal L=CE(y,p)+\alpha D_{KL}(\widetilde y\Vert p).
\]

The extra prediction-logit gradient is target-side \(p-\widetilde y\). The target itself depends on the embedding matrix, so its full parameter derivative also includes the derivative through \(\widetilde y\). It is an embedding-similarity soft-target loss on a tied-embedding RNNLM, not an untied flat-head projector and not an added \(Pp\) term.

### Cluster-based soft targets: Mei et al. (2024)

*Output Regularization With Cluster-Based Soft Targets* [Mei23] explicitly builds classwise soft targets from a class-relationship matrix obtained in cluster space, then combines those targets with ground truth. The available abstract establishes that it is a target-construction/output-regularization method in image classification. The full PDF was not available in this audit, so its exact target formula and derivative remain unverified. It must remain a live adjacent reference; do not claim a full-text exclusion.

### Other class-aware objectives and output factorizations

- **Per23, Beyond One-Hot-Encoding:** uses auxiliary semantic vectors to build a similarity Gram matrix and mixes its class-similarity row with the one-hot label. Its cross-entropy gradient is \(T p-q\), where \(T=\sum_iq_i\); this is a soft-target method, not uniform residual broadcast across class-member rows.
- **Dog19, Label-similarity Curriculum Learning:** constructs time-varying semantic soft targets and applies ordinary soft-target cross-entropy. It changes the target distribution; it does not add a prediction-side class projector.
- **GALA [Zha24h]:** applies target-conditioned logit offsets based on accumulated positive/negative class gradients. Its training gradient is ordinary cross-entropy \(q-y\) on adjusted logits; it does not compute \(P(p-y)\).
- **FlexiGrad [Zho26b]:** performs conflict removal/cooperation weighting between gradients of different hierarchy-level tasks. It operates on task gradients, not token-row class means.
- **SGG [Li25f]:** clusters momentum entries within a layer and scales per-parameter learning rates by cluster-specific factors. It does not average or project the actual vocabulary-row gradient direction.
- **Coupled Adam [Sto25]:** keeps row-specific gradients and first moments but averages bias-corrected second moments across vocabulary rows. It couples denominators, not gradient vectors.
- **Heavy-Tailed Class Imbalance [Kun24] and Rare-Token Fixed Points [Kar26b]:** explain optimizer/frequency effects that can alter rare-token learning without any class projector. They motivate matched optimizer controls and frequency-stratified reporting.
- **LightRNN [Li16], VQ-Logits [Sha25], and Self-organized Hierarchical Softmax [She17]:** share output parameters or cluster the vocabulary, but replace/factorize the flat output distribution. VQ-Logits aggregates token-logit gradients into shared codebook vectors; LightRNN uses separate row/column softmaxes; Self-organized Hierarchical Softmax uses cluster then within-cluster softmax. None retains an independent full token row and adds a normalized class-mean residual to it.



### Further optimizer and class-output neighbors

- **Structured Fisher / Alice [Gon25].** RACS uses a Kronecker structure with diagonal scaling along matrix rows and columns. Alice learns a low-rank gradient subspace \(U\) and uses \(UU^\top\)-type projection plus a complement-space correction. The projector form is a genuine mathematical neighbor: if Alice's \(U\) spans the membership subspace \(\operatorname{col}(A)\), then \(UU^\top=P\). But Alice estimates its subspace from gradient-covariance information; it does not define vocabulary classes, use class means, or isolate the prediction-minus-target residual by token group. This is important prior art for generic subspace preconditioning, not an exact algorithmic anticipation.
- **Gradient starvation [Omo26].** The paper studies rank-limited parameter-gradient subspaces in low-rank LM pretraining and intervenes with gradient truncation / subspace refresh. It does not project the softmax residual into token classes; the mechanism is restricted parameterization and available update directions, not class-row averaging.
- **VQ-Logits [Sha25].** For each code \(c\), its shared codebook gradient aggregates \(\sum_{i:M(i)=c}(p_i-y_i)h\). This is close to the unnormalized class-sum component \(A^\top G\), but VQ-Logits replaces independent token output rows with shared codebook vectors and scatters code logits back to the vocabulary. It does not retain a full untied head and add the normalized mean projector \(AD^{-1}A^\top G\).
- **LightRNN [Li16]** and class-based RNNLM output layers [Shi13, Bot15, Le11] factorize the output distribution into row/column or class/within-class probabilities. They change the output normalization and parameterization; they are not flat-softmax gradient projectors.
- **Self-organized Hierarchical Softmax [She17]** learns clusters online but uses a cluster softmax followed by a within-cluster word softmax. The grouping is dynamic, but the full softmax is replaced rather than augmented with a projector.
- **Urb24** and **Iye26** are the closest loss-level class-probability objectives found in this pass. Their exact derivatives are probability-weighted and target-conditioned; neither broadcasts a constant class-mean residual to each member row.
- **Inan & Khosravi [Ina16b]** is the closest language-model target-similarity objective: it uses \(\\widetilde y=\\mathrm{softmax}(L^\\top L y)\) and adds \(D_{KL}(\\widetilde y\\Vert p)\) to ordinary CE. This is a tied-embedding soft-target loss, not a prediction-side class projector.
- **Mei23** remains unresolved at equation level because its full text was unavailable. Its abstract explicitly describes cluster-derived classwise soft targets, so it remains the closest unverified cluster/target reference.



- **Group Softmax Loss [Kob21]** is a naming false friend. Its groups are feature dimensions in the penultimate representation, not vocabulary classes or output rows. It splits each class logit into ingredient logits and computes separate group-wise softmax losses; the paper explicitly distinguishes this from grouping class categories. It does not apply a class-mean projector to token-row gradients.
- **Interclass Loss [Haq15]** is a hierarchy-distance-weighted classification loss. The accessible copy indicates target-class relationship weighting, but the full source and exact derivative were not available in this audit. Treat it as a target/loss-level neighbor, not as a ruled-out algebraic equivalent.

### Why this is not generally the gradient of another scalar loss

For fixed groups, let \(P=P^\top=P^2\), \(p=\operatorname{softmax}(z)\), and

\[
g(z)=(I+\alpha P)(p-y).
\]

The Jacobian of this transformed logit field is

\[
J_g=(I+\alpha P)J_{\mathrm{softmax}},
\qquad
J_{\mathrm{softmax}}=\operatorname{diag}(p)-pp^\top.
\]

A vector field that is the gradient of a twice-differentiable scalar loss must have a symmetric Jacobian. Here this requires \(PJ_{\mathrm{softmax}}=J_{\mathrm{softmax}}P\), which does not hold for generic nonuniform probabilities within groups. For example, with \(p=(0.55,0.25,0.15,0.05)\), groups \(\{1,2\},\{3,4\}\), and \(\alpha=2\), the Frobenius norm of \(J_g-J_g^\top\) is \(0.2\). Thus the candidate is naturally an optimizer-side preconditioned gradient, not generally the gradient of a conventional scalar loss. It remains a positive-definite preconditioner because \(I+\alpha P\) has eigenvalues \(1\) and \(1+\alpha\) for \(\alpha>-1\).



### Exact normalized-centroid reparameterization collision under SGD

There is an exact fixed-partition parameterization corresponding to the proposed normalized projector, stronger than the unnormalized \(R=U+AC\) baseline.

Let \(D=A^\top A\) and define

\[
W=U+\sqrt{\alpha}\,A D^{-1/2}C.
\]

For a loss with \(G=\nabla_W L\),

\[
\nabla_U L=G,\qquad
\nabla_C L=\sqrt{\alpha}\,D^{-1/2}A^\top G.
\]

If \(U\) and \(C\) are both updated by ordinary SGD with the same learning rate \(\eta\), then

\[
\begin{aligned}
W^+
&=U-\eta G+\sqrt{\alpha}AD^{-1/2}
  \left(C-\eta\sqrt{\alpha}D^{-1/2}A^\top G\right)\\
&=W-\eta\left(I+\alpha AD^{-1}A^\top\right)G\\
&=W-\eta(I+\alpha P)G.
\end{aligned}
\]

Thus, for fixed groups, the candidate projector and a normalized shared-centroid parameterization produce **exactly the same output-matrix trajectory under SGD**, assuming the same learning rate for the residual and centroid parameters. This is not merely a loose analogy to Tree Loss. The original \(U+AC\) parameterization induces the unnormalized \(AA^\top\) term; scaling the centroid basis by \(D^{-1/2}\) gives the normalized projector exactly.

This moves the research question. The abstract operation \(I+\alpha P\) is not a new SGD update geometry: it is a fixed shared-centroid reparameterization written directly as an optimizer update. The empirical contribution, if any, must lie in the practical flat-head implementation, its interaction with adaptive optimizers (Ember/Adam), the use of train-only or evolving token groups, and the resulting frequency-resolved behavior. A matched \(D^{-1/2}\)-scaled centroid+Ember control is therefore required; the existing unnormalized centroid baseline is not sufficient.

For changing groups, the same equivalence holds within each interval of fixed \(A\). At a regrouping boundary, the parameterized representation must be recentered to preserve \(W\) before continuing. Adaptive optimizers are not invariant to this coordinate change because their moment states are maintained separately for \(U\) and \(C\); that is precisely the remaining empirical question.

## What the fork's evidence currently supports

- The flat-head projector improves over plain Ember at V≈50k on the independent seeds tested, but is essentially tied with the upstream shared-centroid method and centroid+Adam.
- On V≈50k, fixed train-context groups lose to online output-row groups; context-first then online regrouping is near-tied with online grouping.
- On V6k at batch128, fixed train-only context groups give a strong short-horizon gain over plain Ember and upstream centroids. The same fixed-context result does not transfer to V≈50k.
- The original dynamic projector is not batch-robust without context initialization.
- The cross-microbatch rho/eta selection experiments do not show a stable advantage over the simpler projected-energy gate.
- The V6k residual decomposition shows that neither prediction-side Pp nor target-side -Py alone reproduces the gain; the difference Pp-Py is essential. Fixed-context full projection beats plain Ember and upstream centroid+Ember at both tested batches and all three fresh seeds.
- Standard target-side semantic smoothing and concept-class CE do not reproduce the projector's validation results.
- The Tree-Loss reparameterization control is near-tied with the direct projector under Ember and Adam.

These results support a **narrow normalized class-mean preconditioner hypothesis**, not a broad claim that dynamic semantic grouping is new or that the method is already a stronger LM-head optimizer.

## Required next comparisons

1. Finish the preregistered direct-projector versus exact normalized-centroid parameterization comparison under Ember and Adam. The fixed-group SGD equivalence is already exact; the remaining question is adaptive-optimizer behavior.
2. Retain context groups through absolute step 250 before switching to online output-row K-means for the batch32 schedule candidate. Batch128 remains fixed context at the tested 100-step horizon.
3. Add output-only Adam epsilon 1e-4 from Han et al., a faithful projected RowNorm baseline from Somasundaram et al., and MuonIO at V≈50k, with per-exposure buckets.
4. Obtain full text for CluOReg [Mei23] and the exact metric-augmentation/class-relationship neighbors that remain unresolved; the Inan & Khosravi paper has now been read in full and is a target-similarity loss, not a projector.
5. Test the surviving mechanism on an untouched shard and at a substantially larger model before making a general LM-pretraining claim.

## Bibliographic links

- Tree Loss: https://doi.org/10.48550/arXiv.2204.07727
- AGG: https://doi.org/10.18653/v1/2022.acl-long.3
- Hypernym Class Prediction: https://doi.org/10.48550/arXiv.2203.10692
- Syntactic Smoothing: https://doi.org/10.48550/arXiv.2410.11462
- Gradient Grouping: https://doi.org/10.18653/v1/2025.acl-long.164
- Som26 RowNorm: https://doi.org/10.48550/arXiv.2608.22253
- Symmetry-Compatible Optimizers: https://doi.org/10.48550/arXiv.2605.18106
- Hu26 Decoupled Routing: https://doi.org/10.48550/arXiv.2604.22407
- Token Geometry / Ember: https://doi.org/10.48550/arXiv.2607.01455
- MuonIO: https://arxiv.org/abs/2610.02705
- Han26: https://arxiv.org/abs/2610.09835
- LM-head Gradient Bottleneck: https://doi.org/10.48550/arXiv.2603.10145

- Superclass scoring rule: https://doi.org/10.1007/978-3-031-70359-1_15
- Concept-Level Training: https://doi.org/10.48550/arXiv.2601.11791
- Inan & Khosravi: https://cs224d.stanford.edu/reports/InanKhosravi.pdf
- Cluster-based soft targets: https://doi.org/10.1109/TNNLS.2023.3262267
- LightRNN: https://doi.org/10.48550/arXiv.1610.09893
- VQ-Logits: https://doi.org/10.48550/arXiv.2505.10202
- Self-organized Hierarchical Softmax: https://doi.org/10.48550/arXiv.1707.08588
- Structured Fisher / Alice: https://doi.org/10.48550/arXiv.2502.07752
- Gradient Starvation: https://doi.org/10.18653/2026.emnlp-main.360
- Coupled Adam: https://doi.org/10.48550/arXiv.2502.08441
- Heavy-Tailed Class Imbalance: https://doi.org/10.48550/arXiv.2402.19449
- Label-similarity Curriculum Learning: https://doi.org/10.48550/arXiv.1911.06902
- Group Softmax Loss: https://doi.org/10.1109/WACV48630.2021.00266
