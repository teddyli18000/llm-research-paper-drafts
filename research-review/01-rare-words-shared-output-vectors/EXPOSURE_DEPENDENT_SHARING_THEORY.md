# Paper 01 — Exposure-dependent value of sharing: a bias–variance model

**Purpose:** derive falsifiable predictions from the empirical V50k token-frequency result. This is a deliberately simplified **estimation analogy**, not a proof of convergence or generalization for the actual nonlinear Adam/Muon language-model training process. Algebra below is exact within its stated toy assumptions.

## 1. Minimal estimation model

Fix a token `i`, a feature coordinate and a latent ideal update direction `θ_i`. Let `n_i>0` be the number of direct observations of token `i`.

- Its private estimate is `X_i = θ_i+ε_i`, `E[ε_i]=0`, `Var(ε_i)=A_i=σ²/n_i`.
- Its group-derived estimate is `Y_g = θ_i+δ_i+η_g`; `δ_i` is the group mismatch (true token-specific deviation or assignment error) and `η_g` is the pooled estimation noise.
- Assume `E[δ_i]=E[η_g]=0`, `Var(δ_i)=τ_i²`, `Var(η_g)=σ²/N_{g,-i}`, and all three errors independent. Define `B_i=τ_i²+σ²/N_{g,-i}`.

The shrinkage estimator is `θhat_i(λ)=(1−λ)X_i+λY_g`. Then

`MSE_i(λ)=(1−λ)² A_i+λ² B_i`.

Differentiate:

`d MSE/dλ=−2(1−λ)A_i+2λB_i`,

so the unique minimizer over `λ∈[0,1]` is

`λ_i*=A_i/(A_i+B_i)=1/(1+B_i n_i/σ²)`.

At this optimum,

`MSE_i(λ_i*)=A_i B_i/(A_i+B_i)`,

which is lower than private-only MSE `A_i` by `A_i²/(A_i+B_i)` whenever `A_i>0` and `B_i<∞`.

## 2. Why the lowest-frequency token bucket can lose

If the group labels came from trustworthy external semantics, `B_i` can remain small even at `n_i=0`; sharing may then be valuable. But the algorithm in Paper01 **self-clusters its undertrained output rows**. In that scenario, `τ_i²` is not a fixed constant independent of `n_i`: for never-seen tokens the group assignment may be almost arbitrary and `B_i` may be extremely high. Then neither the certainty of the group estimate nor the optimal mixing weight is identifiable from `1/n_i` alone.

The observed [V50k frequency result](FULL_VOCAB_FREQUENCY_DECOMPOSITION.md) is consistent with precisely this nonmonotonic trade-off: tokens with **0 or 1–9 direct target observations** can lose, tokens with **10–99 observations** gain strongly, while very frequent tokens have already learned their private directions sufficiently well and need little sharing.

This is a potential explanation, not causal proof. Full-softmax head rows also receive non-target probability gradients on every example, so `n_i` is **target exposure**, not the total number of gradients for row `i`.

## 3. Distinctive testable predictions

1. A group-quality measure estimated **only from train data** (e.g. within-group semantic/contextual agreement, cluster-assignment stability across updates) should distinguish useful from harmful sharing *within the same exposure bucket*. A simple frequency-only weight need not succeed.
2. As pretraining continues, the group estimate's mismatch `τ_i²` may decrease because online output vectors become more informative. This predicts why online KMeans can overtake a static contextual partition at long horizons, but the crossover needs a matched controlled experiment rather than assuming it happens everywhere.
3. Rare-token error is not homogeneous. Expose separate bins `0,1–9,10–31,32–99,100+`; reporting merely “rare words improve” masks the negative-transfer regime.
4. When a modern optimizer (e.g. [Ember v4](https://arxiv.org/html/2607.01455v4)) reduces the private estimator variance `A_i`, the optimal transfer weight may change, yet nonzero group sharing can still help if the group mismatch `B_i` is sufficiently low. The independent [Ember matched experiments](MODERN_OPTIMIZER_AND_TIED_RESULTS.md) show this *can* happen at V6k.

## 4. Limits and novelty boundary

This is elementary shrinkage/empirical-Bayes reasoning, not a newly invented statistical theorem. The paper's possible contribution is to connect the *observed exposure/semantic-phase effect* to a **training-time learned, exactly function-preserving output-head gradient coupling**, and verify this mechanism against strong external semantic groups and modern optimizers.

A proposed reliability-aware group-gradient intervention must be evaluated on new model seeds and a fresh heldout validation slice, since the 10–99 exposure sweet spot was discovered by inspecting earlier V50k outputs. Repeated threshold tuning on that validation split does not constitute independent evidence.
