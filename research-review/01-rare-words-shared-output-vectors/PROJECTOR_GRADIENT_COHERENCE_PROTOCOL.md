# Paper 01 — Causal mechanism probe: projected gradient-energy alignment

**Mechanism-only diagnostic registered after the completed V6k geometry ablation**, not a new model-selection test. It measures the gradient operator's geometry during training, not just final validation CE. A normalized group projector can only help through meaningful alignment if its rank K is much smaller than V.

## Measure

At fixed step `t`, given pre-Ember output gradient matrix `G_t` and the arm's group assignments `A_t`, define `P_t=A_t(A_tᵀA_t)^−1 A_tᵀ`. Record

`c_t = ||P_t G_t||_F² / ||G_t||_F²`.

For an untied full softmax, `1ᵀG_t=0`. Under the isotropic-gradient null independent of a fixed grouping, `E[c_t]=(K−1)/(V−1)=63/5999≈0.0105` at V6000,K64. Learned groups may correlate with gradients and yield larger values. This null is not an actual numerical baseline for nonlinear LM gradients; the **shuffled same-size control is the main empirical comparator**.

## Frozen measurements

- Repeat the V6k/400-step α2 head-only official Ember setup used in the [completed three-seed causal ablation](PROJECTOR_CAUSAL_ABLATION_RESULTS.md).
- The **same model seeds 223/227/229** and three projected arms `dynamic`, `shuffled` and `context`. These are diagnostic replays, not additional independent efficacy claims.
- Collect `c_t` before modifying `G_t` and before optimizer.step at steps **0,25,50,100,200,399**. Each measurement is a scalar ratio calculated from the exact current dense head gradient, including all minibatch target and non-target terms. No new gradient update is introduced beyond the original α2 projector.
- Record final original validation CE, per-frequency bins, and the six coherence values in the run artifact. Verify CE reproduces prior same-seed arms within observed CPU variability.

## Falsification

If `c_t` for real learned/static semantic groups is much larger than shuffled groups and the largest coherence aligns with the observed benefit, that supports the **gradient-subspace alignment** mechanism. If coherence is equal despite very different CE, it is not sufficient, and class semantics may influence the optimizer in a more specific or temporal way. In neither case is it valid to claim that positive CE proves a higher-dimensional projected-gradient alignment without measuring it.
