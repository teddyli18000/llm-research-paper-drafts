# Mechanism identification: prediction-side versus target-side class residual

The normalized class-gradient operator is `G'=G+αP G` for `G=(p-y)hᵀ`, fixed group-mean projector P. This is **not the same** as conventional label smoothing or concept-class CE, as derived in [GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md](GRADIENT_PROJECTION_VS_SEMANTIC_SMOOTHING.md).

## Exact centered decomposition

Since P is a partition projector, `P1=1`, and both p and one-hot y sum to 1. Let `u=1/V · 1` be the vocabulary-uniform vector.

Define

- Prediction-side *centered* class correction: `c_p=(Pp-u)hᵀ`.
- Target-side *centered* class correction: `c_y=-(Py-u)hᵀ`.

Then `c_p+c_y=P(p-y)hᵀ=P G`. Both separately preserve zero-sum across output rows: `1ᵀ c_p=1ᵀ c_y=0`. This removes the known common-logit-shift confound from a naive one-sided ablation.

The corresponding four training interventions are:

1. Plain CE gradient `G`.
2. Prediction-only: `G+α c_p`.
3. Target-only: `G+α c_y`.
4. Full: `G+α(c_p+c_y)`.

All should be trained from matching initial seeds, head optimizer/LR, group assignments and data, and compared at the same α. Unlike target-only label smoothing `G+ε(y-Py)hᵀ`, the target-only **centered** variant above has a different hard-target coefficient and is therefore a distinct causal control, not a generic published loss.

## Falsifiable explanation

- If prediction-only dominates and target-only hurts, the improvement originates mainly in correcting excessive probability mass shared by a class, rather than transferring positive target supervision.
- If target-only dominates, the mechanism is closer to semantic label smoothing and prior class-target methods, weakening the claim of a distinct optimizer.
- If neither alone matches the full update, simultaneous class-probability error balancing may be important.
- Under Ember and Adam the optimizer is nonlinear, so **the effects of separate training runs are not additive** just because gradient algebra is additive. Do not infer a training-outcome decomposition from one-step equations.

A fully independent test of this decomposition is a useful next experiment *after* the currently running full smoothing, concept-probability and Tree Loss reparam comparisons are analyzed. Fix a protocol before collecting training results; avoid continually searching more hyperparameters on the same V6k validation corpus.

This identifies a credible mechanism question that could differentiate a new class-residual optimization paper from the existing shared-class parameterizations, but the equation alone does not confer strong algorithmic novelty.


## Status correction after the first decomposition run

The first one-sided run used raw \(Pp\) and \(-Py\), not the centered terms above. Because both raw terms sum to the same batch-mean hidden vector across vocabulary rows, they contain opposite vocabulary-uniform components. The full difference cancels this component. A uniform row update is softmax-invariant for the current batch but can still affect Ember's adaptive state. Therefore the first one-sided CE degradation is **not a clean test of whether either centered residual is useful**.

The full-projector arm remains valid and retains its measured gain. The raw one-sided results are exploratory only. The preregistered centered follow-up compares \(Pp-U\), \(Py-U\), the full difference, and uniform-only plus/minus controls: [CENTERED_RESIDUAL_PROTOCOL.md](CENTERED_RESIDUAL_PROTOCOL.md), Actions #44052591527.

