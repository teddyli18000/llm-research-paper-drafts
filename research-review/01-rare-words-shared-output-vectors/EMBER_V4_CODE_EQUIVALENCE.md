# Source-level comparison: Ember repository code versus arXiv v4 (2026-10-11)

Checked directly: [arXiv v4 official HTML](https://arxiv.org/html/2607.01455v4), Algorithm 1 and Methods §2.1, plus [pinned GitHub repository](https://github.com/katop1234/ember/tree/76f30117b2d948874cd1c965c4f6813fa9de8caa) `ember.py::ember_update`. The Oct-2026 paper is **newer than the repository README**.

## Apparent discrepancy

The pinned code averages squared gradients over **all vocabulary rows** for the column factor:

```python
col_sum = g32.pow(2).sum(dim=0)
c = beta2*c + (1-beta2)*col_sum/V
s = beta2*s + (1-beta2)*col_sum.sum()/(V*D)
vhat = r[:,None]*c[None,:] / (s*(1-beta2**t))
```

The Oct-2026 **v4 paper** explicitly distinguishes this *all-row variant* from the **active-row column averaging** used in its 50k-step headline experiments. In v4 Algorithm 1, `a_t = max(1,number of rows containing any nonzero gradient)`, and the column factor adds `col_sum/a_t`; normalization is `s_hat=sqrt(mean(rhat)*mean(chat))`.

## Important exact identity for our experiments

**For an untied full-softmax LM head, gradients are dense over the vocabulary rows**: for each token i, `∇R_i = (p_i−1[i=y]) h` (summed over batch). All rows are active in ordinary non-underflow FP32 training, so `a_t=V`. In this case the v4 active-row and repository all-row column updates coincide.

Moreover, for the all-row update:

`mean_i r_t[i] = mean_d c_t[d] = s^{code}_t`,

since all three quantities are initialized at zero and receive the same exponential average of the global mean squared gradient. With `bc=1−beta2^t`,

`s^{v4}_t = sqrt(mean(r_t/bc)·mean(c_t/bc)) = s^{code}_t/bc`.

Thus v4's denominator has

`vhat^{v4}_{i,d} = (r_t[i]/bc)(c_t[d]/bc)/(s^{code}_t/bc) = r_t[i]c_t[d]/(s^{code}_t·bc)`,

**exactly** the pinned GitHub implementation's expression, away from numerical floor/rounding. The difference between Oct-2026 paper and Sept-2026 GitHub version is material for *sparse input embeddings*, but not for the **dense output-only** benchmark in our trial. We should not describe those runs as a full reproduction of the paper's joint input/output tokenizer optimizer, since Ember on the input embedding was intentionally held constant.

## Required numerical audit

The test `ember_v4_equivalence_test.py` compares the pinned repository's `ember_update` against an independent literal implementation of v4 Algorithm 1 on dense random LM-head gradients for multiple `beta2` values and finite steps. Evaluate parameter weights and squared-gradient factor buffers, not merely summary CE.

## Scientific implication

Paper01's **matched head-only comparison remains applicable to v4 Ember's dense LM-head algorithm**, with the qualification that adding Ember to both input and output interfaces is a distinct and yet untested baseline. The important collision is stronger in v4: Ember already demonstrates major real-scale LM-head optimization improvements. Our distinct claim must be about **additional directional inter-token transfer through adaptively learned groups**, not generic token-gradient geometry or rare-token optimizer memory savings.
