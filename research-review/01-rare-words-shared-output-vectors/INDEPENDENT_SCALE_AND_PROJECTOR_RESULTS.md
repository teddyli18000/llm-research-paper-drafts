# Paper 01 — Locked-scale transfer, V50k semantic controls, and new projector pilot

**Verified completed runs (UTC 2026-11-17):**
- [Independent modern optimizer transfer #44002552718](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44002552718): **6/6 matrix jobs successful**, seeds 41,73,109, V16k/50k, fixed preselected head hyperparameters.
- [Full-vocab semantic comparators #44003042366](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44003042366): **3/3 jobs successful**, same seeds and 800-step V≈50k settings.
- [New parameter-free class projector #44004122511](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44004122511): **3/3 jobs successful**, fresh seeds 211,307,401, V6k/400-step.

## 1. Independent modern optimizer transfer

All numbers below are **three-seed means of validation CE**, lower better; not selected best single-run results. Fixed seeds/order as registered in [transfer protocol](EMBER_SCALE_TRANSFER_PROTOCOL.md).

| Output-head optimizer / grouping | V16k width128 / 400step | V50k width64 / 800step |
|---|---:|---:|
| Tuned Adam head, plain | 5.80253 | 6.28850 |
| **Adam + online learned groups** | **5.78330** | **6.25773** |
| Official Ember head-only, beta2=.999 | 5.84310 | 6.28023 |
| **Ember beta2=.999 + online groups** | 5.81530 | **6.25877** |
| Official Ember head-only, beta2=.95 | 5.82323 | 6.39673 |
| Ember beta2=.95 + online groups | 5.80357 | 6.37007 |
| Same-training-data static context groups (Adam) | **5.78023** | see next section |

**Paired marginal benefit of adding dynamic groups:**

| Scale | Adam grouping | Ember beta2=.999 grouping | Ember beta2=.95 grouping |
|---|---:|---:|---:|
| V16k | +0.01923 | +0.02780 | +0.01967 |
| V50k | +0.03077 | +0.02147 | +0.02667 |

All 18 paired contrasts (three seeds × three optimizers × two scales) in this registered matrix have **positive direction**. Group-gradient communication is therefore complementary to the *tested fixed Ember settings*, beyond Adam epsilon, at vocabulary up to near-full GPT-2. However, **the V6k-optimized Ember beta2=.95 setting transfers poorly to V50k** (ungrouped 6.39673, grouped 6.37007). The stronger Adam grouped head (6.25773) and Ember beta2=.999 grouped head (6.25877) are **essentially tied** by measured CE (mean difference 0.00103). On V16k, static contextual Adam (5.78023) is best among these registered settings; both Ember settings remain worse than tuned Adam there.

**Interpretation:** at both tested vocabulary scales grouping adds value within a matched optimizer, but it is false to claim that grouped Ember is automatically the best training algorithm across optimizer families and model widths. A fair *scale-specific Ember hyperparameter sweep* is still required before drawing optimizer-wide conclusions.

## 2. Strong static semantic grouping on near-full GPT-2 vocab

On the same three V≈50k model seeds, original Adam-head rates, static KMeans groups from train-only adjacent-token co-occurrence, and optional late online regrouping:

| Arm | Seed41 | Seed73 | Seed109 | 3-seed mean CE |
|---|---:|---:|---:|---:|
| Static co-occurrence contextual grouping | 6.2822 | 6.2768 | 6.2813 | 6.28010 |
| Static unigram-frequency grouping | 6.2742 | 6.2763 | 6.2843 | 6.27827 |
| **Contextual groups → online at step200** | 6.2727 | 6.2633 | 6.2740 | **6.27000** |
| **Pure online groups from separate matched experiment** | 6.2601 | 6.2570 | 6.2561 | **6.25773** |

The late-online adaptation beats static contextual grouping on **all three seeds** (paired CE improvements +0.0095,+0.0135,+0.0073, average +0.01010). Pure online KMeans beats late-online on all three (mean +0.01227) in the present 800-step horizon.

**But contextual semantic clustering did not beat frequency-only clustering on average at 50k** (6.28010 vs 6.27827). The number of observed token IDs in this 1.64M train-token prefix was 42,466, with **7,790 unobserved mapped IDs**; the sparse co-occurrence matrix has ~1.2M nonzeros. Static context group sizes varied by CPU job even with `KMeans(random_state=0)` (min group 254–329, max 7,981–7,991), so deterministic clustering across hosts and preprocessing cost must remain explicit. Reported static-context preprocessing wall clock was ~2.4–3.1 seconds per run, excluding FineWeb shard download/initial count prep.

**Conclusion at this scale:** online group *adaptation* helps compared to strong local train-only context and frequency baselines under the registered settings. This result does not negate the earlier V6k/tied early-training static-context advantage; the relative usefulness of clustering strategies depends on width, vocabulary, exposure and horizon.

## 3. Parameter-free normalized projector + official Ember (V6k)

This is a **different training method** from the upstream owner's parameter split. A plain single output matrix receives `G̃=(I+αP_t)G` before Ember, with `P_t=A_t(A_tᵀA_t)⁻¹A_tᵀ` and KMeans assignments recomputed every 50 steps. The base architecture, data, evaluation and Ember LR=.05/beta2=.95 match. Four deterministic models per seed? **Five arms per seed**: Ember plain, Ember+original dynamic centroids, plus projected Ember α=.5/1/2.

| Arm | Seed211 | Seed307 | Seed401 | Mean CE |
|---|---:|---:|---:|---:|
| Ember head plain | 5.0024 | 5.0047 | 4.9997 | 5.00227 |
| **Ember + original shared centroids** | 4.9790 | 4.9766 | 4.9761 | 4.97723 |
| Projected Ember α=.5 | 4.9852 | 4.9817 | 4.9791 | 4.98200 |
| Projected Ember α=1 | 4.9794 | 4.9777 | 4.9730 | 4.97670 |
| **Projected Ember α=2** | 4.9789 | 4.9756 | 4.9747 | **4.97640** |

At α=2, the new projector gains **0.02587 CE over standalone Ember**, and is numerically **0.00083 lower** than the original split-parameter method on average (paired original-minus-projector +0.0001,+0.0010,+0.0014). The tiny advantage over the split-parameter original is **not a proven superiority claim**: these are three tiny-model seeds and α was chosen from a small pilot grid. **The important result is that a *parameter-free* bounded-spectrum projector matches the original's effect when combined with Ember**. It removes trainable centroid vectors and parameter/moment state transport. Test global-LR, random/static-group, and no-KMeans comparators to establish whether its particular dynamic grouping geometry is necessary.

## 4. Paper originality and next decision

**Already the author's work:** `R=U+AC`, KMeans reassignments with exact logits, shared centroid gradients, exposure-dependent gain and zero-exposure harm. See [originality ledger](ORIGINAL_IDEA_VS_NEW_CONTRIBUTION.md); reproduction does not create a new algorithm.

**Our distinct candidate:** the normalized `I+αP_t` gradient-preconditioned output head with no centroid parameters, particularly its combination with official Ember. This remains a *candidate* pending controls against global LR rescaling, random/fixed group partitions, and larger-vocabulary independent seeds. A new research paper should focus on these differences, not rebrand the original dynamic centroid algorithm.

Next: execute the registered [out-of-sample evaluation](INDEPENDENT_OUT_OF_SAMPLE_TEST_PLAN.md) and a group-geometry causal ablation. The remaining major gate is meaningful (≥100M) LM scale / realistic batches and actual method overhead.
