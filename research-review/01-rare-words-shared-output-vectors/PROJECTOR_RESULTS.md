# Center-preserving group projector: experimental outcomes

2026-11-16; [verified Actions #43852372143](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43852372143) (success). Theory in [PROJECTOR_LEMMA.md](PROJECTOR_LEMMA.md).

## Experiment

Plain Adam-trained head (no shared trainable output-centroid parameters) receives group-mean gradient mixing `G ← G + αP G`, `P=A(Aᵀ A)⁻¹ Aᵀ`; groups refreshed every 50 steps by KMeans of current output head rows. `P` is symmetric, idempotent, and preserves zero-sum gradients for an untied softmax head. Model V6000,width32,batch32,400 steps,epsilon 1e-5,three matched seeds,reconstructed FineWeb data.

| α | Seed0 CE | Seed1 CE | Seed2 CE | Mean CE | Mean gain over epsilon baseline |
|---:|---:|---:|---:|---:|---:|
| 0 (baseline, prior matched run) | 5.0139 | 5.0118 | 5.0091 | 5.01160 | — |
| 0.25 | 5.0091 | 5.0083 | 5.0017 | 5.00637 | +0.00523 |
| 0.5 | 5.0047 | 5.0022 | 4.9940 | 5.00030 | +0.01130 |
| **1.0** | **5.0003** | **4.9993** | **4.9937** | **4.99777** | **+0.01383** |
| 2.0 | 5.0011 | 5.0005 | 4.9962 | 4.99927 | +0.01233 |
| 4.0 | 5.0064 | 5.0106 | 5.0009 | 5.00597 | +0.00563 |
| Original shared-vector online KMeans (prior run) | 4.9937 | 4.9967 | 4.9925 | 4.99430 | +0.01730 |
| Direct **unnormalized** gradient mixing α=0.01 (prior run) | 5.0001 | 4.9988 | 4.9909 | 4.99660 | +0.01500 |

## Interpretation

The centering-preserving, bounded-spectrum projector yields a reproducible positive gain without extra learnable parameters. It has a reasonable optimum near α≈1 in this tiny-setting sweep, compared with α≈0.01 for the unnormalized group sum (which scales with cluster cardinality). **But its average measured validation CE is not better than the original shared-parameter approach**.

This is mathematically attractive but **not a validated new SOTA optimizer**. The difference between α=1 normalized and α=0.01 unnormalized is small; scaling K, model width, vocabulary and group imbalance may reveal whether bounded-spectrum stability matters. A sufficient test would use a fixed hyperparameter across V6k,V16k,V50k and compare performance sensitivity to K and group sizes. It should also measure actual training-side cluster KMeans costs; inference head is unchanged.
