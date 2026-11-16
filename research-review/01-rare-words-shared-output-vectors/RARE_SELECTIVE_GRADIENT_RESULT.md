# Paper 01 — rare-target-selective gradient transfer experiment

Date: 2026-11-16. [GitHub Actions #43854482667 — success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43854482667). V6000, width32, batch32, 400steps, reconstructed public FineWeb, three initializations and matched training hyperparameters.

## Hypothesis

The V16k [frequency analysis](RARE_TOKEN_BUCKETS_16K.md) showed dynamic cluster sharing has its largest contribution on very rare tokens and can slightly hurt frequent tokens. Test the direct gradient-kernel method with a training-frequency-dependent row weight:

`G'_i = G_i + α · w(f_i) · Σ_(j∈cluster(i)) G_j,`
`w(f)=min(1,τ/f)`; `α=0.01`, `τ ∈ {3e-5,1e-4,3e-4}`.

Group assignments are still learned online via KMeans every 50 steps. Train-corpus token probabilities are used, not validation token frequencies.

## Results

| Method | Seed0 CE | Seed1 CE | Seed2 CE | Mean CE | Mean improvement vs unshared baseline |
|---|---:|---:|---:|---:|---:|
| Epsilon baseline (prior run) | 5.0139 | 5.0118 | 5.0091 | 5.01160 | — |
| Original shared-centroid dynamic head (prior run) | **4.9937** | **4.9967** | **4.9925** | 4.99430 | +0.01730 |
| Direct **ungated** group-gradient kernel α=.01 (prior run) | 5.0001 | 4.9988 | 4.9909 | 4.99660 | +0.01500 |
| Gated gradient, τ=3e-5 | 5.0051 | 5.0006 | 4.9944 | 5.00003 | +0.01157 |
| Gated gradient, τ=1e-4 | 5.0018 | 4.9981 | 4.9929 | 4.99760 | +0.01400 |
| Gated gradient, τ=3e-4 | 5.0003 | 4.9984 | 4.9916 | 4.99677 | +0.01483 |

The naïve frequency attenuation **does not improve aggregate validation CE** relative to the ungated group-gradient kernel, though it remains better than the plain baseline. This is an informative negative result; “rare-token specialization” is not automatically a free improvement. It may also be that common-row gradients support the model's shared hidden-state optimization, or that attenuation is miscalibrated.

The hypothesis about reducing frequent-token harm can be evaluated by frequency-bin outputs already saved in the job artifact. No further alpha or tau tuning should be advertised as an independently validated result without a new validation split. The next stage should prioritize the core algorithm's full-vocabulary and longer-run reliability over tuning this small gain.
