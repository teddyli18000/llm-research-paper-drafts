# Exact Adam-state reset control (moments + step/bias correction)

Date: 2026-11-16. [Actions run #43836617555](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43836617555), all six runs **success**. Same public fineweb data, 400 steps batch32 V6000, three seeds, same LR and head epsilon as the [main confirmation](MULTISEED_REPRODUCTION.md).

The earlier reset experiment set `exp_avg` and `exp_avg_sq` to zero but retained Adam's `step` count and thus bias correction. This follow-up implements **full state reset** by also setting `step=0` after each re-cluster. It is the appropriate interpretation of the original paper pseudocode `reset_adam_state`, rather than merely clearing moments.

| Arm | Seed0 val | Seed1 val | Seed2 val | Mean paired improvement over unshared baseline |
|---|---:|---:|---:|---:|
| Epsilon-only baseline | 5.0139 | 5.0118 | 5.0091 | 0 |
| Dynamic keep all Adam state (earlier run) | 4.9937 | 4.9967 | 4.9925 | **0.01730** |
| Dynamic reset **C** moments only (earlier run) | 4.9938 | 4.9960 | 4.9907 | **0.01810** |
| Dynamic **full reset C** (this run) | **4.9937** | **4.9945** | **4.9875** | **0.01970** |
| Dynamic reset **U+C** moments only (earlier run) | 5.0008 | 5.0014 | 4.9944 | **0.01273** |
| Dynamic **full reset U+C** (this run) | 5.0104 | 5.0120 | 5.0054 | **0.00233** |

**Mechanistic readout:** full resets of the *shared cluster C* Adam state do not eliminate the gain, even improve it in this setting. The benefit is therefore not primarily caused by stale shared-cluster moments. But full resets of **both** the private output rows U and shared C dramatically damage convergence. Private-word Adam history must be retained across each function-preserving change of coordinates.

This is an important distinction for future method design: `R=U+AC` can stay unchanged at regrouping while the optimizer history contains meaningful per-word information. Momentum transport/reset policies affect the next training steps and should be treated as part of the algorithm, not an implementation detail.

Three seeds, tiny model: the result is a reproducible short-training effect, not a claim about 125M+ pretraining. Its strongest use is choosing the correct and theoretically coherent state policy for a scalable prototype.
