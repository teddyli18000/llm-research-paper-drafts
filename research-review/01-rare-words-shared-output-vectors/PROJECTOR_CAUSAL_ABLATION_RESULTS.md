# Paper 01 — Causal group-identity and learning-rate ablation (completed)

**GitHub Actions [#44017278316](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44017278316): success, 3/3 jobs, 27/27 preregistered full trainings.** Raw JSONL artifacts in Actions; code and fixed design in [protocol](PROJECTOR_CAUSAL_ABLATION_PROTOCOL.md). V6000, width32, 400 steps, official head-only Ember β₂=.95, LR=.05, α=2 for all group-projection arms, three previously unseen model seeds 223,227,229.

## Results: final validation CE (lower better)

| Arm | Seed223 | Seed227 | Seed229 | Mean CE | Improvement vs plain Ember |
|---|---:|---:|---:|---:|---:|
| **Static train-only contextual projector** | **4.9707** | **4.9675** | **4.9671** | **4.96843** | **+0.03353** |
| **Dynamic learned KMeans projector** | 4.9749 | 4.9758 | 4.9680 | **4.97290** | **+0.02907** |
| Static random groups | 4.9935 | 4.9906 | 4.9867 | 4.99027 | +0.01170 |
| One-time KMeans refresh only at step50 | 4.9948 | 4.9976 | 4.9871 | 4.99317 | +0.00880 |
| KMeans recomputed but row membership shuffled, group sizes retained | 4.9938 | 4.9972 | 4.9936 | 4.99487 | +0.00710 |
| Plain Ember at **2× head learning rate** | 5.0052 | 4.9961 | 4.9919 | 4.99773 | +0.00423 |
| Static frequency-equal-mass groups | 4.9993 | 4.9999 | 4.9954 | 4.99820 | +0.00377 |
| Plain Ember, LR=.05 | 5.0018 | 5.0048 | 4.9993 | **5.00197** | — |
| Plain Ember at **3× head learning rate** | 5.0119 | 5.0005 | 4.9948 | 5.00240 | −0.00043 |

## What the paired controls establish

1. **Real group identity matters**: dynamic learned KMeans is **+0.02197 CE better** than its same-KMeans-compute, same-group-size but shuffled-membership arm (paired seed differences +0.0189,+0.0214,+0.0256). It is +0.01737 better than fixed random groups. This cannot be explained by KMeans wall compute or group population alone.
2. **The grouping update is directional, not just changing head LR**: ungrouped Ember with 2× or 3× LR does not approach dynamically learned projected gradients. This does not rule out every other optimizer hyperparameter, but rejects this direct same-run LR explanation.
3. **Repeated adaptation matters compared with an early self-partition**: dynamic online KMeans improves over KMeans assigned once at step50 by **+0.02027**. The self-generated cluster membership is initially uninformative and needs continuing updates.
4. **Crucial counterexample to claiming dynamism itself is best**: a strong fixed **train-only semantic/contextual group partition** is better than the dynamic projector on all three seeds (+0.0042,+0.0083,+0.0009, mean **+0.00447 CE**). This implies useful group semantics/gradient coherence matter more than the mechanism's adaptive grouping in this V6k/400-step condition. Strong static semantic baselines must appear in every future manuscript.

## Efficiency

Original training time per arm was approximately **40–51 seconds** on individual GitHub CPUs. These sequential jobs exhibit host/load variation of several seconds, so differences of only 1–3 seconds between variants cannot substantiate a training-speed claim. Fixed-context group-building time is extra and reported in its separate preprocessing job.

## Originality consequence

The source author already invented and tested **online shared-centroid** KMeans grouping and independently identified the 10+ target-exposure threshold. Our **single-head normalized gradient projector before Ember** is a distinct candidate update rule, with a measured benefit supported by unusually strong causal grouping controls. The strongest new scientific opportunity may be **fixed contextual class-gradient preconditioning** rather than adaptive KMeans itself, provided the fixed-context projector scales to V50k and independently beats both original split centroids and modern head-only optimizers.

This is now a well-motivated **novel extension hypothesis**, not evidence that the entire upstream concept was ours. The next already-registered [V16k/V50k dynamic projector comparison](PROJECTOR_SCALE_CONFIRMATION_PROTOCOL.md) and [external test](LOCKED_FINAL_HELDOUT_5ARM_PROTOCOL.md) will determine whether even the dynamic version survives a larger vocabulary.
