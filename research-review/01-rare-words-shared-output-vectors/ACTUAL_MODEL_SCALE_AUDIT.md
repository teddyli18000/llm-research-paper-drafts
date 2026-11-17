# Paper 01 — What V6k, V16k and V50k experiments actually scale

Model code audited directly: [original grid.py](../../01-rare-words-shared-output-vectors/grid.py), `Model.__init__`, `Block.__init__`; head and input embedding each have `V×d` parameters. All current independent replicas have **2 transformer layers** and sequence length **64**. This is a token-interface optimization study, *not* large-language-model evidence. The parameter counts below are calculated from the exact defined layers (RMSNorm adds its scale parameters, value-embedding and smearing options disabled).

## Model weight counts

`V` is the **effective mapped vocabulary count**, not necessarily the `--V` argument, and the original code has an extra unused `Ch[1,d]` for `clus=0`. Table counts are rounded to nearest thousand.

| Test | Effective vocab | Width | Trainable input+output token tables | Transformer block weights | Total model weights (approx.) | Token-table fraction |
|---|---:|---:|---:|---:|---:|---:|
| V6k/width32 | ~6,000 | 32 | 384k | 24k | ~410k | **94%** |
| V16k/width128 | ~16,000 | 128 | 4.10M | 0.39M | ~4.49M | **91%** |
| V50k/width64 | 50,257 | 64 | 6.43M | 0.098M | ~6.54M | **98%** |

Even V50k tests have only **~6.5 million** model weights, most in the embedding and classifier tables, with only ~0.1 million in the transformer layers. Thus the study has enlarged **vocabulary and output-head state**, not scaled the contextual backbone to 100M+ parameters.

The single-head projector keeps the ordinary output-head parameter count `Vd`, with **no** trainable group-centroid matrix. The original author's split head stores `Vd+Kd` learned head parameters plus one integer assignment per token. The difference of `Kd=64×64=4096` learned values at V50k is ~0.13% of the output table and therefore cannot yield major *whole-model* parameter memory savings. The strongest practical difference may instead be **simpler optimizer state transport and no centroid optimizer**, plus any better convergence. Claiming big GPU memory savings without profiling would mislead.

## Why the scale matters scientifically

- At V≈50k, width64, even strong small-model CE improvements say little about a 7B-parameter LLM's ability to learn semantics and identify useful groups; its hidden features and optimizer geometry differ profoundly.
- The original author already reported that **large batches weaken sharing**. Our new normalized projector was worse than plain Ember at batch128×100 despite the same token budget, so a real large-batch scaling claim is currently contradicted, not merely untested.
- Our paired vocabulary and optimizer tests answer a narrower but valid question: **can class-group gradient projection help low-exposure token interfaces in a small model**? The best route to a major conference is to show an actual large-backbone effect or derive a mechanism that predicts where/when it fails, with proper credit to upstream.

The most discriminating new runs are the independent 50k projector tests, the context-first batch128 rescue, and an evaluation with a meaningfully scaled Transformer backbone and realistic optimizer schedule.
