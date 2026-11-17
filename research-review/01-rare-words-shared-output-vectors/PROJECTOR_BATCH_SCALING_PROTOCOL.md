# Paper 01 — Equal-token batch-size falsifier for normalized projected Ember

**Pre-registered 2026-11-16 before any measurements**. The original author's paper already finds the split-centroid group-sharing gain shrinks as batch increases. This trial asks whether the new *single-matrix normalized projection* behaves differently, rather than repeating that upstream finding under a new title.

- V6000, width32, two transformer blocks, 2 heads, public same FineWeb train+val shards, same 819,200 seen train tokens, 600 validation windows; three new model seeds **269,271,277**.
- Official pinned head-only Ember beta2=.95 LR=.05 eps1e−8 and unchanged body/embedding optimizer/learning-rate schedule rules. All trials use group count64, refresh50 and original centroid LR=.02.
- Training regimes (same token budget): **batch32×400 updates×64 tokens** and **batch128×100 updates×64 tokens**. Large batch uses `--acc 4` microbatching; gradient accumulation is mathematically identical to batch128 if sequence grouping/data order match.
- Three methods per regime/seed: (1) Ember plain; (2) upstream original learned centroids + Ember; (3) our normalized projector `G'=(I+2P)G` before Ember. 2×3×3=**18 full runs**.
- Primary: new projector marginal gain `CE(Ember plain)−CE(projector)` at batch128 compared to batch32, and direct comparison to original learned centroids under each batch. Report every paired seed and train exposure bins. Do not tune alpha per batch or select based on this validation.
- Secondary: training wall-time, group-refresh count and relative gain per dollar/compute. The group size is larger at batch128 but only one scheduled refresh occurs (step50), versus seven refreshes in the 400-step batch32 run; this is **part of the causal batch/optimizer-step effect**, not controlled separately.
- This experiment determines whether **the new variant** addresses the original paper's large-batch fragility. It does not establish true million-token global-batch pretraining behavior.
