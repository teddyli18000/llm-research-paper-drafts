# Paper 01 — tied input/output embeddings as an architectural falsifier

**Pre-registered, 2026-11-16.** All prior head-sharing experiments default to `--tie 0`, an untied LM head. Many standard language models tie input embeddings and output classifiers. The original `grid.py` supports `--tie 1`: it uses the same `R=U+AC` for embedding lookup **and** next-token classification. Regrouping still preserves the entire forward function exactly because every row of `R` is unchanged.

## Key theoretical difference

In the untied setting, the softmax gradient matrix `G_out` satisfies `sum_i G_out[i,:] = 0` for each training batch. Tied embeddings introduce an additional sparse input-lookup gradient `G_in`. The full shared token-table gradient is

`G_tied = G_out + G_in`.

Therefore `sum_i G_tied[i,:] = sum_i G_in[i,:]` is **not generally zero**. The earlier interpretation of the group projector as zero-sum preserving applies to the untied classifier component, not the full tied table. Group sharing can now affect contextual representations as well as output logits; a positive result would be a stronger architectural transfer claim, while a negative result would delimit the method's scope.

## Fixed experiment

- Same GPT2-token FineWeb train and val shards, reconstructed V6000 mapping, width32, layers2, heads2, batch32, L64, head epsilon1e-5, LR head0.1/centroid0.02 and fixed original body settings.
- `--tie 1` in **every** arm; model seeds **0,1,2** with identical initialization/data per arm.
- Two full-from-scratch training horizons: **400 and 1600 steps**. `grid.py` learning-rate cooldown is a function of total steps, so these are distinct schedules; compare only same-horizon arms as matched experiments.
- Three grouping conditions per horizon/seed:
  1. Plain tied token table, no shared groups.
  2. Dynamic64 KMeans groups with regroup every50 steps.
  3. Same-training-data co-occurrence group partition (fixed), generated from first819,200 tokens by `context_groups.py`.
- Total **18 full trainings** (3×2×3). CPU one thread for stable reproducibility; same public corpus prefix and 600 heldout validation sequences.

## Primary decision

Report per-seed paired validation CE, token frequency-bucket losses, and dynamic vs static comparison. Determine whether the adaptive grouping advantage persists with tied input/output weights, including the long horizon. If tied embedding sharing hurts, the original paper must be framed as an **untied-output-head optimization technique** rather than generic LM token-interface optimization. No retrospective tuning or group-count search based on these results.
