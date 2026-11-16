# Paper 01 — evidence archive and reproducibility audit

Date: 2026-11-16. Review of the exact fork's source `01-rare-words-shared-output-vectors/`.

## What the original material contains

- `PAPER.md`, `PAPER.html`, `paper.pdf`, `RESULTS.md`, code `grid.py`, `tb.py`, `reproduce.sh`, analysis helpers and figures.
- `results.jsonl` stores **88 JSON records**, not the paper's claimed **249 total study runs**; the paper and RESULTS.md also contain other numerical tables, but the full machine-readable log of all 249 runs is **not in this folder**.
- The `grid.py` script defaults to **fixed torch seed 0**; no `--seed` CLI option, so its archive is not a multi-seed replication of each claim.
- `reproduce.sh` runs two representative arms: baseline vs dynamic grouping (V6000, 400 steps), using the default Mac `--dev mps`.

## Missing data required for an exact re-run

`grid.py` looks for `../llm-per-word-weights/data/counts.npy`, `fineweb_train_000001.bin`, and `fineweb_val_000000.bin`. The `llm-per-word-weights/data` folder is **absent from this repository**. To independently test the idea we downloaded the original public GPT-2-tokenized FineWeb10B train/val shards and **recreated token counts from only one 100M-token train shard**.

Therefore there are two logically different claims:
- **Matched-arm scientific replication:** same reconstructed data and vocabulary, baseline vs cluster changes; our Actions establish this rigorously on 3 seeds and multiple ablations.
- **Byte-for-byte original table replication:** requires the exact original `counts.npy`, full corpus ranking, training data selection and environment. This remains to be recovered if exact numeric reproduction is needed.

## Source–paper differences

- PAPER.md pseudocode calls `reset_adam_state(self.C)` after reassigning clusters; actual `grid.py` does not.
- Pseudocode describes warm-starting K-means; actual `kmeans(R,K)` uses fixed-seed random initial rows at every regroup.
- Our separate instrumented experiments did **not edit upstream source**. We explicitly tested keeping C moments, clearing C moments, clearing U+C moments, and **fully resetting Adam state including step counter**. Full results in [MULTISEED_REPRODUCTION.md](MULTISEED_REPRODUCTION.md) and [FULL_RESET_CONTROL.md](FULL_RESET_CONTROL.md).
- Claims about "rare *words*" in the paper refer to **truncated GPT-2 BPE vocabulary token IDs**, which include partial word pieces and an UNK bucket. The correct scientific generalization target is **low-frequency tokens**.

## Practical implications

The algorithm is independently executable and its main dynamic-cluster effect survives multiple seeds. For a submission-quality code artifact, include a deterministic data download/preprocessing script, final seed list, a reproducible sweep manifest, machine-readable raw results for every table cell, and explicit optimizer-state policy.
