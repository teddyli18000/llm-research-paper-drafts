# Paper 01 — Independent out-of-sample test after method selection

**Locked evaluation design before completing the V16k/V50k winner-selection runs.** Do not run or alter this confirmatory endpoint until the current [Ember scale transfer](EMBER_SCALE_TRANSFER_PROTOCOL.md) and [V50k semantic controls](FULL_VOCAB_SEMANTIC_CONTROL_PROTOCOL.md) produce complete 3-seed results.

## Purpose

The existing lab repeatedly evaluates on 600 windows from `fineweb_val_000000.bin`. Choosing hyperparameters, clustering rules and the next algorithm on that validation split creates a real selection effect, even when model initialization seeds change. A final evidence claim requires a distinct test dataset never used for method selection.

## Frozen test corpus

- Dataset [kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/tree/main) contains `fineweb_train_000001.bin` (our tiny-LM train prefix), `fineweb_val_000000.bin` (our repeated selection validation), and a separately listed **`fineweb_train_000002.bin`**.
- Use `fineweb_train_000002.bin` as the **independent test shard**, taking exactly 600 evenly spaced windows of length64. The suffix "train" is upstream dataset nomenclature; our short experimental model trains only on shard 000001, so this shard is out-of-sample with respect to our study. It must not be read for grouping, count vocabulary construction, threshold tuning, model selection or gradient updates.
- The token ID rank/remap and `UNK` definition stay fixed from shard000001's `counts.npy`, exactly the same mapping used to train each arm.
- The original 600-window validation on `fineweb_val_000000.bin` is retained for cross-study comparison; the new independent shard CE is the **primary final test**.

## Selection and fresh training

- Select at most **two candidate** final learning rules by the **mean** original-validation CE on all three completed independent V50k seeds 41,73,109; precommit rule: include (1) best novel dynamic-group/Ember or projected-Ember method, and (2) strongest same-data fixed contextual group method if one exists.
- Also include a **strong ungrouped head optimizer comparator**, chosen only by the *original validation* data; and the upstream `adam_dynamic` baseline. Do not select hyperparameters on the new independent test shard.
- New training seeds **809,811,821**, held out from all previous method-selection runs.
- Same V50k width64 two-layer LM, 800 steps, batch32×L64×acc4, identical public training prefix, lr/cadence inherited unchanged from the selected runs.
- Train each selected method from scratch under exactly the same training script and data. Evaluate both validation sources at the same fixed endpoint, no best checkpoint selection.
- Report paired per-seed independent-test CE, target-frequency bins, optimizer state memory, group KMeans/preprocessing wall time and total training compute. Include all failures without selecting a replacement seed.

## Inference and novelty

The original method and our adaptations use a standard full-softmax output table after folding shared parameters. This is an algorithmic training-time study, not a claim about runtime-efficient inference. A positive independent V50k test would support a genuine small-model generalization result, but not replace a >100M parameter benchmark with realistic batch size.

The original author's published-in-repository contribution is the dynamic group-sharing method itself: credit it explicitly. The confirmation target is a **substantively distinct extension over modern head optimizers and strong same-data static semantic groups**, not another reproduction of the author's frequency law.
