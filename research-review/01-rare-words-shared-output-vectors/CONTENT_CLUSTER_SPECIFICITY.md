# Paper 01 — Is the benefit really from meaningful groups?

Date: 2026-11-16. This is the crucial counterfactual for the paper's primary mechanism.

## Study 1: control for arbitrary reparameterizations

[GitHub Actions run #43838718914 — success, with actual logs](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43838718914). All arms use V6000, width32, batch32, 400 steps, exact same reconstructed FineWeb data and three seeds, Adam epsilon 1e-5, equal group LR and K=64. Each group update preserves `R=U+AC` exactly at the instant of reassignment.

| Variant | Seed 0 CE | Seed 1 CE | Seed 2 CE | Mean gain against corresponding baseline |
|---|---:|---:|---:|---:|
| Unshared baseline (previous run) | 5.0139 | 5.0118 | 5.0091 | — |
| Original K-means learned groups (previous run) | **4.9937** | **4.9967** | **4.9925** | **+0.01730** |
| Reassign words randomly every 50 steps | 5.0122 | 5.0122 | 5.0106 | **-0.00007** |
| Keep exactly the same assignments, periodically recenter shared/private coordinates | 5.0125 | 5.0142 | 5.0110 | **-0.00097** |
| Warm-start K-means from current clusters | **4.9938** | **4.9990** | **4.9938** | **+0.01607** |

**Mechanism established by direct negative controls:** it is **not** merely the extra learnable shared vectors, arbitrary changing group labels, or repeated function-preserving reparameterization. Learned K-means geometry gives a consistent CE improvement relative to those controls. Warm-starting K-means works almost as well as the original fixed-random-seed K-means initialization.

## Study 2: genuinely semantic but externally pretrained static groups

[GitHub Actions #43838902626 — success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43838902626). GPT-2 small's publicly available tied output-head word vectors (the checkpoint key `wte.weight`, **not** `transformer.wte.weight`) were normalized, clustered into 64 groups with sklearn KMeans, and used **as fixed assignments from the beginning** of the same small language-model training. GPT-2 has extensive additional pretraining on WebText, so this is a *strong informative prior* and is **not compute-matched** to learning groups entirely from scratch.

| Variant | Seed 0 CE | Seed 1 CE | Seed 2 CE | Mean gain against unshared baseline |
|---|---:|---:|---:|---:|
| Unshared epsilon baseline | 5.0139 | 5.0118 | 5.0091 | — |
| Fully self-discovered K-means online | 4.9937 | 4.9967 | 4.9925 | **+0.01730** |
| Fixed groups from pretrained GPT-2 output head | **4.9848** | **4.9836** | **4.9750** | **+0.03047** |

Pretrained **static** semantic groups outperform the fully from-scratch dynamic scheme by **0.0089 / 0.0131 / 0.0175 CE**, mean **0.01317**. This directly falsifies a broad claim that "static groups cannot capture the benefit." Static **random/frequency** grouping does not help, whereas **strong fixed semantic grouping does**.

The more precise central claim is thus **group quality matters**, and online self-discovery is a way to obtain suitable groups **without an external pretrained teacher**. That can still be a useful method contribution, but it is distinct from the original paper's rhetorical static-vs-dynamic dichotomy.

## Follow-up needed

- A follow-up has been launched to test **pretrained semantic groups plus online re-clustering**, starting after 50 or 200 small-model training steps, to see if online adaptation adds value over a strong prior or instead destroys it.
- The exact relative benefit of a *compute-matched* semantic/POS grouping algorithm remains open; an external 124M-parameter GPT-2 teacher is a higher-compute competitor, not an equal-budget alternative.
- For stronger external validity, move beyond the sampled top-6000 GPT-2 tokens/UNK bucket and test a full 32k/50k tokenizer with a modern training regime.

## What to say in a paper

The method introduces a **data-dependent changing class-gradient coupling operator for flat-softmax output rows**, preserving output logits at each cluster update. Repeated meaningful clustering and learned-gradient sharing are necessary **relative to uninformative random/frequency labels and small-model self-generated one-shot groups**. However, a fixed semantically pretrained partition already works even better on these small-model tasks. The paper should give that result prominently; ignoring it would make the novelty story weaker, not stronger.
