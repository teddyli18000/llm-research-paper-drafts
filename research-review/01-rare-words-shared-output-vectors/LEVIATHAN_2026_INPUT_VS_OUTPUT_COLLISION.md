# New 2026 rare-token architecture collision: Leviathan (NeurIPS 2026)

[Batley & Saha, Leviathan: Decoupling Input and Output Representations in Language Models](https://arxiv.org/abs/2601.22040), versions from Jan/May 2026. Full paper rechecked, not inferred from abstracts.

## Mechanism and exact collision

**Leviathan is input-side**, not an output-head gradient optimizer. Learned Embedding Vectorization replaces the standard input token lookup matrix with a differentiable continuous generator: a base-b factorization of token identities, shared seed/codebook components, a smooth B-spline expansion and a learned projection. Inputs with shared components borrow gradient signal. The model's **output head is an unchanged, separately trainable dense untied matrix**, not grouped/projected. The paper does not apply KMeans token groups or gradient mean projection to output rows.

Therefore it **does not disclose** our proposed `G'=(I+αA(AᵀA)⁻¹Aᵀ)G` output-head preconditioner. But it is **major current prior art** for any broad "improve rare tokens by sharing related token learning" claim, and directly challenges paper novelty if our framing ignores input/output distinctions.

## Large-scale benchmark evidence

| Backbone | Total parameters | Training tokens | Tokenizer/vocabulary |
|---|---:|---:|---|
| 200M | 209M | 4.2B | o200k_base, 200,376 |
| 400M | 409M | 8.4B | same |
| 1.2B | 1.218B | 24.1B | same |

Leviathan reports 1.2B validation perplexity about **9% lower** than tied baseline, **2.1× fewer tokens** to reach its final loss, and **81% lower perplexity in the rarest bucket**. That bucket is **1–9 occurrences in a 100M-token corpus-frequency counting sample**, not necessarily 1–9 target occurrences in the full 24.1B-token run. The fully untied naive-embedding comparator is reported at 400M, not at 1.2B. Paper identifies one-seed-per-configuration and corpus/tokenizer limits.

These are substantially larger-scale experiments than our 2-layer toy model, where the majority of parameters reside in input/output vocab tables. The proper future study should compare an output-head projector on the **same untied head** with and without a stronger input-side embedding treatment, and report frequency-tail changes. No implication that one method is better than the other follows from different experiments.

## Publication consequence

An independent Paper01 extension must be described as **training a standard untied output classifier through class-structured gradient projection**. It should not claim to invent learning-signal transfer between rare tokens, untied embeddings, compositional token parameterization, or rare-token performance gains generically. Leviathan specifically strengthens the case that model/backbone/token scale is required before an ICML/NeurIPS-level general LM claim.

Related other 2026 work on table-free input tokens: [Fixed Minimal Token Codes at 1.7B-Class Scale](https://arxiv.org/search/?query=%22Do+Language+Models+Need+a+Trainable+Input+Embedding+Table%22&searchtype=all) also leaves output head standard and untied, so it is a different intervention rather than algorithmic collision.
