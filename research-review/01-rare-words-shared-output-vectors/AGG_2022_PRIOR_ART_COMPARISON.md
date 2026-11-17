# ACL 2022 AGG — prior-art comparison for Paper 01

## Bibliographic record

Sangwon Yu, Jongyoon Song, Heeseung Kim, Seong-min Lee, Woo-Jong Ryu, and Sungroh Yoon. **Rare Tokens Degenerate All Tokens: Improving Neural Text Generation via Adaptive Gradient Gating for Rare Token Embeddings.** ACL 2022, pp. 29–45. DOI: 10.18653/v1/2022.acl-long.3.

Primary sources:
- ACL Anthology: https://aclanthology.org/2022.acl-long.3/
- Full paper: https://aclanthology.org/2022.acl-long.3.pdf
- Reimplementation: https://github.com/jet981217/AGG_pytorch

## What AGG actually does

AGG is a direct and important predecessor in rare-token output-embedding optimization. Its dynamic grouping is based on recent token appearance frequency, not semantic similarity:

\[
a_i=\sum_{t=1}^{K}m_{t,i},\qquad
a_i/K<\alpha\Rightarrow i\in V_r.
\]

The paper decomposes the output-row gradient for a rare token into three terms:

- (a) target-token attraction;
- (b) repulsion from hidden states whose targets are non-rare;
- (c) repulsion from hidden states whose targets are rare.

The paper identifies term (b) as the main cause of embedding degeneration. It uses two frequency-dependent gate vectors to attenuate terms (b) and (c) for rare and very-rare output rows. Its equations (6)–(10) implement this through detached copies and separate logits, preserving the forward parameter values while changing the embedding-gradient path.

Reported WikiText-103 results with a GPT-2-medium architecture, 260K vocabulary and up to 50K training iterations:

| Metric | MLE | AGG |
|---|---:|---:|
| Rare-token perplexity | 438.67 | 75.39 |
| Total perplexity | 15.51 | 15.51 |
| Embedding isotropy I(W) | 0.377 | 0.813 |
| Rare-token unique predictions | 91 | 345 |

The paper also reports WMT14 BLEU improvements of 1.40 and 1.41 for Transformer base and big. Dynamic grouping slightly outperforms static AGG in its ablation (total PPL 15.51 vs 15.55).

## Exact comparison with the original Paper 01 method

| Dimension | ACL 2022 AGG | Paper 01 shared output vectors |
|---|---|---|
| Group membership | Recent appearance frequency | K-means over current output vectors |
| Group meaning | Rare / very rare | Similar output-row geometry and learned usage |
| Parameterization | Original output table; no shared group vector | Own row plus group-shared centroid |
| Gradient operation | Attenuates selected harmful gradient components for rare rows | Adds the group-aggregated gradient to every row in the group |
| Forward values | Preserved by detached parameter copies | Preserved exactly when regrouping by moving the group mean between own and shared parts |
| Main target | Rare-token representation degeneration and frequency bias | Sparse output-row learning through positive transfer from related tokens |

## Novelty consequence

AGG means that these claims are already established and cannot be presented as new:

1. Rare output embeddings have distinctive training dynamics.
2. Dynamically identifying rare tokens during training is useful.
3. Modifying only the output-embedding gradient can improve rare-token behavior.
4. A forward-preserving gradient intervention can improve rare-token generation.

AGG does **not** implement semantic output-row clustering, a shared group parameter, or cross-token gradient transfer. It therefore does not reproduce the Paper 01 mechanism. The defensible distinction is the learned semantic group structure and the transfer of group-member gradients through a shared output component.

The overlap is still close enough that a paper built around “dynamic rare-token grouping and gradient modification” would be poorly positioned without AGG. The research question must be narrower: does model-derived semantic grouping provide useful cross-token transfer beyond frequency-aware gradient gating?

## Required experiments

1. Reproduce an AGG-style frequency gate in the same V6k/width-32 training harness. Use the original paper's recent-frequency memory and its separate rare/very-rare gates; report the exact adaptation if the original tokenizer/task differs.
2. Compare AGG, shared-output grouping, and the best optimizer baseline on identical seeds, tokens, and validation frequency buckets.
3. Measure both rare-target CE/perplexity and embedding isotropy/anisotropy. Overall CE alone does not test the degeneration mechanism that AGG targets.
4. Test AGG + shared-output grouping as a compatibility arm. The AGG paper explicitly reports compatibility with other text-generation methods; an additive gain would be a meaningful result.
5. Inspect whether the shared-gradient gain is concentrated in the 10–99 exposure band already identified in this project, while AGG primarily changes the harmful negative-gradient components for very rare rows.

## Bottom line for this research line

AGG is a strong adjacent ACL 2022 result, not a complete collision. It substantially narrows the novelty claim and creates a required baseline. The shared-vector method remains mechanistically distinct, but its value must be demonstrated as *semantic cross-token transfer beyond frequency-aware rare-row gradient gating*.
