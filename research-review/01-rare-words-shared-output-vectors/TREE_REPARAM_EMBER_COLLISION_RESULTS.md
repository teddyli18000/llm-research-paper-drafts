# Paper 01 — Class-projected Ember is effectively tied with Tree Loss-style reparameterized Ember

**[GitHub Actions #44029716200](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44029716200) succeeded 3/3, all 15 precommitted trainings.** All effective output-head logits are identical immediately after initialization between reparameterized and direct heads, with an explicit numerical assertion inside each run. V6k/width32/2layers/batch32/L64/400steps, K64 fixed train-context classes, α2, new seeds373/379/383, official Ember head LR=.05/beta2=.95 or plain head Adam LR=.05/eps1e−5. Full [protocol](TREE_REPARAM_EMBER_COLLISION_PROTOCOL.md).

## Full paired CE, lower better

| Head update | Seed373 | Seed379 | Seed383 | 3-seed mean CE |
|---|---:|---:|---:|---:|
| Ordinary head-only Ember | 5.0016 | 5.0006 | 5.0007 | **5.00097** |
| **Direct class projector + Ember** | 4.9700 | **4.9598** | 4.9693 | **4.96637** |
| **Equivalent-forward linear class reparameterization + Ember** | 4.9747 | 4.9606 | **4.9660** | **4.96710** |
| Direct class projector + Adam | 4.9859 | 4.9802 | 4.9852 | 4.98377 |
| Linear class reparameterization + Adam | **4.9835** | 4.9804 | 4.9860 | **4.98330** |

Ember direct wins versus reparam on seeds373 and379 (+.0047,+.0008 CE) but loses on seed383 (−.0033); average nominal +.00073 CE **cannot be treated as a real advantage**. Adam direct vs reparam mean difference −.00047 CE, also near-tied. Both head-coordinate methods greatly improve over ordinary Ember on the same seeds.

## Mathematically correct novelty boundary

Our fixed normalized projector `G'=G+αPG`, `P²=P`, corresponds **exactly under vanilla SGD** to a linear reparameterization `R=BU`, with `B=I+(sqrt(1+α)−1)P`; [full proof/source comparison](TREE_LOSS_PRECONDITIONER_COLLISION.md). [AISTATS 2022 Tree Loss](https://proceedings.mlr.press/v151/wang22d.html) already used structurally shared per-class and ancestor parameters under the *same CE objective* for improved generalization. The original paper owner also already used online shared centroid parameters, inducing unnormalized class gradient transfer.

**The empirical test shows that using an adaptive optimizer on directly preconditioned gradients does not give a meaningful accuracy edge over the old-style class-coordinate reparameterization in this static-context V6k regime.** While the two algorithms need not be exactly equivalent under Ember or Adam, the observed differences are too small and change sign across seeds.

The new direct one-table rule remains simpler for a normal output head: no forward-time class mixing or explicit optimizer state reparametrization; possible runtime/memory differences must be separately benchmarked at real scale. On accuracy alone, this **substantially weakens** any claim that we have already found a compelling *independent* ICML/NeurIPS-grade method over published class-sharing ideas. Creditable independent work may still come from a principled *reliability-adaptive* class-group rule that beats these controls on larger LM pretraining, or a significant implementation efficiency result validated on a realistic GPU kernel.

Do not interpret the existing 50k projector gain versus **ungrouped Ember** as proof of algorithmic originality against **Tree Loss** or the author's **group-centroid** approach. Both are stronger conceptual baselines.

The next direct method-collision trials compare label-smoothed and concept-class training objectives, with original source citations: [semantic smoothing](SEMANTIC_SMOOTHING_COLLISION_PROTOCOL.md), [concept-class loss](CONCEPT_CLASS_LOSS_COLLISION_PROTOCOL.md).
