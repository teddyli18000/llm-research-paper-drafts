# 01｜深入研究：训练期间动态共享输出层向量

日期：2026-11-16。源码：[PAPER.md](../../01-rare-words-shared-output-vectors/PAPER.md) · [grid.py](../../01-rare-words-shared-output-vectors/grid.py) · [RESULTS.md](../../01-rare-words-shared-output-vectors/RESULTS.md)

## 1. 方法的数学结构

对词 `w` 的输出向量使用

`R_w=A_w+C_{g(w)}`

其中 `A_w` 是逐词残差，`C_{g(w)}` 是簇共享向量，`g(w)` 是聚类分配。每隔若干训练步，用当前等效向量 `R` 重新运行 K-means：

`g'=kmeans(R)`

`C'_k=mean(R_w | g'(w)=k)`

`A'_w=R_w-C'_{g'(w)}`

由构造可得 `R'_w=R_w`。因此重分组这个操作保持输出层函数不变，而后续梯度更新会经过新簇的共享参数传播。这种**训练中持续变化的共享结构与函数保持重参数化**是本题的关键方法设计。

## 2. 相关工作逐项对照

| 论文 | 已完成的研究 | 与 01 的方法差异 |
|---|---|---|
| [Botha & Blunsom, ICML 2014](https://proceedings.mlr.press/v32/botha14.html) | 将形态学成分与词向量共享用于长尾词学习 | 形态构成分组，而 01 按训练中的输出向量重分组 |
| [LightRNN, 2016](https://arxiv.org/abs/1610.09893) | word-to-cell 动态分配、行列共享参数 | 通过共享的行列概率表示词，01 保留逐词残差并在切换时保持输出行 |
| [Soft Weight-Sharing, ICLR 2017](https://arxiv.org/abs/1702.04008) | 训练中的软权重簇与可学习共享参数 | 聚类正则/压缩，与精确保留输出 head 函数的重分组不同 |
| [GroupReduce, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html) | 对语言模型输出 softmax 矩阵做频率加权组内低秩分解，并迭代调整组分配 | 后处理压缩优化，01 在梯度训练中周期性重分组并优化长尾词 |
| [Chen & Gimpel, NAACL 2018](https://aclanthology.org/N18-2116/) | 学习文本分类的簇嵌入与个性化调整 | 应用对象和共享目标不同 |
| [Adaptive Gradient Gating, ACL 2022](https://aclanthology.org/2022.acl-long.3/) | 分析稀有 token 的目标/非目标梯度与表示退化 | 直接调梯度，01 通过动态参数共享改变更新 |
| [Syntactic Smoothing, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.344/) | 句法相关 token 共享训练信号 | 句法先验与目标函数修改，01 依据模型当前输出几何自动生成簇 |
| [LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/) | 对预训练 LM head 的输出向量做 K-means 并构建簇级预测表示 | 在预训练后替换表示，01 在 LM 训练期间保持原 token logits 并重分组 |

这些参考覆盖了**词级共享、动态分簇、输出层聚类**的不同部分。完整的研究问题则是：**函数保持的在线聚类是否能持续改变梯度耦合，改善长尾词学习？**

## 3. 源码揭示的研究变量

### Adam 状态与重分组

原论文的伪代码包含 `reset_adam_state(self.C)`，实际 `grid.py` 的 `Model.recluster()` 更新簇 ID、`C` 和 `A` 后，优化器保留 `C` 过去的动量与二阶状态。

这意味着实验实现实际上在研究“动态共享结构 + Adam 历史状态”这一组合。可以做四种可比实验：保留 `C` 状态、仅重置 `C`、重置 `C` 与 `A`、按新组映射状态。每组对比重聚类时 `max|R'-R|`、训练曲线和按频率分桶的 token loss，即可分解贡献来源。

### 聚类初始化

论文伪代码使用上一轮的分组初始化；`grid.py` 每轮 K-means 用固定种子随机选初始中心，迭代十轮。这提供另一组干净对照：固定随机初始中心、上一轮中心 warm-start、一次性聚类、不再重分组。

### 梯度解释

交叉熵对输出行的梯度为

`∂L/∂R_w=(p_w-1[w=y])h`

频率不均衡主要体现在正样本出现次数和负样本背景更新的差异；共享簇在正、负样本上同时改变梯度耦合。应报告正目标 token、非目标 token 与频率 bucket 的分开贡献。

## 4. 原作者实验透露的规律

6k 词表下优化过 epsilon 的基线 loss **4.9931**，动态共享方法 **4.9720**。

固定 token 预算的 batch 实验（batch 16/32/64/128），作者报告增益约 **0.025/0.021/0.011/0.007**。这个梯度变化提示新的研究变量：**总 token 固定、优化步数、簇更新次数和平均组内正样本次数**之间的关系。可以进一步建立适用条件的定量规律，而非只报告单点增益。

## 5. 下一个实验阶段

1. **MOMENTS**：按相同 seeds 和重分组间隔做四类 Adam 状态实验，记录簇漂移和 loss。
2. **CLUSTER**：random、frequency、morpheme、POS、固定语义邻域、matched warm-start 的静态/动态 K-means 同配置比较。
3. **BATCH × RARITY**：固定 token 预算，改变有效 batch 与更新次数，分析正样本曝光和类簇质量。
4. **SCALE**：在标准 tokenizer/full-vocabulary、较大模型与真实数据上比较目标词分桶收益、吞吐和内存；推理前后检验折叠等价。
5. **THEORY**：建模簇中心梯度累积与 Adam moments 对低频词向量收敛速度的影响。

## 研究判断

**这是一条有明确方法构造与可验证机制的研究线。** 它与输出词向量共享、动态分组和压缩工作有联系；重点可放在函数保持、优化状态与稀有词信号传播三者的联合作用。成果可以发展为训练优化方法论文或长尾词学习机制论文。


---

## 6. November 17 full-text and mathematical correction

The starting online shared-centroid method \(R=U+AC\), function-preserving K-means refresh, and gradient sharing are already the upstream author's method. They are not an independent novelty claim.

The normalized flat-head projector

\[
P=AD^{-1}A^\top,\qquad G'=(I+\alpha P)G
\]

has an exact fixed-group SGD parameterization \(W=U+\sqrt{\alpha}AD^{-1/2}C\). Same-LR SGD produces exactly the same output-matrix update. The 25-step nonlinear test passes at 2.22e-15 maximum error. Therefore the projector is not a new SGD geometry.

The closest additional mathematical prior art is Hilfiker et al. (2026), *Strictly Proper Scoring Rules for Node Selection Tasks on Graphs*. For a disjoint union of within-class cliques, its graph-diffusion heat kernel converges to the same component-mean projector \(P\) in probability coordinates. The logit gradient of that proper score still contains \(J_{\rm softmax}\), so it is not the same optimizer-side logit update. Conceptizer (2026) and Learning Concepts, Not Tokens (EMNLP 2026) are also close language-model objective neighbors: both reward concept/set probability mass, but their token gradients are probability-weighted within a concept rather than uniform class-mean residuals.

### Updated empirical status

- Direct projection beats the exact normalized-centroid parameterization under Ember and Adam at batch32, but is near-tied at batch128. The direct advantage is concentrated in zero/low-exposure rows. A matched LR sweep is required before attributing it to adaptive-state geometry.
- Context groups through absolute step250, followed by online K-means, is the selected batch32 schedule candidate under the preregistered rule. Batch128 remains context-only at 100 steps.
- Raw second-moment routing fails because it overshoots.
- The first uncentered one-sided \(Pp\)/\(-Py\) ablation was confounded by the vocabulary-uniform mode. Centered one-sided and uniform-only controls are running.

### Revised research judgment

This is no longer a claim of a new dynamic class-sharing mechanism or new SGD preconditioner. The remaining candidate contribution is a practical flat-head adaptive-optimizer effect: whether direct projection produces more useful low-exposure token updates than the exact normalized-centroid parameterization, under matched LR and modern baselines.

A paper-level claim requires the centered residual study, LR sweep, graph-diffusion proper-score comparison, output-only Adam epsilon, projected RowNorm/MuonIO, and validation on an untouched shard and larger model. Until those comparisons hold, the correct status is **promising narrow empirical hypothesis; novelty not yet established**.

