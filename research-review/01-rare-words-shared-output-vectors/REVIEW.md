# 01｜动态聚类共享语言模型输出向量

原始材料：[PAPER.md](../../01-rare-words-shared-output-vectors/PAPER.md) · [RESULTS.md](../../01-rare-words-shared-output-vectors/RESULTS.md) · [深入研究](DEEP_NOVELTY_AUDIT.md)

## 研究问题与方法

稀有 token 的输出向量受到的正样本学习信号较少。方法将输出向量表示为 `R_w=A_w+C_{g(w)}`：每词私有向量 `A_w` 加所属簇共享向量 `C_{g(w)}`。训练期间按当前输出向量做 K-means 并重分簇，再调整私有分量以保持 `R_w` 不变。共享分量汇聚同组词的训练梯度，推理时可折叠回普通输出矩阵。

原作者报告：词表 6,000 时，与优化过输出层 Adam epsilon 的基线相比，验证 loss **4.9931 → 4.9720**；random/frequency 固定分组的收益很小。此结果来自原仓库训练记录。

## 研究贡献与已有文献

- [Botha & Blunsom, ICML 2014](https://proceedings.mlr.press/v32/botha14.html)：形态学构成的共享词向量，提供跨词学习的理论与实验先例。
- [LightRNN, 2016](https://arxiv.org/abs/1610.09893)：训练中重分配共享词表示，关注紧凑结构。
- [Soft Weight-Sharing, ICLR 2017](https://arxiv.org/abs/1702.04008)：动态权重簇和共享优化。
- [GroupReduce, NeurIPS 2018](https://papers.neurips.cc/)：直接研究输出 softmax 矩阵的分组、迭代调整与组内低秩近似，目标是压缩。
- [Syntactic Smoothing, EMNLP 2024](https://arxiv.org/abs/2410.11462)：以句法关系传递学习信号。
- [LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/)：对预训练 LM 输出层 token 向量进行 K-means 聚类，构建簇级表示。

**值得研究的新组合**是 *输出向量的在线自组织分簇、函数保持式重分配、共享梯度优化以及零额外推理成本*。这些参考工作分别覆盖了相关组件，研究目标与完整训练过程各不相同。

## 关键技术问题

原论文的伪代码重分组后执行 `reset_adam_state(self.C)`，实际 `grid.py` 没有该调用，且每次 K-means 用固定随机种子初始化而非上次分组。对这一差异做 Adam 状态消融，能够揭示收益源于跨词共享、参数重设，还是二者的联合作用。

作者报告相同 token 预算下，batch 16/32/64/128 的 loss 收益约 **0.025/0.021/0.011/0.007**；这提示全局 batch、更新次数和簇更新频率值得进一步建模。

## 下一组实验

1. 保留、重置、迁移 Adam moments 的同配置对比，记录函数保持精度和重聚类后学习轨迹。
2. 与静态语义、形态学、词性分组对比，分离“组的质量”和“重分组过程”。
3. 增大有效 batch 和训练长度，分析不同频率 bucket 的收益曲线。
4. 扩展到标准词表与较大模型，报告验证 loss、训练成本、低频 token 泛化。

**论文方向：**语言模型输出层的函数保持动态重参数化与长尾词优化；创新的重点在训练机制及其可解释的优化效果。
