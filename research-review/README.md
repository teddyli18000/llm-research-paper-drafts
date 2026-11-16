# 六篇 LLM 研究草稿：创新点、相关工作与研究进展

研究对象：[原仓库六篇论文](../README.md)；学术审查日期：2026-11-16。每篇按 **问题与贡献 → 已有研究 → 方法差异 → 关键实验** 进行分析。

## 六个方向

| 方向 | 研究问题 | 主要研究贡献与文献关系 | 进展 |
|---|---|---|---|
| [01 动态输出层共享](01-rare-words-shared-output-vectors/REVIEW.md) | 低频词输出向量如何利用其他词的学习信号 | **函数保持的在线输出行重分簇 + 逐词 residual + 共享梯度**。GroupReduce、LightRNN 和 LENS 分别研究相关聚类和共享机制，训练过程不同 | 原作者小规模训练结果；[深入审查](01-rare-words-shared-output-vectors/DEEP_NOVELTY_AUDIT.md) |
| [02 Muon Hypergradient](02-muon-hypergradient-learning-rate/REVIEW.md) | 如何减少 Muon 学习率搜索 | Muon 正交更新专用的学习率适应、early-hold 和宽度规律；与 Hypergradient、MuonMax/Momo 比较 | 原作者 45 组起始 LR 实验 |
| [04 多跳 Pointer Head](04-attention-chain-following-heads/REVIEW.md) | 无中间监督时如何按层稳定跟随事实链 | 从 Memory Mosaics identity retrieval 延伸到逐层 pointer stream 和 no-backtrack | 合成任务 headline 已复现 |
| [05 Pointer Doubling](05-attention-pointer-doubling/REVIEW.md) | 如何通过 Attention 并行完成多跳 | 基于 pointer jumping 的可训练 head；与 Sanford et al. 2024 的对数深度构造比较 | 合成任务 headline 已复现 |
| [06 Word/Pair Value Embedding](06-value-embeddings-word-vs-pair/REVIEW.md) | 组合向量什么时候值得额外参数 | 曝光频率与表示粒度交叉点的定量规律；与 Engram、X-GRAM 的稀疏记忆研究相联系 | 原作者小规模分桶分析 |

## 专题①：动态输出层共享

[深入对照报告](01-rare-words-shared-output-vectors/DEEP_NOVELTY_AUDIT.md) · [①综合文献研究](NOVELTY_DEEP_DIVE_2026-11-16.md)

原论文使用 `R_w=A_w+C_{g(w)}`，通过按模型当前输出行重分组，使训练中的共享结构持续更新，且在重分组瞬间保持 `R_w` 不变。

最近邻 GroupReduce 迭代优化分组低秩输出矩阵，LightRNN 训练中重分配共享词表示，LENS 对输出 head 做 K-means。**这些是方法的参考基础；在线、函数保持的重参数化及其对长尾优化的作用是当前值得验证的组合。**

研究重点是优化器状态如何处理、簇更新频率与 global batch 之间的关系，以及长尾损失收益是否随训练规模变化。PAPER.md 伪代码写了 Adam 状态重置，实际 `grid.py` 没有，此差异值得直接做消融。

## 研究资料与运行记录

- [04/05 原始 headline 复现与 CI 故障审计](REPRODUCTION.md)

**实验范围**：04/05 是 headline 复现；01/02/06 的数值来自各自原论文。

## 主要参考文献

**① 输出词向量**：[Botha & Blunsom 2014](https://proceedings.mlr.press/v32/botha14.html) · [LightRNN 2016](https://arxiv.org/abs/1610.09893) · [Soft Weight-Sharing 2017](https://arxiv.org/abs/1702.04008) · [GroupReduce 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html) · [Syntactic Smoothing 2024](https://arxiv.org/abs/2410.11462) · [LENS 2025](https://aclanthology.org/2025.acl-long.930/)


**其他方向**：[Baydin et al. 2018](https://mlanthology.org/iclr/2018/baydin2018iclr-online/) · [MuonMax/Momo 2026](https://proceedings.mlr.press/v306/crawshaw26a.html) · [Memory Mosaics](https://arxiv.org/abs/2405.06394) · [Sanford et al. 2024](https://proceedings.mlr.press/v235/sanford24a.html) · [Engram](https://arxiv.org/abs/2601.07372) · [X-GRAM](https://arxiv.org/abs/2604.21724)
