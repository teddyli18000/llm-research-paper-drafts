# 2026-11-16｜01 深入撞车报告

**范围**：检查 fork 中 01 原论文、实验记录和关键实现；独立核对 2014–2026 年经典与最新先行论文（包括借助学术全文阅读工具查看具体方法公式）。**重点是 novel claim 是否被已有方法覆盖，而不是作者提供的 toy accuracy 是否好看。**

- [01 深入审查与文献矩阵](01-rare-words-shared-output-vectors/DEEP_NOVELTY_AUDIT.md)
- [此前独立的 04/05 headline 复现实测](REPRODUCTION.md)

## 最新裁决

| 方向 | 原方法是否整体被撞掉？ | 真正可能新在哪里？ | 当前最高风险 | 投入建议 |
|---|---|---|---|---|
| 01（稀有输出词共享） | **没有发现整个方法的同构前作**；但“输出矩阵动态分簇”已有 NeurIPS 2018 GroupReduce，K-means head 已有 ACL 2025 LENS | 在线按等效输出行重分簇、**函数保持**、保留逐词 residual、共享梯度 | PAPER 伪代码说重置 C 的 Adam state；实际 `grid.py` 没有 reset；极大 batch 外推不利 | **优先做 cheap 的 Adam state 消融与近邻基线** |

## 非同一件事：先例已经做过什么

- **01 GroupReduce** 已迭代 refine 语言模型输出层 softmax 的 word clusters，但工作重点是压缩与分组低秩近似；不是有逐词残差的、预训练期间严格无损的动态重参数化。这是“有组件撞车，但可能有新组合”的典型案例。
- **01 LENS** 对 output-head tokens 做 K-means，目的却是生成 cluster-level lexical text representation，不保留原 full-vocabulary logits。

## 两个必须优先核查的源码缺口

1. 01 `PAPER.md` 的 `reset_adam_state(self.C)` 不存在于训练用 `grid.py`；实际聚类也没有用前一轮标签作为 K-means 初始中心。这直接影响作者的机制表述和结果可归因性。

## 建议的投资门槛

- **先跑成本低的可推翻实验**：01 四种 moments 策略 + 静态/动态 cluster + 真实语义组。
- 若 **01** 在 reset moments 后效果消失、在稍大 batch 下趋零，则把它降为“小模型优化器动力学分析”，不扩大训练。
- 两者完成最小关键实验后，再考虑昂贵的 125M+ 规模；**先规模后证伪会浪费算力**。

## 方法与证据级别

**已确认**：GitHub 原始代码里存在上述伪代码与实现、强基线名称与结构不一致；所列多篇近邻有相应明确公式或算法。  
**未完成**：01 的独立重跑、真实规模、可靠的多 seed；目前 **不能认定该方向已经达到了顶会论文门槛**。

## 核心引文（优先原始会议、arXiv 或研究机构官方页面）

### 01
- [Botha & Blunsom, Compositional Morphology, ICML 2014](https://proceedings.mlr.press/v32/botha14.html)
- [LightRNN, 2016](https://arxiv.org/abs/1610.09893)
- [GroupReduce, NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html)
- [Chen & Gimpel, Cluster Embeddings, NAACL 2018](https://aclanthology.org/N18-2116/)
- [Yu et al., Adaptive Gradient Gating, ACL 2022](https://aclanthology.org/2022.acl-long.3/)
- [Diehl Martinez et al., Syntactic Smoothing, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.344/)
- [Lei et al., LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/)

