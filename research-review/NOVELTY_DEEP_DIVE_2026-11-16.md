# 2026-11-16｜01 的深入研究：创新机制与最近邻文献

研究材料：Fork 的原始论文、代码、实验记录及 2014–2026 年最接近论文的方法正文。

- [01 动态输出层共享：方法、文献、源码与后续实验](01-rare-words-shared-output-vectors/DEEP_NOVELTY_AUDIT.md)
- [04/05 原论文关键实验复现](REPRODUCTION.md)

## 技术定位

| 方向 | 研究核心 | 与前作的具体差异 | 已有证据 | 下一组实验 |
|---|---|---|---|---|
| **01** | 动态输出层聚类共享梯度 | LightRNN 动态共享分配；GroupReduce 对输出层迭代聚类做压缩；LENS 对 output head 聚类。**本题是在训练中按等效输出行重分组，并通过逐词 residual 保持 logits** | 原作者 FineWeb 小模型 loss 改善 0.016–0.039 等，batch 增大时收益缩小 | Adam moments 控制、静态/动态组对照、频率分桶与 batch scaling |

## 01：创新机制

表示 `R_w=A_w+C_{g(w)}`，定期对当前输出向量重新聚类，并同时调整 `A_w` 和 `C_g` 使 `R_w` 原位保持。该构造兼有可自适应的共享结构和不改变重分组时预测函数的性质，适合从优化动力学上解释为何低频词受益。

[GroupReduce, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html) 对输出矩阵做频率加权低秩分组及迭代 refinement，目标是压缩；[LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/) 对 pretrained head 做聚类替换。两者提供方法参考，但重分组时的训练函数和优化目标不同。

源码中出现一个很有研究价值的变量：论文伪代码重置共享向量的 Adam 状态，训练程序保留该状态。正好可以用来研究**聚类切换、动量记忆和共享梯度之间的耦合**。

## 研究工作计划

**01：** moments 消融 → 分组构造对照 → 批量大小与词频共同变化 → 标准规模输出层训练。


**论文路线：**①偏训练优化机制与输出层重参数化。投稿论点应由对应实验数据决定。

## 参考文献

### 01

- [Botha & Blunsom 2014, ICML](https://proceedings.mlr.press/v32/botha14.html)
- [LightRNN 2016](https://arxiv.org/abs/1610.09893)
- [GroupReduce 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html)
- [Chen & Gimpel 2018](https://aclanthology.org/N18-2116/)
- [Adaptive Gradient Gating 2022](https://aclanthology.org/2022.acl-long.3/)
- [Syntactic Smoothing 2024](https://aclanthology.org/2024.emnlp-main.344/)
- [LENS 2025](https://aclanthology.org/2025.acl-long.930/)

