# 05｜Attention 中的 Pointer Doubling

原始材料：[PAPER.md](../../05-attention-pointer-doubling/PAPER.md) · [RESULTS.md](../../05-attention-pointer-doubling/RESULTS.md)

## 方法与结果

在 04 的逐层一跳 Pointer 基础上，方法将上一层计算出的跳跃指针再组合，实现随层数指数增长的跳跃范围。增加 doubled no-backtrack 和可门控的第二指针，处理不同长度的链。

原作者报告 **3 层 7 跳、5 层 31 跳**。Fork 的独立 [Actions 复现](../REPRODUCTION.md) 中，15-hop、4 层的对比准确率为 **0.170 → 0.991**。

## 参考研究与新增问题

- **Binary lifting / pointer jumping**：经典并行算法，提供指针倍增构造。
- [Sanford, Hsu & Telgarsky, ICML 2024](https://proceedings.mlr.press/v235/sanford24a.html)：Transformer 对数深度多跳构造及 learned attention doubling 现象。
- [Memory Mosaics](https://arxiv.org/abs/2405.06394)：内容匹配的基础检索结构。
- 同仓 04：提供每层传递一跳的 pointer stream。

**可研究的具体贡献**：把 pointer jumping 组织成明确的可训练 head、通过 no-backtrack 与 gated pointer 改善端点监督的收敛，并分析层数、跳数、查询编码和表示之间的关系。理论上的对数深度性质应从 Sanford 2024 的研究起点继续发展。

## 下一组实验

1. 直接复现和对照 Sanford 2024 的实验设置，比较训练复杂度与长度外推。
2. 测试单模型混合 hop 数、查询中指定跳数以及任意链方向。
3. 处理重复节点、噪声链和不同存储顺序，刻画定位能力。
4. 将 04 的一跳机制与 05 的倍增机制整合成统一实验与理论叙事。

**论文方向：**可训练的符号指针运算与 Attention 的并行多跳计算。
