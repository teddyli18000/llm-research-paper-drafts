# 06｜Word 与 Pair Value Embedding 的训练频率规律

原始材料：[PAPER.md](../../06-value-embeddings-word-vs-pair/PAPER.md) · [RESULTS.md](../../06-value-embeddings-word-vs-pair/RESULTS.md)

## 研究问题与结果

在注意力 value 分支增加按 token 或 token-pair 查询的向量表，比较两者在同参数、同算力等训练预算下的表现。原作者按 token pair 出现次数分桶，报告在其实验配置中，**约 100 次曝光**附近 pair 表的表现开始超过 word 表；实验模型最大约 220 万参数。

这个观察触及一个有用的建模问题：参数应该分配给高频可复用特征，还是细粒度但样本更稀疏的组合特征？

## 研究背景与可能贡献

- [Engram, 2026](https://arxiv.org/abs/2601.07372)：大规模 n-gram 稀疏记忆结构，显示组合 lookup 的可用性。
- [X-GRAM, 2026](https://arxiv.org/abs/2604.21724)：频率感知 n-gram memory 与长尾样本不足的机制分析。

**值得深入的方向**是建立 unigram/bigram 表现交叉点的可预测定量模型，让曝光频率、碰撞率、向量维度、词表结构和训练预算共同解释性能。在这个问题上，准确的条件规律本身就是研究贡献。

## 下一组实验

1. 拟合交叉阈值与 `frequency × capacity × collisions` 的定量关系。
2. 换 tokenizer、词表、训练长度和随机种子做交叉验证。
3. 对齐表容量、参数量、内存带宽和训练速度，比较真实效率。
4. 在较大模型中与 Engram、X-GRAM 的代表性配置对比。

**论文方向：**稀疏组合记忆的频率—容量—数据效率定律。
