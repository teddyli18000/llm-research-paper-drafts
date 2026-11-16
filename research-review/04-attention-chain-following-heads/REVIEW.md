# 04｜逐层跟随事实链的 Pointer Attention

原始材料：[PAPER.md](../../04-attention-chain-following-heads/PAPER.md) · [RESULTS.md](../../04-attention-chain-following-heads/RESULTS.md)

## 方法与结果

模型把相同 token 的历史出现位置作为检索入口，取得其后继 token。作者以 [Memory Mosaics](https://arxiv.org/abs/2405.06394) 的 identity-match head 为起点，增加独立 pointer stream：每层更新指针，沿事实链前进一步，并通过 no-backtrack 约束改善重复访问问题。

原作者在 2–6 hop、8–96 对关系的合成任务上报告接近满分的检索准确率，普通 attention / KV shifting / n-gram heads 在相同短训练预算下表现较弱。Fork 上 [Actions 复现](../REPRODUCTION.md) 得到 4-hop 中 **0.330 → 0.996**。

## 参考研究及创新

- [Memory Mosaics](https://arxiv.org/abs/2405.06394)：identity-match 与后继检索机制。
- [Memory Mosaics at scale](https://arxiv.org/abs/2507.03285)：大规模语言模型实现。
- [Sanford, Hsu & Telgarsky, ICML 2024](https://proceedings.mlr.press/v235/sanford24a.html)：Transformer 的多跳 induction 能力及深度效率。

**本研究的技术贡献集中在**把 identity-based recall 构造成可逐层传递、能够端到端训练的 pointer stream，以及 no-backtrack 对多跳稳定性的作用。这是针对多跳学习动力学的具体架构设计。

## 下一组实验

1. 随机化事实存储顺序、含糊/重复 key、同名遮蔽和干扰事实。
2. 在训练外 hop 数、混合链长和自然代码调用链上测试。
3. 与课程学习、中间监督、已有 pointer/induction 方法同训练预算比较。
4. 将 04 的逐跳指针与 05 的倍增指针放进同一模型设计与分析框架。

**论文方向：**具有结构化多跳归纳偏置的 Attention Head，关注初始化、可训练性和泛化。
