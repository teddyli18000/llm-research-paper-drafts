# 04｜Identity and step-by-step pointer heads for k-hop fact chains

对应原文：[PAPER.md](../../04-attention-chain-following-heads/PAPER.md)，[RESULTS.md](../../04-attention-chain-following-heads/RESULTS.md)。

## 原作者主张

把 key 设为上一 token 的、与 query 使用同权重投影的表示，**从初始化起精确匹配“过去同一 token 后面的位置”**，作者明确承认此部分来自 Memory Mosaics。进一步保留独立 pointer 向量：每层查询到下一跳，就把找到的 token embedding 传给下一层，配合 no-backtrack penalty。合成任务中 2–6 hops、8–96 对键值可达接近 1.0 accuracy，普通 attention / KV shifting / n-gram heads 在作者短训练中常停留在接近 chance。模型很小，文本端只是 2k 词表和有限步数。

## 有关先例

1. **Memory Mosaics (2024)**：关联记忆的“同一 token 匹配并读后继”机制不是作者发明。[arXiv](https://arxiv.org/abs/2405.06394)
2. **Memory Mosaics at scale，NeurIPS 2025 Oral**：已做 10B 级别、真实数据，展示相关单元不是只有 toy 可用。[arXiv](https://arxiv.org/abs/2507.03285)
3. **Sanford, Hsu & Telgarsky, ICML 2024**：研究多跳 induction 的对数深度构造，并有模型学习多跳/倍增的实证。**不能宣称自此首次发现可用 attention 做多跳**。[PMLR](https://proceedings.mlr.press/v235/sanford24a.html)
4. **同仓 #05**：从一层一跳升级为倍增；二者主张彼此高度重叠，独立拆稿有 novelty / self-overlap 风险。

## 可能真正新增的点

**在标准 Transformer 中，把 Memory Mosaics identity head 改造成可传播的独立 pointer stream，并加入阻止回跳的选择机制；合成多跳监督只在终点时缩短训练冷启动。** 但这需要与直接移植 Memory Mosaics / 预置指针状态等更强对照明确区分。

## 论文评估

可写为透明、可解释的**架构归纳偏置论文**；当前独立主会竞争力偏弱。问题不在准确率 1.0，而是题目中序列语法和精确 token 重复本身允许强先验“硬编码正确算法”，真实长文与复杂引用是否受益尚无强证据。

## 最小反证实验

1. 变量名复用、同名遮蔽、自然文档多次提及、反事实噪声以及 value 与 key 不是原样 token 的情形。
2. **混合链长与跳数**，仅监督最终答案、测试训练外长度与 query 中指定 hop 数。
3. 一样的数据/训练预算下加入强训练方法：curriculum、intermediate supervision、真实 Memory Mosaics 和现有多跳构造。
4. 真实代码数据（函数调用链、变量流）和长上下文多跳问答；多个 seed。
5. 与 #05 合并陈述“一跳指针→倍增”，避免写两篇基本都在同一 synthetic k-hop task 上的论文。

## 拍板

**身份匹配原理非原创；pointer stream 及 no-backtrack 可能是具体创新。** 优先级：**第四（建议与 05 合并）**。
