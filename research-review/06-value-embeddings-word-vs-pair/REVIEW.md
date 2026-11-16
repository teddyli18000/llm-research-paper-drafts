# 06｜Value embeddings: unigram vs bigram and exposure frequency

对应原文：[PAPER.md](../../06-value-embeddings-word-vs-pair/PAPER.md)，[RESULTS.md](../../06-value-embeddings-word-vs-pair/RESULTS.md)。

## 原作者的主张

在注意力 value 中加入每词的 lookup vector，或为两个词的组合另存一条 lookup vector。作者按训练中词对出现次数分桶，发现**约 100 次**是词对向量在某些设定下追上单词向量的交叉区间；用部分词表训练的规则预测保留词表（3k）内的桶输赢。在固定 compute 下单词向量常好，固定参数量时扩大普通模型又可能更好。作者声称实验最大 2.2M 参数，150–1600 步，单 seed；长训练和大词表的直接证据有限。

## 先例 / 容易撞的部分

1. **Engram (2026)**：将 n-gram hashed lookup 做成大规模模型的独立稀疏记忆轴，报道 27B 规模、真实任务增益。**“大 n-gram 表改善模型”早已不是新发现**。[arXiv](https://arxiv.org/abs/2601.07372)
2. **X-GRAM (2026)**：明确提出**Zipfian under-training of the long tail**、频率感知 n-gram memory，并在 0.73B / 1.15B 模型比较。**本研究的主要解释因素“词对太少见，训练不足”已经被公开研究**。[arXiv](https://arxiv.org/abs/2604.21724)
3. **Hash Layers / token-routing 与 bigram keys** 等前例在原文 RELATED WORK 有列举；因此 unigram / bigram 本身不是新架构。

## 可能还有的独立贡献

**在相同额外 lookup 容量下，以训练曝光次数为条件，实测 unigram-vs-bigram 交叉点，并将此阈值转化成预先预测不同词表/训练日程结果的简化模型。** 这应是**分析型贡献**，不是声称发现新 memory 模块；固定的“100 次”大概率受 tokenizer、模型宽度、优化器、哈希冲突率和数据熵影响。

## 投稿判断

可以形成**负面/解释性研究**或更完整的 memory scaling empirical law 工作，但目前 1 个 held-out vocab、单 seed、极小模型不足以建立一个跨模型普适定律。与 X-GRAM 的频次解释高度接近，需要提出 X-GRAM 没有给出的、能跨设置预测的量化规律。

## 最小决定性实验

1. 以 **exposure frequency、table collisions、词对熵、占用率、vector dimension** 建立能迁移的可拟合公式，而非用经验常数 100 作主标题。
2. 训练长度与词表都完全 hold out，预注册分桶边界和预测；多个随机种子，误差条、置信区间。
3. 相同 parameters、FLOPs、memory bandwidth、训练时间四种预算全部公平比较。
4. 125M+、32k+ vocab、不同 tokenizer 及自然语料；与 Engram、X-GRAM 的正式实现/设定交叉验证。
5. 给出失败区间：高碰撞率、低维向量、足够长训练、分布漂移。

## 拍板

**“稀有词对欠训练”不是创新；精确转折规律若具有可外推预测力，才可能构成论文创新。** 当前是可用的数据分析题目，不是成熟顶会新算法。优先级：第六。
