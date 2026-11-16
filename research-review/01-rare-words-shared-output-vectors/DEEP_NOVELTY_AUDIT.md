# 01｜深入新颖性审查：Dynamic Cluster-Sharing of Softmax Output Rows

日期：2026-11-16。原始方法：[PAPER.md](../../01-rare-words-shared-output-vectors/PAPER.md)，代码：[grid.py](../../01-rare-words-shared-output-vectors/grid.py)，数据日志：[RESULTS.md](../../01-rare-words-shared-output-vectors/RESULTS.md)。这是原仓库的研究思路，不是已发表论文的优先权认定。

## A. 把创新主张缩到数学上不可替代的部分

原方法对词 w 的输出层向量采用
`R_w = A_w + C_{g(w)}`，其中 `A` 是逐词私有矩阵、`C` 是簇共享矩阵、`g` 是动态簇 ID。训练过程中定期对 **当前等效输出行** `R` 做 k-means，随后重新设置
`g' = kmeans(R)`, `C'_k = mean(R_w | g'(w)=k)`, `A'_w = R_w-C'_{g'(w)}`。
因此 **R' = R（逐词恒等）**，在完全相同的隐藏状态下所有 logits 和 softmax 在重分组瞬间严格不变；之后梯度在 `C` 中聚合，改变训练轨迹。最终将两项相加可得到推理零额外参数/操作的普通输出矩阵。

**不能声称新颖：**稀有词共享表示、learned word classes、对输出矩阵聚类、动态分组、语义平滑、重参数化保持函数，各自已有先例。

**在本轮已核对最接近先例中尚有区别：**将 *语言模型预训练期间的动态输出行聚类*、*逐词 residual + shared vector*、*每次重聚类精确函数保持* 三者一起用于增强稀有词目标预测。这是一个很窄的可能方法贡献，不是全面排他性优先权证明。

## B. 重点先例矩阵（按对主张的威胁程度）

| 论文、年份、链接 | 已经完成的事 | 对当前 01 的影响 | 还存在的区别 |
|---|---|---|---|
| [Botha & Blunsom 2014, ICML — Compositional Morphology](https://proceedings.mlr.press/v32/botha14.html) | 用词自身与形态学成分共享表示缓解稀有词 | “稀有词借邻词参数学习”原则已知 | 固定形态学成分，而非持续根据输出层自身几何重聚类 |
| [LightRNN, 2016](https://arxiv.org/abs/1610.09893) | 对语言模型输入/输出行列共享词向量，训练中交替优化分配 | 动态分配 word-to-shared-parameters 不是首创 | 两个共享向量、无逐词完整 residual；改变词分配后词概率不保持 |
| [GroupReduce, NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/a2b8a85a29b2d64ad6f47275bf1360c6-Abstract.html) | **直接对 LM softmax/output 矩阵做聚类 + 迭代重新分配 + 组内低秩共享** | “输出向量动态组群本身新”已经被撞掉，是最重要的几何近邻之一 | 后处理压缩、通过加权 SVD/误差重分配，不是语言模型 SGD 训练期间的严格 logits-preserving 组共享 |
| [Chen & Gimpel, NAACL 2018](https://aclanthology.org/N18-2116/) | 学可调整的词簇 embedding，cluster adjustment 为每词增加微调参数 | learned clusters + group+per-word tweaks 的构型不完全新 | 目标是文本分类和压缩；非软最大输出层、没有相同的全量 residual/无损重分组 |
| [Soft Weight-Sharing, ICLR 2017](https://arxiv.org/abs/1702.04008) | 训练中可变的 mixture/cluster responsibility 促进共享参数 | 动态 weight clusters 本身早见于前作 | 无输出词专用函数保持重分组 |
| [Syntactic Smoothing, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.344/) | 在 LM 预训练中让句法相近 token 分享更新信号，改善低频项 | **训练中跨词分摊 learning signal 的目的已经被做过** | 使用句法先验与 loss 修改，不是自组织输出行共享 |
| [Rare Tokens Degenerate All Tokens / AGG, ACL 2022](https://aclanthology.org/2022.acl-long.3/) | 分析 rare token 梯度导致的表示退化，针对稀有 token 梯度做门控 | 01 的“梯度分布/低频优化缺陷”不是新诊断 | 梯度筛选 vs 共享输出行参数 |
| [LENS, ACL 2025](https://aclanthology.org/2025.acl-long.930/) | 对 **LM head 输出 token 向量**做 K-means 聚类，改成词簇级 lexical 输出 | “在 output head 上做 K-means”并不新；目标矩阵与操作几乎同名 | 预训练后替换输出词向量、面向词汇检索表示；**改变 logits 与词表**，不是无损动态预训练 |

这里“动态”分两种：GroupReduce 的压缩目标中**迭代细化分簇**，与 01 的**在语言模型训练每 50 步重分簇**不能混为一谈。LENS 不是预训练时重新聚类。

## C. 源码与论文的不一致：优先于扩规模修复的科学问题

### C1. PAPER.md 写了 `reset_adam_state(self.C)`，grid.py 没有

PAPER.md §2 的伪代码在分组后调用 `reset_adam_state(self.C)`；实际 [grid.py 的 `Model.recluster()`](../../01-rare-words-shared-output-vectors/grid.py) 只进行 `s.ah.copy_(asg); s.Ch.copy_(cm); s.head.copy_(R-cm[asg])`，且训练 loop 仅调用 `m.recluster()`，**没有 reset/transport Adam moments**。

影响：`C[k]` 被赋予新的组平均向量，但 Adam 的 `exp_avg/exp_avg_sq` 还记录过去组的历史。严格地说是一个 **optimizer-state / parameter-semantics mismatch**。模型函数仍保持不变，但下一步优化动力学受到过去分组的 moments 影响。因此不能把验证损失的全部提升直接归因于“语义共享梯度”。这不构成“实验作弊”或“实验不成立”，而是需要消融的机制混杂和文稿实现不一致。

### C2. k-means 初始化也不同

PAPER.md 伪代码 `kmeans(..., init=self.c)`；实际 `grid.py` 调用 `kmeans(R, K)`，算法每次用固定种子 1 的随机 64 行初始化中心，再执行 10 次迭代，**没有使用上一轮分组作初始化**。若聚类非平稳，这会影响 group switching 的幅度与 Adam 状态历史匹配。报告中应按实际实现描述，而不是把伪代码当作可复现实验事实。

### C3. 梯度机制应写精确

交叉熵对任何输出词行都有梯度 `∂L/∂R_w=(p_w-1[w=y])h`，不是“非答案行不更新”。稀有词的困难来自正样本不足、非目标的背景梯度和 Adam moments 的动态；不能只用“出现次数少，所以得到的梯度少”解释。ACL 2022 AGG 对稀有词的非目标梯度具有直接关联。

### C4. 本身就有规模外推预警

作者的结果显示固定 token 预算下 batch 16/32/64/128 的增益约 0.025/0.021/0.011/0.007，而现实 LLM 使用极大全局 batch。即使在小模型上可复现，**大 batch 上收益是否仍可测，是可发表性的首个决定项**。报告数值是作者的原实验，不是本次独立复现。

## D. 拍板与可发表范围

- **创新：有一个狭窄、合理的候选方法点；并非整体撞车。**
- **主要风险：**（a）与 GroupReduce/LENS/Syntactic Smoothing 的原理与组件先例非常近；（b）方法效果来自优化器历史而不是主要解释的可能性；（c）大型 batch 下可能消失。
- **可写论文：**可以是《Function-Preserving Dynamic Sharing for Tail Output Embeddings》的训练动力学方法论文，如果真正强对照与大规模生效；目前还不足以宣称 ICML/NeurIPS 主会成熟。
- **建议排序：**先小规模代码与机制清理，再真实模型。**不应先烧 125M+ 的训练预算**。

## E. 最小能推翻或强化贡献的实验（顺序固定）

1. **MOMENTS-4**：保持完全相同模型、随机数与 re-clustering schedule，比较 `keep Adam moments` / `reset C moments` / `reset C+own moments` / `transport moments with reassigned groups`，监测重分组前后 max-logit-diff、Adam moment norms、所有词损失。若只有 keep 模式获益，必须改写机制故事。
2. **CLUSTER-5**：固定同一输出分解与训练预算：random / frequency / morph / POS / LENS-style pretrained head K-means frozen / dynamic k-means；同时增加一次性 static k-means from a matched warm-start，拆分“学到好组”和“频繁重分组”的贡献。
3. **SOFTMAX-COMP**：对齐 Adam epsilon、head LR、weight decay、语义平滑（EMNLP 2024）、AGG（ACL 2022）、等效组共享参数开销；至少 3–5 seeds。
4. **BATCH-TEST**：保持总训练 token 与有效 LR 搜索，改变真实 global batch（特别 ≥ 2k–16k tokens/step），分频率 bucket 报 loss、总体 gain、负迁移；若随 batch 扩张归零，应停止推广为通用 LLM 方案。
5. **SCALE**：以上成功后，在 125M+、32k–50k full vocabulary、真实 text + 合理 token budget 上评估，报告真实 throughput、显存与额外聚类耗时；输出向量必须对低频词提供足够样本。

## 核心评审意见（一句话）

**精确保留输出函数的在线重聚类是目前最能 defend 的组合细节；但必须先排除 Adam 状态混杂，并证明在实际预训练 batch 仍有效。** 
