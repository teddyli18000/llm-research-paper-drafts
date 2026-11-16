# 05｜Pointer doubling / binary lifting in attention

对应原文：[PAPER.md](../../05-attention-pointer-doubling/PAPER.md)，[RESULTS.md](../../05-attention-pointer-doubling/RESULTS.md)。

## 原作者的主张

前一篇 04 的 pointer 每层走一跳；本篇让位置 j 的 pointer 已走 2^l−1 跳，位置 i 复制 j 的 pointer，从而每层可走 2^l 跳，L 层可达 2^L−1 跳。加入 double no-backtrack penalty、第二条受 gate 控制的 pointer 来处理长度不是 2^L−1 的链（类似 binary lifting）。合成数据中 3 层走 7 跳、5 层走 31 跳；有反向链存储顺序要求，混合链长和正常顺序存在失败，文本 loss 无明显优势甚至变坏。

## 严重的先行工作重合

1. **Sanford, Hsu, Telgarsky, ICML 2024 — Transformers, Parallel Computation, and Logarithmic Depth**：理论给出了 O(log k) 深度解决 k-hop induction，且附录实证报告学习到的 attention matrices 存在**doubling phenomenon**。因此“首次发现每层倍增、对数层数足以多跳”**不是原创**。[会议论文](https://proceedings.mlr.press/v235/sanford24a.html) ｜ [含 empirical claim G.2 的论文 PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/sanford24a/sanford24a.pdf)
2. **Binary lifting / pointer jumping**：经典算法思想，本身显然不能视作新的 ML 理论。
3. **Memory Mosaics**：准确内容寻址 / identity induction 的主要模块在 2024 已有。[arXiv](https://arxiv.org/abs/2405.06394)
4. **本仓 #04**：本篇几乎是相同代码骨架、任务和研究线的升级版本；容易被认为应是一篇连续工作。

## 尚可探讨的最窄新贡献

**将旧的对数深度 pointer jumping 算法塞入可训练 Transformer 的特定 head 通路**，在**只有最终监督、短训练**下可学；加 doubled backtrack penalty 维持匹配质量、gated pointer 对跳长编码。与 Sanford 文献不仅要比理论，还要比其 **实证所观察到的 learned doubling**，不能把这项先例忽略掉。

## 审稿风险判断

**作为单独的顶会主会论文：高撞车风险。** 头号拒稿点不是实验太小，而是核心概念已经被 2024 工作提出并观察。若不能证明**特定新组件解决前作解决不了的真实问题**，增加几百个 toy run 无法把现有基本思想变原创。

## 最小决定性实验

1. 复现 Sanford et al. 附录 G 的多跳训练设置和长度混合，比较**训练收敛、泛化、模型规模、算力**。对比点不能只有作者自己的 vanilla attention。
2. 从 query 读取 hop-count，在**单模型混合长度**训练/泛化；目前 gate 按任务编码一个 k 的限制非常严重。
3. 正向/反向、链位置随机化、相同 key 重复、干扰链、自然语言及代码数据；解决 14/4、18/5、20/5 等失败。
4. 125M+ 模型 + 自然 long-context benchmark，证明推理收益大于额外 head 复杂度。
5. 如果只有固定方向合成链收益，收敛成**有启发价值的实验 note**，而不是夸大为实用通用架构。

## 拍板

**创新最弱/撞车风险最高的方向之一；建议不要单独作为投稿主线，而是和 04 合写一篇“从精确检索到可训练倍增”的研究，公开承认 Sanford 2024 优先权。** 优先级：第五（与 04 绑定）。
