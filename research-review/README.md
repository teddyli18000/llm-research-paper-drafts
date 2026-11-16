# 六篇 LLM 研究草稿：独立新颖性与可发表性审查

> 审查日期：2026-11-16。研究材料：本 Fork 原有 01–06/PAPER.md、相关 RESULTS.md、代码与 reproduce.sh；外部文献以论文官方页和 arXiv 原文摘要为主要来源。
> 这是一轮**文献对照 + 研究设计审查**，不等于全面系统文献综述，也不等于已独立复现。任何具体结果在 CI 核验前都属于**原作者声称/记录**。

## 结论矩阵

| # | 研究问题 | 原创性判断 | 投稿方向 | 当前状态 |
|---|---|---|---|---|
| 01 | 从模型输出向量动态发现低频词的共享组 | **可能有窄方法创新**：动态聚类、功能保持式重分配是组合贡献；“共享低频词参数”早已存在 | 可继续研究，但需强对照和规模验证 | Candidate / 未独立复现 |
| 02 | 用 hypergradient 自适应 Muon 学习率 | **增量创新**：hypergradient 并非新；Muon 特定的 early-hold、宽度增益和失效分析可能新 | 短稿/实证优化论文候选；顶会主会需超过新优化器基线 | Incremental / 未独立复现 |
| 04 | 显式 pointer head 逐层跟随事实链 | **部分创新，近邻密集**：identity match 已由 Memory Mosaics 等提出 | 与 05 合并为一条工作线更合理 | Overlap / 未独立复现 |
| 05 | pointer doubling 每层指数增加可达跳数 | **主要算法思想已被先行工作覆盖**；具体可训练实现及门控可能有增量 | 单独顶会主会投稿风险高；建议并入 04 | High collision risk / 未独立复现 |
| 06 | 单词/词对 value embedding 的频次交叉阈值 | **经验分析贡献候选**：百次阈值不是通用定律；稀有 n-gram 欠训练已有工作 | 更适合机制/实证研究，需可靠跨尺度预测 | Analysis candidate / 未独立复现 |

**不等于已经发表，也不等于这六个想法被完全撞掉。** 这里把“研究问题已知 / 一般原理已知 / 精确机制差异 / 可投稿的新增证据”分开判断。

## 建议次序

2. **01**：需要把 k-means 动态输出层参数共享与经典共享、动态软权重共享、形态/词性分组做同配置对照。
3. **02**：验证是否对 125M+ 预训练、长训练日程和 MuonMax / Momo 等新方法有竞争力。
4. **04 + 05**：**建议作为一个“可训练符号指针 + 路径倍增”论文方向**。先承认 2024 年的对数深度理论与训练观察；看是否能补出真实代码、自然多跳任务上的收益。
5. **06**：将“100 次”改成可转移的可检验预测，测试 tokenization、table capacity、training duration、模型规模，而非断言普适阈值。

## 审查标准与实验闸门

- **N0（优先权）**：方法的数学表达、代码实现和最近邻论文逐项对照；不可只凭作者 RELATED WORK 说“我们首创”。
- **N1（可重复性）**：干净环境重跑；多 seed，给均值、标准差/置信区间；核查数据划分、调参公平性。
- **N2（可归因）**：消融、等参数/等 FLOPs 对照和机制干预；与真正最接近的方法而非只和 vanilla Transformer 比。
- **N3（外推）**：有代表性的预训练规模、真实文本/代码/长上下文任务；评估推理/训练成本和稳定性。
- **N4（论文故事）**：给一个准确、可证伪、超出已有论文的新主张，并展示失败边界。

**区分“能写论文”和“够顶会”**：六个方向都能写成实验性研究稿；当前没有任何一篇达到可以负责任地直接宣称 ICML/NeurIPS 主会竞争力已充分证实的程度。

## 文件布局

本目录中的 01–06 与仓库源目录逐一对应，每篇包含：
- 原作者究竟声称了什么
- 文献先例与“相同 / 不同”判断
- 可保护的最窄创新主张
- 论文可发表性、拒稿风险
- 下一轮**最小决定性实验**
- 可点开的文献 URL

## 复现状态

- 检查了源论文、研究日志与脚本，并在 GitHub Linux CPU 上独立运行了 04/05 的 headline 实验；**实测结果和首次 CI 误报原因见 [REPRODUCTION.md](REPRODUCTION.md)**。
- 增设 GitHub Actions 工作流 `.github/workflows/research-reproduction.yml`：仅做 04、05 的 **CPU headline smoke reproduction**，不会覆盖原代码；日志作为 Actions artifact。①/②/⑥需要数据或工作流适配后再配置。
- [有效 Actions 运行 #43792700369](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43792700369) 全通过，并核对各项数值。工作流只验证单次合成任务 headline 现象，**不等于多种子外推或文献新颖性证明**。
- 01/02/06 仍为作者报告值；04/05 的单次实际数值与环境问题均已记录在 REPRODUCTION.md。

## 独立审查的核心外部文献

- [Botha & Blunsom, ICML 2014：形态共享词表示](https://proceedings.mlr.press/v32/botha14.html)
- [Syntactic Smoothing 2024](https://arxiv.org/abs/2410.11462)
- [Soft Weight-Sharing, ICLR 2017](https://arxiv.org/abs/1702.04008)
- [Baydin et al., ICLR 2018：Hypergradient](https://mlanthology.org/iclr/2018/baydin2018iclr-online/)
- [Chu et al., ICML 2025：Hypergradient](https://proceedings.mlr.press/v267/chu25a.html)
- [Crawshaw et al., ICML 2026：MuonMax + Momo](https://proceedings.mlr.press/v306/crawshaw26a.html)
- [Memory Mosaics, 2024](https://arxiv.org/abs/2405.06394)
- [Memory Mosaics at scale, NeurIPS 2025](https://arxiv.org/abs/2507.03285)
- [Sanford, Hsu & Telgarsky, ICML 2024：对数深度与多跳](https://proceedings.mlr.press/v235/sanford24a.html)
- [Engram, 2026](https://arxiv.org/abs/2601.07372)
- [X-GRAM, 2026](https://arxiv.org/abs/2604.21724)

## 方法学限制

本轮重点是**排除明显重复、确立最接近先例、找到足以推翻当前故事的实验**。未穷尽所有论文（尤其 2026 年末尚未索引的预印本），也未检验论文报告数字的统计显著性。文章的最终学术创新性需在完成上述 N0–N4 后重新判定。
