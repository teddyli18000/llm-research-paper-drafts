# 01｜动态聚类共享语言模型输出向量：研究结论（2026-11-17）

**研究阶段已完成。** 最终证据归档：[创新性、可行性与论文定位](EVIDENCE_SYNTHESIS_2026-11-17.md)；[原作者与新增工作的归属台账](ORIGINAL_IDEA_VS_NEW_CONTRIBUTION.md)；[独立测试集 V50k](INDEPENDENT_HELDOUT_V50K_RESULTS.md)。

## 最终决定

1. 原作者的 `R=U+AC`、KMeans 动态重分组、函数保持式共享向量、梯度求和和 rare-token exposure 规律，都是**原作者已有工作**。
2. 我们独立提出并检验的 `G'=(I+αP)G`、`P=A(AᵀA)⁻¹Aᵀ` 是**具体而有限的算法增量**：单矩阵、类均值归一化、可接 Ember/Adam。固定分组下与适当缩放的共享质心 SGD 更新完全等价；不是一套全新的类别共享或 SGD 数学。
3. 自适应优化器下确有可复现差异：V6k 学习率扫描 batch32 的 **18/18** 个配对结果均支持直接投影；batch128 结论随 LR 变化，不能声称普适优越。
4. **最硬证据：V50k untouched FineWeb shard002**，三种子 projector+Ember **6.29370 CE** 优于 plain Ember **6.31510**；但与原作者更强的 centroid+Adam **6.29480** 基本打平（差 0.00110 CE），没有证明我们的方法整体胜过原作者。
5. 图概率评分、语义平滑、concept-class CE 的匹配消融支持该实现与简单损失改写不同；与此同时，类别残差正好是一个图二次评分的 **logit Fisher 自然梯度**，进一步限定所谓“新数学”范围。
6. 目前训练的最大 V50k 模型只有 **约 6.54M 参数**，其中 **98% 是 token 表**，contextual Transformer 主干仅约 0.1M；没有现代 LLM 尺度验证。原始动态 projector 还会在等 token 的 batch128 下输给原作者方法。

**论文定位：** 当前适合一篇以 optimizer-coordinate geometry、rare-token learning 为主题的实证型技术报告或 workshop 论文。**不够支撑 ICML/NeurIPS/ICLR 主会方法论文。** 再做几十轮 V6k 调参不会改变此判断。只有真实 >100M contextual-backbone、强基线独立调参、tie/AGG 和原方法对比中出现明确新优势，才值得升级为主会论文计划。

## 最关键的完成实验

- [72-arm normalized-centroid 学习率扫描，6/6 success](NORMALIZED_CENTROID_LR_SWEEP_RESULTS.md)
- [外部 V50k 独立测试，15 个预注册训练](INDEPENDENT_HELDOUT_V50K_RESULTS.md)
- [V16k 强 Ember 头学习率扫描](TUNED_EMBER_V16K_CHALLENGE_RESULTS.md)
- [Tree Loss/归一化质心 SGD 等价及验证](TREE_LOSS_PRECONDITIONER_COLLISION.md)
- [语义 smoothing / concept-class loss，对照原始日志](CLASS_LOSS_COLLISION_RESULTS.md)
- [30-arm 图扩散 proper-score 对照，6/6 success](GRAPH_DIFFUSION_SCORE_RESULTS.md)
- [类别梯度与 Fisher 自然梯度的严格关系](GRAPH_SCORE_FISHER_GEOMETRY.md)
- [实际模型尺度审计](ACTUAL_MODEL_SCALE_AUDIT.md)

所有新增代码、文档、CI 工作流保存在独立 `research-lab/01-rare-words-shared-output-vectors`、`research-review/01-rare-words-shared-output-vectors` 和相应 scoped workflows；原始六篇论文和实验代码未修改。
