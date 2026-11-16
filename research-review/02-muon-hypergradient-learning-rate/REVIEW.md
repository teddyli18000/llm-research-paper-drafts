# 02｜Hypergradient on Muon

对应原文：[PAPER.md](../../02-muon-hypergradient-learning-rate/PAPER.md)，[RESULTS.md](../../02-muon-hypergradient-learning-rate/RESULTS.md)。

## 原作者的主张

对 Muon 引入利用 gradient 与上一 update 对齐度的**乘法 hypergradient 学习率更新**，叠加“只在前期适应、后期固定”、随 sqrt(width) 调增益以及要求 weight decay。作者报告 15 个小模型配置、45 次起始 LR 试验中 42 次接近所扫出的最佳常数 LR（差 ≤ 0.005）；最大约 1.3M 参数。单次设置没有多 seed，最长几千步；未测试 10^4–10^5 步训练。

## 已有研究与精确差异

1. **Baydin et al., ICLR 2018**：在线 hypergradient 调学习率已用于 SGD/Adam。**核心更新原则已存在**。[ICLR](https://mlanthology.org/iclr/2018/baydin2018iclr-online/)
2. **Funnel / exponentiated update（原文所引 2022）**：用 gradient–update cosine 做乘性更新亦已存在；所以不能把这个规则本身算作创新。
3. **Chu et al., ICML 2025**：重新分析 hypergradient 方法并提出可靠变体；理论竞争不弱。[PMLR](https://proceedings.mlr.press/v267/chu25a.html)
4. **Crawshaw et al., ICML 2026**：MuonMax 和 Momo 显著降低 Muon 的学习率敏感度，并明确把“不需要昂贵扫参”作为目标。**问题和功能最直接撞车**，但算法不相同。[PMLR](https://proceedings.mlr.press/v306/crawshaw26a.html)
5. **MoMo, ICML 2024**：自适应学习率用于 momentum 优化器，提供更强同目标基线。[PMLR](https://proceedings.mlr.press/v235/schaipp24a.html)

## 可以守住的最窄创新

**已知乘法 hypergradient 的 Muon 适配细节**：梯度-正交化更新的正确对齐信号、early-hold 防 short-horizon collapse、宽度缩放、weight-decay 必要性及失败分析。这更像扎实的**优化器应用/系统实证增量**，不是一种全新学习率自动化方法。

## 投稿判断

**能写为可检验方法稿，但 ICML/NeurIPS 主会目前创新强度偏弱。** 关键不是是否 42/45 成功，而是 125M+ 标准基准下是否比 MuonMax+Momo 更省总计算、更稳定且少超参数。不能把训练 1 次和 sweep 5–7 次的对比作为对所有自适应优化器的胜利。

## 必须做的实验

1. 以同等 FLOPs / tokens / 最优调参预算对比 Muon、MuonMax、Momo-Muon、Mechanic、Prodigy、D-Adaptation；显示调参与训练的**总成本**。
2. 至少 125M 模型、真实词表、>10k 步，长日程前后测试是否仍因 short-horizon bias 缩小过度。
3. 多 seed + 不同起始 LR 倍数，统计“未失败概率”；固定基线也调 warmup、decay 和 weight decay。
4. 去除 early-hold / sqrt(width) / weight decay / shared multiplier 的消融；给出跨宽度理论或健壮经验规律。
5. 检测计算开销是 FLOPs、显存还是 wall-clock；Muon's矩阵形状变化是否导致失效。

## 拍板

**典型的旧思想+新优化器适配**。可能确实有工程价值，不能当成原创的自动调学习率思想。优先级：**第三**；没有胜过 MuonMax/Momo 的结果就不值得为顶会单独扩篇。
