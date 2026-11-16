# 02｜Muon 的 Hypergradient 学习率自适应

原始材料：[PAPER.md](../../02-muon-hypergradient-learning-rate/PAPER.md) · [RESULTS.md](../../02-muon-hypergradient-learning-rate/RESULTS.md)

## 研究问题与结果

Muon 的学习率需要调节。原作者利用梯度与上一更新方向的对齐度，进行乘法 hypergradient 调节，并加入前期适应、后期固定、宽度缩放和 weight decay。报告在 15 个小模型配置、45 次初始学习率实验中，有 **42 次**达到所扫最佳固定学习率附近（loss 差 ≤0.005），最大模型约 130 万参数。

## 参考与贡献边界

- [Baydin et al., ICLR 2018](https://mlanthology.org/iclr/2018/baydin2018iclr-online/)：在线 hypergradient 更新学习率的早期系统方法。
- [MoMo, ICML 2024](https://proceedings.mlr.press/v235/schaipp24a.html)、[Chu et al., ICML 2025](https://proceedings.mlr.press/v267/chu25a.html)：自适应优化理论及实证方法。
- [MuonMax / Momo, ICML 2026](https://proceedings.mlr.press/v306/crawshaw26a.html)：同样关注 Muon 的学习率稳健性，算法途径不同。

**可研究的贡献：**Muon 正交化更新的 hypergradient 信号设计、训练早晚阶段适应策略、不同矩阵宽度/形状的稳定性与训练成本规律。这是有实际意义的优化器方法研究；新意集中在 Muon 专属更新机制。

## 后续实验

1. 同预算对比 tuned Muon、MuonMax/Momo 与其他自适应 LR 方法，报告最终损失和完整调参成本。
2. 扩大训练步数与模型规模，绘制 LR 轨迹、更新方向对齐度和失败配置。
3. 逐项消融 early-hold、width scaling、weight decay；跨不同矩阵形状和随机种子。
4. 分析 hypergradient 估计偏差，尝试给出可预测的选择规律。

**论文方向：**Muon 训练过程的自适应学习率方法及稳定性分析。
