# 01｜动态聚类共享语言模型输出向量：研究状态

原始材料：[PAPER.md](../../01-rare-words-shared-output-vectors/PAPER.md) · [grid.py](../../01-rare-words-shared-output-vectors/grid.py) · [RESULTS.md](../../01-rare-words-shared-output-vectors/RESULTS.md)

## 研究对象

原作者的方法将每个输出 token 行表示为私有向量与共享簇质心之和：

\[
R=U+AC
\]

训练期间对当前输出行做 K-means，并通过重分配私有残差保持所有 logits 不变。之后梯度通过共享质心在组内传播。这个核心共享机制、在线 K-means 和函数保持式重分组都已经存在于上游作者的公开工作中，不能作为我们的原创。

我们的独立研究聚焦于其梯度几何的平坦输出头实现：

\[
G=(p-y)h^\top,\qquad
G'=(I+\alpha P)G,\qquad
P=A(A^\top A)^{-1}A^\top.
\]

## 已建立的数学边界

1. \(P\) 是组内均值正交投影，\(P^2=P=P^\top\)，谱为 0/1；\(I+\alpha P\) 的谱为 1 与 \(1+\alpha\)。
2. 固定分组下，这个更新在 SGD 中与归一化共享质心参数化
   \[
   W=U+\sqrt{\alpha}A(A^\top A)^{-1/2}C
   \]
   完全等价。25 步非线性 softmax 测试误差为 \(2.22\times10^{-15}\)。所以它不是新的 SGD 几何。
3. 它一般不是普通 logit 坐标下某个标量 loss 的梯度，因为 \((I+\alpha P)J_{\rm softmax}\) 通常不对称。
4. 图扩散 proper score 在由组内完全图组成的图上，其概率空间热核在长时间极限也收敛到同一个 \(P\)。但回传到 logits 时仍乘 softmax Jacobian，因此不是同一更新。
5. 最有价值的待验证问题是：直接在完整输出头上投影梯度，与精确归一化质心参数化，在 Ember/Adam 的自适应状态下是否有稳定、可解释的差异。

详见：[数学机制](FORMAL_MECHANISM.md) · [归一化投影引理](PROJECTOR_LEMMA.md) · [Tree Loss 碰撞](TREE_LOSS_PRECONDITIONER_COLLISION.md) · [2026 文献审查](2026_LM_HEAD_OPTIMIZER_PRIOR_ART.md)。

## 当前实验结论

### 已完成

- **V≈50k**：动态输出行分组在多组匹配实验中优于 plain Ember；与上游 centroid/Adam 类基线接近。固定 train-context 组在 V50k 不如 online K-means。
- **V6k residual study**：[Actions #44046155032](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44046155032)。固定 train-only context 的完整投影在 batch32/128 均优于 plain Ember 和 upstream centroid，三种子一致。
- **Context→dynamic schedule**：[Actions #44050503629](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44050503629)。step150/200/250 均优于 fixed context；预注册的“相差不超过0.002时选更晚阈值”规则选择 step250。batch128 的 100 步训练不切换。
- **Normalized centroid adaptive collision**：[Actions #44051580714](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44051580714)。直接投影相对精确归一化质心参数化，在 batch32 下 Ember/Adam 均有三种子一致优势；batch128 下两者近乎持平。完整数据见 [结果](NORMALIZED_CENTROID_ADAPTIVE_RESULTS.md)。
- **Raw second-moment routing**：[Actions #44043530225](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44043530225)。原始梯度统计路由造成输出头更新过冲，整体不如共享统计方案；停止继续调全局 raw-stat 路由。
- Tree-Loss reparameterization、semantic smoothing、concept-class probability loss、cross-microbatch reliability 等碰撞实验均已完成，细节见 [WORKLOG.md](../../research-lab/01-rare-words-shared-output-vectors/WORKLOG.md)。

### 需要重新解释的实验

第一轮 prediction-only / target-only residual 分解使用了未中心化的 \(Pp\) 和 \(-Py\)。两者包含相反的 vocabulary-uniform 模式，完整差分会抵消它。该模式不改变当前 softmax，但可能影响 Ember 状态。因此第一轮单侧 CE 不能证明两个中心化残差缺一不可。

[中心化 residual follow-up](CENTERED_RESIDUAL_RESULTS.md) 已完成（Actions #44052591527）。中心化 prediction-only/target-only 仍然显著差于 full projector；uniform-only 控制也明显退化。两项高度相关的残差必须通过差分形成有用更新。完整投影结果不受初始混杂问题影响。

## 当前创新性判断

**不能主张新的 SGD 优化几何、一般性的 class sharing、语义平滑或图概率残差。** 这些均有明确先例或等价形式。

仍有价值的窄问题是：直接平坦输出头投影是否能在现代自适应优化器下，比其精确共享质心参数化更有效地改善低曝光 token，且这种差异是否跨学习率、batch、词表规模和模型规模成立。

目前只有 batch32 的自适应参数化差异较明显，batch128 近乎持平。尚不足以作强方法创新或顶会论文结论。

## 正在推进的下一步

1. 完成 centered residual decomposition，修正单侧机制归因。
2. 完成直接投影 vs 归一化质心的 head-LR sweep（0.025/0.05/0.10）。
3. 加入图扩散 P-score、output-only Adam epsilon、projected RowNorm 和 MuonIO 强基线。
4. 在 untouched shard 和更大模型上验证；若增益不能超过强参数化/优化器基线，则收窄论文定位或停止方法主张。

研究日志：[WORKLOG.md](../../research-lab/01-rare-words-shared-output-vectors/WORKLOG.md)。
