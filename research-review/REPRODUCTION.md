# Research Reproduction Audit

审查日期：2026-11-16；运行环境：GitHub Actions Ubuntu Linux，Python 3.11，PyTorch CPU。使用 Fork 当前代码和原作者的 reproduce.sh（04/05 仅在 runner **临时**把脚本的默认 dev=mps 换成 cpu；未提交任何对原实验代码的改动）。两个实验均使用原论文给出的 headline 参数。

## 最终可信的运行

[Actions #43792700369 — completed / success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43792700369)

| 方向 | 对照模型 | 改动方法 | 对照结果 | 新方法结果 | 复现结论 |
|---|---|---|---|---|---|
| 04：2-hop | tuned full attention | identity heads | acc 0.323 | **acc 0.991** | 成功 |
| 04：4-hop | tuned full attention | pointer + no-backtrack | acc 0.330 | **acc 0.996** | 成功 |
| 05：7-hop / 3 层 | tuned full attention | binary-lifting pointer | acc 0.174 | **acc 0.980** | 成功 |
| 05：15-hop / 4 层 | one-hop pointer | binary-lifting pointer | acc 0.170 | **acc 0.991** | 成功 |

以上数据直接取自本次 GitHub Actions job 输出；并非复制作者 PAPER.md 的预期数字。验证步骤会检查 04、05 全部 8 个子实验有结果、没有 FAILED，且与原文报告方向一致。workflow 日志与 artifacts 保留在上述 run 页面。

## 第一轮误报成功：为何 CI 必须检查子进程结果

[Actions #43792382601 — 状态 success，但 04/05 实验实际上失败](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43792382601)

- 04、05 各四项均输出 **FAILED RuntimeError: PyTorch is not linked with support for mps devices**。原因是作者的 rbg.py / pdg.py 默认 dev=mps，而 GitHub Linux Runner 只支持 CPU。
- 原 grid.py 捕捉子进程报错后打印 FAILED 并继续，最终返回 0，因此最初工作流错误地变绿。
- 修复仅位于 .github/workflows/research-reproduction.yml；在 Runner 临时设置 CPU，并添加对实验子结果与 headline 数值的显式断言。因此第二轮 success 才是有效的 smoke-reproduction 结果。

## 这验证了什么、尚未验证什么

**验证了**：04、05 的原始合成任务 headline 结果在 GitHub Linux CPU 中可以独立重跑，趋势及关键差距与作者相符。

**没有验证**：
- 01、02、06 的数据/结果；它们没有进入这轮 Actions。
- 多 seed 均值、方差、超参搜索的公平性、完整 results.jsonl 的每一项。
- 125M+ 真实语言模型、长训练真实下游任务、GPU kernel 成本。
- **独创性**：代码可重现不等于文献优先权。本仓 05 的倍增在 Sanford/Hsu/Telgarsky ICML 2024 中已经有重要先例。

下一阶段应针对各 REVIEW.md 列出的“最小决定性实验”推进，而不是重复运行相同 toy headline 更多遍。
