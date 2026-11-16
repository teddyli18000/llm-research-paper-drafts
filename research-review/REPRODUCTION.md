# 04/05 合成实验复现记录

审查日期：2026-11-16；运行环境：GitHub Actions Ubuntu Linux，Python 3.11，PyTorch CPU。使用 Fork 当前代码和原作者的 reproduce.sh（04/05 仅在 runner **临时**把脚本的默认 dev=mps 换成 cpu；未提交任何对原实验代码的改动）。两个实验均使用原论文给出的 headline 参数。

## 成功复现结果

[Actions #43792700369 — completed / success](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43792700369)

| 方向 | 对照模型 | 改动方法 | 对照结果 | 新方法结果 | 复现结论 |
|---|---|---|---|---|---|
| 04：2-hop | tuned full attention | identity heads | acc 0.323 | **acc 0.991** | 成功 |
| 04：4-hop | tuned full attention | pointer + no-backtrack | acc 0.330 | **acc 0.996** | 成功 |
| 05：7-hop / 3 层 | tuned full attention | binary-lifting pointer | acc 0.174 | **acc 0.980** | 成功 |
| 05：15-hop / 4 层 | one-hop pointer | binary-lifting pointer | acc 0.170 | **acc 0.991** | 成功 |

以下数据来自 GitHub Actions 的实际运行。验证步骤会检查 04、05 全部 8 个子实验有结果、没有 FAILED，且与原文报告方向一致。workflow 日志与 artifacts 保留在上述 run 页面。

## Linux CPU 适配与首次 CI 故障

[Actions #43792382601 — 状态 success，但 04/05 实验实际上失败](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/43792382601)

- 04、05 各四项均输出 **FAILED RuntimeError: PyTorch is not linked with support for mps devices**。原因是作者的 rbg.py / pdg.py 默认 dev=mps，而 GitHub Linux Runner 只支持 CPU。
- 原 grid.py 捕捉子进程报错后打印 FAILED 并继续，最终返回 0，因此最初工作流错误地变绿。
- 修复仅位于 .github/workflows/research-reproduction.yml；在 Runner 临时设置 CPU，并添加对实验子结果与 headline 数值的显式断言。因此第二轮 success 才是有效的 smoke-reproduction 结果。

## 本次实验覆盖范围

这轮 GitHub Actions 在 Ubuntu CPU 环境复现了 04/05 的 headline 合成任务结果，包含论文中最主要的性能差异。后续研究分别关注：

- **01/02/06** 的独立复现与数据设定；
- **04/05** 的多种子分布、强基线、规模曲线、自然语言与代码任务；
- 训练/推理时间、GPU kernel 和模型容量的定量分析。

