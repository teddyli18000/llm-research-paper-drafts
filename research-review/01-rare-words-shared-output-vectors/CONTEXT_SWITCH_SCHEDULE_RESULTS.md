# Paper 01 — absolute-step context-to-dynamic schedule results

**Run:** [GitHub Actions #44050503629](https://github.com/teddyli18000/llm-research-paper-drafts/actions/runs/44050503629), three jobs successful.  
**Seeds:** 569, 571, 577. **Batch:** 32. **Budget:** 819,200 tokens per arm. Four arms per seed, 12 runs total.

## Overall held-out cross-entropy

Lower is better. Each result is the mean of three seeds.

| Schedule | Mean CE | Per-seed CE |
|---|---:|---|
| Fixed context | 4.97100 | 4.9719, 4.9689, 4.9722 |
| Context → dynamic at step 150 | 4.96650 | 4.9687, 4.9636, 4.9672 |
| Context → dynamic at step 200 | **4.96627** | 4.9675, 4.9639, 4.9674 |
| Context → dynamic at step 250 | 4.96667 | 4.9680, 4.9645, 4.9675 |

Every switch point beats fixed context on all three seeds:

- Step 150: paired deltas −0.0050, −0.0032, −0.0053; mean −0.00450.
- Step 200: −0.0048, −0.0044, −0.0050; mean −0.00473.
- Step 250: −0.0047, −0.0039, −0.0044; mean −0.00433.

## Preregistered decision

The protocol says that when switch points are within 0.002 CE, prefer the later threshold to preserve the context prior longer. All three switch points are within 0.00040 of the best mean. Therefore the selected schedule is **switch at absolute step 250**, not step 200, despite step 200 having the lowest raw mean.

This selection rule was committed before the outcomes were inspected.

## Frequency buckets

| Training exposure | Fixed context | Step 150 | Step 200 | Step 250 |
|---|---:|---:|---:|---:|
| 0 | 12.9957 | 13.1573 | 13.1417 | **12.8310** |
| 1–9 | 10.8568 | 10.7342 | 10.7192 | 10.7349 |
| 10–31 | 9.7210 | 9.6318 | 9.6403 | 9.6518 |
| 32–99 | **8.5021** | 8.5263 | 8.5194 | 8.5115 |
| 100–315 | **7.1360** | 7.1557 | 7.1569 | 7.1571 |
| 316–999 | **5.6720** | 5.6745 | 5.6734 | 5.6757 |
| 1k–3k | 4.6640 | 4.6629 | 4.6622 | 4.6628 |
| 3k+ | 2.8466 | 2.8434 | 2.8440 | 2.8423 |
| Unknown | 1.6870 | 1.6794 | 1.6790 | 1.6826 |

The switch does not improve every exposure band. Its most consistent gains are in the 1–31 exposure bands and the high-frequency/unknown tail; the middle exposure bands regress slightly. Step 250 additionally gives the best zero-exposure bucket in this run. The overall gain is small but seed-consistent.

## Update diagnostics

| Schedule | Projected/raw gradient norm ratio | Mean output-matrix update norm |
|---|---:|---:|
| Fixed context | 1.0582 | 14.9431 |
| Step 150 | 1.0807 | 14.8703 |
| Step 200 | 1.0766 | 14.8640 |
| Step 250 | 1.0716 | 14.8831 |

The switch changes the projector geometry and modestly reduces the actual head-update norm. It does not increase the global update magnitude.

## Decision

Use **context groups through step 250, then output-row K-means refreshed every 50 steps** as the batch32 schedule candidate. Keep batch128 fixed context because its 100-step horizon never reaches the switch point.

This is a three-seed, short-horizon schedule result. It identifies a practical candidate, not a universal optimum.
