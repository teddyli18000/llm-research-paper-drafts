"""Cross-microbatch reliability-gated projection for the Paper 01 LM-head study."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[443, 449, 457], required=True)
p.add_argument("--batch", type=int, choices=[32, 128], required=True)
args = p.parse_args()

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()
patches = [
    (
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F',
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nfrom ember import Ember',
    ),
    (
        'p.add_argument("--coupled", type=int, default=0)',
        'p.add_argument("--coupled", type=int, default=0); '
        'p.add_argument("--proj_mode", choices=["plain", "context", "dynamic", "energy_gate", "crossbatch_select", "crossbatch_gate"], default="plain")',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.proj_mode != "plain" else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.proj_mode in ("context", "energy_gate", "crossbatch_select", "crossbatch_gate"):\n'
        '            ctx = torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy")))\n'
        '            s.register_buffer("context_groups", ctx)\n'
        '            if a.proj_mode == "context": s.ah.copy_(ctx)',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '    opts.append(Ember([m.head], lr=a.hlr, beta2=0.95, eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
        '    gate_trace = []; dynamic_steps = 0; alpha_sum = 0.0\n'
        '    score_ctx = 0.0; score_dyn = 0.0; dynamic_ready = False\n'
        '    rho_ctx_sum = 0.0; rho_dyn_sum = 0.0; rho_count = 0\n'
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.proj_mode in ("dynamic", "energy_gate", "crossbatch_select", "crossbatch_gate") and step > 0 and step % 50 == 0:\n'
        '            with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(), 64))\n'
        '            dynamic_ready = True\n'
        '            if a.proj_mode in ("energy_gate", "crossbatch_select", "crossbatch_gate"):\n'
        '                score_ctx = 0.0; score_dyn = 0.0\n'
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    ),
    (
        'def main():',
        '''def group_project(g, assignment):
    sums = torch.zeros(64, g.shape[1], device=g.device, dtype=g.dtype).index_add_(0, assignment, g)
    counts = torch.bincount(assignment, minlength=64).clamp_min(1).to(g.dtype)
    return sums[assignment] / counts[assignment, None]


def pair_reliability(p1, p2):
    cross = (p1 * p2).sum()
    energy = 0.5 * (p1.square().sum() + p2.square().sum())
    return float((cross / energy.clamp_min(1e-30)).clamp(-1.0, 1.0).item())


def main():''',
    ),
    (
        '        x = X[step * a.batch:(step + 1) * a.batch]\n'
        '        for xc in x.chunk(a.acc):\n'
        '            (m.nll(xc[:, :-1], xc[:, 1:]).sum() / (x.shape[0] * (x.shape[1] - 1))).backward()\n'
        '        for o in opts: o.step()',
        '''        x = X[step * a.batch:(step + 1) * a.batch]
        order = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(a.seed * 100000 + step)).to(D)
        x = x[order]
        micro_head_grads = []
        for xc in x.chunk(a.acc):
            (m.nll(xc[:, :-1], xc[:, 1:]).sum() / (x.shape[0] * (x.shape[1] - 1))).backward()
            if a.proj_mode in ("crossbatch_select", "crossbatch_gate"):
                current = m.head.grad.detach()
                if not micro_head_grads:
                    micro_head_grads.append(current.clone())
                else:
                    micro_head_grads.append((current - micro_head_grads[0]).clone())
        if a.proj_mode != "plain":
            with torch.no_grad():
                g = m.head.grad
                pdyn = group_project(g, m.ah)
                if a.proj_mode in ("context", "energy_gate", "crossbatch_select", "crossbatch_gate"):
                    ctx = m.context_groups
                    pctx = group_project(g, ctx)
                if a.proj_mode == "context":
                    g.add_(pctx, alpha=2.0)
                elif a.proj_mode == "dynamic":
                    g.add_(pdyn, alpha=2.0)
                elif a.proj_mode == "energy_gate":
                    total = g.square().sum().clamp_min(1e-30)
                    ec = float((pctx.square().sum() / total).item())
                    ed = float((pdyn.square().sum() / total).item())
                    use_dynamic = dynamic_ready and step > 0 and score_dyn > 1.5 * score_ctx
                    alpha = 2.0
                    g.add_(pdyn if use_dynamic else pctx, alpha=alpha)
                    dynamic_steps += int(use_dynamic)
                    if step == 0 or step % 50 == 0:
                        score_ctx = ec
                        if dynamic_ready: score_dyn = ed
                    else:
                        score_ctx = 0.9 * score_ctx + 0.1 * ec
                        if dynamic_ready: score_dyn = 0.9 * score_dyn + 0.1 * ed
                    alpha_sum += alpha
                    if step in (0, 25, 50, 99, 100, 200, 399):
                        gate_trace.append([step, round(score_ctx, 6), round(score_dyn, 6), int(use_dynamic), alpha, int(dynamic_ready)])
                elif a.proj_mode in ("crossbatch_select", "crossbatch_gate"):
                    assert len(micro_head_grads) == 2
                    g1, g2 = micro_head_grads
                    pctx1, pctx2 = group_project(g1, ctx), group_project(g2, ctx)
                    pdyn1, pdyn2 = group_project(g1, m.ah), group_project(g2, m.ah)
                    rho_ctx = pair_reliability(pctx1, pctx2)
                    rho_dyn = pair_reliability(pdyn1, pdyn2)
                    use_dynamic = dynamic_ready and step > 0 and score_dyn > score_ctx
                    selected_score = score_dyn if use_dynamic else score_ctx
                    alpha = 2.0 if step == 0 or step % 50 == 0 or a.proj_mode == "crossbatch_select" else 2.0 * max(0.0, min(1.0, selected_score))
                    selected_projection = (pdyn1 + pdyn2) if use_dynamic else (pctx1 + pctx2)
                    g.add_(selected_projection, alpha=alpha)
                    dynamic_steps += int(use_dynamic)
                    alpha_sum += alpha
                    if dynamic_ready:
                        rho_ctx_sum += rho_ctx
                        rho_dyn_sum += rho_dyn
                        rho_count += 1
                    if step == 0 or step % 50 == 0:
                        score_ctx = rho_ctx
                        if dynamic_ready: score_dyn = rho_dyn
                    else:
                        score_ctx = 0.9 * score_ctx + 0.1 * rho_ctx
                        if dynamic_ready: score_dyn = 0.9 * score_dyn + 0.1 * rho_dyn
                    if step in (0, 25, 50, 99, 100, 200, 399):
                        gate_trace.append([step, round(score_ctx, 6), round(score_dyn, 6), int(use_dynamic), round(alpha, 6), round(rho_ctx, 6), round(rho_dyn, 6), int(dynamic_ready)])
        for o in opts: o.step()''',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, gate_trace=gate_trace, gate_dynamic_frac=dynamic_steps / a.steps, '
        'gate_alpha_mean=alpha_sum / a.steps, '
        'rho_context_mean=(rho_ctx_sum / rho_count if rho_count else None), '
        'rho_dynamic_mean=(rho_dyn_sum / rho_count if rho_count else None), '
        'tag=a.tag, args=vars(a), tokens=need',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

steps = 400 if args.batch == 32 else 100
out = lab / f"microbatch_reliability_b{args.batch}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
modes = ("plain", "context", "dynamic", "energy_gate", "crossbatch_select", "crossbatch_gate")
for mode in modes:
    subprocess.run([
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", str(args.batch), "--acc", "2",
        "--steps", str(steps), "--hlr", "0.05", "--clr", "0.02", "--heps", "1e-5",
        "--proj_mode", mode, "--seed", str(args.seed),
        "--tag", f"xrel_{mode}_b{args.batch}_s{args.seed}", "--out", out.name,
    ], check=True)

rows = [json.loads(s) for s in out.read_text().splitlines()]
assert len(rows) == 6 and len({r["tag"] for r in rows}) == 6
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"] == 819200 for r in rows)
assert rows[5]["args"]["acc"] == 2
for gate in rows[4:6]:
    assert 0 <= gate["gate_dynamic_frac"] <= 1
    assert 0 <= gate["gate_alpha_mean"] <= 2
    assert gate["rho_context_mean"] is not None and gate["rho_dynamic_mean"] is not None
for mode, row in zip(modes, rows):
    print(
        f"MICROBATCH seed={args.seed} batch={args.batch} mode={mode} "
        f"CE={row['scores']['val']:.5f} "
        f"dynamic_fraction={row['gate_dynamic_frac']:.4f} "
        f"alpha_mean={row['gate_alpha_mean']:.4f} "
        f"rho_ctx={row['rho_context_mean']} rho_dyn={row['rho_dynamic_mean']} "
        f"trace={row['gate_trace']}",
        flush=True,
    )
