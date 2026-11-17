"""Matched head-learning-rate sweep for direct and normalized-centroid updates."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[641, 643, 647], required=True)
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
        'p.add_argument("--param_mode", choices=["direct", "scaled"], default="direct"); '
        'p.add_argument("--head_opt", choices=["ember", "adam"], default="ember"); '
        'p.add_argument("--proj_alpha", type=float, default=2.0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64; s.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        s.ah.copy_(torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy"))))',
    ),
    (
        '    def H(s):\n        if not s.uh: return s.head\n        C = s.Clast if s.crev else (torch.zeros_like(s.Ch) if s.coff else s.Ch)\n        return s.head + C[s.ah]',
        '''    def H(s):
        if a.param_mode == "scaled":
            counts = torch.bincount(s.ah, minlength=64).clamp_min(1).to(s.head.dtype)
            row_scale = a.proj_alpha ** 0.5 / counts[s.ah].sqrt()
            return s.head + row_scale[:, None] * s.Ch[s.ah]
        if not s.uh: return s.head
        C = s.Clast if s.crev else (torch.zeros_like(s.Ch) if s.coff else s.Ch)
        return s.head + C[s.ah]''',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))\n    opts.append(torch.optim.Adam(groups, betas=(a.b1, a.b2), eps=a.eps))',
        '''    groups = [g for g in groups if all(p is not m.head and p is not m.Ch for p in g["params"])]
    opts.append(torch.optim.Adam(groups, betas=(a.b1, a.b2), eps=a.eps))
    out_params = [m.head, m.Ch] if a.param_mode == "scaled" else [m.head]
    if a.head_opt == "ember":
        opts.append(Ember(out_params, lr=a.hlr, beta2=0.95, eps=1e-8))
    else:
        opts.append(torch.optim.Adam(out_params, lr=a.hlr, betas=(a.b1, a.b2), eps=a.eps))''',
    ),
    (
        'def main():',
        '''def group_project(g, assignment):
    sums = torch.zeros(64, g.shape[1], device=g.device, dtype=g.dtype).index_add_(0, assignment, g)
    counts = torch.bincount(assignment, minlength=64).clamp_min(1).to(g.dtype)
    return sums[assignment] / counts[assignment, None]


def main():''',
    ),
    (
        '        for o in opts: o.step()',
        '''        projected_to_raw = None
        if a.param_mode == "direct":
            raw_head_grad = m.head.grad.detach().clone()
            projected_head_grad = raw_head_grad + a.proj_alpha * group_project(raw_head_grad, m.ah)
            projected_to_raw = float((projected_head_grad.norm() / raw_head_grad.norm().clamp_min(1e-30)).item())
            m.head.grad.copy_(projected_head_grad)
        head_before = m.H().detach().clone()
        for o in opts: o.step()
        head_update_norm = float((m.H().detach() - head_before).norm().item())
        if projected_to_raw is not None:
            projected_norm_sum += projected_to_raw
            projected_norm_count += 1
        head_update_norm_sum += head_update_norm''',
    ),
    (
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
        '    projected_norm_sum = 0.0; projected_norm_count = 0; head_update_norm_sum = 0.0\n'
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, projected_to_raw_norm_mean=(projected_norm_sum / projected_norm_count if projected_norm_count else None), '
        'head_update_norm_mean=(head_update_norm_sum / a.steps), tag=a.tag, args=vars(a), tokens=need',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

steps = 400 if args.batch == 32 else 100
out = lab / f"normalized_centroid_lr_sweep_b{args.batch}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = []
for lr in (0.025, 0.05, 0.10):
    for optimizer in ("ember", "adam"):
        for mode in ("direct", "scaled"):
            arms.append((f"{mode}_{optimizer}_lr{lr}", mode, optimizer, lr))

for name, param_mode, head_opt, head_lr in arms:
    cmd = [
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", str(args.batch), "--acc", "2",
        "--steps", str(steps), "--hlr", str(head_lr), "--clr", "0.02", "--heps", "1e-5",
        "--param_mode", param_mode, "--head_opt", head_opt, "--proj_alpha", "2.0",
        "--seed", str(args.seed), "--tag", f"lr_sweep_{name}_b{args.batch}_s{args.seed}",
        "--out", out.name,
    ]
    subprocess.run(cmd, check=True)

rows = [json.loads(s) for s in out.read_text().splitlines()]
assert len(rows) == len(arms) and len({r["tag"] for r in rows}) == len(arms)
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"] == 819200 for r in rows)
assert all(r["head_update_norm_mean"] > 0 for r in rows)
assert all(r["projected_to_raw_norm_mean"] is not None for r in rows if r["args"]["param_mode"] == "direct")
for row in rows:
    print(
        f"LR_SWEEP seed={args.seed} batch={args.batch} "
        f"mode={row['args']['param_mode']} optimizer={row['args']['head_opt']} "
        f"lr={row['args']['hlr']:.3f} CE={row['scores']['val']:.5f} "
        f"proj_norm_ratio={row['projected_to_raw_norm_mean']} "
        f"head_update_norm={row['head_update_norm_mean']:.6f} bins={row['bins']}",
        flush=True,
    )
