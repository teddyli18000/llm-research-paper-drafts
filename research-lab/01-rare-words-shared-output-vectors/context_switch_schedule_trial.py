"""Compare absolute optimizer-step thresholds for context-to-dynamic regrouping."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[569, 571, 577], required=True)
args = p.parse_args()

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()
patches = [
    (
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F',
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nimport ember\nfrom ember import Ember',
    ),
    (
        'p.add_argument("--coupled", type=int, default=0)',
        'p.add_argument("--coupled", type=int, default=0); '
        'p.add_argument("--proj_mode", choices=["context", "context_late"], default="context"); '
        'p.add_argument("--switch_step", type=int, default=200); '
        'p.add_argument("--proj_alpha", type=float, default=2.0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.proj_mode in ("context", "context_late") else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.proj_mode in ("context", "context_late"):\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy"))))',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '    opts.append(Ember([m.head], lr=a.hlr, beta2=0.95, eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
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
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.proj_mode == "context_late" and step >= a.switch_step and (step - a.switch_step) % a.recl == 0:\n'
        '            with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(), 64))\n'
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    ),
    (
        '        for o in opts: o.step()',
        '''        raw_head_grad = m.head.grad.detach().clone()
        projected_head_grad = raw_head_grad + a.proj_alpha * group_project(raw_head_grad, m.ah)
        projected_to_raw = float((projected_head_grad.norm() / raw_head_grad.norm().clamp_min(1e-30)).item())
        m.head.grad.copy_(projected_head_grad)
        head_before = m.head.detach().clone()
        for o in opts: o.step()
        head_update_norm = float((m.head.detach() - head_before).norm().item())''',
    ),
    (
        '    regroup_records = []\n    t0 = time.time(); X = torch.tensor(rows).to(D)',
        '    regroup_records = []; projected_norm_sum = 0.0; head_update_norm_sum = 0.0\n'
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '        head_update_norm = float((m.head.detach() - head_before).norm().item())',
        '        head_update_norm = float((m.head.detach() - head_before).norm().item())\n'
        '        projected_norm_sum += projected_to_raw\n'
        '        head_update_norm_sum += head_update_norm',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, projected_to_raw_norm_mean=(projected_norm_sum / a.steps), '
        'head_update_norm_mean=(head_update_norm_sum / a.steps), tag=a.tag, args=vars(a), tokens=need',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

steps = 400
out = lab / f"context_switch_schedule_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [
    ("fixed_context", "context", 0),
    ("context_to_dynamic_step150", "context_late", 150),
    ("context_to_dynamic_step200", "context_late", 200),
    ("context_to_dynamic_step250", "context_late", 250),
]
for name, mode, switch_step in arms:
    cmd = [
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", "32", "--acc", "2",
        "--steps", str(steps), "--hlr", "0.05", "--clr", "0.02", "--heps", "1e-5",
        "--proj_mode", mode, "--proj_alpha", "2.0", "--recl", "50",
        "--seed", str(args.seed), "--tag", f"switch_{name}_s{args.seed}",
        "--out", out.name,
    ]
    if mode == "context_late":
        cmd += ["--switch_step", str(switch_step)]
    subprocess.run(cmd, check=True)

rows = [json.loads(s) for s in out.read_text().splitlines()]
assert len(rows) == len(arms) and len({r["tag"] for r in rows}) == len(arms)
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"] == 819200 for r in rows)
assert all(r["projected_to_raw_norm_mean"] > 0 and r["head_update_norm_mean"] > 0 for r in rows)
for row in rows:
    print(
        f"SWITCH seed={args.seed} tag={row['tag']} "
        f"mode={row['args']['proj_mode']} switch_step={row['args']['switch_step']} "
        f"CE={row['scores']['val']:.5f} "
        f"proj_norm_ratio={row['projected_to_raw_norm_mean']:.6f} "
        f"head_update_norm={row['head_update_norm_mean']:.6f} bins={row['bins']}",
        flush=True,
    )
