"""Test whether projected LM-head gradients should share Ember's factored second moments."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[521, 523, 541], required=True)
p.add_argument("--batch", type=int, choices=[32, 128], required=True)
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
        'p.add_argument("--proj_mode", choices=["plain", "centroid", "dynamic", "context", "context_late"], default="plain"); '
        'p.add_argument("--moment_route", choices=["shared", "raw_stats"], default="shared"); '
        'p.add_argument("--switch_step", type=int, default=80); '
        'p.add_argument("--proj_alpha", type=float, default=2.0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.proj_mode in ("dynamic", "context", "context_late") else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.proj_mode in ("context", "context_late"):\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy"))))\n'
        '        elif a.proj_mode == "dynamic":\n'
        '            s.ah.copy_(kmeans(s.head.detach(), 64))',
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


_original_ember_update = ember.ember_update


def routed_ember_update(p_l, g_l, state, lr, beta2, eps, wd, pg=None):
    raw = getattr(p_l, "_raw_stats_grad", None)
    if raw is None:
        return _original_ember_update(p_l, g_l, state, lr, beta2, eps, wd, pg)
    Vloc, D = p_l.shape
    if not state:
        state["t"] = 0
        state["r"] = torch.zeros(Vloc, dtype=torch.float32, device=p_l.device)
        state["c"] = torch.zeros(D, dtype=torch.float32, device=p_l.device)
        state["s"] = torch.zeros((), dtype=torch.float32, device=p_l.device)
        state["n_rows"] = float(Vloc)
    state["t"] += 1
    t, r, c, s, n_rows = state["t"], state["r"], state["c"], state["s"], state["n_rows"]
    g_stats = raw.float()
    col_sum = g_stats.square().sum(dim=0)
    r.mul_(beta2).add_(g_stats.square().mean(dim=1), alpha=1 - beta2)
    c.mul_(beta2).add_(col_sum / n_rows, alpha=1 - beta2)
    s.mul_(beta2).add_(col_sum.sum() / (n_rows * D), alpha=1 - beta2)
    bc = 1 - beta2 ** t
    v_hat = r.unsqueeze(1) * c.unsqueeze(0) / (s.clamp(min=1e-30) * bc)
    denom = v_hat.sqrt_().add_(eps)
    if wd != 0.0:
        p_l.mul_(1 - lr * wd)
    p_l.addcdiv_(g_l, denom.to(p_l.dtype), value=-lr)
    del p_l._raw_stats_grad


ember.ember_update = routed_ember_update


def main():''',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.proj_mode == "dynamic" and step > 0 and step % a.recl == 0:\n'
        '            with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(), 64))\n'
        '        if a.proj_mode == "context_late" and step >= a.switch_step and (step - a.switch_step) % a.recl == 0:\n'
        '            with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(), 64))\n'
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    ),
    (
        '        for o in opts: o.step()',
        '''        projected_to_raw = None
        if a.proj_mode in ("dynamic", "context", "context_late"):
            with torch.no_grad():
                raw_head_grad = m.head.grad.detach().clone()
                projected_head_grad = raw_head_grad + a.proj_alpha * group_project(raw_head_grad, m.ah)
                projected_to_raw = float((projected_head_grad.norm() / raw_head_grad.norm().clamp_min(1e-30)).item())
                m.head.grad.copy_(projected_head_grad)
                if a.moment_route == "raw_stats":
                    m.head._raw_stats_grad = raw_head_grad
        head_before = m.head.detach().clone()
        for o in opts: o.step()
        head_update_norm = float((m.head.detach() - head_before).norm().item())''',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, projected_to_raw_norm_mean=(projected_norm_sum / projected_norm_count if projected_norm_count else None), '
        'head_update_norm_mean=(head_update_norm_sum / a.steps), tag=a.tag, args=vars(a), tokens=need',
    ),
    (
        '    regroup_records = []\n    t0 = time.time(); X = torch.tensor(rows).to(D)',
        '    regroup_records = []; projected_norm_sum = 0.0; projected_norm_count = 0; head_update_norm_sum = 0.0\n'
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '        head_update_norm = float((m.head.detach() - head_before).norm().item())',
        '        head_update_norm = float((m.head.detach() - head_before).norm().item())\n'
        '        head_update_norm_sum += head_update_norm\n'
        '        if projected_to_raw is not None:\n'
        '            projected_norm_sum += projected_to_raw\n'
        '            projected_norm_count += 1',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

steps = 400 if args.batch == 32 else 100
switch_step = steps // 5
out = lab / f"moment_routing_b{args.batch}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [
    ("plain_ember", "plain", "shared", 0),
    ("upstream_centroid_ember", "centroid", "shared", 64),
    ("dynamic_shared_moments", "dynamic", "shared", 0),
    ("dynamic_raw_second_moment", "dynamic", "raw_stats", 0),
    ("context_shared_moments", "context", "shared", 0),
    ("context_raw_second_moment", "context", "raw_stats", 0),
    ("context_to_dynamic_shared", "context_late", "shared", 0),
    ("context_to_dynamic_raw_second_moment", "context_late", "raw_stats", 0),
]
for name, mode, route, clus in arms:
    cmd = [
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", str(args.batch), "--acc", "2",
        "--steps", str(steps), "--hlr", "0.05", "--clr", "0.02", "--heps", "1e-5",
        "--proj_mode", mode, "--moment_route", route, "--proj_alpha", "2.0",
        "--seed", str(args.seed), "--tag", f"moment_{name}_b{args.batch}_s{args.seed}",
        "--out", out.name,
    ]
    if clus:
        cmd += ["--clus", str(clus), "--cinit", "kmeans", "--recl", "50"]
    if mode == "context_late":
        cmd += ["--switch_step", str(switch_step), "--recl", "50"]
    subprocess.run(cmd, check=True)

rows = [json.loads(s) for s in out.read_text().splitlines()]
assert len(rows) == len(arms) and len({r["tag"] for r in rows}) == len(arms)
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"] == 819200 for r in rows)
assert rows[0]["args"]["acc"] == 2
for mode, route in [("dynamic", "raw_stats"), ("context", "raw_stats"), ("context_late", "raw_stats")]:
    row = next(r for r in rows if r["args"]["proj_mode"] == mode and r["args"]["moment_route"] == route)
    assert row["projected_to_raw_norm_mean"] is not None
    assert row["head_update_norm_mean"] > 0
for row in rows:
    print(
        f"MOMENT_ROUTE seed={args.seed} batch={args.batch} mode={row['args']['proj_mode']} "
        f"route={row['args']['moment_route']} CE={row['scores']['val']:.5f} "
        f"proj_norm_ratio={row['projected_to_raw_norm_mean']} "
        f"head_update_norm={row['head_update_norm_mean']:.6f} "
        f"bins={row['bins']}",
        flush=True,
    )
