"""Compare the flat-head projector with graph-diffusion proper-score losses."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[683, 691, 701], required=True)
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
        'p.add_argument("--graph_score", choices=["none", "p_limit", "diffusion", "resistance"], default="none"); '
        'p.add_argument("--graph_weight", type=float, default=1.0); '
        'p.add_argument("--graph_tau", type=float, default=0.02); '
        'p.add_argument("--direct_projector", type=int, default=0); '
        'p.add_argument("--projector_alpha", type=float, default=2.0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if (a.graph_score != "none" or a.direct_projector) else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.graph_score != "none" or a.direct_projector:\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy"))))',
    ),
    (
        '        return F.cross_entropy(lg, tg, reduction="none")',
        '''        base = F.cross_entropy(lg, tg, reduction="none")
        if a.graph_score != "none" and s.training:
            graph_logits = F.linear(h.detach(), s.H())
            prob = F.softmax(graph_logits, dim=-1)
            counts = torch.bincount(s.ah, minlength=64).clamp_min(1).to(prob.dtype)
            group_prob = torch.zeros(len(tg), 64, device=prob.device, dtype=prob.dtype).index_add_(1, s.ah, prob)
            target_group = s.ah[tg]
            group_target = F.one_hot(target_group, num_classes=64).to(prob.dtype)
            group_resid = group_prob - group_target
            p_score = (group_resid.square() / counts).sum(dim=1)
            residual = prob.clone()
            residual[torch.arange(len(tg), device=tg.device), tg] -= 1
            projected = group_resid[:, s.ah] / counts[s.ah]
            within = residual - projected
            within_sq = torch.zeros(len(tg), 64, device=prob.device, dtype=prob.dtype).index_add_(1, s.ah, within.square())
            if a.graph_score == "p_limit":
                score = p_score
            elif a.graph_score == "resistance":
                score = (within_sq / counts).sum(dim=1)
            else:
                decay = torch.exp(-a.graph_tau * counts / 2)
                score = p_score + (within_sq * decay).sum(dim=1)
            return base + a.graph_weight * score
        return base''',
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
        '        for o in opts: o.step()',
        '''        projected_to_raw = None
        if a.direct_projector:
            raw_head_grad = m.head.grad.detach().clone()
            projected_head_grad = raw_head_grad + a.projector_alpha * group_project(raw_head_grad, m.ah)
            projected_to_raw = float((projected_head_grad.norm() / raw_head_grad.norm().clamp_min(1e-30)).item())
            m.head.grad.copy_(projected_head_grad)
        head_before = m.H().detach().clone()
        for o in opts: o.step()
        head_update_norm = float((m.H().detach() - head_before).norm().item())
        head_update_norm_sum += head_update_norm
        if projected_to_raw is not None:
            projected_norm_sum += projected_to_raw
            projected_norm_count += 1''',
    ),
    (
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
        '    head_update_norm_sum = 0.0; projected_norm_sum = 0.0; projected_norm_count = 0\n'
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, head_update_norm_mean=(head_update_norm_sum / a.steps), '
        'projected_to_raw_norm_mean=(projected_norm_sum / projected_norm_count if projected_norm_count else None), '
        'tag=a.tag, args=vars(a), tokens=need',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

steps = 400 if args.batch == 32 else 100
out = lab / f"graph_diffusion_score_b{args.batch}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [
    ("plain_ember", "none", 0),
    ("direct_projector_ember", "none", 1),
    ("graph_p_limit", "p_limit", 0),
    ("graph_diffusion_tau_002", "diffusion", 0),
    ("graph_resistance", "resistance", 0),
]
for name, score_mode, direct in arms:
    cmd = [
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", str(args.batch), "--acc", "2",
        "--steps", str(steps), "--hlr", "0.05", "--clr", "0.02", "--heps", "1e-5",
        "--graph_score", score_mode, "--graph_weight", "1.0", "--graph_tau", "0.02",
        "--direct_projector", str(direct), "--projector_alpha", "2.0",
        "--seed", str(args.seed), "--tag", f"graph_{name}_b{args.batch}_s{args.seed}",
        "--out", out.name,
    ]
    subprocess.run(cmd, check=True)

rows = [json.loads(s) for s in out.read_text().splitlines()]
assert len(rows) == len(arms) and len({r["tag"] for r in rows}) == len(arms)
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"] == 819200 for r in rows)
assert all(r["head_update_norm_mean"] > 0 for r in rows)
assert all(r["projected_to_raw_norm_mean"] is not None for r in rows if r["args"]["direct_projector"])
for row in rows:
    print(
        f"GRAPH_SCORE seed={args.seed} batch={args.batch} tag={row['tag']} "
        f"mode={row['args']['graph_score']} direct={row['args']['direct_projector']} "
        f"CE={row['scores']['val']:.5f} "
        f"proj_norm_ratio={row['projected_to_raw_norm_mean']} "
        f"head_update_norm={row['head_update_norm_mean']:.6f} bins={row['bins']}",
        flush=True,
    )
