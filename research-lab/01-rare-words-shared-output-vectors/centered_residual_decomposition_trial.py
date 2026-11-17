"""Separate centered prediction/target residuals from the vocabulary-uniform mode."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[617, 619, 631], required=True)
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
        'p.add_argument("--residual_mode", choices=["none", "full", "pred_raw", "target_raw", "pred_centered", "target_centered", "uniform_plus", "uniform_minus"], default="none"); '
        'p.add_argument("--projector_alpha", type=float, default=2.0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.residual_mode != "none" else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.residual_mode != "none":\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy"))))',
    ),
    (
        '        lg = F.linear(h, s.H())',
        '''        lg = F.linear(h, s.H())
        if a.residual_mode != "none" and s.training:
            s.last_batches.append((h.detach(), tg.detach(), F.softmax(lg, dim=-1).detach()))''',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '    opts.append(Ember([m.head], lr=a.hlr, beta2=0.95, eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '        x = X[step * a.batch:(step + 1) * a.batch]',
        '        x = X[step * a.batch:(step + 1) * a.batch]\n'
        '        m.last_batches = []',
    ),
    (
        '        for o in opts: o.step()',
        '''        if a.residual_mode != "none":
            with torch.no_grad():
                g = m.head.grad
                raw = g.detach().clone()
                pred_corr = torch.zeros_like(g)
                target_corr = torch.zeros_like(g)
                hidden_sum = torch.zeros(g.shape[1], device=g.device, dtype=g.dtype)
                counts = torch.bincount(m.ah, minlength=64).clamp_min(1).to(g.dtype)
                n_total = x.shape[0] * (x.shape[1] - 1)
                for h, tg, prob in m.last_batches:
                    group_prob = torch.zeros(len(tg), 64, device=prob.device, dtype=prob.dtype).index_add_(1, m.ah, prob)
                    pred_group = group_prob.T @ h / n_total
                    pred_corr.add_(pred_group[m.ah] / counts[m.ah, None])
                    target_group = torch.zeros(64, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, m.ah[tg], h)
                    target_corr.add_(target_group[m.ah] / counts[m.ah, None] / n_total)
                    hidden_sum.add_(h.sum(dim=0))
                mean_hidden = hidden_sum / n_total
                uniform_corr = mean_hidden[None, :].expand_as(g) / g.shape[0]
                pred_centered = pred_corr - uniform_corr
                target_centered = target_corr - uniform_corr
                full_corr = pred_corr - target_corr
                matrix_corr = group_project(raw, m.ah)
                error = float((matrix_corr - full_corr).abs().max().item())
                centered_error = float((pred_centered - target_centered - full_corr).abs().max().item())
                assert error < 2e-5 and centered_error < 2e-5, (error, centered_error)
                assert (pred_centered.sum(dim=0).abs().max().item() < 2e-5)
                assert (target_centered.sum(dim=0).abs().max().item() < 2e-5)
                pred_norm_sum += float(pred_corr.norm().item())
                target_norm_sum += float(target_corr.norm().item())
                pred_centered_norm_sum += float(pred_centered.norm().item())
                target_centered_norm_sum += float(target_centered.norm().item())
                uniform_norm_sum += float(uniform_corr.norm().item())
                full_norm_sum += float(full_corr.norm().item())
                raw_cos_sum += float(((pred_corr * target_corr).sum() / (pred_corr.norm() * target_corr.norm()).clamp_min(1e-30)).item())
                centered_cos_sum += float(((pred_centered * target_centered).sum() / (pred_centered.norm() * target_centered.norm()).clamp_min(1e-30)).item())
                decomposition_error_max = max(decomposition_error_max, error, centered_error)
                if a.residual_mode == "full":
                    g.add_(full_corr, alpha=a.projector_alpha)
                elif a.residual_mode == "pred_raw":
                    g.add_(pred_corr, alpha=a.projector_alpha)
                elif a.residual_mode == "target_raw":
                    g.sub_(target_corr, alpha=a.projector_alpha)
                elif a.residual_mode == "pred_centered":
                    g.add_(pred_centered, alpha=a.projector_alpha)
                elif a.residual_mode == "target_centered":
                    g.sub_(target_centered, alpha=a.projector_alpha)
                elif a.residual_mode == "uniform_plus":
                    g.add_(uniform_corr, alpha=a.projector_alpha)
                elif a.residual_mode == "uniform_minus":
                    g.sub_(uniform_corr, alpha=a.projector_alpha)
        for o in opts: o.step()''',
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
        '    regroup_records = []\n    t0 = time.time(); X = torch.tensor(rows).to(D)',
        '    regroup_records = []; pred_norm_sum = 0.0; target_norm_sum = 0.0\n'
        '    pred_centered_norm_sum = 0.0; target_centered_norm_sum = 0.0\n'
        '    uniform_norm_sum = 0.0; full_norm_sum = 0.0\n'
        '    raw_cos_sum = 0.0; centered_cos_sum = 0.0; decomposition_error_max = 0.0\n'
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, pred_norm_mean=(pred_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'target_norm_mean=(target_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'pred_centered_norm_mean=(pred_centered_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'target_centered_norm_mean=(target_centered_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'uniform_norm_mean=(uniform_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'full_norm_mean=(full_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'raw_pred_target_cosine_mean=(raw_cos_sum / a.steps if a.residual_mode != "none" else None), '
        'centered_pred_target_cosine_mean=(centered_cos_sum / a.steps if a.residual_mode != "none" else None), '
        'decomposition_error_max=decomposition_error_max, tag=a.tag, args=vars(a), tokens=need',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

steps = 400 if args.batch == 32 else 100
out = lab / f"centered_residual_b{args.batch}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [
    ("plain_ember", "none"),
    ("full_projector", "full"),
    ("prediction_only_raw", "pred_raw"),
    ("target_only_raw", "target_raw"),
    ("prediction_only_centered", "pred_centered"),
    ("target_only_centered", "target_centered"),
    ("uniform_plus", "uniform_plus"),
    ("uniform_minus", "uniform_minus"),
]
for name, mode in arms:
    cmd = [
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", str(args.batch), "--acc", "2",
        "--steps", str(steps), "--hlr", "0.05", "--clr", "0.02", "--heps", "1e-5",
        "--residual_mode", mode, "--projector_alpha", "2.0",
        "--seed", str(args.seed), "--tag", f"centered_{name}_b{args.batch}_s{args.seed}",
        "--out", out.name,
    ]
    subprocess.run(cmd, check=True)

rows = [json.loads(s) for s in out.read_text().splitlines()]
assert len(rows) == len(arms) and len({r["tag"] for r in rows}) == len(arms)
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"] == 819200 for r in rows)
for row in rows:
    if row["args"]["residual_mode"] != "none":
        assert row["decomposition_error_max"] < 2e-5
        assert row["pred_centered_norm_mean"] > 0
        assert row["target_centered_norm_mean"] > 0
for row in rows:
    print(
        f"CENTERED_RESIDUAL seed={args.seed} batch={args.batch} tag={row['tag']} "
        f"mode={row['args']['residual_mode']} CE={row['scores']['val']:.5f} "
        f"pred_norm={row['pred_norm_mean']} target_norm={row['target_norm_mean']} "
        f"pred_centered_norm={row['pred_centered_norm_mean']} "
        f"target_centered_norm={row['target_centered_norm_mean']} "
        f"uniform_norm={row['uniform_norm_mean']} "
        f"raw_cos={row['raw_pred_target_cosine_mean']} "
        f"centered_cos={row['centered_pred_target_cosine_mean']} "
        f"decomp_error={row['decomposition_error_max']} bins={row['bins']}",
        flush=True,
    )
