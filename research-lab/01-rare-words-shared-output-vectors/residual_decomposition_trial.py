"""Separate prediction-side Pp and target-side Py terms in the class-projected LM-head gradient."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[547, 557, 563], required=True)
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
        'p.add_argument("--ember_head", type=int, default=0); '
        'p.add_argument("--residual_mode", choices=["none", "full", "pred_only", "target_only"], default="none"); '
        'p.add_argument("--group_source", choices=["context", "dynamic", "context_late"], default="context"); '
        'p.add_argument("--projector_alpha", type=float, default=0.0); '
        'p.add_argument("--smooth_eps", type=float, default=0.0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if (a.residual_mode != "none" or a.smooth_eps) else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if (a.residual_mode != "none" or a.smooth_eps) and a.group_source in ("context", "context_late"):\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy"))))\n'
        '        elif a.residual_mode != "none" and a.group_source == "dynamic":\n'
        '            s.ah.copy_(kmeans(s.head.detach(), 64))',
    ),
    (
        '        lg = F.linear(h, s.H())',
        '''        lg = F.linear(h, s.H())
        if a.residual_mode != "none" and s.training:
            s.last_batches.append((h.detach(), tg.detach(), F.softmax(lg, dim=-1).detach()))
        if a.smooth_eps and s.training:
            lp = F.log_softmax(lg, dim=-1)
            sums = torch.zeros(len(lp), 64, device=lg.device, dtype=lg.dtype).index_add_(1, s.ah, lp)
            c = s.ah[tg]
            sizes = torch.bincount(s.ah, minlength=64).clamp_min(1)
            nll_hard = -lp.gather(1, tg[:, None]).squeeze(1)
            nll_class = -sums.gather(1, c[:, None]).squeeze(1) / sizes[c]
            return (1 - a.smooth_eps) * nll_hard + a.smooth_eps * nll_class''',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    if a.ember_head:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '        opts.append(Ember([m.head], lr=a.hlr, beta2=0.95, eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '        x = X[step * a.batch:(step + 1) * a.batch]',
        '        x = X[step * a.batch:(step + 1) * a.batch]\n'
        '        m.last_batches = []',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.residual_mode != "none" and a.group_source == "dynamic" and step > 0 and step % 50 == 0:\n'
        '            with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(), 64))\n'
        '        if a.residual_mode != "none" and a.group_source == "context_late" and step >= 200 and (step - 200) % 50 == 0:\n'
        '            with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(), 64))\n'
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    ),
    (
        '        for o in opts: o.step()',
        '''        if a.residual_mode != "none":
            with torch.no_grad():
                g = m.head.grad
                raw = g.detach().clone()
                pred_corr = torch.zeros_like(g)
                target_corr = torch.zeros_like(g)
                counts = torch.bincount(m.ah, minlength=64).clamp_min(1)
                n_total = x.shape[0] * (x.shape[1] - 1)
                for h, tg, prob in m.last_batches:
                    group_prob = torch.zeros(len(tg), 64, device=prob.device, dtype=prob.dtype).index_add_(1, m.ah, prob)
                    pred_group = group_prob.T @ h / n_total
                    pred_corr.add_(pred_group[m.ah] / counts[m.ah, None])
                    target_group = torch.zeros(64, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, m.ah[tg], h)
                    target_corr.add_(target_group[m.ah] / counts[m.ah, None] / n_total)
                full_corr = pred_corr - target_corr
                matrix_corr = group_project(raw, m.ah)
                error = float((matrix_corr - full_corr).abs().max().item())
                assert error < 2e-5, error
                decomposition_error_max = max(decomposition_error_max, error)
                pred_norm_sum += float(pred_corr.norm().item())
                target_norm_sum += float(target_corr.norm().item())
                full_norm_sum += float(full_corr.norm().item())
                cosine_sum += float(((pred_corr * target_corr).sum() / (pred_corr.norm() * target_corr.norm()).clamp_min(1e-30)).item())
                if a.residual_mode == "full":
                    g.add_(full_corr, alpha=a.projector_alpha)
                elif a.residual_mode == "pred_only":
                    g.add_(pred_corr, alpha=a.projector_alpha)
                elif a.residual_mode == "target_only":
                    g.sub_(target_corr, alpha=a.projector_alpha)
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
        '    full_norm_sum = 0.0; cosine_sum = 0.0; decomposition_error_max = 0.0\n'
        '    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, pred_correction_norm_mean=(pred_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'target_correction_norm_mean=(target_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'full_correction_norm_mean=(full_norm_sum / a.steps if a.residual_mode != "none" else None), '
        'pred_target_cosine_mean=(cosine_sum / a.steps if a.residual_mode != "none" else None), '
        'decomposition_error_max=decomposition_error_max, tag=a.tag, args=vars(a), tokens=need',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

steps = 400 if args.batch == 32 else 100
out = lab / f"residual_decomposition_b{args.batch}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [
    ("plain_ember", "none", "context", 0.0, 0.0, 0),
    ("upstream_centroid_ember", "none", "context", 0.0, 0.0, 64),
    ("context_full_projector", "full", "context", 2.0, 0.0, 0),
    ("context_prediction_only", "pred_only", "context", 2.0, 0.0, 0),
    ("context_target_only", "target_only", "context", 2.0, 0.0, 0),
    ("dynamic_full_projector", "full", "dynamic", 2.0, 0.0, 0),
    ("context_to_dynamic_step200", "full", "context_late", 2.0, 0.0, 0),
    ("context_label_smoothing_eps_2_3", "none", "context", 0.0, 2.0 / 3.0, 0),
]
for name, residual_mode, group_source, alpha, smooth_eps, clus in arms:
    cmd = [
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", str(args.batch), "--acc", "2",
        "--steps", str(steps), "--hlr", "0.05", "--clr", "0.02", "--heps", "1e-5",
        "--ember_head", "1", "--residual_mode", residual_mode, "--group_source", group_source,
        "--projector_alpha", str(alpha), "--smooth_eps", str(smooth_eps),
        "--seed", str(args.seed), "--tag", f"residual_{name}_b{args.batch}_s{args.seed}",
        "--out", out.name,
    ]
    if clus:
        cmd += ["--clus", str(clus), "--cinit", "kmeans", "--recl", "50"]
    subprocess.run(cmd, check=True)

rows = [json.loads(s) for s in out.read_text().splitlines()]
assert len(rows) == len(arms) and len({r["tag"] for r in rows}) == len(arms)
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"] == 819200 for r in rows)
for row in rows:
    if row["args"]["residual_mode"] != "none":
        assert row["decomposition_error_max"] < 2e-5
        assert row["pred_correction_norm_mean"] > 0
        assert row["target_correction_norm_mean"] > 0
for row in rows:
    print(
        f"RESIDUAL seed={args.seed} batch={args.batch} tag={row['tag']} "
        f"mode={row['args']['residual_mode']} group={row['args']['group_source']} "
        f"CE={row['scores']['val']:.5f} pred_norm={row['pred_correction_norm_mean']} "
        f"target_norm={row['target_correction_norm_mean']} "
        f"pred_target_cos={row['pred_target_cosine_mean']} "
        f"decomp_error={row['decomposition_error_max']} bins={row['bins']}",
        flush=True,
    )
