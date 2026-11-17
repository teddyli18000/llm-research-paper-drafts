"""Compare true within-class label smoothing with class-projected Ember head gradients.

The head-only soft-target gradient is an exact analytic correction at the existing
ordinary cross-entropy hidden activations. The full mode instead optimizes real
cluster-smoothed cross entropy, allowing backbone gradients to change too.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True, choices=[353, 359, 367])
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
        'p.add_argument("--projector_alpha", type=float, default=0.0); '
        'p.add_argument("--smooth_eps", type=float, default=0.0); '
        'p.add_argument("--smooth_mode", default="head"); '
        'p.add_argument("--group_source", default="dynamic")',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if (a.projector_alpha or a.smooth_eps) else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if (a.projector_alpha or a.smooth_eps) and a.group_source == "context":\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA,"context_groups_6000.npy"))))',
    ),
    (
        '        lg = F.linear(h, s.H())',
        '        lg = F.linear(h, s.H())\n'
        '        if a.smooth_eps and s.training:\n'
        '            if a.smooth_mode == "head":\n'
        '                s.last_h=h.detach(); s.last_tg=tg.detach()\n'
        '            elif a.smooth_mode == "full":\n'
        '                lp=F.log_softmax(lg,dim=-1)\n'
        '                sums=torch.zeros(len(lp),64,device=lg.device,dtype=lg.dtype).index_add_(1,s.ah,lp)\n'
        '                c=s.ah[tg]\n'
        '                sizes=torch.bincount(s.ah,minlength=64).clamp_min(1)\n'
        '                nll_hard=-lp.gather(1,tg[:,None]).squeeze(1)\n'
        '                nll_class=-sums.gather(1,c[:,None]).squeeze(1)/sizes[c]\n'
        '                return (1-a.smooth_eps)*nll_hard+a.smooth_eps*nll_class',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    if a.ember_head:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '        opts.append(Ember([m.head],lr=a.hlr,beta2=0.95,eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if (a.projector_alpha or a.smooth_eps) and a.group_source=="dynamic" '
        'and step>0 and step%50==0:\n'
        '            with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(),64))\n'
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    ),
    (
        '        for o in opts: o.step()',
        '        if a.projector_alpha:\n'
        '            with torch.no_grad():\n'
        '                g=m.head.grad\n'
        '                pooled=torch.zeros(64,g.shape[1],device=g.device).index_add_(0,m.ah,g)\n'
        '                counts=torch.bincount(m.ah,minlength=64).clamp_min(1)\n'
        '                g.add_(pooled[m.ah]/counts[m.ah,None],alpha=a.projector_alpha)\n'
        '        if a.smooth_eps and a.smooth_mode=="head":\n'
        '            with torch.no_grad():\n'
        '                h=m.last_h; tg=m.last_tg; n=len(tg)\n'
        '                pooled=torch.zeros(64,h.shape[1],device=h.device).index_add_(0,m.ah[tg],h)\n'
        '                sizes=torch.bincount(m.ah,minlength=64).clamp_min(1)\n'
        '                m.head.grad.index_add_(0,tg,a.smooth_eps*h/n)\n'
        '                m.head.grad.sub_(a.smooth_eps*pooled[m.ah]/sizes[m.ah,None]/n)\n'
        '        for o in opts: o.step()',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

out = lab / f"semantic_smoothing_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [
    ("plain", "plain", 0, "head", "dynamic", False),
    ("upstream_centroids", "centroid", 0, "head", "dynamic", False),
    ("projector_dynamic", "projector", 0, "head", "dynamic", True),
    ("projector_context", "projector", 0, "head", "context", True),
    ("smooth_head_context_eps025", "smooth", 0.25, "head", "context", False),
    ("smooth_head_context_eps050", "smooth", 0.50, "head", "context", False),
    ("smooth_head_context_eps075", "smooth", 0.75, "head", "context", False),
    ("smooth_full_context_eps025", "smooth", 0.25, "full", "context", False),
    ("smooth_full_context_eps050", "smooth", 0.50, "full", "context", False),
    ("smooth_full_context_eps075", "smooth", 0.75, "full", "context", False),
    ("smooth_head_dynamic_eps050", "smooth", 0.50, "head", "dynamic", False),
    ("smooth_full_dynamic_eps050", "smooth", 0.50, "full", "dynamic", False),
]
for name, mechanism, eps, mode, source_group, projection in arms:
    flags = ["--ember_head", "1", "--smooth_eps", str(eps),
             "--smooth_mode", mode, "--group_source", source_group]
    if mechanism == "centroid":
        flags += ["--clus", "64", "--cinit", "kmeans", "--recl", "50"]
    if projection:
        flags += ["--projector_alpha", "2.0"]
    subprocess.run([
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--batch", "32", "--L", "64", "--steps", "400",
        "--hlr", "0.05", "--clr", "0.02", "--heps", "1e-5",
        "--seed", str(args.seed), "--tag", f"smooth_{name}_s{args.seed}",
        "--out", out.name, *flags,
    ], check=True)
rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == len(arms) and len(set(r["tag"] for r in rows)) == len(rows)
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
for (name, *_), rec in zip(arms, rows):
    print(f"SEMANTIC_SMOOTH seed={args.seed} arm={name} "
          f"CE={rec['scores']['val']:.5f} train_s={rec['train_s']:.1f}", flush=True)
