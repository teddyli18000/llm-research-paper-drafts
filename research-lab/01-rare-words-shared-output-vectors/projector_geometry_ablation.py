"""Does dynamic group identity matter for parameter-free class-projected Ember?"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True, choices=[223, 227, 229])
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
        'p.add_argument("--projector_group", default="dynamic")',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.projector_alpha else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.projector_alpha and a.projector_group == "context":\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA, "context_groups_6000.npy"))))',
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
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.projector_alpha and step and step % 50 == 0 '
        'and a.projector_group in ("dynamic", "shuffled", "one_time") '
        'and (a.projector_group != "one_time" or step == 50):\n'
        '            with torch.no_grad():\n'
        '                assigned=kmeans(m.head.detach(),64)\n'
        '                if a.projector_group == "shuffled":\n'
        '                    permutation=torch.randperm(V,generator=torch.Generator().manual_seed(5000+step))\n'
        '                    assigned=assigned[permutation.to(assigned.device)]\n'
        '                m.ah.copy_(assigned)\n'
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
        '        for o in opts: o.step()',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

out = lab / f"projector_geometry_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)

arm_specs = [
    ("plain", 0.0, "dynamic", 0.05),
    ("plain_lr2", 0.0, "dynamic", 0.10),
    ("plain_lr3", 0.0, "dynamic", 0.15),
    ("projected_dynamic", 2.0, "dynamic", 0.05),
    ("projected_shuffled", 2.0, "shuffled", 0.05),
    ("projected_random", 2.0, "random", 0.05),
    ("projected_frequency", 2.0, "frequency", 0.05),
    ("projected_context", 2.0, "context", 0.05),
    ("projected_one_time", 2.0, "one_time", 0.05),
]
for name, alpha, group, lr in arm_specs:
    flags = ["--ember_head", "1", "--projector_alpha", str(alpha),
             "--projector_group", group]
    if group == "frequency" and alpha:
        flags += ["--cinit", "freq"]
    subprocess.run([
        sys.executable, str(runner),
        "--dev", "cpu", "--threads", "1", "--V", "6000", "--width", "32",
        "--layers", "2", "--heads", "2", "--batch", "32", "--L", "64",
        "--steps", "400", "--heps", "1e-5", "--hlr", str(lr), "--clr", "0.02",
        "--seed", str(args.seed), "--tag", f"projgeo_{name}_s{args.seed}",
        "--out", out.name, *flags,
    ], check=True)

rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == len(arm_specs) and len({r["tag"] for r in rows}) == len(rows)
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
vals = {name: record["scores"]["val"] for (name, *_), record in zip(arm_specs, rows)}
for name, val in vals.items():
    print(f"PROJECTOR_GEOMETRY seed={args.seed} arm={name} CE={val:.5f} "
          f"vs_plain={vals['plain']-val:+.5f}", flush=True)
