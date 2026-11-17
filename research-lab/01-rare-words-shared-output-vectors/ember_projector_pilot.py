"""New candidate: normalized dynamic class-gradient projection before official Ember."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, choices=[211, 307, 401], required=True)
args = parser.parse_args()

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
        'p.add_argument("--projector_alpha", type=float, default=0.0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.projector_alpha else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
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
        '        if a.projector_alpha and step and step % 50 == 0:\n'
        '            with torch.no_grad():\n'
        '                m.ah.copy_(kmeans(m.head.detach(), 64))\n'
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

out = lab / f"ember_projector_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)

arms = [
    ("plain", 0, 0.0),
    ("original_groups", 64, 0.0),
    ("projector_a05", 0, 0.5),
    ("projector_a10", 0, 1.0),
    ("projector_a20", 0, 2.0),
]
for name, groups, alpha in arms:
    flags = ["--ember_head", "1", "--projector_alpha", str(alpha)]
    if groups:
        flags += ["--clus", str(groups), "--cinit", "kmeans", "--recl", "50"]
    subprocess.run([
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", "32", "--steps", "400",
        "--heps", "1e-5", "--hlr", "0.05", "--clr", "0.02",
        "--seed", str(args.seed),
        "--tag", f"projected_ember_{name}_s{args.seed}",
        "--out", out.name, *flags,
    ], check=True)

rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == 5 and len({x["tag"] for x in rows}) == 5
assert all(math.isfinite(x["scores"]["val"]) for x in rows)
values = {name: row["scores"]["val"] for (name, _, _), row in zip(arms, rows)}
for name, ce in values.items():
    print(f"PROJECTED_EMBER seed={args.seed} arm={name} CE={ce:.5f} "
          f"plain_gain={values['plain']-ce:.5f}", flush=True)
