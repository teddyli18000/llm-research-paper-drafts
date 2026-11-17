"""Pre-registered paired scale confirmation for the new single-head group projector."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[137, 149, 163], required=True)
p.add_argument("--scale", type=int, choices=[16000, 50258], required=True)
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
        'p.add_argument("--ember_b2", type=float, default=0.999)',
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
        '        opts.append(Ember([m.head], lr=a.hlr, beta2=a.ember_b2, eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.projector_alpha and step and step % 50 == 0:\n'
        '            with torch.no_grad():\n'
        '                m.ah.copy_(kmeans(m.head.detach(),64))\n'
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

if args.scale == 16000:
    width, steps, hlr, clr, accum, b2 = 128, 400, 0.03, 0.005, 1, 0.95
else:
    width, steps, hlr, clr, accum, b2 = 64, 800, 0.05, 0.01, 4, 0.999

out = lab / f"projector_scale_V{args.scale}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
specs = [
    ("ember_plain", True, False, False),
    ("ember_centroid", True, True, False),
    ("ember_projector", True, False, True),
    ("adam_centroid", False, True, False),
]
for name, use_ember, use_centroid, use_projector in specs:
    flags = []
    if use_ember:
        flags += ["--ember_head", "1", "--ember_b2", str(b2)]
    if use_centroid:
        flags += ["--clus", "64", "--cinit", "kmeans", "--recl", "50"]
    if use_projector:
        flags += ["--projector_alpha", "2.0"]
    subprocess.run([
        sys.executable, str(runner),
        "--dev", "cpu", "--threads", "2",
        "--V", str(args.scale), "--width", str(width),
        "--layers", "2", "--heads", "4", "--batch", "32", "--L", "64",
        "--acc", str(accum), "--steps", str(steps),
        "--heps", "1e-5", "--hlr", str(0.05 if use_ember else hlr),
        "--clr", str(clr), "--seed", str(args.seed),
        "--tag", f"projector_scale_{name}_V{args.scale}_s{args.seed}",
        "--out", out.name, *flags,
    ], check=True)

rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == 4 and len({r["tag"] for r in rows}) == 4
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
values = {name: r["scores"]["val"] for (name, *_), r in zip(specs, rows)}
for name, val in values.items():
    print(f"PROJECTOR_SCALE V={args.scale} seed={args.seed} arm={name} "
          f"CE={val:.5f} sec={rows[[n for n,*_ in specs].index(name)]['train_s']:.1f}", flush=True)
for name in ("ember_plain", "ember_centroid", "adam_centroid"):
    print(f"PROJECTOR_SCALE_PAIR V={args.scale} seed={args.seed} "
          f"comparison={name} minus_projector={values[name]-values['ember_projector']:+.5f}",
          flush=True)
