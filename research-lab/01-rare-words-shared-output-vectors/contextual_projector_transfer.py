"""Pair learned versus train-context-fixed normalized Ember projectors at two vocab sizes."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[181, 191, 197], required=True)
p.add_argument("--scale", type=int, choices=[16000, 50258], required=True)
args = p.parse_args()

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()

file_name = "context_groups_16000.npy" if args.scale == 16000 else "context_groups_50258.npy"
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
        '        if a.cinit == "freq": asg = freq_classes(K)',
        f'        if a.cinit in ("contextstatic","contextlate"): '
        f'asg = np.load(os.path.join(DATA,"{file_name}"))\n'
        '        elif a.cinit == "freq": asg = freq_classes(K)',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    if a.ember_head:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '        opts.append(Ember([m.head], lr=a.hlr, beta2='+str(0.95 if args.scale==16000 else 0.999)+', eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.projector_alpha and step and step % 50 == 0 and '
        '(a.projector_group == "dynamic" or (a.projector_group == "late" and step >= 200)):\n'
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

width, steps, clr, acc = ((128,400,0.005,1) if args.scale==16000 else (64,800,0.01,4))
out = lab / f"context_proj_V{args.scale}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [
    ("ember_plain", False, False, "dynamic", "kmeans"),
    ("ember_centroid", True, False, "dynamic", "kmeans"),
    ("proj_dynamic", False, True, "dynamic", "kmeans"),
    ("proj_context_static", False, True, "static", "contextstatic"),
    ("proj_context_late", False, True, "late", "contextlate"),
]
for name,centroid,proj,group,init in arms:
    flags=["--ember_head","1"]
    if centroid:
        flags += ["--clus","64","--cinit","kmeans","--recl","50"]
    if proj:
        flags += ["--projector_alpha","2.0","--projector_group",group,"--cinit",init]
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","2",
        "--V",str(args.scale),"--width",str(width),"--layers","2","--heads","4",
        "--L","64","--batch","32","--acc",str(acc),"--steps",str(steps),
        "--heps","1e-5","--hlr","0.05","--clr",str(clr),"--seed",str(args.seed),
        "--tag",f"contextproj_{name}_V{args.scale}_s{args.seed}",
        "--out",out.name,*flags,
    ],check=True)

rows=[json.loads(x) for x in out.read_text().splitlines()]
assert len(rows)==5 and len(set(r["tag"] for r in rows))==5
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
vals={name:record["scores"]["val"] for (name,*_),record in zip(arms,rows)}
for name,ce in vals.items():
    print(f"CONTEXT_PROJECTOR V={args.scale} seed={args.seed} arm={name} "
          f"CE={ce:.5f}",flush=True)
print(f"CONTEXT_PROJECTOR_PAIR V={args.scale} seed={args.seed} "
      f"dynamic_minus_static={vals['proj_dynamic']-vals['proj_context_static']:+.5f} "
      f"static_minus_late={vals['proj_context_static']-vals['proj_context_late']:+.5f}",
      flush=True)
