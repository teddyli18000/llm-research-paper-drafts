"""Equal-token batch-size test for single-head projected Ember and split-centroid Ember."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[269, 271, 277], required=True)
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
        '        opts.append(Ember([m.head],lr=a.hlr,beta2=0.95,eps=1e-8))\n'
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

steps = 400 if args.batch == 32 else 100
acc = 1 if args.batch == 32 else 4
out = lab / f"projector_batch{args.batch}_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms = [("plain",False,False),("centroid",True,False),("projector",False,True)]
for name, centroid, projected in arms:
    flags = ["--ember_head","1"]
    if centroid:
        flags += ["--clus","64","--cinit","kmeans","--recl","50"]
    if projected:
        flags += ["--projector_alpha","2.0"]
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","1",
        "--V","6000","--width","32","--layers","2","--heads","2",
        "--L","64","--batch",str(args.batch),"--steps",str(steps),
        "--acc",str(acc),"--hlr","0.05","--clr","0.02","--heps","1e-5",
        "--seed",str(args.seed),"--tag",f"batch_{name}_b{args.batch}_s{args.seed}",
        "--out",out.name,*flags,
    ],check=True)

rows=[json.loads(line) for line in out.read_text().splitlines()]
assert len(rows)==3 and len({r["tag"] for r in rows})==3
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
results={name:r["scores"]["val"] for (name,*_),r in zip(arms,rows)}
for name,ce in results.items():
    print(f"PROJECTOR_BATCH batch={args.batch} seed={args.seed} arm={name} CE={ce:.5f}",flush=True)
print(f"PROJECTOR_BATCH_EFFECT batch={args.batch} seed={args.seed} "
      f"plain_minus_projector={results['plain']-results['projector']:+.5f} "
      f"centroid_minus_projector={results['centroid']-results['projector']:+.5f}",
      flush=True)
