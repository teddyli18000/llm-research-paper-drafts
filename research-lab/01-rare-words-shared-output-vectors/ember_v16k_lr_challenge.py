"""Tune official standalone Ember fairly against class-projected Ember at V16k."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--seed",type=int,choices=[331,337,347],required=True)
args=p.parse_args()

lab=Path(__file__).resolve().parent
runner=lab/"grid_study_runtime.py"
source=runner.read_text()
patches=[
    (
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F',
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nfrom ember import Ember',
    ),
    (
        'p.add_argument("--coupled", type=int, default=0)',
        'p.add_argument("--coupled", type=int, default=0); '
        'p.add_argument("--ember_head",type=int,default=0); '
        'p.add_argument("--ember_b2",type=float,default=0.95); '
        'p.add_argument("--projector_alpha",type=float,default=0.0)',
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
        '        opts.append(Ember([m.head],lr=a.hlr,beta2=a.ember_b2,eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.projector_alpha and step and step%50==0:\n'
        '            with torch.no_grad():m.ah.copy_(kmeans(m.head.detach(),64))\n'
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
for old,new in patches:
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
runner.write_text(source)

out=lab/f"ember_v16k_tune_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)

arms=[]
for b2 in (0.95,0.999):
    for lr in (0.01,0.02,0.03,0.05,0.08):
        arms.append((f"plain_b{b2:g}_lr{lr:g}",True,False,False,lr,b2))
for lr in (0.03,0.05):
    arms.append((f"projector_b0.95_lr{lr:g}",True,True,False,lr,0.95))
arms.append(("upstream_centroid_adam",False,False,True,0.03,0.95))

for name,use_ember,projected,centroid,lr,b2 in arms:
    flags=[]
    if use_ember:flags+=["--ember_head","1","--ember_b2",str(b2)]
    if projected:flags+=["--projector_alpha","2.0"]
    if centroid:flags+=["--clus","64","--cinit","kmeans","--recl","50"]
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","2",
        "--V","16000","--width","128","--layers","2","--heads","4",
        "--L","64","--batch","32","--steps","400","--acc","1",
        "--heps","1e-5","--hlr",str(lr),"--clr","0.005",
        "--seed",str(args.seed),"--tag",f"ember_tune_{name}_s{args.seed}",
        "--out",out.name,*flags,
    ],check=True)

rows=[json.loads(s) for s in out.read_text().splitlines()]
assert len(rows)==13 and len(set(x["tag"] for x in rows))==13
assert all(math.isfinite(x["scores"]["val"]) for x in rows)
for (name,*_),row in zip(arms,rows):
    print(f"EMBER_TUNE_V16K seed={args.seed} arm={name} CE={row['scores']['val']:.5f}",flush=True)
