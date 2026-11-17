"""Batch128: differentiate bad early groups from a fundamentally bad projector."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--seed",type=int,choices=[281,283,293],required=True)
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
        'p.add_argument("--ember_head", type=int, default=0); '
        'p.add_argument("--projector_alpha", type=float, default=0.0); '
        'p.add_argument("--projector_group", default="dynamic50")',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.projector_alpha else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.projector_alpha and a.projector_group in ("context_static","context_late"):\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA,"context_groups_6000.npy"))))',
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
        '        if a.projector_alpha:\n'
        '            md=a.projector_group\n'
        '            refresh=(md=="dynamic50" and step>0 and step%50==0) or '
        '(md=="dynamic10" and step>0 and step%10==0) or '
        '(md=="dynamic0_10" and step%10==0) or '
        '(md=="context_late" and step>=20 and step%10==0)\n'
        '            if refresh:\n'
        '                with torch.no_grad(): m.ah.copy_(kmeans(m.head.detach(),64))\n'
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

out=lab/f"high_batch_rescue_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms=[
    ("ember_plain",0.0,"plain",False),
    ("ember_centroid",0.0,"plain",True),
    ("proj_dynamic_50",2.0,"dynamic50",False),
    ("proj_dynamic_10",2.0,"dynamic10",False),
    ("proj_dynamic_0_10",2.0,"dynamic0_10",False),
    ("proj_context_static",2.0,"context_static",False),
    ("proj_context_late",2.0,"context_late",False),
]
for name,alpha,group,centroid in arms:
    flags=["--ember_head","1","--projector_alpha",str(alpha),
           "--projector_group",group]
    if centroid:
        flags+=["--clus","64","--cinit","kmeans","--recl","50"]
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","1",
        "--V","6000","--width","32","--layers","2","--heads","2",
        "--L","64","--batch","128","--acc","4","--steps","100",
        "--heps","1e-5","--hlr","0.05","--clr","0.02",
        "--seed",str(args.seed),"--tag",f"highbatch_{name}_s{args.seed}",
        "--out",out.name,*flags,
    ],check=True)
rows=[json.loads(s) for s in out.read_text().splitlines()]
assert len(rows)==7 and len({r["tag"] for r in rows})==7
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
values={name:r["scores"]["val"] for (name,*_),r in zip(arms,rows)}
for name,loss in values.items():
    print(f"HIGH_BATCH_RESCUE seed={args.seed} arm={name} CE={loss:.5f} "
          f"gain_vs_plain={values['ember_plain']-loss:+.5f}",flush=True)
