"""Compare direct group-projector head gradients to equal-forward TreeLoss-style coordinates."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--seed",type=int,choices=[373,379,383],required=True)
args=p.parse_args()
lab=Path(__file__).resolve().parent
runner=lab/"grid_study_runtime.py"
source=runner.read_text()

patches=[
    (
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F',
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nfrom ember import Ember\nimport math',
    ),
    (
        'p.add_argument("--coupled", type=int, default=0)',
        'p.add_argument("--coupled", type=int, default=0); '
        'p.add_argument("--ember_head",type=int,default=0); '
        'p.add_argument("--projector_alpha",type=float,default=0.0); '
        'p.add_argument("--tree_reparam",type=int,default=0)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if (a.projector_alpha or a.tree_reparam) else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.projector_alpha or a.tree_reparam:\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA,"context_groups_6000.npy"))))',
    ),
    (
        '        if not s.uh: return s.head',
        '        if a.tree_reparam:\n'
        '            pooled=torch.zeros(64,s.head.shape[1],device=s.head.device).index_add_(0,s.ah,s.head)\n'
        '            counts=torch.bincount(s.ah,minlength=64).clamp_min(1)\n'
        '            return s.head+(math.sqrt(3)-1)*pooled[s.ah]/counts[s.ah,None]\n'
        '        if not s.uh: return s.head',
    ),
    (
        '    m = Model().to(D); m.Clast = m.Clast.to(D)',
        '    m = Model().to(D); m.Clast = m.Clast.to(D)\n'
        '    if a.tree_reparam:\n'
        '        with torch.no_grad():\n'
        '            initial=m.head.detach().clone()\n'
        '            sums=torch.zeros(64,m.head.shape[1],device=D).index_add_(0,m.ah,initial)\n'
        '            counts=torch.bincount(m.ah,minlength=64).clamp_min(1)\n'
        '            m.head.copy_(initial+(1/math.sqrt(3)-1)*sums[m.ah]/counts[m.ah,None])\n'
        '            assert (m.H()-initial).abs().max().item()<1e-5',
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

out=lab/f"tree_reparam_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms=[
    ("ember_plain",True,False,False),
    ("ember_direct_context",True,True,False),
    ("ember_reparam_context",True,False,True),
    ("adam_direct_context",False,True,False),
    ("adam_reparam_context",False,False,True),
]
for name,ember,direct,reparam in arms:
    flags=["--ember_head",str(int(ember)),
           "--projector_alpha",str(2.0 if direct else 0.0),
           "--tree_reparam",str(int(reparam))]
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","1",
        "--V","6000","--width","32","--layers","2","--heads","2",
        "--L","64","--batch","32","--steps","400",
        "--heps","1e-5","--hlr","0.05","--clr","0.02",
        "--seed",str(args.seed),"--tag",f"treeloss_{name}_s{args.seed}",
        "--out",out.name,*flags,
    ],check=True)
rows=[json.loads(s) for s in out.read_text().splitlines()]
assert len(rows)==5 and len(set(r["tag"] for r in rows))==5
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
v={name:rec["scores"]["val"] for (name,*_),rec in zip(arms,rows)}
for name,ce in v.items():
    print(f"TREE_REPARAM seed={args.seed} arm={name} CE={ce:.5f}",flush=True)
print(f"TREE_REPARAM_PAIR seed={args.seed} "
      f"ember_reparam_minus_direct={v['ember_reparam_context']-v['ember_direct_context']:+.5f} "
      f"adam_reparam_minus_direct={v['adam_reparam_context']-v['adam_direct_context']:+.5f}",
      flush=True)
