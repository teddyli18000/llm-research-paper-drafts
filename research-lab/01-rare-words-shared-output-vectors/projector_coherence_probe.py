"""Trace group-projected LM-head gradient energy with fixed model seeds and methods."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--seed",type=int,required=True,choices=[223,227,229])
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
        'p.add_argument("--projector_alpha",type=float,default=2.0); '
        'p.add_argument("--projector_group",default="dynamic")',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.projector_alpha else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        s.register_buffer("ah", torch.tensor(asg))',
        '        s.register_buffer("ah", torch.tensor(asg))\n'
        '        if a.projector_group=="context":\n'
        '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA,"context_groups_6000.npy"))))',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    groups=[g for g in groups if g["params"][0] is not m.head]\n'
        '    opts.append(Ember([m.head],lr=a.hlr,beta2=0.95,eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '    regroup_records = []\n    t0 = time.time(); X = torch.tensor(rows).to(D)',
        '    regroup_records = []\n    coherence_trace=[]\n    t0 = time.time(); X = torch.tensor(rows).to(D)',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if step and step%50==0 and a.projector_group in ("dynamic","shuffled"):\n'
        '            with torch.no_grad():\n'
        '                asg=kmeans(m.head.detach(),64)\n'
        '                if a.projector_group=="shuffled":\n'
        '                    perm=torch.randperm(V,generator=torch.Generator().manual_seed(5000+step))\n'
        '                    asg=asg[perm.to(asg.device)]\n'
        '                m.ah.copy_(asg)\n'
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    ),
    (
        '        for o in opts: o.step()',
        '        with torch.no_grad():\n'
        '            g=m.head.grad\n'
        '            pooled=torch.zeros(64,g.shape[1],device=g.device).index_add_(0,m.ah,g)\n'
        '            counts=torch.bincount(m.ah,minlength=64).clamp_min(1)\n'
        '            projected=pooled[m.ah]/counts[m.ah,None]\n'
        '            if step in (0,25,50,100,200,399):\n'
        '                energy=float((projected.square().sum()/g.square().sum()).item())\n'
        '                coherence_trace.append([step,energy])\n'
        '            g.add_(projected,alpha=a.projector_alpha)\n'
        '        for o in opts: o.step()',
    ),
    (
        '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
        '    res = dict(regroup_records=regroup_records, coherence_trace=coherence_trace, '
        'tag=a.tag, args=vars(a), tokens=need',
    ),
]
for old,new in patches:
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
runner.write_text(source)

out=lab/f"projector_coherence_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)
for mode in ("dynamic","shuffled","context"):
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","1",
        "--V","6000","--width","32","--layers","2","--heads","2",
        "--L","64","--batch","32","--steps","400",
        "--heps","1e-5","--hlr","0.05","--clr","0.02",
        "--projector_alpha","2.0","--projector_group",mode,
        "--seed",str(args.seed),"--tag",f"coherence_{mode}_s{args.seed}",
        "--out",out.name,
    ],check=True)
rows=[json.loads(x) for x in out.read_text().splitlines()]
assert len(rows)==3 and len({x["tag"] for x in rows})==3
for row in rows:
    assert len(row["coherence_trace"])==6
    assert all(math.isfinite(c) and 0<=c<=1.00001 for _,c in row["coherence_trace"])
    print(f"COHERENCE seed={args.seed} arm={row['args']['projector_group']} "
          f"CE={row['scores']['val']:.5f} trace={row['coherence_trace']}",flush=True)
