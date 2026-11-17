"""Train-only lagged group-gradient coherence gate, fixed parameter-free head."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--seed",type=int,choices=[431,433,439],required=True)
p.add_argument("--batch",type=int,choices=[32,128],required=True)
args=p.parse_args()

lab=Path(__file__).resolve().parent
runner=lab/"grid_study_runtime.py"
source=runner.read_text()
patches=[
    ('import numpy as np, torch, torch.nn as nn, torch.nn.functional as F',
     'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nfrom ember import Ember'),
    ('p.add_argument("--coupled", type=int, default=0)',
     'p.add_argument("--coupled", type=int, default=0); '
     'p.add_argument("--proj_mode",default="plain")'),
    ('        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
     '        K = 64 if a.proj_mode!="plain" else max(a.clus, 1); '
     's.Ch = nn.Parameter(torch.zeros(K, d))'),
    ('        s.register_buffer("ah", torch.tensor(asg))',
     '        s.register_buffer("ah", torch.tensor(asg))\n'
     '        if a.proj_mode in ("context","gated"):\n'
     '            ctx=torch.tensor(np.load(os.path.join(DATA,"context_groups_6000.npy")))\n'
     '            s.register_buffer("context_groups",ctx)\n'
     '            if a.proj_mode=="context":s.ah.copy_(ctx)'),
    ('    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
     '    groups=[g for g in groups if g["params"][0] is not m.head]\n'
     '    opts.append(Ember([m.head],lr=a.hlr,beta2=0.95,eps=1e-8))\n'
     '    if a.coupled:\n'
     '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))'),
    ('    regroup_records = []\n    t0 = time.time(); X = torch.tensor(rows).to(D)',
     '    regroup_records = []\n'
     '    gate_trace=[]; dynamic_steps=0; score_ctx=0.0; score_dyn=0.0\n'
     '    t0 = time.time(); X = torch.tensor(rows).to(D)'),
    ('        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
     '        if a.proj_mode in ("dynamic","gated") and step>0 and step%50==0:\n'
     '            with torch.no_grad():m.ah.copy_(kmeans(m.head.detach(),64))\n'
     '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()'),
    ('        for o in opts: o.step()',
     '        if a.proj_mode!="plain":\n'
     '            with torch.no_grad():\n'
     '                g=m.head.grad\n'
     '                total=g.square().sum().clamp_min(1e-30)\n'
     '                dims=g.shape[1]\n'
     '                sums=torch.zeros(64,dims,device=g.device).index_add_(0,m.ah,g)\n'
     '                cnt=torch.bincount(m.ah,minlength=64).clamp_min(1)\n'
     '                pdyn=sums[m.ah]/cnt[m.ah,None]\n'
     '                if a.proj_mode=="gated":\n'
     '                    ctx=m.context_groups\n'
     '                    ctxs=torch.zeros(64,dims,device=g.device).index_add_(0,ctx,g)\n'
     '                    ctxn=torch.bincount(ctx,minlength=64).clamp_min(1)\n'
     '                    pctx=ctxs[ctx]/ctxn[ctx,None]\n'
     '                    ec=float((pctx.square().sum()/total).item())\n'
     '                    ed=float((pdyn.square().sum()/total).item())\n'
     '                    use_dynamic=step>0 and score_dyn>1.5*score_ctx\n'
     '                    g.add_(pdyn if use_dynamic else pctx,alpha=2.0)\n'
     '                    dynamic_steps+=int(use_dynamic)\n'
     '                    score_ctx=ec if step==0 else .9*score_ctx+.1*ec\n'
     '                    score_dyn=ed if step==0 else .9*score_dyn+.1*ed\n'
     '                    if step in (0,25,50,99,100,200,399):\n'
     '                        gate_trace.append([step,round(score_ctx,6),round(score_dyn,6),int(use_dynamic)])\n'
     '                else:\n'
     '                    g.add_(pdyn,alpha=2.0)\n'
     '        for o in opts: o.step()'),
    ('    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need',
     '    res = dict(regroup_records=regroup_records, gate_trace=gate_trace, '
     'gate_dynamic_frac=dynamic_steps/a.steps, tag=a.tag, args=vars(a), tokens=need'),
]
for old,replacement in patches:
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,replacement)
runner.write_text(source)

steps=400 if args.batch==32 else 100
acc=1 if args.batch==32 else 4
out=lab/f"coherence_gate_b{args.batch}_s{args.seed}.jsonl"
out.unlink(missing_ok=True)

modes=("plain","context","dynamic","gated")
for mode in modes:
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","1",
        "--V","6000","--width","32","--layers","2","--heads","2",
        "--L","64","--batch",str(args.batch),"--acc",str(acc),
        "--steps",str(steps),"--hlr","0.05","--clr","0.02","--heps","1e-5",
        "--proj_mode",mode,"--seed",str(args.seed),
        "--tag",f"gate_{mode}_b{args.batch}_s{args.seed}","--out",out.name,
    ],check=True)

rows=[json.loads(s) for s in out.read_text().splitlines()]
assert len(rows)==4 and len({r["tag"] for r in rows})==4
assert all(math.isfinite(r["scores"]["val"]) and r["tokens"]==819200 for r in rows)
for mode,row in zip(modes,rows):
    print(f"COHERENCE_GATE batch={args.batch} seed={args.seed} mode={mode} "
          f"CE={row['scores']['val']:.5f} dynamic_fraction={row['gate_dynamic_frac']:.4f} "
          f"trace={row['gate_trace']}",flush=True)
