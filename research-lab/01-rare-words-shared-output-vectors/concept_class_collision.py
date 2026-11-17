"""Hypernym/concept-class auxiliary loss vs normalized class-projected Ember."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--seed",type=int,required=True,choices=[397,409,419])
args=p.parse_args()

lab=Path(__file__).resolve().parent
runner=lab/"grid_study_runtime.py"
source=runner.read_text()
patches=[
    ('import numpy as np, torch, torch.nn as nn, torch.nn.functional as F',
     'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nfrom ember import Ember'),
    ('p.add_argument("--coupled", type=int, default=0)',
     'p.add_argument("--coupled", type=int, default=0); '
     'p.add_argument("--projector_alpha",type=float,default=0.0); '
     'p.add_argument("--concept_weight",type=float,default=0.0); '
     'p.add_argument("--group_source",default="dynamic")'),
    ('        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
     '        K = 64 if (a.projector_alpha or a.concept_weight) else max(a.clus, 1); '
     's.Ch = nn.Parameter(torch.zeros(K, d))'),
    ('        s.register_buffer("ah", torch.tensor(asg))',
     '        s.register_buffer("ah", torch.tensor(asg))\n'
     '        if (a.projector_alpha or a.concept_weight) and a.group_source=="context":\n'
     '            s.ah.copy_(torch.tensor(np.load(os.path.join(DATA,"context_groups_6000.npy"))))'),
    ('        lg = F.linear(h, s.H())',
     '        lg = F.linear(h, s.H())\n'
     '        if a.concept_weight and s.training:\n'
     '            logp=F.log_softmax(lg,dim=-1)\n'
     '            grp_p=torch.zeros(len(logp),64,device=logp.device).index_add_(1,s.ah,logp.exp())\n'
     '            cls=s.ah[tg]\n'
     '            cls_nll=-grp_p.gather(1,cls[:,None]).clamp_min(1e-30).log().squeeze(1)\n'
     '            tok_nll=-logp.gather(1,tg[:,None]).squeeze(1)\n'
     '            return (1-a.concept_weight)*tok_nll+a.concept_weight*cls_nll'),
    ('    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
     '    groups=[g for g in groups if g["params"][0] is not m.head]\n'
     '    opts.append(Ember([m.head],lr=a.hlr,beta2=0.95,eps=1e-8))\n'
     '    if a.coupled:\n'
     '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))'),
    ('        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
     '        if (a.projector_alpha or a.concept_weight) and a.group_source=="dynamic" and step>0 and step%50==0:\n'
     '            with torch.no_grad():m.ah.copy_(kmeans(m.head.detach(),64))\n'
     '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()'),
    ('        for o in opts: o.step()',
     '        if a.projector_alpha:\n'
     '            with torch.no_grad():\n'
     '                g=m.head.grad\n'
     '                pooled=torch.zeros(64,g.shape[1],device=g.device).index_add_(0,m.ah,g)\n'
     '                counts=torch.bincount(m.ah,minlength=64).clamp_min(1)\n'
     '                g.add_(pooled[m.ah]/counts[m.ah,None],alpha=a.projector_alpha)\n'
     '        for o in opts: o.step()')
];
for old,replacement in patches:
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,replacement)
runner.write_text(source)

out=lab/f"concept_class_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)
arms=[
    ("plain",0.0,0.0,"dynamic"),
    ("projector_context",2.0,0.0,"context"),
    ("projector_dynamic",2.0,0.0,"dynamic"),
    ("concept_context_lambda01",0.0,0.1,"context"),
    ("concept_context_lambda03",0.0,0.3,"context"),
    ("concept_context_lambda05",0.0,0.5,"context"),
    ("concept_dynamic_lambda03",0.0,0.3,"dynamic"),
]
for name,alpha,weight,source_group in arms:
    subprocess.run([
        sys.executable,str(runner),"--dev","cpu","--threads","1",
        "--V","6000","--width","32","--layers","2","--heads","2",
        "--L","64","--batch","32","--steps","400",
        "--heps","1e-5","--hlr","0.05","--clr","0.02",
        "--projector_alpha",str(alpha),"--concept_weight",str(weight),
        "--group_source",source_group,
        "--seed",str(args.seed),"--tag",f"concept_{name}_s{args.seed}",
        "--out",out.name,
    ],check=True)
rows=[json.loads(s) for s in out.read_text().splitlines()]
assert len(rows)==7 and len(set(r["tag"] for r in rows))==7
assert all(math.isfinite(r["scores"]["val"]) for r in rows)
for (name,*_),r in zip(arms,rows):
    print(f"CONCEPT_CLASS seed={args.seed} arm={name} CE={r['scores']['val']:.5f}",flush=True)
