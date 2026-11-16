"""Explicitly supervise learned output clusters in a plain softmax head."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
p = root / "grid_study_runtime.py"
s = p.read_text()
s = s.replace(
    'p.add_argument("--seed", type=int, default=0)',
    'p.add_argument("--aux_alpha", type=float, default=0.0); p.add_argument("--seed", type=int, default=0)',
)
s = s.replace(
    '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
    '        K = 64 if a.aux_alpha else max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
)
s = s.replace(
    '        return F.cross_entropy(lg, tg, reduction="none")',
    '''        ce = F.cross_entropy(lg, tg, reduction="none")
        if a.aux_alpha and s.training:
            group_p = torch.zeros(len(lg),64,device=lg.device).index_add_(1,s.ah,lg.softmax(-1))
            tgt_g = s.ah[tg]
            ce = ce - a.aux_alpha * group_p.gather(1,tgt_g[:,None]).squeeze(1).clamp_min(1e-12).log()
        return ce'''
)
s = s.replace(
    '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    '''        if a.aux_alpha and step and step % 50 == 0:
            with torch.no_grad():
                m.ah.copy_(kmeans(m.head.detach(), 64))
        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()'''
)
p.write_text(s)
results=root/"aux_class_loss.jsonl"
results.unlink(missing_ok=True)
for seed in (0,1,2):
    for alpha in (0.01,0.05,0.2):
        subprocess.run([
            sys.executable,str(p),"--dev","cpu","--V","6000","--steps","400",
            "--batch","32","--heps","1e-5","--seed",str(seed),
            "--aux_alpha",str(alpha),"--tag",f"auxclass_a{alpha}_s{seed}",
            "--out",results.name,"--threads","2",
        ],check=True)
rows=[json.loads(x) for x in results.read_text().splitlines()]
assert len(rows)==9
for r in rows:
    print(r["tag"],"val",r["scores"]["val"])
