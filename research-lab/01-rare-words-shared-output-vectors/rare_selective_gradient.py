"""Explore exposure-weighted group-gradient sharing to reduce common-token transfer."""
from pathlib import Path
import subprocess
import sys
import json
from statistics import mean

lab = Path(__file__).resolve().parent
script = lab / "grid_study_runtime.py"
source = script.read_text()
anchors = [
('p.add_argument("--seed", type=int, default=0)',
 'p.add_argument("--rare_alpha", type=float, default=0.0); p.add_argument("--rare_tau", type=float, default=0.0001); p.add_argument("--seed", type=int, default=0)'),
('        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
 '        K = 64 if a.rare_alpha else max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))'),
('        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
 '''        if a.rare_alpha and step and step % 50 == 0:
            with torch.no_grad():
                m.ah.copy_(kmeans(m.head.detach(), 64))
        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()'''),
('        for o in opts: o.step()',
 '''        if a.rare_alpha:
            with torch.no_grad():
                g = m.head.grad
                grouped = torch.zeros(64, g.shape[1], device=g.device).index_add_(0, m.ah, g)
                mask = torch.tensor(np.minimum(1.0, a.rare_tau / np.maximum(pw, 1e-12)), device=g.device, dtype=g.dtype)
                g.add_(mask[:, None] * grouped[m.ah], alpha=a.rare_alpha)
        for o in opts: o.step()''')
]
for old, new in anchors:
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
script.write_text(source)
out=lab/"rare_selective.jsonl"
out.unlink(missing_ok=True)
for seed in (0,1,2):
    for tau in (3e-5,1e-4,3e-4):
        subprocess.run([
            sys.executable,str(script),"--dev","cpu","--threads","2",
            "--V","6000","--steps","400","--batch","32","--heps","1e-5",
            "--seed",str(seed),"--rare_alpha","0.01","--rare_tau",str(tau),
            "--tag",f"rare_tau{tau}_s{seed}","--out",out.name
        ],check=True)
rows=[json.loads(s) for s in out.read_text().splitlines()]
assert len(rows)==9
for tau in (3e-5,1e-4,3e-4):
    vals=[r["scores"]["val"] for r in rows if r["args"]["rare_tau"]==tau]
    print("RARE_TARGETED tau",tau,"seedloss",vals,"mean",mean(vals),flush=True)
