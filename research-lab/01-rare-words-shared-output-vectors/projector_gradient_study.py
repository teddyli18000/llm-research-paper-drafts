"""Test cluster-size-normalized gradient sharing: I + alpha A(A^T A)^(-1)A^T."""
from pathlib import Path
import subprocess
import sys
import json
from statistics import mean

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()
anchor = 'p.add_argument("--seed", type=int, default=0)'
assert source.count(anchor) == 1
source = source.replace(anchor, 'p.add_argument("--projector_alpha", type=float, default=0.0); ' + anchor)
anchor = '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))'
assert source.count(anchor) == 1
source = source.replace(anchor,'        K = 64 if a.projector_alpha else max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))')
anchor = '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()'
assert source.count(anchor) == 1
source = source.replace(anchor, '''        if a.projector_alpha and step and step % 50 == 0:
            with torch.no_grad():
                m.ah.copy_(kmeans(m.head.detach(), 64))
        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()''')
anchor = '        for o in opts: o.step()'
assert source.count(anchor) == 1
source = source.replace(anchor,'''        if a.projector_alpha:
            with torch.no_grad():
                g = m.head.grad
                summed = torch.zeros(64, g.shape[1], device=g.device).index_add_(0, m.ah, g)
                size = torch.bincount(m.ah, minlength=64).clamp(min=1)
                g.add_(summed[m.ah] / size[m.ah, None], alpha=a.projector_alpha)
        for o in opts: o.step()''')
runner.write_text(source)
out = lab / "projector_gradient.jsonl"
out.unlink(missing_ok=True)
for seed in (0, 1, 2):
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
        subprocess.run([
            sys.executable, str(runner), "--dev", "cpu", "--threads", "2",
            "--V", "6000", "--steps", "400", "--batch", "32", "--heps", "1e-5",
            "--seed", str(seed), "--projector_alpha", str(alpha),
            "--tag", f"proj_{alpha}_s{seed}", "--out", out.name
        ], check=True)
rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == 15
for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
    values = [r["scores"]["val"] for r in rows if r["args"]["projector_alpha"] == alpha]
    print(f"PROJECTOR alpha={alpha}: values={values}; mean_val={mean(values):.6f}", flush=True)
