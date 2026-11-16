"""Compare original extra-parameter group head to a direct group-gradient preconditioner."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
p = root / "grid_study_runtime.py"
s = p.read_text()
s = s.replace(
    'p.add_argument("--seed", type=int, default=0)',
    'p.add_argument("--kernel_alpha", type=float, default=0.0); p.add_argument("--seed", type=int, default=0)',
)
s = s.replace(
    '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
    '        K = 64 if a.kernel_alpha else max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
)
s = s.replace(
    '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    '''        if a.kernel_alpha and step and step % 50 == 0:
            with torch.no_grad():
                m.ah.copy_(kmeans(m.head.detach(), 64))
        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()'''
)
s = s.replace(
    '        for o in opts: o.step()',
    '''        if a.kernel_alpha:
            with torch.no_grad():
                g = m.head.grad
                pooled = torch.zeros(64, g.shape[1], device=g.device).index_add_(0, m.ah, g)
                g.add_(pooled[m.ah], alpha=a.kernel_alpha)
        for o in opts: o.step()'''
)
p.write_text(s)
result = root / "gradient_kernel_study.jsonl"
result.unlink(missing_ok=True)
for seed in (0, 1, 2):
    for alpha in (0.01, 0.05, 0.2):
        cmd = [
            sys.executable, str(p), "--dev", "cpu", "--V", "6000",
            "--steps", "400", "--batch", "32", "--heps", "1e-5",
            "--seed", str(seed), "--kernel_alpha", str(alpha),
            "--tag", f"kernel_alpha{alpha}_s{seed}",
            "--out", result.name, "--threads", "2",
        ]
        subprocess.run(cmd, check=True)
records = [json.loads(line) for line in result.read_text().splitlines()]
assert len(records) == 9
for record in records:
    print(record["tag"], "val", record["scores"]["val"])
